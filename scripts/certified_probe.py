"""M23: certified-removal probe with one frozen CIFAR-10 extractor."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.metrics import trr_metrics
from topotrace.mnist import make_random_forget_split
from topotrace.resnet import ResNet18C, get_embeddings
from topotrace.stats import bootstrap_ci, permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

SIGMA = .01
REG = .01
METHODS = ("certified", "noop", "retrain2")


def gradient(X, y, W, b):
    p = (X @ W.T).softmax(1)
    p[torch.arange(len(y), device=X.device), y] -= 1
    return p.T @ X / len(y) + REG * W + b / len(y)


def solve(X, y, b):
    W = torch.zeros((10, X.shape[1]), dtype=X.dtype, device=X.device,
                    requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [W], max_iter=200, history_size=20, tolerance_grad=1e-10,
        tolerance_change=1e-16, line_search_fn="strong_wolfe")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(X @ W.T, y) + REG / 2 * W.square().sum()
                + (b * W).sum() / len(y))
        loss.backward()
        return loss

    for _ in range(3):
        optimizer.step(closure)
        if gradient(X, y, W.detach(), b).norm().item() < 1e-10:
            break
    residual = gradient(X, y, W.detach(), b).norm().item() * len(y)
    return W.detach(), {"closure_calls": calls, "gradient_residual_norm": residual}


def remove(W, Xr, yr, Xf, yf, b):
    n_r, n_f = len(yr), len(yf)
    pf = (Xf @ W.T).softmax(1)
    pf[torch.arange(n_f, device=W.device), yf] -= 1
    rhs = pf.T @ Xf / n_r + REG * n_f / n_r * W
    p = (Xr @ W.T).softmax(1)

    def hessian(V):
        scores = Xr @ V.T
        weighted = p * (scores - (p * scores).sum(1, keepdim=True))
        return weighted.T @ Xr / n_r + REG * V

    diag = (p * (1 - p)).T @ Xr.square() / n_r + REG
    delta = torch.zeros_like(W)
    r = rhs.clone()
    z = r / diag
    direction = z.clone()
    rz = (r * z).sum()
    target = rhs.norm() * 1e-10
    iterations = 0
    for iterations in range(1, 501):
        Hd = hessian(direction)
        alpha = rz / (direction * Hd).sum()
        delta += alpha * direction
        r -= alpha * Hd
        if r.norm() <= target:
            break
        z = r / diag
        new_rz = (r * z).sum()
        direction = z + new_rz / rz * direction
        rz = new_rz

    certified = W + delta
    certificate = gradient(Xr, yr, certified, b).norm().item() * n_r
    return certified, {"cg_iterations": iterations,
                       "cg_relative_residual": (r.norm() / rhs.norm()).item(),
                       "retain_gradient_residual_norm": certificate}


def bh(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def main():
    out = ROOT / "results" / "m4_random"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    X, y, X_test, y_test = load_cifar10(str(ROOT / "data"))
    forget_idx, retain_idx = make_random_forget_split(y, frac=.05, seed=0)
    probe_idx = np.load(out / "probe_idx.npy")
    extractor = ResNet18C()
    extractor.load_state_dict(torch.load(
        out / "models" / "original_0.pt", map_location="cpu", weights_only=True))
    extractor.eval()
    print(f"extracting frozen features on {device}", flush=True)
    train_features = get_embeddings(extractor, X, device=device)["penultimate"]
    test_features = get_embeddings(extractor, X_test, device=device)["penultimate"]
    del extractor, X, X_test

    train = torch.as_tensor(train_features, dtype=dtype, device=device)
    test = torch.as_tensor(test_features, dtype=dtype, device=device)
    labels = torch.as_tensor(y, dtype=torch.long, device=device)
    test_labels = torch.as_tensor(y_test, dtype=torch.long, device=device)
    retain = train[retain_idx]
    retain_labels = labels[retain_idx]
    forget = train[forget_idx]
    forget_labels = labels[forget_idx]
    probe = train[probe_idx]

    rng = np.random.default_rng(0)
    perturbations = rng.normal(0, SIGMA, (20, 10, train.shape[1]))
    bs = torch.as_tensor(perturbations, dtype=dtype, device=device)
    weights = {name: [] for name in
               ("original", "retrain", "retrain2", "certified", "noop")}
    fits, certificates = {"original": [], "retrain": [], "retrain2": []}, []
    for draw in range(10):
        original, info_o = solve(train, labels, bs[draw])
        retrain, info_r = solve(retain, retain_labels, bs[draw])
        certified, info_c = remove(
            original, retain, retain_labels, forget, forget_labels, bs[draw])
        retrain2, info_r2 = solve(retain, retain_labels, bs[10 + draw])
        for name, value in (("original", original), ("retrain", retrain),
                            ("retrain2", retrain2), ("certified", certified),
                            ("noop", original)):
            weights[name].append(value)
        fits["original"].append(info_o)
        fits["retrain"].append(info_r)
        fits["retrain2"].append(info_r2)
        certificates.append(info_c)
        print(f"draw {draw}: fit residuals {info_o['gradient_residual_norm']:.3e} "
              f"{info_r['gradient_residual_norm']:.3e} "
              f"{info_r2['gradient_residual_norm']:.3e}; certificate "
              f"{info_c['retain_gradient_residual_norm']:.3e}", flush=True)

    subsets = (("retain", retain, retain_labels),
               ("forget", forget, forget_labels),
               ("test", test, test_labels))
    utility = {}
    for condition, models in weights.items():
        per_draw = [{name: (features @ W.T).argmax(1).eq(target).double().mean().item()
                     for name, features, target in subsets} for W in models]
        utility[condition] = {
            "per_draw": per_draw,
            "mean": {name: float(np.mean([row[name] for row in per_draw]))
                     for name, _, _ in subsets},
        }

    fingerprints = {name: [(probe @ W.T).cpu().numpy() for W in models]
                    for name, models in weights.items()}
    diagrams = {name: [persistence_diagrams(chordal_distance_matrix(logits))
                       for logits in models]
                for name, models in fingerprints.items() if name != "noop"}
    diagrams["noop"] = diagrams["original"]
    audit, labels_bh, pvalues = {}, [], []
    for hom in (0, 1):
        imager = make_imager([diagram[hom]
                              for condition in ("original", "retrain")
                              for diagram in diagrams[condition]])
        vectors = {condition: [vectorize(diagram, imager, dim=hom)
                               for diagram in values]
                   for condition, values in diagrams.items()}
        O, R = vectors["original"], vectors["retrain"]
        imprint = trr_metrics(O, R, R)
        ci = bootstrap_ci(O, R, lambda A, B: trr_metrics(A, B, B)["I_topo"])
        gate_p = permutation_pvalue(O, R)
        cell = {key: imprint[key] for key in ("D_RR", "D_OR", "I_topo")}
        cell.update(CI=ci, gate_p=gate_p,
                    gate_open=bool(gate_p < .05 and ci[0] > 0), methods={})
        for method in METHODS:
            metrics = trr_metrics(O, R, vectors[method])
            pvalue = permutation_pvalue(vectors[method], R)
            cell["methods"][method] = {
                key: metrics[key] for key in ("D_UR", "TRR", "alpha", "eta")}
            cell["methods"][method]["p"] = pvalue
            labels_bh.append((f"H{hom}", method))
            pvalues.append(pvalue)
        audit[f"H{hom}"] = cell
    for (cell, method), qvalue in zip(labels_bh, bh(pvalues)):
        audit[cell]["methods"][method].update(
            q=float(qvalue), reject=bool(qvalue <= .05))

    certified_distance = [torch.linalg.vector_norm(c - r).item()
                          for c, r in zip(weights["certified"], weights["retrain"])]
    oracle_spread = [torch.linalg.vector_norm(r - r2).item()
                     for r, r2 in zip(weights["retrain"], weights["retrain2"])]
    results = {
        "protocol": {
            "dataset": "CIFAR-10 random 5%", "extractor": "original_0.pt",
            "feature_dim": 512, "draws": 10, "sigma": SIGMA,
            "lambda": "0.01 * |S|", "perturbation_seed": 0,
            "perturbation_generator": "numpy.random.default_rng (PCG64)",
            "perturbation_sha256": hashlib.sha256(perturbations.tobytes()).hexdigest(),
            "device": str(device), "dtype": str(dtype),
        },
        "solver": {"fits": fits, "certified": certificates},
        "utility": utility,
        "parameter_distance": {
            "certified_vs_retrain": certified_distance,
            "retrain_vs_retrain2": oracle_spread,
            "mean_certified_vs_retrain": float(np.mean(certified_distance)),
            "mean_retrain_vs_retrain2": float(np.mean(oracle_spread)),
        },
        "audit": audit,
    }
    (out / "certified_probe.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'condition':10s} {'retain':>8s} {'forget':>8s} {'test':>8s}")
    for condition, values in utility.items():
        mean = values["mean"]
        print(f"{condition:10s} {mean['retain']:8.4f} {mean['forget']:8.4f} "
              f"{mean['test']:8.4f}")
    print(f"\n{'cell':4s} {'method':10s} {'I_topo':>10s} {'CI':>23s} "
          f"{'gate p':>8s} {'TRR':>8s} {'p':>8s} {'q':>8s} {'decision':>9s}")
    for cell, values in audit.items():
        for method, row in values["methods"].items():
            ci = values["CI"]
            print(f"{cell:4s} {method:10s} {values['I_topo']:+10.6f} "
                  f"({ci[0]:+.6f},{ci[1]:+.6f}) {values['gate_p']:8.4f} "
                  f"{row['TRR']:+8.3f} {row['p']:8.4f} {row['q']:8.4f} "
                  f"{'REJECT' if row['reject'] else 'RETAIN':>9s}")
    print("\nparameter L2 means: certified-retrain="
          f"{np.mean(certified_distance):.6g}, retrain-retrain2="
          f"{np.mean(oracle_spread):.6g}")
    print("certificate proxy mean: "
          f"{np.mean([row['retain_gradient_residual_norm'] for row in certificates]):.6g}")


if __name__ == "__main__":
    main()
