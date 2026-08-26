"""Tune and audit SalUn and RMU on CIFAR-10 random 5%."""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.attacks import loss_mia_auc
from topotrace.cifar import load_cifar10
from topotrace.metrics import trr_metrics
from topotrace.mnist import make_random_forget_split
from topotrace.resnet import ResNet18C, evaluate, get_embeddings
from topotrace.stats import permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)
from topotrace.unlearn import rmu, salun

METHODS = ("salun", "rmu")
LAYERS = ("penultimate", "logits")
TARGET = .9222


def load_model(path):
    model = ResNet18C()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def accuracy(model, X, y, forget_idx, retain_idx):
    return {"retain": evaluate(model, X, y, retain_idx),
            "forget": evaluate(model, X, y, forget_idx)}


def bh(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1)
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    out = ROOT / "results" / "m4_random"
    models_dir = out / "models"
    result_path = out / "salun_rmu.json"
    X, y, X_test, y_test = load_cifar10()
    forget_idx, retain_idx = make_random_forget_split(y, frac=.05, seed=0)
    probe = X[np.load(out / "probe_idx.npy")]
    original = load_model(models_dir / "original_0.pt")
    grids = {
        "salun": [dict(sparsity=sparsity, epochs=epochs, lr=lr)
                  for sparsity, epochs, lr in product(
                      (.1, .2, .3, .5), (1, 2), (1e-5, 1e-4))],
        "rmu": [dict(coeff=coeff, steps=steps, alpha=alpha)
                for coeff, steps, alpha in product(
                    (.25, .5, 1., 2.), (50, 100, 300), (1., 5.))],
    }
    calls = {
        "salun": lambda config: salun(
            original, X, y, forget_idx, retain_idx, seed=2000, **config),
        "rmu": lambda config: rmu(
            original, X, y, forget_idx, retain_idx, seed=2000, **config),
    }

    tuning, selected = {}, {}
    print(f"GPU: {torch.cuda.get_device_name(0)}\nTUNING (seed 0)", flush=True)
    for method in METHODS:
        rows = []
        for config in grids[method]:
            candidate = calls[method](config)
            row = {"config": config, **accuracy(
                candidate, X, y, forget_idx, retain_idx)}
            row["eligible"] = row["retain"] >= .97
            row["target_error"] = abs(row["forget"] - TARGET)
            rows.append(row)
            print(f"{method:5s} {str(config):38s} retain={row['retain']:.4f} "
                  f"forget={row['forget']:.4f}", flush=True)
        eligible = [row for row in rows if row["eligible"]]
        best = (min(eligible, key=lambda row: row["target_error"])
                if eligible else max(rows, key=lambda row: row["retain"]))
        selected[method] = best["config"]
        tuning[method] = {"rows": rows, "selected": best,
                          "constraint_met": bool(eligible)}

    print("\nSELECTION TABLE", flush=True)
    for method in METHODS:
        row = tuning[method]["selected"]
        print(f"{method:5s} {str(row['config']):38s} retain={row['retain']:.4f} "
              f"forget={row['forget']:.4f} error={row['target_error']:.4f} "
              f"constraint_met={tuning[method]['constraint_met']}", flush=True)
    result_path.write_text(json.dumps(
        {"target_forget_accuracy": TARGET, "tuning": tuning}, indent=2))

    embeddings = {}
    accs = {method: [] for method in METHODS}
    mia = {method: [] for method in METHODS}
    for seed in range(10):
        original = load_model(models_dir / f"original_{seed}.pt")
        print(f"\nseed {seed}: selected unlearning", flush=True)
        for method in METHODS:
            model = (salun(original, X, y, forget_idx, retain_idx,
                           seed=2000 + seed, **selected[method])
                     if method == "salun" else
                     rmu(original, X, y, forget_idx, retain_idx,
                         seed=2000 + seed, **selected[method]))
            torch.save(model.state_dict(), models_dir / f"{method}_{seed}.pt")
            values = accuracy(model, X, y, forget_idx, retain_idx)
            values["test"] = evaluate(model, X_test, y_test)
            accs[method].append(values)
            mia[method].append(loss_mia_auc(
                model, X, y, forget_idx, X_test, y_test, seed=seed))
            encoded = get_embeddings(model, probe)
            for layer in LAYERS:
                embeddings[f"{method}_{seed}_{layer}"] = encoded[layer]
            print(f"  {method:5s} retain={values['retain']:.4f} "
                  f"forget={values['forget']:.4f} test={values['test']:.4f} "
                  f"MIA={mia[method][-1]:.4f}", flush=True)
    np.savez_compressed(out / "embeddings_salun_rmu.npz", **embeddings)

    conditions = ("original", "retrain", "retrain2")
    with np.load(out / "embeddings.npz") as old:
        base = {layer: {condition: [old[f"{condition}_{seed}_{layer}"]
                                    for seed in range(10)]
                        for condition in conditions} for layer in LAYERS}
    audit, labels, pvalues = {}, [], []
    for layer in LAYERS:
        diagrams = {condition: [persistence_diagrams(
            chordal_distance_matrix(values)) for values in models]
                    for condition, models in base[layer].items()}
        diagrams.update({method: [persistence_diagrams(chordal_distance_matrix(
            embeddings[f"{method}_{seed}_{layer}"])) for seed in range(10)]
                         for method in METHODS})
        for hom in (0, 1):
            imager = make_imager([diagram[hom]
                                  for condition in ("original", "retrain")
                                  for diagram in diagrams[condition]])
            vectors = {condition: [vectorize(diagram, imager, dim=hom)
                                   for diagram in values]
                       for condition, values in diagrams.items()}
            cell = f"{layer}_H{hom}"
            anchor = trr_metrics(vectors["original"], vectors["retrain"],
                                 vectors["retrain2"])
            audit[cell] = {"retrain2": {
                key: anchor[key] for key in ("TRR", "alpha", "eta")},
                "methods": {}}
            for method in METHODS:
                metrics = trr_metrics(vectors["original"], vectors["retrain"],
                                      vectors[method])
                p = permutation_pvalue(vectors[method], vectors["retrain"])
                audit[cell]["methods"][method] = {
                    key: metrics[key] for key in ("TRR", "alpha", "eta")}
                audit[cell]["methods"][method]["p"] = p
                labels.append((cell, method))
                pvalues.append(p)
    for (cell, method), q in zip(labels, bh(pvalues)):
        audit[cell]["methods"][method].update(
            q=float(q), reject=bool(q <= .05))

    results = {"target_forget_accuracy": TARGET, "tuning": tuning,
               "accuracy": accs, "mia_auc": mia, "audit": audit}
    result_path.write_text(json.dumps(results, indent=2))
    print("\nMODEL SUMMARY", flush=True)
    print(f"{'method':5s} {'retain':>8s} {'forget':>8s} {'test':>8s} {'MIA':>8s}")
    for method in METHODS:
        means = {key: np.mean([row[key] for row in accs[method]])
                 for key in ("retain", "forget", "test")}
        print(f"{method:5s} {means['retain']:8.4f} {means['forget']:8.4f} "
              f"{means['test']:8.4f} {np.mean(mia[method]):8.4f}")
    print("\nAUDIT SUMMARY", flush=True)
    print(f"{'cell':15s} {'method':5s} {'TRR':>8s} {'alpha':>8s} "
          f"{'eta':>8s} {'p':>8s} {'q':>8s}")
    for cell, values in audit.items():
        for method, row in values["methods"].items():
            print(f"{cell:15s} {method:5s} {row['TRR']:+8.3f} "
                  f"{row['alpha']:+8.3f} {row['eta']:8.3f} "
                  f"{row['p']:8.4f} {row['q']:8.4f}")


if __name__ == "__main__":
    main()
