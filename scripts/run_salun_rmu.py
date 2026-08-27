"""Refine RMU at the oracle operating point on CIFAR-10 random 5%."""
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
from topotrace.unlearn import rmu

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
    if not result_path.exists():
        raise RuntimeError("M24 salun_rmu.json is required for M25 refinement")
    results = json.loads(result_path.read_text())
    tuning = results["tuning"]
    preserved = {
        "stage2": json.dumps({
            "salun": tuning["salun"],
            "rmu": {key: tuning["rmu"][key]
                    for key in ("rows", "selected", "constraint_met")},
        }),
        "salun_accuracy": json.dumps(results["accuracy"]["salun"]),
        "salun_mia": json.dumps(results["mia_auc"]["salun"]),
        "salun_audit": json.dumps({
            cell: values["methods"]["salun"]
            for cell, values in results["audit"].items()}),
        "anchors": json.dumps({cell: values["retrain2"]
                               for cell, values in results["audit"].items()}),
    }
    X, y, X_test, y_test = load_cifar10()
    forget_idx, retain_idx = make_random_forget_split(y, frac=.05, seed=0)
    probe = X[np.load(out / "probe_idx.npy")]
    original = load_model(models_dir / "original_0.pt")
    grid = [dict(coeff=coeff, steps=steps, alpha=5.)
            for coeff, steps in product((2., 4.), (350, 400, 450, 500, 600))]
    rows = []
    print(f"GPU: {torch.cuda.get_device_name(0)}\nRMU REFINEMENT (seed 0)",
          flush=True)
    for config in grid:
        candidate = rmu(original, X, y, forget_idx, retain_idx,
                        seed=2000, **config)
        row = {"config": config,
               **accuracy(candidate, X, y, forget_idx, retain_idx)}
        row["eligible"] = row["retain"] >= .97
        row["target_error"] = abs(row["forget"] - TARGET)
        rows.append(row)
        print(f"{str(config):42s} retain={row['retain']:.4f} "
              f"forget={row['forget']:.4f} eligible={row['eligible']} "
              f"target_error={row['target_error']:.4f}", flush=True)
    eligible = [row for row in rows if row["eligible"]]
    best = (min(eligible, key=lambda row: row["target_error"])
            if eligible else max(rows, key=lambda row: row["retain"]))
    tuning["rmu"]["refinement"] = {
        "rows": rows, "selected": best, "constraint_met": bool(eligible)}
    stage2 = tuning["rmu"]["selected"]
    improved = bool(eligible) and best["target_error"] < stage2["target_error"]
    final = best if improved else stage2
    tuning["rmu"]["final_stage"] = "refinement" if improved else "stage2"
    print("\nFINAL RMU CONFIGURATION", flush=True)
    print(f"stage={tuning['rmu']['final_stage']} config={final['config']}",
          flush=True)
    if not improved:
        print("Refinement did not beat stage 2; keeping stage 2 results.",
              flush=True)

    if improved:
        embeddings_path = out / "embeddings_salun_rmu.npz"
        with np.load(embeddings_path) as old:
            embeddings = {key: old[key] for key in old.files}
            salun_embeddings = {key: value.copy()
                                for key, value in embeddings.items()
                                if key.startswith("salun_")}
        accs, mia = [], []
        for seed in range(10):
            original = load_model(models_dir / f"original_{seed}.pt")
            model = rmu(original, X, y, forget_idx, retain_idx,
                        seed=2000 + seed, **final["config"])
            torch.save(model.state_dict(), models_dir / f"rmu_{seed}.pt")
            values = accuracy(model, X, y, forget_idx, retain_idx)
            values["test"] = evaluate(model, X_test, y_test)
            accs.append(values)
            mia.append(loss_mia_auc(
                model, X, y, forget_idx, X_test, y_test, seed=seed))
            encoded = get_embeddings(model, probe)
            for layer in LAYERS:
                embeddings[f"rmu_{seed}_{layer}"] = encoded[layer]
            print(f"seed {seed}: retain={values['retain']:.4f} "
                  f"forget={values['forget']:.4f} test={values['test']:.4f} "
                  f"MIA={mia[-1]:.4f}", flush=True)
        assert all(np.array_equal(embeddings[key], value)
                   for key, value in salun_embeddings.items())
        np.savez_compressed(embeddings_path, **embeddings)
        results["accuracy"]["rmu"] = accs
        results["mia_auc"]["rmu"] = mia

        with np.load(out / "embeddings.npz") as old:
            base = {layer: {condition: [old[f"{condition}_{seed}_{layer}"]
                                        for seed in range(10)]
                            for condition in ("original", "retrain")}
                    for layer in LAYERS}
        labels = []
        for layer in LAYERS:
            diagrams = {condition: [persistence_diagrams(
                chordal_distance_matrix(values)) for values in models]
                        for condition, models in base[layer].items()}
            diagrams["rmu"] = [persistence_diagrams(chordal_distance_matrix(
                embeddings[f"rmu_{seed}_{layer}"])) for seed in range(10)]
            for hom in (0, 1):
                imager = make_imager([diagram[hom]
                                      for condition in ("original", "retrain")
                                      for diagram in diagrams[condition]])
                vectors = {condition: [vectorize(diagram, imager, dim=hom)
                                       for diagram in values]
                           for condition, values in diagrams.items()}
                cell = f"{layer}_H{hom}"
                metrics = trr_metrics(vectors["original"], vectors["retrain"],
                                      vectors["rmu"])
                p = permutation_pvalue(vectors["rmu"], vectors["retrain"])
                results["audit"][cell]["methods"]["rmu"] = {
                    key: metrics[key] for key in ("TRR", "alpha", "eta")}
                results["audit"][cell]["methods"]["rmu"]["p"] = p
        for cell in results["audit"]:
            for method in METHODS:
                labels.append((cell, method))
        pvalues = [results["audit"][cell]["methods"][method]["p"]
                   for cell, method in labels]
        for (cell, method), q in zip(labels, bh(pvalues)):
            if method == "rmu":
                results["audit"][cell]["methods"][method].update(
                    q=float(q), reject=bool(q <= .05))

    assert preserved["stage2"] == json.dumps({
        "salun": tuning["salun"],
        "rmu": {key: tuning["rmu"][key]
                for key in ("rows", "selected", "constraint_met")},
    })
    assert preserved["salun_accuracy"] == json.dumps(
        results["accuracy"]["salun"])
    assert preserved["salun_mia"] == json.dumps(results["mia_auc"]["salun"])
    assert preserved["salun_audit"] == json.dumps({
        cell: values["methods"]["salun"]
        for cell, values in results["audit"].items()})
    assert preserved["anchors"] == json.dumps({
        cell: values["retrain2"] for cell, values in results["audit"].items()})
    result_path.write_text(json.dumps(results, indent=2))
    accs, mia = results["accuracy"]["rmu"], results["mia_auc"]["rmu"]
    means = {key: np.mean([row[key] for row in accs])
             for key in ("retain", "forget", "test")}
    print("\nRMU 10-SEED MEANS", flush=True)
    print(f"retain={means['retain']:.4f} forget={means['forget']:.4f} "
          f"test={means['test']:.4f} MIA={np.mean(mia):.4f}", flush=True)
    print("\nRMU AUDIT", flush=True)
    print(f"{'cell':15s} {'TRR':>8s} {'alpha':>8s} {'eta':>8s} "
          f"{'p':>8s} {'q':>8s} {'reject':>8s}")
    for cell, values in results["audit"].items():
        row = values["methods"]["rmu"]
        print(f"{cell:15s} {row['TRR']:+8.3f} {row['alpha']:+8.3f} "
              f"{row['eta']:8.3f} {row['p']:8.4f} {row['q']:8.4f} "
              f"{str(row['reject']):>8s}")


if __name__ == "__main__":
    main()
