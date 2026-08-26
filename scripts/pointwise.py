"""Pointwise representation metrics from cached embeddings."""

import json
import sys
from pathlib import Path

import numpy as np

METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")


def prepare(X):
    X = np.asarray(X, dtype=np.float32)
    unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    centered = X - X.mean(axis=0)
    gram = centered @ centered.T
    gram /= np.linalg.norm(gram) + 1e-12
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(similarity, -10, axis=1)[:, -10:]
    graph = np.zeros(similarity.shape, dtype=bool)
    graph[np.arange(len(X))[:, None], neighbors] = True
    return gram, unit, graph


def compare(A, B):
    return (
        float(np.vdot(A[0], B[0])),
        float(np.mean(np.sum(A[1] * B[1], axis=1))),
        float(np.mean(np.sum(A[2] & B[2], axis=1) / 10)),
    )


def permutation_pvalue(a, b, n_perm=10_000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    pool, n = np.concatenate((a, b)), len(a)
    observed = abs(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    count = sum(abs(p[:n].mean() - p[n:].mean()) >= observed
                for p in (rng.permutation(pool) for _ in range(n_perm)))
    return (count + 1) / (n_perm + 1)


def main():
    out = Path(sys.argv[1])
    with np.load(out / "embeddings.npz") as archive:
        seeds = sorted(int(k.split("_")[1]) for k in archive.files
                       if k.startswith("retrain_") and k.endswith("_penultimate"))
        retrains = [prepare(archive[f"retrain_{s}_penultimate"]) for s in seeds]
        baseline_values = [np.mean([compare(a, b) for j, b in enumerate(retrains)
                                    if i != j], axis=0)
                           for i, a in enumerate(retrains)]
        method_values = {}
        for method in METHODS:
            models = [prepare(archive[f"{method}_{s}_penultimate"]) for s in seeds]
            method_values[method] = [np.mean([compare(model, r) for r in retrains],
                                             axis=0) for model in models]

    names = ("cka", "cosine", "knn_overlap")
    baseline = np.asarray(baseline_values)
    results = {
        "baseline": {"n_seeds": len(baseline), **{
            name: {"median": float(np.median(baseline[:, i]))}
            for i, name in enumerate(names)}},
        "methods": {},
    }
    for method, values in method_values.items():
        values = np.asarray(values)
        results["methods"][method] = {"n_seeds": len(values), **{
            name: {
                "median": float(np.median(values[:, i])),
                "p": permutation_pvalue(values[:, i], baseline[:, i]),
            } for i, name in enumerate(names)}}

    print(f"{'method':9s} {'CKA':>18s} {'cosine':>18s} {'10-NN':>18s}")
    row = results["baseline"]
    print(f"{'retrain':9s} " + " ".join(
        f"{row[name]['median']:9.4f} {'-':>8s}" for name in names))
    for method, row in results["methods"].items():
        print(f"{method:9s} " + " ".join(
            f"{row[name]['median']:9.4f} {row[name]['p']:8.4g}" for name in names))
    (out / "pointwise.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
