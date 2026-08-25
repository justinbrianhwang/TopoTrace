"""Sensitivity ablations from cached M4 embeddings."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.stats import permutation_pvalue
from topotrace.topology import make_imager, persistence_diagrams, vectorize

CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")


def distances(X, kind):
    X = np.asarray(X, dtype=np.float64)
    if kind == "correlation":
        X = X - X.mean(axis=1, keepdims=True)
    if kind in ("chordal", "correlation"):
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        D2 = 2.0 - 2.0 * (X @ X.T)
    else:
        norms = np.sum(X * X, axis=1)
        D2 = norms[:, None] + norms[None, :] - 2.0 * (X @ X.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def vectors(diagrams, kind):
    h1 = [d[1] for ds in diagrams.values() for d in ds]
    if kind == "persistence_image":
        imager = make_imager(h1)
        return {c: [vectorize(d, imager) for d in ds]
                for c, ds in diagrams.items()}
    if kind == "betti_curve":
        maximum = max((float(d[:, 1].max()) for d in h1 if d.size), default=0.0)
        thresholds = np.linspace(0.0, maximum, 50)
        fn = lambda d: np.sum((d[:, 0, None] <= thresholds) &
                              (thresholds < d[:, 1, None]), axis=0)
    elif kind == "top_k":
        def fn(d):
            result = np.zeros(20)
            persistence = np.sort(d[:, 1] - d[:, 0])[::-1][:20]
            result[:len(persistence)] = persistence
            return result
    else:
        def fn(d):
            persistence = d[:, 1] - d[:, 0]
            total = persistence.sum()
            entropy = -np.sum((persistence / total) *
                              np.log(persistence / total)) if total else 0.0
            return np.array([entropy, total])
    return {c: [np.asarray(fn(d[1]), dtype=np.float64) for d in ds]
            for c, ds in diagrams.items()}


def summarize(v):
    imprint = trr_metrics(v["original"], v["retrain"], v["retrain"])
    return {
        "D_RR": imprint["D_RR"],
        "D_OR": imprint["D_OR"],
        "I_topo": imprint["I_topo"],
        "p": permutation_pvalue(v["original"], v["retrain"]),
        "TRR": {m: trr_metrics(v["original"], v["retrain"], v[m])["TRR"]
                for m in METHODS},
    }


def show(group, variant, row):
    trrs = " ".join(f"{m}={row['TRR'][m]:+.4f}" for m in METHODS)
    print(f"{group}/{variant}: D_RR={row['D_RR']:.6f} "
          f"D_OR={row['D_OR']:.6f} I_topo={row['I_topo']:+.6f} "
          f"p={row['p']:.6f} {trrs}", flush=True)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "m4_random"
    with np.load(out / "embeddings.npz") as archive:
        seeds = sorted(int(k.split("_")[1]) for k in archive.files
                       if k.startswith("original_") and k.endswith("_penultimate"))
        embeddings = {c: [archive[f"{c}_{s}_penultimate"] for s in seeds]
                      for c in CONDITIONS}

    diagram_cache = {}

    def diagrams(kind="chordal", rows=range(600)):
        key = kind, tuple(rows)
        if key not in diagram_cache:
            diagram_cache[key] = {
                c: [persistence_diagrams(distances(X[list(key[1])], kind))
                    for X in embeddings[c]] for c in CONDITIONS
            }
        return diagram_cache[key]

    baseline_vectors = vectors(diagrams(), "persistence_image")
    baseline = summarize(baseline_vectors)
    results = {"distance": {}, "vectorization": {}, "probe_subset": {},
               "point_count": {}, "oracle": {}}

    for kind in ("chordal", "euclidean", "correlation"):
        row = baseline if kind == "chordal" else summarize(
            vectors(diagrams(kind), "persistence_image"))
        results["distance"][kind] = row
        show("distance", kind, row)

    for kind in ("persistence_image", "betti_curve", "top_k",
                 "persistence_entropy_total"):
        row = baseline if kind == "persistence_image" else summarize(
            vectors(diagrams(), kind))
        results["vectorization"][kind] = row
        show("vectorization", kind, row)

    for name, rows in (("all_600", range(600)), ("forget_only", range(300)),
                       ("neighbors_only", range(300, 600))):
        row = baseline if name == "all_600" else summarize(
            vectors(diagrams(rows=rows), "persistence_image"))
        results["probe_subset"][name] = row
        show("probe_subset", name, row)

    results["point_count"]["600"] = baseline
    show("point_count", "600", baseline)
    rng = np.random.default_rng(0)
    for size in (150, 300, 450):
        rows = rng.choice(600, size=size, replace=False)
        row = summarize(vectors(diagrams(rows=rows), "persistence_image"))
        results["point_count"][str(size)] = row
        show("point_count", str(size), row)

    results["oracle"]["distribution"] = baseline
    show("oracle", "distribution", baseline)
    single = {}
    for method in METHODS:
        ratios = []
        for target in baseline_vectors["retrain"]:
            numerator = np.median([np.linalg.norm(v - target)
                                   for v in baseline_vectors[method]])
            denominator = np.median([np.linalg.norm(v - target)
                                     for v in baseline_vectors["original"]])
            ratios.append(float(numerator / denominator))
        single[method] = {"min": min(ratios), "max": max(ratios)}
    results["oracle"]["single_retrain"] = single
    print("oracle/single_retrain: " + " ".join(
        f"{m}=[{single[m]['min']:+.4f},{single[m]['max']:+.4f}]"
        for m in METHODS), flush=True)

    (out / "ablations.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
