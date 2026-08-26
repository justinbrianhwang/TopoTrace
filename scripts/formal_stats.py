"""Formal statistical analyses from cached embeddings."""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.stats import bootstrap_ci, permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")


def d_ur(U, R):
    return float(np.median([np.linalg.norm(u - r) for u in U for r in R]))


def bh(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1)
    return q


def main():
    out = Path(sys.argv[1])
    with np.load(out / "embeddings.npz") as archive:
        seeds = sorted(int(k.split("_")[1]) for k in archive.files
                       if k.startswith("original_") and k.endswith("_penultimate"))
        embeddings = {layer: {c: [archive[f"{c}_{s}_{layer}"] for s in seeds]
                              for c in CONDITIONS}
                      for layer in ("penultimate", "logits")}

    diagrams = {layer: {c: [persistence_diagrams(chordal_distance_matrix(X))
                            for X in models]
                        for c, models in by_condition.items()}
                for layer, by_condition in embeddings.items()}
    vectors, equivalence = {}, {}
    for layer in ("penultimate", "logits"):
        for hom in (0, 1):
            key = f"{layer}_H{hom}"
            imager = make_imager([d[hom] for c in ("original", "retrain") for d in diagrams[layer][c]])
            v = {c: [vectorize(d, imager, dim=hom) for d in ds]
                 for c, ds in diagrams[layer].items()}
            vectors[key] = v
            rr = [np.linalg.norm(a - b) for a, b in combinations(v["retrain"], 2)]
            delta = float(np.percentile(rr, 95))
            rows = {}
            for method in METHODS:
                ci = bootstrap_ci(v[method], v["retrain"], d_ur)
                decision = ("oracle-equivalent" if ci[1] < delta else
                            "outside" if ci[0] > delta else "inconclusive")
                rows[method] = {"D_UR": d_ur(v[method], v["retrain"]),
                                "CI": ci, "decision": decision}
            equivalence[key] = {"delta": delta, "methods": rows}

    analysis_path = out / "analysis.json"
    analysis = json.loads(analysis_path.read_text()) if analysis_path.exists() else {}
    labels, pvalues = [], []
    for cell, v in vectors.items():
        for method in METHODS:
            labels.append((cell, method))
            saved = analysis.get(cell, {}).get("methods", {}).get(method, {})
            pvalues.append(saved["p"] if "p" in saved else
                           permutation_pvalue(v[method], v["retrain"]))
    qvalues = bh(pvalues)
    fdr = {cell: {} for cell in vectors}
    for (cell, method), p, q in zip(labels, pvalues, qvalues):
        fdr[cell][method] = {"p": float(p), "q": float(q), "reject": bool(q <= .05)}

    metrics = json.loads((out / "metrics.json").read_text())
    mia = json.loads((out / "mia.json").read_text())
    relearn = json.loads((out / "relearn.json").read_text())
    destroyed = [m for m in METHODS if metrics[m]["acc"]["retain"] < .5]
    included = [m for m in METHODS if m not in destroyed and m in mia and m in relearn]
    missing = [m for m in METHODS if m not in destroyed and m not in included]
    v = vectors["penultimate_H1"]
    topo, mia_auc, relearn_auc = [], [], []
    for method in included:
        for i, model in enumerate(v[method]):
            topo.append(d_ur([model], v["retrain"]) /
                        d_ur([v["original"][i]], v["retrain"]))
            mia_auc.append(mia[method][i])
            relearn_auc.append(relearn[method]["auc"][i])
    correlations = {"included": included, "excluded_destroyed": destroyed,
                    "excluded_missing_outcomes": missing, "n": len(topo)}
    for name, outcome in (("mia_auc", mia_auc), ("relearn_auc", relearn_auc)):
        rho, p = spearmanr(topo, outcome)
        correlations[name] = {"rho": float(rho), "p": float(p)}

    rng = np.random.default_rng(0)
    audit_values = []
    for _ in range(20):
        rows = rng.integers(len(embeddings["penultimate"]["original"][0]),
                           size=len(embeddings["penultimate"]["original"][0]))
        ds = {c: [persistence_diagrams(chordal_distance_matrix(X[rows]))
                  for X in embeddings["penultimate"][c]] for c in CONDITIONS}
        imager = make_imager([d[1] for c in ("original", "retrain") for d in ds[c]])
        boot = {c: [vectorize(d, imager) for d in ds[c]]
                for c in ("original", "retrain")}
        audit_values.append(trr_metrics(boot["original"], boot["retrain"],
                                        boot["retrain"])["I_topo"])
    audit = {"n": 20, "mean": float(np.mean(audit_values)),
             "std": float(np.std(audit_values, ddof=1)), "values": audit_values}

    results = {"equivalence": equivalence,
               "bh_fdr": {"alpha": .05, "cells": fdr},
               "correlations": correlations, "audit_bootstrap": audit}
    print("equivalence")
    for cell, result in equivalence.items():
        print(f"{cell} delta={result['delta']:.6g}")
        print("  " + " ".join(f"{m}={r['D_UR']:.4g}[{r['decision']}]"
                              for m, r in result["methods"].items()))
    print("BH-FDR")
    for cell, rows in fdr.items():
        print(f"{cell}: " + " ".join(
            f"{m}={r['q']:.4g}{'*' if r['reject'] else ''}" for m, r in rows.items()))
    print("correlations " + " ".join(
        f"{name}=rho:{correlations[name]['rho']:+.4f},p:{correlations[name]['p']:.4g}"
        for name in ("mia_auc", "relearn_auc")))
    print(f"excluded destroyed={destroyed} missing outcomes={missing}")
    print(f"audit penultimate_H1 I_topo={audit['mean']:.6g}+/-{audit['std']:.6g}")
    (out / "formal_stats.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
