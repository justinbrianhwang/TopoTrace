"""Analyze the M2 layer and homology sweep from saved embeddings."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.stats import bootstrap_ci, permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")
LAYERS = ("penultimate", "logits")


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "m2_class"
    results = {}

    with np.load(out / "embeddings.npz") as embeddings:
        for layer in LAYERS:
            seeds = sorted(int(key.split("_")[1]) for key in embeddings.files
                           if key.startswith("original_") and key.endswith(f"_{layer}"))
            diagrams = {
                condition: [persistence_diagrams(chordal_distance_matrix(
                    embeddings[f"{condition}_{seed}_{layer}"])) for seed in seeds]
                for condition in CONDITIONS
            }

            for hom_dim in (0, 1):
                # frozen-grid protocol: the persistence-image grid is fitted
                # on original+oracle diagrams ONLY, before any unlearned
                # model is vectorized (audit independence)
                imager = make_imager([d[hom_dim] for c in ("original", "retrain")
                                      for d in diagrams[c]])
                vectors = {name: [vectorize(d, imager, dim=hom_dim) for d in ds]
                           for name, ds in diagrams.items()}
                v_O, v_R = vectors["original"], vectors["retrain"]
                imprint = trr_metrics(v_O, v_R, v_R)
                ci = bootstrap_ci(
                    v_O, v_R,
                    lambda A, B: trr_metrics(A, B, B)["I_topo"])
                p = permutation_pvalue(v_O, v_R)
                key = f"{layer}_H{hom_dim}"
                cell = {
                    "D_RR": imprint["D_RR"], "D_OR": imprint["D_OR"],
                    "I_topo": imprint["I_topo"], "CI": ci, "p": p,
                }
                results[key] = cell
                print(f"{layer:11s} H{hom_dim} D_RR={cell['D_RR']:.6f} "
                      f"D_OR={cell['D_OR']:.6f} I_topo={cell['I_topo']:+.6f} "
                      f"CI=({ci[0]:+.6f}, {ci[1]:+.6f}) p={p:.6f}")

                if p < 0.05:
                    cell["methods"] = {}
                    print(f"  {'method':9s} {'TRR':>9s} {'alpha':>9s} "
                          f"{'eta':>9s} {'p':>9s}")
                    for method in METHODS:
                        metrics = trr_metrics(v_O, v_R, vectors[method])
                        method_p = permutation_pvalue(vectors[method], v_R)
                        row = {name: metrics[name] for name in ("TRR", "alpha", "eta")}
                        row["p"] = method_p
                        cell["methods"][method] = row
                        print(f"  {method:9s} {row['TRR']:+9.4f} "
                              f"{row['alpha']:+9.4f} {row['eta']:9.4f} {method_p:9.6f}")

    (out / "analysis.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
