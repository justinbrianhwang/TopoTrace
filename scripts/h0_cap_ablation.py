"""Ablate conventions for the essential H0 persistence bar."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.stats import permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
METHODS = ("noop", "finetune", "neggrad", "scrub", "ssd")


def main():
    out = ROOT / "results" / "m4_random"
    with np.load(out / "embeddings.npz") as archive:
        seeds = sorted(int(k.rsplit("_", 2)[1]) for k in archive.files
                       if k.startswith("original_") and k.endswith("_penultimate"))
        distances = {c: [chordal_distance_matrix(
            archive[f"{c}_{s}_penultimate"]) for s in seeds] for c in CONDITIONS}
    current = {c: [persistence_diagrams(D) for D in ds]
               for c, ds in distances.items()}
    global_cap = max(float(D.max()) for ds in distances.values() for D in ds)
    diagrams = {
        "per_model_cap": current,
        "drop_essential": {c: [[d[0][:-1]] for d in ds] for c, ds in current.items()},
        "global_cap": {c: [[np.vstack((d[0][:-1], (0., global_cap)))] for d in ds]
                       for c, ds in current.items()},
    }
    results = {"global_cap": global_cap, "conventions": {}}
    essential_fraction = None
    for name, by_condition in diagrams.items():
        imager = make_imager([d[0] for c in ("original", "retrain")
                              for d in by_condition[c]])
        v = {c: [vectorize(d, imager, dim=0) for d in ds]
             for c, ds in by_condition.items()}
        O, R = v["original"], v["retrain"]
        imprint = trr_metrics(O, R, R)
        row = {"I_topo": imprint["I_topo"],
               "gate_p": permutation_pvalue(O, R),
               "anchor_TRR": trr_metrics(O, R, v["retrain2"])["TRR"],
               "method_TRR": {m: trr_metrics(O, R, v[m])["TRR"] for m in METHODS}}
        results["conventions"][name] = row
        if name == "per_model_cap":
            full = [x for values in v.values() for x in values]
            essential = [vectorize([d[0][-1:]], imager, dim=0)
                         for ds in current.values() for d in ds]
            essential_fraction = float(np.mean(np.linalg.norm(essential, axis=1)) /
                                       np.mean(np.linalg.norm(full, axis=1)))
    results["essential_bar_norm_fraction"] = essential_fraction
    (out / "h0_cap_ablation.json").write_text(json.dumps(results, indent=2))

    print(f"{'convention':16s} {'I_topo':>10s} {'gate p':>9s} {'anchor':>9s}  methods")
    for name, row in results["conventions"].items():
        methods = " ".join(f"{m}={v:+.3f}" for m, v in row["method_TRR"].items())
        print(f"{name:16s} {row['I_topo']:+10.6f} {row['gate_p']:9.6f} "
              f"{row['anchor_TRR']:+9.3f}  {methods}")
    print(f"essential-bar mean-norm fraction: {essential_fraction:.6f}")


if __name__ == "__main__":
    main()
