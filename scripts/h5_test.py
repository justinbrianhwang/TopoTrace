"""Direct bootstrap test of targeted versus class-matched imprints."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)


def imprint(O, R):
    return trr_metrics(O, R, R)["I_topo"]


def main():
    vectors = {}
    for scenario in ("targeted", "matched"):
        with np.load(ROOT / "results" / f"m4_{scenario}" / "embeddings.npz") as archive:
            seeds = sorted(int(k.rsplit("_", 2)[1]) for k in archive.files
                           if k.startswith("original_") and k.endswith("_penultimate"))
            for layer in ("penultimate", "logits"):
                diagrams = {c: [persistence_diagrams(chordal_distance_matrix(
                    archive[f"{c}_{s}_{layer}"])) for s in seeds]
                            for c in ("original", "retrain")}
                for hom in (0, 1):
                    imager = make_imager([d[hom] for ds in diagrams.values() for d in ds])
                    vectors[(scenario, layer, hom)] = {
                        c: [vectorize(d, imager, dim=hom) for d in ds]
                        for c, ds in diagrams.items()}

    rng, results = np.random.default_rng(0), {}
    for layer in ("penultimate", "logits"):
        for hom in (0, 1):
            T, M = (vectors[(s, layer, hom)] for s in ("targeted", "matched"))
            targeted, matched = imprint(T["original"], T["retrain"]), imprint(
                M["original"], M["retrain"])
            boot = []
            for _ in range(1000):
                groups = []
                for v in (T, M):
                    groups.append(imprint(
                        [v["original"][i] for i in rng.integers(len(v["original"]),
                                                                 size=len(v["original"]))],
                        [v["retrain"][i] for i in rng.integers(len(v["retrain"]),
                                                                size=len(v["retrain"]))]))
                boot.append(groups[0] - groups[1])
            ci = [float(x) for x in np.quantile(boot, [.025, .975])]
            results[f"{layer}_H{hom}"] = {
                "targeted_I_topo": targeted, "matched_I_topo": matched,
                "difference": targeted - matched, "CI": ci,
                "excludes_zero": ci[0] > 0 or ci[1] < 0,
            }
    path = ROOT / "results" / "h5_difference.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"{'cell':16s} {'difference':>11s} {'95% CI':>25s} {'excludes 0':>11s}")
    for cell, row in results.items():
        print(f"{cell:16s} {row['difference']:+11.6f} "
              f"({row['CI'][0]:+.6f}, {row['CI'][1]:+.6f}) "
              f"{str(row['excludes_zero']):>11s}")


if __name__ == "__main__":
    main()
