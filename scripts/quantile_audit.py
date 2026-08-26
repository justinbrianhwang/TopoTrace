"""Full audit protocol with pairwise-distance quantile fingerprints."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.stats import bootstrap_ci, permutation_pvalue
from topotrace.topology import chordal_distance_matrix

LAYERS = ("penultimate", "logits")
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")
APPROXIMATE = ("finetune", "neggrad", "scrub", "ssd")


def bh(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    ranked = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(ranked, 1)
    return q


def fingerprint(X):
    D = chordal_distance_matrix(X)
    return np.quantile(D[np.triu_indices(len(D), 1)], np.linspace(.02, .98, 50))


def main():
    summaries = []
    for archive_path in sorted((ROOT / "results").glob("*/embeddings.npz")):
        out, results, tests = archive_path.parent, {}, []
        with np.load(archive_path) as archive:
            for layer in LAYERS:
                keys = [k for k in archive.files if k.endswith(f"_{layer}")]
                conditions = sorted({k.rsplit("_", 2)[0] for k in keys})
                seeds = {c: sorted(int(k.rsplit("_", 2)[1]) for k in keys
                                   if k.startswith(f"{c}_")) for c in conditions}
                v = {c: [fingerprint(archive[f"{c}_{s}_{layer}"]) for s in seeds[c]]
                     for c in conditions}
                O, R = v["original"], v["retrain"]
                imprint = trr_metrics(O, R, R)
                ci = bootstrap_ci(O, R, lambda A, B: trr_metrics(A, B, B)["I_topo"])
                p = permutation_pvalue(O, R)
                anchor = trr_metrics(O, R, v["retrain2"])["TRR"]
                cell = {"I_topo": imprint["I_topo"], "CI": ci, "p": p,
                        "gate_open": p < .05 and ci[0] > 0,
                        "anchor_TRR": anchor, "anchor_admissible": abs(anchor) <= .5}
                results[layer] = cell
                if cell["gate_open"]:
                    cell["methods"] = {}
                    for method in METHODS:
                        if method in v:
                            metrics = trr_metrics(O, R, v[method])
                            row = {k: metrics[k] for k in ("TRR", "alpha", "eta")}
                            row["p"] = permutation_pvalue(v[method], R)
                            cell["methods"][method] = row
                            tests.append((layer, method, row))
        for (_, _, row), q in zip(tests, bh([row["p"] for _, _, row in tests])):
            row["q"] = float(q)
            row["reject"] = bool(q <= .05)
        (out / "quantile_audit.json").write_text(json.dumps(results, indent=2))
        for layer, cell in results.items():
            methods = cell.get("methods", {})
            worst = max((methods[m]["q"] for m in APPROXIMATE if m in methods),
                        default=None)
            sign = "positive" if cell["CI"][0] > 0 else (
                "negative" if cell["CI"][1] < 0 else "crosses_zero")
            summaries.append((out.name, layer, cell["p"], sign,
                              cell["anchor_TRR"], worst))

    print(f"{'scenario':25s} {'layer':11s} {'gate p':>9s} {'CI sign':>12s} "
          f"{'anchor TRR':>11s} {'worst BH q':>10s}")
    for scenario, layer, p, sign, anchor, worst in summaries:
        q = "-" if worst is None else f"{worst:.6f}"
        print(f"{scenario:25s} {layer:11s} {p:9.6f} {sign:>12s} "
              f"{anchor:+11.4f} {q:>10s}")


if __name__ == "__main__":
    main()
