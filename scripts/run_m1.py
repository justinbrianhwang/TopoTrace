"""Milestone 1: synthetic ring — persistence diagrams + TRR anchors.

Run: conda run -n tda python scripts/run_m1.py
Outputs: results/m1/figure_m1.png, results/m1/metrics.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.models import get_embeddings
from topotrace.synthetic import make_forget_split, make_ring_dataset
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)
from topotrace.train import train

SEEDS = range(10)
LAYER = "h2"  # penultimate


def main():
    out = ROOT / "results" / "m1"
    out.mkdir(parents=True, exist_ok=True)

    # Fixed dataset and probe set across all models (plan §8.4)
    X, y = make_ring_dataset(seed=0)
    forget_idx, retain_idx = make_forget_split(X, y)
    all_idx = np.arange(len(X))
    # Local relative topology (plan §9.2): forget arc + its retain ring
    # neighborhood = all ring-class points. Forget-arc-only has no H1 signal.
    probe = X[y == 0]

    conditions = {"original": [], "retrain": [], "retrain2": [], "noop": [],
                  "finetune": []}
    for s in SEEDS:
        orig = train(X, y, all_idx, seed=s)
        conditions["original"].append(orig)
        conditions["noop"].append(orig)
        conditions["retrain"].append(train(X, y, retain_idx, seed=1000 + s))
        # independent retrain seeds: fair "exact retrain as method" anchor
        conditions["retrain2"].append(train(X, y, retain_idx, seed=3000 + s))
        conditions["finetune"].append(
            train(X, y, retain_idx, seed=2000 + s, epochs=50, init_model=orig))

    # Persistence diagrams on probe embeddings
    dgms = {name: [] for name in conditions}
    for name, models in conditions.items():
        for m in models:
            Z = get_embeddings(m, probe)[LAYER]
            D = chordal_distance_matrix(Z)
            dgms[name].append(persistence_diagrams(D))

    # Shared persistence-image grid, then TRR metrics (plan §10-11)
    imager = make_imager([d[1] for ds in dgms.values() for d in ds])
    vecs = {name: [vectorize(d, imager) for d in ds] for name, ds in dgms.items()}

    metrics = {}
    for name in ("noop", "finetune", "retrain2"):
        metrics[name] = trr_metrics(vecs["original"], vecs["retrain"], vecs[name])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # Figure: H0/H1 persistence diagrams, seed 0, four conditions
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    titles = ["original", "retrain2", "noop", "finetune"]
    for ax, name in zip(axes, titles):
        h0, h1 = dgms[name][0]
        ax.scatter(h0[:, 0], h0[:, 1], s=10, label="H0", alpha=0.6)
        if len(h1):
            ax.scatter(h1[:, 0], h1[:, 1], s=25, marker="^", label="H1", c="crimson")
        lim = ax.get_xlim()[1]
        ax.plot([0, lim], [0, lim], "k--", lw=0.5)
        ax.set_title(f"{name}\nTRR={metrics[name]['TRR']:.2f}" if name in metrics
                     else name)
        ax.legend(fontsize=8)
    fig.suptitle("Synthetic ring — ring-class probe persistence diagrams (seed 0)")
    fig.tight_layout()
    fig.savefig(out / "figure_m1.png", dpi=150)

    for name, m in metrics.items():
        print(f"{name:9s} TRR={m['TRR']:+.3f} alpha={m['alpha']:+.3f} "
              f"eta={m['eta']:.3f} I_topo={m['I_topo']:+.4f}")


if __name__ == "__main__":
    main()
