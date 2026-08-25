"""Run the full synthetic M1 benchmark suite on CPU."""
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
from topotrace.stats import permutation_pvalue
from topotrace.synthetic import (make_bridge, make_canary, make_figure_eight,
                                 make_forget_split, make_ring_dataset,
                                 make_satellites, split_bridge, split_canary,
                                 split_figure_eight, split_satellites)
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)
from topotrace.train import train

SEEDS = range(10)
BENCHMARKS = {
    "ring": (make_ring_dataset, make_forget_split, 1),
    "satellites": (make_satellites, split_satellites, 0),
    "bridge": (make_bridge, split_bridge, 0),
    "figure_eight": (make_figure_eight, split_figure_eight, 1),
    "canary": (make_canary, split_canary, 1),
}
METHODS = ("noop", "retrain2", "finetune")


def predictions(model, X):
    return get_embeddings(model, X)["logits"].argmax(1)


def plot_diagram(ax, dgm, title, limit):
    ax.scatter(dgm[:, 0], dgm[:, 1], s=12)
    ax.plot([0, limit], [0, limit], "k--", lw=0.5)
    ax.set(xlim=(-0.02 * limit, limit), ylim=(-0.02 * limit, limit),
           title=title, xlabel="birth", ylabel="death")


def main():
    out = ROOT / "results" / "m1_suite"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    fig, axes = plt.subplots(len(BENCHMARKS), 3, figsize=(12, 18))

    for row, (name, (make_data, split_data, plot_dim)) in enumerate(BENCHMARKS.items()):
        X, y = make_data(0)
        forget_idx, retain_idx = split_data(X, y)
        all_idx = np.arange(len(y))
        probe = X[y == 0]
        conditions = {key: [] for key in ("original", "retrain", "retrain2",
                                           "noop", "finetune")}
        for seed in SEEDS:
            original = train(X, y, all_idx, seed=seed)
            conditions["original"].append(original)
            conditions["noop"].append(original)
            conditions["retrain"].append(
                train(X, y, retain_idx, seed=1000 + seed))
            conditions["retrain2"].append(
                train(X, y, retain_idx, seed=3000 + seed))
            conditions["finetune"].append(
                train(X, y, retain_idx, seed=2000 + seed, epochs=50,
                      init_model=original))

        diagrams = {
            key: [persistence_diagrams(chordal_distance_matrix(
                get_embeddings(model, probe)["h2"])) for model in models]
            for key, models in conditions.items()
        }
        X_test, y_test = make_data(777)
        benchmark = {
            "test_accuracy": float(np.mean([
                (predictions(model, X_test) == y_test).mean()
                for model in conditions["original"]
            ])),
            "dimensions": {},
        }
        if name == "canary":
            benchmark["forget_prediction_agreement"] = float(np.mean([
                (predictions(original, X[forget_idx]) ==
                 predictions(retrain, X[forget_idx])).mean()
                for original, retrain in zip(conditions["original"],
                                             conditions["retrain"])
            ]))

        for dim in (0, 1):
            imager = make_imager([
                dgm[dim] for group in diagrams.values() for dgm in group
            ])
            vectors = {
                key: [vectorize(dgm, imager, dim) for dgm in group]
                for key, group in diagrams.items()
            }
            baseline = trr_metrics(vectors["original"], vectors["retrain"],
                                   vectors["original"])
            cell = {
                "I_topo": baseline["I_topo"],
                "p": permutation_pvalue(vectors["original"], vectors["retrain"]),
                "methods": {},
            }
            for method in METHODS:
                values = trr_metrics(vectors["original"], vectors["retrain"],
                                     vectors[method])
                cell["methods"][method] = {
                    key: values[key] for key in ("TRR", "alpha", "eta")
                }
            benchmark["dimensions"][f"H{dim}"] = cell
        results[name] = benchmark

        ax = axes[row, 0]
        ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="coolwarm", alpha=0.55)
        ax.scatter(X[forget_idx, 0], X[forget_idx, 1], s=18, facecolors="none",
                   edgecolors="gold", label="forget")
        ax.set(title=name, aspect="equal")
        ax.legend(fontsize=8)
        shown = [diagrams[key][0][plot_dim] for key in ("original", "retrain")]
        limit = max(float(dgm.max()) for dgm in shown if dgm.size) * 1.05
        plot_diagram(axes[row, 1], shown[0], f"original H{plot_dim}", limit)
        plot_diagram(axes[row, 2], shown[1], f"retrain H{plot_dim}", limit)

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    fig.tight_layout()
    fig.savefig(out / "figure_suite.png", dpi=150)

    print(f"{'benchmark':12s} dim method     I_topo       p    TRR  alpha    eta   acc agree")
    for name, benchmark in results.items():
        agree = benchmark.get("forget_prediction_agreement")
        for dim, cell in benchmark["dimensions"].items():
            for method, values in cell["methods"].items():
                print(f"{name:12s} {dim:>2s}  {method:8s} {cell['I_topo']:7.4f} "
                      f"{cell['p']:7.4f} {values['TRR']:6.2f} "
                      f"{values['alpha']:6.2f} {values['eta']:6.2f} "
                      f"{benchmark['test_accuracy']:5.3f} "
                      f"{agree if agree is not None else '':>5}")


if __name__ == "__main__":
    main()
