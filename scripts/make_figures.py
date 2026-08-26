"""Render the paper's data figures from cached JSON results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS, OUT = ROOT / "results", ROOT / "paper" / "figures"
COLORS = {"retrain": "#2166ac", "retrain2": "#2166ac", "original": "#555555",
          "noop": "#555555", "finetune": "#e08214", "neggrad": "#b2182b",
          "scrub": "#7b3294", "ssd": "#35978f"}
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")
LEGEND = {"frameon": False, "fontsize": 7, "handlelength": 1.5,
          "columnspacing": .8, "handletextpad": .4}
MANIFEST = []


def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": .6, "pdf.fonttype": 42,
                         "legend.frameon": False})


def load(path):
    return json.loads(path.read_text())


def save(fig, name, inputs):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"),
                     "inputs": [str(p.relative_to(ROOT)).replace("\\", "/") for p in inputs]})


def fig_imprint_gate():
    cases = [("MNIST\nclass", "m2_class"),
             ("Fashion\nrandom", "exp_fashionmnist_random"),
             ("Fashion\nclass", "exp_fashionmnist_class"),
             ("SVHN\nrandom", "exp_svhn_random"),
             ("SVHN\nclass", "exp_svhn_class"),
             ("C10\nrandom 1%", "m4_random1"),
             ("C10\nrandom 5%", "m4_random"),
             ("C10\nrandom 10%", "m4_random10"),
             ("C10\nclass", "m4_class"),
             ("C10\ntargeted", "m4_targeted"),
             ("C10\nmatched", "m4_matched"),
             ("C100\nclass", "exp_cifar100_class")]
    found = [(label, RESULTS / folder / "analysis.json") for label, folder in cases
             if (RESULTS / folder / "analysis.json").exists()]
    if not found:
        print("skip fig_imprint_gate.pdf: no analysis.json inputs")
        return
    data = [(label, load(path)) for label, path in found]
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.5), sharex=True)
    x = np.arange(len(data))
    for ax, key, title in zip(axes, ("penultimate_H1", "penultimate_H0"),
                              ("Penultimate H1", "Penultimate H0")):
        rows = [d[key] for _, d in data]
        y = np.array([r["I_topo"] for r in rows])
        ci = np.array([r["CI"] for r in rows])
        bars = ax.bar(x, y, color="#2166ac", edgecolor="white", linewidth=.4,
                      yerr=np.vstack((y - ci[:, 0], ci[:, 1] - y)), capsize=2)
        for bar, row in zip(bars, rows):
            if row["p"] < .05:
                bar.set_hatch("///")
                offset = .04 * max(np.ptp(y), max(abs(y)), 1e-6)
                ax.text(bar.get_x() + bar.get_width() / 2,
                        row["CI"][1] + offset, "*", ha="center", va="bottom")
        ax.axhline(0, color="#555555", lw=.6)
        ax.set(title=title, ylabel=r"$I_{topo}$")
    axes[-1].set_xticks(x, [label for label, _ in data], rotation=35, ha="right")
    fig.tight_layout()
    save(fig, "fig_imprint_gate.pdf", [path for _, path in found])


def fig_layer_profile():
    path = RESULTS / "m4_random" / "layer_profile.json"
    if not path.exists():
        print(f"skip fig_layer_profile.pdf: missing {path.relative_to(ROOT)}")
        return
    data = load(path)
    layers = list(data["H0"])
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), sharey=True)
    for ax, hom in zip(axes, ("H0", "H1")):
        rows = data[hom]
        hollow = np.array([rows[layer]["p"] >= .05 for layer in layers])
        for method in METHODS:
            y = np.clip([rows[layer]["methods"][method]["TRR"] for layer in layers], -1, 5)
            ax.plot(x, y, color=COLORS[method], lw=1, label=method)
            ax.scatter(x[~hollow], y[~hollow], color=COLORS[method], s=15, zorder=3)
            ax.scatter(x[hollow], y[hollow], facecolors="white", edgecolors=COLORS[method],
                       linewidths=.8, s=15, zorder=3)
        ax.set(title=hom, xlabel="Layer", xticks=x, xticklabels=layers, ylim=(-1, 5))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("TRR")
    axes[1].legend(ncol=2, **LEGEND)
    fig.tight_layout()
    save(fig, "fig_layer_profile.pdf", [path])


def fig_progress_artifact():
    scenarios = {"random": "o", "class": "s", "targeted": "^", "matched": "D"}
    paths = {name: RESULTS / f"m4_{name}" / "metrics.json" for name in scenarios}
    destructive = RESULTS / "m4_random" / "destructive.json"
    missing = [p for p in [*paths.values(), destructive] if not p.exists()]
    if missing:
        print("skip fig_progress_artifact.pdf: missing " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
        return
    data = {name: load(path) for name, path in paths.items()}
    noise = load(destructive)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    for scenario, marker in scenarios.items():
        for method in METHODS:
            row = data[scenario][method]
            ax.scatter(row["alpha"], row["eta"], color=COLORS[method], marker=marker,
                       s=30, linewidths=.5, edgecolors="white")
    sigmas = sorted(noise, key=float)
    ax.plot([noise[s]["alpha"] for s in sigmas], [noise[s]["eta"] for s in sigmas],
            color="#777777", marker="x", lw=.7, ms=5)
    for sigma in ("0.2", "0.5", "1.0"):
        row = noise[sigma]
        ax.annotate(rf"$\sigma$={sigma}", (row["alpha"], row["eta"]),
                    xytext=(3, -8) if sigma == "0.2" else (3, 2),
                    textcoords="offset points", fontsize=6, color="#555555")
    ax.axvline(0, color="#aaaaaa", lw=.6)
    ax.axvline(1, color="#aaaaaa", lw=.6)
    ax.axhline(0, color="#aaaaaa", lw=.6)
    ax.set(xlabel=r"progress $\alpha$", ylabel=r"artifact $\eta$")
    method_handles = [Line2D([], [], marker="o", ls="", color=COLORS[m], label=m) for m in METHODS]
    scenario_handles = [Line2D([], [], marker=mark, ls="", color="#333333", label=name)
                        for name, mark in scenarios.items()]
    method_legend = {**LEGEND, "fontsize": 6, "handlelength": 1,
                     "columnspacing": .6, "handletextpad": .2}
    first = ax.legend(handles=method_handles, ncol=6, loc="lower center",
                      bbox_to_anchor=(.5, 1.14), **method_legend)
    ax.add_artist(first)
    ax.legend(handles=scenario_handles, ncol=4, loc="lower center",
              bbox_to_anchor=(.5, 1.04), **LEGEND)
    fig.tight_layout(rect=(0, 0, 1, .78))
    save(fig, "fig_progress_artifact.pdf", [*paths.values(), destructive])


def fig_metric_disagreement():
    base = RESULTS / "m4_random"
    paths = [base / name for name in ("metrics.json", "mia.json", "pointwise.json", "formal_stats.json")]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("skip fig_metric_disagreement.pdf: missing " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
        return
    metrics, mia, pointwise, formal = map(load, paths)
    methods = ("finetune", "neggrad", "scrub", "ssd")
    matrix = []
    retrain_forget = metrics["retrain2"]["acc"]["forget"]
    for method in methods:
        row = pointwise["methods"][method]
        matrix.append([abs(metrics[method]["acc"]["forget"] - retrain_forget) < .02,
                       abs(np.mean(mia[method]) - .5) < .02,
                       row["cosine"]["p"] >= .05, row["cka"]["p"] >= .05,
                       row["knn_overlap"]["p"] >= .05,
                       formal["bh_fdr"]["cells"]["penultimate_H1"][method]["q"] >= .05])
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6.9, 2.2))
    ax.imshow(matrix, cmap=ListedColormap(["#d6604d", "#5aae61"]), vmin=0, vmax=1,
              aspect="auto")
    for i, row in enumerate(matrix):
        for j, passed in enumerate(row):
            ax.text(j, i, "pass" if passed else "flag", ha="center", va="center",
                    color="white", fontsize=7)
    ax.set(xticks=range(6), yticks=range(4), yticklabels=methods,
           xticklabels=("forget-acc\ngap", "MIA", "cosine", "CKA", "kNN", "TopoTrace"))
    ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, "fig_metric_disagreement.pdf", paths)


def fig_ratio_dose():
    paths = [RESULTS / folder / "analysis.json" for folder in
             ("m4_random1", "m4_random", "m4_random10")]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("skip fig_ratio_dose.pdf: missing " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
        return
    data = list(map(load, paths))
    ratios = np.array([1, 5, 10])
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for key, label, color, marker in (("penultimate_H0", "H0", "#555555", "o"),
                                       ("penultimate_H1", "H1", "#2166ac", "s")):
        rows = [d[key] for d in data]
        y = np.array([r["I_topo"] for r in rows])
        ci = np.array([r["CI"] for r in rows])
        ax.errorbar(ratios, y, yerr=np.vstack((y - ci[:, 0], ci[:, 1] - y)),
                    color=color, marker=marker, capsize=2, lw=1, label=label)
    ax.axhline(0, color="#aaaaaa", lw=.6)
    ax.set(xlabel="Deletion ratio (%)", ylabel=r"$I_{topo}$", xticks=ratios)
    ax.legend(**LEGEND)
    fig.tight_layout()
    save(fig, "fig_ratio_dose.pdf", paths)


def fig_distinguisher():
    path = RESULTS / "m4_random" / "distinguisher.json"
    if not path.exists():
        print(f"skip fig_distinguisher.pdf: missing {path.relative_to(ROOT)}")
        return
    data = load(path)
    methods = ("finetune", "neggrad", "scrub", "ssd", "noop")
    features = list(data[methods[0]])
    x, width = np.arange(len(features)), .16
    fig, ax = plt.subplots(figsize=(6.9, 2.6))
    for i, method in enumerate(methods):
        ax.bar(x + (i - 2) * width, [data[method][feature]["auc"] for feature in features],
               width, color=COLORS[method], label=method)
    ax.axhline(.5, color="#555555", ls="--", lw=.7)
    ax.set(ylabel="Leave-one-seed-out AUC", xticks=x, xticklabels=features, ylim=(.45, 1.03))
    ax.legend(ncol=5, **LEGEND)
    fig.tight_layout()
    save(fig, "fig_distinguisher.pdf", [path])


def fig_destructive():
    path = RESULTS / "m4_random" / "destructive.json"
    if not path.exists():
        print(f"skip fig_destructive.pdf: missing {path.relative_to(ROOT)}")
        return
    data = load(path)
    sigma = np.array(sorted(map(float, data)))
    rows = [data[str(s)] if str(s) in data else data[f"{s:.1f}"] for s in sigma]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6))
    for ax, key, label, color in zip(axes, ("TRR", "eta"), ("TRR", r"artifact $\eta$"),
                                      ("#2166ac", "#b2182b")):
        ax.plot(sigma, [r[key] for r in rows], color=color, marker="o", lw=1, label=label)
        ax.set(xlabel=r"noise $\sigma$", ylabel=label)
        test_ax = ax.twinx()
        test_ax.plot(sigma, [r["test"] for r in rows], color="#777777", marker="s",
                     ms=3, ls="--", lw=.8, label="test accuracy")
        test_ax.set_ylabel("Test accuracy", color="#555555")
        handles = ax.lines + test_ax.lines
        ax.legend(handles, [line.get_label() for line in handles], loc="best", **LEGEND)
    fig.tight_layout()
    save(fig, "fig_destructive.pdf", [path])


def fig_relearn():
    path = RESULTS / "m4_class" / "relearn.json"
    if not path.exists():
        print(f"skip fig_relearn.pdf: missing {path.relative_to(ROOT)}")
        return
    data = load(path)
    conditions = ("retrain", "retrain2", "finetune", "neggrad", "scrub", "ssd")
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    for condition in conditions:
        curves = np.asarray(data[condition]["step_curves"], dtype=float)
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        steps = 5 * np.arange(1, len(mean) + 1)
        ls = "--" if condition == "retrain2" else "-"
        ax.plot(steps, mean, color=COLORS[condition], ls=ls, lw=1, label=condition)
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[condition], alpha=.12)
    ax.set(xlabel="Optimizer step", ylabel="Forget accuracy", ylim=(0, 1.03))
    ax.legend(ncol=2, **LEGEND)
    fig.tight_layout()
    save(fig, "fig_relearn.pdf", [path])


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (fig_imprint_gate, fig_layer_profile, fig_progress_artifact,
                 fig_metric_disagreement, fig_ratio_dose, fig_distinguisher,
                 fig_destructive, fig_relearn):
        draw()
    (OUT / "figures_manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    print(f"wrote {len(MANIFEST)} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
