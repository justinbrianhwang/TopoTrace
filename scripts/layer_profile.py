"""Layerwise topological profile for saved CIFAR-10 ResNet checkpoints."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.metrics import trr_metrics
from topotrace.resnet import ResNet18C, get_all_embeddings
from topotrace.stats import permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

LAYERS = ("stem", "layer1", "layer2", "layer3", "layer4",
          "penultimate", "logits")
CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "m4_random"
    X, *_ = load_cifar10()
    probe = X[np.load(out / "probe_idx.npy")]
    checkpoints = sorted((out / "models").glob("*.pt"),
                         key=lambda p: (p.stem.rsplit("_", 1)[0],
                                        int(p.stem.rsplit("_", 1)[1])))
    embeddings = {}
    diagrams = {layer: {condition: [] for condition in CONDITIONS}
                for layer in LAYERS}

    print("extracting embeddings ...", flush=True)
    for path in checkpoints:
        condition, seed = path.stem.rsplit("_", 1)
        model = ResNet18C()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        values = get_all_embeddings(model, probe)
        for layer in LAYERS:
            embeddings[f"{condition}_{seed}_{layer}"] = values[layer]
            diagrams[layer][condition].append(persistence_diagrams(
                chordal_distance_matrix(values[layer])))
        if condition == "original":
            for layer in LAYERS:
                embeddings[f"noop_{seed}_{layer}"] = values[layer]
                diagrams[layer]["noop"].append(diagrams[layer][condition][-1])
        print(path.stem, flush=True)
    np.savez_compressed(out / "embeddings_layers.npz", **embeddings)

    results = {f"H{hom}": {} for hom in (0, 1)}
    for hom in (0, 1):
        for layer in LAYERS:
            layer_diagrams = diagrams[layer]
            imager = make_imager([d[hom] for ds in layer_diagrams.values() for d in ds])
            vectors = {condition: [vectorize(d, imager, hom) for d in ds]
                       for condition, ds in layer_diagrams.items()}
            metrics = {method: trr_metrics(vectors["original"], vectors["retrain"],
                                           vectors[method])
                       for method in METHODS}
            results[f"H{hom}"][layer] = {
                "I_topo": metrics["noop"]["I_topo"],
                "p": permutation_pvalue(vectors["original"], vectors["retrain"]),
                "methods": metrics,
            }
    (out / "layer_profile.json").write_text(json.dumps(results, indent=2))

    for hom in (0, 1):
        print(f"\nH{hom}  " + " ".join(f"{m:>9}" for m in METHODS) +
              "     I_topo         p")
        for layer in LAYERS:
            row = results[f"H{hom}"][layer]
            print(f"{layer:11}" + " ".join(
                f"{row['methods'][m]['TRR']:+9.3f}" for m in METHODS) +
                f"  {row['I_topo']:+10.6f} {row['p']:9.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    x = np.arange(len(LAYERS))
    for hom, ax in enumerate(axes):
        rows = results[f"H{hom}"]
        hollow = np.array([rows[layer]["p"] >= .05 for layer in LAYERS])
        for method in METHODS:
            y = np.clip([rows[layer]["methods"][method]["TRR"]
                         for layer in LAYERS], -1, 5)
            line, = ax.plot(x, y, label=method)
            ax.scatter(x[~hollow], y[~hollow], color=line.get_color(), s=24)
            ax.scatter(x[hollow], y[hollow], facecolors="none",
                       edgecolors=line.get_color(), s=24)
        ax.set(title=f"H{hom}", xlabel="Layer", xticks=x,
               xticklabels=LAYERS, ylim=(-1, 5))
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("TRR")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure_layer_profile.png", dpi=200)


if __name__ == "__main__":
    main()
