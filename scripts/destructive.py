"""Evaluate parameter-noise destructive controls on saved M4 originals."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.metrics import trr_metrics
from topotrace.mnist import make_class_forget_split, make_random_forget_split
from topotrace.resnet import ResNet18C, evaluate, get_embeddings
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)
from topotrace.unlearn import noise_destroy

SIGMAS = (.02, .05, .1, .2, .5, 1.0)
CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "m4_random"
    scenario = out.name.removeprefix("m4_")
    X, y, X_test, y_test = load_cifar10()
    if scenario == "class":
        forget_idx, retain_idx = make_class_forget_split(y, cls=9)
    elif scenario in ("targeted", "matched"):
        splits = np.load(ROOT / "results" / "m4_splits.npz")
        forget_idx, retain_idx = (splits[f"{scenario}_forget"],
                                  splits[f"{scenario}_retain"])
    elif scenario.startswith("random"):
        frac = float(scenario[6:] or 5) / 100
        forget_idx, retain_idx = make_random_forget_split(y, frac=frac, seed=0)
    else:
        raise SystemExit(f"unknown M4 scenario: {scenario}")

    checkpoints = sorted((out / "models").glob("original_*.pt"),
                         key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    probe = X[np.load(out / "probe_idx.npy")]
    noise_diagrams = {sigma: [] for sigma in SIGMAS}
    accuracies = {sigma: {name: [] for name in ("retain", "forget", "test")}
                  for sigma in SIGMAS}

    for checkpoint in checkpoints:
        seed = int(checkpoint.stem.rsplit("_", 1)[1])
        original = ResNet18C()
        original.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                            weights_only=True))
        for sigma in SIGMAS:
            print(f"seed {seed} sigma {sigma:g}", flush=True)
            model = noise_destroy(original, sigma, seed)
            accuracies[sigma]["retain"].append(evaluate(model, X, y, retain_idx))
            accuracies[sigma]["forget"].append(evaluate(model, X, y, forget_idx))
            accuracies[sigma]["test"].append(evaluate(model, X_test, y_test))
            embedding = get_embeddings(model, probe)["penultimate"]
            noise_diagrams[sigma].append(persistence_diagrams(
                chordal_distance_matrix(embedding)))

    with np.load(out / "embeddings.npz") as archive:
        seeds = [int(path.stem.rsplit("_", 1)[1]) for path in checkpoints]
        diagrams = {
            name: [persistence_diagrams(chordal_distance_matrix(
                archive[f"{name}_{seed}_penultimate"])) for seed in seeds]
            for name in CONDITIONS
        }

    imager = make_imager([diagram[1] for values in
                          (*diagrams.values(), *noise_diagrams.values())
                          for diagram in values])
    vectors = {name: [vectorize(diagram, imager) for diagram in values]
               for name, values in diagrams.items()}
    results = {}
    for sigma in SIGMAS:
        metrics = trr_metrics(
            vectors["original"], vectors["retrain"],
            [vectorize(diagram, imager) for diagram in noise_diagrams[sigma]])
        results[str(sigma)] = {
            **{name: float(np.mean(values))
               for name, values in accuracies[sigma].items()},
            **{name: metrics[name] for name in ("TRR", "alpha", "eta")},
        }

    print(f"\n{'sigma':>5s} {'ret_acc':>8s} {'fgt_acc':>8s} {'test_acc':>8s} "
          f"{'TRR':>8s} {'alpha':>8s} {'eta':>8s}")
    for sigma, row in results.items():
        print(f"{sigma:>5s} {row['retain']:8.4f} {row['forget']:8.4f} "
              f"{row['test']:8.4f} {row['TRR']:+8.3f} "
              f"{row['alpha']:+8.3f} {row['eta']:8.3f}")
    (out / "destructive.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
