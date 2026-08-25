"""Analyze augmentation-orbit topology for saved CIFAR-10 checkpoints."""

import argparse
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
from topotrace.resnet import ResNet18C, _augment, get_embeddings
from topotrace.stats import permutation_pvalue
from topotrace.topology import chordal_distance_matrix, persistence_diagrams

CONDITIONS = ("original", "retrain", "retrain2", "finetune", "neggrad",
              "scrub", "ssd")
METHODS = ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd")


def orbit_stats(embeddings):
    diagram = persistence_diagrams(
        chordal_distance_matrix(embeddings), maxdim=0)[0]
    persistence = diagram[:, 1] - diagram[:, 0]
    persistence = persistence[persistence > 0]
    total = persistence.sum()
    entropy = -np.sum((persistence / total) * np.log(persistence / total)) \
        if total else 0.0
    return np.array([total, entropy])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--n-augs", type=int, default=64)
    args = parser.parse_args()
    if args.n_samples < 1 or args.n_augs < 1:
        parser.error("n-samples and n-augs must be positive")

    X, y, _, _ = load_cifar10()
    scenario = args.result_dir.name.rsplit("_", 1)[-1]
    if scenario == "class":
        forget_idx, _ = make_class_forget_split(y, 9)
    elif scenario.startswith("random"):
        forget_idx, _ = make_random_forget_split(
            y, float(scenario[6:] or 5) / 100, 0)
    elif scenario in ("targeted", "matched"):
        forget_idx = np.load(ROOT / "results" / "m4_splits.npz")[
            f"{scenario}_forget"]
    else:
        parser.error("unknown scenario suffix in result directory name")
    if args.n_samples > len(forget_idx):
        parser.error("n-samples exceeds the forget split size")

    sample_idx = np.random.default_rng(0).choice(
        forget_idx, args.n_samples, replace=False)
    orbits = torch.cat([
        _augment(torch.as_tensor(X[i]).repeat(args.n_augs, 1, 1, 1),
                 torch.Generator().manual_seed(1000 + rank))
        for rank, i in enumerate(sample_idx)
    ]).numpy()

    vectors = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for seed in range(10):
            print(f"{condition} seed {seed}", flush=True)
            model = ResNet18C()
            model.load_state_dict(torch.load(
                args.result_dir / "models" / f"{condition}_{seed}.pt",
                map_location="cpu", weights_only=True))
            embeddings = get_embeddings(model, orbits)["penultimate"]
            vectors[condition].append(np.concatenate([
                orbit_stats(orbit) for orbit in
                embeddings.reshape(args.n_samples, args.n_augs, -1)
            ]))
    vectors["noop"] = vectors["original"]

    original, retrain = vectors["original"], vectors["retrain"]
    imprint = trr_metrics(original, retrain, retrain)
    result = {name: imprint[name] for name in ("D_RR", "D_OR", "I_topo")}
    result["p"] = permutation_pvalue(original, retrain)
    result["methods"] = {}
    for method in METHODS:
        metrics = trr_metrics(original, retrain, vectors[method])
        result["methods"][method] = {
            name: metrics[name] for name in ("TRR", "alpha", "eta")}

    (args.result_dir / "aug_orbit.json").write_text(
        json.dumps(result, indent=2, default=float))
    print(f"\nD_RR={result['D_RR']:.6f} D_OR={result['D_OR']:.6f} "
          f"I_topo={result['I_topo']:+.6f} p={result['p']:.6f}")
    print(f"{'method':9s} {'TRR':>9s} {'alpha':>9s} {'eta':>9s}")
    for method, row in result["methods"].items():
        print(f"{method:9s} {row['TRR']:+9.4f} {row['alpha']:+9.4f} "
              f"{row['eta']:9.4f}")


if __name__ == "__main__":
    main()
