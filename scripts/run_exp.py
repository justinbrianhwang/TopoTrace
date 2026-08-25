"""Run the generic dataset experiment suite."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

if "--smoke" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace import cnn, resnet
from topotrace.cifar import (load_cifar10, load_cifar100,
                             make_subclass_forget_split)
from topotrace.metrics import trr_metrics
from topotrace.mnist import (load_fashion_mnist, make_class_forget_split,
                             make_probe, make_random_forget_split)
from topotrace.svhn import load_svhn
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)
from topotrace.unlearn import finetune, neggrad, scrub, ssd

LAYER = "penultimate"
REGISTRY = {
    "cifar10": (load_cifar10, resnet.train_resnet, resnet.get_embeddings,
                resnet.evaluate, 10, 9, "resnet"),
    "cifar100": (load_cifar100, resnet.train_resnet, resnet.get_embeddings,
                 resnet.evaluate, 100, 30, "resnet"),
    "svhn": (load_svhn, resnet.train_resnet, resnet.get_embeddings,
             resnet.evaluate, 10, 9, "resnet"),
    "fashionmnist": (load_fashion_mnist, cnn.train_cnn, cnn.get_embeddings,
                     cnn.evaluate, 10, 9, "cnn"),
}


def run(dataset, scenario, n_seeds, smoke, out):
    loader, train, get_embeddings, evaluate, num_classes, target, kind = REGISTRY[dataset]
    X, y, X_test, y_test = loader()
    if smoke:
        X, y = X[:3000], y[:3000]
    if scenario == "class":
        split = make_subclass_forget_split if dataset == "cifar100" else make_class_forget_split
        forget_idx, retain_idx = split(y, target)
    else:
        forget_idx, retain_idx = make_random_forget_split(y, frac=.05, seed=0)
    probe_size = 100 if smoke else 300
    probe_idx = make_probe(X, forget_idx, retain_idx, probe_size, probe_size, seed=0)
    all_idx = np.arange(len(y))
    np.save(out / "probe_idx.npy", probe_idx)

    train_kwargs = {"epochs": 1, "device": "cpu"} if smoke else {}
    if kind == "resnet":
        train_kwargs["num_classes"] = num_classes
    seeds = range(2 if smoke else n_seeds)
    models = {name: [] for name in (
        "original", "retrain", "retrain2", "noop", "finetune", "neggrad",
        "scrub", "ssd")}
    for seed in seeds:
        print(f"seed {seed}: training original/retrain/retrain2 ...", flush=True)
        original = train(X, y, all_idx, seed=seed, **train_kwargs)
        models["original"].append(original)
        models["noop"].append(original)
        models["retrain"].append(train(X, y, retain_idx, seed=1000 + seed, **train_kwargs))
        models["retrain2"].append(train(X, y, retain_idx, seed=3000 + seed, **train_kwargs))
        print(f"seed {seed}: unlearning ...", flush=True)
        epoch_kwargs = {"epochs": 1} if smoke else {}
        models["finetune"].append(finetune(
            original, X, y, forget_idx, retain_idx, seed=2000 + seed,
            **epoch_kwargs))
        models["neggrad"].append(neggrad(
            original, X, y, forget_idx, retain_idx, seed=2000 + seed))
        models["scrub"].append(scrub(
            original, X, y, forget_idx, retain_idx, seed=2000 + seed,
            **epoch_kwargs))
        models["ssd"].append(ssd(original, X, y, forget_idx, retain_idx))

    acc = {}
    for name, trained in models.items():
        if name == "noop":
            continue
        acc[name] = {
            "retain": float(np.mean([evaluate(m, X, y, retain_idx) for m in trained])),
            "forget": float(np.mean([evaluate(m, X, y, forget_idx) for m in trained])),
            "test": float(np.mean([evaluate(m, X_test, y_test) for m in trained])),
        }

    import torch
    (out / "models").mkdir(exist_ok=True)
    for name, trained in models.items():
        if name == "noop":
            continue
        for i, model in enumerate(trained):
            torch.save(model.state_dict(), out / "models" / f"{name}_{i}.pt")

    print("computing persistence ...", flush=True)
    probe = X[probe_idx]
    diagrams = {name: [] for name in models}
    embeddings = {}
    for name, trained in models.items():
        for i, model in enumerate(trained):
            values = get_embeddings(model, probe)
            embeddings[f"{name}_{i}_penultimate"] = values["penultimate"]
            embeddings[f"{name}_{i}_logits"] = values["logits"]
            diagrams[name].append(persistence_diagrams(
                chordal_distance_matrix(values[LAYER])))
    np.savez_compressed(out / "embeddings.npz", **embeddings)

    imager = make_imager([d[1] for values in diagrams.values() for d in values])
    vectors = {name: [vectorize(d, imager) for d in values]
               for name, values in diagrams.items()}
    metrics = {}
    for name in ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd"):
        metrics[name] = trr_metrics(
            vectors["original"], vectors["retrain"], vectors[name])
        metrics[name]["acc"] = acc.get(name, acc.get("original"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\n{'method':9s} {'TRR':>7s} {'alpha':>7s} {'eta':>6s} "
          f"{'ret_acc':>8s} {'fgt_acc':>8s} {'test_acc':>8s}")
    for name, values in metrics.items():
        accuracy = values["acc"]
        print(f"{name:9s} {values['TRR']:+7.3f} {values['alpha']:+7.3f} "
              f"{values['eta']:6.3f} {accuracy['retain']:8.4f} "
              f"{accuracy['forget']:8.4f} {accuracy['test']:8.4f}")
    print(f"\nI_topo={metrics['noop']['I_topo']:.6f} "
          f"D_RR={metrics['noop']['D_RR']:.6f} "
          f"D_OR={metrics['noop']['D_OR']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=REGISTRY)
    parser.add_argument("scenario", choices=("random", "class"))
    parser.add_argument("n_seeds", nargs="?", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.n_seeds < 1:
        parser.error("n_seeds must be positive")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    if args.smoke:
        out = Path(tempfile.mkdtemp(
            prefix=f"_smoke_{args.dataset}_{args.scenario}_", dir=results))
    else:
        out = results / f"exp_{args.dataset}_{args.scenario}"
        out.mkdir(parents=True, exist_ok=True)
    try:
        run(args.dataset, args.scenario, args.n_seeds, args.smoke, out)
    finally:
        if args.smoke:
            shutil.rmtree(out)


if __name__ == "__main__":
    main()
