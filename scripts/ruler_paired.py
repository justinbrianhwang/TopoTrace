"""Paired-seed RULER comparison for the CIFAR-10 random-deletion sweep."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.mnist import make_random_forget_split
from topotrace.resnet import ResNet18C, get_embeddings, train_resnet
from topotrace.stats import permutation_pvalue

METHODS = ("retrain2", "finetune", "neggrad", "scrub", "ssd")


def load_model(path):
    model = ResNet18C()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def score(method, oracle):
    cosine = np.sum(method * oracle, axis=1) / (
        np.linalg.norm(method, axis=1) * np.linalg.norm(oracle, axis=1) + 1e-12)
    return float(cosine[:300].mean() - np.median(cosine[300:600]))


def bh(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(ranked)
    result[order] = np.minimum(ranked, 1)
    return result


def analyze(embeddings, paired, offset):
    values = {method: [score(embeddings[f"{method}_{seed}_penultimate"],
                             paired[f"retrain_paired_{(seed + offset) % 10}_penultimate"])
                       for seed in range(10)] for method in METHODS}
    anchor = [np.array([x]) for x in values["retrain2"]]
    pvalues = [permutation_pvalue([np.array([x]) for x in values[method]], anchor,
                                  n_perm=10_000, seed=0)
               for method in METHODS]
    qvalues = bh(pvalues)
    return {method: {"median_score": float(np.median(values[method])),
                     "permutation_p": float(p), "q": float(q),
                     "reject": bool(q <= .05), "seed_scores": values[method]}
            for method, p, q in zip(METHODS, pvalues, qvalues)}


def main():
    out = ROOT / "results" / "m4_random"
    models = out / "models"
    X, y, _, _ = load_cifar10()
    _, retain_idx = make_random_forget_split(y, frac=.05, seed=0)
    probe = X[np.load(out / "probe_idx.npy")]

    paired = {}
    for seed in range(10):
        path = models / f"retrain_paired_{seed}.pt"
        if path.exists():
            model = load_model(path)
            print(f"seed {seed}: loaded", flush=True)
        else:
            print(f"seed {seed}: training paired oracle", flush=True)
            model = train_resnet(X, y, retain_idx, seed=seed)
            torch.save(model.state_dict(), path)
        for layer, values in get_embeddings(model, probe).items():
            paired[f"retrain_paired_{seed}_{layer}"] = values
    np.savez_compressed(out / "embeddings_paired.npz", **paired)

    with np.load(out / "embeddings.npz") as embeddings:
        results = {"paired": analyze(embeddings, paired, 0),
                   "mismatched": analyze(embeddings, paired, 1)}
    (out / "ruler_paired.json").write_text(json.dumps(results, indent=2))

    for pairing, rows in results.items():
        print(f"\n{pairing}")
        print(f"{'method':10s} | {'median score':>12s} | {'p':>8s} | {'q':>8s} | reject")
        for method, row in rows.items():
            print(f"{method:10s} | {row['median_score']:+12.6f} | "
                  f"{row['permutation_p']:8.5f} | {row['q']:8.5f} | {row['reject']}")


if __name__ == "__main__":
    main()
