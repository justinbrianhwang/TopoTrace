"""Milestone 4: CIFAR-10 + ResNet-18 (plan main experiment).

Run: conda run -n tda python scripts/run_m4.py [random|class] [n_seeds]
Outputs: results/m4_<scenario>/ (metrics.json, embeddings.npz, probe_idx.npy)
Analyze with: scripts/analyze_m2.py results/m4_<scenario>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.metrics import trr_metrics
from topotrace.mnist import (make_class_forget_split, make_probe,
                             make_random_forget_split)
from topotrace.resnet import evaluate, get_embeddings, train_resnet
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)
from topotrace.unlearn import finetune, neggrad, scrub, ssd

LAYER = "penultimate"


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "random"
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    out = ROOT / "results" / f"m4_{scenario}"
    out.mkdir(parents=True, exist_ok=True)

    X, y, X_test, y_test = load_cifar10()
    if scenario == "class":
        forget_idx, retain_idx = make_class_forget_split(y, cls=9)
    elif scenario in ("targeted", "matched"):
        splits = np.load(ROOT / "results" / "m4_splits.npz")
        forget_idx = splits[f"{scenario}_forget"]
        retain_idx = splits[f"{scenario}_retain"]
    else:
        forget_idx, retain_idx = make_random_forget_split(y, frac=0.05, seed=0)
    probe_idx = make_probe(X, forget_idx, retain_idx, seed=0)
    all_idx = np.arange(len(y))
    np.save(out / "probe_idx.npy", probe_idx)

    models = {n: [] for n in ("original", "retrain", "retrain2", "noop",
                              "finetune", "neggrad", "scrub", "ssd")}
    for s in seeds:
        print(f"seed {s}: training original/retrain/retrain2 ...", flush=True)
        orig = train_resnet(X, y, all_idx, seed=s)
        models["original"].append(orig)
        models["noop"].append(orig)
        models["retrain"].append(train_resnet(X, y, retain_idx, seed=1000 + s))
        models["retrain2"].append(train_resnet(X, y, retain_idx, seed=3000 + s))
        print(f"seed {s}: unlearning ...", flush=True)
        models["finetune"].append(finetune(orig, X, y, forget_idx, retain_idx, seed=2000 + s))
        models["neggrad"].append(neggrad(orig, X, y, forget_idx, retain_idx, seed=2000 + s))
        models["scrub"].append(scrub(orig, X, y, forget_idx, retain_idx, seed=2000 + s))
        models["ssd"].append(ssd(orig, X, y, forget_idx, retain_idx))

    acc = {}
    for name, ms in models.items():
        if name == "noop":
            continue
        acc[name] = {
            "retain": float(np.mean([evaluate(m, X, y, retain_idx) for m in ms])),
            "forget": float(np.mean([evaluate(m, X, y, forget_idx) for m in ms])),
            "test": float(np.mean([evaluate(m, X_test, y_test) for m in ms])),
        }

    import torch
    (out / "models").mkdir(exist_ok=True)
    for name, ms in models.items():
        if name == "noop":
            continue
        for i, m in enumerate(ms):
            torch.save(m.state_dict(), out / "models" / f"{name}_{i}.pt")

    print("computing persistence ...", flush=True)
    probe = X[probe_idx]
    dgms = {name: [] for name in models}
    emb_store = {}
    for name, ms in models.items():
        for i, m in enumerate(ms):
            E = get_embeddings(m, probe)
            emb_store[f"{name}_{i}_penultimate"] = E["penultimate"]
            emb_store[f"{name}_{i}_logits"] = E["logits"]
            dgms[name].append(
                persistence_diagrams(chordal_distance_matrix(E[LAYER])))
    np.savez_compressed(out / "embeddings.npz", **emb_store)

    imager = make_imager([d[1] for ds in dgms.values() for d in ds])
    vecs = {n: [vectorize(d, imager) for d in ds] for n, ds in dgms.items()}

    metrics = {}
    for name in ("noop", "retrain2", "finetune", "neggrad", "scrub", "ssd"):
        metrics[name] = trr_metrics(vecs["original"], vecs["retrain"], vecs[name])
        metrics[name]["acc"] = acc.get(name, acc.get("original"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"\n{'method':9s} {'TRR':>7s} {'alpha':>7s} {'eta':>6s} "
          f"{'ret_acc':>8s} {'fgt_acc':>8s} {'test_acc':>8s}")
    for name, m in metrics.items():
        a = m["acc"]
        print(f"{name:9s} {m['TRR']:+7.3f} {m['alpha']:+7.3f} {m['eta']:6.3f} "
              f"{a['retain']:8.4f} {a['forget']:8.4f} {a['test']:8.4f}")
    print(f"\nI_topo={metrics['noop']['I_topo']:.6f} "
          f"D_RR={metrics['noop']['D_RR']:.6f} D_OR={metrics['noop']['D_OR']:.6f}")


if __name__ == "__main__":
    main()
