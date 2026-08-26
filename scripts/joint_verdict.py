"""Apply the prespecified joint utility/forget/topology verdict."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.mnist import make_random_forget_split
from topotrace.resnet import ResNet18C, evaluate
from topotrace.unlearn import noise_destroy

BASE = ("retrain", "retrain2", "original", "finetune", "neggrad", "scrub", "ssd")
SIGMAS = (.02, .05, .1, .2, .5, 1.0)


def load_model(path):
    model = ResNet18C()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def accuracy(model, X, y, X_test, y_test, forget_idx, retain_idx):
    return {"retain": evaluate(model, X, y, retain_idx),
            "forget": evaluate(model, X, y, forget_idx),
            "test": evaluate(model, X_test, y_test)}


def bh(rows):
    p = np.array([row[2] for row in rows])
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    return {(rows[i][0], rows[i][1]): float(min(q[j], 1))
            for j, i in enumerate(order)}


def main():
    out = ROOT / "results" / "m4_random"
    models = out / "models"
    X, y, X_test, y_test = load_cifar10()
    forget_idx, retain_idx = make_random_forget_split(y, frac=.05, seed=0)

    per_seed = {name: [] for name in BASE}
    for seed in range(10):
        for name in BASE:
            per_seed[name].append(accuracy(load_model(models / f"{name}_{seed}.pt"),
                                           X, y, X_test, y_test,
                                           forget_idx, retain_idx))
        original = load_model(models / f"original_{seed}.pt")
        for sigma in SIGMAS:
            name = f"noise_{sigma:g}"
            per_seed.setdefault(name, []).append(accuracy(
                noise_destroy(original, sigma, seed), X, y, X_test, y_test,
                forget_idx, retain_idx))
        print(f"seed {seed} evaluated", flush=True)

    faithful = json.loads((out / "faithful_baselines.json").read_text())
    per_seed.update(faithful["accuracy"])
    ranges = {key: [min(row[key] for row in per_seed["retrain"]),
                    max(row[key] for row in per_seed["retrain"])]
              for key in ("retain", "forget", "test")}

    analysis = json.loads((out / "analysis.json").read_text())
    rows = [(cell, method, row["p"])
            for cell, values in analysis.items() if values["gate_open"]
            for method, row in values["methods"].items()]
    base_q = bh(rows)
    topo = {"retrain": [1.0]}
    for name in BASE[1:]:
        method = "noop" if name == "original" else name
        topo[name] = [base_q[(cell, method)] for cell, values in analysis.items()
                      if values["gate_open"]]
    for name in faithful["accuracy"]:
        topo[name] = [values["methods"][name]["q"]
                      for values in faithful["audit"].values()]

    results = {}
    for name, values in per_seed.items():
        mean = {key: float(np.mean([row[key] for row in values]))
                for key in ranges}
        utility = all(ranges[key][0] <= mean[key] <= ranges[key][1]
                      for key in ("retain", "test"))
        forget = ranges["forget"][0] <= mean["forget"] <= ranges["forget"][1]
        qs = topo.get(name, [])
        topology = all(q >= .05 for q in qs)
        failures = [label for label, passed in
                    (("UTILITY", utility), ("FORGET", forget),
                     ("TOPOLOGY", topology)) if not passed]
        modes = []
        if not utility:
            below = (mean["retain"] < ranges["retain"][0]
                     or mean["test"] < ranges["test"][0])
            modes.append("utility-loss" if below else "retain-overfit")
        if not forget:
            modes.append("under-forgetting" if mean["forget"] > ranges["forget"][1]
                         else "over-forgetting")
        if not topology:
            modes.append("residual")
        mode = "+".join(modes) if modes else None
        results[name] = {"accuracy": values, "mean": mean,
                         "topology_q": min(qs) if qs else None,
                         "utility_consistent": utility,
                         "forget_consistent": forget,
                         "topology_consistent": topology,
                         "verdict": "audit-consistent" if not failures else "flagged",
                         "failures": failures, "mode": mode}

    payload = {"oracle_ranges": ranges, "conditions": results}
    (out / "joint_verdict.json").write_text(json.dumps(payload, indent=2))
    print(f"\n{'condition':14s} | retain | forget | test   | topo q | verdict          | mode")
    for name, row in results.items():
        mean, q = row["mean"], row["topology_q"]
        q_text = f"{q:.4f}" if q is not None else " n/a "
        print(f"{name:14s} | {mean['retain']:.4f} | {mean['forget']:.4f} | "
              f"{mean['test']:.4f} | {q_text}  | {row['verdict']:16s} | "
              f"{row['mode'] or '-'}")


if __name__ == "__main__":
    main()
