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
ACCURACIES = ("retain", "forget", "test")


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


def fractions(hits, inside=True):
    return {**{key: float(np.mean([row[key] == inside for row in hits]))
               for key in ACCURACIES},
            "conjunction": float(np.mean([all(row.values()) == inside
                                           for row in hits]))}


def main():
    out = ROOT / "results" / "m4_random"
    models = out / "models"
    extra = [json.loads((out / name).read_text())
             for name in ("faithful_baselines.json", "salun_rmu.json")]
    if "--cache" in sys.argv:
        cached = json.loads((out / "joint_verdict.json").read_text())
        per_seed = {name: row["accuracy"]
                    for name, row in cached["conditions"].items()}
        print("loaded cached per-seed accuracies", flush=True)
    else:
        X, y, X_test, y_test = load_cifar10()
        forget_idx, retain_idx = make_random_forget_split(y, frac=.05, seed=0)
        per_seed = {name: [] for name in BASE}
        for seed in range(10):
            for name in BASE:
                per_seed[name].append(accuracy(
                    load_model(models / f"{name}_{seed}.pt"), X, y, X_test,
                    y_test, forget_idx, retain_idx))
            original = load_model(models / f"original_{seed}.pt")
            for sigma in SIGMAS:
                name = f"noise_{sigma:g}"
                per_seed.setdefault(name, []).append(accuracy(
                    noise_destroy(original, sigma, seed), X, y, X_test, y_test,
                    forget_idx, retain_idx))
            print(f"seed {seed} evaluated", flush=True)
    for payload in extra:
        per_seed.update(payload["accuracy"])
    oracle = per_seed["retrain"]
    ranges = {key: [min(row[key] for row in oracle),
                    max(row[key] for row in oracle)]
              for key in ACCURACIES}
    loo_hits = []
    for i, row in enumerate(oracle):
        others = oracle[:i] + oracle[i + 1:]
        loo_hits.append({key: min(x[key] for x in others) <= row[key]
                         <= max(x[key] for x in others) for key in ACCURACIES})
    rng, split_hits = np.random.default_rng(0), []
    for _ in range(200):
        half_a = rng.choice(10, 5, replace=False)
        half_b = set(range(10)) - set(half_a)
        split_ranges = {key: [min(oracle[i][key] for i in half_a),
                              max(oracle[i][key] for i in half_a)]
                        for key in ACCURACIES}
        split_hits.extend({key: split_ranges[key][0] <= oracle[i][key]
                           <= split_ranges[key][1] for key in ACCURACIES}
                          for i in half_b)
    anchor_hits = [{key: ranges[key][0] <= row[key] <= ranges[key][1]
                    for key in ACCURACIES} for row in per_seed["retrain2"]]
    range_characteristics = {
        "leave_one_out": {"fraction_inside": fractions(loo_hits)},
        "five_five_splits": {
            "n_splits": 200,
            "fraction_wrongly_flagged": fractions(split_hits, False)},
        "held_out_anchor": {
            "fraction_inside_oracle_range": fractions(anchor_hits)}}

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
    for payload in extra:
        for name in payload["accuracy"]:
            topo[name] = [values["methods"][name]["q"]
                          for values in payload["audit"].values()]

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

    payload = {"oracle_ranges": ranges, "conditions": results,
               "range_operating_characteristics": range_characteristics}
    (out / "joint_verdict.json").write_text(json.dumps(payload, indent=2))
    print(f"\n{'condition':14s} | retain | forget | test   | topo q | verdict          | mode")
    for name, row in results.items():
        mean, q = row["mean"], row["topology_q"]
        q_text = f"{q:.4f}" if q is not None else " n/a "
        print(f"{name:14s} | {mean['retain']:.4f} | {mean['forget']:.4f} | "
              f"{mean['test']:.4f} | {q_text}  | {row['verdict']:16s} | "
              f"{row['mode'] or '-'}")
    print("\nrange operating characteristics")
    print(json.dumps(range_characteristics, indent=2))


if __name__ == "__main__":
    main()
