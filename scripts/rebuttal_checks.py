"""Offline rebuttal checks for the M4 random-seed sweep."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.metrics import trr_metrics
from topotrace.stats import permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
RULER_METHODS = ("retrain2", "noop", "finetune", "neggrad", "scrub", "ssd")
TEST_METHODS = ("finetune", "neggrad", "scrub", "ssd", "retrain2")


def mean_difference_pvalue(a, b, n_perm=10_000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate((a, b))
    rng = np.random.default_rng(seed)
    count = sum(abs(x[:len(a)].mean() - x[len(a):].mean()) >= observed
                for x in (pool[rng.permutation(len(pool))] for _ in range(n_perm)))
    return (count + 1) / (n_perm + 1)


def ruler(embeddings, seeds):
    oracle = {s: embeddings[f"retrain_{s}_penultimate"] for s in seeds}
    fingerprints = {}
    for method in RULER_METHODS:
        fingerprints[method] = [[float(np.mean(np.sum(x * oracle[t], axis=1) /
            (np.linalg.norm(x, axis=1) * np.linalg.norm(oracle[t], axis=1) + 1e-12)))
            for t in seeds] for s in seeds
            for x in [embeddings[f"{method}_{s}_penultimate"]]]
    seed_values = {method: np.mean(values, axis=1)
                   for method, values in fingerprints.items()}
    anchor = seed_values["retrain2"]
    return {method: {
        "median_pairwise_mean_cosine": float(np.median(seed_values[method])),
        "permutation_p": mean_difference_pvalue(seed_values[method], anchor),
        "seed_values": seed_values[method].tolist(),
        "fingerprints": values,
    } for method, values in fingerprints.items()}


def auc(features, models, labels, seeds):
    X = np.array([features[model] for model in models])
    groups = np.array([seed for _, seed in models])
    scores = np.empty(len(models))
    for seed in seeds:
        test = groups == seed
        classifier = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000))
        classifier.fit(X[~test], labels[~test])
        scores[test] = classifier.decision_function(X[test])
    return float(roc_auc_score(labels, scores))


def distinguisher(features, seeds):
    models = [(condition, seed) for condition in ("retrain", "retrain2", "scrub")
              for seed in seeds]
    observed_labels = np.array([condition == "scrub" for condition, _ in models])
    rng = np.random.default_rng(0)
    choices = [rng.integers(3, size=len(seeds)) for _ in range(200)]
    null_labels = [np.array([i == choice[seed] for i, condition in
                            enumerate(("retrain", "retrain2", "scrub"))
                            for seed in seeds]) for choice in choices]
    results = {}
    for name in ("pen_H1", "concat"):
        observed = auc(features[name], models, observed_labels, seeds)
        null = [auc(features[name], models, labels, seeds) for labels in null_labels]
        p95 = float(np.percentile(null, 95))
        results[name] = {"observed_auc": observed,
                         "null_auc_mean": float(np.mean(null)),
                         "null_auc_95th_percentile": p95,
                         "observed_exceeds_null_95th": bool(observed > p95),
                         "null_aucs": null}
    return results


def oracle_split(diagrams, seeds):
    half_a, half_b = seeds[:5], seeds[5:]
    results = {}
    for layer, short in (("penultimate", "pen"), ("logits", "logits")):
        for dim in (0, 1):
            imager = make_imager([diagrams[(condition, seed, layer)][dim]
                                  for condition, subset in (("original", seeds),
                                                            ("retrain", half_a))
                                  for seed in subset])
            vectors = lambda condition, subset: [vectorize(
                diagrams[(condition, seed, layer)], imager, dim) for seed in subset]
            original = vectors("original", seeds)
            retrain_a = vectors("retrain", half_a)
            retrain_b = vectors("retrain", half_b)
            gate = trr_metrics(original, retrain_a, retrain_a)
            methods = {}
            for method in TEST_METHODS:
                values = vectors(method, seeds)
                metrics = trr_metrics(original, retrain_b, values)
                methods[method] = {"TRR": metrics["TRR"],
                                   "permutation_p": permutation_pvalue(values, retrain_b)}
            results[f"{short}_H{dim}"] = {
                "gate_I_topo": gate["I_topo"],
                "gate_permutation_p": permutation_pvalue(original, retrain_a),
                "methods": methods,
                "pattern_survives": (all(methods[m]["permutation_p"] < .05
                                         for m in TEST_METHODS[:-1]) and
                                     methods["retrain2"]["permutation_p"] >= .05),
            }
    return results


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "m4_random"
    with np.load(out / "embeddings.npz") as embeddings:
        seeds = sorted(int(key.split("_")[1]) for key in embeddings.files
                       if key.startswith("original_") and key.endswith("_logits"))
        ruler_results = ruler(embeddings, seeds)
        diagrams = {(condition, seed, layer): persistence_diagrams(
            chordal_distance_matrix(embeddings[f"{condition}_{seed}_{layer}"]))
            for condition in CONDITIONS for seed in seeds
            for layer in ("penultimate", "logits")}

    cells = (("pen_H1", "penultimate", 1), ("pen_H0", "penultimate", 0),
             ("logits_H1", "logits", 1))
    features = {}
    for name, layer, dim in cells:
        imager = make_imager([diagrams[(condition, seed, layer)][dim]
                              for condition in ("original", "retrain")
                              for seed in seeds])
        features[name] = {(condition, seed): vectorize(
            diagrams[(condition, seed, layer)], imager, dim)
            for condition in CONDITIONS for seed in seeds}
    features["concat"] = {model: np.concatenate([features[name][model]
                                                  for name, _, _ in cells])
                          for model in features["pen_H1"]}

    distinguish_results = distinguisher(features, seeds)
    split_results = oracle_split(diagrams, seeds)
    results = {"ruler": ruler_results, "distinguisher": distinguish_results,
               "oracle_split": split_results}

    print("RULER cosine")
    print(f"{'method':10s} {'median':>10s} {'p':>10s}")
    for method, row in ruler_results.items():
        print(f"{method:10s} {row['median_pairwise_mean_cosine']:10.6f} "
              f"{row['permutation_p']:10.6f}")
    print("\nDistinguisher permutation null")
    print(f"{'feature':10s} {'observed':>10s} {'null mean':>10s} {'null p95':>10s} {'exceeds':>8s}")
    for name, row in distinguish_results.items():
        print(f"{name:10s} {row['observed_auc']:10.3f} {row['null_auc_mean']:10.3f} "
              f"{row['null_auc_95th_percentile']:10.3f} "
              f"{str(row['observed_exceeds_null_95th']):>8s}")
    print("\nOracle-split gate")
    for name, row in split_results.items():
        print(f"{name}: I_topo={row['gate_I_topo']:+.6f} "
              f"gate_p={row['gate_permutation_p']:.6f}")
        print("  " + " ".join(f"{method}={cell['permutation_p']:.6f}"
                                for method, cell in row["methods"].items()) +
              f" survives={row['pattern_survives']}")
    (out / "rebuttal_checks.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
