"""Classify retrained and approximate-unlearned models by topology."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

CONDITIONS = ("original", "retrain", "retrain2", "noop", "finetune",
              "neggrad", "scrub", "ssd")
METHODS = ("finetune", "neggrad", "scrub", "ssd", "noop")
CELLS = (("pen_H1", "penultimate", 1),
         ("pen_H0", "penultimate", 0),
         ("logits_H1", "logits", 1))


def evaluate(features, negatives, positives, seeds):
    models = negatives + positives
    labels = np.array([0] * len(negatives) + [1] * len(positives))
    groups = np.array([seed for _, seed in models])
    X = np.array([features[model] for model in models])

    def predict(y):
        scores = np.empty(len(models))
        for seed in seeds:
            test = groups == seed
            classifier = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000))
            classifier.fit(X[~test], y[~test])
            scores[test] = classifier.decision_function(X[test])
        return scores

    scores = predict(labels)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        y = labels.copy()
        for seed in seeds:
            y[groups == seed] = rng.permutation(y[groups == seed])
        null.append(roc_auc_score(y, predict(y)))
    observed = roc_auc_score(labels, scores)
    p95 = float(np.percentile(null, 95))
    return {"auc": observed,
            "balanced_accuracy": balanced_accuracy_score(labels, scores >= 0),
            "null_auc_mean": float(np.mean(null)),
            "null_auc_95th_percentile": p95,
            "observed_exceeds_null_95th": bool(observed > p95)}


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "m4_random"
    with np.load(out / "embeddings.npz") as embeddings:
        seeds = sorted(int(key.split("_")[1]) for key in embeddings.files
                       if key.startswith("original_") and key.endswith("_logits"))
        features = {}
        for name, layer, dim in CELLS:
            diagrams = {(condition, seed): persistence_diagrams(
                chordal_distance_matrix(embeddings[f"{condition}_{seed}_{layer}"]))
                for condition in CONDITIONS for seed in seeds}
            imager = make_imager([diagrams[(condition, seed)][dim]
                                  for condition in ("original", "retrain")
                                  for seed in seeds])
            features[name] = {model: vectorize(diagram, imager, dim)
                              for model, diagram in diagrams.items()}

    features["concat"] = {model: np.concatenate(
        [features[name][model] for name, _, _ in CELLS])
        for model in features["pen_H1"]}
    negatives = [(condition, seed) for condition in ("retrain", "retrain2")
                 for seed in seeds]
    tasks = {method: (negatives, [(method, seed) for seed in seeds])
             for method in METHODS}
    tasks["any_approximate"] = (negatives, [(method, seed)
                                             for method in METHODS[:-1]
                                             for seed in seeds])
    tasks["anchor_control"] = ([("retrain", seed) for seed in seeds],
                               [("retrain2", seed) for seed in seeds])
    results = {task: {name: evaluate(feature, negatives, positives, seeds)
                      for name, feature in features.items()}
               for task, (negatives, positives) in tasks.items()}

    names = list(features)
    print(f"{'task':16s}" + "".join(f"{name:>28s}" for name in names))
    for task, row in results.items():
        print(f"{task:16s}" + "".join(
            (f"{cell['auc']:.3f} ({cell['balanced_accuracy']:.3f}) "
             f"{cell['null_auc_mean']:.3f}/{cell['null_auc_95th_percentile']:.3f}"
             f"{'*' if cell['observed_exceeds_null_95th'] else ''}").rjust(28)
            for cell in row.values()))
    (out / "distinguisher.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
