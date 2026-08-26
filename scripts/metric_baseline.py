"""Reviewer-response computations on m4_random (penultimate):
A) imager fit ONLY on original+retrain diagrams (frozen) -> recompute stats
B) non-PH baseline: distance-quantile fingerprint under identical protocol
"""
import sys

sys.path.insert(0, r"f:\coding\TDA\TOPOTrace\src")
import numpy as np
from topotrace.metrics import trr_metrics
from topotrace.stats import permutation_pvalue
from topotrace.topology import (chordal_distance_matrix, make_imager,
                                persistence_diagrams, vectorize)

npz = np.load(r"f:\coding\TDA\TOPOTrace\results\m4_random\embeddings.npz")
CONDS = ("original", "retrain", "retrain2", "noop", "finetune", "neggrad",
         "scrub", "ssd")
SEEDS = range(10)

D = {c: [chordal_distance_matrix(npz[f"{c}_{s}_penultimate"]) for s in SEEDS]
     for c in CONDS}
dg = {c: [persistence_diagrams(d) for d in ds] for c, ds in D.items()}

for dim in (0, 1):
    # A) frozen imager: fit on original+retrain only
    imager = make_imager([d[dim] for c in ("original", "retrain")
                          for d in dg[c]])
    v = {c: [vectorize(d, imager, dim=dim) for d in ds]
         for c, ds in dg.items()}
    m = trr_metrics(v["original"], v["retrain"], v["retrain"])
    p = permutation_pvalue(v["original"], v["retrain"])
    print(f"[frozen-imager H{dim}] I_topo={m['I_topo']:+.6f} p={p:.4f}")
    for c in ("retrain2", "finetune", "neggrad", "scrub", "ssd"):
        t = trr_metrics(v["original"], v["retrain"], v[c])
        pm = permutation_pvalue(v[c], v["retrain"])
        print(f"  {c:9s} TRR={t['TRR']:+7.3f} p={pm:.4f}")

# B) non-PH baseline: 50 quantiles of the pairwise-distance distribution
qs = np.linspace(0.02, 0.98, 50)
vq = {}
for c in CONDS:
    vq[c] = []
    for d in D[c]:
        tri = d[np.triu_indices(len(d), 1)]
        vq[c].append(np.quantile(tri, qs))
m = trr_metrics(vq["original"], vq["retrain"], vq["retrain"])
p = permutation_pvalue(vq["original"], vq["retrain"])
print(f"\n[distance-quantile baseline] I_topo={m['I_topo']:+.6f} p={p:.4f}")
for c in ("retrain2", "finetune", "neggrad", "scrub", "ssd"):
    t = trr_metrics(vq["original"], vq["retrain"], vq[c])
    pm = permutation_pvalue(vq[c], vq["retrain"])
    print(f"  {c:9s} TRR={t['TRR']:+7.3f} p={pm:.4f}")
