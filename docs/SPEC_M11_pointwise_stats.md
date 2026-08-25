# M11 spec — pointwise representation metrics + formal statistics
(plan §12.3, §14.4-14.5, §10.3)

All CPU, from cached results. Files (write ONLY these):

```
scripts/pointwise.py     # §12.3 pointwise representation comparison
scripts/formal_stats.py  # §14 equivalence/BH table + correlations + §10.3
```

## pointwise.py

CLI: `... scripts/pointwise.py results/m4_random`
From embeddings.npz (keys {condition}_{seed}_{layer}, rows = fixed probe):
for each method m in {noop, retrain2, finetune, neggrad, scrub, ssd},
penultimate layer, compare method model s vs retrain model t pairwise
(all s,t):
- linear CKA between the two embedding matrices
- mean rowwise cosine similarity
- k-NN overlap: fraction of shared neighbors in each model's own 10-NN
  graph (mean over rows)
Report median over (s,t) pairs per metric per method; baseline row =
retrain-vs-retrain pairs (t != t'). Also per-method permutation p vs that
baseline (topotrace.stats.permutation_pvalue needs vectors — instead do a
simple two-sample permutation on the pair-value distributions with mean
difference statistic; implement locally).
Save <dir>/pointwise.json; print table. The paper question: do pointwise
metrics separate methods from retrains where TopoTrace does (or fail to)?

## formal_stats.py

CLI: `... scripts/formal_stats.py results/m4_random` (works for any dir
with embeddings.npz + analysis.json)
1. Equivalence analysis (plan §14.5): per (layer, hom) cell, vectors as in
   analyze_m2 (rebuild them the same way); delta = 95th percentile of
   pairwise retrain-retrain distances; per method: D_UR (median), bootstrap
   CI (resample seeds); decision: "oracle-equivalent" if CI upper < delta,
   "outside" if CI lower > delta, else "inconclusive".
2. BH-FDR correction (plan §14.4): collect all per-method permutation
   p-values across the 4 cells (from analysis.json if present, else
   recompute) and report BH-adjusted q-values at alpha=.05.
3. Correlations (H4): per-seed TRR_s for each method (per-seed distance to
   oracle normalized as in ablate.py oracle section) vs per-seed MIA AUC
   (mia.json) and vs relearning auc (relearn.json, matching conditions);
   Spearman rho + p (scipy). Pool methods excluding destroyed models
   (retain acc < .5 — read metrics.json accs; note exclusions in output).
4. §10.3 audit-subset bootstrap: for the baseline cell (penultimate H1),
   recompute I_topo on 20 bootstrap row-resamples of the probe (same rows
   for all models per resample); report mean±std — probe-sampling
   stability.
Save <dir>/formal_stats.json; print all tables compactly.

Verify both on results/m4_random (real run). scipy is available in env tda.
Keep code minimal.
