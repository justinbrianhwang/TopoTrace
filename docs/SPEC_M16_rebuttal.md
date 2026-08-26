# M16 spec — rebuttal computations (adversarial-review response)

All offline from cached artifacts under results/m4_random (embeddings.npz,
keys {condition}_{seed}_{layer}, conditions original/retrain/retrain2/noop/
finetune/neggrad/scrub/ssd, seeds 0..9, layers penultimate/logits; rows
0..299 forget probe, 300..599 retain neighbors).

New file: `scripts/rebuttal_checks.py` (write ONLY this). CPU only.
CLI: `conda run -n tda python scripts/rebuttal_checks.py results/m4_random`
Save everything to <dir>/rebuttal_checks.json and print compact tables.

1. RULER-style per-record cosine baseline (unpaired variant): for each
   (method model s, oracle retrain t) pair compute mean per-record cosine
   similarity between their penultimate embeddings (row-wise cosine,
   mean over 600 rows). Method fingerprint per seed = vector of its 10
   cosines to the oracles. Report per method (incl. retrain2 anchor):
   median pairwise-mean cosine, and a two-sample permutation p using the
   mean-difference statistic between the method's (s,t) cosine values and
   the retrain2 anchor's (baseline for what exact retrains look like).
   This documents how a per-record cosine audit behaves WITHOUT paired
   seeds on our data.
2. Distinguisher permutation null: reuse the leave-one-seed-out logistic
   regression protocol of scripts/distinguish.py (feature set pen_H1 and
   concat) for task scrub-vs-retrains, but ALSO run it 200 times with
   class labels randomly permuted (seed-consistent permutation: permute
   which models are labeled class 1). Report the observed AUC and the
   null AUC distribution (mean, 95th percentile) — does the observed AUC
   exceed the permutation null?
3. Oracle-split gate robustness: split the 10 oracle retrains into two
   disjoint halves (seeds 0-4 / 5-9, i.e., first five and last five).
   Recompute the gate (I_topo + permutation p, original vs oracle-half A)
   using ONLY half A; then run the method tests (finetune/neggrad/scrub/
   ssd/retrain2 vs oracle-half B) using ONLY half B, with the
   persistence-image grid fitted on original + half A. Penultimate H0 and
   H1. Report gate p, and per-method p — does the all-methods-rejected /
   anchor-retained pattern survive when gate and method tests share no
   oracle seeds?
Reuse topotrace.topology/stats/metrics helpers. Keep code minimal;
runtime should be minutes (persistence diagrams can be computed once and
reused across parts 2-3).
