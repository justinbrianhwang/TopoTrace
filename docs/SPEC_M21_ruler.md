# M21 — genuine RULER head-to-head (paired-seed protocol reproduced)

Closes the "no real RULER comparison" barrier. RULER pairs each audited
model with a retrained oracle trained from the SAME initialization seed;
our main oracles use disjoint seeds (1000+s), so a faithful comparison
needs paired-seed retrains. New file ONLY: `scripts/ruler_paired.py`.
GPU allowed (~1 h).

CIFAR-10, random 5% deletion, results/m4_random.

1. Train 10 paired-seed oracles: `train_resnet(X, y, retain_idx, seed=s)`
   for s in 0..9 — the SAME seed as original_s, so initialization (and
   the shuffling generator) match the original by construction. Save
   state dicts as results/m4_random/models/retrain_paired_{s}.pt and
   probe embeddings into results/m4_random/embeddings_paired.npz
   (keys retrain_paired_{s}_{penultimate,logits}).
2. RULER-style $M_2$, paired: for each method in (retrain2, finetune,
   neggrad, scrub, ssd) and each seed s, compute the mean per-record
   cosine similarity between the method model's penultimate embeddings
   and its PAIRED oracle retrain_paired_{s}, over the FORGET rows of the
   probe (rows 0..299). Calibrate exactly as RULER does: subtract the
   median of the same per-record cosine computed over the RETAIN-neighbor
   rows (300..599). The result is one calibrated score per seed per
   method.
3. Inference at the seed level: for each method, a two-sided permutation
   test (10^4) of its 10 calibrated scores against the paired-seed
   ANCHOR distribution — i.e. the same statistic computed between
   retrain_paired_{s} and a second exact-retrain cohort (use the existing
   retrain2 models, seeds 3000+s, as the "method" whose true status is
   exact retraining). Report per method: median calibrated score,
   permutation p, BH q over the family of methods.
4. Stability probe (RULER's own caveat): repeat step 2 pairing each
   method with a MISMATCHED oracle (retrain_paired_{(s+1) mod 10}) and
   report how the per-method verdicts change. This quantifies how much
   the paired protocol depends on the pairing itself.
5. Save results/m4_random/ruler_paired.json; print a compact table:
   method | median score | p | q | reject, for both paired and mismatched
   pairings.

Reuse topotrace.{resnet,mnist,stats}. Verify by running end-to-end.
Keep code minimal.
