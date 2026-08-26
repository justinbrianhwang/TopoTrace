# M17A — frozen protocol end-to-end + valid seed-level inference

Fixes from the final adversarial review (findings 2, 5, 11, 15). Edit ONLY:
scripts/distinguish.py, scripts/destructive.py, scripts/rebuttal_checks.py,
scripts/pointwise.py. CPU only, all from results/m4_random cached artifacts.

1. distinguish.py:
   - Fit each feature-set persistence-image grid on ORIGINAL + RETRAIN
     diagrams only (frozen) before vectorizing any other model.
   - Add task "anchor_control": positives = retrain2 models (10),
     negatives = retrain (oracle) models (10) — does the pipeline
     spuriously separate two exact-retrain cohorts?
   - For EVERY task x feature set add a label-permutation null (200
     permutations of which models are labeled positive): report observed
     AUC, null mean, null 95th percentile, exceeds-null flag.
   - Save the extended results to distinguisher.json (same path), rerun.
2. destructive.py: fit the grid on ORIGINAL + RETRAIN diagrams only
   (exclude methods AND all noise levels), rerun, overwrite
   destructive.json.
3. rebuttal_checks.py:
   - Cosine part: replace pair-level permutation with SEED-LEVEL inference
     (the paper's independent unit): per method seed s, statistic = mean
     over the 10 oracles of mean per-record cosine -> 10 values per
     method; two-sample permutation (mean difference, 10^4 perms) of the
     method's 10 seed values vs the retrain2 anchor's 10 seed values.
   - Oracle-split part: extend from penultimate-only to all four cells
     (penultimate/logits x H0/H1).
   - Distinguisher-null part: use the frozen grid as in (1).
   - Rerun, overwrite rebuttal_checks.json.
4. pointwise.py: same seed-level fix — per method seed, aggregate each
   metric (CKA, cosine, kNN overlap) over the oracle models to one value
   per seed (10 per method); baseline = per-oracle-seed mean over the
   other oracles (10 values); permutation on seed labels. Rerun,
   overwrite pointwise.json.

Verify each script end-to-end on results/m4_random; print compact tables.
Keep code minimal.
