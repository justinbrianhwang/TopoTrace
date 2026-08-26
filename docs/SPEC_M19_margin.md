# M19 — equivalence-margin uncertainty propagation + operating characteristics

Fixes final-review must-fix 9 (remaining part). CPU only. Edit ONLY
scripts/formal_stats.py (extend; do not break existing outputs).

1. JOINT bootstrap for the proximity decision: in each of 1000 bootstrap
   draws, resample the oracle seeds (with replacement) AND the method
   seeds; recompute delta = q95 of the resampled oracle's pairwise
   distances and D_UR of the resampled method vs resampled oracle; record
   the difference D_UR - delta. Decision per method per cell:
   "within-radius" if the 95% CI of (D_UR - delta) is entirely below 0,
   "outside" if entirely above, else inconclusive. Store alongside the
   existing fields as "joint_decision" and "dur_minus_delta_CI".
2. OPERATING CHARACTERISTICS of the proximity rule using held-out exact
   retrains: for each of 200 random 5/5 splits of the 10 oracle seeds,
   treat half A as the oracle (delta from its 10 pairs, grid refit not
   needed — reuse vectors), audit half B as if it were a method
   (D_UR of B vs A) with the joint rule of (1); report the fraction of
   splits where the exact retrain is declared within-radius / outside /
   inconclusive per cell. This estimates the rule's false-nonequivalence
   behavior at n=5. Store as "proximity_operating_characteristics".
3. Rerun on results/m4_random; print the new decision table and the
   operating characteristics.

Keep code minimal; reuse the vectors already computed in the script.
