# M24 — extended tuning for SalUn/RMU + range-rule operating characteristics

Two independent tasks from the v3 review's optional list.

## A. Extended tuning (GPU) — `scripts/run_salun_rmu.py` EDIT + rerun

Neither SalUn nor RMU reached the oracle operating point under the small
grid, so the audit's added value at that point is untested for them.
Widen the search (still prespecified, seed 0 only, all configs logged,
same selection rule: minimize |forget_acc - 0.9222| subject to
retain_acc >= 0.97):

- salun: sparsity in (0.1, 0.2, 0.3, 0.5), epochs in (1, 2),
  lr in (1e-5, 1e-4)   -- lower sparsity and lr should forget less
- rmu: coeff in (0.25, 0.5, 1.0, 2.0), steps in (50, 100, 300),
  alpha in (1.0, 5.0)  -- weaker steering, stronger retain anchor

If a config now satisfies the constraint, apply it to all 10 seeds and
rerun the audit exactly as before, overwriting
results/m4_random/salun_rmu.json (keep the full tuning log inside it and
record `constraint_met` per method). If still unmet, keep the best
configuration and say so. Report both the tuning table and the audit.

## B. Range-rule operating characteristics (CPU) — `scripts/joint_verdict.py` EDIT + rerun

The joint verdict's utility/forget conditions use the oracle's per-seed
[min, max] range, whose false-rejection behavior is unquantified. Add:

1. Leave-one-out: for each of the 10 oracle seeds, is it inside the
   [min, max] range formed by the other 9, for retain, forget, and test
   separately and for the conjunction? Report the fraction inside.
2. 5/5 splits: over 200 random 5/5 splits of the oracle cohort, compute
   the range from half A and check each of the five held-out seeds in
   half B; report the fraction of held-out exact retrains that the rule
   would wrongly flag, per accuracy and for the conjunction.
3. Also apply (1) to the held-out anchor cohort (retrain2): fraction of
   its 10 seeds inside the oracle's full 10-seed range.

Store under a new key "range_operating_characteristics" in
results/m4_random/joint_verdict.json and print it. Do not change the
existing verdict logic or outputs.

Both tasks: verify by running. Keep code minimal.
