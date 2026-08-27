# M25 — RMU refinement to the oracle's forget operating point

One task, from the v4 review's optional list. GPU required. Env: conda `tda`.

## Motivation

The widened grid (M24) put RMU inside the retain constraint but left its
mean forget accuracy at 0.9466, still 0.016 above the oracle range
[0.918, 0.9308]; target is 0.9222. Tuned SCRUB matches the oracle to
within 0.002. We want RMU held to the same standard, so that a rejection
cannot be attributed to a forget-accuracy mismatch.

The grid trend at retain weight alpha=5 is monotone in steps and retain
accuracy is flat there:

    c=2 T=50  a=5  retain=0.9781 forget=0.9748
    c=2 T=100 a=5  retain=0.9772 forget=0.9752
    c=2 T=300 a=5  retain=0.9764 forget=0.9420

so more steps should reach 0.9222 without breaking retain >= 0.97.

## Task — `scripts/run_salun_rmu.py` EDIT + rerun

Add a **third, separately logged refinement stage for RMU only**. Do not
touch the SalUn path, the existing grids, or their logs — stage 2 output
must survive verbatim in the JSON.

Refinement grid (prespecified, seed 0 only, alpha fixed at the
stage-2 selected value of 5):

    coeff in (2.0, 4.0), steps in (350, 400, 450, 500, 600)

10 configurations. Same selection rule as before: minimize
|forget_acc - 0.9222| subject to retain_acc >= 0.97.

Store the refinement under `tuning["rmu"]["refinement"]` with the same
shape as the existing stage (`rows`, `selected`, `constraint_met`), and
keep the stage-2 `rows`/`selected` where they are. Record which stage
supplied the final configuration under `tuning["rmu"]["final_stage"]`.

If the refinement finds an eligible configuration with smaller
`target_error` than stage 2's, it becomes RMU's final configuration:
apply it to all 10 seeds and rerun the audit exactly as before
(same probe, same frozen persistence-image grid, same statistics),
overwriting `accuracy["rmu"]`, `mia_auc["rmu"]`, and the `rmu` entries
in `audit`. SalUn's 10-seed results and the anchor entries must be
byte-identical to what is already in the file — recompute them only if
that is cheaper than preserving them, and verify they match.

If no refinement configuration beats stage 2, keep stage 2's and say so.

## Reporting

Print, and leave in the JSON:

1. the full 10-row refinement table (config, retain, forget, eligible,
   target_error)
2. the final RMU configuration and which stage it came from
3. the 10-seed mean retain / forget / test accuracy and mean MIA AUC
4. the per-cell audit for RMU: TRR, alpha, eta, p, q, reject

Report the audit outcome faithfully **whether or not RMU is still
rejected**. A configuration that matches the oracle on all output
criteria and is no longer rejected topologically is a legitimate and
important negative result; do not tune further to avoid it, and do not
soften the reporting if it happens.

Verify by running. Keep the diff minimal.
