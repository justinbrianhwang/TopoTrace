# M20 — a concrete, checkable joint verdict rule

Fixes the remaining "joint TRR/eta/utility verdict has no decision rule"
criticism. New file ONLY: `scripts/joint_verdict.py`. GPU allowed.

Goal: turn the qualitative "read TRR with eta and utility" into a
prespecified rule with per-seed oracle reference ranges, and show it
separates genuine unlearning attempts from the destructive noise control.

CLI: `conda run -n tda python scripts/joint_verdict.py`
(operates on results/m4_random; CIFAR-10 random 5%).

1. Per-seed accuracies. Load CIFAR-10 and the random-5% split. For each
   checkpoint in results/m4_random/models/ ({condition}_{seed}.pt into
   topotrace.resnet.ResNet18C) compute retain / forget / test accuracy
   per seed, for conditions: retrain (oracle), retrain2 (anchor),
   original, finetune, neggrad, scrub, ssd. Also load the faithful
   variants' per-seed accuracies from results/m4_random/
   faithful_baselines.json ("accuracy" key) instead of recomputing.
2. Noise control per seed: for sigma in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
   apply topotrace.unlearn.noise_destroy to each original checkpoint
   (seed as the noise seed) and compute the same three accuracies.
3. Oracle reference ranges: for each accuracy, the [min, max] over the 10
   oracle (retrain) seeds.
4. THE RULE (report per condition):
   - UTILITY-CONSISTENT: mean retain and mean test accuracy both lie
     within the oracle range (inclusive).
   - FORGET-CONSISTENT: mean forget accuracy lies within the oracle range.
   - TOPOLOGY-CONSISTENT: not rejected (BH q >= 0.05, from
     results/m4_random/analysis.json / faithful_baselines.json) in ANY
     gate-open cell.
   - VERDICT: "audit-consistent" iff all three hold; otherwise "flagged",
     with the failing conditions listed, and a mode label:
       * "residual" if only TOPOLOGY fails,
       * "destructive" if UTILITY fails,
       * "under-forgetting"/"over-forgetting" if only FORGET fails
         (above / below the oracle range).
5. Save results/m4_random/joint_verdict.json (ranges, per-condition
   accuracies, per-condition verdict) and print a compact table:
   condition | retain | forget | test | topo q | verdict | mode.

Verify by running it. Keep code minimal.
