# M17B — full-protocol quantile baseline, H0-cap ablation, H5 test

Fixes findings 3, 12, 14 of the final adversarial review. New files ONLY:
scripts/quantile_audit.py, scripts/h0_cap_ablation.py, scripts/h5_test.py.
CPU only, cached embeddings.

1. quantile_audit.py — the distance-quantile fingerprint run through the
   COMPLETE audit protocol. For every results/* dir containing
   embeddings.npz (glob them): per layer (penultimate, logits) the
   fingerprint = 50 quantiles (linspace .02...98) of the chordal
   pairwise-distance upper triangle. Then exactly as the main audit:
   I_topo, seed-level bootstrap CI (1000 resamples), gate permutation p
   (energy distance, 10^4); method tests (noop/retrain2/finetune/neggrad/
   scrub/ssd where present) with BH correction over the method x layer
   family per scenario; anchor TRR with admissibility |TRR| <= 0.5.
   Save <dir>/quantile_audit.json per dir + print one summary matrix
   (scenario x layer: gate p, CI sign, anchor TRR, worst BH q).
2. h0_cap_ablation.py — on results/m4_random penultimate H0, compare three
   conventions for the infinite H0 death: (a) current per-model cap (max
   finite death, as in topotrace.topology.persistence_diagrams), (b) DROP
   the infinite bar entirely, (c) global fixed cap = max pairwise chordal
   distance over ALL models' probe embeddings. For each: frozen-grid
   persistence-image vectors, I_topo, gate p, anchor TRR, method TRRs, and
   the fraction of the mean fingerprint norm contributed by the essential
   bar's pixel mass under (a). Save results/m4_random/h0_cap_ablation.json.
3. h5_test.py — direct targeted-vs-matched difference test: recompute
   frozen-grid persistence-image vectors for original+retrain in
   results/m4_targeted and results/m4_matched (all four cells); statistic
   = I_topo(targeted) - I_topo(matched) per cell; seed-level bootstrap
   (resample seeds independently within each scenario, 1000 draws) CI for
   the difference. Save results/h5_difference.json; print per-cell
   difference + CI + whether CI excludes 0.

Reuse topotrace.{topology,metrics,stats}. Verify by running all three.
Keep code minimal.
