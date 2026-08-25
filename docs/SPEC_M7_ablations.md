# M7 spec — sensitivity ablations from cached embeddings (plan §16, §10.3)

All offline from `results/m4_random/embeddings.npz`
(keys `{condition}_{seed}_{layer}`, conditions original/retrain/retrain2/
noop/finetune/neggrad/scrub/ssd, seeds 0..9, layers penultimate/logits;
rows 0..299 of each embedding are FORGET probe samples, 300..599 are
retain-neighbor samples).

New file: `scripts/ablate.py` (write ONLY this). CLI:
`conda run -n tda python scripts/ablate.py results/m4_random`

Baseline cell for every ablation (unless stated): penultimate, H1,
chordal distance, persistence image, all 600 probe rows.

For EACH variant below: compute per-model vectors, then report
D_RR, D_OR, I_topo, permutation p (topotrace.stats.permutation_pvalue on
original vs retrain), and TRR for noop/retrain2/finetune/neggrad/scrub/ssd
(topotrace.metrics.trr_metrics). One printed row per variant + method
sub-rows only for TRR (compact: one line per variant with TRRs inline).

Ablations:
A. Distance function: chordal (baseline) / euclidean / correlation
   (d = sqrt(2-2*rowwise-pearson), clip>=0).
B. Vectorization (on chordal H1 diagrams):
   - persistence image (baseline, topotrace.topology.make_imager/vectorize)
   - Betti curve: 50 thresholds linspace(0, global max death), vector of
     counts alive at each threshold
   - top-K persistence: sorted (death-birth) descending, K=20, zero-padded
   - persistence entropy + total persistence (2-dim vector)
C. Probe subset: all 600 (baseline) / forget-only rows :300 /
   neighbors-only rows 300: — the forget-only vs relative-PH question (RQ6).
D. Point count: random row subsets of size 150/300/450 (rng(0), same subset
   for every model).
E. Oracle: distribution (baseline, all 10 retrains) vs single retrain —
   for each single retrain seed t, TRR_single(method) =
   median_s ||v_method_s - v_R_t|| / median_s ||v_O_s - v_R_t||; report the
   MIN and MAX over t per method to show single-oracle instability.

Save everything to results/m4_random/ablations.json (nested dict mirroring
the printed structure).

Verify: run the full command (CPU, ripser on 600x600 ~ seconds per model,
8 cond x 10 seeds x few variants — several minutes). A GPU job is running;
this script must not touch the GPU (numpy only). Keep code minimal.
