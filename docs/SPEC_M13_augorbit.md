# M13 spec — augmentation-orbit topology (plan §9.3, secondary analysis)

Question: for individual forget samples, does the topology of the sample's
augmentation orbit {h(a_j(x))} differ between unlearned and retrained
models?

New file: `scripts/aug_orbit.py` (write ONLY this).

CLI: `... scripts/aug_orbit.py results/m4_random [--n-samples 30]
[--n-augs 64]`
1. CIFAR-10; forget split as in eval_mia (dir suffix logic, random ->
   frac .05 seed 0). Pick n_samples forget samples (rng(0)).
2. Augmentations: the same random-crop(pad4)+flip family used in training
   (reuse/replicate topotrace.resnet._augment on a batch of n_augs copies,
   torch.Generator(seed=1000+sample_rank) so every model sees the SAME
   orbit inputs).
3. For each checkpoint (conditions original/retrain/retrain2/finetune/
   neggrad/scrub/ssd, 10 seeds) and each chosen sample: penultimate
   embeddings of the orbit (n_augs, 512) -> chordal distances -> H0
   diagram -> orbit statistics: total persistence and persistence entropy
   (H1 on 64 points is unstable; H0 only).
4. Per-sample per-model feature = [H0 total persistence, H0 entropy];
   model-level vector = concat over the n_samples samples (2*n_samples).
5. I_topo + permutation p (original vs retrain) on these vectors, TRR per
   method — same metrics/stats helpers as everywhere.
6. Save <dir>/aug_orbit.json; print the imprint row + method TRR table.

GPU free — extraction is 80 models x 30 samples x 64 augs (batched),
minutes. Verify end-to-end on the real data. Keep code minimal.
