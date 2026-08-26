# M14 spec — paper data-figure PDF set (plan §22)

Produce publication-quality matplotlib PDFs from cached results. The user
draws concept figures (framework, PH explainer) in Illustrator — this spec
covers DATA figures only.

New file: `scripts/make_figures.py` (write ONLY this).
CLI: `conda run -n tda python scripts/make_figures.py`
Output: `paper/figures/*.pdf` (create dirs).

Global style (define once):
- matplotlib rcParams: font.family serif, font.size 8, axes.linewidth .6,
  pdf.fonttype 42; figure widths 3.3in (single col) / 6.9in (double).
- Fixed colors: oracle/retrain #2166ac (blue), original/noop #555555
  (gray), finetune #e08214, neggrad #b2182b, scrub #7b3294, ssd #35978f.
  One shared legend style. Label methods with these exact names.

Figures (all inputs exist under results/):

1. fig_imprint_gate.pdf — bar chart of I_topo with bootstrap-CI error bars
   for penultimate-H1 (and pen-H0 as second panel) across conditions:
   MNIST class (results/m2_class/analysis.json if present else skip),
   FashionMNIST random/class, SVHN random/class, CIFAR-10 random1/random/
   random10/class/targeted/matched, CIFAR-100 class (exp_* and m4_* dirs'
   analysis.json). Hatch/star bars with p<0.05. Log-scale y if needed.
2. fig_layer_profile.pdf — re-render results/m4_random/layer_profile.json
   in the shared style (two panels H0/H1, TRR vs layer, hollow markers at
   p>=.05, clamp [-1, 5]).
3. fig_progress_artifact.pdf — alpha vs eta scatter, one marker per
   method per scenario (m4_random/class/targeted/matched metrics.json),
   method = color, scenario = marker shape; annotate the destructive noise
   models from results/m4_random/destructive.json as gray crosses with a
   thin path in order of sigma. Reference lines alpha=0,1 and eta=0.
4. fig_metric_disagreement.pdf — heatmap methods x metrics for CIFAR-10
   random: rows finetune/neggrad/scrub/ssd; columns: forget-acc gap vs
   retrain (|fgt - retrain fgt| < .02), MIA pass (|auc-.5| < .02,
   mia.json), cosine pass (p>=.05, pointwise.json), CKA pass, kNN pass,
   TopoTrace pass (BH q>=.05, formal_stats.json). Green = passes (looks
   unlearned), red = flagged. This is the "conventional pass / topology
   flagged" figure.
5. fig_ratio_dose.pdf — I_topo (pen H0 and H1) vs deletion ratio
   {1,5,10}% with CI bars (analysis.json of m4_random1/random/random10),
   plus horizontal line at 0.
6. fig_distinguisher.pdf — grouped bar chart of leave-one-seed-out AUC per
   method x feature set (results/m4_random/distinguisher.json), hline at
   0.5 chance.
7. fig_destructive.pdf — TRR and eta vs sigma (two y-axes or two panels)
   with test-accuracy overlay, from destructive.json.
8. fig_relearn.pdf — mean step-level relearning curves for the class
   scenario (results/m4_class/relearn.json "step_curves" per condition;
   x = optimizer step recorded every 5, y = forget acc; shaded ±1 std).

Skip a figure gracefully (print a note) if its input file is missing.
Also write paper/figures/figures_manifest.json listing produced files and
their inputs. Verify by running the script and confirming every PDF > 5KB.
Keep code minimal — one function per figure, shared style helper.
