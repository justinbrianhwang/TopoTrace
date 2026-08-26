# M15 spec — paper LaTeX tables (plan §23)

Generate Tables 1-5 as standalone LaTeX (booktabs style: \toprule/\midrule/
\bottomrule, no vertical rules) from cached results.

New file: `scripts/make_tables.py` (write ONLY this).
CLI: `conda run -n tda python scripts/make_tables.py`
Output: `paper/tables/table{1..5}_<slug>.tex` — NOTE: paper/ is gitignored
on purpose (journal material); never git-add anything under paper/.

Numbers: 3-4 significant digits; p-values as <0.001 when smaller; mark
BH-significant entries with $^*$.

1. table1_settings.tex — experimental settings summary: rows = the runs
   (synthetic suite, MNIST 4 scenarios, FashionMNIST 2, SVHN 2, CIFAR-10
   random1/5/10+class+targeted+matched(+pretrained), CIFAR-100 class);
   columns = model, #seeds, forget size/definition, probe size. Hardcode
   the descriptive columns (they are protocol facts), read nothing.
2. table2_main.tex — CIFAR-10 random 5% main table: rows = methods
   (noop, retrain2, finetune, neggrad, scrub, ssd); columns = Retain acc,
   Forget acc, Test acc, MIA AUC, TRR, alpha, eta (from
   results/m4_random/metrics.json + mia.json; accs/aucs mean over seeds).
3. table3_equivalence.tex — from results/m4_random/formal_stats.json
   equivalence section: rows = method x cell (penultimate/logits x H0/H1),
   columns = D_UR, CI, delta, BH q, decision. Compact: group rows by
   method, one line per cell.
4. table4_ablation.tex — from results/m4_random/ablations.json: one block
   per ablation family (distance, vectorization, probe subset, point
   count), columns = variant, I_topo, p, TRR(retrain2), TRR(scrub); plus
   the single-oracle min-max row pair.
5. table5_operational.tex — distinguisher AUCs (results/m4_random/
   distinguisher.json: rows = tasks, cols = feature sets) with a second
   panel: relearning epoch-AUC mean±std per condition for m4_class
   (relearn.json) and the TRR-vs-MIA / TRR-vs-relearn Spearman rho (p)
   from formal_stats.json correlations.

Each .tex must compile inside a tabular/table environment standalone
(include \begin{table}...\caption{...}\label{tab:...}\end{table}).
Verify: run the script, then check each file exists and contains
\bottomrule. Print a short preview of each table to stdout. CPU only.
Keep code minimal.
