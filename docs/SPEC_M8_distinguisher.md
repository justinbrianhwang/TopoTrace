# M8 spec — topology-based distinguisher (plan §15.1)

Question: can a simple classifier tell unlearned models from exact-retrained
models using ONLY topology fingerprints? Offline from
`results/m4_random/embeddings.npz` (key format as in SPEC_M7).

New file: `scripts/distinguish.py` (write ONLY this). CLI:
`conda run -n tda python scripts/distinguish.py results/m4_random`

Feature sets (compute per model):
1. pen_H1: persistence image, penultimate, H1 (chordal; shared imager per
   feature set fit on all models' diagrams of that cell)
2. pen_H0: same, H0
3. logits_H1: same, logits H1
4. concat: concatenation of 1-3

Classification task per unlearning method m in
{finetune, neggrad, scrub, ssd, noop}:
- class 0 = retrain + retrain2 models (20), class 1 = method m models (10).
- Leave-one-seed-out CV: fold s holds out ALL models with seed s (both
  classes); train logistic regression (sklearn, StandardScaler + 
  LogisticRegression(max_iter=1000)) on the rest; collect held-out decision
  scores. Report ROC-AUC and balanced accuracy over pooled held-out scores,
  per feature set.
- Also a pooled "any approximate method" task: class 1 = union of finetune/
  neggrad/scrub/ssd (40 models), same CV.

Print a table: rows = task, columns = feature sets, cells = AUC (balanced
acc in parens). Save results/m4_random/distinguisher.json.

Verify: run the full command (CPU only, minutes). Keep code minimal;
no plotting.
