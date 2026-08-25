# TopoTrace

**Oracle-calibrated topological auditing of machine unlearning.**

> TopoTrace evaluates whether the multi-scale topology induced by forgotten
> data in an unlearned model becomes statistically indistinguishable from the
> topology of models that never observed those data.

After a deletion request, an *unlearned* model \(M_U\) is supposed to behave
as if it had been retrained from scratch without the forget set
(\(M_R\)). TopoTrace audits this claim **inside the representation space**:
it computes persistent-homology fingerprints of a fixed probe set at each
layer and asks whether \(M_U\) falls inside the distribution of independently
retrained oracle models — not merely whether output metrics look right.

## Key ideas

- **Oracle calibration** — the reference is a *distribution* of exact
  retrains (10+ seeds), not a single retrained model. Retrain–retrain
  variation \(D_{RR}\) defines the noise floor.
- **Topological Imprint Gate** — residuals are only interpreted when the
  deletion itself produces a detectable imprint (\(D_{OR} > D_{RR}\),
  permutation test + bootstrap CI).
- **Topological Residual Ratio (TRR)** — where a method sits between
  no-op (≈1) and exact retrain (≈0); values ≫1 flag destructive updates.
- **Progress–Artifact decomposition** (α, η) — separates movement toward
  the retrain oracle from changes retraining cannot explain.
- **Topology-targeted deletion** — a benchmark that deletes the samples
  supporting persistent features, selected by an independent selector model.

## Current results (CIFAR-10, ResNet-18, random 5% deletion, 10 seeds)

- The deletion imprint is significant in every audited cell
  (penultimate/logits × H0/H1; permutation p ≤ 0.0004, CIs > 0).
- Independent exact retrains are **indistinguishable** from the oracle
  (p > 0.24 everywhere) — the audit is calibrated.
- Every approximate method tested (fine-tuning, NegGrad, SCRUB, SSD) is
  **distinguishable** from the oracle (p ≤ 0.0005 everywhere).
- **SCRUB passes a loss-based membership-inference audit (AUC 0.509 vs
  0.508 for exact retrain) yet is flagged by TopoTrace** — topology captures
  residual signal that conventional metrics miss.
- MNIST pilot: no scenario (random / class / topology-targeted) produces an
  oracle-distinguishable imprint — evidence that 5-seed positives can be
  oracle-variation artifacts, and that audit power depends on
  representation richness.

## Layout

```
src/topotrace/
  synthetic.py  models.py  train.py     # M1: synthetic ring sanity check
  mnist.py      cnn.py                  # M2: MNIST pilot
  cifar.py      resnet.py               # M4: CIFAR-10 main experiments
  unlearn.py                            # finetune / NegGrad / SCRUB / SSD
  targeted.py                           # topology-targeted deletion (D5/D6)
  topology.py   metrics.py  stats.py    # PH, TRR/α/η, permutation + bootstrap
  attacks.py                            # loss-based MIA
scripts/
  run_m1.py                             # synthetic anchors (TRR 0 / 1)
  run_m2.py [random|class|targeted|matched]
  run_m4.py [random|class|targeted|matched] [n_seeds]
  make_m4_splits.py                     # CIFAR targeted/matched forget sets
  analyze_m2.py <results_dir>           # layer × homology sweep + statistics
  eval_mia.py   <results_dir>           # MIA AUC per checkpoint
  relearn.py    <results_dir>           # forget-set relearning curves
docs/                                   # per-milestone specs
TopoTrace_research_plan.md              # full research plan
```

## Setup

```bash
conda create -n tda python=3.12
conda activate tda
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install numpy scikit-learn matplotlib ripser persim gudhi
```

Datasets load from `data/` (torchvision layout, auto-download on first
use). To reuse an existing torchvision data directory instead of a local
copy, make `data/` a symlink/junction to it — it is gitignored either way.

## Reproduce

```bash
python scripts/run_m1.py                      # synthetic ring anchors
python scripts/run_m4.py random 10            # CIFAR-10 main run (GPU, ~3 h)
python scripts/analyze_m2.py results/m4_random
python scripts/eval_mia.py results/m4_random
```

Every run records probe sample IDs, per-model embeddings, and metrics under
`results/` for offline re-analysis.

## License

[MIT](LICENSE)
