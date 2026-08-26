# M18 — faithful baselines (NegGrad+, tuned SCRUB/SSD) audited end-to-end

Fixes final-review must-fix 7. GPU available. Files:

```
src/topotrace/unlearn.py   # EDIT: add neggrad_plus (nothing else changed)
scripts/run_faithful.py    # new
```

## unlearn.py addition

```python
def neggrad_plus(model, X, y, forget_idx, retain_idx, seed: int,
                 epochs: int = 2, lr: float = 1e-4, beta: float = 0.95,
                 batch_size: int = 128) -> SmallCNN:
    """NegGrad+ (Kurmanji et al. 2023): joint objective over interleaved
    minibatches, loss = beta * CE(retain batch) - (1-beta) * CE(forget
    batch), Adam. Each step samples one retain batch and one forget batch
    (forget batches cycle). Deepcopy the model; eval CPU return as the
    other methods."""
```

## scripts/run_faithful.py

CLI: `conda run -n tda python scripts/run_faithful.py` (CIFAR-10, dir
results/m4_random). Steps:

1. Load CIFAR-10 + random-5% split (topotrace.mnist.make_random_forget_split
   frac .05 seed 0) + probe_idx.npy. Load original checkpoints
   results/m4_random/models/original_{0..9}.pt into ResNet18C.
2. TUNING (prespecified budget, seed 0 only, disclosed): for each method
   pick the config minimizing |forget_acc - 0.9222| (retrain forget acc)
   subject to retain_acc >= 0.97, from these grids:
   - neggrad_plus: beta in (0.90, 0.95, 0.99), epochs in (1, 2)
   - scrub: lr in (1e-4, 5e-4), epochs in (2, 4), alpha in (0.5, 1.0)
     (topotrace.unlearn.scrub)
   - ssd: dampening_alpha in (5, 10, 30), lam in (0.5, 1.0)
   Log every config's retain/forget acc; save the selection table.
3. Apply the selected configs to all 10 original seeds (unlearning seed
   2000+s as elsewhere). Conditions named neggrad_plus, scrub_tuned,
   ssd_tuned. Save state dicts under results/m4_random/models/ and
   penultimate+logits probe embeddings appended into a NEW file
   results/m4_random/embeddings_faithful.npz (same key format).
4. AUDIT under the frozen protocol: load original+retrain(+retrain2)
   vectors from the existing embeddings.npz, fit the persistence-image
   grid on original+retrain diagrams only (per cell: penultimate/logits x
   H0/H1), vectorize the three new conditions, compute per cell: method
   permutation p vs oracle, TRR/alpha/eta; BH over the 3x4 family. Also
   compute retain/forget/test accuracy and loss-MIA AUC
   (topotrace.attacks.loss_mia_auc) per model.
5. Save results/m4_random/faithful_baselines.json (tuning table, accs,
   MIA, per-cell audit) and print a compact summary table.

Verify end-to-end (expect ~1-1.5 h total; training steps are the slow
part). Keep code minimal.
