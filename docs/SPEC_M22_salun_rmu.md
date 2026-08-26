.# M22 — SalUn and RMU (representation-steering) audited

Closes "SalUn / representation-steering methods never audited". Files:

```
src/topotrace/unlearn.py   # EDIT: add salun and rmu (nothing else changed)
scripts/run_salun_rmu.py   # new
```

## unlearn.py additions

```python
def salun(model, X, y, forget_idx, retain_idx, seed: int, sparsity: float = 0.5,
          epochs: int = 2, lr: float = 1e-4, batch_size: int = 128):
    """SalUn (Fan et al., ICLR 2024). (1) Weight saliency: accumulate
    |grad| of the forget-set cross-entropy at the original weights over up
    to 20 batches; keep the top `sparsity` fraction of entries per
    parameter tensor as a binary mask. (2) Random-label unlearning: for
    `epochs`, alternate a forget batch with RANDOM labels (drawn per batch
    from a seeded generator, excluding the true label) and a retain batch
    with true labels, minimizing cross-entropy in both cases; after each
    optimizer step, restore the non-masked entries to their original
    values so only salient weights move. Adam, deepcopy, CPU eval return."""

def rmu(model, X, y, forget_idx, retain_idx, seed: int, steps: int = 300,
        lr: float = 1e-4, coeff: float = 6.0, alpha: float = 1.0,
        batch_size: int = 128):
    """RMU (Li et al., ICML 2024), adapted from LLM layers to a vision
    penultimate layer. Freeze a copy of the original as the reference. Draw
    one fixed random unit vector u in R^512 (seeded). Each step samples a
    forget batch and a retain batch and minimizes
      || h_f(student) - coeff * u ||^2  +  alpha * || h_r(student) - h_r(reference) ||^2
    where h is the penultimate embedding (model.embed). Adam, deepcopy,
    CPU eval return."""
```

## scripts/run_salun_rmu.py

CLI: `conda run -n tda python scripts/run_salun_rmu.py`. CIFAR-10 random
5% (results/m4_random), the same protocol as scripts/run_faithful.py:

1. Load the 10 original checkpoints and the probe.
2. TUNING (prespecified, seed 0 only, all configs logged): choose the
   config minimizing |forget_acc - 0.9222| subject to retain_acc >= 0.97:
   - salun: sparsity in (0.3, 0.5), epochs in (1, 2)
   - rmu: coeff in (2.0, 6.0), steps in (150, 300)
   If NO config satisfies the retain constraint, select the config with
   the highest retain accuracy and record that the constraint was not met.
3. Apply the selected configs to all 10 seeds (unlearning seed 2000+s),
   save state dicts under results/m4_random/models/ and probe embeddings
   into results/m4_random/embeddings_salun_rmu.npz.
4. AUDIT under the frozen protocol exactly as run_faithful.py does (grid
   fitted on original+retrain diagrams only; four cells; BH over the
   2x4 family), plus retain/forget/test accuracy and loss-MIA AUC.
5. Save results/m4_random/salun_rmu.json; print a compact summary.

Verify end-to-end (GPU, ~1 h). Keep code minimal.
