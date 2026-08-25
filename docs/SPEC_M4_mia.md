# M4 MIA spec — conventional privacy metric (plan §12.2)

Goal: loss-based membership-inference AUC per model so TopoTrace results can
be compared against a conventional unlearning metric (Gate 3 evidence).

New files (write ONLY these):

```
src/topotrace/attacks.py
scripts/eval_mia.py
```

Environment: `conda run -n tda python ...` as before.

## attacks.py

```python
def sample_losses(model, X, y, idx, batch_size: int = 512,
                  device=None) -> np.ndarray:
    """Per-sample cross-entropy losses (no reduction), float64, eval/no-grad,
    batched, device auto like topotrace.resnet._default_device."""

def loss_mia_auc(model, X, y, forget_idx, X_test, y_test,
                 n_test: int = 3000, seed: int = 0) -> float:
    """Yeom-style loss-threshold MIA: scores = -loss; positives = forget
    samples, negatives = `n_test` random test samples (rng(seed)).
    Returns sklearn roc_auc_score. ~0.5 => forget set indistinguishable
    from unseen data; >0.5 => membership signal remains."""
```

Demo in `__main__`: build two tiny logistic-like models is overkill — instead
use a SmallCNN on MNIST (data/ cached): train 2 epochs on first 5000, assert
loss_mia_auc(model, ..., forget_idx=first 500 training samples) > 0.5 and
that a fresh UNtrained SmallCNN gives AUC in (0.3, 0.7). Runs in ~1 min.

## eval_mia.py

CLI: `conda run -n tda python scripts/eval_mia.py results/m4_random`
- loads CIFAR-10 via topotrace.cifar and the scenario's forget split:
  scenario from the dir name (random -> make_random_forget_split(y, .05, 0),
  class -> make_class_forget_split(y, 9)).
- iterates over results/<dir>/models/*.pt (files named {condition}_{seed}.pt),
  loads each into topotrace.resnet.ResNet18C, computes loss_mia_auc.
- prints mean±std AUC per condition and writes results/<dir>/mia.json
  ({condition: [auc per seed]}).
- If the models dir does not exist yet (a training run may still be
  writing it), print a clear message and exit 0 — do not wait.

Verify: run the attacks.py demo until it passes. For eval_mia.py, verify
end-to-end with 2-3 tiny SmallCNN checkpoints you save to a scratch dir
mimicking the models/ layout (MNIST arrays, loading into SmallCNN instead of
ResNet via a --model cnn flag on the CLI; default stays resnet), then delete
them. Keep code minimal.
