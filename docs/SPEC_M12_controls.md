# M12 spec — destructive control + pretrained-init support
(plan §7.2 destructive control, §16 pretrained-vs-scratch)

Files:

```
src/topotrace/unlearn.py   # EDIT: add noise_destroy (nothing else changed)
src/topotrace/resnet.py    # EDIT: pretrained flag on train_resnet
scripts/destructive.py     # new
```

## unlearn.py addition

```python
def noise_destroy(model, sigma: float = 0.1, seed: int = 0):
    """Destructive control: deepcopy + add N(0, sigma*std(param)) noise to
    every float parameter tensor (per-tensor std scaling), torch.manual_seed
    for determinism. No training. Returns CPU eval model."""
```

## resnet.py edit

`train_resnet(..., pretrained: bool = False)`: when True and init_model is
None, load torchvision resnet18(weights=IMAGENET1K_V1) into self.net
EXCEPT conv1 (shape mismatch with CIFAR stem) and fc (class count) — load
with strict=False after deleting those keys from the state dict; keep the
seeded fresh init for conv1/fc.

## destructive.py

CLI: `... scripts/destructive.py results/m4_random`
For sigma in (0.02, 0.05, 0.1, 0.2): apply noise_destroy to each saved
original checkpoint (10 seeds); evaluate retain/forget/test accuracy;
extract penultimate embeddings on the saved probe; compute TRR/alpha/eta
vs the cached original/retrain vectors (rebuild vectors from embeddings.npz
exactly as analyze_m2 does for penultimate H1, adding the noise models to
the shared imager fit). Print sigma-table: accs + TRR + alpha + eta; the
expected signature is utility collapse with large eta (topology moves
AWAY from both original and retrain). Save <dir>/destructive.json.

Verify destructive.py end-to-end on the real data (GPU free, minutes).
For the pretrained flag: verify only that
`train_resnet(X, y, idx[:2000], seed=0, epochs=1, pretrained=True)` runs
and returns a model (1-epoch smoke, GPU) — the full pretrained experiment
is launched separately by the PM. Keep code minimal.
