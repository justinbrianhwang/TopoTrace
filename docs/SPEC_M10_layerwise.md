# M10 spec — full layerwise topological profile (plan §8.1, RQ5, Figure 4)

Checkpoints in results/m4_random/models/ ({condition}_{seed}.pt, ResNet18C).
Produce the Layer × Homology × Method TRR profile.

Files:

```
src/topotrace/resnet.py     # EDIT: add get_all_embeddings (nothing else)
scripts/layer_profile.py    # new
```

## resnet.py addition

```python
def get_all_embeddings(model, X, batch_size=256, device=None) -> dict:
    """Like get_embeddings but returns spatially-avgpooled activations after
    each stage: {"stem": post conv1/bn1/relu (n,64), "layer1": (n,64),
    "layer2": (n,128), "layer3": (n,256), "layer4": (n,512),
    "penultimate": (n,512), "logits": (n,10)}. stem..layer4 are
    adaptive-avg-pooled over spatial dims. float32 numpy."""
```

## layer_profile.py

CLI: `conda run -n tda python scripts/layer_profile.py results/m4_random`
1. Load CIFAR-10, probe = np.load(<dir>/probe_idx.npy).
2. For every checkpoint (+ noop = original checkpoints): get_all_embeddings
   on X[probe]; per layer: chordal distances -> H0/H1 diagrams (reuse
   topotrace.topology). Cache diagrams in memory; also save the layer
   embeddings to <dir>/embeddings_layers.npz ({condition}_{seed}_{layer}).
3. Per (layer, hom): shared imager, vectors, then I_topo + permutation p
   (original vs retrain) and TRR/alpha/eta per method (noop, retrain2,
   finetune, neggrad, scrub, ssd) via topotrace.metrics.
4. Save <dir>/layer_profile.json; print a compact matrix: rows = layer,
   cols = methods, cells = TRR, one block per hom dim, plus I_topo/p per
   row.
5. Figure <dir>/figure_layer_profile.png: x = layer index (stem..logits),
   y = TRR, one line per method, two panels (H0, H1); mark layers where
   p >= 0.05 with hollow markers; clamp displayed TRR to [-1, 5].

GPU is free — use it for embedding extraction (~80 checkpoints x 7 layers,
minutes). PH on 600 points x 7 layers x 80 models is CPU (~15 min).
Verify end-to-end on the real data (run the command). Keep code minimal.
