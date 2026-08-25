# M3 spec — topology-targeted deletion on MNIST (plan §13, D5/D6)

Context: random and class deletion leave no oracle-distinguishable
topological imprint on MNIST/SmallCNN (results/m2_*). Next prescribed step:
delete the samples that SUPPORT persistent features, plus a matched random
control.

New file: `src/topotrace/targeted.py` (write ONLY this file).
Reuse `topotrace.cnn` (SmallCNN, train_cnn, get_embeddings) and
`topotrace.topology` (chordal_distance_matrix). Environment as before
(`conda run -n tda python ...`; MNIST is already in `data/`).

## Interfaces

```python
def score_by_cycle_support(Z: np.ndarray, maxdim: int = 1) -> np.ndarray:
    """Z: (m,d) embeddings of a candidate pool.
    ripser on chordal distance matrix with do_cocycles=True.
    score_i = sum over H1 cocycles c_j of persistence(c_j) for every i
    appearing as a vertex in any edge of c_j (plan §13.2). Returns (m,)
    float64, zeros for non-support samples."""

def make_targeted_split(X, y, n_forget: int = 3000, pool_size: int = 2000,
                        selector_seed: int = 9999, seed: int = 0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """1. Train selector SmallCNN on all data (train_cnn, seed=selector_seed,
       epochs=3) — separate seed from all evaluated models (plan §13.1).
    2. Pool: `pool_size` random train indices (rng(seed)).
    3. Score pool via score_by_cycle_support on selector penultimate
       embeddings; support set = pool samples with score > 0.
    4. Score EVERY train sample = -min chordal distance (in selector
       penultimate space) to any support sample (so support samples
       themselves rank first); break ties by rng noise 1e-9 scale.
    5. forget = top n_forget scores. Returns (forget_idx, retain_idx)."""

def make_matched_split(y, targeted_forget_idx, seed: int = 0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Random forget set with EXACTLY the same per-class counts as the
    targeted set (plan D6), disjointness with targeted NOT required.
    Returns (forget_idx, retain_idx)."""
```

`__main__` demo: build both splits (may take a few minutes: one selector
training + one ripser run on 2000 points), save
`results/m3/splits.npz` with arrays targeted_forget, targeted_retain,
matched_forget, matched_retain, then print per-class histograms of both
forget sets and assert: sizes equal n_forget, histograms identical,
targeted retain/forget disjoint and cover arange(len(y)).

Verify by running it: `conda run -n tda python src/topotrace/targeted.py`.
Note ripser H1 on 2000x2000 needs a few GB RAM — if it OOMs, drop
pool_size default to 1000. Keep code minimal.
