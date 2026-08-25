# TopoTrace Milestone 1 Spec (Synthetic Ring)

Goal (from research plan §30): one figure with persistence diagrams for
Original / Exact-Retrain / No-op / Fine-tuned models on a synthetic ring,
plus TRR anchors: Retrain ≈ 0, No-op ≈ 1, Fine-tune in between.

Environment: run/test with `conda run -n tda python ...`
(Python 3.12, torch 2.11, numpy, ripser, persim, scikit-learn, matplotlib).

Language: Python, plain functions, no classes unless necessary, no config
framework, no CLI args beyond what is specified. Type hints welcome.
All randomness controlled by explicit integer `seed` arguments
(use `numpy.random.default_rng(seed)` / `torch.manual_seed(seed)`).

Package layout (write ONLY your assigned files):

```
src/topotrace/__init__.py        # empty, already created
src/topotrace/synthetic.py       # Agent A
src/topotrace/models.py          # Agent A
src/topotrace/train.py           # Agent A
src/topotrace/topology.py        # Agent B
src/topotrace/metrics.py         # Agent B
scripts/run_m1.py                # PM (do not write)
```

Modules are imported as `topotrace.synthetic` etc. (scripts add `src/` to
`sys.path`). Do not add packaging files.

---

## Agent A interfaces

### synthetic.py

```python
def make_ring_dataset(seed: int, n_ring: int = 500, n_blob: int = 250,
                      noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """2D binary classification.
    Class 0: points on unit circle (radius 1) + gaussian noise*N(0,I).
    Class 1: gaussian blob at origin, std 0.2.
    Returns X (n,2) float32, y (n,) int64."""

def make_forget_split(X, y, arc_frac: float = 0.25
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Forget set = ring-class points whose angle atan2(x1,x0) lies in
    [0, 2*pi*arc_frac) (a contiguous arc — its removal breaks loop support).
    Returns (forget_idx, retain_idx) index arrays over X."""
```

### models.py

```python
class MLP(torch.nn.Module):
    """2 -> 64 -> 64 -> 2. ReLU. forward(x) -> logits."""

def get_embeddings(model, X: np.ndarray) -> dict[str, np.ndarray]:
    """Run model on X (n,2) float32, no grad, eval mode.
    Returns {"h1": (n,64), "h2": (n,64), "logits": (n,2)} float32 numpy,
    where h1/h2 are post-ReLU hidden activations."""
```

### train.py

```python
def train(X, y, idx, seed: int, epochs: int = 200, lr: float = 1e-2,
          init_model: MLP | None = None) -> MLP:
    """Train MLP on X[idx], y[idx]. Full-batch Adam, CrossEntropyLoss.
    If init_model given, deep-copy it and continue training (fine-tune);
    else fresh init with torch.manual_seed(seed).
    Deterministic given (seed, idx). Returns trained model (eval mode)."""
```

Conditions built by caller (PM):
- Original_s = train(all idx, seed=s)
- Retrain_s  = train(retain idx, seed=1000+s)
- Noop_s     = Original_s (same object)
- Finetune_s = train(retain idx, seed=2000+s, epochs=50, init_model=Original_s)

Include a `demo()` under `if __name__ == "__main__":` in train.py that
trains one Original model and asserts train accuracy > 0.95.

## Agent B interfaces

### topology.py

```python
def chordal_distance_matrix(Z: np.ndarray) -> np.ndarray:
    """L2-normalize rows (eps=1e-12), return sqrt(max(0, 2-2*Z@Z.T)),
    symmetric, zero diagonal. (research plan §8.2–8.3)"""

def persistence_diagrams(D: np.ndarray, maxdim: int = 1) -> list[np.ndarray]:
    """ripser.ripser(D, distance_matrix=True, maxdim).
    Returns [H0 (k0,2), H1 (k1,2)] with np.inf deaths in H0 replaced by
    the max finite death in the diagram set (or D.max() if none)."""

def vectorize(dgms: list[np.ndarray], imager) -> np.ndarray:
    """Persistence image of H1 diagram via a shared persim.PersistenceImager,
    flattened float64. Empty diagram -> zero vector."""

def make_imager(all_h1_dgms: list[np.ndarray]) -> "persim.PersistenceImager":
    """One PersistenceImager fit on the union of all H1 diagrams
    (so every model is vectorized on the same grid). pixel_size chosen so
    the grid is roughly 20x20."""
```

### metrics.py  (research plan §11)

```python
def trr_metrics(v_O: list[np.ndarray], v_R: list[np.ndarray],
                v_U: list[np.ndarray], eps: float = 1e-12) -> dict:
    """v_* are per-seed topology vectors.
    D_RR = median pairwise ||v_R_t - v_R_t'|| (t != t')
    D_OR = median over all (s,t) ||v_O_s - v_R_t||
    D_UR = median over all (s,t) ||v_U_s - v_R_t||
    TRR  = (D_UR - D_RR) / (D_OR - D_RR + eps)
    alpha, eta per §11.6–11.7 using mean vectors v_O_bar, v_R_bar and
    v_U_bar: alpha = <v_U_bar - v_O_bar, v_R_bar - v_O_bar> / (||v_R_bar - v_O_bar||^2 + eps),
    eta = ||(v_U_bar - v_O_bar) - alpha*(v_R_bar - v_O_bar)|| / (||v_R_bar - v_O_bar|| + eps).
    Returns dict with keys: D_RR, D_OR, D_UR, TRR, alpha, eta, I_topo (=D_OR-D_RR)."""
```

Include in metrics.py a `demo()` under `__main__` with synthetic vectors
asserting: identical U and R distributions -> TRR ≈ 0; U == O -> TRR ≈ 1.

---

Both agents: after writing, run your demo with
`conda run -n tda python <file>` and fix until it passes. Keep code short;
no logging frameworks, no docs beyond docstrings.
