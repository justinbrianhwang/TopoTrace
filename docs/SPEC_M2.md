# TopoTrace Milestone 2 Spec (MNIST pilot)

Goal (plan §21.1 row 4): MNIST + Small CNN + random 5% deletion, baselines
{no-op, exact retrain, fine-tune, NegGrad, SCRUB, SSD}, 5 seeds, H0/H1 TRR
audit on the penultimate layer + conventional accuracy metrics.

Environment: `conda run -n tda python ...` (torch 2.11 cu126, torchvision,
numpy, ripser, persim, sklearn). Use CUDA if available, fall back to CPU.
Reuse existing `topotrace.topology` / `topotrace.metrics` — do not modify.

Style: same as SPEC_M1 — plain functions, explicit `seed` args, minimal code,
no config frameworks. MNIST downloads to `data/` (gitignored).

New files:

```
src/topotrace/mnist.py       # Agent A
src/topotrace/cnn.py         # Agent A
src/topotrace/unlearn.py     # Agent B
scripts/run_m2.py            # PM (do not write)
```

---

## Agent A interfaces

### mnist.py

```python
def load_mnist(root: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """torchvision MNIST, download=True. Returns X_train (60000,1,28,28)
    float32 in [0,1] normalized with mean .1307 / std .3081, y_train int64,
    X_test, y_test."""

def make_random_forget_split(y: np.ndarray, frac: float = 0.05, seed: int = 0
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random forget set of size frac*len(y). Returns
    (forget_idx, retain_idx)."""

def make_probe(X: np.ndarray, forget_idx, retain_idx, n_forget: int = 300,
               n_neighbors: int = 300, seed: int = 0) -> np.ndarray:
    """Fixed model-independent probe (plan §8.4/§9.2): subsample n_forget
    forget samples, then their raw-pixel nearest retain samples (union of
    1-NN per forget sample, deduped, topped up with next-nearest until
    n_neighbors retained samples; use sklearn NearestNeighbors on flattened
    pixels). Returns probe_idx into X (forget part first)."""
```

### cnn.py

```python
class SmallCNN(torch.nn.Module):
    """conv(1->32,3) ReLU, conv(32->64,3) ReLU, maxpool2, flatten,
    fc(->128) ReLU, fc(->10). forward -> logits."""

def get_embeddings(model, X: np.ndarray, batch_size: int = 512,
                   device: str | None = None) -> dict[str, np.ndarray]:
    """eval, no_grad, batched. Returns {"penultimate": (n,128),
    "logits": (n,10)} float32 numpy."""

def train_cnn(X, y, idx, seed: int, epochs: int = 5, lr: float = 1e-3,
              batch_size: int = 128, init_model=None,
              device: str | None = None) -> SmallCNN:
    """Adam + CrossEntropy, shuffled minibatches (torch.Generator seeded).
    Fresh init with torch.manual_seed(seed) unless init_model given
    (then deepcopy and continue). Returns model on CPU, eval mode."""

def evaluate(model, X, y, idx=None, batch_size: int = 512) -> float:
    """Accuracy on X[idx] (all if idx None)."""
```

Demo in cnn.py `__main__`: train 1 epoch on first 5000 MNIST samples,
assert test-subset accuracy > 0.9.

## Agent B interfaces — unlearn.py

All take the original model, never mutate it (deepcopy inside), return a new
eval-mode CPU model. `X, y` are the full train arrays; idx arrays select
rows. Import `train_cnn` from `topotrace.cnn` per the Agent A interface
above (files are developed in parallel — code to the spec signature).

```python
def finetune(model, X, y, forget_idx, retain_idx, seed: int,
             epochs: int = 2, lr: float = 1e-4) -> SmallCNN:
    """train_cnn on retain_idx with init_model=model."""

def neggrad(model, X, y, forget_idx, retain_idx, seed: int,
            steps: int = 100, lr: float = 1e-4,
            batch_size: int = 128) -> SmallCNN:
    """Gradient ASCENT on forget set: minimize -CE on forget minibatches
    for `steps` steps (Adam). Stop early if forget accuracy < 0.05."""

def scrub(model, X, y, forget_idx, retain_idx, seed: int,
          epochs: int = 2, lr: float = 1e-4, batch_size: int = 128,
          alpha: float = 0.5, gamma: float = 1.0) -> SmallCNN:
    """Simplified SCRUB: teacher = frozen copy of original.
    Each epoch: (1) max-step over forget minibatches maximizing
    KL(student || teacher) (i.e. minimize -KL), (2) min-step over retain
    minibatches minimizing gamma*CE + alpha*KL(student || teacher).
    KL over softmax with temperature 2."""

def ssd(model, X, y, forget_idx, retain_idx,
        dampening_alpha: float = 10.0, lam: float = 1.0,
        batch_size: int = 128, max_batches: int = 50) -> SmallCNN:
    """Selective Synaptic Dampening (Foster et al. 2024, simplified).
    Importance(param) = mean over batches of squared grad of CE loss,
    computed separately on forget (up to max_batches) and retain batches.
    For each parameter tensor elementwise where
    imp_f > dampening_alpha * imp_r:
        theta *= min(lam * imp_r / imp_f, 1).
    No training steps. Deterministic given inputs."""
```

Demo in unlearn.py `__main__`: build a tiny SmallCNN on 2000 MNIST samples
(1 epoch), pick 100 forget samples, run each method, assert each returns a
model whose parameters differ from the original (except that finetune/scrub
still classify: retain-subset accuracy > 0.5) and neggrad forget accuracy
drops below original's. Keep the demo under ~60s runtime.

---

Both agents: verify with `conda run -n tda python src/topotrace/<file>.py`
and fix until the demo passes. If torchvision is still installing, retry
after a minute. Do not touch other files.
