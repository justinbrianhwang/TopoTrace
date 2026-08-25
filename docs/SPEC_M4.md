# M4 spec — CIFAR-10 + ResNet-18 (plan main experiment)

Reuse everything from M2/M3; only the dataset and model are new. Interfaces
mirror `mnist.py` / `cnn.py` exactly so `unlearn.py`, `targeted.py`,
`analyze_m2.py` keep working (they only need the shared signatures).

New files (write ONLY these):

```
src/topotrace/cifar.py    # load_cifar10
src/topotrace/resnet.py   # ResNet18C, get_embeddings, train_resnet, evaluate
```

Environment: `conda run -n tda python ...`, torch 2.13+cu130, CUDA RTX 5090.
CIFAR-10 is already downloaded under `data/` (torchvision layout) — the
sandbox blocks downloads, so keep download=True but expect a cache hit.

## cifar.py

```python
def load_cifar10(root: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """torchvision CIFAR10. X_train (50000,3,32,32) float32, per-channel
    normalized (mean .4914,.4822,.4465 / std .2470,.2435,.2616), y int64;
    plus X_test, y_test."""
```

(Forget-split helpers are dataset-agnostic — reuse topotrace.mnist ones.)

## resnet.py

```python
class ResNet18C(torch.nn.Module):
    """torchvision resnet18(num_classes=10), CIFAR stem: conv1 replaced by
    3x3 stride-1 conv, maxpool replaced by Identity. embed(x) -> (n,512)
    post-avgpool features; forward(x) -> logits."""

def get_embeddings(model, X, batch_size: int = 512, device=None) -> dict:
    """{"penultimate": (n,512), "logits": (n,10)} — same contract as cnn.py."""

def train_resnet(X, y, idx, seed: int, epochs: int = 20, lr: float = 0.1,
                 batch_size: int = 128, init_model=None, device=None) -> ResNet18C:
    """SGD momentum .9, weight decay 5e-4, cosine annealing over `epochs`.
    Per-batch augmentation on GPU tensors: random crop (pad 4, torch ops)
    + random horizontal flip, seeded torch.Generator. AMP autocast+GradScaler
    on cuda. Fresh init under torch.manual_seed(seed) unless init_model
    (deepcopy, continue training). Returns CPU eval model.
    Print epoch train loss every 5 epochs (flush=True)."""

def evaluate(model, X, y, idx=None, batch_size: int = 512) -> float:
    """Accuracy, same contract as cnn.py."""
```

Demo in resnet.py `__main__`: train 2 epochs on first 10000 CIFAR samples,
assert test accuracy > 0.45. (~2 min on GPU.)

Verify: `conda run -n tda python src/topotrace/resnet.py` until passing.
Keep code minimal; no schedulers/frameworks beyond what is stated.
