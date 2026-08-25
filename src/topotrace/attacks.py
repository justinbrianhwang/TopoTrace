"""Membership-inference attacks."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from topotrace.resnet import _default_device


def sample_losses(model, X, y, idx, batch_size: int = 512,
                  device=None) -> np.ndarray:
    """Return per-sample cross-entropy losses for indexed rows."""
    device = torch.device(device or _default_device())
    original_device = next(model.parameters()).device
    model.to(device).eval()
    losses = []
    with torch.no_grad():
        idx = np.asarray(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            logits = model(torch.as_tensor(X[batch_idx], device=device))
            target = torch.as_tensor(y[batch_idx], device=device)
            losses.append(torch.nn.functional.cross_entropy(
                logits, target, reduction="none").cpu().numpy())
    model.to(original_device)
    return np.concatenate(losses).astype(np.float64)


def loss_mia_auc(model, X, y, forget_idx, X_test, y_test,
                 n_test: int = 3000, seed: int = 0) -> float:
    """Return Yeom-style loss-threshold membership-inference AUC."""
    test_idx = np.random.default_rng(seed).choice(
        len(y_test), n_test, replace=False)
    member = sample_losses(model, X, y, forget_idx)
    nonmember = sample_losses(model, X_test, y_test, test_idx)
    labels = np.r_[np.ones(len(member)), np.zeros(len(nonmember))]
    return float(roc_auc_score(labels, -np.r_[member, nonmember]))


def demo():
    from topotrace.cnn import SmallCNN, train_cnn
    from topotrace.mnist import load_mnist

    X, y, X_test, y_test = load_mnist()
    train_idx = np.arange(5000)
    forget_idx = np.arange(500)
    model = train_cnn(X, y, train_idx, seed=0, epochs=2)
    assert loss_mia_auc(model, X, y, forget_idx, X_test, y_test) > .5
    torch.manual_seed(0)
    assert .3 < loss_mia_auc(
        SmallCNN(), X, y, forget_idx, X_test, y_test) < .7


if __name__ == "__main__":
    demo()
