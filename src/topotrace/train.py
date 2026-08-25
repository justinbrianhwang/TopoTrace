from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topotrace.models import MLP


def train(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    seed: int,
    epochs: int = 200,
    lr: float = 1e-2,
    init_model: MLP | None = None,
) -> MLP:
    """Train a fresh MLP or fine-tune a deep copy using full-batch Adam."""
    torch.manual_seed(seed)
    model = copy.deepcopy(init_model) if init_model is not None else MLP()
    x = torch.as_tensor(X[idx], dtype=torch.float32)
    target = torch.as_tensor(y[idx], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_fn(model(x), target).backward()
        optimizer.step()
    return model.eval()


def demo() -> None:
    from topotrace.synthetic import make_ring_dataset

    X, y = make_ring_dataset(seed=0)
    model = train(X, y, np.arange(len(y)), seed=0)
    with torch.no_grad():
        accuracy = (model(torch.from_numpy(X)).argmax(1).numpy() == y).mean()
    assert accuracy > 0.95, accuracy


if __name__ == "__main__":
    demo()
