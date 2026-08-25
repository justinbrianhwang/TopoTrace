"""Small CNN training and embedding helpers."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


def _default_device():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
            return "cuda"
    return "cpu"


class SmallCNN(nn.Module):
    """Two-convolution MNIST classifier with 128-wide embeddings."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def embed(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        return torch.relu(self.fc1(torch.flatten(x, 1)))

    def forward(self, x):
        return self.fc2(self.embed(x))


def get_embeddings(
    model, X: np.ndarray, batch_size: int = 512, device: str | None = None
) -> dict[str, np.ndarray]:
    """Return batched penultimate activations and logits."""
    device = device or _default_device()
    original_device = next(model.parameters()).device
    model.to(device).eval()
    penultimate, logits = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.as_tensor(X[start : start + batch_size], device=device)
            embedding = model.embed(batch)
            penultimate.append(embedding.cpu().numpy())
            logits.append(model.fc2(embedding).cpu().numpy())
    model.to(original_device)
    return {
        "penultimate": np.concatenate(penultimate).astype(np.float32, copy=False),
        "logits": np.concatenate(logits).astype(np.float32, copy=False),
    }


def train_cnn(
    X,
    y,
    idx,
    seed: int,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 128,
    init_model=None,
    device: str | None = None,
) -> SmallCNN:
    """Train on indexed rows with shuffled Adam minibatches."""
    torch.manual_seed(seed)
    device = device or _default_device()
    model = deepcopy(init_model) if init_model is not None else SmallCNN()
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X_train = torch.as_tensor(np.asarray(X[idx], dtype=np.float32))
    y_train = torch.as_tensor(np.asarray(y[idx], dtype=np.int64))
    generator = torch.Generator().manual_seed(seed)

    for _ in range(epochs):
        for batch_idx in torch.randperm(len(y_train), generator=generator).split(batch_size):
            logits = model(X_train[batch_idx].to(device))
            loss = loss_fn(logits, y_train[batch_idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.cpu().eval()


def evaluate(model, X, y, idx=None, batch_size: int = 512) -> float:
    """Return classification accuracy on all or selected rows."""
    if idx is None:
        idx = np.arange(len(y))
    predictions = get_embeddings(model, X[idx], batch_size)["logits"].argmax(1)
    return float(np.mean(predictions == np.asarray(y)[idx]))


def demo():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from topotrace.mnist import load_mnist

    X_train, y_train, X_test, y_test = load_mnist()
    model = train_cnn(X_train, y_train, np.arange(5000), seed=0, epochs=1)
    assert evaluate(model, X_test, y_test, np.arange(2000)) > 0.85


if __name__ == "__main__":
    demo()
