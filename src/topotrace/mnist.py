"""MNIST data helpers."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_mnist(
    root: str = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalized torchvision MNIST arrays."""
    from torchvision.datasets import MNIST

    train, test = MNIST(root, train=True, download=True), MNIST(
        root, train=False, download=True
    )

    def arrays(dataset):
        X = dataset.data.numpy().astype(np.float32)[:, None] / 255.0
        return (X - 0.1307) / 0.3081, dataset.targets.numpy().astype(np.int64)

    X_train, y_train = arrays(train)
    X_test, y_test = arrays(test)
    return X_train, y_train, X_test, y_test


def load_fashion_mnist(
    root: str = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load normalized torchvision FashionMNIST arrays."""
    from torchvision.datasets import FashionMNIST

    train, test = FashionMNIST(root, train=True, download=True), FashionMNIST(
        root, train=False, download=True
    )

    def arrays(dataset):
        X = dataset.data.numpy().astype(np.float32)[:, None] / 255.0
        return (X - .2860) / .3530, dataset.targets.numpy().astype(np.int64)

    X_train, y_train = arrays(train)
    X_test, y_test = arrays(test)
    return X_train, y_train, X_test, y_test


def make_random_forget_split(
    y: np.ndarray, frac: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return uniformly shuffled forget and retain indices."""
    order = np.random.default_rng(seed).permutation(len(y))
    n_forget = int(frac * len(y))
    return order[:n_forget], order[n_forget:]


def make_class_forget_split(y: np.ndarray, cls: int = 9
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Class deletion (plan §6.3 D2): forget = all samples of `cls`."""
    return np.flatnonzero(y == cls), np.flatnonzero(y != cls)


def make_probe(
    X: np.ndarray,
    forget_idx,
    retain_idx,
    n_forget: int = 300,
    n_neighbors: int = 300,
    seed: int = 0,
) -> np.ndarray:
    """Return sampled forget points followed by nearby retained points."""
    rng = np.random.default_rng(seed)
    forget_idx, retain_idx = np.asarray(forget_idx), np.asarray(retain_idx)
    sampled = rng.choice(forget_idx, min(n_forget, len(forget_idx)), replace=False)
    target = min(n_neighbors, len(retain_idx))
    if not target:
        return sampled

    rankings = NearestNeighbors(n_neighbors=target).fit(
        X[retain_idx].reshape(len(retain_idx), -1)
    ).kneighbors(X[sampled].reshape(len(sampled), -1), return_distance=False)
    chosen = []
    seen = set()
    for rank in range(target):
        for row in rankings:
            idx = int(retain_idx[row[rank]])
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
                if len(chosen) == target:
                    return np.concatenate((sampled, np.asarray(chosen, dtype=sampled.dtype)))
    return np.concatenate((sampled, np.asarray(chosen, dtype=sampled.dtype)))
