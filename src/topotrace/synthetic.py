import numpy as np


def make_ring_dataset(
    seed: int, n_ring: int = 500, n_blob: int = 250, noise: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return a noisy unit ring (class 0) and an origin blob (class 1)."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n_ring)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    ring += noise * rng.standard_normal((n_ring, 2))
    blob = rng.normal(0, 0.2, (n_blob, 2))
    X = np.vstack((ring, blob)).astype(np.float32)
    y = np.r_[np.zeros(n_ring), np.ones(n_blob)].astype(np.int64)
    return X, y


def make_forget_split(
    X: np.ndarray, y: np.ndarray, arc_frac: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices on the selected class-0 arc and all remaining indices."""
    angles = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2 * np.pi)
    forget = (y == 0) & (angles < 2 * np.pi * arc_frac)
    return np.flatnonzero(forget), np.flatnonzero(~forget)
