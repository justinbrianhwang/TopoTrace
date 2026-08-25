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


def make_satellites(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return three triangle clusters, a satellite, and a distant blob."""
    rng = np.random.default_rng(seed)
    angles = np.arange(3) * 2 * np.pi / 3
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    main = vertices[np.arange(440) % 3] + rng.normal(0, 0.1, (440, 2))
    satellite = rng.normal((2, 0), 0.05, (60, 2))
    blob = rng.normal((0, 3), 0.2, (250, 2))
    X = np.vstack((main, satellite, blob)).astype(np.float32)
    y = np.r_[np.zeros(500), np.ones(250)].astype(np.int64)
    return X, y


def split_satellites(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forget class-0 points within 0.5 of the satellite center."""
    forget = (y == 0) & (np.linalg.norm(X - (2, 0), axis=1) < 0.5)
    return np.flatnonzero(forget), np.flatnonzero(~forget)


def make_bridge(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two clusters joined by a narrow bridge and a distant blob."""
    rng = np.random.default_rng(seed)
    clusters = np.vstack((rng.normal((-1, 0), 0.1, (220, 2)),
                          rng.normal((1, 0), 0.1, (220, 2))))
    bridge = np.column_stack((rng.uniform(-1, 1, 60),
                              rng.normal(0, 0.03, 60)))
    blob = rng.normal((0, 3), 0.2, (250, 2))
    X = np.vstack((clusters, bridge, blob)).astype(np.float32)
    y = np.r_[np.zeros(500), np.ones(250)].astype(np.int64)
    return X, y


def split_bridge(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forget the central part of the class-0 bridge."""
    forget = (y == 0) & (np.abs(X[:, 0]) < 0.6) & (np.abs(X[:, 1]) < 0.15)
    return np.flatnonzero(forget), np.flatnonzero(~forget)


def make_figure_eight(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two tangent noisy circles and a distant blob."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, (2, 250))
    circles = np.vstack(tuple(
        np.column_stack((np.cos(a) + center, np.sin(a)))
        for a, center in zip(angles, (-1, 1))
    ))
    circles += rng.normal(0, 0.05, circles.shape)
    blob = rng.normal((0, 3), 0.2, (250, 2))
    X = np.vstack((circles, blob)).astype(np.float32)
    y = np.r_[np.zeros(500), np.ones(250)].astype(np.int64)
    return X, y


def split_figure_eight(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forget the outward-facing arc of the right circle."""
    angles = np.arctan2(X[:, 1], X[:, 0] - 1)
    forget = (y == 0) & (angles >= -np.pi / 4) & (angles < np.pi / 4)
    return np.flatnonzero(forget), np.flatnonzero(~forget)


def make_canary(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return linearly separated backgrounds with a class-0 topology canary."""
    rng = np.random.default_rng(seed)
    left = rng.uniform((-2.2, -1), (-0.2, 1), (300, 2))
    angles = rng.uniform(0, 2 * np.pi, 200)
    ring = np.column_stack((-1.2 + 0.6 * np.cos(angles),
                            0.6 * np.sin(angles)))
    right = left * (-1, 1)
    X = np.vstack((left, ring, right)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def split_canary(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forget class-0 points close to the canary circle."""
    radius = np.linalg.norm(X - (-1.2, 0), axis=1)
    forget = (y == 0) & (np.abs(radius - 0.6) < 0.12)
    return np.flatnonzero(forget), np.flatnonzero(~forget)
