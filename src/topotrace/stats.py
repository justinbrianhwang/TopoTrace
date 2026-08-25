"""Statistical helpers for topology-vector comparisons."""

import numpy as np


def energy_distance(A: list[np.ndarray], B: list[np.ndarray]) -> float:
    """Return the two-sample energy distance using off-diagonal within means."""
    A, B = np.stack(A), np.stack(B)

    def within(X):
        n = len(X)
        return 0.0 if n < 2 else np.linalg.norm(
            X[:, None] - X[None, :], axis=-1
        ).sum() / (n * (n - 1))

    return float(2 * np.linalg.norm(A[:, None] - B[None, :], axis=-1).mean()
                 - within(A) - within(B))


def permutation_pvalue(A, B, n_perm: int = 10000, seed: int = 0) -> float:
    """Return a two-sample permutation p-value for the energy distance."""
    A, B = list(A), list(B)
    pool, n_A = A + B, len(A)
    observed = energy_distance(A, B)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(len(pool))
        count += energy_distance([pool[i] for i in order[:n_A]],
                                 [pool[i] for i in order[n_A:]]) >= observed
    return (1 + count) / (1 + n_perm)


def bootstrap_ci(A, B, stat_fn, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for ``stat_fn``."""
    A, B = list(A), list(B)
    rng = np.random.default_rng(seed)
    values = [stat_fn([A[i] for i in rng.integers(len(A), size=len(A))],
                      [B[i] for i in rng.integers(len(B), size=len(B))])
              for _ in range(n_boot)]
    tail = (1 - level) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


def demo() -> None:
    """Check separated and identical Gaussian samples."""
    rng = np.random.default_rng(0)
    A = list(rng.normal(size=(20, 3)))
    B = list(rng.normal(loc=5, size=(20, 3)))
    assert permutation_pvalue(A, B, n_perm=999) < 0.01
    assert permutation_pvalue(A, A, n_perm=999) > 0.1
    print("stats demo passed")


if __name__ == "__main__":
    demo()
