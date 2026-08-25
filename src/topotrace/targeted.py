"""Topology-targeted MNIST deletion splits."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from ripser import ripser

from topotrace.cnn import get_embeddings, train_cnn
from topotrace.topology import chordal_distance_matrix


def score_by_cycle_support(Z: np.ndarray, maxdim: int = 1) -> np.ndarray:
    """Score samples by the persistence of H1 cocycles they support."""
    result = ripser(chordal_distance_matrix(Z), distance_matrix=True,
                    maxdim=maxdim, do_cocycles=True)
    scores = np.zeros(len(Z), dtype=np.float64)
    for interval, cocycle in zip(result["dgms"][1], result["cocycles"][1]):
        if cocycle.size:
            vertices = np.unique(cocycle[:, :2].astype(np.intp))
            scores[vertices] += interval[1] - interval[0]
    return scores


def make_targeted_split(X, y, n_forget: int = 3000, pool_size: int = 2000,
                        selector_seed: int = 9999, seed: int = 0,
                        train_fn=train_cnn, embed_fn=get_embeddings
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Select samples nearest to persistent-cycle support samples."""
    rng = np.random.default_rng(seed)
    selector = train_fn(X, y, np.arange(len(y)), seed=selector_seed,
                        epochs=3)
    pool = rng.choice(len(y), pool_size, replace=False)
    Z = embed_fn(selector, X)["penultimate"].astype(np.float64)
    support = pool[score_by_cycle_support(Z[pool]) > 0]

    unit = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    support_Z = unit[support]
    scores = np.empty(len(y), dtype=np.float64)
    for start in range(0, len(y), 1024):
        cosine = unit[start:start + 1024] @ support_Z.T
        scores[start:start + 1024] = -np.sqrt(
            np.maximum(0.0, 2.0 - 2.0 * cosine.max(axis=1)))
    scores += rng.random(len(y)) * 1e-9

    forget = np.argsort(scores)[-n_forget:]
    retain = np.flatnonzero(~np.isin(np.arange(len(y)), forget))
    return forget, retain


def make_matched_split(y, targeted_forget_idx, seed: int = 0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Randomly forget the targeted split's exact per-class counts."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(np.asarray(y)[targeted_forget_idx])
    forget = np.concatenate([
        rng.choice(np.flatnonzero(np.asarray(y) == cls), count, replace=False)
        for cls, count in enumerate(counts) if count
    ])
    retain = np.flatnonzero(~np.isin(np.arange(len(y)), forget))
    return forget, retain


if __name__ == "__main__":
    from topotrace.mnist import load_mnist

    X, y, _, _ = load_mnist()
    targeted_forget, targeted_retain = make_targeted_split(X, y)
    matched_forget, matched_retain = make_matched_split(y, targeted_forget)

    output = Path("results/m3")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "splits.npz", targeted_forget=targeted_forget,
             targeted_retain=targeted_retain, matched_forget=matched_forget,
             matched_retain=matched_retain)

    targeted_hist = np.bincount(y[targeted_forget], minlength=10)
    matched_hist = np.bincount(y[matched_forget], minlength=10)
    print("targeted:", targeted_hist)
    print("matched: ", matched_hist)
    assert len(targeted_forget) == len(matched_forget) == 3000
    assert np.array_equal(targeted_hist, matched_hist)
    assert not np.intersect1d(targeted_forget, targeted_retain).size
    assert np.array_equal(
        np.sort(np.concatenate((targeted_forget, targeted_retain))),
        np.arange(len(y)))
    print("targeted demo passed")
