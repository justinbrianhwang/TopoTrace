"""Persistent-topology helpers for TopoTrace."""

import numpy as np
from persim import PersistenceImager
from ripser import ripser


def chordal_distance_matrix(Z: np.ndarray) -> np.ndarray:
    """Return pairwise chordal distances between L2-normalized rows of Z."""
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (Z @ Z.T)))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def persistence_diagrams(D: np.ndarray, maxdim: int = 1) -> list[np.ndarray]:
    """Compute Vietoris--Rips diagrams from a precomputed distance matrix."""
    dgms = [np.asarray(dgm, dtype=np.float64) for dgm in ripser(
        D, distance_matrix=True, maxdim=maxdim
    )["dgms"]]
    finite = [dgm[np.isfinite(dgm[:, 1]), 1] for dgm in dgms if dgm.size]
    cap = max((death.max() for death in finite if death.size), default=float(np.max(D)))
    dgms[0] = dgms[0].copy()
    dgms[0][~np.isfinite(dgms[0][:, 1]), 1] = cap
    return dgms


def vectorize(dgms: list[np.ndarray], imager, dim: int = 1) -> np.ndarray:
    """Return a flattened persistence image on a shared grid."""
    dgm = dgms[dim]
    if dim == 0:
        dgm = dgm[dgm[:, 1] != dgm[:, 0]]
    image = np.zeros(imager.resolution) if dgm.size == 0 else imager.transform(dgm)
    return np.asarray(image, dtype=np.float64).ravel()


def make_imager(all_h1_dgms: list[np.ndarray]) -> "PersistenceImager":
    """Fit one roughly 20-by-20 persistence-image grid to all H1 diagrams."""
    nonempty = [np.asarray(dgm, dtype=np.float64) for dgm in all_h1_dgms if dgm.size]
    if not nonempty:
        return PersistenceImager(pixel_size=0.05)

    union = np.vstack(nonempty)
    birth_persistence = union.copy()
    birth_persistence[:, 1] -= birth_persistence[:, 0]
    spans = np.ptp(birth_persistence, axis=0)
    scale = float(spans.max())
    if scale == 0:
        scale = max(float(np.abs(birth_persistence).max()), 1.0)
    ranges = []
    for values, span in zip(birth_persistence.T, spans):
        bounds = (float(values.min()), float(values.max()))
        if span == 0:
            bounds = (bounds[0] - scale / 2.0, bounds[1] + scale / 2.0)
        ranges.append(bounds)
    imager = PersistenceImager(
        birth_range=ranges[0], pers_range=ranges[1], pixel_size=scale / 20.0
    )
    imager.fit(union)

    for name, span in zip(("birth_range", "pers_range"), spans):
        if span == 0:
            center = sum(getattr(imager, name)) / 2.0
            setattr(imager, name, (center - scale / 2.0, center + scale / 2.0))
    return imager
