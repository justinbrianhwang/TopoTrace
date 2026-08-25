"""Oracle-calibrated topology metrics."""

from itertools import combinations

import numpy as np


def trr_metrics(v_O: list[np.ndarray], v_R: list[np.ndarray],
                v_U: list[np.ndarray], eps: float = 1e-12) -> dict:
    """Compute residual, progress, and artifact metrics from topology vectors."""
    rr = [np.linalg.norm(a - b) for a, b in combinations(v_R, 2)]
    D_RR = float(np.median(rr)) if rr else 0.0
    D_OR = float(np.median([np.linalg.norm(a - b) for a in v_O for b in v_R]))
    D_UR = float(np.median([np.linalg.norm(a - b) for a in v_U for b in v_R]))

    O, R, U = (np.mean(np.stack(v), axis=0) for v in (v_O, v_R, v_U))
    direction = R - O
    change = U - O
    direction_norm = np.linalg.norm(direction)
    alpha = float(np.dot(change, direction) / (direction_norm ** 2 + eps))
    eta = float(np.linalg.norm(change - alpha * direction) / (direction_norm + eps))

    return {
        "D_RR": D_RR,
        "D_OR": D_OR,
        "D_UR": D_UR,
        "TRR": (D_UR - D_RR) / (D_OR - D_RR + eps),
        "alpha": alpha,
        "eta": eta,
        "I_topo": D_OR - D_RR,
    }


def demo() -> None:
    """Check the exact-retrain and no-op TRR anchors."""
    original = [np.array([0.0, 0.0])] * 2
    retrained = [np.array([1.0, 0.0])] * 2
    assert abs(trr_metrics(original, retrained, retrained)["TRR"]) < 1e-9
    assert abs(trr_metrics(original, retrained, original)["TRR"] - 1.0) < 1e-9
    print("metrics demo passed")


if __name__ == "__main__":
    demo()
