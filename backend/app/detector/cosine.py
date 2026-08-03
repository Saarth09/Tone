from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def mean_top_k_similarity(similarities: list[float]) -> float:
    if not similarities:
        return 1.0
    return float(np.mean(similarities))


def drift_from_similarity(similarity: float) -> float:
    """Convert similarity in [0,1] to a drift score (higher = more drift)."""
    return float(max(0.0, min(1.0, 1.0 - similarity)))
