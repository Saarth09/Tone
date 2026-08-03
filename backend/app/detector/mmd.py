from __future__ import annotations

import numpy as np


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gaussian RBF kernel matrix between rows of x and y."""
    x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y**2, axis=1).reshape(1, -1)
    dists = x_norm + y_norm - 2 * np.dot(x, y.T)
    dists = np.maximum(dists, 0.0)
    return np.exp(-gamma * dists)


def median_gamma(x: np.ndarray, y: np.ndarray) -> float:
    """Median heuristic for RBF bandwidth (Gretton et al.)."""
    z = np.vstack([x, y])
    if len(z) < 2:
        return 1.0
    # subsample for speed
    rng = np.random.default_rng(0)
    if len(z) > 200:
        z = z[rng.choice(len(z), size=200, replace=False)]
    z_norm = np.sum(z**2, axis=1).reshape(-1, 1)
    dists = z_norm + z_norm.T - 2 * np.dot(z, z.T)
    dists = dists[np.triu_indices_from(dists, k=1)]
    dists = dists[dists > 0]
    if len(dists) == 0:
        return 1.0
    median = float(np.median(dists))
    # gamma = 1 / (2 * sigma^2) with sigma^2 ≈ median of ||x-y||^2
    return 1.0 / max(median, 1e-6)


def mmd2(
    x: np.ndarray,
    y: np.ndarray,
    gamma: float | None = None,
    biased: bool = True,
) -> float:
    """
    Squared Maximum Mean Discrepancy between samples X and Y.

    Uses a Gaussian RBF kernel. Returns a non-negative score; higher means
    the two distributions differ more in embedding space.
    Biased estimator is preferred for small sample sizes.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays")
    if len(x) == 0 or len(y) == 0:
        return 0.0

    if gamma is None:
        gamma = median_gamma(x, y)

    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)

    n, m = len(x), len(y)
    if biased:
        return float(max(0.0, k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()))

    # Unbiased estimator
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    term_xx = k_xx.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_yy = k_yy.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_xy = k_xy.mean()
    return float(max(0.0, term_xx + term_yy - 2 * term_xy))


def normalize_mmd(raw: float, scale: float = 4.0) -> float:
    """Map raw MMD² into roughly [0, 1] for dashboard scoring."""
    return float(1.0 - np.exp(-raw * scale))
