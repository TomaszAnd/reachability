"""
Shared mathematical utilities for quantum reachability analysis.

Functions extracted from mathematics.py that are used across multiple modules.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Breakdown tolerance for Lanczos/Krylov iteration
KRYLOV_BREAKDOWN_TOL: float = 1e-14

# Dimension threshold for switching from numpy to scipy eigh.
# On Apple M1 Accelerate, scipy heevr (MRRR) is 1.3-2x faster at d>=256.
# On Intel/AMD, numpy heevd may be faster. Run scripts/benchmark_platform.py to check.
SCIPY_EIGH_THRESHOLD: int = 256


def eigendecompose(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of a Hermitian matrix H = U diag(E) U†.

    Uses dimension-adaptive backend selection:
    - d < 256: numpy.linalg.eigh (heevd, divide & conquer)
    - d >= 256: scipy.linalg.eigh default (heevr, MRRR algorithm)

    On Apple M1 Accelerate, heevr is 2x faster than heevd at d>=256.
    At smaller d, heevd is faster due to lower overhead.

    Args:
        H: Hermitian matrix (d x d numpy array)

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors are column-wise

    Raises:
        RuntimeError: If eigendecomposition fails or produces non-finite results
    """
    try:
        d = H.shape[0]
        if d >= SCIPY_EIGH_THRESHOLD:
            from scipy.linalg import eigh as scipy_eigh
            eigenvalues, eigenvectors = scipy_eigh(
                H, overwrite_a=True, check_finite=False
            )
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H)

        if np.any(~np.isfinite(eigenvalues)) or np.any(~np.isfinite(eigenvectors)):
            raise RuntimeError("Eigendecomposition produced non-finite results")

        return eigenvalues, eigenvectors

    except Exception as e:
        raise RuntimeError(f"Eigendecomposition failed: {e}") from e


def compute_binomial_sem(p: float, n: int) -> float:
    """
    Standard error of the mean for binomial proportion.

    SEM = sqrt(p(1-p)/n).

    Args:
        p: Observed proportion in [0, 1]
        n: Number of trials (positive integer)

    Returns:
        Standard error of the mean
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if p == 0.0 or p == 1.0 or n == 1:
        return 0.0
    return np.sqrt(p * (1 - p) / n)


def clip_to_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clip parameter vector to specified bounds.

    Args:
        x: Parameter vector to clip
        bounds: List of (lower, upper) bounds. If length 1, broadcast to all dims.

    Returns:
        Clipped parameter vector
    """
    x = np.asarray(x)
    if len(bounds) == 1:
        bounds = bounds * len(x)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return np.clip(x, lower, upper)


