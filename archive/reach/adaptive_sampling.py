#!/usr/bin/env python3
"""
Adaptive rho sampling for phase transition detection.

Strategy:
1. Start with coarse uniform sampling
2. Identify transition region (where 0.1 < P < 0.9)
3. Add dense sampling in transition region
4. Optionally iterate for finer resolution
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AdaptiveSamplingConfig:
    """Configuration for adaptive sampling."""
    # Initial coarse sampling
    rho_min: float = 0.005
    rho_max: float = 0.20
    n_coarse: int = 10

    # Transition detection
    p_low: float = 0.1   # P below this is "pre-transition"
    p_high: float = 0.9  # P above this is "post-transition"

    # Dense sampling
    n_dense: int = 15
    margin: float = 0.02  # Extra margin around detected transition

    # Refinement
    n_refine_iterations: int = 1


def detect_transition_region(
    rho_values: np.ndarray,
    p_values: np.ndarray,
    p_low: float = 0.1,
    p_high: float = 0.9,
) -> Tuple[float, float]:
    """
    Detect the transition region from P(rho) data.

    Returns:
        (rho_start, rho_end) of transition region
    """
    # Sort by rho
    order = np.argsort(rho_values)
    rho_sorted = rho_values[order]
    p_sorted = p_values[order]

    rho_start = rho_sorted[0]
    rho_end = rho_sorted[-1]

    # Transition starts when P drops below p_high
    for i, (rho, p) in enumerate(zip(rho_sorted, p_sorted)):
        if p < p_high and rho_start == rho_sorted[0]:
            rho_start = rho_sorted[max(0, i - 1)]
        if p < p_low:
            rho_end = rho
            break

    return rho_start, rho_end


def estimate_rho_c(
    rho_values: np.ndarray,
    p_values: np.ndarray,
) -> float:
    """
    Estimate critical density rho_c where P(rho_c) ~ 0.5.

    Uses linear interpolation between nearest points.
    """
    order = np.argsort(rho_values)
    rho_sorted = rho_values[order]
    p_sorted = p_values[order]

    for i in range(len(p_sorted) - 1):
        if p_sorted[i] >= 0.5 >= p_sorted[i + 1]:
            t = (0.5 - p_sorted[i]) / (p_sorted[i + 1] - p_sorted[i])
            return rho_sorted[i] + t * (rho_sorted[i + 1] - rho_sorted[i])

    # Fallback: midpoint
    return (rho_sorted[0] + rho_sorted[-1]) / 2


def generate_adaptive_rho_points(
    coarse_rho: np.ndarray,
    coarse_P: np.ndarray,
    config: Optional[AdaptiveSamplingConfig] = None,
) -> np.ndarray:
    """
    Generate additional rho points based on coarse sampling results.

    Args:
        coarse_rho: Initial rho values
        coarse_P: Corresponding P(unreachable) values
        config: Sampling configuration

    Returns:
        Combined array of rho values (coarse + dense)
    """
    if config is None:
        config = AdaptiveSamplingConfig()

    # Detect transition
    rho_start, rho_end = detect_transition_region(
        coarse_rho, coarse_P, config.p_low, config.p_high
    )

    # Add margin
    rho_start = max(config.rho_min, rho_start - config.margin)
    rho_end = min(config.rho_max, rho_end + config.margin)

    # Generate dense points in transition
    dense_rho = np.linspace(rho_start, rho_end, config.n_dense)

    # Combine and deduplicate
    all_rho = np.unique(np.concatenate([coarse_rho, dense_rho]))

    return np.sort(all_rho)


def get_dimension_specific_config(
    ensemble: str,
    d: int,
) -> AdaptiveSamplingConfig:
    """
    Get adaptive sampling config based on ensemble and dimension.

    Uses transition estimates from previous runs.
    """
    # Transition estimates (rho_c at tau=0.99, spectral criterion)
    transition_estimates = {
        'canonical': {8: 0.15, 16: 0.10, 32: 0.06, 64: 0.03},
        'GEO2': {8: 0.08, 16: 0.035, 32: 0.02, 64: 0.015},
    }

    rho_c = transition_estimates.get(ensemble, {}).get(d, 0.05)

    return AdaptiveSamplingConfig(
        rho_min=max(0.005, rho_c - 0.08),
        rho_max=min(0.30, rho_c + 0.10),
        n_coarse=8,
        n_dense=20,
        margin=0.015,
    )
