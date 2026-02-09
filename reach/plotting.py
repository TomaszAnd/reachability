"""
Publication-quality plotting utilities for reachability analysis.

Provides fit functions and color schemes used by the production plotting scripts.
The actual plot generation lives in scripts/production/plot_overnight_v7style.py;
this module provides the shared building blocks.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


# ============================================================================
# Unified Color Scheme
# ============================================================================

DIM_COLORS = {
    8: '#1f77b4',    # Blue
    16: '#ff7f0e',   # Orange
    32: '#2ca02c',   # Green
    64: '#d62728',   # Red
}

DIM_MARKERS = {
    8: 'o',    # circle
    16: 's',   # square
    32: '^',   # triangle up
    64: 'D',   # diamond
}

CRIT_COLORS = {
    'moment': '#1f77b4',    # Blue
    'spectral': '#d62728',  # Red
    'krylov': '#2ca02c',    # Green
}

CRIT_MARKERS = {
    'moment': 'o',
    'spectral': 's',
    'krylov': '^',
}


# ============================================================================
# Fit Functions
# ============================================================================

def fermi_dirac(rho: np.ndarray, rho_c: float, delta: float) -> np.ndarray:
    """P = 1/(1 + exp((rho - rho_c) / delta)) for Spectral/Krylov."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))


def simple_exponential(rho: np.ndarray, lam: float) -> np.ndarray:
    """P = exp(-rho / lambda) for Moment criterion."""
    return np.exp(-rho / lam)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def binomial_sem(P: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Binomial standard error for proportion P with n trials."""
    return np.sqrt(P * (1 - P) / n)


def fit_fermi(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit Fermi-Dirac to P(rho) data. Returns dict with rho_c, delta, R2."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        idx = np.argmin(np.abs(P - 0.5))
        rho_c_init = rho[idx]
        popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                            p0=[rho_c_init, 0.005],
                            bounds=([0, 0.0001], [1.0, 0.5]), maxfev=10000)
        r2 = compute_r2(P[mask], fermi_dirac(rho[mask], *popt))
        return {'rho_c': popt[0], 'delta': popt[1], 'R2': r2}
    except Exception:
        return None


def fit_exponential(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit simple exponential to P(rho) data. Returns dict with lambda, R2."""
    try:
        mask = (P > 0.01) & (P < 0.99)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, _ = curve_fit(simple_exponential, rho[mask], P[mask],
                            p0=[0.01], bounds=([0.0001], [1.0]), maxfev=10000)
        r2 = compute_r2(P[mask], simple_exponential(rho[mask], *popt))
        return {'lambda': popt[0], 'R2': r2, 'type': 'exponential'}
    except Exception:
        return None
