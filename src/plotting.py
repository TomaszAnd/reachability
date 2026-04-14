"""
Publication-quality plotting utilities for reachability analysis.

Provides fit functions and color schemes used by the production plotting scripts.
The actual plot generation lives in scripts/production/plot_overnight_v7style.py;
this module provides the shared building blocks.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

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
    'moment': '#2ca02c',    # Green
    'spectral': '#1f77b4',  # Blue
    'krylov': '#ff7f0e',    # Orange
}

CRIT_MARKERS = {
    'moment': 'o',
    'spectral': 's',
    'krylov': '^',
}

CRIT_LABELS = {
    'moment': 'Moment',
    'spectral': 'Spectral',
    'krylov': 'Krylov',
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


def fermi_dirac_floor(rho: np.ndarray, rho_c: float, delta: float,
                      P_floor: float) -> np.ndarray:
    """P = P_floor + (1 - P_floor) / (1 + exp((rho - rho_c) / delta)).

    Accounts for non-zero tail from optimizer limitations in Spectral.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return P_floor + (1.0 - P_floor) / (1.0 + np.exp(x))


def fit_fermi_floor(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit Fermi-Dirac with floor parameter. Returns dict with rho_c, delta, P_floor, R2."""
    try:
        mask = (P > 0.01) & (P < 0.98)
        if np.sum(mask) < 4:
            mask = np.ones(len(P), dtype=bool)
        idx = np.argmin(np.abs(P - 0.5))
        rho_c_init = rho[idx]
        # Estimate floor from tail (last 20% of data)
        n_tail = max(1, len(P) // 5)
        P_floor_init = np.clip(np.mean(np.sort(P)[:n_tail]), 0, 0.2)
        popt, _ = curve_fit(
            fermi_dirac_floor, rho[mask], P[mask],
            p0=[rho_c_init, 0.005, P_floor_init],
            bounds=([0, 0.0001, 0], [1.0, 0.5, 0.3]),
            maxfev=10000,
        )
        r2 = compute_r2(P[mask], fermi_dirac_floor(rho[mask], *popt))
        return {'rho_c': popt[0], 'delta': popt[1], 'P_floor': popt[2],
                'R2': r2, 'type': 'fermi_floor'}
    except Exception:
        return None


def power_law(d: np.ndarray, a: float, alpha: float) -> np.ndarray:
    """rho_c = a * d^(-alpha) for scaling fits."""
    return a * np.power(d, -alpha)


def fit_power_law(d_vals: np.ndarray, rho_c_vals: np.ndarray,
                  weights: Optional[np.ndarray] = None) -> Optional[Dict]:
    """Fit rho_c = a * d^(-alpha) via log-log linear regression.

    Returns dict with a, alpha, R2.
    """
    try:
        log_d = np.log(d_vals)
        log_rho = np.log(rho_c_vals)
        if weights is not None:
            w = weights / weights.sum()
            coeffs = np.polyfit(log_d, log_rho, 1, w=w)
        else:
            coeffs = np.polyfit(log_d, log_rho, 1)
        alpha = -coeffs[0]
        a = np.exp(coeffs[1])
        y_pred = power_law(d_vals, a, alpha)
        r2 = compute_r2(rho_c_vals, y_pred)
        return {'a': a, 'alpha': alpha, 'R2': r2}
    except Exception:
        return None


def log_logistic(rho: np.ndarray, rho_c: float, beta: float) -> np.ndarray:
    """Log-logistic survival function for Spectral/Krylov phase transitions.

    P(0) = 1 exactly. P(rho_c) = 0.5. P(inf) = 0.
    rho_c = midpoint, beta = steepness (higher = sharper transition).
    """
    out = np.ones_like(rho, dtype=float)
    mask = rho > 0
    ratio = (rho_c / rho[mask]) ** beta
    out[mask] = ratio / (1.0 + ratio)
    return out


def fit_log_logistic(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit log-logistic to P(rho) data. Returns dict with rho_c, beta, R2."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        idx = np.argmin(np.abs(P - 0.5))
        rho_c_init = rho[idx] if rho[idx] > 0 else 0.001
        popt, _ = curve_fit(log_logistic, rho[mask], P[mask],
                            p0=[rho_c_init, 5.0],
                            bounds=([1e-6, 0.5], [1.0, 50.0]), maxfev=10000)
        r2 = compute_r2(P[mask], log_logistic(rho[mask], *popt))
        return {'rho_c': popt[0], 'beta': popt[1], 'R2': r2, 'type': 'log_logistic'}
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


def fit_exponential_full(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit simple exponential on ALL data (no mask). For step-like transitions.

    Uses all points including P=0, which constrains the decay constant even
    with few non-zero points. R2 computed on full dataset.
    """
    try:
        # Initial guess from first non-zero point
        nonzero = P > 0.01
        if nonzero.any():
            lam_init = rho[nonzero].mean()
        else:
            lam_init = rho.mean()
        lam_init = max(lam_init, 0.0001)
        popt, _ = curve_fit(simple_exponential, rho, P,
                            p0=[lam_init], bounds=([1e-8], [1.0]),
                            maxfev=10000)
        r2 = compute_r2(P, simple_exponential(rho, *popt))
        return {'lambda': popt[0], 'R2': r2, 'type': 'exponential'}
    except Exception:
        return None
