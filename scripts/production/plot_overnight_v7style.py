#!/usr/bin/env python3
"""
Overnight Publication Plots - v7/v3 Style Reproduction

Reproduces the exact visual style of:
- fig/canonical/combined_criteria_d26_v7_exp.png  (from generate_publication_dual_versions.py)
- fig/canonical/final_summary_3panel_v7_exp.png   (same)
- fig/canonical/Kc_vs_d_v7_exp.png                (same)
- fig/geo2/geo2_main_v3.png                       (from plot_geo2_v3.py)
- fig/geo2/geo2_d64_summary_v3.png                (same)

Data sources (MERGED):
- Overnight data (spectral + moment): data/overnight/*.csv
- Corrected Krylov data (m=min(K,d)): data/krylov_corrected/*.csv

The overnight krylov_* columns used m=d (stale), so we merge with corrected data
which uses m=min(K,d) for genuine Krylov phase transitions.

Formatting modifications from v7:
1. Removed figure-level suptitles
2. Removed fit equation text boxes from plots
3. Fit curve opacity set to alpha=0.5
4. Font sizes increased by ~2pt

All fit equations and parameters are documented in PUBLICATION_PLOTS.md.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
from datetime import datetime

# =============================================================================
# STYLE - Matching v7/v3 with +2pt font increase
# =============================================================================

# v7 base: font.size=11, axes.labelsize=12, axes.titlesize=13,
#           xtick/ytick=10, legend=9, figure.titlesize=14
# +2pt each:
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# Unified dimension colors (same for canonical and GEO2)
DIM_COLORS = {
    8: '#1f77b4',    # Blue
    16: '#ff7f0e',   # Orange
    32: '#2ca02c',   # Green
    64: '#d62728',   # Red
}
CANONICAL_COLORS = DIM_COLORS  # backward compat alias

# Unified dimension markers (same for canonical and GEO2)
DIM_MARKERS = {
    8: 'o',    # circle
    16: 's',   # square
    32: '^',   # triangle up
    64: 'D',   # diamond
}

# Criterion colors (for single-dimension combined plots)
CRIT_COLORS = {
    'moment': 'blue',
    'spectral': 'red',
    'krylov': 'green',
}

# Criterion markers (for single-dimension combined plots)
CRIT_FMTS = {
    'moment': 'o',
    'spectral': 's',
    'krylov': '^',
}

# GEO2 criterion colors now unified with canonical
GEO2_CRIT_COLORS = CRIT_COLORS

# GEO2 dimension colors/markers now unified with canonical
GEO2_DIM_COLORS = DIM_COLORS
GEO2_DIM_MARKERS = DIM_MARKERS


# =============================================================================
# FIT FUNCTIONS - Identical to v7
# =============================================================================

def simple_exponential(rho, lam):
    """P = exp(-rho/lambda) for Moment criterion."""
    return np.exp(-rho / lam)


def fermi_dirac(rho, rho_c, delta):
    """P = 1/(1 + exp((rho-rho_c)/delta)) for Spectral/Krylov."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_exponential(rho, P):
    """Fit simple exponential. Returns dict or None. Identical to v7."""
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


def fit_moment_best(rho, P):
    """Fit moment data: try exponential first, fall back to Fermi-Dirac if R² < 0.5."""
    exp_fit = fit_exponential(rho, P)
    if exp_fit and exp_fit['R2'] >= 0.5:
        return exp_fit
    # Try Fermi-Dirac for sharp transitions
    fermi_fit = fit_fermi(rho, P)
    if fermi_fit:
        fermi_fit['type'] = 'fermi'
        # Prefer Fermi if it's better, or if exponential failed
        if exp_fit is None or fermi_fit['R2'] > exp_fit['R2']:
            return fermi_fit
    if exp_fit:
        return exp_fit
    return None


def fit_fermi(rho, P):
    """Fit Fermi-Dirac. Returns dict or None. Handles sharp transitions."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)

        # Better initial guess: find where P crosses 0.5
        idx = np.argmin(np.abs(P - 0.5))
        rho_c_init = rho[idx]

        # Allow very small delta for sharp transitions
        popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                            p0=[rho_c_init, 0.005],
                            bounds=([0, 0.0001], [1.0, 0.5]), maxfev=10000)
        r2 = compute_r2(P[mask], fermi_dirac(rho[mask], *popt))
        return {'rho_c': popt[0], 'delta': popt[1], 'R2': r2}
    except Exception:
        return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_merged_data(ensemble):
    """
    Load overnight spectral/moment data merged with corrected Krylov data.

    The overnight krylov_* columns used m=d (full Hilbert space dimension),
    which is stale. We drop them and merge with corrected Krylov data that
    uses m=min(K,d) for genuine phase transitions.

    Args:
        ensemble: 'canonical' or 'GEO2'

    Returns:
        DataFrame with spectral/moment from overnight + krylov from corrected
    """
    overnight_dir = Path(__file__).parent.parent.parent / 'data' / 'overnight'
    krylov_dir = Path(__file__).parent.parent.parent / 'data' / 'krylov_corrected'

    if ensemble == 'canonical':
        overnight_file = overnight_dir / 'canonical_overnight_20260128_224236.csv'
        # Use dense Krylov data (v24) if available, fall back to corrected (v23)
        krylov_file = krylov_dir / 'canonical_krylov_dense_20260205_170823.csv'
        if not krylov_file.exists():
            krylov_file = krylov_dir / 'canonical_krylov_corrected_20260204_222726.csv'
    elif ensemble == 'GEO2':
        overnight_file = overnight_dir / 'geo2_overnight_20260128_224613.csv'
        # Use dense Krylov data (v24) if available, fall back to corrected (v23)
        krylov_file = krylov_dir / 'geo2_krylov_dense_20260205_170823.csv'
        if not krylov_file.exists():
            krylov_file = krylov_dir / 'geo2_krylov_corrected_20260204_222747.csv'
    else:
        return None

    if not overnight_file.exists():
        print(f"  Warning: {overnight_file} not found")
        return None

    df_overnight = pd.read_csv(overnight_file)

    # Drop stale Krylov columns from overnight
    krylov_cols_to_drop = [c for c in df_overnight.columns if c.startswith('krylov_')]
    df_base = df_overnight.drop(columns=krylov_cols_to_drop)

    # Average duplicate (d, K, tau) rows in base data
    group_cols = ['ensemble', 'd', 'K', 'rho', 'tau']
    num_cols = [c for c in df_base.columns if c not in group_cols and c != 'n_trials']
    agg = {c: 'mean' for c in num_cols}
    agg['n_trials'] = 'first'
    df_base = (df_base.groupby(group_cols, as_index=False)
               .agg(agg)
               .sort_values(['d', 'tau', 'rho'])
               .reset_index(drop=True))

    # Ensure integer K for merge
    df_base['K'] = df_base['K'].astype(int)
    df_base['d'] = df_base['d'].astype(int)

    # Load and merge corrected Krylov data if available
    if krylov_file.exists():
        df_krylov = pd.read_csv(krylov_file)
        df_krylov['K'] = df_krylov['K'].astype(int)
        df_krylov['d'] = df_krylov['d'].astype(int)

        # Select only Krylov columns + join keys
        krylov_cols = ['d', 'K', 'm', 'krylov_P', 'krylov_mean', 'krylov_std']
        krylov_merge = df_krylov[[c for c in krylov_cols if c in df_krylov.columns]].copy()

        # Left join: keep all overnight points, add Krylov where available
        df_merged = df_base.merge(krylov_merge, on=['d', 'K'], how='left')

        n_matched = df_merged['krylov_P'].notna().sum()
        print(f"  Merged {n_matched}/{len(df_merged)} rows with corrected Krylov data")
    else:
        print(f"  Warning: {krylov_file} not found, using overnight Krylov (stale)")
        # Fallback to original loading
        df_overnight = pd.read_csv(overnight_file)
        group_cols = ['ensemble', 'd', 'K', 'rho', 'tau']
        num_cols = [c for c in df_overnight.columns if c not in group_cols and c != 'n_trials']
        agg = {c: 'mean' for c in num_cols}
        agg['n_trials'] = 'first'
        df_merged = (df_overnight.groupby(group_cols, as_index=False)
                     .agg(agg)
                     .sort_values(['d', 'tau', 'rho'])
                     .reset_index(drop=True))

    return df_merged


def load_overnight_data():
    """Load and preprocess overnight CSVs with corrected Krylov data."""
    df_canonical = load_merged_data('canonical')
    df_geo2 = load_merged_data('GEO2')
    return df_canonical, df_geo2


def load_corrected_krylov_data(ensemble):
    """Load corrected Krylov data directly (for fitting purposes)."""
    krylov_dir = Path(__file__).parent.parent.parent / 'data' / 'krylov_corrected'

    if ensemble == 'canonical':
        # Use dense Krylov data (v24) if available
        krylov_file = krylov_dir / 'canonical_krylov_dense_20260205_170823.csv'
        if not krylov_file.exists():
            krylov_file = krylov_dir / 'canonical_krylov_corrected_20260204_222726.csv'
    elif ensemble == 'GEO2':
        # Use dense Krylov data (v24) if available
        krylov_file = krylov_dir / 'geo2_krylov_dense_20260205_170823.csv'
        if not krylov_file.exists():
            krylov_file = krylov_dir / 'geo2_krylov_corrected_20260204_222747.csv'
    else:
        return None

    if not krylov_file.exists():
        return None

    df = pd.read_csv(krylov_file)
    df['K'] = df['K'].astype(int)
    df['d'] = df['d'].astype(int)
    return df


def fit_krylov_from_corrected(ensemble, d, tau=0.99):
    """Fit Krylov criterion directly from corrected data (not merged)."""
    df = load_corrected_krylov_data(ensemble)
    if df is None:
        return None

    sub = df[(df['d'] == d) & (df['tau'] == tau)].sort_values('rho')
    if len(sub) < 3:
        return None

    rho = sub['rho'].values
    P = sub['krylov_P'].values

    return fit_fermi(rho, P)


def binomial_sem(P, n):
    """Compute binomial standard error for proportion P with n trials."""
    return np.sqrt(P * (1 - P) / n)


def get_curve(df, d, tau, crit_col):
    """Extract sorted (rho, K, P, sem) for a given d and tau."""
    mask = (df['d'] == d) & (df['tau'] == tau)
    sub = df[mask].sort_values('rho').copy()
    if len(sub) == 0:
        return None, None, None, None
    rho = sub['rho'].values
    K = sub['K'].values
    P = sub[crit_col].values
    n = sub['n_trials'].values
    sem = binomial_sem(P, n)
    return rho, K, P, sem


# =============================================================================
# PLOT 1: COMBINED CRITERIA - CANONICAL (v7 combined_criteria_d26 style)
# =============================================================================

def plot_combined_criteria_canonical(df, output_path, d=64, tau=0.99):
    """
    All three criteria on single plot for one dimension.
    Style: combined_criteria_d26_v7_exp.png
    """
    # v7 figsize=(10, 6), +0 (keep identical)
    fig, ax = plt.subplots(figsize=(10, 6))

    all_fits = {}

    for crit_col, crit_name, color, fmt in [
        ('moment_P', 'Moment', CRIT_COLORS['moment'], CRIT_FMTS['moment']),
        ('spectral_P', 'Spectral', CRIT_COLORS['spectral'], CRIT_FMTS['spectral']),
        ('krylov_P', 'Krylov', CRIT_COLORS['krylov'], CRIT_FMTS['krylov']),
    ]:
        rho, K, P, sem = get_curve(df, d, tau, crit_col)
        if rho is None:
            continue

        # v7 style: errorbar with markersize=6, capsize=2; data at full opacity
        ax.errorbar(rho, P, yerr=sem, fmt=fmt, color=color, label=crit_name,
                    markersize=6, capsize=2)

        # Fit curve - v7 uses '-', linewidth=2; MODIFIED: alpha=0.5 (was 0.8)
        if 'moment' in crit_col:
            fit = fit_exponential(rho, P)
            if fit and fit['R2'] > 0.3:
                rho_fine = np.linspace(0, rho.max(), 200)
                ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit
        else:
            fit = fit_fermi(rho, P)
            if fit and fit['R2'] > 0.3:
                rho_fine = np.linspace(0, rho.max(), 200)
                ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit

    # v7 style axes - fontsize +2pt
    ax.set_xlabel(r'$\rho$ = K/d$^2$', fontsize=15)
    ax.set_ylabel('P(unreachable)', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 2: COMBINED CRITERIA - GEO2 (geo2_d64_summary_v3 style)
# =============================================================================

def plot_combined_criteria_geo2(df, output_path, d=32, tau=0.99):
    """
    All three criteria on single plot for one GEO2 dimension.
    Style: geo2_d64_summary_v3.png (single-dimension view)
    """
    # v3 figsize=(10, 7)
    fig, ax = plt.subplots(figsize=(10, 7))

    all_fits = {}

    # Load corrected Krylov data directly for better fitting
    df_krylov_corrected = load_corrected_krylov_data('GEO2')

    for crit_col, crit_name in [
        ('moment_P', 'moment'),
        ('spectral_P', 'spectral'),
        ('krylov_P', 'krylov'),
    ]:
        color = CRIT_COLORS[crit_name]
        fmt = CRIT_FMTS[crit_name]
        rho, K, P, sem = get_curve(df, d, tau, crit_col)
        if rho is None or len(rho) == 0:
            # For Krylov, try loading directly from corrected data
            if crit_name == 'krylov' and df_krylov_corrected is not None:
                sub = df_krylov_corrected[(df_krylov_corrected['d'] == d) &
                                          (df_krylov_corrected['tau'] == tau)].sort_values('rho')
                if len(sub) > 0:
                    rho = sub['rho'].values
                    K = sub['K'].values
                    P = sub['krylov_P'].values
                    n = sub['n_trials'].values
                    sem = binomial_sem(P, n)
            if rho is None or len(rho) == 0:
                continue

        # Unified style matching canonical combined criteria
        ax.errorbar(rho, P, yerr=sem, fmt=fmt, color=color,
                    markersize=6, capsize=2,
                    label=f'{crit_name.title()}')

        # Fit - unified solid style matching canonical
        if 'moment' in crit_col:
            fit = fit_moment_best(rho, P)
            if fit and fit['R2'] > 0.3:
                rho_fine = np.linspace(rho.min(), rho.max(), 100)
                if fit.get('type') == 'fermi':
                    ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5, linewidth=2)
                else:
                    ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                            '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit
        elif crit_name == 'krylov':
            # Use corrected Krylov data directly for fitting
            fit = fit_krylov_from_corrected('GEO2', d, tau)
            if fit and fit['R2'] > 0.3:
                rho_fine = np.linspace(rho.min(), rho.max(), 100)
                ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit
        else:
            fit = fit_fermi(rho, P)
            if fit and fit['R2'] > 0.3:
                rho_fine = np.linspace(rho.min(), rho.max(), 100)
                ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit

    # Axes style matching canonical
    ax.set_xlabel(r'$\rho$ = K/d$^2$', fontsize=15)
    ax.set_ylabel('P(unreachable)', fontsize=15)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 3: CANONICAL 3-PANEL (final_summary_3panel_v7 style)
# =============================================================================

def plot_canonical_3panel(df, output_path, tau=0.99):
    """
    1x3 grid: Moment | Spectral | Krylov, all dimensions.
    Style: final_summary_3panel_v7_exp.png
    """
    # v7 figsize=(16, 5)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    criteria = [
        ('moment_P', 'moment', 'Moment Criterion'),
        ('spectral_P', 'spectral', f'Spectral Criterion ($\\tau$={tau})'),
        ('krylov_P', 'krylov', f'Krylov Criterion ($\\tau$={tau})'),
    ]

    all_fits = {}
    dims = sorted(df[df['tau'] == tau]['d'].unique())

    for ax, (crit_col, crit_key, title) in zip(axes, criteria):
        ax.set_title(title, fontsize=12)
        crit_fits = {}

        for d in dims:
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')
            rho, K, P, sem = get_curve(df, d, tau, crit_col)
            if rho is None:
                continue

            # dimension-colored markers with unified scheme
            ax.errorbar(rho, P, yerr=sem, fmt=marker, color=color,
                        label=f'd={int(d)}', markersize=5, capsize=2)

            # Fit - v7 uses '-', alpha=0.8; MODIFIED: alpha=0.5
            if 'moment' in crit_col:
                fit = fit_exponential(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(0, rho.max(), 200)
                    ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit
            else:
                fit = fit_fermi(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(0, rho.max(), 200)
                    ax.plot(rho_fine,
                            fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit

        # v7 axes style, fontsize +2pt
        ax.set_xlabel(r'$\rho$ = K/d$^2$', fontsize=14)
        ax.set_ylabel('P(unreachable)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, title='Dimension')
        ax.set_ylim(-0.05, 1.05)

        # NO equation text box
        all_fits[crit_key] = crit_fits

    # NO fig.suptitle()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 4: GEO2 3-PANEL (geo2_main_v3 style)
# =============================================================================

def plot_geo2_3panel(df, output_path, tau=0.99):
    """
    1x3 grid: Moment | Spectral | Krylov, all dimensions.
    Style: geo2_main_v3.png

    For Krylov, uses corrected data directly (m=min(K,d)) rather than merged data.
    """
    # v3 figsize=(16, 5.5)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    criteria = [
        ('moment_P', 'moment', 'Moment Criterion'),
        ('spectral_P', 'spectral', f'Spectral Criterion ($\\tau$={tau})'),
        ('krylov_P', 'krylov', f'Krylov Criterion ($\\tau$={tau})'),
    ]

    all_fits = {}
    dims = sorted(df[df['tau'] == tau]['d'].unique())

    # Load corrected Krylov data directly for better data + fitting
    df_krylov_corrected = load_corrected_krylov_data('GEO2')

    for ax, (crit_col, crit_key, title) in zip(axes, criteria):
        ax.set_title(title, fontsize=12)
        crit_fits = {}

        for d in dims:
            color = GEO2_DIM_COLORS.get(int(d), 'gray')
            marker = GEO2_DIM_MARKERS.get(int(d), 'o')

            # For Krylov, use corrected data directly
            if crit_key == 'krylov' and df_krylov_corrected is not None:
                sub = df_krylov_corrected[(df_krylov_corrected['d'] == d) &
                                          (df_krylov_corrected['tau'] == tau)].sort_values('rho')
                if len(sub) > 0:
                    rho = sub['rho'].values
                    K = sub['K'].values
                    P = sub['krylov_P'].values
                    n = sub['n_trials'].values
                    sem = binomial_sem(P, n)
                else:
                    continue
            else:
                rho, K, P, sem = get_curve(df, d, tau, crit_col)
                if rho is None:
                    continue

            # unified style: filled markers, same as canonical
            ax.errorbar(rho, P, yerr=sem, fmt=marker, color=color,
                        label=f'd={int(d)}', markersize=5, capsize=2)

            # Fit - unified solid line style matching canonical
            if 'moment' in crit_col:
                fit = fit_moment_best(rho, P)
                if fit and fit['R2'] > 0.5:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    if fit.get('type') == 'fermi':
                        ax.plot(rho_fine,
                                fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                                '-', color=color, alpha=0.5)
                    else:
                        ax.plot(rho_fine,
                                simple_exponential(rho_fine, fit['lambda']),
                                '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit
            elif crit_key == 'krylov':
                # Fit from corrected Krylov data directly
                fit = fit_krylov_from_corrected('GEO2', int(d), tau)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    ax.plot(rho_fine,
                            fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit
            else:
                fit = fit_fermi(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    ax.plot(rho_fine,
                            fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit

        # axes style matching canonical
        ax.set_xlabel(r'$\rho$ = K/d$^2$', fontsize=14)
        ax.set_ylabel('P(unreachable)', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, title='Dimension')

        all_fits[crit_key] = crit_fits

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 5: K_c vs d SCALING (Kc_vs_d_v7 style)
# =============================================================================

def plot_Kc_vs_d(fits_canonical, fits_geo2, output_path):
    """
    K_c scaling with dimension for both ensembles.
    Style: Kc_vs_d_v7_exp.png
    """
    # v7 figsize=(10, 7)
    fig, ax = plt.subplots(figsize=(10, 7))

    all_Kc_params = {}

    entries = []
    if fits_canonical:
        entries.append(('Canonical', fits_canonical, '-'))
    if fits_geo2:
        entries.append(('GEO2', fits_geo2, '--'))

    for ens_name, fits, ls in entries:
        for crit_key, color, marker, label_suffix in [
            ('moment', CRIT_COLORS['moment'], CRIT_FMTS['moment'], ''),
            ('spectral', CRIT_COLORS['spectral'], CRIT_FMTS['spectral'],
             r' ($\tau$=0.99)'),
            ('krylov', CRIT_COLORS['krylov'], CRIT_FMTS['krylov'],
             r' ($\tau$=0.99)'),
        ]:
            if crit_key not in fits or len(fits[crit_key]) < 2:
                continue

            d_vals = np.array(sorted(fits[crit_key].keys()), dtype=float)
            if crit_key == 'moment':
                # K_c depends on fit type
                Kc_vals = []
                for d in d_vals:
                    f = fits[crit_key][int(d)]
                    if f.get('type') == 'fermi':
                        Kc_vals.append(f['rho_c'] * d**2)
                    else:
                        Kc_vals.append(f['lambda'] * np.log(2) * d**2)
                Kc_vals = np.array(Kc_vals)
            else:
                # K_c = d^2 * rho_c
                Kc_vals = np.array([
                    fits[crit_key][int(d)]['rho_c'] * d**2
                    for d in d_vals
                ])

            label = f'{ens_name} {crit_key.title()}{label_suffix}'

            # v7 style: 'o-'/'s-'/'^-', markersize=8, linewidth=2
            ax.plot(d_vals, Kc_vals, marker + ls, color=color,
                    label=label, markersize=8, linewidth=2)

            # Linear fit: K_c = a*d + b (v7 uses linear, not power-law)
            if len(d_vals) >= 2:
                coeffs = np.polyfit(d_vals, Kc_vals, 1)
                d_fine = np.linspace(d_vals.min() * 0.9, d_vals.max() * 1.1,
                                     100)
                Kc_fit = np.polyval(coeffs, d_fine)
                # v7: '--', alpha=0.5
                ax.plot(d_fine, Kc_fit, '--', color=color, alpha=0.5)

                r2 = compute_r2(Kc_vals, np.polyval(coeffs, d_vals))
                key = f'{ens_name}_{crit_key}'
                all_Kc_params[key] = {
                    'slope': coeffs[0], 'intercept': coeffs[1], 'R2': r2,
                    'd_vals': d_vals.tolist(), 'Kc_vals': Kc_vals.tolist(),
                }

    # v7 axes, +2pt
    ax.set_xlabel('Dimension d', fontsize=15)
    ax.set_ylabel(r'K$_c$ (Critical # Hamiltonians)', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=13)

    # NO equation text boxes

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_Kc_params


# =============================================================================
# PLOT 6: GEO2 COMBINED CRITERIA VS K (like Plot 2 but K axis)
# =============================================================================

def plot_combined_criteria_geo2_vs_K(df, output_path, d=32, tau=0.99):
    """Same as combined_criteria_geo2 but with K on x-axis."""
    fig, ax = plt.subplots(figsize=(10, 7))

    all_fits = {}

    for crit_col, crit_name in [
        ('moment_P', 'moment'),
        ('spectral_P', 'spectral'),
        ('krylov_P', 'krylov'),
    ]:
        color = CRIT_COLORS[crit_name]
        fmt = CRIT_FMTS[crit_name]
        rho, K, P, sem = get_curve(df, d, tau, crit_col)
        if rho is None:
            continue

        # Unified style matching canonical
        ax.errorbar(K, P, yerr=sem, fmt=fmt, color=color,
                    markersize=6, capsize=2,
                    label=f'{crit_name.title()}')

        # Fit on K scale - unified solid style
        if 'moment' in crit_col:
            fit = fit_moment_best(rho, P)
            if fit and fit['R2'] > 0.3:
                K_fine = np.linspace(K.min(), K.max(), 100)
                rho_fine = K_fine / d**2
                if fit.get('type') == 'fermi':
                    ax.plot(K_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5, linewidth=2)
                else:
                    ax.plot(K_fine, simple_exponential(rho_fine, fit['lambda']),
                            '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit
        else:
            fit = fit_fermi(rho, P)
            if fit and fit['R2'] > 0.3:
                K_fine = np.linspace(K.min(), K.max(), 100)
                rho_fine = K_fine / d**2
                ax.plot(K_fine,
                        fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit

    ax.set_xlabel('K (Number of Hamiltonians)', fontsize=15)
    ax.set_ylabel('P(unreachable)', fontsize=15)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 7: GEO2 3-PANEL VS K (like Plot 4 but K axis)
# =============================================================================

def plot_geo2_3panel_vs_K(df, output_path, tau=0.99):
    """Same as geo2_3panel but with K on x-axis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    criteria = [
        ('moment_P', 'moment', 'Moment Criterion'),
        ('spectral_P', 'spectral', f'Spectral Criterion ($\\tau$={tau})'),
        ('krylov_P', 'krylov', f'Krylov Criterion ($\\tau$={tau})'),
    ]

    all_fits = {}
    dims = sorted(df[df['tau'] == tau]['d'].unique())

    for ax, (crit_col, crit_key, title) in zip(axes, criteria):
        ax.set_title(title, fontsize=12)
        crit_fits = {}

        for d in dims:
            color = GEO2_DIM_COLORS.get(int(d), 'gray')
            marker = GEO2_DIM_MARKERS.get(int(d), 'o')
            rho, K, P, sem = get_curve(df, d, tau, crit_col)
            if rho is None:
                continue

            # unified style: filled markers, same as canonical
            ax.errorbar(K, P, yerr=sem, fmt=marker, color=color,
                        label=f'd={int(d)}', markersize=5, capsize=2)

            if 'moment' in crit_col:
                fit = fit_moment_best(rho, P)
                if fit and fit['R2'] > 0.5:
                    K_fine = np.linspace(K.min(), K.max(), 100)
                    rho_fine = K_fine / d**2
                    if fit.get('type') == 'fermi':
                        ax.plot(K_fine,
                                fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                                '-', color=color, alpha=0.5)
                    else:
                        ax.plot(K_fine,
                                simple_exponential(rho_fine, fit['lambda']),
                                '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit
            else:
                fit = fit_fermi(rho, P)
                if fit and fit['R2'] > 0.3:
                    K_fine = np.linspace(K.min(), K.max(), 100)
                    rho_fine = K_fine / d**2
                    ax.plot(K_fine,
                            fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit

        ax.set_xlabel('K (Number of Hamiltonians)', fontsize=14)
        ax.set_ylabel('P(unreachable)', fontsize=14)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, title='Dimension')

        all_fits[crit_key] = crit_fits

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 8: CANONICAL COMBINED CRITERIA VS K
# =============================================================================

def plot_combined_criteria_canonical_vs_K(df, output_path, d=64, tau=0.99):
    """Same as combined_criteria_canonical but with K on x-axis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_fits = {}

    for crit_col, crit_name, color, fmt in [
        ('moment_P', 'Moment', CRIT_COLORS['moment'], CRIT_FMTS['moment']),
        ('spectral_P', 'Spectral', CRIT_COLORS['spectral'], CRIT_FMTS['spectral']),
        ('krylov_P', 'Krylov', CRIT_COLORS['krylov'], CRIT_FMTS['krylov']),
    ]:
        rho, K, P, sem = get_curve(df, d, tau, crit_col)
        if rho is None:
            continue

        ax.errorbar(K, P, yerr=sem, fmt=fmt, color=color, label=crit_name,
                    markersize=6, capsize=2)

        if 'moment' in crit_col:
            fit = fit_exponential(rho, P)
            if fit and fit['R2'] > 0.3:
                K_fine = np.linspace(K.min(), K.max(), 200)
                rho_fine = K_fine / d**2
                ax.plot(K_fine, simple_exponential(rho_fine, fit['lambda']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit
        else:
            fit = fit_fermi(rho, P)
            if fit and fit['R2'] > 0.3:
                K_fine = np.linspace(K.min(), K.max(), 200)
                rho_fine = K_fine / d**2
                ax.plot(K_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                        '-', color=color, alpha=0.5, linewidth=2)
                all_fits[crit_name] = fit

    ax.set_xlabel('K (Number of Hamiltonians)', fontsize=15)
    ax.set_ylabel('P(unreachable)', fontsize=15)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# PLOT 9: CANONICAL 3-PANEL VS K
# =============================================================================

def plot_canonical_3panel_vs_K(df, output_path, tau=0.99):
    """
    1x3 grid: Moment | Spectral | Krylov, all dimensions, K on x-axis.
    Mirror of plot_geo2_3panel_vs_K but for canonical data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    criteria = [
        ('moment_P', 'moment', 'Moment Criterion'),
        ('spectral_P', 'spectral', f'Spectral Criterion ($\\tau$={tau})'),
        ('krylov_P', 'krylov', f'Krylov Criterion ($\\tau$={tau})'),
    ]

    all_fits = {}
    dims = sorted(df[df['tau'] == tau]['d'].unique())

    for ax, (crit_col, crit_key, title) in zip(axes, criteria):
        ax.set_title(title, fontsize=12)
        crit_fits = {}

        for d in dims:
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')
            rho, K, P, sem = get_curve(df, d, tau, crit_col)
            if rho is None:
                continue

            ax.errorbar(K, P, yerr=sem, fmt=marker, color=color,
                        label=f'd={int(d)}', markersize=5, capsize=2)

            if 'moment' in crit_col:
                fit = fit_exponential(rho, P)
                if fit and fit['R2'] > 0.3:
                    K_fine = np.linspace(K.min(), K.max(), 200)
                    rho_fine = K_fine / d**2
                    ax.plot(K_fine, simple_exponential(rho_fine, fit['lambda']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit
            else:
                fit = fit_fermi(rho, P)
                if fit and fit['R2'] > 0.3:
                    K_fine = np.linspace(K.min(), K.max(), 200)
                    rho_fine = K_fine / d**2
                    ax.plot(K_fine,
                            fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                            '-', color=color, alpha=0.5)
                    crit_fits[int(d)] = fit

        ax.set_xlabel('K (Number of Hamiltonians)', fontsize=14)
        ax.set_ylabel('P(unreachable)', fontsize=14)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11, title='Dimension')

        all_fits[crit_key] = crit_fits

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return all_fits


# =============================================================================
# DOCUMENTATION
# =============================================================================

def generate_documentation(
    fits_canonical_3p, fits_geo2_3p,
    fits_canonical_combined, fits_geo2_combined,
    Kc_params, output_path
):
    """Generate PUBLICATION_PLOTS.md with all fit parameters."""

    lines = [
        "# Publication Plot Documentation\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "",
        "## Data Provenance\n",
        "",
        "### IMPORTANT: Merged Data Sources",
        "",
        "**Spectral and Moment data** come from overnight runs (valid).",
        "**Krylov data** comes from CORRECTED runs with `m=min(K,d)` (valid).",
        "",
        "The overnight Krylov columns used `m=d` (full Hilbert space), which caused:",
        "- GEO2 Krylov to be trivially P=0 everywhere (Krylov subspace = full R^d)",
        "- Canonical Krylov to have different (less meaningful) transitions",
        "",
        "The corrected Krylov experiments use `m=min(K,d)`, creating genuine phase transitions.\n",
        "### Canonical Ensemble",
        "- **Spectral/Moment**: `data/overnight/canonical_overnight_20260128_224236.csv`",
        "  - 327 data points, 142.9 hours runtime",
        "- **Krylov (corrected)**: `data/krylov_corrected/canonical_krylov_corrected_20260204_222726.csv`",
        "  - 108 data points, ~10 hours runtime, m=min(K,d)",
        "- **Dimensions**: d in {8, 16, 32, 64}",
        "- **Threshold**: tau = 0.99 (plots)",
        "- **Trials per point**: 200\n",
        "### GEO2 Ensemble",
        "- **Spectral/Moment**: `data/overnight/geo2_overnight_20260128_224613.csv`",
        "  - 101 data points, 47.2 hours runtime",
        "- **Krylov (corrected)**: `data/krylov_corrected/geo2_krylov_corrected_20260204_222747.csv`",
        "  - 96 data points, ~5.4 hours runtime, m=min(K,d)",
        "- **Dimensions**: d in {8, 16, 32, 64}",
        "- **Threshold**: tau = 0.99",
        "- **Trials per point**: 200 (100 for d=64)\n",
        "## Plot Generation\n",
        "- **Script**: `scripts/production/plot_overnight_v7style.py`",
        "- **Based on**:",
        "  - `scripts/canonical/generate_publication_dual_versions.py` (canonical v7 style)",
        "  - `scripts/geo2/plot_geo2_v3.py` (GEO2 v3 style)",
        f"- **Date generated**: {datetime.now().strftime('%Y-%m-%d')}",
        "- **Style reference**: Unified style documented in CLAUDE.md\n",
        "---\n",
    ]

    # Plot descriptions
    lines.extend([
        "## Plot Descriptions\n",
        "### Plot 1: `combined_criteria_canonical.png`",
        "All three criteria (Spectral, Krylov, Moment) on a single plot for d=64, tau=0.99.",
        "Style: `combined_criteria_d26_v7_exp.png`. X-axis: rho = K/d^2. Y-axis: P(unreachable).\n",
        "### Plot 2: `combined_criteria_geo2.png`",
        "Same as Plot 1 for GEO2 ensemble, d=64, tau=0.99.",
        "Style: `geo2_d64_summary_v3.png`. Note: Krylov shows P~0 everywhere (correct physics).\n",
        "### Plot 3: `canonical_3panel.png`",
        "3-panel figure (Moment | Spectral | Krylov). Each panel shows all dimensions (d=8,16,32,64).",
        "Style: `final_summary_3panel_v7_exp.png`. tau=0.99.\n",
        "### Plot 4: `geo2_3panel.png`",
        "Same 3-panel layout for GEO2 ensemble, tau=0.99.",
        "Style: `geo2_main_v3.png`.\n",
        "### Plot 5: `Kc_vs_d.png`",
        "Critical K_c vs dimension d for both ensembles, all fitted criteria.",
        "Style: `Kc_vs_d_v7_exp.png`. Linear scale.\n",
        "### Plot 6: `geo2_combined_criteria_vs_K.png`",
        "Same as Plot 2 but with K (absolute count) on x-axis instead of rho.\n",
        "### Plot 7: `geo2_3panel_vs_K.png`",
        "Same as Plot 4 but with K on x-axis.\n",
        "### Plot 8: `canonical_combined_criteria_vs_K_d{d}_tau099.png`",
        "All three criteria on single plot with K on x-axis, for each canonical dimension.\n",
        "### Plot 9: `canonical_3panel_vs_K.png`",
        "Same as Plot 3 but with K (absolute count) on x-axis instead of rho.",
        "X-axis range extends to K_max from data.\n",
        "---\n",
    ])

    # Fit functions
    lines.extend([
        "## Fit Function Specifications\n",
        "### Fermi-Dirac (Spectral and Krylov Criteria)\n",
        "$$P(\\rho) = \\frac{1}{1 + \\exp\\!\\left(\\frac{\\rho - \\rho_c}{\\Delta}\\right)}$$\n",
        "- $\\rho_c$: transition density (P = 0.5 at $\\rho = \\rho_c$)",
        "- $\\Delta$: transition width",
        "- $K_c = \\rho_c \\times d^2$\n",
        "### Simple Exponential (Moment Criterion)\n",
        "$$P(\\rho) = \\exp\\!\\left(-\\frac{\\rho}{\\lambda}\\right)$$\n",
        "- $\\lambda$: decay scale",
        "- $\\rho_{1/2} = \\lambda \\ln 2$",
        "- $K_c = d^2 \\lambda \\ln 2$\n",
        "### Linear K_c Scaling\n",
        "$$K_c = a \\cdot d + b$$\n",
        "---\n",
    ])

    # Fitted parameter tables - Canonical 3-panel
    if fits_canonical_3p:
        lines.append("## Canonical Ensemble Fit Parameters (tau = 0.99)\n")
        for crit_key, crit_label in [
            ('spectral', 'Spectral'), ('krylov', 'Krylov'), ('moment', 'Moment')
        ]:
            if crit_key not in fits_canonical_3p:
                continue
            crit_fits = fits_canonical_3p[crit_key]
            if not crit_fits:
                continue

            lines.append(f"### {crit_label} Criterion\n")
            if crit_key == 'moment':
                lines.append("| d | Fit Type | Parameters | K_c | R^2 |")
                lines.append("|---|----------|------------|-----|-----|")
                for d in sorted(crit_fits.keys()):
                    f = crit_fits[d]
                    if f.get('type') == 'fermi':
                        Kc = f['rho_c'] * d**2
                        params = f"rho_c={f['rho_c']:.5f}, Delta={f['delta']:.5f}"
                        lines.append(f"| {d} | Fermi-Dirac | {params} | "
                                     f"{Kc:.1f} | {f['R2']:.3f} |")
                    else:
                        rho_half = f['lambda'] * np.log(2)
                        Kc = rho_half * d**2
                        params = f"lambda={f['lambda']:.5f}, rho_1/2={rho_half:.5f}"
                        lines.append(f"| {d} | Exponential | {params} | "
                                     f"{Kc:.1f} | {f['R2']:.3f} |")
            else:
                lines.append("| d | rho_c | Delta | K_c | R^2 |")
                lines.append("|---|-------|-------|-----|-----|")
                for d in sorted(crit_fits.keys()):
                    f = crit_fits[d]
                    Kc = f['rho_c'] * d**2
                    lines.append(f"| {d} | {f['rho_c']:.5f} | "
                                 f"{f['delta']:.5f} | {Kc:.1f} | "
                                 f"{f['R2']:.3f} |")
            lines.append("")
        lines.append("---\n")

    # Fitted parameter tables - GEO2 3-panel
    if fits_geo2_3p:
        lines.append("## GEO2 Ensemble Fit Parameters (tau = 0.99)\n")
        for crit_key, crit_label in [
            ('spectral', 'Spectral'), ('krylov', 'Krylov'), ('moment', 'Moment')
        ]:
            if crit_key not in fits_geo2_3p:
                continue
            crit_fits = fits_geo2_3p[crit_key]
            if not crit_fits:
                continue

            lines.append(f"### {crit_label} Criterion\n")
            if crit_key == 'moment':
                lines.append("| d | Fit Type | Parameters | K_c | R^2 |")
                lines.append("|---|----------|------------|-----|-----|")
                for d in sorted(crit_fits.keys()):
                    f = crit_fits[d]
                    if f.get('type') == 'fermi':
                        Kc = f['rho_c'] * d**2
                        params = f"rho_c={f['rho_c']:.5f}, Delta={f['delta']:.5f}"
                        lines.append(f"| {d} | Fermi-Dirac | {params} | "
                                     f"{Kc:.1f} | {f['R2']:.3f} |")
                    else:
                        rho_half = f['lambda'] * np.log(2)
                        Kc = rho_half * d**2
                        params = f"lambda={f['lambda']:.5f}, rho_1/2={rho_half:.5f}"
                        lines.append(f"| {d} | Exponential | {params} | "
                                     f"{Kc:.1f} | {f['R2']:.3f} |")
            else:
                lines.append("| d | rho_c | Delta | K_c | R^2 |")
                lines.append("|---|-------|-------|-----|-----|")
                for d in sorted(crit_fits.keys()):
                    f = crit_fits[d]
                    Kc = f['rho_c'] * d**2
                    lines.append(f"| {d} | {f['rho_c']:.5f} | "
                                 f"{f['delta']:.5f} | {Kc:.1f} | "
                                 f"{f['R2']:.3f} |")
            lines.append("")
        lines.append("---\n")

    # K_c scaling table
    if Kc_params:
        lines.extend([
            "## K_c Linear Scaling\n",
            "$$K_c = a \\cdot d + b$$\n",
            "| Ensemble | Criterion | a (slope) | b (intercept) | R^2 |",
            "|----------|-----------|-----------|---------------|-----|",
        ])
        for key in sorted(Kc_params.keys()):
            p = Kc_params[key]
            ens, crit = key.split('_', 1)
            lines.append(f"| {ens} | {crit.title()} | "
                         f"{p['slope']:.2f} | {p['intercept']:.1f} | "
                         f"{p['R2']:.3f} |")
        lines.append("")
        lines.append("---\n")

    # Notes on Krylov correction
    lines.extend([
        "## Notes on Krylov m-Parameter Correction\n",
        "**Background**: The original overnight experiments used `m=d` (full Hilbert space ",
        "dimension) for the Krylov subspace. This caused:",
        "",
        "1. **GEO2 Krylov was trivially P=0**: Dense GEO2 Hamiltonians never cause early ",
        "   Arnoldi breakdown, so the Krylov basis spans all of R^d regardless of K.",
        "2. **Canonical Krylov was also affected**: Though Arnoldi breakdown occurs earlier ",
        "   for sparse canonical operators, using m=d still inflates reachability scores.",
        "",
        "**Correction**: The corrected experiments use `m=min(K,d)`, so the Krylov rank ",
        "grows with K (number of Hamiltonians), creating genuine phase transitions.",
        "",
        "**Result**: GEO2 Krylov now shows real transitions (P drops from 1.0 to 0.0 as ",
        "K increases), and is included in K_c vs d scaling.\n",
        "---\n",
    ])

    # Formatting changes
    lines.extend([
        "## Formatting Changes from v7/v3\n",
        "1. **Removed figure-level suptitles** (`fig.suptitle()` calls removed)",
        "2. **Removed fit equation text boxes** from plot legends "
        "(equations documented here)",
        "3. **Fit curves at 50% opacity** (`alpha=0.5` on all fit lines)",
        "4. **Font sizes increased ~2pt** from v7/v3 originals for "
        "single-column LaTeX readability",
        "",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("OVERNIGHT PUBLICATION PLOTS - UNIFIED STYLE")
    print("=" * 70)

    output_dir = Path(__file__).parent.parent.parent / 'fig' / 'overnight'
    output_dir.mkdir(parents=True, exist_ok=True)

    pub_dir = Path(__file__).parent.parent.parent / 'fig' / 'publication'
    pub_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/11] Loading data...")
    df_canonical, df_geo2 = load_overnight_data()
    if df_canonical is not None:
        print(f"  Canonical: {len(df_canonical)} rows, "
              f"dims={sorted(df_canonical['d'].unique())}, "
              f"taus={sorted(df_canonical['tau'].unique())}")
    if df_geo2 is not None:
        print(f"  GEO2: {len(df_geo2)} rows, "
              f"dims={sorted(df_geo2['d'].unique())}, "
              f"taus={sorted(df_geo2['tau'].unique())}")

    # Plot 1: Combined criteria - Canonical (all dimensions)
    print("\n[2/11] Combined criteria - Canonical (all d)...")
    fits_can_combined = {}
    if df_canonical is not None:
        can_dims = sorted(df_canonical[df_canonical['tau'] == 0.99]['d'].unique())
        for d in can_dims:
            fname = f'combined_criteria_canonical_d{int(d)}_tau099.png'
            fits = plot_combined_criteria_canonical(
                df_canonical, output_dir / fname, d=int(d), tau=0.99)
            print(f"  d={int(d)}: {list(fits.keys())}")
            if int(d) == 64:
                fits_can_combined = fits

    # Plot 2: Combined criteria - GEO2 (all dimensions)
    print("\n[3/11] Combined criteria - GEO2 (all d)...")
    fits_geo2_combined = {}
    if df_geo2 is not None:
        geo_dims = sorted(df_geo2[df_geo2['tau'] == 0.99]['d'].unique())
        for d in geo_dims:
            fname = f'combined_criteria_geo2_d{int(d)}_tau099.png'
            fits = plot_combined_criteria_geo2(
                df_geo2, output_dir / fname, d=int(d), tau=0.99)
            print(f"  d={int(d)}: {list(fits.keys())}")
            if int(d) == 32:
                fits_geo2_combined = fits

    # Plot 3
    print("\n[4/11] Canonical 3-panel...")
    fits_can_3p = {}
    if df_canonical is not None:
        fits_can_3p = plot_canonical_3panel(
            df_canonical, output_dir / 'canonical_3panel.png', tau=0.99)
        for k, v in fits_can_3p.items():
            print(f"  {k}: fitted d={sorted(v.keys())}")

    # Plot 4
    print("\n[5/11] GEO2 3-panel...")
    fits_geo2_3p = {}
    if df_geo2 is not None:
        fits_geo2_3p = plot_geo2_3panel(
            df_geo2, output_dir / 'geo2_3panel.png', tau=0.99)
        for k, v in fits_geo2_3p.items():
            print(f"  {k}: fitted d={sorted(v.keys())}")

    # Plot 5
    print("\n[6/11] K_c vs d scaling...")
    Kc_params = plot_Kc_vs_d(fits_can_3p, fits_geo2_3p,
                              output_dir / 'Kc_vs_d.png')
    for k, v in Kc_params.items():
        print(f"  {k}: K_c = {v['slope']:.2f}d + {v['intercept']:.1f} "
              f"(R2={v['R2']:.3f})")

    # Plot 6: GEO2 combined criteria vs K (all dimensions)
    print("\n[7/11] GEO2 combined criteria vs K (all d)...")
    if df_geo2 is not None:
        geo_dims = sorted(df_geo2[df_geo2['tau'] == 0.99]['d'].unique())
        for d in geo_dims:
            fname = f'geo2_combined_criteria_vs_K_d{int(d)}_tau099.png'
            plot_combined_criteria_geo2_vs_K(
                df_geo2, output_dir / fname, d=int(d), tau=0.99)
            print(f"  d={int(d)}")

    # Plot 7
    print("\n[8/11] GEO2 3-panel vs K...")
    if df_geo2 is not None:
        plot_geo2_3panel_vs_K(
            df_geo2, output_dir / 'geo2_3panel_vs_K.png', tau=0.99)

    # Plot 8: Canonical combined criteria vs K (all dimensions)
    print("\n[9/11] Canonical combined criteria vs K (all d)...")
    if df_canonical is not None:
        can_dims = sorted(df_canonical[df_canonical['tau'] == 0.99]['d'].unique())
        for d in can_dims:
            fname = f'canonical_combined_criteria_vs_K_d{int(d)}_tau099.png'
            plot_combined_criteria_canonical_vs_K(
                df_canonical, output_dir / fname, d=int(d), tau=0.99)
            print(f"  d={int(d)}")

    # Plot 9: Canonical 3-panel vs K
    print("\n[10/11] Canonical 3-panel vs K...")
    if df_canonical is not None:
        plot_canonical_3panel_vs_K(
            df_canonical, output_dir / 'canonical_3panel_vs_K.png', tau=0.99)

    # Documentation
    print("\n[11/11] Generating documentation...")
    generate_documentation(
        fits_can_3p, fits_geo2_3p,
        fits_can_combined, fits_geo2_combined,
        Kc_params, output_dir / 'PUBLICATION_PLOTS.md')

    # Copy all plots to fig/publication/
    import shutil
    print("\nCopying to fig/publication/...")
    for src in sorted(output_dir.glob('*.png')):
        shutil.copy2(src, pub_dir / src.name)
        print(f"  Copied {src.name}")

    # Summary
    print("\n" + "=" * 70)
    print("GENERATED FILES:")
    print("=" * 70)
    for f in sorted(output_dir.glob('*')):
        if f.is_file() and f.name in [
            'combined_criteria_canonical_d64_tau099.png',
            'combined_criteria_geo2_d32_tau099.png',
            'canonical_3panel.png',
            'geo2_3panel.png',
            'Kc_vs_d.png',
            'geo2_combined_criteria_vs_K_d32_tau099.png',
            'geo2_3panel_vs_K.png',
            'canonical_combined_criteria_vs_K_d64_tau099.png',
            'canonical_3panel_vs_K.png',
            'PUBLICATION_PLOTS.md',
        ]:
            size = f.stat().st_size / 1024
            print(f"  {f.name:45s} ({size:7.0f} KB)")
    print("=" * 70)


if __name__ == '__main__':
    main()
