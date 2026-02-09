#!/usr/bin/env python3
"""
Publication Plots from Overnight Experiments

Generates publication-quality plots from completed overnight data:
- Canonical ensemble: d=8,16,32,64; τ=0.99,0.95,0.90; 200 trials
- GEO2 ensemble: d=8,16,32,64; τ=0.99; 200 trials (100 for d=64)

Based on v7 publication figures (generate_publication_dual_versions.py)
and production plotting (plot_final_figures.py), adapted for overnight CSVs.

Formatting: No figure-level titles, no fit equation legends on plots,
fit curves at alpha=0.5, data points at alpha=1.0, LaTeX single-column sizing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
from datetime import datetime

# =============================================================================
# PUBLICATION STYLE
# =============================================================================

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# Single-column (3.4") and full-width (7.0") for two-column LaTeX
COL_WIDTH = 3.4
FULL_WIDTH = 7.0

# Dimension colors and markers
DIM_COLORS = {8: '#1f77b4', 16: '#ff7f0e', 32: '#2ca02c', 64: '#d62728'}
DIM_MARKERS = {8: 'o', 16: 's', 32: '^', 64: 'D'}

# Criterion colors
CRIT_COLORS = {'moment': '#2E86AB', 'spectral': '#A23B72', 'krylov': '#F18F01'}
CRIT_MARKERS = {'moment': 'o', 'spectral': 's', 'krylov': '^'}

# Tau styles
TAU_COLORS = {0.99: '#1f77b4', 0.95: '#2ca02c', 0.90: '#d62728'}


# =============================================================================
# FIT FUNCTIONS
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
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_exponential(rho, P):
    """Fit simple exponential to Moment data. Returns dict or None."""
    try:
        mask = (P > 0.01) & (P < 0.99)
        if np.sum(mask) < 3:
            mask = P > 0.01  # Relax upper bound
        if np.sum(mask) < 2:
            return None
        popt, _ = curve_fit(simple_exponential, rho[mask], P[mask],
                            p0=[0.01], bounds=([1e-5], [1.0]), maxfev=10000)
        r2 = compute_r2(P[mask], simple_exponential(rho[mask], *popt))
        Kc = popt[0] * np.log(2)  # K_c/d² = λ ln(2), but we track rho_c
        return {'lambda': popt[0], 'rho_c_half': popt[0] * np.log(2), 'R2': r2}
    except Exception:
        return None


def fit_fermi(rho, P):
    """Fit Fermi-Dirac to Spectral/Krylov data. Returns dict or None."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 2:
            return None
        popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                            p0=[np.median(rho[mask]), 0.02],
                            bounds=([0, 1e-6], [1.0, 0.5]), maxfev=10000)
        r2 = compute_r2(P[mask], fermi_dirac(rho[mask], *popt))
        return {'rho_c': popt[0], 'delta': popt[1], 'R2': r2}
    except Exception:
        return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_overnight_data():
    """Load canonical and GEO2 overnight CSVs."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'overnight'

    canonical_file = data_dir / 'canonical_overnight_20260128_224236.csv'
    geo2_file = data_dir / 'geo2_overnight_20260128_224613.csv'

    df_canonical = None
    df_geo2 = None

    if canonical_file.exists():
        df_canonical = pd.read_csv(canonical_file)
        # Average duplicate (d, K, tau) entries
        group_cols = ['ensemble', 'd', 'K', 'rho', 'tau']
        agg_cols = {c: 'mean' for c in df_canonical.columns if c not in group_cols}
        agg_cols['n_trials'] = 'first'
        df_canonical = df_canonical.groupby(group_cols, as_index=False).agg(agg_cols)
        df_canonical = df_canonical.sort_values(['d', 'tau', 'rho']).reset_index(drop=True)
        print(f"Loaded canonical: {len(df_canonical)} rows, "
              f"dims={sorted(df_canonical['d'].unique())}, "
              f"taus={sorted(df_canonical['tau'].unique())}")
    else:
        print(f"Warning: {canonical_file} not found")

    if geo2_file.exists():
        df_geo2 = pd.read_csv(geo2_file)
        # Average duplicate (d, K, tau) entries
        group_cols = ['ensemble', 'd', 'K', 'rho', 'tau']
        agg_cols = {c: 'mean' for c in df_geo2.columns if c not in group_cols}
        agg_cols['n_trials'] = 'first'
        df_geo2 = df_geo2.groupby(group_cols, as_index=False).agg(agg_cols)
        df_geo2 = df_geo2.sort_values(['d', 'tau', 'rho']).reset_index(drop=True)
        print(f"Loaded GEO2: {len(df_geo2)} rows, "
              f"dims={sorted(df_geo2['d'].unique())}, "
              f"taus={sorted(df_geo2['tau'].unique())}")
    else:
        print(f"Warning: {geo2_file} not found")

    return df_canonical, df_geo2


# =============================================================================
# FITTING HELPERS
# =============================================================================

def fit_all_criteria(df, tau=0.99):
    """
    Fit all criteria for all dimensions at given tau.
    Returns dict: {criterion: {d: fit_result}}.
    """
    if df is None:
        return {}

    df_tau = df[df['tau'] == tau] if 'tau' in df.columns else df

    results = {}
    for crit_col, crit_name, fit_func in [
        ('moment_P', 'moment', fit_exponential),
        ('spectral_P', 'spectral', fit_fermi),
        ('krylov_P', 'krylov', fit_fermi),
    ]:
        results[crit_name] = {}
        if crit_col not in df_tau.columns:
            continue
        for d in sorted(df_tau['d'].unique()):
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')
            rho = df_d['rho'].values
            P = df_d[crit_col].values

            # Skip if all zeros or all ones
            if np.all(P < 0.02) or np.all(P > 0.98):
                continue

            fit = fit_func(rho, P)
            if fit is not None and fit['R2'] > 0.3:
                results[crit_name][int(d)] = fit

    return results


# =============================================================================
# PLOT 1: 3-PANEL DECAY CURVES (per ensemble)
# =============================================================================

def plot_3panel_decay(df, fits, ensemble_name, output_path, tau=0.99):
    """
    1×3 grid: Spectral | Krylov | Moment.
    Each panel shows all dimensions with fit curves.
    """
    if df is None:
        return

    df_tau = df[df['tau'] == tau]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH / 3 + 0.3))

    criteria = [
        ('spectral_P', 'spectral', 'Spectral'),
        ('krylov_P', 'krylov', 'Krylov'),
        ('moment_P', 'moment', 'Moment'),
    ]

    for ax, (crit_col, crit_key, crit_label) in zip(axes, criteria):
        ax.set_title(crit_label, fontsize=14)

        for d in sorted(df_tau['d'].unique()):
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')
            rho = df_d['rho'].values
            P = df_d[crit_col].values

            # Data points at full opacity
            ax.plot(rho, P, marker=marker, color=color, linestyle='none',
                    markersize=4, label=f'd={int(d)}', alpha=1.0, zorder=3)

            # Fit curve at alpha=0.5
            d_int = int(d)
            if crit_key in fits and d_int in fits[crit_key]:
                rho_fine = np.linspace(0, rho.max() * 1.05, 200)
                f = fits[crit_key][d_int]
                if crit_key == 'moment':
                    P_fit = simple_exponential(rho_fine, f['lambda'])
                else:
                    P_fit = fermi_dirac(rho_fine, f['rho_c'], f['delta'])
                ax.plot(rho_fine, P_fit, '-', color=color, alpha=0.5,
                        linewidth=1.5, zorder=2)

        ax.set_xlabel(r'$\rho = K/d^2$', fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, None)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper right', fontsize=9, title='Dimension',
                  title_fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# PLOT 2: COMBINED CRITERIA (per ensemble, representative dimension)
# =============================================================================

def plot_combined_criteria(df, fits, ensemble_name, output_path, tau=0.99,
                           target_dims=None):
    """
    Show all criteria for a given ensemble in a 2×2 grid of dimensions.
    """
    if df is None:
        return

    df_tau = df[df['tau'] == tau]
    dims = sorted(df_tau['d'].unique())

    if target_dims is not None:
        dims = [d for d in dims if d in target_dims]

    n_dims = len(dims)
    if n_dims == 0:
        return

    ncols = min(n_dims, 2)
    nrows = (n_dims + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(FULL_WIDTH, FULL_WIDTH / 2 * nrows / ncols + 0.3),
                              squeeze=False)

    criteria = [
        ('moment_P', 'moment', 'Moment', CRIT_COLORS['moment'], CRIT_MARKERS['moment']),
        ('spectral_P', 'spectral', 'Spectral', CRIT_COLORS['spectral'], CRIT_MARKERS['spectral']),
        ('krylov_P', 'krylov', 'Krylov', CRIT_COLORS['krylov'], CRIT_MARKERS['krylov']),
    ]

    for idx, d in enumerate(dims):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        d_int = int(d)

        df_d = df_tau[df_tau['d'] == d].sort_values('rho')
        rho = df_d['rho'].values

        for crit_col, crit_key, crit_label, color, marker in criteria:
            if crit_col not in df_d.columns:
                continue
            P = df_d[crit_col].values

            ax.plot(rho, P, marker=marker, color=color, linestyle='none',
                    markersize=5, label=crit_label, alpha=1.0, zorder=3)

            # Fit curve
            if crit_key in fits and d_int in fits[crit_key]:
                rho_fine = np.linspace(0, rho.max() * 1.05, 200)
                f = fits[crit_key][d_int]
                if crit_key == 'moment':
                    P_fit = simple_exponential(rho_fine, f['lambda'])
                else:
                    P_fit = fermi_dirac(rho_fine, f['rho_c'], f['delta'])
                ax.plot(rho_fine, P_fit, '-', color=color, alpha=0.5,
                        linewidth=1.5, zorder=2)

        ax.set_title(f'd = {d_int}', fontsize=14)
        ax.set_xlabel(r'$\rho = K/d^2$', fontsize=14)
        if col == 0:
            ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, None)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper right')

    # Hide unused axes
    for idx in range(n_dims, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# PLOT 3: K_c VS d SCALING
# =============================================================================

def plot_Kc_vs_d(fits_canonical, fits_geo2, output_path):
    """
    K_c vs d for both ensembles. Power-law fits.
    """
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.85))

    all_fit_params = {}

    for ensemble_name, fits, ls, marker_offset in [
        ('Canonical', fits_canonical, '-', 0),
        ('GEO2', fits_geo2, '--', 0),
    ]:
        if not fits:
            continue

        for crit_key, color, marker in [
            ('spectral', CRIT_COLORS['spectral'], CRIT_MARKERS['spectral']),
            ('krylov', CRIT_COLORS['krylov'], CRIT_MARKERS['krylov']),
            ('moment', CRIT_COLORS['moment'], CRIT_MARKERS['moment']),
        ]:
            if crit_key not in fits or len(fits[crit_key]) < 2:
                continue

            d_vals = np.array(sorted(fits[crit_key].keys()), dtype=float)
            if crit_key == 'moment':
                Kc_vals = np.array([fits[crit_key][int(d)]['rho_c_half'] * d**2
                                     for d in d_vals])
            else:
                Kc_vals = np.array([fits[crit_key][int(d)]['rho_c'] * d**2
                                     for d in d_vals])

            label = f'{ensemble_name} {crit_key.capitalize()}'
            mfc = color if ls == '-' else 'none'
            ax.plot(d_vals, Kc_vals, marker=marker, color=color,
                    linestyle=ls, markersize=7, linewidth=1.5,
                    label=label, alpha=1.0,
                    markerfacecolor=mfc, markeredgecolor=color, zorder=3)

            # Power-law fit in log space
            if len(d_vals) >= 2:
                log_d = np.log(d_vals)
                log_Kc = np.log(Kc_vals)
                coeffs = np.polyfit(log_d, log_Kc, 1)
                alpha_exp = coeffs[0]
                A = np.exp(coeffs[1])
                d_fine = np.linspace(d_vals.min() * 0.8, d_vals.max() * 1.1, 100)
                Kc_fit = A * d_fine ** alpha_exp
                ax.plot(d_fine, Kc_fit, ls, color=color, alpha=0.3,
                        linewidth=1, zorder=1)

                r2 = compute_r2(log_Kc, np.polyval(coeffs, log_d))
                all_fit_params[f'{ensemble_name}_{crit_key}'] = {
                    'A': A, 'alpha': alpha_exp, 'R2': r2
                }

    ax.set_xlabel(r'Dimension $d$', fontsize=14)
    ax.set_ylabel(r'$K_c$', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=8, loc='upper left', ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")

    return all_fit_params


# =============================================================================
# PLOT 4: CANONICAL VS GEO2 OVERLAY
# =============================================================================

def plot_overlay_comparison(df_canonical, df_geo2, fits_canonical, fits_geo2,
                            output_path, tau=0.99):
    """
    1×3 grid overlaying Canonical (solid) and GEO2 (dashed) per criterion.
    """
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, FULL_WIDTH / 3 + 0.3))

    criteria = [
        ('spectral_P', 'spectral', 'Spectral'),
        ('krylov_P', 'krylov', 'Krylov'),
        ('moment_P', 'moment', 'Moment'),
    ]

    # Get common dimensions
    dims_can = sorted(df_canonical['d'].unique()) if df_canonical is not None else []
    dims_geo = sorted(df_geo2['d'].unique()) if df_geo2 is not None else []
    all_dims = sorted(set(list(dims_can) + list(dims_geo)))

    for ax, (crit_col, crit_key, crit_label) in zip(axes, criteria):
        ax.set_title(crit_label, fontsize=14)

        for d in all_dims:
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')

            # Canonical (solid markers)
            if df_canonical is not None:
                df_c = df_canonical[(df_canonical['d'] == d) &
                                    (df_canonical['tau'] == tau)].sort_values('rho')
                if len(df_c) > 0 and crit_col in df_c.columns:
                    ax.plot(df_c['rho'], df_c[crit_col],
                            marker=marker, color=color, linestyle='none',
                            markersize=4, alpha=0.8,
                            label=f'Can d={int(d)}', zorder=3)

            # GEO2 (open markers)
            if df_geo2 is not None:
                df_g = df_geo2[(df_geo2['d'] == d) &
                               (df_geo2['tau'] == tau)].sort_values('rho')
                if len(df_g) > 0 and crit_col in df_g.columns:
                    ax.plot(df_g['rho'], df_g[crit_col],
                            marker=marker, color=color, linestyle='none',
                            markersize=4, alpha=0.8,
                            markerfacecolor='none', markeredgewidth=1.5,
                            label=f'GEO2 d={int(d)}', zorder=3)

        ax.set_xlabel(r'$\rho = K/d^2$', fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, None)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=7, loc='upper right', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# PLOT 5: MULTI-TAU SPECTRAL COMPARISON (canonical only)
# =============================================================================

def plot_multi_tau_spectral(df, output_path):
    """
    Show spectral P(unreachable) vs ρ for all tau values and dimensions.
    1×3 grid: one panel per tau.
    """
    if df is None:
        return

    taus = sorted(df['tau'].unique())
    n_tau = len(taus)
    if n_tau == 0:
        return

    fig, axes = plt.subplots(1, n_tau,
                              figsize=(FULL_WIDTH, FULL_WIDTH / 3 + 0.3),
                              squeeze=False)
    axes = axes[0]

    for ax, tau in zip(axes, taus):
        df_tau = df[df['tau'] == tau]
        ax.set_title(fr'$\tau = {tau}$', fontsize=14)

        for d in sorted(df_tau['d'].unique()):
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')
            rho = df_d['rho'].values
            P = df_d['spectral_P'].values

            ax.plot(rho, P, marker=marker, color=color, linestyle='none',
                    markersize=4, label=f'd={int(d)}', alpha=1.0, zorder=3)

            # Fit
            fit = fit_fermi(rho, P)
            if fit is not None and fit['R2'] > 0.3:
                rho_fine = np.linspace(0, rho.max() * 1.05, 200)
                P_fit = fermi_dirac(rho_fine, fit['rho_c'], fit['delta'])
                ax.plot(rho_fine, P_fit, '-', color=color, alpha=0.5,
                        linewidth=1.5, zorder=2)

        ax.set_xlabel(r'$\rho = K/d^2$', fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, None)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=9, loc='upper right', title='Dimension',
                  title_fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# PLOT 6: 2×3 COMPARISON GRID
# =============================================================================

def plot_comparison_grid(df_canonical, df_geo2, output_path, tau=0.99):
    """
    2×3 grid: rows = ensemble (Canonical, GEO2), cols = criterion.
    """
    fig, axes = plt.subplots(2, 3,
                              figsize=(FULL_WIDTH, FULL_WIDTH * 0.6),
                              sharey=True)

    criteria = [
        ('spectral_P', 'Spectral'),
        ('krylov_P', 'Krylov'),
        ('moment_P', 'Moment'),
    ]

    for row, (df_raw, ens_name) in enumerate([
        (df_canonical, 'Canonical'),
        (df_geo2, 'GEO2'),
    ]):
        if df_raw is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center',
                                    va='center',
                                    transform=axes[row, col].transAxes,
                                    fontsize=12)
            continue

        df = df_raw[df_raw['tau'] == tau] if 'tau' in df_raw.columns else df_raw

        for col, (crit_col, crit_name) in enumerate(criteria):
            ax = axes[row, col]

            if crit_col not in df.columns:
                continue

            for d in sorted(df['d'].unique()):
                df_d = df[df['d'] == d].sort_values('rho')
                color = DIM_COLORS.get(int(d), 'gray')
                marker = DIM_MARKERS.get(int(d), 'o')

                ax.plot(df_d['rho'], df_d[crit_col],
                        color=color, marker=marker,
                        markersize=3, linewidth=0, linestyle='none',
                        label=f'd={int(d)}', alpha=0.9)

            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)

            if row == 0:
                ax.set_title(crit_name, fontsize=14)
            if row == 1:
                ax.set_xlabel(r'$\rho = K/d^2$', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'{ens_name}\n' + r'$P(\mathrm{unreachable})$',
                              fontsize=11)
                ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


# =============================================================================
# DOCUMENTATION
# =============================================================================

def generate_documentation(fits_canonical, fits_geo2, Kc_fit_params, output_path):
    """Generate PUBLICATION_PLOTS.md with all fit parameters."""

    lines = []
    lines.append("# Publication Plot Documentation\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Data Provenance
    lines.append("## Data Provenance\n")
    lines.append("### Canonical Ensemble")
    lines.append("- **Data file**: `data/overnight/canonical_overnight_20260128_224236.csv`")
    lines.append("- **Points**: 327")
    lines.append("- **Runtime**: 142.9 hours")
    lines.append("- **Parameters**: d in {8, 16, 32, 64}, tau in {0.90, 0.95, 0.99}, n_trials = 200")
    lines.append("- **Generated by**: `scripts/production/overnight_canonical.py`\n")

    lines.append("### GEO2 Ensemble")
    lines.append("- **Data file**: `data/overnight/geo2_overnight_20260128_224613.csv`")
    lines.append("- **Points**: 101")
    lines.append("- **Runtime**: 47.2 hours")
    lines.append("- **Parameters**: d in {8, 16, 32, 64}, tau = 0.99, n_trials = 200 (100 for d=64)")
    lines.append("- **Generated by**: `scripts/production/overnight_geo2.py`\n")

    lines.append("## Plot Generation")
    lines.append("- **Script**: `scripts/production/plot_overnight_publication.py`")
    lines.append("- **Based on**: `scripts/canonical/generate_publication_dual_versions.py` (v7)")
    lines.append(f"- **Date generated**: {datetime.now().strftime('%Y-%m-%d')}\n")

    lines.append("---\n")

    # Fit Functions
    lines.append("## Fit Functions\n")
    lines.append("### Moment Criterion: Simple Exponential\n")
    lines.append("$$P(\\rho) = e^{-\\rho/\\lambda}$$\n")
    lines.append("- At $\\rho = 0$: $P = 1$ (fully reachable)")
    lines.append("- $K_c = d^2 \\cdot \\lambda \\cdot \\ln(2)$ (at $P = 0.5$)\n")

    lines.append("### Spectral/Krylov Criteria: Fermi-Dirac\n")
    lines.append("$$P(\\rho) = \\frac{1}{1 + \\exp\\left(\\frac{\\rho - \\rho_c}{\\Delta}\\right)}$$\n")
    lines.append("- $K_c = d^2 \\cdot \\rho_c$ (at $P = 0.5$)\n")

    lines.append("---\n")

    # Fit Parameters Tables
    for ensemble_name, fits in [("Canonical", fits_canonical), ("GEO2", fits_geo2)]:
        if not fits:
            continue

        lines.append(f"## {ensemble_name} Ensemble Fit Parameters (tau = 0.99)\n")

        # Spectral
        if 'spectral' in fits and fits['spectral']:
            lines.append("### Spectral Criterion\n")
            lines.append("| Dimension | rho_c | Delta | K_c | R^2 |")
            lines.append("|-----------|-------|-------|-----|-----|")
            for d in sorted(fits['spectral'].keys()):
                f = fits['spectral'][d]
                Kc = f['rho_c'] * d ** 2
                lines.append(f"| d={d} | {f['rho_c']:.5f} | {f['delta']:.5f} | "
                             f"{Kc:.1f} | {f['R2']:.3f} |")
            lines.append("")

        # Krylov
        if 'krylov' in fits and fits['krylov']:
            lines.append("### Krylov Criterion\n")
            lines.append("| Dimension | rho_c | Delta | K_c | R^2 |")
            lines.append("|-----------|-------|-------|-----|-----|")
            for d in sorted(fits['krylov'].keys()):
                f = fits['krylov'][d]
                Kc = f['rho_c'] * d ** 2
                lines.append(f"| d={d} | {f['rho_c']:.5f} | {f['delta']:.5f} | "
                             f"{Kc:.1f} | {f['R2']:.3f} |")
            lines.append("")

        # Moment
        if 'moment' in fits and fits['moment']:
            lines.append("### Moment Criterion\n")
            lines.append("| Dimension | lambda | rho_c(half) | K_c | R^2 |")
            lines.append("|-----------|--------|-------------|-----|-----|")
            for d in sorted(fits['moment'].keys()):
                f = fits['moment'][d]
                Kc = f['rho_c_half'] * d ** 2
                lines.append(f"| d={d} | {f['lambda']:.5f} | "
                             f"{f['rho_c_half']:.5f} | {Kc:.1f} | "
                             f"{f['R2']:.3f} |")
            lines.append("")

        lines.append("---\n")

    # K_c scaling
    if Kc_fit_params:
        lines.append("## K_c Scaling (Power Law Fits)\n")
        lines.append("$$K_c = A \\cdot d^{\\alpha}$$\n")
        lines.append("| Ensemble | Criterion | A | alpha | R^2 |")
        lines.append("|----------|-----------|---|-------|-----|")
        for key, params in sorted(Kc_fit_params.items()):
            ens, crit = key.split('_', 1)
            lines.append(f"| {ens} | {crit.capitalize()} | "
                         f"{params['A']:.4f} | {params['alpha']:.3f} | "
                         f"{params['R2']:.3f} |")
        lines.append("")

    lines.append("---\n")

    # Formatting Notes
    lines.append("## Formatting Notes\n")
    lines.append("- **No figure-level titles** (suptitle removed)")
    lines.append("- **No fit equations in legends** (equations documented here only)")
    lines.append("- **Fit curves at 50% opacity** (alpha=0.5)")
    lines.append("- **Font sizes**: axis labels 14pt, tick labels 12pt, legend 11pt")
    lines.append("- **DPI**: 300")
    lines.append("- **Figure width**: 3.4\" (single column) or 7.0\" (full width)")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Saved: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("OVERNIGHT PUBLICATION FIGURES")
    print("=" * 70)

    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'fig' / 'overnight'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/8] Loading data...")
    df_canonical, df_geo2 = load_overnight_data()

    # Fit all criteria
    print("\n[2/8] Fitting curves...")
    fits_canonical = fit_all_criteria(df_canonical, tau=0.99)
    fits_geo2 = fit_all_criteria(df_geo2, tau=0.99)

    for name, fits in [("Canonical", fits_canonical), ("GEO2", fits_geo2)]:
        if fits:
            for crit in ['spectral', 'krylov', 'moment']:
                if crit in fits:
                    dims = sorted(fits[crit].keys())
                    print(f"  {name} {crit}: fitted d={dims}")

    # Plot 1: Canonical 3-panel
    print("\n[3/8] Canonical 3-panel decay...")
    plot_3panel_decay(df_canonical, fits_canonical, 'Canonical',
                      output_dir / 'canonical_3panel.png', tau=0.99)

    # Plot 2: GEO2 3-panel
    print("\n[4/8] GEO2 3-panel decay...")
    plot_3panel_decay(df_geo2, fits_geo2, 'GEO2',
                      output_dir / 'geo2_3panel.png', tau=0.99)

    # Plot 3: Combined criteria canonical
    print("\n[5/8] Combined criteria (Canonical)...")
    plot_combined_criteria(df_canonical, fits_canonical, 'Canonical',
                           output_dir / 'combined_criteria_canonical.png',
                           tau=0.99)

    # Plot 4: Combined criteria GEO2
    print("\n[5/8] Combined criteria (GEO2)...")
    plot_combined_criteria(df_geo2, fits_geo2, 'GEO2',
                           output_dir / 'combined_criteria_geo2.png',
                           tau=0.99)

    # Plot 5: K_c vs d
    print("\n[6/8] K_c vs d scaling...")
    Kc_fit_params = plot_Kc_vs_d(fits_canonical, fits_geo2,
                                  output_dir / 'Kc_vs_d.png')

    # Plot 6: Canonical vs GEO2 overlay
    print("\n[7/8] Canonical vs GEO2 overlay...")
    plot_overlay_comparison(df_canonical, df_geo2, fits_canonical, fits_geo2,
                            output_dir / 'canonical_vs_geo2_overlay.png',
                            tau=0.99)

    # Plot 7: Multi-tau spectral (canonical only)
    print("\n[7/8] Multi-tau spectral comparison...")
    plot_multi_tau_spectral(df_canonical,
                            output_dir / 'canonical_spectral_multi_tau.png')

    # Plot 8: 2×3 comparison grid
    print("\n[8/8] Comparison grid...")
    plot_comparison_grid(df_canonical, df_geo2,
                          output_dir / 'comparison_grid.png', tau=0.99)

    # Documentation
    print("\n[Doc] Generating documentation...")
    generate_documentation(fits_canonical, fits_geo2, Kc_fit_params,
                           output_dir / 'PUBLICATION_PLOTS.md')

    # Summary
    print("\n" + "=" * 70)
    print("GENERATED FILES:")
    print("=" * 70)

    for f in sorted(output_dir.glob('*')):
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"  {f.name:45s} ({size:7.0f} KB)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
