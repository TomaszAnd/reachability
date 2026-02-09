#!/usr/bin/env python3
"""
Final Publication Figures for Quantum Reachability

Generates plots matching the style of:
- fig/canonical/final_summary_3panel_v7_pow2.png (multi-tau 3-panel)
- fig/geo2/geo2_main_v3.png (main GEO2 figure with all criteria)

Data sources:
- Canonical: data/production/canonical_partial_all.csv (d=8,16,32,64, tau=0.90,0.95,0.99)
- GEO2: data/production/GEO2_quick_extracted.csv (d=8,16,32, tau=0.99 only)

Output:
- fig/canonical/canonical_3panel_tau_unified.png
- fig/canonical/canonical_main_unified.png
- fig/geo2/geo2_3panel_criterion_unified.png
- fig/geo2/geo2_main_unified.png
- fig/comparison/canonical_vs_geo2_all_criteria.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

# Style configuration matching original plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# Color scheme by dimension
DIM_COLORS = {8: '#1f77b4', 16: '#ff7f0e', 32: '#2ca02c', 64: '#d62728'}
DIM_MARKERS = {8: 'o', 16: 's', 32: '^', 64: 'D'}

# Criterion colors (for GEO2-style plots)
CRIT_COLORS = {'moment': '#2E86AB', 'spectral': '#A23B72', 'krylov': '#F18F01'}

# Tau colors (for multi-tau plots)
TAU_COLORS = {0.99: '#1f77b4', 0.95: '#2ca02c', 0.90: '#d62728'}
TAU_MARKERS = {0.99: 'o', 0.95: 's', 0.90: '^'}


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
    """Fit simple exponential to Moment data."""
    try:
        mask = (P > 0.01) & (P < 0.99)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, _ = curve_fit(simple_exponential, rho[mask], P[mask],
                           p0=[0.01], bounds=([0.0001], [1.0]), maxfev=10000)
        r2 = compute_r2(P[mask], simple_exponential(rho[mask], *popt))
        return {'lambda': popt[0], 'R2': r2}
    except:
        return None


def fit_fermi(rho, P):
    """Fit Fermi-Dirac to Spectral/Krylov data."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                           p0=[np.median(rho[mask]), 0.02],
                           bounds=([0, 0.001], [1.0, 0.5]), maxfev=10000)
        r2 = compute_r2(P[mask], fermi_dirac(rho[mask], *popt))
        return {'rho_c': popt[0], 'delta': popt[1], 'R2': r2}
    except:
        return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load canonical and GEO2 data from CSV files."""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'production'

    # Canonical
    canonical_file = data_dir / 'canonical_partial_all.csv'
    if canonical_file.exists():
        df_canonical = pd.read_csv(canonical_file)
        print(f"Loaded canonical: {len(df_canonical)} rows")
    else:
        df_canonical = None
        print("Warning: canonical_partial_all.csv not found")

    # GEO2
    geo2_file = data_dir / 'GEO2_quick_extracted.csv'
    if geo2_file.exists():
        df_geo2 = pd.read_csv(geo2_file)
        print(f"Loaded GEO2: {len(df_geo2)} rows")
    else:
        df_geo2 = None
        print("Warning: GEO2_quick_extracted.csv not found")

    return df_canonical, df_geo2


# =============================================================================
# PLOT 1: CANONICAL 3-PANEL BY TAU (matching final_summary_3panel_v7 style)
# =============================================================================

def plot_canonical_3panel_tau(df, output_path):
    """
    3-panel plot: one column per tau value (0.99, 0.95, 0.90).
    Shows all 3 criteria with dimension coloring.
    """
    if df is None or len(df) == 0:
        print("No canonical data for 3-panel tau plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    taus = [0.99, 0.95, 0.90]
    criteria = [('spectral_P', 'Spectral'), ('krylov_P', 'Krylov'), ('moment_P', 'Moment')]

    for ax, tau in zip(axes, taus):
        df_tau = df[df['tau'] == tau]
        ax.set_title(f'All Criteria (τ = {tau})', fontsize=14, fontweight='bold')

        fits_for_box = {}

        for d in sorted(df_tau['d'].unique()):
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')

            rho = df_d['rho'].values

            # Plot all criteria for this dimension
            for crit_col, crit_name in criteria:
                if crit_col in df_d.columns:
                    P = df_d[crit_col].values

                    if crit_name == 'Spectral':
                        ls = '-'
                        label = f'd={int(d)}'
                    elif crit_name == 'Krylov':
                        ls = '--'
                        label = None
                    else:  # Moment
                        ls = ':'
                        label = None

                    ax.plot(rho, P, marker=marker, color=color,
                           linestyle=ls, linewidth=1.5, markersize=4,
                           label=label, alpha=0.8)

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel('P(unreachable)', fontsize=12)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Add legend for line styles
        if ax == axes[0]:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', linestyle='-', label='Spectral'),
                Line2D([0], [0], color='gray', linestyle='--', label='Krylov'),
                Line2D([0], [0], color='gray', linestyle=':', label='Moment'),
            ]
            ax.legend(handles=legend_elements, loc='center right', fontsize=8)

    fig.suptitle('Canonical Ensemble: Phase Transitions by Threshold',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path.name}")


# =============================================================================
# PLOT 2: CANONICAL MAIN (all criteria, dimension ordering)
# =============================================================================

def plot_canonical_main(df, output_path, tau=0.99):
    """
    Main canonical figure: 1×3 grid showing Spectral, Krylov, Moment.
    Each panel shows all dimensions with equation text boxes.
    """
    if df is None or len(df) == 0:
        print("No canonical data for main plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    df_tau = df[df['tau'] == tau]
    criteria = [
        ('spectral_P', 'Spectral Criterion'),
        ('krylov_P', 'Krylov Criterion'),
        ('moment_P', 'Moment Criterion'),
    ]

    for ax, (crit_col, title) in zip(axes, criteria):
        ax.set_title(f'{title} (τ={tau})', fontsize=14, fontweight='bold')

        fits_for_box = {}

        for d in sorted(df_tau['d'].unique()):
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')

            rho = df_d['rho'].values
            P = df_d[crit_col].values if crit_col in df_d.columns else None

            if P is None:
                continue

            ax.plot(rho, P, marker=marker, color=color,
                   linestyle='-', linewidth=2, markersize=5,
                   label=f'd={int(d)}', alpha=0.9)

            # Fit curves
            if 'moment' in crit_col:
                fit = fit_exponential(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                           ':', color=color, alpha=0.5, linewidth=1.5)
                    fits_for_box[int(d)] = fit
            else:
                fit = fit_fermi(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                           ':', color=color, alpha=0.5, linewidth=1.5)
                    fits_for_box[int(d)] = fit

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel('P(unreachable)', fontsize=12)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Equation text box
        if fits_for_box:
            if 'moment' in crit_col:
                lam_mean = np.mean([fits_for_box[d]['lambda'] for d in fits_for_box])
                r2_mean = np.mean([fits_for_box[d]['R2'] for d in fits_for_box])
                info = (r"$P = e^{-\rho/\lambda}$" + "\n"
                        + "─" * 18 + "\n")
                for d in sorted(fits_for_box.keys()):
                    info += f"d={d}: λ={fits_for_box[d]['lambda']:.4f}\n"
                info += "─" * 18 + f"\nR²≈{r2_mean:.2f}"
            else:
                info = (r"$P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$" + "\n"
                        + "─" * 18 + "\n")
                for d in sorted(fits_for_box.keys()):
                    f = fits_for_box[d]
                    info += f"d={d}: ρc={f['rho_c']:.3f}\n"

            ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                   family='monospace')

    fig.suptitle('Canonical Ensemble: All Three Criteria',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path.name}")


# =============================================================================
# PLOT 3: GEO2 3-PANEL BY CRITERION (matching geo2_main_v3 style)
# =============================================================================

def plot_geo2_3panel_criterion(df, output_path, tau=0.99):
    """
    3-panel plot for GEO2: one column per criterion.
    Shows all dimensions with equation text boxes.
    """
    if df is None or len(df) == 0:
        print("No GEO2 data for 3-panel criterion plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    df_tau = df[df['tau'] == tau] if 'tau' in df.columns else df

    criteria = [
        ('moment_P', 'Moment Criterion'),
        ('spectral_P', 'Spectral Criterion (τ=0.99)'),
        ('krylov_P', 'Krylov Criterion (τ=0.99)'),
    ]

    for ax, (crit_col, title) in zip(axes, criteria):
        ax.set_title(title, fontsize=14, fontweight='bold')

        fits_for_box = {}

        for d in sorted(df_tau['d'].unique()):
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')

            rho = df_d['rho'].values
            P = df_d[crit_col].values if crit_col in df_d.columns else None

            if P is None:
                continue

            ax.plot(rho, P, marker=marker, color=color,
                   linestyle='-', linewidth=2, markersize=5,
                   label=f'd={int(d)}', alpha=0.9)

            # Fit curves
            if 'moment' in crit_col:
                fit = fit_exponential(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                           ':', color=color, alpha=0.5, linewidth=1.5)
                    fits_for_box[int(d)] = fit
            else:
                fit = fit_fermi(rho, P)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho.min(), rho.max(), 100)
                    ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                           ':', color=color, alpha=0.5, linewidth=1.5)
                    fits_for_box[int(d)] = fit

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel('P(unreachable)', fontsize=12)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        # Equation text box
        if fits_for_box:
            if 'moment' in crit_col:
                lam_mean = np.mean([fits_for_box[d]['lambda'] for d in fits_for_box])
                r2_mean = np.mean([fits_for_box[d]['R2'] for d in fits_for_box])
                info = (r"$P = e^{-\rho/\lambda}$" + "\n"
                        + "─" * 20 + "\n")
                for d in sorted(fits_for_box.keys()):
                    info += f"d={d}: λ={fits_for_box[d]['lambda']:.4f}, R²={fits_for_box[d]['R2']:.2f}\n"
                info += "─" * 20 + "\n"
                info += f"Mean: λ≈{lam_mean:.4f}"
            else:
                info = (r"$P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$" + "\n"
                        + "─" * 20 + "\n")
                for d in sorted(fits_for_box.keys()):
                    f = fits_for_box[d]
                    info += f"d={d}: ρc={f['rho_c']:.3f}, Δ={f['delta']:.3f}, R²={f['R2']:.2f}\n"

            ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                   family='monospace')

    fig.suptitle('GEO2 Ensemble: Unreachability Phase Transitions',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path.name}")


# =============================================================================
# PLOT 4: GEO2 MAIN (dimension comparison)
# =============================================================================

def plot_geo2_main(df, output_path, tau=0.99):
    """
    Main GEO2 figure: All criteria with dimension-based coloring.
    """
    if df is None or len(df) == 0:
        print("No GEO2 data for main plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    df_tau = df[df['tau'] == tau] if 'tau' in df.columns else df

    criteria = [
        ('spectral_P', 'Spectral', '-'),
        ('krylov_P', 'Krylov', '--'),
        ('moment_P', 'Moment', ':'),
    ]

    for d in sorted(df_tau['d'].unique()):
        df_d = df_tau[df_tau['d'] == d].sort_values('rho')
        color = DIM_COLORS.get(int(d), 'gray')
        marker = DIM_MARKERS.get(int(d), 'o')

        rho = df_d['rho'].values

        for crit_col, crit_name, ls in criteria:
            if crit_col in df_d.columns:
                P = df_d[crit_col].values

                label = f'd={int(d)} {crit_name}' if crit_name == 'Spectral' else None
                ax.plot(rho, P, marker=marker, color=color,
                       linestyle=ls, linewidth=2, markersize=4,
                       label=label, alpha=0.8)

    ax.set_xlabel('ρ = K/d²', fontsize=14)
    ax.set_ylabel('P(unreachable)', fontsize=14)
    ax.set_title(f'GEO2 Ensemble: All Criteria (τ={tau})', fontsize=16, fontweight='bold')
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    # Add legend for dimensions and line styles
    ax.legend(loc='upper right', fontsize=10)

    # Add line style legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='Spectral'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Krylov'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Moment'),
    ]
    ax2 = ax.legend(handles=legend_elements, loc='center right', fontsize=10,
                   title='Criterion')
    ax.add_artist(ax2)

    # Key findings text box
    info = (
        "Key findings:\n"
        "─" * 25 + "\n"
        "• Higher d transitions FIRST\n"
        "  (correct dimension ordering)\n"
        "• Moment transitions before Spectral\n"
        "• Krylov uninformative (P≈0 or 1)"
    )
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path.name}")


# =============================================================================
# PLOT 5: CANONICAL VS GEO2 COMPARISON
# =============================================================================

def plot_comparison_all_criteria(df_canonical, df_geo2, output_path, tau=0.99):
    """
    2×3 grid: rows = ensemble (Canonical, GEO2), columns = criterion.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

    criteria = [
        ('spectral_P', 'Spectral'),
        ('krylov_P', 'Krylov'),
        ('moment_P', 'Moment'),
    ]

    ensembles = [
        (df_canonical, 'Canonical', 0),
        (df_geo2, 'GEO2', 1),
    ]

    for df_raw, ens_name, row in ensembles:
        if df_raw is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                   transform=axes[row, col].transAxes, fontsize=14)
            continue

        df = df_raw[df_raw['tau'] == tau] if 'tau' in df_raw.columns else df_raw

        for col, (crit_col, crit_name) in enumerate(criteria):
            ax = axes[row, col]

            if crit_col not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                continue

            for d in sorted(df['d'].unique()):
                df_d = df[df['d'] == d].sort_values('rho')
                color = DIM_COLORS.get(int(d), 'gray')
                marker = DIM_MARKERS.get(int(d), 'o')

                ax.plot(df_d['rho'], df_d[crit_col],
                       color=color, marker=marker,
                       markersize=5, linewidth=2,
                       label=f'd={int(d)}')

            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(crit_name, fontsize=14, fontweight='bold')
            if row == 1:
                ax.set_xlabel('ρ = K/d²', fontsize=12)
            if col == 0:
                ax.set_ylabel(f'{ens_name}\nP(unreachable)', fontsize=12)
                ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Three Criteria Comparison (τ = {tau})',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path.name}")


# =============================================================================
# PLOT 6: OVERLAY COMPARISON
# =============================================================================

def plot_overlay_comparison(df_canonical, df_geo2, output_path, tau=0.99):
    """
    1×3 grid: Overlay Canonical (solid) and GEO2 (dashed) for each criterion.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    criteria = [
        ('spectral_P', 'Spectral'),
        ('krylov_P', 'Krylov'),
        ('moment_P', 'Moment'),
    ]

    for ax, (crit_col, crit_name) in zip(axes, criteria):
        ax.set_title(f'{crit_name} Criterion', fontsize=14, fontweight='bold')

        # Find common dimensions
        dims_can = set(df_canonical['d'].unique()) if df_canonical is not None else set()
        dims_geo = set(df_geo2['d'].unique()) if df_geo2 is not None else set()
        common_dims = dims_can.intersection(dims_geo)
        all_dims = dims_can.union(dims_geo)

        for d in sorted(all_dims):
            color = DIM_COLORS.get(int(d), 'gray')
            marker = DIM_MARKERS.get(int(d), 'o')

            # Canonical (solid)
            if df_canonical is not None and d in dims_can:
                df_c = df_canonical[(df_canonical['d'] == d) & (df_canonical['tau'] == tau)].sort_values('rho')
                if crit_col in df_c.columns and len(df_c) > 0:
                    ax.plot(df_c['rho'], df_c[crit_col],
                           color=color, marker=marker,
                           markersize=5, linewidth=2, linestyle='-',
                           label=f'Can d={int(d)}', alpha=0.9)

            # GEO2 (dashed)
            if df_geo2 is not None and d in dims_geo:
                df_g = df_geo2[(df_geo2['d'] == d) & (df_geo2['tau'] == tau)].sort_values('rho')
                if crit_col in df_g.columns and len(df_g) > 0:
                    ax.plot(df_g['rho'], df_g[crit_col],
                           color=color, marker=marker,
                           markersize=5, linewidth=2, linestyle='--',
                           markerfacecolor='none', markeredgewidth=1.5,
                           label=f'GEO2 d={int(d)}', alpha=0.9)

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel('P(unreachable)', fontsize=12)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    fig.suptitle(f'Canonical vs GEO2 Comparison (τ = {tau})',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {output_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FINAL PUBLICATION FIGURES")
    print("Matching canonical/final_summary_3panel_v7 and geo2/geo2_main_v3 styles")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading data...")
    df_canonical, df_geo2 = load_data()

    if df_canonical is not None:
        print(f"  Canonical: dims={sorted(df_canonical['d'].unique())}, "
              f"taus={sorted(df_canonical['tau'].unique())}")
    if df_geo2 is not None:
        print(f"  GEO2: dims={sorted(df_geo2['d'].unique())}, "
              f"taus={sorted(df_geo2['tau'].unique())}")

    # Create output directories
    canonical_dir = Path(__file__).parent.parent.parent / 'fig' / 'canonical'
    geo2_dir = Path(__file__).parent.parent.parent / 'fig' / 'geo2'
    comparison_dir = Path(__file__).parent.parent.parent / 'fig' / 'comparison'

    canonical_dir.mkdir(parents=True, exist_ok=True)
    geo2_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\n[2/7] Canonical 3-panel by tau...")
    plot_canonical_3panel_tau(df_canonical, canonical_dir / 'canonical_3panel_tau_unified.png')

    print("\n[3/7] Canonical main figure...")
    plot_canonical_main(df_canonical, canonical_dir / 'canonical_main_unified.png')

    print("\n[4/7] GEO2 3-panel by criterion...")
    plot_geo2_3panel_criterion(df_geo2, geo2_dir / 'geo2_3panel_criterion_unified.png')

    print("\n[5/7] GEO2 main figure...")
    plot_geo2_main(df_geo2, geo2_dir / 'geo2_main_unified.png')

    print("\n[6/7] Comparison 2×3 grid...")
    plot_comparison_all_criteria(df_canonical, df_geo2,
                                 comparison_dir / 'canonical_vs_geo2_all_criteria.png')

    print("\n[7/7] Overlay comparison...")
    plot_overlay_comparison(df_canonical, df_geo2,
                           comparison_dir / 'canonical_vs_geo2_overlay.png')

    # Summary
    print("\n" + "=" * 70)
    print("GENERATED FILES:")
    print("=" * 70)

    outputs = [
        ('fig/canonical/canonical_3panel_tau_unified.png', 'Multi-tau 3-panel'),
        ('fig/canonical/canonical_main_unified.png', 'Main figure with equation boxes'),
        ('fig/geo2/geo2_3panel_criterion_unified.png', '3-panel by criterion'),
        ('fig/geo2/geo2_main_unified.png', 'Main figure'),
        ('fig/comparison/canonical_vs_geo2_all_criteria.png', '2×3 comparison grid'),
        ('fig/comparison/canonical_vs_geo2_overlay.png', 'Overlay comparison'),
    ]

    for path, desc in outputs:
        fpath = Path(__file__).parent.parent.parent / path
        if fpath.exists():
            size = fpath.stat().st_size / 1024
            print(f"  ✓ {path} ({size:.0f} KB) - {desc}")
        else:
            print(f"  ✗ {path} - NOT GENERATED")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
