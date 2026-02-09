#!/usr/bin/env python3
"""
Overnight Checkpoint Analysis and Plotting

Analyzes checkpoint data from overnight production runs and generates
publication-quality plots. Works with partial data (experiments in progress).

Generates:
- fig/overnight/canonical_spectral_transitions.png
- fig/overnight/geo2_spectral_transitions.png
- fig/overnight/canonical_vs_geo2_comparison.png
- fig/overnight/all_criteria_grid.png
- fig/overnight/P_vs_K.png
- fig/overnight/OVERNIGHT_ANALYSIS.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings
from datetime import datetime

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'gray',
})

# Color scheme
DIM_COLORS = {8: '#1f77b4', 16: '#ff7f0e', 32: '#2ca02c', 64: '#d62728'}
DIM_MARKERS = {8: 'o', 16: 's', 32: '^', 64: 'D'}
CRIT_COLORS = {'moment': '#2E86AB', 'spectral': '#A23B72', 'krylov': '#F18F01'}


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


def fit_fermi(rho, P, verbose=False):
    """Fit Fermi-Dirac to data."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)

        # Initial guess: rho_c at P=0.5, small delta
        mid_idx = np.argmin(np.abs(P - 0.5))
        rho_c_init = rho[mid_idx] if len(rho) > 0 else np.median(rho)

        popt, pcov = curve_fit(
            fermi_dirac, rho[mask], P[mask],
            p0=[rho_c_init, 0.02],
            bounds=([0, 0.001], [1.0, 0.5]),
            maxfev=10000
        )
        y_pred = fermi_dirac(rho[mask], *popt)
        r2 = compute_r2(P[mask], y_pred)

        return {'rho_c': popt[0], 'delta': popt[1], 'R2': r2, 'n_points': np.sum(mask)}
    except Exception as e:
        if verbose:
            print(f"  Fit failed: {e}")
        return None


def load_checkpoint_data(base_path):
    """Load all checkpoint files from overnight directory."""
    overnight_dir = Path(base_path) / "data" / "overnight"

    canonical_files = list(overnight_dir.glob("canonical_checkpoint_*.csv"))
    geo2_files = list(overnight_dir.glob("geo2_checkpoint_*.csv"))

    # Load most recent/largest checkpoint for each
    canonical_df = None
    geo2_df = None

    for f in canonical_files:
        df = pd.read_csv(f)
        if canonical_df is None or len(df) > len(canonical_df):
            canonical_df = df
            canonical_file = f

    for f in geo2_files:
        df = pd.read_csv(f)
        if geo2_df is None or len(df) > len(geo2_df):
            geo2_df = df
            geo2_file = f

    return {
        'canonical': {'df': canonical_df, 'file': canonical_file if canonical_df is not None else None},
        'geo2': {'df': geo2_df, 'file': geo2_file if geo2_df is not None else None}
    }


def validate_data(df, ensemble_name):
    """Validate data quality and report issues."""
    issues = []

    if df is None or len(df) == 0:
        return [f"{ensemble_name}: No data found"]

    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"{ensemble_name}: NaN values in columns: {nan_cols}")

    # Check P values in [0, 1]
    p_cols = [c for c in df.columns if c.endswith('_P')]
    for col in p_cols:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range > 0:
                issues.append(f"{ensemble_name}: {out_of_range} values out of [0,1] in {col}")

    # Check dimensions present
    dims_present = sorted(df['d'].unique())
    issues.append(f"{ensemble_name}: Dimensions present: {dims_present}")

    # Check tau values
    taus_present = sorted(df['tau'].unique())
    issues.append(f"{ensemble_name}: Tau values: {taus_present}")

    # Row count
    issues.append(f"{ensemble_name}: {len(df)} data points")

    return issues


def check_dimension_ordering(df, criterion='spectral_P', tau=0.99):
    """Check that higher d transitions at lower rho (physics validation)."""
    results = {}
    df_tau = df[df['tau'] == tau] if 'tau' in df.columns else df

    for d in sorted(df_tau['d'].unique()):
        df_d = df_tau[df_tau['d'] == d].sort_values('rho')
        if len(df_d) < 3:
            continue

        rho = df_d['rho'].values
        P = df_d[criterion].values

        fit = fit_fermi(rho, P)
        if fit:
            results[d] = fit['rho_c']

    # Check ordering
    dims = sorted(results.keys())
    ordering_correct = True
    for i in range(len(dims) - 1):
        if results[dims[i]] < results[dims[i+1]]:
            ordering_correct = False
            break

    return results, ordering_correct


def aggregate_by_rho(df, d, tau=None, criterion='spectral_P'):
    """Aggregate multiple measurements at same (d, rho) point."""
    mask = df['d'] == d
    if tau is not None and 'tau' in df.columns:
        mask = mask & (df['tau'] == tau)

    df_filtered = df[mask].copy()
    if len(df_filtered) == 0:
        return None, None, None

    # Group by rho and aggregate
    grouped = df_filtered.groupby('rho').agg({
        criterion: ['mean', 'std', 'count'],
        'K': 'first'  # K should be same for same rho
    }).reset_index()

    grouped.columns = ['rho', 'P_mean', 'P_std', 'n', 'K']
    grouped = grouped.sort_values('rho')

    # Replace NaN std with 0 (single measurements)
    grouped['P_std'] = grouped['P_std'].fillna(0)

    return grouped['rho'].values, grouped['P_mean'].values, grouped['P_std'].values


def plot_spectral_transitions(data, ensemble, output_dir, tau=0.99):
    """Plot P(unreachable) vs rho for spectral criterion."""
    df = data['df']
    if df is None or len(df) == 0:
        print(f"  No data for {ensemble}")
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    fit_results = {}

    for d in sorted(df['d'].unique()):
        color = DIM_COLORS.get(d, 'gray')
        marker = DIM_MARKERS.get(d, 'o')

        rho, P, P_std = aggregate_by_rho(df, d, tau=tau, criterion='spectral_P')
        if rho is None or len(rho) < 3:
            continue

        # Plot data points with error bars
        ax.errorbar(rho, P, yerr=P_std, fmt=marker, color=color,
                   markersize=6, capsize=3, alpha=0.8, label=f'd={d}')

        # Fit and plot
        fit = fit_fermi(rho, P)
        if fit and fit['R2'] > 0.9:
            rho_fine = np.linspace(rho.min(), rho.max(), 100)
            P_fit = fermi_dirac(rho_fine, fit['rho_c'], fit['delta'])
            ax.plot(rho_fine, P_fit, '-', color=color, alpha=0.7, linewidth=1.5)
            fit_results[d] = fit

    ax.set_xlabel(r'Operator density $\rho = K/d^2$')
    ax.set_ylabel(r'$P(\mathrm{unreachable})$')
    ax.set_title(f'{ensemble} Spectral Criterion Phase Transitions (τ={tau})')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add fit info box
    if fit_results:
        lines = [f"$\\rho_c$ values (τ={tau}):"]
        for d in sorted(fit_results.keys()):
            lines.append(f"  d={d}: {fit_results[d]['rho_c']:.4f}")
        ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    filename = f"{ensemble.lower()}_spectral_transitions.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")

    return fit_results


def plot_comparison(canonical_data, geo2_data, output_dir, tau=0.99):
    """Plot canonical vs GEO2 comparison."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot canonical (solid lines)
    if canonical_data['df'] is not None:
        for d in sorted(canonical_data['df']['d'].unique()):
            color = DIM_COLORS.get(d, 'gray')
            rho, P, _ = aggregate_by_rho(canonical_data['df'], d, tau=tau, criterion='spectral_P')
            if rho is not None and len(rho) >= 3:
                ax.plot(rho, P, 'o-', color=color, markersize=5,
                       label=f'Canonical d={d}', alpha=0.9)

    # Plot GEO2 (dashed lines)
    if geo2_data['df'] is not None:
        for d in sorted(geo2_data['df']['d'].unique()):
            color = DIM_COLORS.get(d, 'gray')
            rho, P, _ = aggregate_by_rho(geo2_data['df'], d, tau=tau, criterion='spectral_P')
            if rho is not None and len(rho) >= 3:
                ax.plot(rho, P, 's--', color=color, markersize=5,
                       label=f'GEO2 d={d}', alpha=0.9)

    ax.set_xlabel(r'Operator density $\rho = K/d^2$')
    ax.set_ylabel(r'$P(\mathrm{unreachable})$')
    ax.set_title(f'Canonical vs GEO2 Phase Transitions (Spectral, τ={tau})')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "canonical_vs_geo2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved canonical_vs_geo2_comparison.png")


def plot_all_criteria_grid(canonical_data, geo2_data, output_dir, tau=0.99):
    """Plot 2x3 grid: rows=ensemble, cols=criterion."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    criteria = ['spectral_P', 'krylov_P', 'moment_P']
    crit_labels = ['Spectral', 'Krylov', 'Moment']
    ensembles = [('Canonical', canonical_data), ('GEO2', geo2_data)]

    for row, (ens_name, ens_data) in enumerate(ensembles):
        df = ens_data['df']
        if df is None:
            continue

        for col, (crit, crit_label) in enumerate(zip(criteria, crit_labels)):
            ax = axes[row, col]

            if crit not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{ens_name} - {crit_label}')
                continue

            for d in sorted(df['d'].unique()):
                color = DIM_COLORS.get(d, 'gray')
                marker = DIM_MARKERS.get(d, 'o')

                rho, P, _ = aggregate_by_rho(df, d, tau=tau, criterion=crit)
                if rho is not None and len(rho) >= 2:
                    ax.plot(rho, P, marker=marker, color=color,
                           markersize=4, linewidth=1, label=f'd={d}')

            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel(r'$P(\mathrm{unreach})$')
            ax.set_title(f'{ens_name} - {crit_label}')
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "all_criteria_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved all_criteria_grid.png")


def plot_P_vs_K(canonical_data, geo2_data, output_dir, tau=0.99):
    """Plot P vs K (number of Hamiltonians) instead of rho."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ensembles = [('Canonical', canonical_data), ('GEO2', geo2_data)]

    for i, (ens_name, ens_data) in enumerate(ensembles):
        ax = axes[i]
        df = ens_data['df']

        if df is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{ens_name}')
            continue

        # Filter by tau
        df_tau = df[df['tau'] == tau] if 'tau' in df.columns else df

        for d in sorted(df_tau['d'].unique()):
            color = DIM_COLORS.get(d, 'gray')
            marker = DIM_MARKERS.get(d, 'o')

            df_d = df_tau[df_tau['d'] == d].copy()
            # Aggregate by K
            grouped = df_d.groupby('K')['spectral_P'].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('K')

            ax.errorbar(grouped['K'], grouped['mean'], yerr=grouped['std'].fillna(0),
                       fmt=marker, color=color, markersize=5, capsize=2,
                       label=f'd={d}', alpha=0.8)

        ax.set_xlabel('Number of Hamiltonians K')
        ax.set_ylabel(r'$P(\mathrm{unreachable})$')
        ax.set_title(f'{ens_name} Spectral (τ={tau})')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "P_vs_K.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved P_vs_K.png")


def plot_multi_tau(canonical_data, output_dir):
    """Plot canonical with multiple tau values."""
    df = canonical_data['df']
    if df is None:
        return

    taus = sorted(df['tau'].unique())
    if len(taus) <= 1:
        return

    fig, axes = plt.subplots(1, len(taus), figsize=(5*len(taus), 5))
    if len(taus) == 1:
        axes = [axes]

    for i, tau in enumerate(taus):
        ax = axes[i]
        df_tau = df[df['tau'] == tau]

        for d in sorted(df_tau['d'].unique()):
            color = DIM_COLORS.get(d, 'gray')
            marker = DIM_MARKERS.get(d, 'o')

            rho, P, P_std = aggregate_by_rho(df_tau, d, criterion='spectral_P')
            if rho is not None and len(rho) >= 3:
                ax.errorbar(rho, P, yerr=P_std, fmt=marker, color=color,
                           markersize=5, capsize=2, label=f'd={d}', alpha=0.8)

        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$P(\mathrm{unreach})$')
        ax.set_title(f'τ = {tau}')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Canonical Spectral Criterion - Multiple τ', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "canonical_multi_tau.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved canonical_multi_tau.png")


def generate_analysis_summary(data, validation_results, fit_results, output_dir):
    """Generate OVERNIGHT_ANALYSIS.md summary document."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Overnight Production Run Analysis",
        f"\n**Generated**: {timestamp}",
        "\n**Status**: Experiments still in progress (checkpoint data)",
        "\n---\n",
        "## Data Summary\n"
    ]

    for ens_name, ens_data in [('Canonical', data['canonical']), ('GEO2', data['geo2'])]:
        df = ens_data['df']
        if df is None:
            lines.append(f"### {ens_name}\n- No data available\n")
            continue

        lines.append(f"### {ens_name}\n")
        lines.append(f"- **File**: `{ens_data['file'].name}`")
        lines.append(f"- **Rows**: {len(df)}")
        lines.append(f"- **Dimensions**: {sorted(df['d'].unique())}")
        if 'tau' in df.columns:
            lines.append(f"- **τ values**: {sorted(df['tau'].unique())}")
        lines.append(f"- **ρ range**: [{df['rho'].min():.4f}, {df['rho'].max():.4f}]")
        lines.append(f"- **K range**: [{df['K'].min()}, {df['K'].max()}]")
        lines.append("")

    lines.append("---\n")
    lines.append("## Validation Results\n")
    for result in validation_results:
        lines.append(f"- {result}")

    lines.append("\n---\n")
    lines.append("## Transition Densities (ρ_c)\n")
    lines.append("### Spectral Criterion (τ=0.99)\n")
    lines.append("| Ensemble | d=8 | d=16 | d=32 | d=64 |")
    lines.append("|----------|-----|------|------|------|")

    for ens_name, fits in fit_results.items():
        if fits is None:
            continue
        row = [ens_name]
        for d in [8, 16, 32, 64]:
            if d in fits:
                row.append(f"{fits[d]['rho_c']:.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n---\n")
    lines.append("## Key Observations\n")

    # Check dimension ordering
    for ens_name, fits in fit_results.items():
        if fits is None or len(fits) < 2:
            continue
        dims = sorted(fits.keys())
        rho_cs = [fits[d]['rho_c'] for d in dims]
        if all(rho_cs[i] >= rho_cs[i+1] for i in range(len(rho_cs)-1)):
            lines.append(f"- **{ens_name}**: Dimension ordering correct (higher d → lower ρ_c)")
        else:
            lines.append(f"- **{ens_name}**: ⚠️ Dimension ordering anomaly detected")

    # GEO2 Krylov observation
    if data['geo2']['df'] is not None:
        df = data['geo2']['df']
        if 'krylov_P' in df.columns:
            krylov_always_zero = (df['krylov_P'] == 0).all()
            if krylov_always_zero:
                lines.append("- **GEO2 Krylov**: Always P=0 (expected - dense Hamiltonians fill Krylov space)")

    lines.append("\n---\n")
    lines.append("## Generated Figures\n")
    lines.append("1. `canonical_spectral_transitions.png` - Canonical phase transitions")
    lines.append("2. `geo2_spectral_transitions.png` - GEO2 phase transitions")
    lines.append("3. `canonical_vs_geo2_comparison.png` - Side-by-side comparison")
    lines.append("4. `all_criteria_grid.png` - 2x3 grid of all criteria")
    lines.append("5. `P_vs_K.png` - P vs number of Hamiltonians")
    lines.append("6. `canonical_multi_tau.png` - Multiple τ comparison")

    lines.append("\n---\n")
    lines.append("## Notes\n")
    lines.append("- Data from checkpoint files (experiments still running)")
    lines.append("- Fit quality may improve with more data points")
    lines.append("- GEO2 d=64 data incomplete")

    with open(output_dir / "OVERNIGHT_ANALYSIS.md", 'w') as f:
        f.write('\n'.join(lines))

    print("  Saved OVERNIGHT_ANALYSIS.md")


def main():
    base_path = Path("/Users/tomas/PycharmProjects/reachability/reachability")
    output_dir = base_path / "fig" / "overnight"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Overnight Checkpoint Analysis")
    print("=" * 60)

    # Load data
    print("\n1. Loading checkpoint data...")
    data = load_checkpoint_data(base_path)

    for ens_name, ens_data in data.items():
        if ens_data['df'] is not None:
            print(f"   {ens_name}: {len(ens_data['df'])} rows from {ens_data['file'].name}")
        else:
            print(f"   {ens_name}: No data found")

    # Validate
    print("\n2. Validating data...")
    validation_results = []
    for ens_name, ens_data in data.items():
        validation_results.extend(validate_data(ens_data['df'], ens_name.title()))
    for result in validation_results:
        print(f"   {result}")

    # Generate plots
    print("\n3. Generating plots...")

    fit_results = {}

    # Canonical spectral transitions
    if data['canonical']['df'] is not None:
        fits = plot_spectral_transitions(data['canonical'], 'Canonical', output_dir, tau=0.99)
        fit_results['Canonical'] = fits

        # Multi-tau plot
        plot_multi_tau(data['canonical'], output_dir)

    # GEO2 spectral transitions
    if data['geo2']['df'] is not None:
        fits = plot_spectral_transitions(data['geo2'], 'GEO2', output_dir, tau=0.99)
        fit_results['GEO2'] = fits

    # Comparison
    if data['canonical']['df'] is not None or data['geo2']['df'] is not None:
        plot_comparison(data['canonical'], data['geo2'], output_dir, tau=0.99)

    # All criteria grid
    plot_all_criteria_grid(data['canonical'], data['geo2'], output_dir, tau=0.99)

    # P vs K
    plot_P_vs_K(data['canonical'], data['geo2'], output_dir, tau=0.99)

    # Generate summary
    print("\n4. Generating analysis summary...")
    generate_analysis_summary(data, validation_results, fit_results, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
