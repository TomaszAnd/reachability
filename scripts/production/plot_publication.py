#!/usr/bin/env python3
"""
Generate publication-quality canonical vs GEO2 comparison plots.

Uses partial data:
- Canonical: d=8,16,32,64(partial)
- GEO2: d=8,16,32
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {8: '#1f77b4', 16: '#ff7f0e', 32: '#2ca02c', 64: '#d62728'}
MARKERS = {8: 'o', 16: 's', 32: '^', 64: 'D'}


def load_data(data_dir):
    """Load all available CSV data, separated by ensemble."""
    data_dir = Path(data_dir)

    canonical_dfs = []
    geo2_dfs = []

    for csv_file in data_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            continue
        name = csv_file.name.lower()

        if 'ensemble' in df.columns:
            can = df[df['ensemble'] == 'canonical']
            geo = df[df['ensemble'] == 'GEO2']
            if len(can) > 0:
                canonical_dfs.append(can)
            if len(geo) > 0:
                geo2_dfs.append(geo)
        elif 'canonical' in name:
            canonical_dfs.append(df)
        elif 'geo2' in name:
            geo2_dfs.append(df)

        print(f"  Loaded: {csv_file.name} ({len(df)} rows)")

    df_canonical = pd.concat(canonical_dfs, ignore_index=True).drop_duplicates(
        subset=['d', 'rho', 'tau'], keep='last') if canonical_dfs else None
    df_geo2 = pd.concat(geo2_dfs, ignore_index=True).drop_duplicates(
        subset=['d', 'rho', 'tau'], keep='last') if geo2_dfs else None

    return df_canonical, df_geo2


def plot_side_by_side_3panel(df_canonical, df_geo2, output_path, tau=0.99):
    """Side-by-side: Canonical | GEO2 | Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    df_c = df_canonical[df_canonical['tau'] == tau] if (
        df_canonical is not None and 'tau' in df_canonical.columns) else df_canonical
    df_g = df_geo2[df_geo2['tau'] == tau] if (
        df_geo2 is not None and 'tau' in df_geo2.columns) else df_geo2

    # Panel 1: Canonical
    ax1 = axes[0]
    if df_c is not None:
        for d in sorted(df_c['d'].unique()):
            df_d = df_c[df_c['d'] == d].sort_values('rho')
            ax1.plot(df_d['rho'], df_d['spectral_P'],
                     color=COLORS.get(int(d), 'gray'),
                     marker=MARKERS.get(int(d), 'o'),
                     markersize=6, linewidth=1.5,
                     label=f'd={int(d)}')
    ax1.set_xlabel(r'Control density $\rho = K/d^2$')
    ax1.set_ylabel('P(unreachable)')
    ax1.set_title('Canonical Ensemble')
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: GEO2
    ax2 = axes[1]
    if df_g is not None:
        for d in sorted(df_g['d'].unique()):
            df_d = df_g[df_g['d'] == d].sort_values('rho')
            ax2.plot(df_d['rho'], df_d['spectral_P'],
                     color=COLORS.get(int(d), 'gray'),
                     marker=MARKERS.get(int(d), 'o'),
                     markersize=6, linewidth=1.5,
                     label=f'd={int(d)}')
    ax2.set_xlabel(r'Control density $\rho = K/d^2$')
    ax2.set_title('GEO2 Ensemble')
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Overlay
    ax3 = axes[2]
    common_dims = set()
    if df_c is not None:
        common_dims.update(int(d) for d in df_c['d'].unique())
    if df_g is not None:
        common_dims = common_dims.intersection(int(d) for d in df_g['d'].unique())

    for d in sorted(common_dims):
        color = COLORS.get(d, 'gray')
        marker = MARKERS.get(d, 'o')

        if df_c is not None:
            df_cd = df_c[df_c['d'] == d].sort_values('rho')
            if len(df_cd) > 0:
                ax3.plot(df_cd['rho'], df_cd['spectral_P'],
                         color=color, marker=marker,
                         markersize=5, linewidth=1.5, linestyle='-',
                         label=f'Can d={d}')

        if df_g is not None:
            df_gd = df_g[df_g['d'] == d].sort_values('rho')
            if len(df_gd) > 0:
                ax3.plot(df_gd['rho'], df_gd['spectral_P'],
                         color=color, marker=marker,
                         markersize=5, linewidth=1.5, linestyle='--',
                         markerfacecolor='none',
                         label=f'GEO2 d={d}')

    ax3.set_xlabel(r'Control density $\rho = K/d^2$')
    ax3.set_title('Comparison (solid=Canonical, dashed=GEO2)')
    ax3.set_ylim(-0.02, 1.02)
    ax3.legend(loc='upper right', ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(r'Spectral Criterion ($\tau$ = ' + str(tau) + ')', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_criterion_comparison_2x3(df_canonical, df_geo2, output_path, tau=0.99):
    """2x3 grid: rows = ensemble, columns = criterion."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

    criteria = [('spectral_P', 'Spectral'), ('krylov_P', 'Krylov'), ('moment_P', 'Moment')]
    ensembles = [
        (df_canonical, 'Canonical', 0),
        (df_geo2, 'GEO2', 1),
    ]

    for df_raw, ens_name, row in ensembles:
        if df_raw is None:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                     transform=axes[row, col].transAxes)
            continue

        df = df_raw[df_raw['tau'] == tau] if 'tau' in df_raw.columns else df_raw

        for col, (crit_col, crit_name) in enumerate(criteria):
            ax = axes[row, col]

            if crit_col not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                         transform=ax.transAxes)
                continue

            for d in sorted(df['d'].unique()):
                df_d = df[df['d'] == d].sort_values('rho')
                ax.plot(df_d['rho'], df_d[crit_col],
                        color=COLORS.get(int(d), 'gray'),
                        marker=MARKERS.get(int(d), 'o'),
                        markersize=5, linewidth=1.5,
                        label=f'd={int(d)}')

            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(crit_name, fontsize=14)
            if row == 1:
                ax.set_xlabel(r'$\rho = K/d^2$')
            if col == 0:
                ax.set_ylabel(f'{ens_name}\nP(unreachable)')
                ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(r'Three Criteria Comparison ($\tau$ = ' + str(tau) + ')',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_dimension_ordering(df, ensemble, output_path, tau=0.99):
    """Dimension ordering check for a single ensemble."""
    if df is None or len(df) == 0:
        return

    df_plot = df[df['tau'] == tau] if 'tau' in df.columns else df
    dims = sorted(df_plot['d'].unique())

    fig, ax = plt.subplots(figsize=(10, 7))

    for d in dims:
        df_d = df_plot[df_plot['d'] == d].sort_values('rho')
        ax.plot(df_d['rho'], df_d['spectral_P'],
                color=COLORS.get(int(d), 'gray'),
                marker=MARKERS.get(int(d), 'o'),
                markersize=6, linewidth=2,
                label=f'd={int(d)}')

    ax.set_xlabel(r'Control density $\rho = K/d^2$', fontsize=14)
    ax.set_ylabel('P(unreachable) - Spectral', fontsize=14)
    ax.set_title(f'{ensemble.upper()} Dimension Ordering ($\\tau$ = {tau})', fontsize=16)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.05,
            'Correct ordering: Higher d drops first\n(at fixed rho, more operators = more reachable)',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    data_dir = Path("data/production")
    output_dir = Path("fig/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_canonical, df_geo2 = load_data(data_dir)

    if df_canonical is not None:
        print(f"\nCanonical: {len(df_canonical)} rows, dims={sorted(df_canonical['d'].unique())}")
    if df_geo2 is not None:
        print(f"GEO2: {len(df_geo2)} rows, dims={sorted(df_geo2['d'].unique())}")

    print("\nGenerating plots...")

    # 1. Side-by-side comparison
    plot_side_by_side_3panel(
        df_canonical, df_geo2,
        output_dir / "canonical_vs_geo2_sidebyside.png"
    )

    # 2. 2x3 all-criteria grid
    plot_criterion_comparison_2x3(
        df_canonical, df_geo2,
        output_dir / "all_criteria_2x3.png"
    )

    # 3. Dimension ordering checks
    if df_canonical is not None:
        plot_dimension_ordering(df_canonical, 'canonical',
                                output_dir / 'canonical_dimension_ordering.png')
    if df_geo2 is not None:
        plot_dimension_ordering(df_geo2, 'geo2',
                                output_dir / 'geo2_dimension_ordering.png')

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
