#!/usr/bin/env python3
"""
Plot unified comparison between Canonical and GEO2 ensembles.

Generates:
1. 3-panel figure for Canonical (one panel per criterion)
2. 3-panel figure for GEO2 (one panel per criterion)
3. Direct comparison overlay plot (canonical vs GEO2)

Usage:
    python scripts/production/plot_unified_comparison.py \
        --canonical data/production/canonical_unified_*.csv \
        --geo2 data/production/geo2_unified_*.csv \
        --output-dir fig/comparison
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

COLORS = {
    8: '#1f77b4',
    16: '#ff7f0e',
    32: '#2ca02c',
    64: '#d62728',
}

MARKERS = {
    8: 'o',
    16: 's',
    32: '^',
    64: 'D',
}


def plot_3panel_criteria(df, ensemble, output_path, tau=0.99):
    """Generate 3-panel figure (one panel per criterion) at fixed tau."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    dimensions = sorted(df['d'].unique())
    df_tau = df[df['tau'] == tau]

    criteria = [
        ('spectral_P', 'Spectral'),
        ('krylov_P', 'Krylov'),
        ('moment_P', 'Moment'),
    ]

    for ax_idx, (col, label) in enumerate(criteria):
        ax = axes[ax_idx]

        for d in dimensions:
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')

            ax.plot(df_d['rho'], df_d[col],
                    color=COLORS.get(d, 'gray'),
                    marker=MARKERS.get(d, 'o'),
                    markersize=5,
                    label=f'd={d}',
                    linewidth=1.5)

        ax.set_xlabel('Control density rho = K/d^2')
        ax.set_title(f'{label} Criterion')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('P(unreachable)')
            ax.legend(loc='upper right')

    fig.suptitle(f'{ensemble.upper()} Ensemble (tau = {tau})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_3panel_taus(df, ensemble, output_path, criterion='spectral_P',
                     criterion_label='Spectral', tau_values=[0.99, 0.95, 0.90]):
    """Generate 3-panel figure (one panel per tau)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    dimensions = sorted(df['d'].unique())

    for ax_idx, tau in enumerate(tau_values):
        ax = axes[ax_idx]
        df_tau = df[df['tau'] == tau]

        for d in dimensions:
            df_d = df_tau[df_tau['d'] == d].sort_values('rho')

            ax.plot(df_d['rho'], df_d[criterion],
                    color=COLORS.get(d, 'gray'),
                    marker=MARKERS.get(d, 'o'),
                    markersize=5,
                    label=f'd={d}',
                    linewidth=1.5)

        ax.set_xlabel('Control density rho = K/d^2')
        ax.set_title(f'tau = {tau}')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('P(unreachable)')
            ax.legend(loc='upper right')

    fig.suptitle(f'{ensemble.upper()} - {criterion_label} Criterion', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison_overlay(df_canonical, df_geo2, output_path, tau=0.99):
    """Overlay canonical and GEO2 on same plot for direct comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    dimensions = sorted(set(df_canonical['d'].unique()) & set(df_geo2['d'].unique()))

    criteria = [
        ('spectral_P', 'Spectral'),
        ('krylov_P', 'Krylov'),
        ('moment_P', 'Moment'),
    ]

    for ax_idx, (col, label) in enumerate(criteria):
        ax = axes[ax_idx]

        for d in dimensions:
            # Canonical (solid lines)
            df_c = df_canonical[
                (df_canonical['d'] == d) & (df_canonical['tau'] == tau)
            ].sort_values('rho')
            if not df_c.empty:
                ax.plot(df_c['rho'], df_c[col],
                        color=COLORS.get(d, 'gray'),
                        marker=MARKERS.get(d, 'o'),
                        markersize=5,
                        linestyle='-',
                        linewidth=1.5,
                        label=f'Canon d={d}')

            # GEO2 (dashed lines)
            df_g = df_geo2[
                (df_geo2['d'] == d) & (df_geo2['tau'] == tau)
            ].sort_values('rho')
            if not df_g.empty:
                ax.plot(df_g['rho'], df_g[col],
                        color=COLORS.get(d, 'gray'),
                        marker=MARKERS.get(d, 'o'),
                        markersize=5,
                        linestyle='--',
                        linewidth=1.5,
                        markerfacecolor='none',
                        label=f'GEO2 d={d}')

        ax.set_xlabel('Control density rho = K/d^2')
        ax.set_title(f'{label} Criterion')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('P(unreachable)')
            ax.legend(loc='upper right', fontsize=8, ncol=2)

    fig.suptitle(f'Canonical vs GEO2 Comparison (tau = {tau})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def load_data(path):
    """Load data from CSV or pickle file."""
    path = str(path)
    if path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if 'dataframe' in data:
            return data['dataframe']
        return pd.DataFrame(data['results'])


def main():
    parser = argparse.ArgumentParser(description='Plot unified ensemble comparison')
    parser.add_argument('--canonical', required=True, help='Path to canonical CSV or PKL')
    parser.add_argument('--geo2', required=True, help='Path to GEO2 CSV or PKL')
    parser.add_argument('--output-dir', default='fig/comparison', help='Output directory')
    args = parser.parse_args()

    df_canonical = load_data(args.canonical)
    df_geo2 = load_data(args.geo2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Canonical data: {len(df_canonical)} rows, dims={sorted(df_canonical['d'].unique())}")
    print(f"GEO2 data: {len(df_geo2)} rows, dims={sorted(df_geo2['d'].unique())}")

    # 3-panel by criterion (one per ensemble)
    plot_3panel_criteria(df_canonical, 'canonical',
                         output_dir / 'canonical_unified_3panel.png')
    plot_3panel_criteria(df_geo2, 'geo2',
                         output_dir / 'geo2_unified_3panel.png')

    # 3-panel by tau (spectral only, one per ensemble)
    plot_3panel_taus(df_canonical, 'canonical',
                     output_dir / 'canonical_spectral_3tau.png')
    plot_3panel_taus(df_geo2, 'geo2',
                     output_dir / 'geo2_spectral_3tau.png')

    # Comparison overlay
    plot_comparison_overlay(df_canonical, df_geo2,
                            output_dir / 'canonical_vs_geo2_comparison.png')

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
