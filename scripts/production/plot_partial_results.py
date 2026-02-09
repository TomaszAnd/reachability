#!/usr/bin/env python3
"""
Plot results from partial production data.

Handles missing dimensions, single ensemble, and comparison when both available.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
})

COLORS = {8: '#1f77b4', 16: '#ff7f0e', 32: '#2ca02c', 64: '#d62728'}
MARKERS = {8: 'o', 16: 's', 32: '^', 64: 'D'}


def load_all_data(data_dir):
    """Load all CSV files and combine by ensemble."""
    data_dir = Path(data_dir)

    canonical_dfs = []
    geo2_dfs = []

    for csv_file in sorted(data_dir.glob("*.csv")):
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

    df_canonical = pd.concat(canonical_dfs, ignore_index=True) if canonical_dfs else None
    df_geo2 = pd.concat(geo2_dfs, ignore_index=True) if geo2_dfs else None

    # Deduplicate (keep last)
    if df_canonical is not None and len(df_canonical) > 0:
        df_canonical = df_canonical.drop_duplicates(subset=['d', 'rho', 'tau'], keep='last')
    if df_geo2 is not None and len(df_geo2) > 0:
        df_geo2 = df_geo2.drop_duplicates(subset=['d', 'rho', 'tau'], keep='last')

    return df_canonical, df_geo2


def plot_3panel_criteria(df, ensemble, output_path, tau=0.99):
    """3-panel by criterion for single ensemble."""
    if df is None or len(df) == 0:
        print(f"No data for {ensemble}")
        return

    df_plot = df[df['tau'] == tau] if 'tau' in df.columns else df

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    criteria = [('spectral_P', 'Spectral'), ('krylov_P', 'Krylov'), ('moment_P', 'Moment')]

    dims = sorted(df_plot['d'].unique())

    for ax_idx, (col, title) in enumerate(criteria):
        ax = axes[ax_idx]

        for d in dims:
            df_d = df_plot[df_plot['d'] == d].sort_values('rho')
            if col in df_d.columns:
                ax.plot(df_d['rho'], df_d[col],
                        color=COLORS.get(int(d), 'gray'),
                        marker=MARKERS.get(int(d), 'o'),
                        markersize=6, linewidth=1.5,
                        label=f'd={int(d)}')

        ax.set_xlabel(r'Control density $\rho = K/d^2$')
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:
            ax.set_ylabel('P(unreachable)')
            ax.legend(loc='upper right')

    fig.suptitle(f'{ensemble.upper()} Ensemble ($\\tau$ = {tau})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(df_canonical, df_geo2, output_path, tau=0.99):
    """Overlay both ensembles - spectral criterion only."""
    if df_canonical is None and df_geo2 is None:
        print("No data for comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    all_dims = set()
    if df_canonical is not None:
        df_c = df_canonical[df_canonical['tau'] == tau] if 'tau' in df_canonical.columns else df_canonical
        all_dims.update(df_c['d'].unique())
    else:
        df_c = None

    if df_geo2 is not None:
        df_g = df_geo2[df_geo2['tau'] == tau] if 'tau' in df_geo2.columns else df_geo2
        all_dims.update(df_g['d'].unique())
    else:
        df_g = None

    for d in sorted(all_dims):
        d = int(d)
        color = COLORS.get(d, 'gray')
        marker = MARKERS.get(d, 'o')

        if df_c is not None:
            dd = df_c[df_c['d'] == d].sort_values('rho')
            if len(dd) > 0:
                ax.plot(dd['rho'], dd['spectral_P'],
                        color=color, marker=marker,
                        markersize=6, linewidth=1.5, linestyle='-',
                        label=f'Canonical d={d}')

        if df_g is not None:
            dd = df_g[df_g['d'] == d].sort_values('rho')
            if len(dd) > 0:
                ax.plot(dd['rho'], dd['spectral_P'],
                        color=color, marker=marker,
                        markersize=6, linewidth=1.5, linestyle='--',
                        markerfacecolor='none',
                        label=f'GEO2 d={d}')

    ax.set_xlabel(r'Control density $\rho = K/d^2$', fontsize=14)
    ax.set_ylabel('P(unreachable)', fontsize=14)
    ax.set_title(f'Canonical vs GEO2 - Spectral Criterion ($\\tau$ = {tau})', fontsize=16)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_dimension_ordering(df, ensemble, output_path, tau=0.99):
    """Check dimension ordering: higher d should drop first at fixed rho."""
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

    # Annotate ordering
    ax.text(0.02, 0.05,
            'Expected: Higher d drops first\n(more operators needed at higher d)',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/production')
    parser.add_argument('--output-dir', default='fig/comparison')
    parser.add_argument('--tau', type=float, default=0.99)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_canonical, df_geo2 = load_all_data(args.data_dir)

    print(f"Canonical: {len(df_canonical) if df_canonical is not None else 0} rows")
    print(f"GEO2: {len(df_geo2) if df_geo2 is not None else 0} rows")

    if df_canonical is not None:
        plot_3panel_criteria(df_canonical, 'canonical',
                             output_dir / 'canonical_3panel.png', args.tau)
        plot_dimension_ordering(df_canonical, 'canonical',
                                output_dir / 'canonical_dimension_ordering.png', args.tau)

    if df_geo2 is not None:
        plot_3panel_criteria(df_geo2, 'geo2',
                             output_dir / 'geo2_3panel.png', args.tau)
        plot_dimension_ordering(df_geo2, 'geo2',
                                output_dir / 'geo2_dimension_ordering.png', args.tau)

    if df_canonical is not None or df_geo2 is not None:
        plot_comparison(df_canonical, df_geo2,
                        output_dir / 'canonical_vs_geo2_comparison.png', args.tau)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
