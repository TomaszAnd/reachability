#!/usr/bin/env python3
"""
Generate publication-quality plots from overnight production data.

Usage:
    python scripts/plot_publication.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# STYLE (v7 style from plot_overnight_v7style.py)
# =============================================================================

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

DIM_COLORS = {
    16: '#ff7f0e',   # Orange
    32: '#2ca02c',   # Green
    64: '#d62728',   # Red
}

DIM_MARKERS = {
    16: 's',   # square
    32: '^',   # triangle up
    64: 'D',   # diamond
}

CRIT_COLORS = {
    'moment': 'blue',
    'spectral': 'red',
    'krylov': 'green',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir):
    """Load all CSV files from data directories."""
    data = {'canonical': {}, 'qubitgrid': {}}

    for model, subdir in [('canonical', 'canonical'), ('qubitgrid', 'qubitgrid')]:
        model_dir = data_dir / subdir
        for d in [16, 32, 64]:
            # Match both tau0.99 and tau0_99 naming
            files = sorted(model_dir.glob(f'{model}_d{d}_tau0*_*.csv'))
            if files:
                latest = files[-1]
                data[model][d] = pd.read_csv(latest)
                print(f"  Loaded {model} d={d}: {len(data[model][d])} rows from {latest.name}")

    return data


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_canonical_3panel(data, output_dir):
    """Plot 1: Canonical 3-panel (moment, spectral, krylov)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, crit in zip(axes, ['moment', 'spectral', 'krylov']):
        for d in [16, 32, 64]:
            if d not in data['canonical']:
                continue
            df = data['canonical'][d]
            if f'{crit}_P' not in df.columns:
                continue
            ax.errorbar(
                df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                fmt=f'{DIM_MARKERS[d]}-', color=DIM_COLORS[d], label=f'd={d}',
                capsize=3, markersize=6, linewidth=1.5,
            )
        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(f'Canonical - {crit.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 0.26)

    plt.tight_layout()
    plt.savefig(output_dir / 'publication_canonical_3panel.png')
    plt.savefig(output_dir / 'publication_canonical_3panel.pdf')
    print("Saved publication_canonical_3panel.png/pdf")
    plt.close()


def plot_qubitgrid_2panel(data, output_dir):
    """Plot 2: QubitGrid 2-panel (moment, spectral)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, crit in zip(axes, ['moment', 'spectral']):
        for d in [16, 32, 64]:
            if d not in data['qubitgrid']:
                continue
            df = data['qubitgrid'][d]
            if f'{crit}_P' not in df.columns:
                continue
            ax.errorbar(
                df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                fmt=f'{DIM_MARKERS[d]}-', color=DIM_COLORS[d], label=f'd={d}',
                capsize=3, markersize=6, linewidth=1.5,
            )
        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(f'QubitGrid - {crit.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 0.13)

    plt.tight_layout()
    plt.savefig(output_dir / 'publication_qubitgrid_2panel.png')
    plt.savefig(output_dir / 'publication_qubitgrid_2panel.pdf')
    print("Saved publication_qubitgrid_2panel.png/pdf")
    plt.close()


def plot_combined_2x3(data, output_dir):
    """Plot 3: Combined 2x3 panel."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Canonical
    for ax, crit in zip(axes[0], ['moment', 'spectral', 'krylov']):
        for d in [16, 32, 64]:
            if d not in data['canonical']:
                continue
            df = data['canonical'][d]
            if f'{crit}_P' not in df.columns:
                continue
            ax.errorbar(
                df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                fmt=f'{DIM_MARKERS[d]}-', color=DIM_COLORS[d], label=f'd={d}',
                capsize=3, markersize=5,
            )
        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(f'Canonical - {crit.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 0.26)

    # Row 2: QubitGrid (moment, spectral, empty)
    for ax, crit in zip(axes[1][:2], ['moment', 'spectral']):
        for d in [16, 32, 64]:
            if d not in data['qubitgrid']:
                continue
            df = data['qubitgrid'][d]
            if f'{crit}_P' not in df.columns:
                continue
            ax.errorbar(
                df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                fmt=f'{DIM_MARKERS[d]}-', color=DIM_COLORS[d], label=f'd={d}',
                capsize=3, markersize=5,
            )
        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(f'QubitGrid - {crit.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 0.13)

    # Empty panel with note
    axes[1, 2].text(
        0.5, 0.5,
        'QubitGrid Krylov\nomitted\n(P = 0 for all $\\rho$)',
        ha='center', va='center', fontsize=14,
        transform=axes[1, 2].transAxes,
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
    )
    axes[1, 2].set_axis_off()

    plt.tight_layout()
    plt.savefig(output_dir / 'publication_combined_2x3.png')
    plt.savefig(output_dir / 'publication_combined_2x3.pdf')
    print("Saved publication_combined_2x3.png/pdf")
    plt.close()


def plot_spectral_comparison(data, output_dir):
    """Plot 4: Spectral comparison (Canonical vs QubitGrid)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model, title, xlim in [
        (axes[0], 'canonical', 'Canonical - Spectral', 0.26),
        (axes[1], 'qubitgrid', 'QubitGrid - Spectral', 0.13),
    ]:
        for d in [16, 32, 64]:
            if d not in data[model]:
                continue
            df = data[model][d]
            if 'spectral_P' not in df.columns:
                continue
            ax.errorbar(
                df['rho'], df['spectral_P'], yerr=df['spectral_sem'],
                fmt=f'{DIM_MARKERS[d]}-', color=DIM_COLORS[d], label=f'd={d}',
                capsize=3, markersize=6, linewidth=1.5,
            )
        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, xlim)

    plt.tight_layout()
    plt.savefig(output_dir / 'publication_spectral_comparison.png')
    plt.savefig(output_dir / 'publication_spectral_comparison.pdf')
    print("Saved publication_spectral_comparison.png/pdf")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'fig' / 'publication'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(data_dir)

    print("\nGenerating plots...")
    plot_canonical_3panel(data, output_dir)
    plot_qubitgrid_2panel(data, output_dir)
    plot_combined_2x3(data, output_dir)
    plot_spectral_comparison(data, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    main()
