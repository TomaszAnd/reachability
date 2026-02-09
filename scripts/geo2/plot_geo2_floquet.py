#!/usr/bin/env python3
"""
Plot GEO2 Floquet Engineering Experimental Results

Generate publication-quality plots comparing:
1. Regular Moment criterion (baseline, P ≈ 0)
2. Floquet Moment criterion (order 1 and 2)

Expected outcome:
- Regular Moment: Flat at P ≈ 0 (too weak, λ-independent)
- Floquet Moment (order 1): Slight improvement (time-averaged)
- Floquet Moment (order 2): Clear transition (commutators make it λ-dependent!)

Plot types:
-----------
1. Main comparison: 3 criteria vs ρ for each dimension
2. Floquet order comparison: Order 1 vs Order 2
3. Multi-dimension overlay: All dimensions on one plot
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (14, 10),
    'figure.dpi': 200,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})


def load_results(filepath: str) -> Dict:
    """Load experimental results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_main_comparison(results: Dict, output_dir: Path):
    """
    Plot main comparison: Regular vs Floquet Moment criteria.

    Creates one plot per dimension with 3 curves:
    - Regular Moment (baseline)
    - Floquet Moment Order 1
    - Floquet Moment Order 2
    """
    data = results['data']
    config = results['config']

    for d in sorted(data.keys()):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Extract data
        rho_values = sorted(data[d].keys())
        P_regular = [data[d][rho]['P_regular_moment'] for rho in rho_values]
        P_floquet1 = [data[d][rho]['P_floquet_moment_order1'] for rho in rho_values]
        P_floquet2 = [data[d][rho]['P_floquet_moment_order2'] for rho in rho_values]

        # Plot curves
        ax.plot(rho_values, P_regular, 'o-', label='Regular Moment (baseline)',
                color='gray', alpha=0.7)
        ax.plot(rho_values, P_floquet1, 's-', label='Floquet Moment (order 1)',
                color='orange', alpha=0.8)
        ax.plot(rho_values, P_floquet2, '^-', label='Floquet Moment (order 2)',
                color='red', linewidth=2.5)

        # Formatting
        ax.set_xlabel(r'Density $\rho = K/d^2$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14, fontweight='bold')
        ax.set_title(f'GEO2 Floquet Engineering: d={d}\n'
                     f'{config["driving_type"]} driving, T={config["T"]}, '
                     f'Magnus order={config["magnus_order"]}',
                     fontsize=12)
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Save
        output_file = output_dir / f'geo2_floquet_main_d{d}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_floquet_order_comparison(results: Dict, output_dir: Path):
    """
    Plot comparison between Magnus order 1 and order 2.

    Shows the effect of including commutator terms [H_j, H_k] in the
    effective Hamiltonian.
    """
    data = results['data']
    config = results['config']

    for d in sorted(data.keys()):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Extract data
        rho_values = sorted(data[d].keys())
        P_floquet1 = [data[d][rho]['P_floquet_moment_order1'] for rho in rho_values]
        P_floquet2 = [data[d][rho]['P_floquet_moment_order2'] for rho in rho_values]

        # Plot curves
        ax.plot(rho_values, P_floquet1, 'o-', label='Order 1 (time-averaged only)',
                color='orange', linewidth=2, markersize=6)
        ax.plot(rho_values, P_floquet2, '^-', label='Order 2 (with commutators)',
                color='red', linewidth=2.5, markersize=8)

        # Add shading between curves to show difference
        ax.fill_between(rho_values, P_floquet1, P_floquet2,
                        alpha=0.2, color='red', label='Order 2 enhancement')

        # Formatting
        ax.set_xlabel(r'Density $\rho = K/d^2$', fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14, fontweight='bold')
        ax.set_title(f'Floquet Magnus Order Comparison: d={d}\n'
                     f'Effect of commutator terms $[H_j, H_k]$',
                     fontsize=12)
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Save
        output_file = output_dir / f'geo2_floquet_order_comparison_d{d}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_multi_dimension_overlay(results: Dict, output_dir: Path):
    """
    Plot all dimensions on a single plot for Floquet order 2.

    Shows scaling of critical density ρ_c with dimension.
    """
    data = results['data']
    config = results['config']

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(data)))

    for idx, d in enumerate(sorted(data.keys())):
        # Extract data
        rho_values = sorted(data[d].keys())
        P_floquet2 = [data[d][rho]['P_floquet_moment_order2'] for rho in rho_values]

        # Plot
        ax.plot(rho_values, P_floquet2, 'o-', label=f'd={d}',
                color=colors[idx], linewidth=2.5, markersize=7)

    # Formatting
    ax.set_xlabel(r'Density $\rho = K/d^2$', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=14, fontweight='bold')
    ax.set_title(f'GEO2 Floquet Engineering: Multi-Dimension Comparison\n'
                 f'Floquet Moment (order 2) - {config["driving_type"]} driving',
                 fontsize=12)
    ax.legend(loc='best', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Save
    output_file = output_dir / 'geo2_floquet_multidim.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_3panel_comparison(results: Dict, output_dir: Path):
    """
    Create 3-panel figure similar to GEO2 v3:
    Panel 1: Regular Moment
    Panel 2: Floquet Moment (order 1)
    Panel 3: Floquet Moment (order 2)
    """
    data = results['data']
    config = results['config']

    # Use largest dimension for 3-panel plot
    d = max(data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Extract data
    rho_values = sorted(data[d].keys())
    P_regular = [data[d][rho]['P_regular_moment'] for rho in rho_values]
    P_floquet1 = [data[d][rho]['P_floquet_moment_order1'] for rho in rho_values]
    P_floquet2 = [data[d][rho]['P_floquet_moment_order2'] for rho in rho_values]

    # Panel 1: Regular Moment
    axes[0].plot(rho_values, P_regular, 'o-', color='gray', linewidth=2.5)
    axes[0].set_title('Regular Moment\n(λ-independent, too weak)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(r'$\rho = K/d^2$', fontsize=12)
    axes[0].set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)

    # Panel 2: Floquet Order 1
    axes[1].plot(rho_values, P_floquet1, 's-', color='orange', linewidth=2.5)
    axes[1].set_title('Floquet Moment (order 1)\n(time-averaged)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(r'$\rho = K/d^2$', fontsize=12)
    axes[1].set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)

    # Panel 3: Floquet Order 2
    axes[2].plot(rho_values, P_floquet2, '^-', color='red', linewidth=2.5)
    axes[2].set_title('Floquet Moment (order 2)\n(with commutators, λ-dependent)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel(r'$\rho = K/d^2$', fontsize=12)
    axes[2].set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.05, 1.05)

    # Overall title
    fig.suptitle(f'GEO2 Floquet Engineering Comparison (d={d})',
                 fontsize=14, fontweight='bold', y=1.02)

    # Save
    output_file = output_dir / f'geo2_floquet_3panel_d{d}.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def print_summary_statistics(results: Dict):
    """Print summary statistics from the experiment."""
    data = results['data']
    config = results['config']

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    for d in sorted(data.keys()):
        print(f"\nDimension d={d}:")

        rho_values = sorted(data[d].keys())

        # Find crossing points (P ≈ 0.5)
        for criterion in ['regular_moment', 'floquet_moment_order1', 'floquet_moment_order2']:
            P_values = [data[d][rho][f'P_{criterion}'] for rho in rho_values]

            # Find approximate crossing
            crossing_idx = None
            for i, P in enumerate(P_values):
                if P >= 0.5:
                    crossing_idx = i
                    break

            if crossing_idx is not None:
                rho_c = rho_values[crossing_idx]
                K_c = data[d][rho_c]['K']
                print(f"  {criterion:30s}: ρ_c ≈ {rho_c:.3f}, K_c ≈ {K_c}")
            else:
                print(f"  {criterion:30s}: No crossing found (P < 0.5)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Plot GEO2 Floquet Engineering Results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('input_file', type=str,
                        help='Path to experimental results (.pkl file)')
    parser.add_argument('--output-dir', type=str, default='fig/geo2_floquet',
                        help='Output directory for plots')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics only (no plots)')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.input_file}")
    results = load_results(args.input_file)

    # Print summary
    print_summary_statistics(results)

    if args.summary:
        return 0

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots in: {output_dir}")

    # Generate plots
    plot_main_comparison(results, output_dir)
    plot_floquet_order_comparison(results, output_dir)
    plot_multi_dimension_overlay(results, output_dir)
    plot_3panel_comparison(results, output_dir)

    print("\nPlotting complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
