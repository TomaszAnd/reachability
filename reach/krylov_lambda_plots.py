"""
Explicit m(λ) visualizations for lambda-dependence analysis.

This module generates detailed plots showing how Krylov dimension m varies
as a function of lambda direction on the unit sphere S^{K-1}.

It is supposed to check whehter Krylov dimension depend on λ
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import krylov_comparison, mathematics, models

logger = logging.getLogger(__name__)


def sample_lambda_directions(
    K: int,
    n_lambda: int,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    Sample lambda vectors uniformly on unit sphere S^{K-1}.

    Samples i.i.d. N(0,1)^K and normalizes to unit L2 norm.
    Only direction matters; scaling is irrelevant for Krylov dimension.

    Args:
        K: Dimension of lambda space
        n_lambda: Number of samples
        rng: Random number generator

    Returns:
        Array of shape (n_lambda, K) with unit-norm rows
    """
    # Sample from N(0,1)^K
    lambdas = rng.randn(n_lambda, K)

    # Normalize each row to unit norm
    norms = np.linalg.norm(lambdas, axis=1, keepdims=True)
    lambdas = lambdas / norms

    return lambdas


def compute_m_lambda_single_set(
    method: str,
    d: int,
    K: int,
    n_lambda: int,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, List]:
    """
    Compute m(λ) for a single generator set {H_k}.

    Args:
        method: 'canonical' or 'projector'
        d: Dimension
        K: Number of Hamiltonians
        n_lambda: Number of lambda samples
        rng: Random number generator

    Returns:
        Tuple of (m_values, lambdas) where m_values is 1D array of Krylov dimensions
    """
    # Generate one fixed set of Hamiltonians
    if method == 'canonical':
        hams = krylov_comparison.generate_canonical_pauli_hamiltonian(
            dim=d, K=K, seed=rng.randint(0, 2**31 - 1)
        )
    elif method == 'projector':
        hams = krylov_comparison.generate_random_projector_hamiltonian(
            dim=d, K=K, seed=rng.randint(0, 2**31 - 1), rank=1
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Initial state |0⟩
    psi = models.fock_state(d, 0)

    # Sample lambda directions
    lambdas = sample_lambda_directions(K, n_lambda, rng)

    # Compute m(λ) for each direction
    m_values = np.zeros(n_lambda, dtype=int)

    for i, lam in enumerate(lambdas):
        # Construct H(λ)
        H_lambda = mathematics.construct_hamiltonian(lam, hams)

        # Compute Krylov dimension
        m = krylov_comparison.compute_krylov_dimension(H_lambda, psi)
        m_values[i] = m

    return m_values, lambdas


def plot_m_scatter(
    m_values: np.ndarray,
    method: str,
    d: int,
    K: int,
    outpath: Path,
    set_id: Optional[int] = None
):
    """Plot m vs sample index (scatter)."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    n_lambda = len(m_values)
    indices = np.arange(n_lambda)

    # Scatter plot
    ax.scatter(indices, m_values, alpha=0.6, s=30, c='#1f77b4', edgecolors='black', linewidth=0.5)

    # Horizontal line at d
    ax.axhline(y=d, color='red', linestyle='--', linewidth=2, label=f'd={d}', alpha=0.7)

    # Styling
    ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Krylov Dimension m(λ)', fontsize=14, fontweight='bold')

    method_label = 'Canonical Pauli' if method == 'canonical' else 'Random Projectors'
    set_label = f' (Set {set_id})' if set_id is not None else ''
    ax.set_title(
        f'Krylov Dimension m(λ) vs Sample Index\n'
        f'{method_label}, d={d}, K={K}{set_label}',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    # Statistics annotation
    mean_m = np.mean(m_values)
    std_m = np.std(m_values)
    min_m = np.min(m_values)
    max_m = np.max(m_values)

    stats_text = (
        f'N={n_lambda}\n'
        f'mean = {mean_m:.2f}\n'
        f'std = {std_m:.4f}\n'
        f'range = [{min_m}, {max_m}]'
    )

    if std_m < 1e-3:
        stats_text += '\n\nλ-independent\n(std < 1e-3)'
        box_color = 'lightgreen'
    else:
        stats_text += '\n\n⚠ λ-dependence\ndetected!'
        box_color = 'yellow'

    ax.text(
        0.98, 0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='black')
    )

    plt.tight_layout()

    set_suffix = f'_set{set_id}' if set_id is not None else ''
    filename = f'm_vs_lambda_scatter_{method}_d{d}_K{K}{set_suffix}.png'
    outfile = outpath / filename
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Scatter plot saved: {outfile}")


def plot_m_distribution(
    m_values: np.ndarray,
    method: str,
    d: int,
    K: int,
    outpath: Path,
    set_id: Optional[int] = None
):
    """Plot distribution of m(λ) with violin + box + histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200)

    # Left panel: Violin + Box
    ax1 = axes[0]

    # Violin plot
    parts = ax1.violinplot(
        [m_values],
        positions=[1],
        showmeans=True,
        showextrema=True,
        widths=0.7
    )

    # Color the violin
    for pc in parts['bodies']:
        pc.set_facecolor('#8dd3c7')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')

    # Overlay box plot
    bp = ax1.boxplot(
        [m_values],
        positions=[1],
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(color='red', linewidth=3)
    )

    ax1.set_ylabel('Krylov Dimension m(λ)', fontsize=14, fontweight='bold')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['m(λ) distribution'])
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--')

    # Horizontal line at d
    ax1.axhline(y=d, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Right panel: Histogram
    ax2 = axes[1]

    # Determine bins
    unique_vals = np.unique(m_values)
    if len(unique_vals) == 1:
        # All same value
        bins = [unique_vals[0] - 0.5, unique_vals[0] + 0.5]
    else:
        bins = np.arange(m_values.min() - 0.5, m_values.max() + 1.5, 1)

    counts, bins, patches = ax2.hist(
        m_values,
        bins=bins,
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7,
        color='#8dd3c7'
    )

    ax2.set_xlabel('Krylov Dimension m', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.25, axis='y', linestyle='--')

    # Vertical line at d
    ax2.axvline(x=d, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'd={d}')
    ax2.legend(fontsize=11)

    # Overall title
    method_label = 'Canonical Pauli' if method == 'canonical' else 'Random Projectors'
    set_label = f' (Set {set_id})' if set_id is not None else ''
    fig.suptitle(
        f'Distribution of m(λ) - {method_label}, d={d}, K={K}{set_label}',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )

    # Statistics annotation
    mean_m = np.mean(m_values)
    std_m = np.std(m_values)

    stats_text = (
        f'N = {len(m_values)}\n'
        f'mean = {mean_m:.2f}\n'
        f'std = {std_m:.4f}\n'
        f'min = {m_values.min()}\n'
        f'max = {m_values.max()}'
    )

    if std_m < 1e-3:
        stats_text += '\n\nλ-independent'
        box_color = 'lightgreen'
    else:
        stats_text += '\n\n⚠ λ-dependent'
        box_color = 'yellow'

    ax2.text(
        0.98, 0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='black')
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    set_suffix = f'_set{set_id}' if set_id is not None else ''
    filename = f'm_vs_lambda_dist_{method}_d{d}_K{K}{set_suffix}.png'
    outfile = outpath / filename
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Distribution plot saved: {outfile}")


def plot_m_violin_by_set(
    all_m_values: List[np.ndarray],
    method: str,
    d: int,
    K: int,
    outpath: Path
):
    """Plot violin by generator set showing per-set distribution."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)

    R = len(all_m_values)
    positions = np.arange(1, R + 1)

    # Violin plot
    parts = ax.violinplot(
        all_m_values,
        positions=positions,
        showmeans=True,
        showextrema=True,
        widths=0.7
    )

    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#8dd3c7')
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')

    # Overlay box plots
    bp = ax.boxplot(
        all_m_values,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(color='red', linewidth=2)
    )

    # Horizontal line at d
    ax.axhline(y=d, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'd={d}')

    # Styling
    ax.set_xlabel('Generator Set ID', fontsize=14, fontweight='bold')
    ax.set_ylabel('Krylov Dimension m(λ)', fontsize=14, fontweight='bold')

    method_label = 'Canonical Pauli' if method == 'canonical' else 'Random Projectors'
    ax.set_title(
        f'Krylov Dimension m(λ) Across Generator Sets\n'
        f'{method_label}, d={d}, K={K}, R={R} sets',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{i}' for i in positions])
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.legend(fontsize=11, loc='best')

    # Count sets with any variation
    stds = [np.std(m_vals) for m_vals in all_m_values]
    n_variable = sum(1 for s in stds if s > 1e-3)

    # Annotation
    if n_variable == 0:
        annot_text = f'✓ All {R} sets λ-independent\n(std < 1e-3)'
        box_color = 'lightgreen'
    else:
        annot_text = f'⚠ {n_variable}/{R} sets show\nλ-dependence (std ≥ 1e-3)'
        box_color = 'yellow'

    ax.text(
        0.02, 0.98,
        annot_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='black', linewidth=2)
    )

    plt.tight_layout()

    filename = f'm_vs_lambda_violin_byset_{method}_d{d}_K{K}.png'
    outfile = outpath / filename
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Violin-by-set plot saved: {outfile}")


def plot_m_vs_theta(
    method: str,
    d: int,
    K: int,
    n_theta: int,
    rng: np.random.RandomState,
    outpath: Path
):
    """
    Plot m(θ) for K=2 case where λ = (cos θ, sin θ).

    Only works for K=2.
    """
    if K != 2:
        logger.warning(f"1-D slice only implemented for K=2, got K={K}")
        return

    # Generate Hamiltonians
    if method == 'canonical':
        hams = krylov_comparison.generate_canonical_pauli_hamiltonian(
            dim=d, K=K, seed=rng.randint(0, 2**31 - 1)
        )
    elif method == 'projector':
        hams = krylov_comparison.generate_random_projector_hamiltonian(
            dim=d, K=K, seed=rng.randint(0, 2**31 - 1), rank=1
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Initial state |0⟩
    psi = models.fock_state(d, 0)

    # Sweep theta from 0 to 2π
    thetas = np.linspace(0, 2 * np.pi, n_theta)
    m_values = np.zeros(n_theta, dtype=int)

    for i, theta in enumerate(thetas):
        # λ = (cos θ, sin θ)
        lam = np.array([np.cos(theta), np.sin(theta)])

        # Construct H(λ)
        H_lambda = mathematics.construct_hamiltonian(lam, hams)

        # Compute Krylov dimension
        m = krylov_comparison.compute_krylov_dimension(H_lambda, psi)
        m_values[i] = m

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    ax.plot(thetas, m_values, 'o-', linewidth=2, markersize=4, alpha=0.7, color='#1f77b4')

    # Horizontal line at d
    ax.axhline(y=d, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'd={d}')

    # Styling
    ax.set_xlabel('θ (radians)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Krylov Dimension m(θ)', fontsize=14, fontweight='bold')

    method_label = 'Canonical Pauli' if method == 'canonical' else 'Random Projectors'
    ax.set_title(
        f'Krylov Dimension m(θ) for λ = (cos θ, sin θ)\n'
        f'{method_label}, d={d}, K={K}',
        fontsize=15,
        fontweight='bold',
        pad=15
    )

    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    # Set x-axis ticks at multiples of π/2
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    # Statistics annotation
    mean_m = np.mean(m_values)
    std_m = np.std(m_values)

    stats_text = (
        f'N = {n_theta}\n'
        f'mean = {mean_m:.2f}\n'
        f'std = {std_m:.4f}'
    )

    if std_m < 1e-3:
        stats_text += '\n\nλ-independent'
        box_color = 'lightgreen'
    else:
        stats_text += '\n\n⚠ λ-dependent'
        box_color = 'yellow'

    ax.text(
        0.98, 0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, edgecolor='black')
    )

    plt.tight_layout()

    filename = f'm_vs_theta_{method}_d{d}_K2.png'
    outfile = outpath / filename
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Theta-slice plot saved: {outfile}")


def run_m_lambda_analysis(
    methods: List[str],
    dims: List[int],
    Ks: List[int],
    n_lambda: int = 1000,
    repeats: int = 10,
    do_slice: bool = False,
    seed: int = 123,
    outdir: str = 'fig_summary'
) -> Dict:
    """
    Run complete m(λ) analysis for all (method, d, K) combinations.

    Args:
        methods: List of methods to test
        dims: List of dimensions
        Ks: List of K values
        n_lambda: Number of lambda samples per set
        repeats: Number of generator sets to test (R)
        do_slice: Whether to do 1-D theta slice for K=2
        seed: Random seed
        outdir: Output directory

    Returns:
        Summary dictionary with statistics
    """
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    # Summary storage
    summary = {
        'total_combinations': 0,
        'combinations_with_variability': 0,
        'details': []
    }

    total_combos = len(methods) * len(dims) * len(Ks)
    combo_idx = 0

    for method in methods:
        # Validate dimensions for canonical
        if method == 'canonical':
            for d in dims:
                if d < 2 or (d & (d - 1)) != 0:
                    raise ValueError(
                        f"Canonical method requires power-of-2 dimensions, got d={d}"
                    )

        logger.info(f"\n{'='*80}")
        logger.info(f"METHOD: {method.upper()}")
        logger.info(f"{'='*80}\n")

        for d in dims:
            for K in Ks:
                combo_idx += 1
                logger.info(
                    f"[{combo_idx}/{total_combos}] Processing {method}, d={d}, K={K}"
                )

                # Collect m(λ) for R generator sets
                all_m_values = []
                set_stats = []

                for set_id in range(repeats):
                    # Compute m(λ) for this set
                    m_vals, lambdas = compute_m_lambda_single_set(
                        method, d, K, n_lambda, rng
                    )
                    all_m_values.append(m_vals)

                    # Statistics
                    mean_m = np.mean(m_vals)
                    std_m = np.std(m_vals)
                    min_m = np.min(m_vals)
                    max_m = np.max(m_vals)
                    all_equal = (min_m == max_m == d)

                    set_stats.append({
                        'set_id': set_id,
                        'mean_m': mean_m,
                        'std_m': std_m,
                        'min_m': min_m,
                        'max_m': max_m,
                        'all_equal_to_d': all_equal
                    })

                    logger.debug(
                        f"  Set {set_id}: mean={mean_m:.2f}, std={std_m:.4f}, "
                        f"range=[{min_m}, {max_m}]"
                    )

                # Check for variability across sets
                stds = [s['std_m'] for s in set_stats]
                has_variability = any(s > 1e-3 for s in stds)
                n_variable_sets = sum(1 for s in stds if s > 1e-3)

                if has_variability:
                    summary['combinations_with_variability'] += 1
                    logger.warning(
                        f"⚠ λ-dependence detected for {method}, d={d}, K={K}: "
                        f"{n_variable_sets}/{repeats} sets show std > 1e-3"
                    )
                else:
                    logger.info(
                        f"✓ λ-independent for {method}, d={d}, K={K}: "
                        f"all {repeats} sets have std < 1e-3"
                    )

                summary['total_combinations'] += 1
                summary['details'].append({
                    'method': method,
                    'd': d,
                    'K': K,
                    'n_variable_sets': n_variable_sets,
                    'total_sets': repeats
                })

                # Generate plots
                # 1. Scatter for first set
                plot_m_scatter(all_m_values[0], method, d, K, outpath, set_id=0)

                # 2. Distribution for first set
                plot_m_distribution(all_m_values[0], method, d, K, outpath, set_id=0)

                # 3. Violin by set
                plot_m_violin_by_set(all_m_values, method, d, K, outpath)

                # 4. 1-D slice if K=2 and requested
                if do_slice and K == 2:
                    plot_m_vs_theta(method, d, K, n_theta=721, rng=rng, outpath=outpath)

                # 5. Write CSV summary
                csv_path = outpath / f'm_vs_lambda_summary_{method}_d{d}_K{K}.csv'
                with open(csv_path, 'w', newline='') as f:
                    fieldnames = [
                        'method', 'd', 'K', 'set_id', 'N_lambda',
                        'mean_m', 'std_m', 'min_m', 'max_m', 'all_equal_to_d'
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for stats in set_stats:
                        writer.writerow({
                            'method': method,
                            'd': d,
                            'K': K,
                            'set_id': stats['set_id'],
                            'N_lambda': n_lambda,
                            'mean_m': f"{stats['mean_m']:.4f}",
                            'std_m': f"{stats['std_m']:.6f}",
                            'min_m': stats['min_m'],
                            'max_m': stats['max_m'],
                            'all_equal_to_d': stats['all_equal_to_d']
                        })

                logger.info(f"  ✓ CSV summary saved: {csv_path}")

    return summary


def print_summary_report(summary: Dict):
    """Print summary report of lambda dependence analysis."""
    logger.info("\n" + "="*80)
    logger.info("M(λ) ANALYSIS SUMMARY")
    logger.info("="*80 + "\n")

    total = summary['total_combinations']
    with_var = summary['combinations_with_variability']

    logger.info(f"Total (method, d, K) combinations tested: {total}")
    logger.info(f"Combinations showing λ-dependence: {with_var}")
    logger.info(f"Combinations with λ-independence: {total - with_var}")

    if with_var == 0:
        logger.info("\n✓✓✓ ALL COMBINATIONS ARE λ-INDEPENDENT ✓✓✓")
    else:
        logger.warning(f"\n⚠ {with_var} combinations show λ-dependence:")
        for detail in summary['details']:
            if detail['n_variable_sets'] > 0:
                logger.warning(
                    f"  - {detail['method']}, d={detail['d']}, K={detail['K']}: "
                    f"{detail['n_variable_sets']}/{detail['total_sets']} sets variable"
                )

    logger.info("")
