#!/usr/bin/env python3
"""
CLI runner for Krylov criterion comparison experiments.

This module provides command-line interface for running:
1. Lambda-dependence grid experiments (Question 2)
2. Criteria comparison experiments (Question 3)

Usage:
    python -m reach.krylov_cli run-lambda-grid --methods a,b --dims 8,16,32,64 ...
    python -m reach.krylov_cli run-criteria-a --dims 8,16,32,64 --taus 0.90,0.95 ...
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import krylov_comparison, krylov_lambda_plots, mathematics, models, optimize, settings

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging with timestamp and regular flushing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Force flush every message
    for handler in logging.root.handlers:
        handler.setLevel(logging.INFO)


def ensure_output_dir(outdir: str = 'fig_summary') -> Path:
    """Create output directory if it doesn't exist."""
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)
    return outpath


def validate_power_of_2(dims: List[int], method: str = 'canonical') -> None:
    """
    Validate that dimensions are powers of 2 for Pauli method.

    Args:
        dims: List of dimensions to validate
        method: Generation method ('canonical' requires power of 2)

    Raises:
        ValueError: If method is canonical and any dim is not power of 2
    """
    if method != 'canonical':
        return

    for d in dims:
        if d < 2 or (d & (d - 1)) != 0:
            # Find nearest powers of 2
            lower = 1 << (d.bit_length() - 1)
            upper = 1 << d.bit_length()
            raise ValueError(
                f"Canonical Pauli method requires d=2^n dimensions. "
                f"Got d={d}. Nearest valid: {lower} or {upper}"
            )


def run_lambda_dependence_grid(
    methods: List[str],
    dims: List[int],
    Ks: List[int],
    n_lambda: int = 200,
    seed: int = 42,
    outdir: str = 'fig_summary'
) -> Dict[str, np.ndarray]:
    """
    Run lambda-dependence grid experiment (Question 2).

    For each (method, d, K) combination:
    - Generate K Hamiltonians
    - Sample n_lambda random λ vectors
    - Compute Krylov dimension m(λ) for each
    - Compute statistics: mean, std, min, max
    - Determine if lambda-independent (std < 1e-3)

    Args:
        methods: List of methods ('canonical' and/or 'projector')
        dims: List of dimensions to test
        Ks: List of K values to test
        n_lambda: Number of lambda samples per point
        seed: Random seed
        outdir: Output directory for CSV and plots

    Returns:
        Dictionary with results keyed by method
    """
    outpath = ensure_output_dir(outdir)
    csv_path = outpath / 'krylov_lambda_dependence_grid.csv'

    # Validate dimensions for canonical method
    if 'canonical' in methods:
        validate_power_of_2(dims, 'canonical')

    rng = np.random.RandomState(seed)

    # CSV header
    csv_exists = csv_path.exists()
    fieldnames = [
        'method', 'd', 'K', 'n_lambda', 'mean_m', 'std_m', 'min_m', 'max_m',
        'lambda_independent', 'seed', 'timestamp'
    ]

    # Open CSV for appending
    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    # Storage for heatmap data
    results = {method: np.zeros((len(dims), len(Ks))) for method in methods}

    total_points = len(methods) * len(dims) * len(Ks)
    point_idx = 0

    try:
        for method_idx, method in enumerate(methods):
            logger.info(f"\n{'='*80}")
            logger.info(f"METHOD: {method.upper()}")
            logger.info(f"{'='*80}\n")

            for d_idx, d in enumerate(dims):
                for k_idx, K in enumerate(Ks):
                    point_idx += 1
                    start_time = time.time()

                    logger.info(
                        f"[λ-grid {point_idx}/{total_points}] "
                        f"method={method}, d={d}, K={K}..."
                    )

                    # Generate Hamiltonians
                    try:
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
                    except ValueError as e:
                        logger.error(f"Failed to generate Hamiltonians: {e}")
                        continue

                    # Initial state |0⟩
                    psi = models.fock_state(d, 0)

                    # Sample lambda vectors and compute Krylov dimensions
                    krylov_dims = []
                    for trial in range(n_lambda):
                        # Sample λ from N(0, 1)
                        lambdas = rng.randn(K)

                        # Construct H(λ)
                        H_lambda = mathematics.construct_hamiltonian(lambdas, hams)

                        # Compute Krylov dimension
                        m = krylov_comparison.compute_krylov_dimension(H_lambda, psi)
                        krylov_dims.append(m)

                    # Compute statistics
                    krylov_dims = np.array(krylov_dims)
                    mean_m = float(np.mean(krylov_dims))
                    std_m = float(np.std(krylov_dims))
                    min_m = int(np.min(krylov_dims))
                    max_m = int(np.max(krylov_dims))
                    lambda_independent = std_m < 1e-3

                    elapsed = time.time() - start_time

                    # Log result
                    logger.info(
                        f"[λ-grid] method={method[0]}, d={d}, K={K}: "
                        f"mean={mean_m:.2f}, std={std_m:.4f}, min={min_m}, max={max_m}, "
                        f"indep={lambda_independent} (elapsed {elapsed:.1f}s)"
                    )

                    # Store for heatmap
                    results[method][d_idx, k_idx] = std_m

                    # Write to CSV
                    row = {
                        'method': method,
                        'd': d,
                        'K': K,
                        'n_lambda': n_lambda,
                        'mean_m': f'{mean_m:.4f}',
                        'std_m': f'{std_m:.6f}',
                        'min_m': min_m,
                        'max_m': max_m,
                        'lambda_independent': lambda_independent,
                        'seed': seed,
                        'timestamp': datetime.now().isoformat()
                    }
                    writer.writerow(row)
                    csv_file.flush()  # Flush immediately

    finally:
        csv_file.close()

    logger.info(f"\n✓ Lambda-dependence grid CSV written to {csv_path}")

    # Generate heatmaps
    for method in methods:
        plot_lambda_std_heatmap(
            results[method],
            dims=dims,
            Ks=Ks,
            method=method,
            outpath=outpath
        )

    return results


def plot_lambda_std_heatmap(
    std_grid: np.ndarray,
    dims: List[int],
    Ks: List[int],
    method: str,
    outpath: Path
):
    """
    Plot heatmap of std(m) across (d, K) grid.

    Args:
        std_grid: 2D array of std(m) values, shape (len(dims), len(Ks))
        dims: List of dimension values
        Ks: List of K values
        method: Method name for filename
        outpath: Output directory path
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Create heatmap
    sns.heatmap(
        std_grid,
        annot=True,
        fmt='.3g',  # 3 significant figures
        cmap='YlOrRd',
        xticklabels=[f'K={k}' for k in Ks],
        yticklabels=[f'd={d}' for d in dims],
        cbar_kws={'label': 'std(m)'},
        ax=ax,
        vmin=0,
        linewidths=0.5,
        linecolor='gray'
    )

    # Styling
    ax.set_xlabel('Number of Hamiltonians', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension', fontsize=12, fontweight='bold')

    method_label = 'Canonical Pauli Basis' if method == 'canonical' else 'Random Projectors'
    ax.set_title(
        f'Krylov Dimension Std Dev - {method_label}\n'
        f'(λ-independent ≡ std < 1e-3)',
        fontsize=13,
        fontweight='bold',
        pad=15
    )

    # Add legend text
    ax.text(
        0.02, 0.98,
        'λ-independent = std < 1e-3',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    )

    plt.tight_layout()

    filename = f'krylov_lambda_std_heatmap_{method}.png'
    outfile = outpath / filename
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Heatmap saved: {outfile}")


def run_criteria_comparison(
    dims: List[int],
    Ks: List[int],
    taus: List[float],
    trials: int = 200,
    lambda_mode: str = 'uniform',
    seed: int = 42,
    outdir: str = 'fig_summary'
) -> Dict[str, Dict]:
    """
    Run criteria comparison for canonical Pauli method (Question 3).

    Compares three criteria across (d, K) parameter space:
    - Spectral (for each τ)
    - Moment
    - Krylov

    Args:
        dims: List of dimensions (must be powers of 2)
        Ks: List of K values
        taus: List of threshold values for spectral criterion
        trials: Number of Monte Carlo trials per point
        lambda_mode: 'uniform' or 'seeded-normal' for fixed lambda
        seed: Random seed
        outdir: Output directory

    Returns:
        Dictionary with results by criterion
    """
    outpath = ensure_output_dir(outdir)
    csv_path = outpath / 'criteria_comparison_dK.csv'

    # Validate dimensions
    validate_power_of_2(dims, 'canonical')

    rng = np.random.RandomState(seed)

    # CSV header
    csv_exists = csv_path.exists()
    fieldnames = [
        'criterion', 'tau', 'd', 'K', 'trials', 'unreachable_count',
        'p_unreach', 'sem_unreach', 'mean_S_star', 'std_S_star',
        'lambda_mode', 'seed', 'timestamp'
    ]

    csv_file = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    # Storage for results
    results = {
        'spectral': {},
        'moment': {},
        'krylov': {}
    }

    total_points = len(dims) * len(Ks)
    point_idx = 0

    try:
        for d in dims:
            for K in Ks:
                point_idx += 1
                logger.info(
                    f"\n[crit {point_idx}/{total_points}] d={d}, K={K}"
                )

                # Run trials for this (d, K) point
                for tau in taus:
                    run_criterion_point(
                        d, K, tau, trials, lambda_mode, rng, writer, csv_file, results
                    )

    finally:
        csv_file.close()

    logger.info(f"\n✓ Criteria comparison CSV written to {csv_path}")

    # Generate plots
    generate_criteria_plots(results, dims, Ks, taus, outpath)

    return results


def run_criterion_point(
    d: int,
    K: int,
    tau: float,
    trials: int,
    lambda_mode: str,
    rng: np.random.RandomState,
    writer: csv.DictWriter,
    csv_file,
    results: Dict
):
    """Run all three criteria for a single (d, K, tau) point."""

    # Counters
    spectral_unreachable = 0
    moment_unreachable = 0
    krylov_unreachable = 0
    S_star_values = []

    start_time = time.time()

    for trial in range(trials):
        # Generate Hamiltonians
        hams = krylov_comparison.generate_canonical_pauli_hamiltonian(
            dim=d, K=K, seed=rng.randint(0, 2**31 - 1)
        )

        # Fixed lambda weights
        if lambda_mode == 'uniform':
            lambdas = np.ones(K) / np.sqrt(K)  # Unit norm
        elif lambda_mode == 'seeded-normal':
            lambdas = rng.randn(K)
            lambdas /= np.linalg.norm(lambdas)  # Normalize
        else:
            raise ValueError(f"Unknown lambda_mode: {lambda_mode}")

        # States: |0⟩ → |d-1⟩
        psi = models.fock_state(d, 0)
        phi = models.fock_state(d, d - 1)

        # --- Spectral criterion ---
        opt_result = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            method='L-BFGS-B',
            restarts=1,
            maxiter=100,
            seed=rng.randint(0, 2**31 - 1)
        )
        S_star = opt_result['best_value']
        S_star_values.append(S_star)

        if S_star < tau:
            spectral_unreachable += 1

        # --- Moment criterion ---
        is_reachable_moment = krylov_comparison.moment_criterion(psi, phi, hams)
        if not is_reachable_moment:
            moment_unreachable += 1

        # --- Krylov criterion ---
        is_reachable_krylov = krylov_comparison.krylov_criterion(
            psi, phi, hams, lambdas=lambdas
        )
        if not is_reachable_krylov:
            krylov_unreachable += 1

    elapsed = time.time() - start_time

    # Compute statistics
    p_unreach_spectral = spectral_unreachable / trials
    sem_spectral = mathematics.compute_binomial_sem(p_unreach_spectral, trials)
    mean_S = np.mean(S_star_values)
    std_S = np.std(S_star_values)

    p_unreach_moment = moment_unreachable / trials
    sem_moment = mathematics.compute_binomial_sem(p_unreach_moment, trials)

    p_unreach_krylov = krylov_unreachable / trials
    sem_krylov = mathematics.compute_binomial_sem(p_unreach_krylov, trials)

    # Log
    logger.info(
        f"[crit] d={d}, K={K}, spectral τ={tau:.2f} → "
        f"p_unreach={p_unreach_spectral:.2f}±{sem_spectral:.3f}, "
        f"mean S*={mean_S:.2f}, T={trials} (elapsed {elapsed:.1f}s)"
    )
    logger.info(
        f"[crit] d={d}, K={K}, moment → p_unreach={p_unreach_moment:.2f}±{sem_moment:.3f}"
    )
    logger.info(
        f"[crit] d={d}, K={K}, krylov → p_unreach={p_unreach_krylov:.2f}±{sem_krylov:.3f}"
    )

    # Store results
    key = (d, K, tau)
    results['spectral'][key] = {
        'p_unreach': p_unreach_spectral,
        'sem': sem_spectral,
        'mean_S': mean_S,
        'std_S': std_S
    }
    results['moment'][(d, K)] = {
        'p_unreach': p_unreach_moment,
        'sem': sem_moment
    }
    results['krylov'][(d, K)] = {
        'p_unreach': p_unreach_krylov,
        'sem': sem_krylov
    }

    # Write CSV rows
    # Spectral
    writer.writerow({
        'criterion': 'spectral',
        'tau': tau,
        'd': d,
        'K': K,
        'trials': trials,
        'unreachable_count': spectral_unreachable,
        'p_unreach': f'{p_unreach_spectral:.6f}',
        'sem_unreach': f'{sem_spectral:.6f}',
        'mean_S_star': f'{mean_S:.6f}',
        'std_S_star': f'{std_S:.6f}',
        'lambda_mode': lambda_mode,
        'seed': rng.randint(0, 2**31 - 1),
        'timestamp': datetime.now().isoformat()
    })
    csv_file.flush()

    # Moment (only once per d, K)
    if tau == min(results['spectral'].keys(), key=lambda x: x[2])[2]:  # First tau
        writer.writerow({
            'criterion': 'moment',
            'tau': '',
            'd': d,
            'K': K,
            'trials': trials,
            'unreachable_count': moment_unreachable,
            'p_unreach': f'{p_unreach_moment:.6f}',
            'sem_unreach': f'{sem_moment:.6f}',
            'mean_S_star': '',
            'std_S_star': '',
            'lambda_mode': lambda_mode,
            'seed': rng.randint(0, 2**31 - 1),
            'timestamp': datetime.now().isoformat()
        })
        csv_file.flush()

        # Krylov
        writer.writerow({
            'criterion': 'krylov',
            'tau': '',
            'd': d,
            'K': K,
            'trials': trials,
            'unreachable_count': krylov_unreachable,
            'p_unreach': f'{p_unreach_krylov:.6f}',
            'sem_unreach': f'{sem_krylov:.6f}',
            'mean_S_star': '',
            'std_S_star': '',
            'lambda_mode': lambda_mode,
            'seed': rng.randint(0, 2**31 - 1),
            'timestamp': datetime.now().isoformat()
        })
        csv_file.flush()


def generate_criteria_plots(
    results: Dict,
    dims: List[int],
    Ks: List[int],
    taus: List[float],
    outpath: Path,
    add_krylov_annotation: bool = True,
    n_lambda: int = 500,
    n_repeats: int = 8
):
    """Generate all plots for criteria comparison.

    Args:
        add_krylov_annotation: If True, add λ-independence verification annotation
        n_lambda: Number of λ samples used in verification (for annotation)
        n_repeats: Number of generator sets tested (for annotation)
    """

    # 1. Small multiples: one figure per d, showing all criteria vs K
    for d in dims:
        plot_criteria_vs_K(results, d, Ks, taus, outpath)

    # 2. Heatmaps: one per criterion (spectral has multiple for each tau)
    for tau in taus:
        plot_criterion_heatmap(
            results, 'spectral', dims, Ks, tau, outpath,
            add_krylov_annotation, n_lambda, n_repeats
        )

    plot_criterion_heatmap(
        results, 'moment', dims, Ks, None, outpath,
        add_krylov_annotation, n_lambda, n_repeats
    )
    plot_criterion_heatmap(
        results, 'krylov', dims, Ks, None, outpath,
        add_krylov_annotation, n_lambda, n_repeats
    )


def plot_criteria_vs_K(
    results: Dict,
    d: int,
    Ks: List[int],
    taus: List[float],
    outpath: Path
):
    """Plot all criteria vs K for a fixed dimension d."""

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

    # Extract data
    K_vals = np.array(Ks)

    # Spectral for each tau
    colors_spectral = ['#1f77b4', '#ff7f0e']  # Two hues
    for i, tau in enumerate(taus):
        p_vals = []
        err_vals = []
        for K in Ks:
            key = (d, K, tau)
            if key in results['spectral']:
                p_vals.append(results['spectral'][key]['p_unreach'])
                err_vals.append(results['spectral'][key]['sem'])
            else:
                p_vals.append(0)
                err_vals.append(0)

        ax.errorbar(
            K_vals, p_vals, yerr=err_vals,
            fmt='o-',
            linewidth=2.5,
            markersize=7,
            capsize=5,
            capthick=2,
            label=f'Spectral (τ={tau:.2f})',
            color=colors_spectral[i]
        )

    # Moment
    p_vals_moment = []
    err_vals_moment = []
    for K in Ks:
        key = (d, K)
        if key in results['moment']:
            p_vals_moment.append(results['moment'][key]['p_unreach'])
            err_vals_moment.append(results['moment'][key]['sem'])
        else:
            p_vals_moment.append(0)
            err_vals_moment.append(0)

    ax.errorbar(
        K_vals, p_vals_moment, yerr=err_vals_moment,
        fmt='s--',
        linewidth=2.5,
        markersize=7,
        capsize=5,
        capthick=2,
        label='Moment',
        color='#2ca02c'
    )

    # Krylov
    p_vals_krylov = []
    err_vals_krylov = []
    for K in Ks:
        key = (d, K)
        if key in results['krylov']:
            p_vals_krylov.append(results['krylov'][key]['p_unreach'])
            err_vals_krylov.append(results['krylov'][key]['sem'])
        else:
            p_vals_krylov.append(0)
            err_vals_krylov.append(0)

    ax.errorbar(
        K_vals, p_vals_krylov, yerr=err_vals_krylov,
        fmt='^:',
        linewidth=2.5,
        markersize=7,
        capsize=5,
        capthick=2,
        label='Krylov',
        color='#d62728'
    )

    # Styling
    ax.set_xlabel('Number of Hamiltonians (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unreachability Probability', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Criteria Comparison - d={d} (Canonical Pauli)\n|0⟩ → |{d-1}⟩',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(K_vals)

    plt.tight_layout()

    filename = f'criteria_vs_K_d{d}_methodA.png'
    outfile = outpath / filename
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Small-multiples plot saved: {outfile}")


def plot_criterion_heatmap(
    results: Dict,
    criterion: str,
    dims: List[int],
    Ks: List[int],
    tau: Optional[float],
    outpath: Path,
    add_krylov_annotation: bool = True,
    n_lambda: int = 500,
    n_repeats: int = 8
):
    """Plot heatmap of p_unreach for a single criterion.

    Args:
        add_krylov_annotation: If True, add annotation about λ-independence verification
            to Krylov and Moment heatmaps (membership criteria)
        n_lambda: Number of λ samples used in verification (for annotation)
        n_repeats: Number of generator sets tested (for annotation)
    """

    # Build grid
    grid = np.zeros((len(dims), len(Ks)))

    for i, d in enumerate(dims):
        for j, K in enumerate(Ks):
            if criterion == 'spectral':
                key = (d, K, tau)
                if key in results[criterion]:
                    grid[i, j] = results[criterion][key]['p_unreach']
            else:
                key = (d, K)
                if key in results[criterion]:
                    grid[i, j] = results[criterion][key]['p_unreach']

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    sns.heatmap(
        grid,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',  # Red = unreachable, Green = reachable
        xticklabels=[f'K={k}' for k in Ks],
        yticklabels=[f'd={d}' for d in dims],
        cbar_kws={'label': 'P(unreachable)'},
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )

    # Title
    if criterion == 'spectral':
        title = f'Spectral Criterion (τ={tau:.2f}) - Unreachability Heatmap'
    else:
        title = f'{criterion.capitalize()} Criterion - Unreachability Heatmap'

    ax.set_title(
        f'{title}\nCanonical Pauli Method, |0⟩ → |d-1⟩',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('Number of Hamiltonians', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension', fontsize=12, fontweight='bold')

    # Add λ-independence annotation for membership criteria (Krylov and Moment)
    if add_krylov_annotation and criterion in ['krylov', 'moment']:
        annotation_text = (
            f'Krylov λ-indep. verified on $S^{{K-1}}$\n'
            f'$N_\\lambda={n_lambda}$, $R={n_repeats}$, tolerance=$10^{{-3}}$'
        )
        ax.text(
            0.02, 0.98,
            annotation_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5)
        )

    plt.tight_layout()

    if criterion == 'spectral':
        filename = f'heatmap_spectral_methodA_tau{tau:.2f}.png'
    else:
        filename = f'heatmap_{criterion}_methodA.png'

    outfile = outpath / filename
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Heatmap saved: {outfile}")


def print_lambda_summary(csv_path: Path, methods: List[str]):
    """Print summary of lambda-dependence results."""
    import pandas as pd

    logger.info("\n" + "="*80)
    logger.info("LAMBDA-DEPENDENCE SUMMARY")
    logger.info("="*80 + "\n")

    df = pd.read_csv(csv_path)

    for method in methods:
        method_df = df[df['method'] == method]
        total = len(method_df)
        independent = (method_df['lambda_independent'] == True).sum()
        fraction = independent / total if total > 0 else 0

        logger.info(f"{method.upper():>12s}: {independent}/{total} points λ-independent ({fraction:.1%})")

        if independent == total:
            logger.info(f"             ✓ ALL points are λ-independent (std < 1e-3)")
        elif independent == 0:
            logger.info(f"             ✗ NO points are λ-independent")
        else:
            logger.info(f"             ⚠ MIXED results")

    logger.info("")


def print_criteria_summary(csv_path: Path, dims: List[int]):
    """Print summary of criteria comparison results."""
    import pandas as pd

    logger.info("\n" + "="*80)
    logger.info("CRITERIA COMPARISON SUMMARY")
    logger.info("="*80 + "\n")

    df = pd.read_csv(csv_path)

    for d in dims:
        d_df = df[df['d'] == d]

        # Find K where criteria disagree the most
        Ks = sorted(d_df['K'].unique())
        max_diff = 0
        max_diff_K = None

        for K in Ks:
            # Get probabilities for this (d, K)
            spectral_df = d_df[(d_df['K'] == K) & (d_df['criterion'] == 'spectral')]
            moment_df = d_df[(d_df['K'] == K) & (d_df['criterion'] == 'moment')]
            krylov_df = d_df[(d_df['K'] == K) & (d_df['criterion'] == 'krylov')]

            if len(spectral_df) > 0 and len(moment_df) > 0 and len(krylov_df) > 0:
                # Compare spectral (max tau) vs others
                p_spectral = spectral_df['p_unreach'].max()
                p_moment = moment_df['p_unreach'].values[0]
                p_krylov = krylov_df['p_unreach'].values[0]

                diff = max(
                    abs(p_spectral - p_moment),
                    abs(p_spectral - p_krylov),
                    abs(p_moment - p_krylov)
                )

                if diff > max_diff:
                    max_diff = diff
                    max_diff_K = K

        logger.info(
            f"d={d:3d}: Max disagreement at K={max_diff_K} (Δp={max_diff:.3f})"
        )


def run_m_vs_lambda_diagnostic(
    methods: List[str],
    test_points: List[Tuple[int, int]],
    n_lambda: int = 500,
    seed: int = 42,
    outdir: str = 'fig_summary'
) -> Dict[str, Dict[Tuple[int, int], Dict]]:
    """
    Run m-vs-lambda diagnostic for specific (d,K) test points.

    For each method and (d,K) pair:
    - Sample n_lambda uniformly on S^{K-1}
    - Compute m(λ) for each sample
    - Save compact CSV
    - Generate 2×2 subplot figure

    Args:
        methods: List of method names ('canonical', 'projector')
        test_points: List of (d, K) tuples to test
        n_lambda: Number of lambda samples per point
        seed: Random seed
        outdir: Output directory

    Returns:
        Dictionary with statistics per method and (d,K)
    """
    outpath = ensure_output_dir(outdir)

    logger.info("\n" + "="*80)
    logger.info("M-VS-LAMBDA DIAGNOSTIC")
    logger.info("="*80)
    logger.info(f"Methods: {methods}")
    logger.info(f"Test points (d,K): {test_points}")
    logger.info(f"Lambda samples: {n_lambda}")
    logger.info(f"Seed: {seed}")

    results = {}

    for method in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"METHOD: {method.upper()}")
        logger.info(f"{'='*80}")

        results[method] = {}

        # Prepare figure with 2×2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
        axes = axes.flatten()

        for idx, (d, K) in enumerate(test_points):
            logger.info(f"\n[{idx+1}/{len(test_points)}] Testing d={d}, K={K}")

            # Generate Hamiltonians
            rng = np.random.RandomState(seed + idx)

            if method == 'canonical':
                hams = krylov_comparison.generate_canonical_pauli_hamiltonian(
                    d, K, seed=seed + idx, allow_single_site=True
                )
            elif method == 'projector':
                hams = krylov_comparison.generate_random_projector_hamiltonian(
                    d, K, seed=seed + idx, rank=1
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Sample lambda directions
            lambdas = rng.randn(n_lambda, K)
            lambdas = lambdas / np.linalg.norm(lambdas, axis=1, keepdims=True)

            # Compute m(λ) for each sample
            m_values = []
            for lamb in lambdas:
                # Construct H(λ) = sum_k λ_k H_k
                H_lambda = sum(lamb[k] * hams[k] for k in range(K))

                # Initial state
                psi = models.fock_state(d, 0)

                # Compute Krylov dimension
                m = krylov_comparison.compute_krylov_dimension(H_lambda, psi)
                m_values.append(m)

            m_values = np.array(m_values)

            # Statistics
            m_min = m_values.min()
            m_max = m_values.max()
            m_mean = m_values.mean()
            m_std = m_values.std()

            logger.info(f"  m: min={m_min}, max={m_max}, mean={m_mean:.3f}, std={m_std:.6f}")

            # Save CSV
            csv_filename = f'm_vs_lambda_d{d}_K{K}_{method}.csv'
            csv_path = outpath / csv_filename

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['idx', 'm', 'd', 'K', 'method'])
                for i, m in enumerate(m_values, start=1):
                    writer.writerow([i, int(m), d, K, method])

            logger.info(f"  ✓ CSV saved: {csv_filename}")

            # Store results
            results[method][(d, K)] = {
                'm_values': m_values,
                'min': m_min,
                'max': m_max,
                'mean': m_mean,
                'std': m_std,
                'lambda_independent': m_std < 1e-3
            }

            # Plot in subplot
            ax = axes[idx]

            sample_indices = np.arange(1, n_lambda + 1)
            ax.plot(sample_indices, m_values, 'o', markersize=3, alpha=0.6, color='steelblue')
            ax.axhline(y=d, color='red', linestyle='--', linewidth=2, label=f'm = d = {d}')

            # Styling
            ax.set_xlabel('Sample index', fontsize=11, fontweight='bold')
            ax.set_ylabel('m(λ)', fontsize=11, fontweight='bold')
            ax.set_title(
                f'd={d}, K={K}\nmin={m_min}, mean={m_mean:.1f}, max={m_max}',
                fontsize=11,
                fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            # Add λ-independence indicator
            if m_std < 1e-3:
                indicator_text = '✓ λ-independent'
                indicator_color = 'green'
            else:
                indicator_text = '✗ λ-dependent'
                indicator_color = 'orange'

            ax.text(
                0.98, 0.02,
                indicator_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=indicator_color, alpha=0.3)
            )

        # Overall figure title
        method_label = 'Canonical Pauli Basis' if method == 'canonical' else 'Random Projectors'
        fig.suptitle(
            f'm(λ) Diagnostic - {method_label}\n(should be constant = d)',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Save figure
        fig_filename = f'm_vs_lambda_{method}.png'
        fig_path = outpath / fig_filename
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"\n✓ Figure saved: {fig_filename}")

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Krylov criterion comparison experiments'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Lambda-grid command
    lambda_parser = subparsers.add_parser(
        'run-lambda-grid',
        help='Run lambda-dependence grid experiment'
    )
    lambda_parser.add_argument(
        '--methods',
        type=str,
        default='canonical,projector',
        help='Comma-separated methods (canonical,projector)'
    )
    lambda_parser.add_argument(
        '--dims',
        type=str,
        default='8,16,32,64',
        help='Comma-separated dimensions'
    )
    lambda_parser.add_argument(
        '--Ks',
        type=str,
        default='4,8,12,16',
        help='Comma-separated K values'
    )
    lambda_parser.add_argument(
        '--n-lambda',
        type=int,
        default=200,
        help='Number of lambda samples per point'
    )
    lambda_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    lambda_parser.add_argument(
        '--outdir',
        type=str,
        default='fig_summary',
        help='Output directory'
    )

    # Criteria comparison command
    criteria_parser = subparsers.add_parser(
        'run-criteria-a',
        help='Run criteria comparison for canonical Pauli method'
    )
    criteria_parser.add_argument(
        '--dims',
        type=str,
        default='8,16,32,64',
        help='Comma-separated dimensions (powers of 2)'
    )
    criteria_parser.add_argument(
        '--Ks',
        type=str,
        default='4,8,12,16',
        help='Comma-separated K values'
    )
    criteria_parser.add_argument(
        '--taus',
        type=str,
        default='0.90,0.95',
        help='Comma-separated tau thresholds'
    )
    criteria_parser.add_argument(
        '--trials',
        type=int,
        default=200,
        help='Number of trials per point'
    )
    criteria_parser.add_argument(
        '--lambda-mode',
        type=str,
        default='uniform',
        choices=['uniform', 'seeded-normal'],
        help='Lambda weight mode'
    )
    criteria_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    criteria_parser.add_argument(
        '--outdir',
        type=str,
        default='fig_summary',
        help='Output directory'
    )

    # M(lambda) visualization command
    mlambda_parser = subparsers.add_parser(
        'plot-m-vs-lambda',
        help='Generate explicit m(λ) visualizations for lambda-dependence analysis'
    )
    mlambda_parser.add_argument(
        '--methods',
        type=str,
        default='canonical,projector',
        help='Comma-separated methods (canonical,projector)'
    )
    mlambda_parser.add_argument(
        '--dims',
        type=str,
        default='8,16,32',
        help='Comma-separated dimensions (powers of 2 for canonical)'
    )
    mlambda_parser.add_argument(
        '--Ks',
        type=str,
        default='4,8,12',
        help='Comma-separated K values'
    )
    mlambda_parser.add_argument(
        '--n-lambda',
        type=int,
        default=1000,
        help='Number of lambda samples per generator set'
    )
    mlambda_parser.add_argument(
        '--repeats',
        type=int,
        default=10,
        help='Number of generator sets to test (R)'
    )
    mlambda_parser.add_argument(
        '--slice',
        action='store_true',
        help='Generate 1-D theta slice for K=2 cases'
    )
    mlambda_parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed'
    )
    mlambda_parser.add_argument(
        '--outdir',
        type=str,
        default='fig_summary',
        help='Output directory'
    )

    # M-vs-lambda diagnostic command
    diagnostic_parser = subparsers.add_parser(
        'run-m-diagnostic',
        help='Run compact m-vs-lambda diagnostic for specific (d,K) test points'
    )
    diagnostic_parser.add_argument(
        '--methods',
        type=str,
        default='canonical,projector',
        help='Comma-separated methods (canonical,projector)'
    )
    diagnostic_parser.add_argument(
        '--test-points',
        type=str,
        default='8,4;16,8;32,12;64,16',
        help='Semicolon-separated (d,K) pairs (e.g., "8,4;16,8;32,12;64,16")'
    )
    diagnostic_parser.add_argument(
        '--n-lambda',
        type=int,
        default=500,
        help='Number of lambda samples per point'
    )
    diagnostic_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    diagnostic_parser.add_argument(
        '--outdir',
        type=str,
        default='fig_summary',
        help='Output directory'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    setup_logging()

    try:
        if args.command == 'run-lambda-grid':
            methods = args.methods.split(',')
            dims = [int(x) for x in args.dims.split(',')]
            Ks = [int(x) for x in args.Ks.split(',')]

            logger.info("="*80)
            logger.info("LAMBDA-DEPENDENCE GRID EXPERIMENT")
            logger.info("="*80)
            logger.info(f"Methods: {methods}")
            logger.info(f"Dimensions: {dims}")
            logger.info(f"K values: {Ks}")
            logger.info(f"Lambda samples: {args.n_lambda}")
            logger.info(f"Seed: {args.seed}")
            logger.info(f"Output: {args.outdir}")
            logger.info("="*80 + "\n")

            results = run_lambda_dependence_grid(
                methods=methods,
                dims=dims,
                Ks=Ks,
                n_lambda=args.n_lambda,
                seed=args.seed,
                outdir=args.outdir
            )

            csv_path = Path(args.outdir) / 'krylov_lambda_dependence_grid.csv'
            print_lambda_summary(csv_path, methods)

        elif args.command == 'run-criteria-a':
            dims = [int(x) for x in args.dims.split(',')]
            Ks = [int(x) for x in args.Ks.split(',')]
            taus = [float(x) for x in args.taus.split(',')]

            logger.info("="*80)
            logger.info("CRITERIA COMPARISON EXPERIMENT (Canonical Pauli)")
            logger.info("="*80)
            logger.info(f"Dimensions: {dims}")
            logger.info(f"K values: {Ks}")
            logger.info(f"Tau values: {taus}")
            logger.info(f"Trials per point: {args.trials}")
            logger.info(f"Lambda mode: {args.lambda_mode}")
            logger.info(f"Seed: {args.seed}")
            logger.info(f"Output: {args.outdir}")
            logger.info("="*80 + "\n")

            results = run_criteria_comparison(
                dims=dims,
                Ks=Ks,
                taus=taus,
                trials=args.trials,
                lambda_mode=args.lambda_mode,
                seed=args.seed,
                outdir=args.outdir
            )

            csv_path = Path(args.outdir) / 'criteria_comparison_dK.csv'
            print_criteria_summary(csv_path, dims)

        elif args.command == 'plot-m-vs-lambda':
            methods = args.methods.split(',')
            dims = [int(x) for x in args.dims.split(',')]
            Ks = [int(x) for x in args.Ks.split(',')]

            logger.info("="*80)
            logger.info("M(λ) VISUALIZATION EXPERIMENT")
            logger.info("="*80)
            logger.info(f"Methods: {methods}")
            logger.info(f"Dimensions: {dims}")
            logger.info(f"K values: {Ks}")
            logger.info(f"Lambda samples per set: {args.n_lambda}")
            logger.info(f"Generator sets (repeats): {args.repeats}")
            logger.info(f"1-D slice (K=2): {args.slice}")
            logger.info(f"Seed: {args.seed}")
            logger.info(f"Output: {args.outdir}")
            logger.info("="*80 + "\n")

            summary = krylov_lambda_plots.run_m_lambda_analysis(
                methods=methods,
                dims=dims,
                Ks=Ks,
                n_lambda=args.n_lambda,
                repeats=args.repeats,
                do_slice=args.slice,
                seed=args.seed,
                outdir=args.outdir
            )

            krylov_lambda_plots.print_summary_report(summary)

        elif args.command == 'run-m-diagnostic':
            methods = args.methods.split(',')
            # Parse test points: "8,4;16,8;32,12;64,16" -> [(8,4), (16,8), (32,12), (64,16)]
            test_points = []
            for pair_str in args.test_points.split(';'):
                d, K = pair_str.split(',')
                test_points.append((int(d), int(K)))

            logger.info("="*80)
            logger.info("M-VS-LAMBDA DIAGNOSTIC EXPERIMENT")
            logger.info("="*80)
            logger.info(f"Methods: {methods}")
            logger.info(f"Test points: {test_points}")
            logger.info(f"Lambda samples: {args.n_lambda}")
            logger.info(f"Seed: {args.seed}")
            logger.info(f"Output: {args.outdir}")
            logger.info("="*80 + "\n")

            results = run_m_vs_lambda_diagnostic(
                methods=methods,
                test_points=test_points,
                n_lambda=args.n_lambda,
                seed=args.seed,
                outdir=args.outdir
            )

            # Print summary and check for violations
            logger.info("\n" + "="*80)
            logger.info("DIAGNOSTIC SUMMARY")
            logger.info("="*80)

            violations = []
            for method in methods:
                logger.info(f"\nMethod: {method.upper()}")
                for (d, K), stats in results[method].items():
                    status = '✓' if stats['lambda_independent'] else '✗'
                    logger.info(
                        f"  {status} (d={d:2d}, K={K:2d}): "
                        f"std={stats['std']:.6f}, "
                        f"min={stats['min']}, max={stats['max']}"
                    )
                    if not stats['lambda_independent']:
                        violations.append((method, d, K, stats['std']))

            if violations:
                logger.warning("\n⚠ WARNING: Lambda-dependence detected!")
                logger.warning("Offending points:")
                for method, d, K, std in violations:
                    logger.warning(f"  - {method}: (d={d}, K={K}) with std={std:.6f}")
            else:
                logger.info("\n✓ All test points are λ-independent (std < 1e-3)")

    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1

    logger.info("\n" + "="*80)
    logger.info("✓ EXPERIMENT COMPLETE")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
