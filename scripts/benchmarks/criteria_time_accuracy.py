#!/usr/bin/env python3
"""
Criteria Time & Accuracy Benchmark.

Benchmarks all three reachability criteria (Spectral, Krylov, Moment) across
dimensions, ensembles, and optimizer methods. Produces:
  - Timing data per (criterion, d, K, ensemble)
  - Accuracy against known ground-truth systems
  - Optimizer method comparison
  - LaTeX tables and publication plots

Usage:
    # Profile first (quick, ~2 min)
    python scripts/benchmarks/criteria_time_accuracy.py --profile-only

    # Full benchmark
    python scripts/benchmarks/criteria_time_accuracy.py

    # Optimizer comparison only
    python scripts/benchmarks/criteria_time_accuracy.py --optimizer-only
"""

import argparse
import time
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import null_space

# Add parent to path for reach imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from reach import models, mathematics, optimize, settings
import qutip

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_GRID = {
    'dims': [8, 16, 32, 64],
    'K_values': [5, 10, 20],
    'ensembles': ['canonical', 'GEO2', 'GUE'],
    'tau': 0.99,
    'n_trials': 10,  # per (d, K, ensemble) point
    'seed': 42,
}

# Reduced grid for optimizer comparison
OPTIMIZER_GRID = {
    'dims': [8, 16],
    'K_values': [5, 10],
    'ensembles': ['GUE'],
    'methods': ['L-BFGS-B', 'TNC', 'SLSQP', 'CG', 'Powell', 'Nelder-Mead'],
    'n_trials': 10,
    'tau': 0.99,
    'seed': 42,
}

# GEO2 lattice configs by dimension
GEO2_LATTICE = {
    8: {'nx': 1, 'ny': 3},
    16: {'nx': 2, 'ny': 2},
    32: {'nx': 1, 'ny': 5},
    64: {'nx': 2, 'ny': 3},
}

OUTPUT_DIR = 'fig/benchmarks'
TEX_DIR = 'tex'


# =============================================================================
# Ground-Truth Systems
# =============================================================================

def create_provably_reachable_target(d, K, hams, psi, seed=42):
    """
    Create a target state that is provably reachable via time evolution.

    Strategy: Pick random λ, evolve psi under H(λ) for random time t.
    The resulting |φ⟩ = exp(-iH(λ)t)|ψ⟩ is reachable by construction.

    Returns:
        phi: QuTiP Qobj target state (provably reachable)
        lambda_true: The λ that generates it
        t_true: The time used
    """
    rng = np.random.RandomState(seed)
    lambda_true = rng.uniform(-1, 1, K)
    t_true = rng.uniform(0.5, 5.0)

    H = sum(l * H_k for l, H_k in zip(lambda_true, hams))
    U = (-1j * H * t_true).expm()
    phi = U * psi

    return phi, lambda_true, t_true


def create_provably_unreachable_target(d, K, hams, psi, seed=42):
    """
    Create a target state that is provably unreachable.

    Strategy: For K << d², pick a state orthogonal to the reachable subspace.
    We construct this by finding a state in the null space of the
    Lie algebra generators projected onto the initial state's sector.

    For small K, the dynamical Lie algebra is a strict subalgebra of su(d),
    so there exist unreachable states.

    Returns:
        phi: QuTiP Qobj target state (provably unreachable if K << d²)
    """
    rng = np.random.RandomState(seed)

    # Build the reachable set approximation via repeated commutators
    # For K << d², the span of {H_i, [H_i, H_j], ...} is much smaller than d²
    psi_vec = psi.full().flatten()

    # Generate many H(λ)|ψ⟩ vectors to span the reachable tangent space
    tangent_vecs = []
    for _ in range(min(50, 5 * K)):
        lam = rng.uniform(-1, 1, K)
        H = sum(l * H_k for l, H_k in zip(lam, hams))
        Hpsi = (H * psi).full().flatten()
        tangent_vecs.append(Hpsi)

    # Also include H^2|ψ⟩ vectors
    for _ in range(min(20, 2 * K)):
        lam = rng.uniform(-1, 1, K)
        H = sum(l * H_k for l, H_k in zip(lam, hams))
        H2psi = (H * H * psi).full().flatten()
        tangent_vecs.append(H2psi)

    T = np.array(tangent_vecs)
    if T.shape[0] > 0:
        # Find null space of T (states orthogonal to all tangent vectors)
        ns = null_space(T)
        if ns.shape[1] > 0:
            # Pick a random state from null space
            coeffs = rng.randn(ns.shape[1]) + 1j * rng.randn(ns.shape[1])
            phi_vec = ns @ coeffs
            phi_vec /= np.linalg.norm(phi_vec)
            phi = qutip.Qobj(phi_vec.reshape(-1, 1))
            return phi

    # Fallback: random state far from reachable subspace (not guaranteed)
    phi = models.random_states(1, d, seed=seed + 999)[0]
    return phi


# =============================================================================
# Timing Utilities
# =============================================================================

def time_spectral(psi, phi, hams, method='L-BFGS-B', maxiter=100,
                  restarts=3, ftol=1e-6, seed=42):
    """Time spectral overlap optimization. Returns (score, time_s)."""
    t0 = time.perf_counter()
    result = optimize.maximize_spectral_overlap(
        psi, phi, hams,
        method=method, maxiter=maxiter, restarts=restarts,
        ftol=ftol, seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return result['best_value'], elapsed


def time_krylov(psi, phi, hams, m, method='L-BFGS-B', maxiter=100,
                restarts=3, ftol=1e-6, seed=42):
    """Time Krylov score optimization. Returns (score, time_s)."""
    t0 = time.perf_counter()
    result = optimize.maximize_krylov_score(
        psi, phi, hams, m=m,
        method=method, maxiter=maxiter, restarts=restarts,
        ftol=ftol, seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return result['best_value'], elapsed


def time_moment(psi, phi, hams):
    """Time moment criterion evaluation. Returns (unreachable_bool, time_s)."""
    K = len(hams)
    t0 = time.perf_counter()

    L = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams])
    kernel = null_space(L.reshape(1, -1))

    if kernel.size == 0:
        unreachable = False
    else:
        Q = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                anticomm = (hams[i] * hams[j] + hams[j] * hams[i]) / 2
                Q[i, j] = qutip.expect(anticomm, phi) - qutip.expect(anticomm, psi)

        Q_proj = kernel.T @ Q @ kernel
        eigvals = np.linalg.eigvalsh(Q_proj)
        unreachable = bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10))

    elapsed = time.perf_counter() - t0
    return unreachable, elapsed


# =============================================================================
# Profiling
# =============================================================================

def run_profiling():
    """Quick profiling to identify bottlenecks before full benchmark."""
    print("=" * 60)
    print("PROFILING: Identifying bottlenecks")
    print("=" * 60)

    results = {}

    for d in [8, 16, 32]:
        for K in [5, 10]:
            for ensemble in ['GUE', 'canonical']:
                psi = models.fock_state(d, 0)
                phi = models.random_states(1, d, seed=100)[0]

                if ensemble == 'GEO2' and d in GEO2_LATTICE:
                    hams = models.random_hamiltonian_ensemble(
                        d, K, ensemble, seed=42, **GEO2_LATTICE[d]
                    )
                else:
                    hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=42)

                m = min(K, d)

                # Time each criterion (single evaluation)
                _, t_spec = time_spectral(psi, phi, hams, maxiter=50, restarts=1)
                _, t_krylov = time_krylov(psi, phi, hams, m, maxiter=50, restarts=1)
                _, t_moment = time_moment(psi, phi, hams)

                key = f"{ensemble} d={d} K={K}"
                results[key] = {
                    'spectral_s': t_spec,
                    'krylov_s': t_krylov,
                    'moment_s': t_moment,
                }
                print(f"  {key}: spectral={t_spec:.3f}s, "
                      f"krylov={t_krylov:.3f}s, moment={t_moment:.4f}s")

    print("\n--- Profiling Summary ---")
    print(f"{'Config':<30} {'Spectral':>10} {'Krylov':>10} {'Moment':>10}")
    for key, v in results.items():
        print(f"{key:<30} {v['spectral_s']:>10.3f} {v['krylov_s']:>10.3f} {v['moment_s']:>10.4f}")

    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(grid=None):
    """Run full timing + accuracy benchmark."""
    if grid is None:
        grid = BENCHMARK_GRID

    print("=" * 60)
    print("FULL BENCHMARK: Time & Accuracy")
    print("=" * 60)

    all_results = []
    rng = np.random.RandomState(grid['seed'])

    for ensemble in grid['ensembles']:
        for d in grid['dims']:
            # Skip infeasible combinations
            if ensemble == 'GEO2' and d not in GEO2_LATTICE:
                continue

            for K in grid['K_values']:
                # Skip K > d² (trivial)
                if K > d * d:
                    continue

                print(f"\n--- {ensemble} d={d} K={K} ---")

                # Generate Hamiltonians
                ham_seed = rng.randint(0, 2**31 - 1)
                if ensemble == 'GEO2':
                    # GEO2 has fixed operator count L = 3n + 9|E|
                    params = GEO2_LATTICE[d]
                    n_qubits = params['nx'] * params['ny']
                    n_edges = (params['nx'] - 1) * params['ny'] + params['nx'] * (params['ny'] - 1)
                    L = 3 * n_qubits + 9 * n_edges
                    if K > L:
                        print(f"  Skipping: K={K} > L={L} (max GEO2 operators)")
                        continue
                    hams = models.random_hamiltonian_ensemble(
                        d, K, ensemble, seed=ham_seed, **params
                    )
                else:
                    hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=ham_seed)

                psi = models.fock_state(d, 0)
                m = min(K, d)

                opt_settings = settings.get_production_settings(d, ensemble)
                maxiter = opt_settings['maxiter']
                restarts = opt_settings['restarts']
                ftol = opt_settings['ftol']

                # Per-trial timing and accuracy
                spec_times, krylov_times, moment_times = [], [], []
                spec_scores, krylov_scores = [], []
                moment_results_list = []

                # Ground-truth accuracy
                gt_reachable_spec, gt_reachable_kry = 0, 0
                gt_unreachable_mom = 0
                n_reachable_trials = 0
                n_unreachable_trials = 0

                for trial in range(grid['n_trials']):
                    trial_seed = rng.randint(0, 2**31 - 1)

                    # --- Reachable target (ground truth) ---
                    phi_reach, _, _ = create_provably_reachable_target(
                        d, K, hams, psi, seed=trial_seed
                    )

                    s_score, s_time = time_spectral(
                        psi, phi_reach, hams,
                        maxiter=maxiter, restarts=restarts, ftol=ftol,
                        seed=trial_seed,
                    )
                    r_score, r_time = time_krylov(
                        psi, phi_reach, hams, m,
                        maxiter=maxiter, restarts=restarts, ftol=ftol,
                        seed=trial_seed,
                    )
                    m_unreach, m_time = time_moment(psi, phi_reach, hams)

                    spec_times.append(s_time)
                    krylov_times.append(r_time)
                    moment_times.append(m_time)
                    spec_scores.append(s_score)
                    krylov_scores.append(r_score)
                    moment_results_list.append(m_unreach)

                    # Check accuracy: reachable targets should NOT be classified as unreachable
                    if s_score >= grid['tau']:
                        gt_reachable_spec += 1
                    if r_score >= grid['tau']:
                        gt_reachable_kry += 1
                    if not m_unreach:
                        # Moment says "possibly reachable" = correct for known-reachable
                        gt_unreachable_mom += 1  # counts correct classifications
                    n_reachable_trials += 1

                    # --- Random target (unknown ground truth, for timing only) ---
                    phi_rand = models.random_states(1, d, seed=trial_seed + 5000)[0]

                    s_score2, s_time2 = time_spectral(
                        psi, phi_rand, hams,
                        maxiter=maxiter, restarts=restarts, ftol=ftol,
                        seed=trial_seed + 1000,
                    )
                    r_score2, r_time2 = time_krylov(
                        psi, phi_rand, hams, m,
                        maxiter=maxiter, restarts=restarts, ftol=ftol,
                        seed=trial_seed + 1000,
                    )
                    _, m_time2 = time_moment(psi, phi_rand, hams)

                    spec_times.append(s_time2)
                    krylov_times.append(r_time2)
                    moment_times.append(m_time2)

                # Aggregate
                result = {
                    'ensemble': ensemble,
                    'd': d,
                    'K': K,
                    'rho': K / d**2,
                    'm': m,
                    'n_trials': grid['n_trials'],
                    'tau': grid['tau'],
                    # Timing
                    'spectral_mean_s': np.mean(spec_times),
                    'spectral_std_s': np.std(spec_times),
                    'krylov_mean_s': np.mean(krylov_times),
                    'krylov_std_s': np.std(krylov_times),
                    'moment_mean_s': np.mean(moment_times),
                    'moment_std_s': np.std(moment_times),
                    # Scores
                    'spectral_mean_score': np.mean(spec_scores),
                    'krylov_mean_score': np.mean(krylov_scores),
                    'moment_unreachable_frac': np.mean(moment_results_list),
                    # Accuracy on known-reachable targets
                    'spectral_true_positive_rate': gt_reachable_spec / max(n_reachable_trials, 1),
                    'krylov_true_positive_rate': gt_reachable_kry / max(n_reachable_trials, 1),
                    'moment_correct_rate': gt_unreachable_mom / max(n_reachable_trials, 1),
                }

                print(f"  Spectral: {result['spectral_mean_s']:.3f}s "
                      f"(TPR={result['spectral_true_positive_rate']:.2f})")
                print(f"  Krylov:   {result['krylov_mean_s']:.3f}s "
                      f"(TPR={result['krylov_true_positive_rate']:.2f})")
                print(f"  Moment:   {result['moment_mean_s']:.4f}s "
                      f"(correct={result['moment_correct_rate']:.2f})")

                all_results.append(result)

    return all_results


# =============================================================================
# Optimizer Comparison
# =============================================================================

def run_optimizer_comparison(grid=None):
    """Compare optimizer methods on reduced grid."""
    if grid is None:
        grid = OPTIMIZER_GRID

    print("=" * 60)
    print("OPTIMIZER COMPARISON")
    print("=" * 60)

    all_results = []
    rng = np.random.RandomState(grid['seed'])

    for d in grid['dims']:
        for K in grid['K_values']:
            for ensemble in grid['ensembles']:
                ham_seed = rng.randint(0, 2**31 - 1)
                hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=ham_seed)
                psi = models.fock_state(d, 0)

                for method in grid['methods']:
                    scores = []
                    times = []
                    nfevs = []

                    for trial in range(grid['n_trials']):
                        trial_seed = rng.randint(0, 2**31 - 1)
                        phi = models.random_states(1, d, seed=trial_seed)[0]

                        t0 = time.perf_counter()
                        try:
                            result = optimize.maximize_spectral_overlap(
                                psi, phi, hams,
                                method=method, maxiter=100, restarts=2,
                                ftol=1e-6, seed=trial_seed,
                            )
                            elapsed = time.perf_counter() - t0
                            scores.append(result['best_value'])
                            times.append(elapsed)
                            nfevs.append(result['nfev'])
                        except Exception as e:
                            elapsed = time.perf_counter() - t0
                            logger.warning(f"  {method} failed: {e}")
                            scores.append(0.0)
                            times.append(elapsed)
                            nfevs.append(0)

                    opt_result = {
                        'ensemble': ensemble,
                        'd': d,
                        'K': K,
                        'method': method,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'mean_time': np.mean(times),
                        'std_time': np.std(times),
                        'mean_nfev': np.mean(nfevs),
                        'n_trials': grid['n_trials'],
                    }
                    all_results.append(opt_result)

                    print(f"  {ensemble} d={d} K={K} {method:<12}: "
                          f"S*={opt_result['mean_score']:.4f}±{opt_result['std_score']:.4f}, "
                          f"t={opt_result['mean_time']:.3f}s, nfev={opt_result['mean_nfev']:.0f}")

    return all_results


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_latex_tables(benchmark_results, optimizer_results, outdir=TEX_DIR):
    """Generate publication-ready LaTeX tables."""
    os.makedirs(outdir, exist_ok=True)

    # Table 1: Main benchmark (time per criterion)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean evaluation time (seconds) per criterion across dimensions and ensembles. "
        r"Spectral and Krylov use L-BFGS-B with production settings. "
        r"Moment is analytical (no optimization).}",
        r"\label{tab:criteria_timing}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Ensemble & $d$ & $K$ & Spectral (s) & Krylov (s) & Moment (s) & Speedup \\",
        r"\midrule",
    ]

    for r in benchmark_results:
        speedup = r['spectral_mean_s'] / max(r['moment_mean_s'], 1e-6)
        lines.append(
            f"  {r['ensemble']} & {r['d']} & {r['K']} & "
            f"{r['spectral_mean_s']:.3f} & {r['krylov_mean_s']:.3f} & "
            f"{r['moment_mean_s']:.4f} & {speedup:.0f}$\\times$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    filepath = os.path.join(outdir, 'criteria_comparison_table.tex')
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {filepath}")

    # Table 2: Accuracy on ground-truth targets
    lines2 = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{True positive rate on provably reachable targets ($\tau = 0.99$). "
        r"TPR = fraction of known-reachable targets correctly classified as reachable.}",
        r"\label{tab:criteria_accuracy}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Ensemble & $d$ & $K$ & Spectral TPR & Krylov TPR & Moment Correct \\",
        r"\midrule",
    ]

    for r in benchmark_results:
        lines2.append(
            f"  {r['ensemble']} & {r['d']} & {r['K']} & "
            f"{r['spectral_true_positive_rate']:.2f} & "
            f"{r['krylov_true_positive_rate']:.2f} & "
            f"{r['moment_correct_rate']:.2f} \\\\"
        )

    lines2.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    filepath2 = os.path.join(outdir, 'criteria_accuracy_table.tex')
    with open(filepath2, 'w') as f:
        f.write('\n'.join(lines2))
    print(f"Saved: {filepath2}")

    # Table 3: Optimizer comparison
    if optimizer_results:
        lines3 = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Optimizer method comparison for spectral overlap maximization (GUE ensemble). "
            r"Mean $S^*$ score and wall time across random targets.}",
            r"\label{tab:optimizer_comparison}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Method & $d=8,K=5$ $S^*$ & Time (s) & $d=16,K=10$ $S^*$ & Time (s) \\",
            r"\midrule",
        ]

        for method in OPTIMIZER_GRID['methods']:
            vals = {}
            for r in optimizer_results:
                if r['method'] == method:
                    key = f"d{r['d']}K{r['K']}"
                    vals[key] = r

            r1 = vals.get('d8K5', {})
            r2 = vals.get('d16K10', {})

            lines3.append(
                f"  {method} & "
                f"{r1.get('mean_score', 0):.4f} & {r1.get('mean_time', 0):.3f} & "
                f"{r2.get('mean_score', 0):.4f} & {r2.get('mean_time', 0):.3f} \\\\"
            )

        lines3.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        filepath3 = os.path.join(outdir, 'optimizer_comparison_table.tex')
        with open(filepath3, 'w') as f:
            f.write('\n'.join(lines3))
        print(f"Saved: {filepath3}")


# =============================================================================
# Plotting
# =============================================================================

def plot_time_vs_dimension(results, outdir=OUTPUT_DIR):
    """Plot 1: Mean time vs d for each criterion and ensemble."""
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ensembles = sorted(set(r['ensemble'] for r in results))
    colors = {'canonical': '#1f77b4', 'GUE': '#ff7f0e', 'GEO2': '#2ca02c'}
    K_ref = 10  # Reference K value

    for ax, criterion in zip(axes, ['spectral', 'krylov', 'moment']):
        time_key = f'{criterion}_mean_s'

        for ens in ensembles:
            subset = [r for r in results if r['ensemble'] == ens and r['K'] == K_ref]
            if not subset:
                continue
            subset.sort(key=lambda r: r['d'])
            ds = [r['d'] for r in subset]
            ts = [r[time_key] for r in subset]

            ax.plot(ds, ts, 'o-', color=colors.get(ens, 'gray'),
                    label=ens, markersize=6)

        ax.set_xlabel('Dimension d', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title(f'{criterion.capitalize()} (K={K_ref})', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(outdir, 'time_vs_dimension.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_time_vs_K(results, outdir=OUTPUT_DIR):
    """Plot 2: Mean time vs K for each criterion at fixed d."""
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    d_ref = 16
    colors = {'canonical': '#1f77b4', 'GUE': '#ff7f0e', 'GEO2': '#2ca02c'}
    ensembles = sorted(set(r['ensemble'] for r in results))

    for ax, criterion in zip(axes, ['spectral', 'krylov', 'moment']):
        time_key = f'{criterion}_mean_s'

        for ens in ensembles:
            subset = [r for r in results if r['ensemble'] == ens and r['d'] == d_ref]
            if not subset:
                continue
            subset.sort(key=lambda r: r['K'])
            Ks = [r['K'] for r in subset]
            ts = [r[time_key] for r in subset]

            ax.plot(Ks, ts, 'o-', color=colors.get(ens, 'gray'),
                    label=ens, markersize=6)

        ax.set_xlabel('K (Hamiltonians)', fontsize=12)
        ax.set_ylabel('Time (s)', fontsize=12)
        ax.set_title(f'{criterion.capitalize()} (d={d_ref})', fontsize=12)
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(outdir, 'time_vs_K.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_accuracy_heatmap(results, outdir=OUTPUT_DIR):
    """Plot 3: Accuracy heatmap (TPR on reachable targets)."""
    os.makedirs(outdir, exist_ok=True)

    ensembles = sorted(set(r['ensemble'] for r in results))
    fig, axes = plt.subplots(1, len(ensembles), figsize=(6 * len(ensembles), 5))
    if len(ensembles) == 1:
        axes = [axes]

    for ax, ens in zip(axes, ensembles):
        subset = [r for r in results if r['ensemble'] == ens]
        if not subset:
            continue

        dims = sorted(set(r['d'] for r in subset))
        Ks = sorted(set(r['K'] for r in subset))

        tpr_grid = np.full((len(dims), len(Ks)), np.nan)
        for r in subset:
            i = dims.index(r['d'])
            j = Ks.index(r['K'])
            tpr_grid[i, j] = r['spectral_true_positive_rate']

        im = ax.imshow(tpr_grid, cmap='RdYlGn', vmin=0, vmax=1,
                       aspect='auto', origin='lower')
        ax.set_xticks(range(len(Ks)))
        ax.set_xticklabels(Ks)
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels(dims)
        ax.set_xlabel('K', fontsize=12)
        ax.set_ylabel('d', fontsize=12)
        ax.set_title(f'Spectral TPR — {ens}', fontsize=12)

        # Annotate cells
        for i in range(len(dims)):
            for j in range(len(Ks)):
                val = tpr_grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=9, color='black' if val > 0.5 else 'white')

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    filepath = os.path.join(outdir, 'accuracy_heatmap.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_time_accuracy_scatter(results, outdir=OUTPUT_DIR):
    """Plot 4: Time vs accuracy scatter for all criteria."""
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    markers = {'spectral': 's', 'krylov': '^', 'moment': 'o'}
    colors_ens = {'canonical': '#1f77b4', 'GUE': '#ff7f0e', 'GEO2': '#2ca02c'}

    for r in results:
        for criterion, marker in markers.items():
            time_key = f'{criterion}_mean_s'
            if criterion == 'moment':
                acc = r['moment_correct_rate']
            elif criterion == 'spectral':
                acc = r['spectral_true_positive_rate']
            else:
                acc = r['krylov_true_positive_rate']

            ax.scatter(r[time_key], acc,
                       marker=marker, color=colors_ens.get(r['ensemble'], 'gray'),
                       s=20 + 3 * r['d'], alpha=0.6)

    # Legend entries
    for crit, marker in markers.items():
        ax.scatter([], [], marker=marker, color='gray', label=crit.capitalize(), s=60)
    for ens, color in colors_ens.items():
        ax.scatter([], [], marker='o', color=color, label=ens, s=60)

    ax.set_xlabel('Mean time (s)', fontsize=12)
    ax.set_ylabel('Accuracy (TPR on reachable targets)', fontsize=12)
    ax.set_xscale('log')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(outdir, 'time_accuracy_scatter.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def plot_optimizer_heatmap(optimizer_results, outdir=OUTPUT_DIR):
    """Plot 5: Optimizer method comparison heatmap."""
    os.makedirs(outdir, exist_ok=True)

    if not optimizer_results:
        print("No optimizer results to plot.")
        return

    methods = OPTIMIZER_GRID['methods']
    configs = sorted(set((r['d'], r['K']) for r in optimizer_results))

    score_grid = np.full((len(methods), len(configs)), np.nan)
    time_grid = np.full((len(methods), len(configs)), np.nan)

    for r in optimizer_results:
        i = methods.index(r['method'])
        j = configs.index((r['d'], r['K']))
        score_grid[i, j] = r['mean_score']
        time_grid[i, j] = r['mean_time']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(score_grid, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                     aspect='auto')
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([f'd={d},K={K}' for d, K in configs], rotation=45, ha='right')
    ax1.set_title('Mean S* Score', fontsize=12)
    for i in range(len(methods)):
        for j in range(len(configs)):
            val = score_grid[i, j]
            if not np.isnan(val):
                ax1.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(time_grid, cmap='YlOrRd', aspect='auto')
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([f'd={d},K={K}' for d, K in configs], rotation=45, ha='right')
    ax2.set_title('Mean Time (s)', fontsize=12)
    for i in range(len(methods)):
        for j in range(len(configs)):
            val = time_grid[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.tight_layout()
    filepath = os.path.join(outdir, 'optimizer_heatmap.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Markdown Report
# =============================================================================

def write_benchmark_report(benchmark_results, optimizer_results, profile_results,
                           outdir='scripts/benchmarks'):
    """Write CRITERIA_BENCHMARK.md report."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [
        "# Criteria Time & Accuracy Benchmark",
        f"",
        f"**Generated:** {timestamp}",
        f"**Script:** `scripts/benchmarks/criteria_time_accuracy.py`",
        "",
        "---",
        "",
        "## 1. Profiling Summary",
        "",
        "| Config | Spectral (s) | Krylov (s) | Moment (s) |",
        "|--------|-------------|-----------|----------|",
    ]

    if profile_results:
        for key, v in profile_results.items():
            lines.append(
                f"| {key} | {v['spectral_s']:.3f} | "
                f"{v['krylov_s']:.3f} | {v['moment_s']:.4f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Full Benchmark Results",
        "",
        "### Timing",
        "",
        "| Ensemble | d | K | Spectral (s) | Krylov (s) | Moment (s) |",
        "|----------|---|---|-------------|-----------|----------|",
    ])

    for r in benchmark_results:
        lines.append(
            f"| {r['ensemble']} | {r['d']} | {r['K']} | "
            f"{r['spectral_mean_s']:.3f} | {r['krylov_mean_s']:.3f} | "
            f"{r['moment_mean_s']:.4f} |"
        )

    lines.extend([
        "",
        "### Accuracy (True Positive Rate on Reachable Targets)",
        "",
        "| Ensemble | d | K | Spectral TPR | Krylov TPR | Moment Correct |",
        "|----------|---|---|-------------|-----------|----------------|",
    ])

    for r in benchmark_results:
        lines.append(
            f"| {r['ensemble']} | {r['d']} | {r['K']} | "
            f"{r['spectral_true_positive_rate']:.2f} | "
            f"{r['krylov_true_positive_rate']:.2f} | "
            f"{r['moment_correct_rate']:.2f} |"
        )

    if optimizer_results:
        lines.extend([
            "",
            "---",
            "",
            "## 3. Optimizer Comparison",
            "",
            "| Method | d | K | Mean S* | Std S* | Mean Time (s) | Mean nfev |",
            "|--------|---|---|---------|--------|--------------|-----------|",
        ])

        for r in optimizer_results:
            lines.append(
                f"| {r['method']} | {r['d']} | {r['K']} | "
                f"{r['mean_score']:.4f} | {r['std_score']:.4f} | "
                f"{r['mean_time']:.3f} | {r['mean_nfev']:.0f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Plots",
        "",
        "| Plot | Description |",
        "|------|-------------|",
        "| `fig/benchmarks/time_vs_dimension.png` | Time scaling with d |",
        "| `fig/benchmarks/time_vs_K.png` | Time scaling with K |",
        "| `fig/benchmarks/accuracy_heatmap.png` | TPR heatmap by (d, K) |",
        "| `fig/benchmarks/time_accuracy_scatter.png` | Time-accuracy Pareto front |",
        "| `fig/benchmarks/optimizer_heatmap.png` | Optimizer comparison |",
        "",
        "---",
        "",
        "## 5. Key Findings",
        "",
        "*(Auto-populated after running benchmark)*",
        "",
    ])

    filepath = os.path.join(outdir, 'CRITERIA_BENCHMARK.md')
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Criteria Time & Accuracy Benchmark')
    parser.add_argument('--profile-only', action='store_true',
                        help='Run profiling only (quick)')
    parser.add_argument('--optimizer-only', action='store_true',
                        help='Run optimizer comparison only')
    parser.add_argument('--skip-optimizer', action='store_true',
                        help='Skip optimizer comparison in full run')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override number of trials')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)

    print(f"Criteria Time & Accuracy Benchmark")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    profile_results = None
    benchmark_results = []
    optimizer_results = []

    if args.n_trials:
        BENCHMARK_GRID['n_trials'] = args.n_trials
        OPTIMIZER_GRID['n_trials'] = args.n_trials

    # Step 1: Profile
    profile_results = run_profiling()

    if args.profile_only:
        print("\nProfile-only mode. Done.")
        return

    # Step 2: Full benchmark or optimizer only
    if args.optimizer_only:
        optimizer_results = run_optimizer_comparison()
    else:
        benchmark_results = run_benchmark()
        if not args.skip_optimizer:
            optimizer_results = run_optimizer_comparison()

    # Step 3: Generate outputs
    if benchmark_results:
        plot_time_vs_dimension(benchmark_results)
        plot_time_vs_K(benchmark_results)
        plot_accuracy_heatmap(benchmark_results)
        plot_time_accuracy_scatter(benchmark_results)

    if optimizer_results:
        plot_optimizer_heatmap(optimizer_results)

    if benchmark_results or optimizer_results:
        generate_latex_tables(benchmark_results, optimizer_results)

    write_benchmark_report(benchmark_results, optimizer_results, profile_results)

    # Save raw results as JSON
    raw_data = {
        'timestamp': datetime.now().isoformat(),
        'profile': profile_results,
        'benchmark': benchmark_results,
        'optimizer': optimizer_results,
        'grid': BENCHMARK_GRID,
        'optimizer_grid': OPTIMIZER_GRID,
    }
    json_path = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, 'w') as f:
        json.dump(raw_data, f, indent=2, default=convert)
    print(f"Saved: {json_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
