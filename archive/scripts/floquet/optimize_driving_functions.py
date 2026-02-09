#!/usr/bin/env python3
"""
Optimize driving functions for Floquet-engineered QEC code preparation.

This script:
1. Compares different driving function families
2. Analyzes Fourier overlap structure
3. Optimizes driving parameters to maximize criterion strength
4. Generates comparison figures

Usage:
    python3 scripts/optimize_driving_functions.py --mode compare
    python3 scripts/optimize_driving_functions.py --mode optimize --K 4
    python3 scripts/optimize_driving_functions.py --mode full --K 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from datetime import datetime
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

from reach import floquet, models, moment_criteria


# ============================================================================
# 5-QUBIT CODE SETUP
# ============================================================================

def create_5qubit_code_logical_zero():
    """
    Create the [[5,1,3]] perfect code logical |0_L⟩ state.

    The logical zero state is an equal superposition of 16 basis states
    with specific signs, forming the +1 eigenspace of the stabilizer group.

    Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
    |0_L⟩ is the +1 eigenstate of all stabilizers AND Z_L = ZZZZZ

    Returns:
        np.ndarray: Normalized state vector of dimension 32
    """
    d = 32  # 2^5
    psi = np.zeros(d, dtype=complex)

    # Weight 0 (1 state)
    psi[0b00000] = +1

    # Weight 2 (4 states)
    psi[0b10010] = +1
    psi[0b01001] = +1
    psi[0b10100] = +1
    psi[0b01010] = +1

    # Weight 3 (10 states) - all negative
    psi[0b11011] = -1
    psi[0b00110] = -1
    psi[0b11000] = -1
    psi[0b11101] = -1
    psi[0b00011] = -1
    psi[0b11110] = -1
    psi[0b01111] = -1
    psi[0b10001] = -1
    psi[0b01100] = -1
    psi[0b10111] = -1

    # Weight 4 (1 state)
    psi[0b00101] = +1

    # Normalize
    psi /= np.linalg.norm(psi)

    return psi


# ============================================================================
# DRIVING FUNCTION FAMILIES
# ============================================================================

def create_monochromatic(K, omega=1.0):
    """All operators driven at same frequency (zero DC)."""
    return [lambda t, w=omega: np.sin(w * t) for _ in range(K)]


def create_monochromatic_offset(K, omega=1.0, offset=1.0, amplitude=0.5):
    """All operators at same frequency with non-zero DC offset."""
    return [lambda t, w=omega, off=offset, amp=amplitude:
            off + amp * np.sin(w * t) for _ in range(K)]


def create_bichromatic(K, omega1=1.0, omega2=None):
    """Alternating between two incommensurate frequencies."""
    if omega2 is None:
        omega2 = np.sqrt(2)  # Incommensurate with 1.0
    return [lambda t, w=(omega1 if k % 2 == 0 else omega2): np.sin(w * t)
            for k in range(K)]


def create_polychromatic(K, base_omega=1.0, spread=0.5):
    """Each operator at different frequency (spread around base)."""
    omegas = base_omega + spread * np.linspace(-1, 1, K)
    return [lambda t, w=w: np.sin(w * t) for w in omegas]


def create_random_phases(K, omega=1.0, seed=42):
    """Same frequency, random phase offsets."""
    rng = np.random.RandomState(seed)
    phases = rng.uniform(0, 2*np.pi, K)
    return [lambda t, w=omega, p=p: np.sin(w * t + p) for p in phases]


def create_optimized(K, omegas, phases):
    """Custom frequencies and phases from optimization."""
    return [lambda t, w=w, p=p: np.sin(w * t + p)
            for w, p in zip(omegas, phases)]


def create_square_wave(K, omega=1.0):
    """Square wave driving (sharp transitions)."""
    return [lambda t, w=omega * (k + 1): np.sign(np.sin(w * t))
            for k in range(K)]


def create_amplitude_modulated(K, omega_carrier=5.0, omega_mod=1.0):
    """Amplitude-modulated sinusoidal driving."""
    return [lambda t, wc=omega_carrier * (k + 1), wm=omega_mod:
            np.sin(wc * t) * (1 + 0.5*np.cos(wm * t)) for k in range(K)]


# ============================================================================
# FOURIER OVERLAP COMPUTATION
# ============================================================================

def compute_fourier_overlap_matrix(driving_funcs, period, n_terms=10):
    """
    Compute F_{jk} matrix for all operator pairs using existing floquet function.

    Args:
        driving_funcs: List of K driving functions
        period: Driving period T
        n_terms: Number of Fourier terms for overlap computation

    Returns:
        K x K antisymmetric matrix of Fourier overlaps
    """
    K = len(driving_funcs)
    F = np.zeros((K, K))

    for j in range(K):
        for k in range(j+1, K):
            F_jk = floquet.compute_fourier_overlap(
                driving_funcs[j], driving_funcs[k], period, n_terms=n_terms
            )
            F[j, k] = F_jk
            F[k, j] = -F_jk  # Antisymmetric

    return F


def analyze_fourier_matrix(F_matrix):
    """
    Analyze structure of Fourier overlap matrix.

    Returns:
        Dictionary with analysis metrics
    """
    K = F_matrix.shape[0]

    # Get upper triangular elements (non-redundant)
    upper_tri = F_matrix[np.triu_indices(K, k=1)]

    return {
        'frobenius_norm': np.linalg.norm(F_matrix),
        'max_element': np.max(np.abs(F_matrix)),
        'mean_element': np.mean(np.abs(upper_tri)),
        'std_element': np.std(np.abs(upper_tri)),
        'n_large_elements': np.sum(np.abs(upper_tri) > 0.1),
        'sparsity': np.sum(np.abs(F_matrix) < 0.01) / (K * K),
    }


# ============================================================================
# CRITERION EVALUATION
# ============================================================================

def compute_min_eigenvalue_with_x_search(L_F, Q_F, x_range=(-10, 10), n_points=1000):
    """
    Find the minimum eigenvalue of Q_F + x L_F L_F^T over all x.

    If max_x min_eig(Q_F + x L_F L_F^T) > 0, criterion succeeds.

    Returns:
        best_min_eig: Maximum achievable minimum eigenvalue
        best_x: Value of x achieving it
    """
    L_F_outer = np.outer(L_F, L_F)
    x_values = np.linspace(x_range[0], x_range[1], n_points)

    best_min_eig = -np.inf
    best_x = None

    for x in x_values:
        M = Q_F + x * L_F_outer
        min_eig = np.min(np.linalg.eigvalsh(M))

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_x = x

    return best_min_eig, best_x


def compute_criterion_matrices(psi, phi, hams, lambdas, driving_functions, period, order=2):
    """
    Compute L_F and Q_F matrices for given parameters.

    Returns L_F vector and Q_F matrix used in criterion evaluation.
    """
    K = len(hams)

    # Compute derivatives ∂H_F/∂λ_k
    dH_F_dlambda = []
    for k in range(K):
        lambda_bar_k = floquet.compute_time_average(driving_functions[k], period)
        derivative = lambda_bar_k * hams[k]

        if order >= 2:
            for j in range(K):
                if j != k:
                    F_jk = floquet.compute_fourier_overlap(
                        driving_functions[j], driving_functions[k], period
                    )
                    commutator = hams[j] @ hams[k] - hams[k] @ hams[j]
                    derivative += lambdas[j] * F_jk * commutator / (2 * 1j)

        derivative = (derivative + derivative.conj().T) / 2
        dH_F_dlambda.append(derivative)

    # Compute L_F
    L_F = np.zeros(K)
    for k in range(K):
        exp_val_phi = np.real(phi.conj() @ dH_F_dlambda[k] @ phi)
        exp_val_psi = np.real(psi.conj() @ dH_F_dlambda[k] @ psi)
        L_F[k] = exp_val_phi - exp_val_psi

    # Compute Q_F
    Q_F = np.zeros((K, K))
    for k in range(K):
        for m in range(K):
            anticomm = (dH_F_dlambda[k] @ dH_F_dlambda[m] +
                       dH_F_dlambda[m] @ dH_F_dlambda[k]) / 2
            exp_val_phi = np.real(phi.conj() @ anticomm @ phi)
            exp_val_psi = np.real(psi.conj() @ anticomm @ psi)
            Q_F[k, m] = exp_val_phi - exp_val_psi

    return L_F, Q_F


def evaluate_criterion_strength(psi, phi, hams, driving_funcs, period,
                                order=2, n_lambda_trials=200, seed=42):
    """
    Evaluate Floquet criterion and return strength metrics.

    Args:
        psi: Initial state
        phi: Target state
        hams: List of Hamiltonian matrices
        driving_funcs: List of driving functions
        period: Driving period T
        order: Magnus expansion order
        n_lambda_trials: Number of random λ samples
        seed: Random seed

    Returns:
        Dictionary with:
        - success: Whether criterion proves unreachability
        - min_eigenvalue: Best achievable min eigenvalue (criterion strength)
        - lambda_opt: Optimal coupling vector
        - F_matrix: Fourier overlap matrix
        - F_analysis: Analysis of F matrix structure
    """
    K = len(driving_funcs)

    # Compute Fourier overlap matrix
    F_matrix = compute_fourier_overlap_matrix(driving_funcs, period)
    F_analysis = analyze_fourier_matrix(F_matrix)

    # Run criterion with lambda optimization
    success, lambda_opt, x_opt, eigvals = \
        moment_criteria.floquet_moment_criterion_optimized(
            psi, phi, hams[:K],
            driving_functions=driving_funcs,
            period=period,
            order=order,
            n_lambda_trials=n_lambda_trials,
            seed=seed
        )

    min_eigenvalue = min(eigvals) if eigvals is not None and len(eigvals) > 0 else None

    return {
        'success': success,
        'min_eigenvalue': min_eigenvalue,
        'lambda_opt': lambda_opt,
        'x_opt': x_opt,
        'F_matrix': F_matrix,
        'F_analysis': F_analysis,
    }


# ============================================================================
# FAST GRID-BASED OPTIMIZATION
# ============================================================================

def compute_analytical_fourier_overlap(omega_j, omega_k, phi_j, phi_k, period):
    """
    Compute Fourier overlap for sinusoidal driving functions analytically.

    For f_j(t) = sin(ω_j t + φ_j) and f_k(t) = sin(ω_k t + φ_k):
    The overlap depends on frequency relationships.

    Returns approximate overlap based on frequency difference.
    """
    # For different frequencies, overlap is non-zero
    # The magnitude depends on how "incommensurate" the frequencies are
    if abs(omega_j - omega_k) < 1e-10:
        # Same frequency: overlap depends on phase difference
        return np.sin(phi_j - phi_k) / (2 * omega_j)
    else:
        # Different frequencies: complex beat pattern
        delta_omega = omega_j - omega_k
        # Approximate overlap magnitude
        return np.sin(phi_j - phi_k) / delta_omega


def fast_grid_optimization(psi, phi, hams, period, K, n_grid=10,
                           n_lambda_trials=100, seed=42, verbose=True):
    """
    Fast optimization using grid search over frequency patterns.

    Instead of expensive continuous optimization, we:
    1. Test a grid of frequency ratios
    2. Pre-compute Fourier overlaps for each configuration
    3. Evaluate criterion strength

    This is much faster than differential_evolution.
    """
    if verbose:
        print(f"Fast grid optimization for K={K}...")
        print(f"  Grid size: {n_grid} frequency ratios")
        print(f"  λ trials per configuration: {n_lambda_trials}")

    # Frequency patterns to test
    # Pattern 1: All different (polychromatic)
    # Pattern 2: Alternating (bichromatic)
    # Pattern 3: Random ratios

    base_omega = 2 * np.pi / period

    # Test configurations
    configs = []

    # Bichromatic variants (vary ratio between two frequencies)
    ratios = [np.sqrt(2), np.sqrt(3), np.sqrt(5), 1.618, 2.0, 2.5, 3.0]
    for ratio in ratios:
        omegas = [base_omega if k % 2 == 0 else base_omega * ratio for k in range(K)]
        phases = [0.0] * K
        configs.append(('bichromatic_r{:.2f}'.format(ratio), omegas, phases))

    # Polychromatic variants (spread frequencies)
    for spread in [0.3, 0.5, 0.7, 1.0]:
        omegas = base_omega * (1 + spread * np.linspace(-1, 1, K))
        phases = [0.0] * K
        configs.append(('polychromatic_s{:.1f}'.format(spread), list(omegas), phases))

    # Random phase variants (same frequency, different phases)
    rng = np.random.RandomState(seed)
    for trial in range(3):
        phases = list(rng.uniform(0, 2*np.pi, K))
        omegas = [base_omega * np.sqrt(2) if k % 2 == 0 else base_omega for k in range(K)]
        configs.append(('random_phases_{}'.format(trial), omegas, phases))

    # Evaluate each configuration
    results = []

    for name, omegas, phases in configs:
        driving_funcs = create_optimized(K, omegas, phases)

        # Run criterion evaluation
        success, lambda_opt, x_opt, eigvals = \
            moment_criteria.floquet_moment_criterion_optimized(
                psi, phi, hams[:K],
                driving_functions=driving_funcs,
                period=period,
                order=2,
                n_lambda_trials=n_lambda_trials,
                seed=seed
            )

        min_eig = min(eigvals) if eigvals is not None and len(eigvals) > 0 else None

        results.append({
            'name': name,
            'omegas': omegas,
            'phases': phases,
            'success': success,
            'min_eigenvalue': min_eig,
            'lambda_opt': lambda_opt,
        })

        if verbose:
            status = "YES" if success else "NO"
            eig_str = f"{min_eig:.6f}" if min_eig else "N/A"
            print(f"  {name:<25}: {status} (min_eig={eig_str})")

    # Find best configuration
    successful = [r for r in results if r['success']]
    if successful:
        best = max(successful, key=lambda r: r['min_eigenvalue'] or 0)
    else:
        best = results[0]

    if verbose:
        print()
        print(f"Best configuration: {best['name']}")
        print(f"  Min eigenvalue: {best['min_eigenvalue']}")

    return {
        'optimal_omegas': np.array(best['omegas']),
        'optimal_phases': np.array(best['phases']),
        'optimal_strength': best['min_eigenvalue'],
        'best_config_name': best['name'],
        'all_results': results,
    }


def optimize_driving_parameters(psi, phi, hams, period, K,
                                maxiter=100, seed=42, verbose=True, fast=True):
    """
    Optimize driving function parameters (frequencies, phases) to maximize
    criterion discriminative power.

    Args:
        psi: Initial state
        phi: Target state
        hams: List of Hamiltonian matrices (at least K)
        period: Driving period T
        K: Number of operators to use
        maxiter: Maximum optimization iterations
        seed: Random seed
        verbose: Print progress
        fast: Use fast grid search instead of differential evolution

    Returns:
        Dictionary with optimal parameters and metrics
    """
    if fast:
        return fast_grid_optimization(
            psi, phi, hams, period, K,
            n_grid=10, n_lambda_trials=100, seed=seed, verbose=verbose
        )

    # Original slow differential evolution approach
    eval_count = [0]
    best_so_far = [float('inf')]

    def objective(params):
        omegas = params[:K]
        phases = params[K:2*K]

        # Create driving functions
        driving_funcs = create_optimized(K, omegas, phases)

        # We need to search over λ as well - use a few random trials
        rng = np.random.RandomState(seed + eval_count[0])

        best_min_eig = -np.inf

        for _ in range(20):  # Reduced λ trials for speed in inner loop
            lambdas = rng.randn(K) / np.sqrt(K)

            try:
                L_F, Q_F = compute_criterion_matrices(
                    psi, phi, hams[:K], lambdas, driving_funcs, period, order=2
                )
                min_eig, _ = compute_min_eigenvalue_with_x_search(L_F, Q_F)

                if min_eig > best_min_eig:
                    best_min_eig = min_eig
            except Exception:
                pass

        eval_count[0] += 1
        obj_val = -best_min_eig if best_min_eig > -np.inf else 1e6

        if obj_val < best_so_far[0]:
            best_so_far[0] = obj_val
            if verbose and eval_count[0] % 20 == 0:
                print(f"  Iter {eval_count[0]}: best min_eig = {-best_so_far[0]:.6f}")

        return obj_val

    # Bounds: omegas in [0.3, 5.0], phases in [-π, π]
    bounds = [(0.3, 5.0)] * K + [(-np.pi, np.pi)] * K

    if verbose:
        print(f"Optimizing driving parameters for K={K}...")
        print(f"  Bounds: ω ∈ [0.3, 5.0], φ ∈ [-π, π]")
        print(f"  Max iterations: {maxiter}")

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter,
        workers=1, disp=False, atol=1e-6, tol=1e-6,
        mutation=(0.5, 1.0), recombination=0.7, popsize=15
    )

    optimal_omegas = result.x[:K]
    optimal_phases = result.x[K:2*K]
    optimal_strength = -result.fun if result.fun < 1e5 else None

    if verbose:
        print(f"\nOptimization complete after {eval_count[0]} evaluations")
        print(f"  Optimal min eigenvalue: {optimal_strength}")

    return {
        'optimal_omegas': optimal_omegas,
        'optimal_phases': optimal_phases,
        'optimal_strength': optimal_strength,
        'n_evaluations': eval_count[0],
        'optimization_result': {
            'success': result.success,
            'fun': result.fun,
            'nfev': result.nfev,
            'nit': result.nit,
        }
    }


# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def run_family_comparison(K=4, seed=42, n_lambda_trials=200):
    """Compare different driving function families."""

    print("="*70)
    print("DRIVING FUNCTION FAMILY COMPARISON")
    print("="*70)
    print()
    print(f"K = {K} operators")
    print(f"λ search trials = {n_lambda_trials}")
    print(f"Seed = {seed}")
    print()

    # Setup 5-qubit code
    d = 32
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0  # |00000⟩
    phi = create_5qubit_code_logical_zero()

    print("Generating GEO2LOCAL Hamiltonians (5-qubit chain)...")
    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d, k=51, ensemble="GEO2", nx=5, ny=1, seed=seed
    )
    hams = floquet.hamiltonians_to_numpy(hams_qutip)
    print(f"Generated {len(hams)} operators")
    print()

    period = 2 * np.pi

    # Define families to test
    families = {
        'monochromatic': create_monochromatic(K, omega=1.0),
        'monochromatic_offset': create_monochromatic_offset(K, omega=1.0, offset=1.0),
        'bichromatic': create_bichromatic(K, omega1=1.0, omega2=np.sqrt(2)),
        'polychromatic': create_polychromatic(K, base_omega=1.0, spread=0.5),
        'random_phases': create_random_phases(K, omega=1.0, seed=seed),
        'square_wave': create_square_wave(K, omega=1.0),
    }

    results = {}

    print("Testing driving families:")
    print("-" * 70)
    print(f"{'Family':<25} | {'Success':<8} | {'Min λ':<12} | {'||F||_F':<10}")
    print("-" * 70)

    for name, driving_funcs in families.items():
        result = evaluate_criterion_strength(
            psi, phi, hams, driving_funcs, period,
            order=2, n_lambda_trials=n_lambda_trials, seed=seed
        )

        results[name] = result

        status = "YES" if result['success'] else "NO"
        min_eig = f"{result['min_eigenvalue']:.6f}" if result['min_eigenvalue'] is not None else "N/A"
        f_norm = f"{result['F_analysis']['frobenius_norm']:.4f}"

        print(f"{name:<25} | {status:<8} | {min_eig:<12} | {f_norm:<10}")

    print("-" * 70)
    print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    successful = [name for name, r in results.items() if r['success']]
    failed = [name for name, r in results.items() if not r['success']]

    print(f"Successful families ({len(successful)}): {', '.join(successful) if successful else 'None'}")
    print(f"Failed families ({len(failed)}): {', '.join(failed) if failed else 'None'}")
    print()

    if successful:
        best_name = max(successful, key=lambda n: results[n]['min_eigenvalue'] or 0)
        print(f"Best performing: {best_name}")
        print(f"  Min eigenvalue: {results[best_name]['min_eigenvalue']:.6f}")
        print(f"  ||F||_F: {results[best_name]['F_analysis']['frobenius_norm']:.4f}")

    return results


def run_driving_optimization(K=4, seed=42, maxiter=100):
    """Optimize driving parameters."""

    print("="*70)
    print("DRIVING PARAMETER OPTIMIZATION")
    print("="*70)
    print()
    print(f"K = {K} operators")
    print(f"Max iterations = {maxiter}")
    print(f"Seed = {seed}")
    print()

    # Setup 5-qubit code
    d = 32
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    phi = create_5qubit_code_logical_zero()

    print("Generating GEO2LOCAL Hamiltonians (5-qubit chain)...")
    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d, k=51, ensemble="GEO2", nx=5, ny=1, seed=seed
    )
    hams = floquet.hamiltonians_to_numpy(hams_qutip)
    print(f"Generated {len(hams)} operators")
    print()

    period = 2 * np.pi

    # Run optimization
    print("Starting differential evolution optimization...")
    print()
    opt_result = optimize_driving_parameters(
        psi, phi, hams, period, K, maxiter=maxiter, seed=seed, verbose=True
    )

    # Compare with baseline (bichromatic)
    print()
    print("="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    print()

    baseline_driving = create_bichromatic(K)
    baseline_result = evaluate_criterion_strength(
        psi, phi, hams, baseline_driving, period,
        order=2, n_lambda_trials=200, seed=seed
    )

    print("Baseline (bichromatic):")
    print(f"  Success: {baseline_result['success']}")
    print(f"  Min eigenvalue: {baseline_result['min_eigenvalue']}")
    print(f"  ||F||_F: {baseline_result['F_analysis']['frobenius_norm']:.4f}")
    print()

    print("Optimized:")
    print(f"  Frequencies: {np.round(opt_result['optimal_omegas'], 4)}")
    print(f"  Phases: {np.round(opt_result['optimal_phases'], 4)}")
    print(f"  Min eigenvalue: {opt_result['optimal_strength']}")
    print()

    # Compute F matrix for optimized parameters
    opt_driving = create_optimized(K, opt_result['optimal_omegas'], opt_result['optimal_phases'])
    opt_F_matrix = compute_fourier_overlap_matrix(opt_driving, period)
    opt_F_analysis = analyze_fourier_matrix(opt_F_matrix)
    print(f"  ||F||_F: {opt_F_analysis['frobenius_norm']:.4f}")
    print()

    if opt_result['optimal_strength'] and baseline_result['min_eigenvalue']:
        improvement = (opt_result['optimal_strength'] - baseline_result['min_eigenvalue']) \
                      / abs(baseline_result['min_eigenvalue']) * 100
        print(f"Improvement: {improvement:+.1f}%")

    return {
        'baseline': baseline_result,
        'optimized': opt_result,
        'optimized_F_matrix': opt_F_matrix,
        'optimized_F_analysis': opt_F_analysis,
    }


def generate_comparison_figures(results, output_dir='results'):
    """Generate visualization figures for comparison results."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Extract data
    families = list(results.keys())
    successes = [results[f]['success'] for f in families]
    min_eigs = [results[f]['min_eigenvalue'] or 0 for f in families]
    f_norms = [results[f]['F_analysis']['frobenius_norm'] for f in families]

    # Figure 1: Bar chart of criterion strength
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['green' if s else 'red' for s in successes]

    ax = axes[0]
    bars = ax.bar(range(len(families)), min_eigs, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.set_ylabel('Min eigenvalue (criterion strength)')
    ax.set_title('Floquet Criterion Strength by Driving Family')

    ax = axes[1]
    ax.bar(range(len(families)), f_norms, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.set_ylabel('||F||_F (Frobenius norm)')
    ax.set_title('Fourier Overlap Matrix Norm')

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_path / f'driving_comparison_{timestamp}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved figure: {fig_path}")

    return fig_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize driving functions for Floquet QEC preparation'
    )
    parser.add_argument('--mode', choices=['compare', 'optimize', 'full'],
                        default='compare',
                        help='Mode: compare families, optimize params, or both')
    parser.add_argument('--K', type=int, default=4,
                        help='Number of operators to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n-lambda-trials', type=int, default=200,
                        help='Number of lambda search trials')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='Max iterations for optimization')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip figure generation')

    args = parser.parse_args()

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    if args.mode == 'compare':
        results = run_family_comparison(
            K=args.K, seed=args.seed, n_lambda_trials=args.n_lambda_trials
        )

        if not args.no_plot:
            generate_comparison_figures(results, output_dir='results')

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f'driving_comparison_K{args.K}_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved to {filename}")

    elif args.mode == 'optimize':
        results = run_driving_optimization(
            K=args.K, seed=args.seed, maxiter=args.maxiter
        )

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f'driving_optimization_K{args.K}_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved to {filename}")

    elif args.mode == 'full':
        print()
        print("#" * 70)
        print("# PHASE 1: FAMILY COMPARISON")
        print("#" * 70)
        print()

        comparison = run_family_comparison(
            K=args.K, seed=args.seed, n_lambda_trials=args.n_lambda_trials
        )

        if not args.no_plot:
            generate_comparison_figures(comparison, output_dir='results')

        print()
        print("#" * 70)
        print("# PHASE 2: DRIVING OPTIMIZATION")
        print("#" * 70)
        print()

        optimization = run_driving_optimization(
            K=args.K, seed=args.seed, maxiter=args.maxiter
        )

        results = {
            'comparison': comparison,
            'optimization': optimization,
            'args': vars(args),
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f'driving_full_analysis_K{args.K}_{timestamp}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved to {filename}")


if __name__ == '__main__':
    main()
