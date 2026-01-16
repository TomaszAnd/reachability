"""
Monte Carlo analysis functions for quantum reachability (compute only).

Pipeline Role:
This module implements all Monte Carlo experiments for estimating unreachability
probabilities and analyzing parameter sensitivity. It consumes Hamiltonians from
models.py, uses optimize.py to maximize S(λ), and outputs raw numerical data
for viz.py to render.

Strictly Compute-Only:
NO plotting code allowed. All visualization delegated to viz.py for clean
separation of concerns.

Key Probability Estimates:

1. **Unreachability Probability** (spectral criterion):
   P_unreach(d, K; τ) = Pr[ max_{λ ∈ [-1,1]ᴷ} S(λ) < τ ]

   Estimated via Monte Carlo sampling over Hamiltonian ensembles and target states.
   If S* < τ, the target is classified as unreachable.

2. **Moment Criterion** (τ-free, Gram matrix-based):
   Uses definiteness check on Gram matrix eigenvalues. Included for comparison
   with spectral criterion. Does NOT use threshold τ.

Statistical Tools:
- **Binomial SEM**: SEM(p) = √(p(1-p)/N) for error bars
- All estimates floor-clamped at settings.DISPLAY_FLOOR to handle log plots

Analysis Types:
- Unreachability vs (d, K) grids
- Threshold sensitivity (τ sweep)
- Optimizer method comparison (S* distributions)
- Convergence analysis (iterations vs probability)
- Landscape generation (S(λ₁, λ₂) over parameter grid)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import qutip
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter

from . import mathematics, models, optimize, settings
from .logging_utils import EnhancedDataLogger

logger = logging.getLogger(__name__)


def continuous_krylov_vs_spectral_comparison(
    d: int,
    k_values: List[int],
    ensemble: str,
    nks: int = 50,
    nst: int = 20,
    m: Optional[int] = None,
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare continuous Krylov scores with spectral overlap scores.

    This function computes both R*_Krylov (optimized Krylov score) and S*
    (optimized spectral overlap) for the same set of Hamiltonians and states,
    enabling direct comparison between the two continuous criteria.

    Args:
        d: Hilbert space dimension
        k_values: List of K values (number of Hamiltonians) to test
        ensemble: "GOE" or "GUE"
        nks: Number of Hamiltonian samples per K
        nst: Number of target states per Hamiltonian
        m: Krylov rank (default: d)
        method: Optimization method
        maxiter: Maximum iterations for optimization
        seed: Random seed

    Returns:
        Dictionary containing:
        - 'k_values': Array of K values
        - 'krylov_scores': List of arrays of R* values (one array per K)
        - 'spectral_scores': List of arrays of S* values (one array per K)
        - 'krylov_mean': Array of mean R* per K
        - 'krylov_std': Array of std R* per K
        - 'spectral_mean': Array of mean S* per K
        - 'spectral_std': Array of std S* per K
        - 'correlation': Array of correlation coefficients per K
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    if m is None:
        m = d

    logger.info(
        f"Continuous Krylov vs Spectral comparison: d={d}, K={k_values}, "
        f"ensemble={ensemble}, m={m}, trials={nks * nst}"
    )

    results = {
        "k_values": np.array(k_values),
        "krylov_scores": [],
        "spectral_scores": [],
        "krylov_mean": [],
        "krylov_std": [],
        "spectral_mean": [],
        "spectral_std": [],
        "correlation": [],
    }

    for K in k_values:
        if K < 2 or K >= d:
            logger.warning(f"Skipping K={K} (invalid range)")
            results["krylov_scores"].append(np.array([]))
            results["spectral_scores"].append(np.array([]))
            results["krylov_mean"].append(0.0)
            results["krylov_std"].append(0.0)
            results["spectral_mean"].append(0.0)
            results["spectral_std"].append(0.0)
            results["correlation"].append(0.0)
            continue

        logger.info(f"Computing for K={K}, m={min(m, K)}")

        krylov_scores = []
        spectral_scores = []

        # Use m = min(m, K) for consistency
        m_actual = min(m, K)

        for _ in range(nks):
            # Generate random Hamiltonian ensemble
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=rng.randint(0, 2**31 - 1))

            # Initial state
            psi = models.fock_state(d, 0)

            # Sample random target states
            targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

            for phi in targets:
                # Optimize Krylov score
                krylov_result = optimize.maximize_krylov_score(
                    psi,
                    phi,
                    hams,
                    m=m_actual,
                    method=method,
                    restarts=settings.DEFAULT_RESTARTS,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )
                krylov_scores.append(krylov_result["best_value"])

                # Optimize spectral overlap
                spectral_result = optimize.maximize_spectral_overlap(
                    psi,
                    phi,
                    hams,
                    method=method,
                    restarts=settings.DEFAULT_RESTARTS,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )
                spectral_scores.append(spectral_result["best_value"])

        # Convert to arrays
        krylov_arr = np.array(krylov_scores)
        spectral_arr = np.array(spectral_scores)

        # Compute statistics
        krylov_mean = float(np.mean(krylov_arr))
        krylov_std = float(np.std(krylov_arr))
        spectral_mean = float(np.mean(spectral_arr))
        spectral_std = float(np.std(spectral_arr))

        # Compute correlation
        if len(krylov_arr) > 1:
            correlation = float(np.corrcoef(krylov_arr, spectral_arr)[0, 1])
        else:
            correlation = 0.0

        # Store results
        results["krylov_scores"].append(krylov_arr)
        results["spectral_scores"].append(spectral_arr)
        results["krylov_mean"].append(krylov_mean)
        results["krylov_std"].append(krylov_std)
        results["spectral_mean"].append(spectral_mean)
        results["spectral_std"].append(spectral_std)
        results["correlation"].append(correlation)

        logger.info(
            f"  K={K}: R*={krylov_mean:.4f}±{krylov_std:.4f}, "
            f"S*={spectral_mean:.4f}±{spectral_std:.4f}, "
            f"corr={correlation:.4f}"
        )

    # Convert lists to arrays
    results["krylov_mean"] = np.array(results["krylov_mean"])
    results["krylov_std"] = np.array(results["krylov_std"])
    results["spectral_mean"] = np.array(results["spectral_mean"])
    results["spectral_std"] = np.array(results["spectral_std"])
    results["correlation"] = np.array(results["correlation"])

    logger.info("Continuous Krylov vs Spectral comparison complete")
    return results


def monte_carlo_unreachability(
    dims: List[int],
    ks: List[int],
    ensemble: str,
    tau: float = settings.DEFAULT_TAU,
    nks: int = settings.FULL_SAMPLING[0],
    nst: int = settings.FULL_SAMPLING[1],
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute unreachability probabilities across (dimension, k) parameter space.

    For each (d,k) pair, estimates P_unreach(d,k;τ) = Pr[max_λ S(λ) < τ]
    using Monte Carlo sampling over Hamiltonian ensembles and target states.

    Args:
        dims: List of Hilbert space dimensions
        ks: List of Hamiltonian counts
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method for maximizing S(λ)
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary mapping (d,k) → probability
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Monte Carlo analysis: {ensemble}, τ={tau}, nks={nks}, nst={nst}")

    results = {}

    for d in dims:
        for k in ks:
            if k >= d:  # Skip invalid parameter combinations
                continue

            logger.info(f"Computing P_unreach for d={d}, k={k}")

            unreachable_count = 0
            total_count = 0

            for _ in range(nks):
                # Generate random Hamiltonian ensemble
                hams = models.random_hamiltonian_ensemble(
                    d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
                )

                # Use |0⟩ as initial state
                psi = models.fock_state(d, 0)

                # Sample random target states
                targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

                for phi in targets:
                    # Maximize spectral overlap
                    result = optimize.maximize_spectral_overlap(
                        psi,
                        phi,
                        hams,
                        method=method,
                        restarts=1,
                        maxiter=maxiter,
                        seed=rng.randint(0, 2**31 - 1),
                    )

                    # Check unreachability
                    if result["best_value"] < tau:
                        unreachable_count += 1
                    total_count += 1

            # Compute probability
            probability = unreachable_count / total_count if total_count > 0 else 0.0
            results[(d, k)] = max(settings.DISPLAY_FLOOR, probability)

            logger.info(f"  P_unreach({d},{k}) = {probability:.6f}")

    return results


def probability_vs_k_single_d(
    d: int,
    ks: List[int],
    ensemble: str,
    tau: float = settings.DEFAULT_TAU,
    nks: int = settings.FULL_SAMPLING[0],
    nst: int = settings.FULL_SAMPLING[1],
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute unreachability probabilities vs K for a single fixed dimension.

    For fixed d, estimates P_unreach(d,K;τ) = Pr[max_λ S(λ) < τ] across
    a range of K values, with binomial SEM error bars.

    Args:
        d: Hilbert space dimension (fixed)
        ks: List of Hamiltonian counts to sweep (e.g., [1,2,...,14])
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method for maximizing S(λ)
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary {'k': ks, 'p': probs, 'err': sems} where:
        - k: Array of K values
        - p: Array of unreachability probabilities
        - err: Array of binomial SEM error bars
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Single-d K sweep: d={d}, K={ks}, {ensemble}, τ={tau}")

    probs = []
    sems = []

    for k in ks:
        if k < 2:
            logger.warning(f"Skipping k={k} (k < 2, minimum required)")
            probs.append(settings.DISPLAY_FLOOR)
            sems.append(0.0)
            continue
        if k >= d:
            logger.warning(f"Skipping k={k} (k >= d={d})")
            probs.append(settings.DISPLAY_FLOOR)
            sems.append(0.0)
            continue

        logger.info(f"Computing P_unreach for d={d}, k={k}")

        unreachable_count = 0
        total_count = 0

        for _ in range(nks):
            hams = models.random_hamiltonian_ensemble(
                d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
            )
            psi = models.fock_state(d, 0)

            targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

            for phi in targets:
                result = optimize.maximize_spectral_overlap(
                    psi,
                    phi,
                    hams,
                    method=method,
                    restarts=settings.DEFAULT_RESTARTS,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )

                total_count += 1
                if result["best_value"] < tau:
                    unreachable_count += 1

        # Compute probability and binomial SEM
        probability = unreachable_count / total_count if total_count > 0 else 0.0
        probability = max(settings.DISPLAY_FLOOR, probability)

        # Binomial SEM: sqrt(p(1-p)/N)
        sem = np.sqrt(probability * (1 - probability) / total_count) if total_count > 0 else 0.0

        probs.append(probability)
        sems.append(sem)

        logger.info(f"  P_unreach(d={d},k={k}) = {probability:.6f} ± {sem:.6f}")

    return {"k": np.array(ks), "p": np.array(probs), "err": np.array(sems)}


def probability_vs_iterations(
    d: int,
    k: int,
    ensemble: str,
    iters: Tuple[int, ...],
    tau: float = settings.DEFAULT_TAU,
    nks_iter: int = 50,
    nst_iter: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Analyze unreachability probability vs optimization iterations for L-BFGS-B.

    Computes P_unreach vs maxiter to study optimization convergence effects.

    Args:
        d: Hilbert space dimension
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        iters: Tuple of iteration counts to test
        tau: Unreachability threshold
        nks_iter: Number of Hamiltonian samples
        nst_iter: Number of target states per Hamiltonian
        seed: Random seed

    Returns:
        Dictionary with keys: 'iterations', 'probabilities', 'errors', 'runtimes'
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Iteration analysis: d={d}, k={k}, {ensemble}, iters={iters}")

    probabilities = []
    errors = []
    runtimes = []

    for max_iter in iters:
        logger.info(f"Testing maxiter={max_iter}")

        unreachable_count = 0
        total_count = 0
        iter_runtime = 0.0

        for _ in range(nks_iter):
            # Generate Hamiltonians
            hams = models.random_hamiltonian_ensemble(
                d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
            )
            psi = models.fock_state(d, 0)

            # Sample targets
            targets = models.random_states(nst_iter, d, seed=rng.randint(0, 2**31 - 1))

            for phi in targets:
                result = optimize.maximize_spectral_overlap(
                    psi,
                    phi,
                    hams,
                    method="L-BFGS-B",
                    restarts=1,
                    maxiter=max_iter,
                    seed=rng.randint(0, 2**31 - 1),
                )

                iter_runtime += result["runtime_s"]

                if result["best_value"] < tau:
                    unreachable_count += 1
                total_count += 1

        # Compute statistics
        p = unreachable_count / total_count if total_count > 0 else 0.0
        probabilities.append(p)
        errors.append(mathematics.compute_binomial_sem(p, total_count))
        runtimes.append(iter_runtime)

        logger.info(f"  maxiter={max_iter}: P={p:.4f} ± {errors[-1]:.4f}")

    return {
        "iterations": np.array(iters),
        "probabilities": np.array(probabilities),
        "errors": np.array(errors),
        "runtimes": np.array(runtimes),
    }


def probability_vs_tau(
    dims: List[int],
    taus: np.ndarray,
    k: int,
    ensemble: str,
    nks_tau: int = 60,
    nst_tau: int = 20,
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute unreachability probability vs threshold τ for different dimensions.

    Args:
        dims: List of dimensions to analyze
        taus: Array of threshold values
        k: Number of Hamiltonians (fixed)
        ensemble: "GOE" or "GUE"
        nks_tau: Number of Hamiltonian samples
        nst_tau: Number of target states per Hamiltonian
        method: Optimization method
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary mapping d → {'tau': taus, 'p': probabilities, 'err': errors}
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Tau sweep: k={k}, {ensemble}, dims={dims}")

    results = {}

    for d in dims:
        if k >= d:
            continue

        logger.info(f"Processing dimension d={d}")
        probs = []
        errors = []

        for tau in taus:
            unreachable_count = 0
            total_count = 0

            for _ in range(nks_tau):
                hams = models.random_hamiltonian_ensemble(
                    d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
                )
                psi = models.fock_state(d, 0)

                targets = models.random_states(nst_tau, d, seed=rng.randint(0, 2**31 - 1))

                for phi in targets:
                    result = optimize.maximize_spectral_overlap(
                        psi,
                        phi,
                        hams,
                        method=method,
                        restarts=1,
                        maxiter=maxiter,
                        seed=rng.randint(0, 2**31 - 1),
                    )

                    if result["best_value"] < tau:
                        unreachable_count += 1
                    total_count += 1

            p = unreachable_count / total_count if total_count > 0 else 0.0
            probs.append(p)
            errors.append(mathematics.compute_binomial_sem(p, total_count))

        results[d] = {"tau": taus, "p": np.array(probs), "err": np.array(errors)}

        logger.info(f"  d={d}: P range [{np.min(probs):.4f}, {np.max(probs):.4f}]")

    return results


def optimizer_comparison(
    dims: List[int],
    methods: List[str],
    Kmax: int,
    ensemble: str,
    tau: float = settings.DEFAULT_TAU,
    nks_opt: int = 40,
    nst_opt: int = 15,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compare unreachability across optimization methods for different dimensions.

    Args:
        dims: List of dimensions
        methods: List of optimization method names
        Kmax: Maximum number of Hamiltonians
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        nks_opt: Number of Hamiltonian samples
        nst_opt: Number of target states per Hamiltonian
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary: {d: {'k': ks, method: {'p': probs, 'err': errors}}}
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Optimizer comparison: {ensemble}, methods={methods}")

    results = {}

    for d in dims:
        logger.info(f"Dimension d={d}")
        results[d] = {"k": []}

        # Initialize method results
        for method in methods:
            results[d][method] = {"p": [], "err": []}

        # Compute for each k
        for k in range(2, min(Kmax + 1, d)):
            results[d]["k"].append(k)

            for method in methods:
                unreachable_count = 0
                total_count = 0

                for _ in range(nks_opt):
                    hams = models.random_hamiltonian_ensemble(
                        d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
                    )
                    psi = models.fock_state(d, 0)

                    targets = models.random_states(nst_opt, d, seed=rng.randint(0, 2**31 - 1))

                    for phi in targets:
                        result = optimize.maximize_spectral_overlap(
                            psi,
                            phi,
                            hams,
                            method=method,
                            restarts=1,
                            maxiter=maxiter,
                            seed=rng.randint(0, 2**31 - 1),
                        )

                        if result["best_value"] < tau:
                            unreachable_count += 1
                        total_count += 1

                p = unreachable_count / total_count if total_count > 0 else 0.0
                results[d][method]["p"].append(p)
                results[d][method]["err"].append(mathematics.compute_binomial_sem(p, total_count))

        # Convert to arrays
        results[d]["k"] = np.array(results[d]["k"])
        for method in methods:
            results[d][method]["p"] = np.array(results[d][method]["p"])
            results[d][method]["err"] = np.array(results[d][method]["err"])

    return results


def optimizer_Sstar_comparison(
    dims: List[int],
    methods: List[str],
    k: int,
    ensemble: str,
    nks_opt: int = 40,
    nst_opt: int = 15,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compare S* (max spectral overlap) distributions across optimization methods.

    Collects S* values (not unreachability probabilities) for each optimizer method
    across different dimensions, enabling direct comparison of optimizer performance.

    Args:
        dims: List of dimensions to test
        methods: List of optimization method names
        k: Fixed number of Hamiltonians
        ensemble: "GOE" or "GUE"
        nks_opt: Number of Hamiltonian samples
        nst_opt: Number of target states per Hamiltonian
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary: {method: {d: {'mean_S': float, 'sem_S': float}}}
        where mean_S and sem_S are statistics of S* distribution
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Optimizer S* comparison: {ensemble}, methods={methods}, k={k}")

    results = {method: {} for method in methods}

    for d in dims:
        if k >= d:
            logger.warning(f"Skipping d={d} (k={k} >= d)")
            continue

        logger.info(f"Dimension d={d}")

        for method in methods:
            S_star_values = []

            for _ in range(nks_opt):
                hams = models.random_hamiltonian_ensemble(
                    d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
                )
                psi = models.fock_state(d, 0)

                targets = models.random_states(nst_opt, d, seed=rng.randint(0, 2**31 - 1))

                for phi in targets:
                    result = optimize.maximize_spectral_overlap(
                        psi,
                        phi,
                        hams,
                        method=method,
                        restarts=1,
                        maxiter=maxiter,
                        seed=rng.randint(0, 2**31 - 1),
                    )

                    S_star_values.append(result["best_value"])

            # Compute mean and SEM of S* distribution
            S_star_array = np.array(S_star_values)
            mean_S = np.mean(S_star_array)
            sem_S = np.std(S_star_array, ddof=1) / np.sqrt(len(S_star_array))

            results[method][d] = {"mean_S": float(mean_S), "sem_S": float(sem_S)}

            logger.info(f"  {method}, d={d}: mean(S*)={mean_S:.4f} ± {sem_S:.4f}")

    return results


def landscape_spectral_overlap(
    d: int,
    k: int,
    ensemble: str,
    grid: int = settings.DEFAULT_GRID_SIZE,
    n_targets: int = settings.DEFAULT_LANDSCAPE_TARGETS,
    lambda_range: Tuple[float, float] = (-1.5, 1.5),
    seed: Optional[int] = None,
    no_smooth: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectral overlap landscape S(λ₁, λ₂) over parameter grid.

    Fixes λ₃, λ₄, ... = 0 and varies (λ₁, λ₂) to compute S(λ₁, λ₂) surface.
    This is the actual spectral overlap values, not unreachability probabilities.

    Args:
        d: Hilbert space dimension
        k: Number of Hamiltonians (k ≥ 2)
        ensemble: "GOE" or "GUE"
        grid: Grid resolution (grid × grid points). Will be made odd if even.
        n_targets: Number of target states to average over
        lambda_range: (min, max) range for λ₁, λ₂
        seed: Random seed
        no_smooth: If True, skip Gaussian smoothing and interpolation (default: False)

    Returns:
        (L1, L2, S) where:
        - L1, L2: Meshgrid arrays for λ₁, λ₂
        - S: Grid of spectral overlap values S(λ₁, λ₂)
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    # Enforce odd grid so λ=0 is exactly sampled (center pixel)
    if grid % 2 == 0:
        grid += 1
        logger.info(f"Grid size adjusted to {grid} (must be odd for centered λ=0)")

    logger.info(f"Computing landscape: d={d}, k={k}, {ensemble}, grid={grid}×{grid}")

    # Generate fixed Hamiltonian ensemble
    hams = models.random_hamiltonian_ensemble(d, k, ensemble, seed=rng.randint(0, 2**31 - 1))

    # Setup parameter grid with exact λ=0 at center
    lambda_vals = np.linspace(lambda_range[0], lambda_range[1], grid)
    L1, L2 = np.meshgrid(lambda_vals, lambda_vals)
    S_grid = np.zeros_like(L1)

    # Fixed initial state
    psi = models.fock_state(d, 0)

    # Generate target states to average over
    targets = models.random_states(n_targets, d, seed=rng.randint(0, 2**31 - 1))

    # Compute S(λ₁, λ₂) at each grid point
    for i in range(grid):
        for j in range(grid):
            overlap_sum = 0.0

            # Set parameters: λ₁, λ₂ from grid, others = 0
            lambdas = np.zeros(k)
            lambdas[0] = L1[i, j]
            if k > 1:
                lambdas[1] = L2[i, j]

            # Average over target states
            for phi in targets:
                try:
                    S = mathematics.spectral_overlap(lambdas, psi, phi, hams)
                    overlap_sum += S
                except Exception as e:
                    logger.debug(
                        f"Landscape evaluation failed at ({L1[i,j]:.2f}, {L2[i,j]:.2f}): {e}"
                    )
                    continue

            S_grid[i, j] = overlap_sum / n_targets

        # Progress logging
        if (i + 1) % (grid // 4) == 0:
            logger.info(f"  Progress: {(i + 1) * 100 // grid}%")

    # Conditionally apply smoothing and interpolation
    if no_smooth:
        # Return raw grid without smoothing or interpolation
        S_final = np.clip(S_grid, 0, 1)
        logger.info(f"Landscape complete (no smoothing): S range [{np.min(S_final):.4f}, {np.max(S_final):.4f}]")
        return L1, L2, S_final
    else:
        # Apply smoothing for cleaner visualization (legacy behavior)
        S_smooth = gaussian_filter(S_grid, sigma=settings.LANDSCAPE_SMOOTH_SIGMA)

        # Interpolate to finer grid for smoother rendering
        fine_grid = int(grid * settings.LANDSCAPE_FINE_MULTIPLIER)
        lambda_fine = np.linspace(lambda_range[0], lambda_range[1], fine_grid)
        L1_fine, L2_fine = np.meshgrid(lambda_fine, lambda_fine)

        # Spline interpolation
        try:
            spline = RectBivariateSpline(lambda_vals, lambda_vals, S_smooth, kx=3, ky=3)
            S_fine = spline(lambda_fine, lambda_fine)
            S_fine = np.clip(S_fine, 0, 1)  # Ensure valid range
        except Exception as e:
            logger.warning(f"Spline interpolation failed: {e}, using original grid")
            L1_fine, L2_fine, S_fine = L1, L2, S_smooth

        logger.info(f"Landscape complete: S range [{np.min(S_fine):.4f}, {np.max(S_fine):.4f}]")
        return L1_fine, L2_fine, S_fine


def punreach_vs_dimension_K(
    d_range: List[int],
    K_range: List[int],
    ensemble: str,
    epsilons: List[float],
    nks: int = 40,
    nst: int = 15,
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute P(unreachability) vs (dimension, K) heatmaps for different ε.

    Computes P(S* < ε) over (d, K) parameter space for each threshold ε.

    Args:
        d_range: List of dimensions
        K_range: List of K values
        ensemble: "GOE" or "GUE"
        epsilons: List of unreachability thresholds
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary with keys: 'd_vals', 'K_vals', 'eps_{ε}' for each ε
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"P(unreachability) heatmaps: {ensemble}, epsilons={epsilons}")

    # Create meshgrid
    d_vals = np.array(d_range)
    K_vals = np.array(K_range)

    results = {"d_vals": d_vals, "K_vals": K_vals}

    for eps in epsilons:
        logger.info(f"Computing for ε={eps}")

        P_grid = np.zeros((len(d_vals), len(K_vals)))

        for i, d in enumerate(d_vals):
            for j, K in enumerate(K_vals):
                if K >= d:  # Invalid parameter combination
                    P_grid[i, j] = np.nan
                    continue

                unreachable_count = 0
                total_count = 0

                for _ in range(nks):
                    hams = models.random_hamiltonian_ensemble(
                        d, K, ensemble, seed=rng.randint(0, 2**31 - 1)
                    )
                    psi = models.fock_state(d, 0)

                    targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

                    for phi in targets:
                        result = optimize.maximize_spectral_overlap(
                            psi,
                            phi,
                            hams,
                            method=method,
                            restarts=1,
                            maxiter=maxiter,
                            seed=rng.randint(0, 2**31 - 1),
                        )

                        if result["best_value"] < eps:
                            unreachable_count += 1
                        total_count += 1

                P_grid[i, j] = unreachable_count / total_count if total_count > 0 else 0.0

        results[f"eps_{eps:.2f}".replace(".", "_")] = P_grid

    return results


def collect_Sstar_for_dims(
    dims: List[int],
    ensemble: str,
    k: int = 4,
    method: str = "L-BFGS-B",
    nks: int = settings.BIG_NKS,
    nst: int = settings.BIG_NST,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    For each d, collect an array of S* values (max spectral overlap) from random trials.

    Uses the same pipeline as optimizer_Sstar_comparison but collects raw S* values
    instead of computing statistics, for histogram generation.

    Args:
        dims: List of dimensions to analyze
        ensemble: "GOE" or "GUE"
        k: Fixed number of Hamiltonians
        method: Optimization method (default: L-BFGS-B)
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        maxiter: Maximum optimization iterations
        seed: Random seed

    Returns:
        Dictionary mapping d → np.array([S* values])
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Collecting S* values: {ensemble}, k={k}, method={method}")

    results = {}

    for d in dims:
        if k >= d:
            logger.warning(f"Skipping d={d} (k={k} >= d)")
            continue

        logger.info(f"Dimension d={d}")
        S_star_values = []

        for _ in range(nks):
            hams = models.random_hamiltonian_ensemble(
                d, k, ensemble, seed=rng.randint(0, 2**31 - 1)
            )
            psi = models.fock_state(d, 0)

            targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

            for phi in targets:
                result = optimize.maximize_spectral_overlap(
                    psi,
                    phi,
                    hams,
                    method=method,
                    restarts=1,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )

                S_star_values.append(result["best_value"])

        results[d] = np.array(S_star_values)
        logger.info(f"  d={d}: collected {len(S_star_values)} S* values")

    return results


def moment_criterion_probabilities(
    dims: List[int],
    k_values: List[int],
    ensemble: str,
    nks: int = 80,
    nst: int = 20,
    seed: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute P(unreachability) using the moment criterion (Gram matrix definiteness).

    Implements the moment-based reachability criterion using Gram matrix
    eigenvalue analysis for comparison with the spectral criterion.

    Args:
        dims: List of Hilbert space dimensions
        k_values: List of Hamiltonian counts
        ensemble: "GOE" or "GUE"
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping (d,k) → P(unreachability)
    """
    from scipy.linalg import null_space

    from . import models

    # Initialize reproducible random state
    if seed is not None:
        models.setup_environment(seed)

    log = logging.getLogger(__name__)
    log.info(
        f"Computing moment criterion probabilities: ensemble={ensemble}, dims={dims}, k_values={k_values}"
    )

    def expect_array_of_operators(ops, state):
        """Helper function from reference notebook."""
        if qutip.isoper(ops):
            return np.real_if_close(qutip.expect(ops, state))
        else:
            ret = [expect_array_of_operators(subops, state) for subops in ops]
            return np.array(ret)

    def check_eigenvalues(matrix):
        """Check if matrix has all positive or all negative eigenvalues."""
        eigenvalues = np.linalg.eigvalsh(matrix)
        return np.all(eigenvalues > 0) or np.all(eigenvalues < 0)

    results = {}

    for d in dims:
        log.info(f"Processing dimension d={d}")

        for k in k_values:
            if k >= d:  # Skip invalid cases
                continue

            log.info(f"  Computing for k={k}")

            unreachable_count = 0
            total_count = 0

            # Generate initial state (ground state)
            init_state = qutip.fock(d, 0)

            for _ in range(nks):
                # Generate k random projectors (Hamiltonians)
                if ensemble == "GOE":
                    hs = [qutip.ket2dm(qutip.rand_ket(d, seed=None)) for _ in range(k)]
                elif ensemble == "GUE":
                    hs = [qutip.ket2dm(qutip.rand_ket(d, seed=None)) for _ in range(k)]
                else:
                    raise ValueError(f"Unknown ensemble: {ensemble}")

                # Compute anticommutators {H_i, H_j}/2
                hs_anticomms = [[None for _ in range(k)] for _ in range(k)]
                for i in range(k):
                    for j in range(k):
                        hs_anticomms[i][j] = (hs[i] @ hs[j] + hs[j] @ hs[i]) / 2

                # Compute expectations for initial state
                energies_zero = expect_array_of_operators(hs, init_state)
                energy_sq_zero = expect_array_of_operators(hs_anticomms, init_state)

                # Test nst random target states
                for _ in range(nst):
                    target_state = qutip.rand_ket(d, seed=None)

                    # Compute energy differences
                    energies_state = expect_array_of_operators(hs, target_state)
                    diff = energies_state - energies_zero

                    # Find null space of energy difference
                    kernel = null_space(diff.reshape(1, -1))

                    if kernel.size > 0:
                        # Compute second moment differences
                        energy_sq_state = expect_array_of_operators(hs_anticomms, target_state)
                        m_final = kernel.T @ (energy_sq_state - energy_sq_zero) @ kernel

                        # Check definiteness (moment criterion)
                        if check_eigenvalues(m_final):
                            unreachable_count += 1

                    total_count += 1

            # Store probability
            probability = unreachable_count / total_count if total_count > 0 else 0.0
            results[(d, k)] = probability

            log.info(f"    P(unreachable) = {probability:.4f} ({unreachable_count}/{total_count})")

    log.info("Moment criterion computation complete")
    return results


def monte_carlo_unreachability_vs_m(
    d: int,
    m_values: List[int],
    K: int,
    ensemble: str,
    criteria: Tuple[str, ...] = ("krylov", "spectral", "moment"),
    tau: float = settings.DEFAULT_TAU,
    nks: int = settings.FULL_SAMPLING[0],
    nst: int = settings.FULL_SAMPLING[1],
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
    rank_tol: float = settings.KRYLOV_RANK_TOL,
) -> Dict[str, Any]:
    """
    Compute P(unreachability) vs Krylov rank m for 3 criteria (overlay plot).

    For fixed (d, K), sweeps Krylov rank m and evaluates:
    - Krylov criterion: Check if φ ∈ K_m(H, ψ) via rank test
    - Spectral overlap: Check if max_λ S(λ) < τ
    - Moment criterion: Gram matrix definiteness check

    Args:
        d: Hilbert space dimension
        m_values: List of Krylov ranks to sweep (e.g., [1, 2, ..., d])
        K: Number of Hamiltonians (fixed for this sweep)
        ensemble: "GOE" or "GUE"
        criteria: Tuple of criteria to evaluate (default: all 3)
        tau: Threshold for spectral overlap criterion
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method for spectral overlap
        maxiter: Max iterations for spectral optimization
        seed: Random seed
        rank_tol: Rank tolerance for Krylov criterion

    Returns:
        {
            'm': np.array(m_values),
            'p_krylov': np.array([...]),      # P(unreachable) via Krylov
            'err_krylov': np.array([...]),    # Binomial SEM
            'p_spectral': np.array([...]),    # P(unreachable) via spectral
            'err_spectral': np.array([...]),
            'p_old': np.array([...]),         # P(unreachable) via old
            'err_old': np.array([...]),
            'mean_best_overlap_spectral': np.array([...]),  # Mean S* for spectral (per m)
            'sem_best_overlap_spectral': np.array([...]),   # SEM of S* for spectral (per m)
        }
        (Only includes keys for requested criteria; best_overlap stats only for spectral)
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(
        f"Three-criteria m-sweep: d={d}, K={K}, {ensemble}, m_values={m_values}, criteria={criteria}"
    )

    # Initialize results
    results: Dict[str, Any] = {"m": np.array(m_values)}

    # Process each criterion
    for criterion in criteria:
        logger.info(f"  Processing criterion: {criterion}")
        probs = []
        errs = []
        # For spectral criterion, also collect best overlap statistics
        mean_overlaps = [] if criterion == "spectral" else None
        sem_overlaps = [] if criterion == "spectral" else None

        for m in m_values:
            if m < 1 or m > d:
                logger.warning(f"    Skipping m={m} (out of range [1, {d}])")
                probs.append(settings.DISPLAY_FLOOR)
                errs.append(0.0)
                if criterion == "spectral":
                    mean_overlaps.append(0.0)
                    sem_overlaps.append(0.0)
                continue

            logger.info(f"    Computing for m={m}")

            unreachable_count = 0
            total_count = 0
            # For spectral: collect all best_value results
            spectral_best_values = [] if criterion == "spectral" else None

            for _ in range(nks):
                # Generate random Hamiltonian ensemble
                hams = models.random_hamiltonian_ensemble(
                    d, K, ensemble, seed=rng.randint(0, 2**31 - 1)
                )

                # Initial state: |0⟩
                psi = models.fock_state(d, 0)

                # Sample random target states
                targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

                # Pre-compute moment criterion data (once per Hamiltonian ensemble)
                if criterion == "moment":
                    from scipy.linalg import null_space

                    def expect_array_of_operators(ops, state):
                        if qutip.isoper(ops):
                            return np.real_if_close(qutip.expect(ops, state))
                        else:
                            ret = [expect_array_of_operators(subops, state) for subops in ops]
                            return np.array(ret)

                    def check_eigenvalues(matrix):
                        eigenvalues = np.linalg.eigvalsh(matrix)
                        return np.all(eigenvalues > 0) or np.all(eigenvalues < 0)

                    # Compute anticommutators (once per ensemble)
                    hs_anticomms = [[None for _ in range(K)] for _ in range(K)]
                    for i in range(K):
                        for j in range(K):
                            hs_anticomms[i][j] = (hams[i] @ hams[j] + hams[j] @ hams[i]) / 2

                    # Compute expectations for initial state (once per ensemble)
                    energies_zero = expect_array_of_operators(hams, psi)
                    energy_sq_zero = expect_array_of_operators(hs_anticomms, psi)

                for phi in targets:
                    is_unreach = False

                    if criterion == "krylov":
                        # Construct single Hamiltonian with random coefficients
                        lambdas = rng.uniform(-1.0, 1.0, K)
                        H_combined = sum(lam * H for lam, H in zip(lambdas, hams))
                        # Convert to numpy array
                        H_matrix = H_combined.full() if hasattr(H_combined, "full") else H_combined
                        is_unreach = mathematics.is_unreachable_krylov(
                            H_matrix, psi, phi, m, rank_tol=rank_tol
                        )

                    elif criterion == "spectral":
                        # Maximize spectral overlap
                        result = optimize.maximize_spectral_overlap(
                            psi,
                            phi,
                            hams,
                            method=method,
                            restarts=settings.DEFAULT_RESTARTS,
                            maxiter=maxiter,
                            seed=rng.randint(0, 2**31 - 1),
                        )
                        # Collect best_value for statistics
                        spectral_best_values.append(result["best_value"])
                        is_unreach = result["best_value"] < tau

                    elif criterion == "moment":
                        # Moment criterion (Gram matrix-based) - uses pre-computed data
                        # Compute energy differences
                        energies_state = expect_array_of_operators(hams, phi)
                        diff = energies_state - energies_zero

                        # Find null space
                        kernel = null_space(diff.reshape(1, -1))

                        if kernel.size > 0:
                            energy_sq_state = expect_array_of_operators(hs_anticomms, phi)
                            m_final = kernel.T @ (energy_sq_state - energy_sq_zero) @ kernel

                            if check_eigenvalues(m_final):
                                is_unreach = True

                    if is_unreach:
                        unreachable_count += 1
                    total_count += 1

            # Compute probability and binomial SEM
            p = unreachable_count / total_count if total_count > 0 else 0.0
            p = max(settings.DISPLAY_FLOOR, p)
            sem = mathematics.compute_binomial_sem(p, total_count) if total_count > 0 else 0.0

            probs.append(p)
            errs.append(sem)

            # For spectral: compute mean and SEM of best overlap values
            if criterion == "spectral" and spectral_best_values:
                overlap_arr = np.array(spectral_best_values)
                mean_overlap = float(np.mean(overlap_arr))
                sem_overlap = float(np.std(overlap_arr, ddof=1) / np.sqrt(len(overlap_arr)))
                mean_overlaps.append(mean_overlap)
                sem_overlaps.append(sem_overlap)
                logger.info(
                    f"      P_unreach(m={m}) = {p:.6f} ± {sem:.6f}, "
                    f"mean(S*) = {mean_overlap:.6f} ± {sem_overlap:.6f}"
                )
            else:
                logger.info(f"      P_unreach(m={m}) = {p:.6f} ± {sem:.6f}")

        # Store results for this criterion
        results[f"p_{criterion}"] = np.array(probs)
        results[f"err_{criterion}"] = np.array(errs)

        # Store spectral overlap statistics if applicable
        if criterion == "spectral":
            results["mean_best_overlap_spectral"] = np.array(mean_overlaps)
            results["sem_best_overlap_spectral"] = np.array(sem_overlaps)

    logger.info("Three-criteria m-sweep complete")
    return results


def monte_carlo_unreachability_vs_K_three(
    d: int,
    k_values: List[int],
    ensemble: str,
    tau: float = settings.DEFAULT_TAU,
    krylov_m_strategy: str = settings.DEFAULT_KRYLOV_M_STRATEGY,
    krylov_m_fixed: Optional[int] = None,
    nks: int = settings.FULL_SAMPLING[0],
    nst: int = settings.FULL_SAMPLING[1],
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
    rank_tol: float = settings.KRYLOV_RANK_TOL,
) -> Dict[str, Any]:
    """
    Compute P(unreachability) vs K for all 3 criteria (replica of single-d plot).

    Sweeps K (number of Hamiltonians) and evaluates all 3 criteria:
    - Krylov: m determined by strategy ("K" → m=K, or fixed value)
    - Spectral: max_λ S(λ) < τ
    - Moment: Gram matrix-based criterion

    Args:
        d: Hilbert space dimension (fixed)
        k_values: List of K values to sweep (e.g., [1, 2, ..., 14])
        ensemble: "GOE" or "GUE"
        tau: Threshold for spectral overlap
        krylov_m_strategy: "K" (m=K) or "fixed" (use krylov_m_fixed)
        krylov_m_fixed: Fixed m value if strategy="fixed"
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method for spectral overlap
        maxiter: Max iterations for spectral optimization
        seed: Random seed
        rank_tol: Rank tolerance for Krylov criterion

    Returns:
        {
            'k': np.array(k_values),
            'p_krylov': [...], 'err_krylov': [...],
            'p_spectral': [...], 'err_spectral': [...],
            'p_old': [...], 'err_old': [...],
            'mean_best_overlap_spectral': [...],  # Mean S* for spectral (per K)
            'sem_best_overlap_spectral': [...],   # SEM of S* for spectral (per K)
            'm_label': str  # e.g., "m = K" or "m = 5 (fixed)"
        }
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    # Determine m_label for annotation
    if krylov_m_strategy == "K":
        m_label = "m = K"
    else:
        m_label = f"m = {krylov_m_fixed} (fixed)"

    logger.info(
        f"Three-criteria K-sweep: d={d}, {ensemble}, k_values={k_values}, "
        f"tau={tau}, Krylov strategy: {m_label}"
    )

    # Initialize results
    results: Dict[str, Any] = {"k": np.array(k_values), "m_label": m_label}

    # Storage for each criterion
    probs_krylov, errs_krylov = [], []
    probs_spectral, errs_spectral = [], []
    probs_old, errs_old = [], []
    mean_overlaps_spectral, sem_overlaps_spectral = [], []
    mean_overlaps_krylov, sem_overlaps_krylov = [], []  # NEW: Krylov score statistics

    for K in k_values:
        if K >= d:
            logger.warning(f"  Skipping K={K} (K >= d={d})")
            probs_krylov.append(settings.DISPLAY_FLOOR)
            errs_krylov.append(0.0)
            probs_spectral.append(settings.DISPLAY_FLOOR)
            errs_spectral.append(0.0)
            probs_old.append(settings.DISPLAY_FLOOR)
            errs_old.append(0.0)
            mean_overlaps_spectral.append(0.0)
            sem_overlaps_spectral.append(0.0)
            mean_overlaps_krylov.append(0.0)
            sem_overlaps_krylov.append(0.0)
            continue

        # Determine Krylov rank m for this K
        if krylov_m_strategy == "K":
            m = K
        else:
            m = krylov_m_fixed if krylov_m_fixed is not None else K

        # Clamp m to valid range
        m = max(1, min(m, d))

        logger.info(f"  Computing for K={K} (Krylov m={m})")

        # Counters for criteria
        unreach_old = 0
        unreach_spectral = 0  # Counted inline for spectral
        total_count = 0
        # Collect best values for τ-dependent criteria (spectral, krylov)
        spectral_best_values = []
        krylov_best_values = []

        for _ in range(nks):
            # Generate random Hamiltonian ensemble
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=rng.randint(0, 2**31 - 1))

            # Initial state
            psi = models.fock_state(d, 0)

            # Sample random target states
            targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

            # Pre-compute old criterion data (once per Hamiltonian ensemble)
            from scipy.linalg import null_space

            def expect_array_of_operators(ops, state):
                if qutip.isoper(ops):
                    return np.real_if_close(qutip.expect(ops, state))
                else:
                    ret = [expect_array_of_operators(subops, state) for subops in ops]
                    return np.array(ret)

            def check_eigenvalues(matrix):
                eigenvalues = np.linalg.eigvalsh(matrix)
                return np.all(eigenvalues > 0) or np.all(eigenvalues < 0)

            # Compute anticommutators (once per ensemble)
            hs_anticomms = [[None for _ in range(K)] for _ in range(K)]
            for i in range(K):
                for j in range(K):
                    hs_anticomms[i][j] = (hams[i] @ hams[j] + hams[j] @ hams[i]) / 2

            # Compute expectations for initial state (once per ensemble)
            energies_zero = expect_array_of_operators(hams, psi)
            energy_sq_zero = expect_array_of_operators(hs_anticomms, psi)

            for phi in targets:
                # --- Continuous Krylov criterion (optimized over λ, τ-dependent) ---
                krylov_result = optimize.maximize_krylov_score(
                    psi,
                    phi,
                    hams,
                    m=m,
                    method=method,
                    restarts=settings.DEFAULT_RESTARTS,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )
                krylov_best_values.append(krylov_result["best_value"])

                # --- Spectral overlap criterion ---
                result = optimize.maximize_spectral_overlap(
                    psi,
                    phi,
                    hams,
                    method=method,
                    restarts=settings.DEFAULT_RESTARTS,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )
                # Collect best_value for statistics
                spectral_best_values.append(result["best_value"])
                if result["best_value"] < tau:
                    unreach_spectral += 1

                # --- Moment criterion (uses pre-computed data) ---
                energies_state = expect_array_of_operators(hams, phi)
                diff = energies_state - energies_zero

                kernel = null_space(diff.reshape(1, -1))

                if kernel.size > 0:
                    energy_sq_state = expect_array_of_operators(hs_anticomms, phi)
                    m_final = kernel.T @ (energy_sq_state - energy_sq_zero) @ kernel

                    if check_eigenvalues(m_final):
                        unreach_old += 1

                total_count += 1

        # Compute probabilities and SEMs for τ-dependent criteria (krylov, spectral)
        # Threshold Krylov scores at tau
        krylov_arr = np.array(krylov_best_values)
        unreach_krylov = np.sum(krylov_arr < tau)
        p_krylov = max(settings.DISPLAY_FLOOR, unreach_krylov / total_count if total_count > 0 else 0.0)
        sem_krylov = mathematics.compute_binomial_sem(p_krylov, total_count) if total_count > 0 else 0.0

        # Threshold Spectral scores at tau (already counted inline, just compute probability)
        p_spectral = max(
            settings.DISPLAY_FLOOR, unreach_spectral / total_count if total_count > 0 else 0.0
        )
        sem_spectral = (
            mathematics.compute_binomial_sem(p_spectral, total_count) if total_count > 0 else 0.0
        )

        # Moment criterion (τ-independent)
        p_old = max(settings.DISPLAY_FLOOR, unreach_old / total_count if total_count > 0 else 0.0)
        sem_old = mathematics.compute_binomial_sem(p_old, total_count) if total_count > 0 else 0.0

        probs_krylov.append(p_krylov)
        errs_krylov.append(sem_krylov)
        probs_spectral.append(p_spectral)
        errs_spectral.append(sem_spectral)
        probs_old.append(p_old)
        errs_old.append(sem_old)

        # Compute mean and SEM of spectral best overlap values
        if spectral_best_values:
            overlap_arr = np.array(spectral_best_values)
            mean_overlap_spectral = float(np.mean(overlap_arr))
            sem_overlap_spectral = float(np.std(overlap_arr, ddof=1) / np.sqrt(len(overlap_arr)))
        else:
            mean_overlap_spectral = 0.0
            sem_overlap_spectral = 0.0
        mean_overlaps_spectral.append(mean_overlap_spectral)
        sem_overlaps_spectral.append(sem_overlap_spectral)

        # Compute mean and SEM of Krylov best scores (NEW)
        if krylov_best_values:
            krylov_score_arr = np.array(krylov_best_values)
            mean_overlap_krylov = float(np.mean(krylov_score_arr))
            sem_overlap_krylov = float(np.std(krylov_score_arr, ddof=1) / np.sqrt(len(krylov_score_arr)))
        else:
            mean_overlap_krylov = 0.0
            sem_overlap_krylov = 0.0
        mean_overlaps_krylov.append(mean_overlap_krylov)
        sem_overlaps_krylov.append(sem_overlap_krylov)

        logger.info(
            f"    K={K}: P_krylov={p_krylov:.4f}±{sem_krylov:.4f}, "
            f"P_spectral={p_spectral:.4f}±{sem_spectral:.4f}, "
            f"P_old={p_old:.4f}±{sem_old:.4f}, "
            f"mean(S*)={mean_overlap_spectral:.4f}±{sem_overlap_spectral:.4f}, "
            f"mean(R*)={mean_overlap_krylov:.4f}±{sem_overlap_krylov:.4f}"
        )

    # Store results
    results["p_krylov"] = np.array(probs_krylov)
    results["err_krylov"] = np.array(errs_krylov)
    results["p_spectral"] = np.array(probs_spectral)
    results["err_spectral"] = np.array(errs_spectral)
    results["p_old"] = np.array(probs_old)
    results["err_old"] = np.array(errs_old)
    results["mean_best_overlap_spectral"] = np.array(mean_overlaps_spectral)
    results["sem_best_overlap_spectral"] = np.array(sem_overlaps_spectral)
    results["mean_best_overlap_krylov"] = np.array(mean_overlaps_krylov)  # NEW
    results["sem_best_overlap_krylov"] = np.array(sem_overlaps_krylov)  # NEW

    logger.info("Three-criteria K-sweep complete")
    return results


def monte_carlo_unreachability_vs_density(
    dims: List[int],
    rho_max: float,
    rho_step: float,
    taus: List[float],
    ensemble: str,
    k_cap: int = 200,
    nks: int = settings.FULL_SAMPLING[0],
    nst: int = settings.FULL_SAMPLING[1],
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
    rank_tol: float = settings.KRYLOV_RANK_TOL,
    data_logger: Optional[EnhancedDataLogger] = None,
    optimize_lambda: bool = True,
    **ensemble_params,
) -> Dict[str, Any]:
    """
    Compute P(unreachability) vs density ρ=K/d² for all 3 criteria across multiple dimensions and τ values.

    This function efficiently handles multiple τ values for the spectral criterion by:
    1. Running MC once per (d, K) to collect raw spectral best overlap values
    2. Thresholding these values at each τ without rerunning randomness
    3. For old/krylov criteria (τ-independent), computing once per (d, K)

    Key difference from K-sweep: sweeps over ρ grid, computes K = min(round(ρ × d²), k_cap),
    and removes K < d constraint (uses m = min(K, d) for Krylov).

    Args:
        dims: List of dimensions to analyze (e.g., [20, 30, 40, 50])
        rho_max: Maximum density value (e.g., 0.15)
        rho_step: Density step size (e.g., 0.01)
        taus: List of τ thresholds for spectral criterion (e.g., [0.90, 0.95, 0.99])
        ensemble: "GOE", "GUE", or "GEO2"
        k_cap: Maximum K value cap (default: 200)
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method for spectral overlap
        maxiter: Max iterations for spectral optimization
        seed: Random seed
        rank_tol: Rank tolerance for Krylov criterion
        data_logger: Optional EnhancedDataLogger for logging raw trial data
        optimize_lambda: If True (default), optimize λ to maximize Spectral/Krylov criteria.
                        If False, use fixed random λ ~ N(0, 1/√K) without optimization.
        **ensemble_params: Ensemble-specific parameters (for GEO2: nx, ny, periodic)

    Returns:
        {
            'dims': dims,
            'taus': taus,
            'rho_grid': rho_grid,  # Common ρ grid used
            # For each (d, tau, criterion): dict with keys 'K', 'rho', 'p', 'err', 'mean_overlap', 'sem_overlap'
            (d, tau, criterion): {
                'K': np.array([K values]),
                'rho': np.array([K/d² values]),
                'p': np.array([P(unreachable) values]),
                'err': np.array([SEM values]),
                'mean_overlap': np.array([...]),  # spectral only
                'sem_overlap': np.array([...]),   # spectral only
            },
            ...
        }
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    # Build ρ grid
    rho_grid = np.arange(0, rho_max + rho_step / 2, rho_step)
    if rho_grid[0] == 0:
        rho_grid = rho_grid[1:]  # Skip ρ=0

    logger.info(
        f"Density sweep: dims={dims}, ρ=0..{rho_max} (step {rho_step}), "
        f"taus={taus}, {ensemble}, k_cap={k_cap}"
    )

    # Get ensemble-specific optimization settings
    opt_settings = settings.get_optimization_settings(ensemble)
    opt_restarts = opt_settings['restarts']
    opt_maxiter = opt_settings['maxiter']
    logger.info(
        f"Using ensemble-specific settings for {ensemble}: "
        f"restarts={opt_restarts}, maxiter={opt_maxiter}"
    )

    results = {"dims": dims, "taus": taus, "rho_grid": rho_grid}

    # For each dimension
    for d in dims:
        logger.info(f"Processing dimension d={d}")

        # Build K values from ρ grid for this d
        k_values_raw = [min(round(rho * d**2), k_cap) for rho in rho_grid]
        # Deduplicate and sort
        k_values = sorted(set(k_values_raw))
        if not k_values:
            logger.warning(f"  No valid K values for d={d}")
            continue

        # For each K, run MC once and collect raw data
        # Storage: k_idx -> {criterion: data}
        raw_data_by_k = {}

        for K in k_values:
            logger.info(f"  Computing MC for d={d}, K={K}")

            # Counters for τ-independent criteria (moment only)
            unreach_old = 0
            total_count = 0
            # Collect best values for τ-dependent criteria (spectral, krylov)
            spectral_best_values = []
            krylov_best_values = []

            # Trial index for logging
            trial_idx = 0

            for _ in range(nks):
                # Generate random Hamiltonian ensemble
                hams = models.random_hamiltonian_ensemble(
                    d, K, ensemble, seed=rng.randint(0, 2**31 - 1), **ensemble_params
                )

                # Initial state
                psi = models.fock_state(d, 0)

                # Sample random target states
                targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

                # Pre-compute old criterion data
                from scipy.linalg import null_space

                def expect_array_of_operators(ops, state):
                    if qutip.isoper(ops):
                        return np.real_if_close(qutip.expect(ops, state))
                    else:
                        ret = [expect_array_of_operators(subops, state) for subops in ops]
                        return np.array(ret)

                def check_eigenvalues(matrix):
                    eigenvalues = np.linalg.eigvalsh(matrix)
                    return np.all(eigenvalues > 0) or np.all(eigenvalues < 0)

                # Anticommutators for moment criterion
                hs_anticomms = [[None for _ in range(K)] for _ in range(K)]
                for i in range(K):
                    for j in range(K):
                        hs_anticomms[i][j] = (hams[i] @ hams[j] + hams[j] @ hams[i]) / 2

                energies_zero = expect_array_of_operators(hams, psi) # <- do innego pliku razem z krylovem etc?
                energy_sq_zero = expect_array_of_operators(hs_anticomms, psi)

                for phi in targets:
                    # Continuous Krylov criterion (τ-dependent)
                    m_krylov = min(K, d)
                    if optimize_lambda:
                        # Optimize λ to maximize Krylov score
                        krylov_result = optimize.maximize_krylov_score(
                            psi,
                            phi,
                            hams,
                            m=m_krylov,
                            method=method,
                            restarts=opt_restarts,
                            maxiter=opt_maxiter,
                            seed=rng.randint(0, 2**31 - 1),
                        )
                        krylov_best_values.append(krylov_result["best_value"])
                    else:
                        # Use fixed random λ ~ N(0, 1/√K) without optimization
                        lambda_fixed = rng.randn(K) / np.sqrt(K)
                        krylov_score_fixed = mathematics.krylov_score(
                            lambda_fixed, psi, phi, hams, m=m_krylov
                        )
                        krylov_best_values.append(krylov_score_fixed)

                    # Spectral overlap criterion
                    if optimize_lambda:
                        # Optimize λ to maximize spectral overlap
                        result = optimize.maximize_spectral_overlap(
                            psi,
                            phi,
                            hams,
                            method=method,
                            restarts=opt_restarts,
                            maxiter=opt_maxiter,
                            seed=rng.randint(0, 2**31 - 1),
                        )
                        spectral_best_values.append(result["best_value"])
                    else:
                        # Use fixed random λ ~ N(0, 1/√K) without optimization
                        # CRITICAL FIX: Use the SAME spectral_overlap formula as optimized mode
                        lambda_fixed = rng.randn(K) / np.sqrt(K)
                        spectral_overlap_fixed = mathematics.spectral_overlap(
                            lambda_fixed, psi, phi, hams
                        )
                        spectral_best_values.append(spectral_overlap_fixed)

                    # Moment criterion
                    energies_state = expect_array_of_operators(hams, phi)
                    diff = energies_state - energies_zero
                    kernel = null_space(diff.reshape(1, -1))

                    # Initialize moment data for logging
                    moment_eigenvalues = np.array([])
                    moment_definite = False

                    if kernel.size > 0:
                        energy_sq_state = expect_array_of_operators(hs_anticomms, phi)
                        m_final = kernel.T @ (energy_sq_state - energy_sq_zero) @ kernel

                        # Compute eigenvalues for logging
                        moment_eigenvalues = np.linalg.eigvalsh(m_final)
                        moment_definite = check_eigenvalues(m_final)

                        if moment_definite:
                            unreach_old += 1

                    total_count += 1

                    # Log trial data if logger is provided
                    if data_logger is not None:
                        data_logger.log_trial(
                            d=d,
                            K=K,
                            trial_idx=trial_idx,
                            spectral_score=spectral_best_values[-1],  # Just added
                            krylov_score=krylov_best_values[-1],       # Just added
                            moment_eigenvalues=moment_eigenvalues,
                            moment_definite=moment_definite,
                        )

                    trial_idx += 1

            # Store raw results for this K
            raw_data_by_k[K] = {
                "total_count": total_count,
                "unreach_old": unreach_old,  # Moment criterion (τ-independent)
                "spectral_best_values": np.array(spectral_best_values),  # Spectral (τ-dependent)
                "krylov_best_values": np.array(krylov_best_values),  # Krylov (τ-dependent, NEW)
            }

            logger.info(f"    Collected {total_count} trials for (d={d}, K={K})")

        # Now process results for each (tau, criterion) combination
        for tau in taus:
            for criterion in ["spectral", "moment", "krylov"]:
                K_list, rho_list, p_list, err_list = [], [], [], []
                mean_overlap_list, sem_overlap_list = [], []

                for K in k_values:
                    data = raw_data_by_k[K]
                    total = data["total_count"]

                    if criterion == "spectral":
                        # Threshold spectral best values at this tau
                        best_vals = data["spectral_best_values"]
                        unreach = np.sum(best_vals < tau)
                        p = unreach / total if total > 0 else 0.0

                        # Compute mean and SEM of best overlap
                        mean_overlap = float(np.mean(best_vals))
                        sem_overlap = float(np.std(best_vals, ddof=1) / np.sqrt(len(best_vals))) if len(best_vals) > 1 else 0.0
                        mean_overlap_list.append(mean_overlap)
                        sem_overlap_list.append(sem_overlap)

                    elif criterion == "krylov":
                        # Threshold Krylov best values at this tau (NEW: τ-dependent)
                        best_vals = data["krylov_best_values"]
                        unreach = np.sum(best_vals < tau)
                        p = unreach / total if total > 0 else 0.0

                        # Compute mean and SEM of Krylov scores
                        mean_overlap = float(np.mean(best_vals))
                        sem_overlap = float(np.std(best_vals, ddof=1) / np.sqrt(len(best_vals))) if len(best_vals) > 1 else 0.0
                        mean_overlap_list.append(mean_overlap)
                        sem_overlap_list.append(sem_overlap)

                    elif criterion == "moment":
                        unreach = data["unreach_old"]
                        p = unreach / total if total > 0 else 0.0

                    p = max(settings.DISPLAY_FLOOR, p)
                    sem = mathematics.compute_binomial_sem(p, total) if total > 0 else 0.0

                    K_list.append(K)
                    rho_list.append(K / (d**2))
                    p_list.append(p)
                    err_list.append(sem)

                # Store results for this (d, tau, criterion)
                key = (d, tau, criterion)
                results[key] = {
                    "K": np.array(K_list),
                    "rho": np.array(rho_list),
                    "p": np.array(p_list),
                    "err": np.array(err_list),
                }

                # Store mean/SEM for τ-dependent criteria (spectral, krylov)
                if criterion in ["spectral", "krylov"]:
                    results[key]["mean_overlap"] = np.array(mean_overlap_list)
                    results[key]["sem_overlap"] = np.array(sem_overlap_list)

                logger.info(
                    f"    Stored results for (d={d}, τ={tau:.2f}, {criterion})"
                )

    logger.info("Density sweep complete")
    return results


def monte_carlo_unreachability_vs_K_multi_tau(
    d: int,
    k_max: int,
    taus: List[float],
    ensemble: str,
    nks: int = settings.FULL_SAMPLING[0],
    nst: int = settings.FULL_SAMPLING[1],
    method: str = settings.DEFAULT_METHOD,
    maxiter: int = settings.DEFAULT_MAXITER,
    seed: Optional[int] = None,
    rank_tol: float = settings.KRYLOV_RANK_TOL,
    **ensemble_params,
) -> Dict[str, Any]:
    """
    Compute P(unreachability) vs K for all 3 criteria, with multiple τ for spectral.

    Similar to monte_carlo_unreachability_vs_K_three but handles multiple τ values
    efficiently by collecting spectral overlaps once and thresholding multiple times.

    Args:
        d: Hilbert space dimension (fixed)
        k_max: Maximum K value (sweep from 2 to k_max)
        taus: List of τ thresholds for spectral criterion
        ensemble: "GOE", "GUE", or "GEO2"
        nks: Number of Hamiltonian samples
        nst: Number of target states per Hamiltonian
        method: Optimization method for spectral overlap
        maxiter: Max iterations for spectral optimization
        seed: Random seed
        rank_tol: Rank tolerance for Krylov criterion
        **ensemble_params: Ensemble-specific parameters (for GEO2: nx, ny, periodic)

    Returns:
        {
            'k': np.array([K values]),
            'taus': taus,
            'd': d,
            # For each tau: spectral results
            (tau, 'spectral'): {'p': ..., 'err': ..., 'mean_overlap': ..., 'sem_overlap': ...},
            # Old and Krylov (tau-independent)
            'old': {'p': ..., 'err': ...},
            'krylov': {'p': ..., 'err': ...},
        }
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    k_values = list(range(2, k_max + 1))

    logger.info(
        f"K-sweep multi-tau: d={d}, K=2..{k_max}, taus={taus}, {ensemble}"
    )

    results = {"k": np.array(k_values), "taus": taus, "d": d}

    # Collect raw data for each K
    raw_data_by_k = {}

    for K in k_values:
        logger.info(f"  Computing MC for K={K}")

        unreach_krylov, unreach_old = 0, 0
        total_count = 0
        spectral_best_values = []

        for _ in range(nks):
            hams = models.random_hamiltonian_ensemble(
                d, K, ensemble, seed=rng.randint(0, 2**31 - 1), **ensemble_params
            )
            psi = models.fock_state(d, 0)
            targets = models.random_states(nst, d, seed=rng.randint(0, 2**31 - 1))

            # Pre-compute old criterion data
            from scipy.linalg import null_space

            def expect_array_of_operators(ops, state):
                if qutip.isoper(ops):
                    return np.real_if_close(qutip.expect(ops, state))
                else:
                    ret = [expect_array_of_operators(subops, state) for subops in ops]
                    return np.array(ret)

            def check_eigenvalues(matrix):
                eigenvalues = np.linalg.eigvalsh(matrix)
                return np.all(eigenvalues > 0) or np.all(eigenvalues < 0)

            hs_anticomms = [[None for _ in range(K)] for _ in range(K)]
            for i in range(K):
                for j in range(K):
                    hs_anticomms[i][j] = (hams[i] @ hams[j] + hams[j] @ hams[i]) / 2

            energies_zero = expect_array_of_operators(hams, psi)
            energy_sq_zero = expect_array_of_operators(hs_anticomms, psi)

            for phi in targets:
                # Krylov (m = min(K, d))
                m_krylov = min(K, d)
                lambdas = rng.uniform(-1.0, 1.0, K)
                H_combined = sum(lam * H for lam, H in zip(lambdas, hams))
                H_matrix = H_combined.full() if hasattr(H_combined, "full") else H_combined

                if mathematics.is_unreachable_krylov(H_matrix, psi, phi, m_krylov, rank_tol=rank_tol):
                    unreach_krylov += 1

                # Spectral
                result = optimize.maximize_spectral_overlap(
                    psi, phi, hams, method=method,
                    restarts=settings.DEFAULT_RESTARTS,
                    maxiter=maxiter,
                    seed=rng.randint(0, 2**31 - 1),
                )
                spectral_best_values.append(result["best_value"])

                # Old
                energies_state = expect_array_of_operators(hams, phi)
                diff = energies_state - energies_zero
                kernel = null_space(diff.reshape(1, -1))

                if kernel.size > 0:
                    energy_sq_state = expect_array_of_operators(hs_anticomms, phi)
                    m_final = kernel.T @ (energy_sq_state - energy_sq_zero) @ kernel
                    if check_eigenvalues(m_final):
                        unreach_old += 1

                total_count += 1

        raw_data_by_k[K] = {
            "total_count": total_count,
            "unreach_krylov": unreach_krylov,
            "unreach_old": unreach_old,
            "spectral_best_values": np.array(spectral_best_values),
        }

    # Process spectral for each tau
    for tau in taus:
        p_list, err_list, mean_list, sem_list = [], [], [], []

        for K in k_values:
            data = raw_data_by_k[K]
            total = data["total_count"]
            best_vals = data["spectral_best_values"]

            unreach = np.sum(best_vals < tau)
            p = unreach / total if total > 0 else 0.0
            p = max(settings.DISPLAY_FLOOR, p)
            sem = mathematics.compute_binomial_sem(p, total) if total > 0 else 0.0

            mean_overlap = float(np.mean(best_vals))
            sem_overlap = float(np.std(best_vals, ddof=1) / np.sqrt(len(best_vals))) if len(best_vals) > 1 else 0.0

            p_list.append(p)
            err_list.append(sem)
            mean_list.append(mean_overlap)
            sem_list.append(sem_overlap)

        results[(tau, "spectral")] = {
            "p": np.array(p_list),
            "err": np.array(err_list),
            "mean_overlap": np.array(mean_list),
            "sem_overlap": np.array(sem_list),
        }

    # Process old and krylov (tau-independent)
    for criterion in ["old", "krylov"]:
        p_list, err_list = [], []

        for K in k_values:
            data = raw_data_by_k[K]
            total = data["total_count"]
            unreach = data[f"unreach_{criterion}"]

            p = unreach / total if total > 0 else 0.0
            p = max(settings.DISPLAY_FLOOR, p)
            sem = mathematics.compute_binomial_sem(p, total) if total > 0 else 0.0

            p_list.append(p)
            err_list.append(sem)

        results[criterion] = {
            "p": np.array(p_list),
            "err": np.array(err_list),
        }

    logger.info("K-sweep multi-tau complete")
    return results
