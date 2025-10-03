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

1. **Unreachability Probability** (new criterion):
   P_unreach(d, K; τ) = Pr[ max_{λ ∈ [-1,1]ᴷ} S(λ) < τ ]

   Estimated via Monte Carlo sampling over Hamiltonian ensembles and target states.
   If S* < τ, the target is classified as unreachable.

2. **Old Criterion** (τ-free, moment-based):
   Uses definiteness check on second moment matrix. Included for comparison
   with new spectral overlap criterion. Does NOT use threshold τ.

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

logger = logging.getLogger(__name__)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectral overlap landscape S(λ₁, λ₂) over parameter grid.

    Fixes λ₃, λ₄, ... = 0 and varies (λ₁, λ₂) to compute S(λ₁, λ₂) surface.
    This is the actual spectral overlap values, not unreachability probabilities.

    Args:
        d: Hilbert space dimension
        k: Number of Hamiltonians (k ≥ 2)
        ensemble: "GOE" or "GUE"
        grid: Grid resolution (grid × grid points)
        n_targets: Number of target states to average over
        lambda_range: (min, max) range for λ₁, λ₂
        seed: Random seed

    Returns:
        (L1, L2, S) where:
        - L1, L2: Meshgrid arrays for λ₁, λ₂
        - S: Grid of spectral overlap values S(λ₁, λ₂)
    """
    if seed is None:
        seed = settings.SEED
    rng = models.setup_rng(seed)

    logger.info(f"Computing landscape: d={d}, k={k}, {ensemble}, grid={grid}×{grid}")

    # Generate fixed Hamiltonian ensemble
    hams = models.random_hamiltonian_ensemble(d, k, ensemble, seed=rng.randint(0, 2**31 - 1))

    # Setup parameter grid
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

    # Apply smoothing for cleaner visualization
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


def old_criterion_probabilities(
    dims: List[int],
    k_values: List[int],
    ensemble: str,
    nks: int = 80,
    nst: int = 20,
    seed: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute P(unreachability) using the old moment-based criterion.

    Implements the classical moment-based reachability criterion from the
    reference notebook for comparison with the new spectral overlap method.

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
        f"Computing old criterion probabilities: ensemble={ensemble}, dims={dims}, k_values={k_values}"
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

                        # Check definiteness (old criterion)
                        if check_eigenvalues(m_final):
                            unreachable_count += 1

                    total_count += 1

            # Store probability
            probability = unreachable_count / total_count if total_count > 0 else 0.0
            results[(d, k)] = probability

            log.info(f"    P(unreachable) = {probability:.4f} ({unreachable_count}/{total_count})")

    log.info("Old criterion computation complete")
    return results
