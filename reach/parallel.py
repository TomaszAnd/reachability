#!/usr/bin/env python3
"""
Parallel Monte Carlo evaluation for reachability criteria.

Uses joblib for embarrassingly parallel trial execution.
"""
import numpy as np
from typing import List, Dict, Optional
from joblib import Parallel, delayed
import qutip

from reach import models, mathematics, optimize


def evaluate_single_trial(
    ensemble: str,
    d: int,
    K: int,
    tau: float,
    seed: int,
    criteria: List[str] = ['spectral', 'krylov', 'moment'],
    optimizer_settings: Optional[Dict] = None,
    lattice_params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate all criteria for a single (psi, phi, hams) triple.

    Args:
        ensemble: 'canonical', 'GUE', or 'GEO2'
        d: Hilbert space dimension
        K: Number of Hamiltonians
        tau: Reachability threshold
        seed: Random seed for reproducibility
        criteria: List of criteria to evaluate
        optimizer_settings: Dict with maxiter, restarts, ftol
        lattice_params: Dict with nx, ny for GEO2

    Returns:
        Dict with results for each criterion
    """
    if optimizer_settings is None:
        optimizer_settings = {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6}

    # Generate states and Hamiltonians
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed + 1000)[0]

    if ensemble == 'GEO2' and lattice_params:
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **lattice_params)
    else:
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

    result = {'seed': seed}

    # Spectral criterion
    if 'spectral' in criteria:
        res = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            maxiter=optimizer_settings['maxiter'],
            restarts=optimizer_settings['restarts'],
            ftol=optimizer_settings['ftol'],
            seed=seed
        )
        result['spectral_score'] = res['best_value']
        result['spectral_unreachable'] = res['best_value'] < tau

    # Krylov criterion
    if 'krylov' in criteria:
        res = optimize.maximize_krylov_score(
            psi, phi, hams,
            m=min(K, d),
            maxiter=optimizer_settings['maxiter'],
            restarts=optimizer_settings['restarts'],
            ftol=optimizer_settings['ftol'],
            seed=seed
        )
        result['krylov_score'] = res['best_value']
        result['krylov_unreachable'] = res['best_value'] < tau

    # Moment criterion (vectorized with numpy for speed)
    if 'moment' in criteria:
        from scipy.linalg import null_space

        # Pre-extract numpy arrays (avoids repeated QObj→numpy conversion)
        hams_np = [H.full() for H in hams]
        psi_np = psi.full().flatten()
        phi_np = phi.full().flatten()

        # First-order condition: expectation differences
        L = np.array([
            np.real(phi_np.conj() @ H @ phi_np - psi_np.conj() @ H @ psi_np)
            for H in hams_np
        ])
        kernel = null_space(L.reshape(1, -1))

        if kernel.size == 0:
            result['moment_unreachable'] = False
        else:
            # Second-order condition: anticommutator matrix (vectorized)
            Q = np.zeros((K, K))
            for i in range(K):
                for j in range(i, K):
                    # {H_i, H_j} / 2 = (H_i H_j + H_j H_i) / 2
                    anticomm = (hams_np[i] @ hams_np[j] + hams_np[j] @ hams_np[i]) / 2
                    q_val = np.real(
                        phi_np.conj() @ anticomm @ phi_np
                        - psi_np.conj() @ anticomm @ psi_np
                    )
                    Q[i, j] = q_val
                    Q[j, i] = q_val  # Symmetric

            Q_proj = kernel.T @ Q @ kernel
            eigvals = np.linalg.eigvalsh(Q_proj)
            result['moment_unreachable'] = bool(
                np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10)
            )

    return result


def parallel_evaluate(
    ensemble: str,
    d: int,
    K: int,
    tau: float,
    n_trials: int,
    seed_base: int = 0,
    criteria: List[str] = ['spectral', 'krylov', 'moment'],
    optimizer_settings: Optional[Dict] = None,
    lattice_params: Optional[Dict] = None,
    n_jobs: int = -1,
    verbose: int = 0,
) -> Dict:
    """
    Parallel evaluation of reachability criteria over multiple trials.

    Args:
        ensemble: 'canonical', 'GUE', or 'GEO2'
        d: Hilbert space dimension
        K: Number of Hamiltonians
        tau: Reachability threshold
        n_trials: Number of Monte Carlo trials
        seed_base: Base seed for reproducibility
        criteria: List of criteria to evaluate
        optimizer_settings: Dict with maxiter, restarts, ftol
        lattice_params: Dict with nx, ny for GEO2
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Verbosity level (0, 1, or 10)

    Returns:
        Dict with P(unreachable) for each criterion and statistics
    """
    seeds = [seed_base + i for i in range(n_trials)]

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(evaluate_single_trial)(
            ensemble, d, K, tau, seed,
            criteria, optimizer_settings, lattice_params
        )
        for seed in seeds
    )

    # Aggregate results
    output = {
        'ensemble': ensemble,
        'd': d,
        'K': K,
        'rho': K / d**2,
        'tau': tau,
        'n_trials': n_trials,
    }

    for criterion in criteria:
        key = f'{criterion}_unreachable'
        unreachable_count = sum(1 for r in results if r.get(key, False))
        output[f'{criterion}_P'] = unreachable_count / n_trials

        if f'{criterion}_score' in results[0]:
            scores = [r[f'{criterion}_score'] for r in results]
            output[f'{criterion}_mean'] = np.mean(scores)
            output[f'{criterion}_std'] = np.std(scores)

    return output
