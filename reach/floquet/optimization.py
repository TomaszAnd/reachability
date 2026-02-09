"""
Fidelity optimization for quantum state preparation (TIME-DEPENDENT).

This module provides ground-truth reachability tests by directly optimizing
the time-evolution fidelity |⟨φ|U(t)|ψ⟩|² for a given Hamiltonian.

This is the GOLD STANDARD for testing whether a state is reachable:
- If max fidelity ≈ 1 → state is reachable
- If max fidelity < 1 → either need more time or state is unreachable

Use this to validate reachability criteria and compare static vs Floquet Hamiltonians.

Module Relationship:
    - optimization.py (THIS FILE): Time-dependent fidelity via U(t) = exp(-iHt)
    - optimize.py: Time-FREE spectral overlap S(λ) and Krylov score R(λ)

This module is primarily used by Floquet experiments to compare time-evolved
fidelity against the time-free spectral criteria.
"""

from typing import Tuple

import numpy as np
import scipy.linalg
from scipy.optimize import minimize, differential_evolution


def compute_fidelity(
    psi: np.ndarray,
    phi: np.ndarray,
    H: np.ndarray,
    t: float
) -> float:
    """
    Compute fidelity |⟨φ|U(t)|ψ⟩|² for time evolution under H.

    Args:
        psi: Initial state vector
        phi: Target state vector
        H: Hamiltonian (as numpy array)
        t: Evolution time

    Returns:
        Fidelity in [0, 1]
    """
    # Time evolution operator U(t) = exp(-iHt)
    U = scipy.linalg.expm(-1j * H * t)

    # Evolve initial state
    psi_evolved = U @ psi

    # Compute fidelity
    overlap = phi.conj() @ psi_evolved
    fidelity = np.abs(overlap)**2

    return fidelity


def optimize_fidelity(
    psi: np.ndarray,
    phi: np.ndarray,
    H: np.ndarray,
    t_max: float = 10.0,
    method: str = 'L-BFGS-B',
    n_restarts: int = 5
) -> Tuple[float, float]:
    """
    Find maximum fidelity over time t ∈ [0, t_max].

    Args:
        psi: Initial state
        phi: Target state
        H: Hamiltonian
        t_max: Maximum evolution time
        method: Optimization method ('L-BFGS-B' or 'differential_evolution')
        n_restarts: Number of random restarts

    Returns:
        Tuple of (max_fidelity, optimal_time)
    """
    def neg_fidelity(t_array):
        """Negative fidelity for minimization."""
        t = t_array[0]
        return -compute_fidelity(psi, phi, H, t)

    if method == 'differential_evolution':
        # Global optimization (slower but more reliable)
        result = differential_evolution(
            neg_fidelity,
            bounds=[(0, t_max)],
            maxiter=100,
            seed=42
        )
        return -result.fun, result.x[0]

    else:
        # Local optimization with multiple restarts
        best_fidelity = 0.0
        best_time = 0.0

        # Try multiple starting points
        for i in range(n_restarts):
            t0 = i * t_max / (n_restarts - 1) if i < n_restarts - 1 else t_max / 2

            result = minimize(
                neg_fidelity,
                x0=[t0],
                bounds=[(0, t_max)],
                method='L-BFGS-B',
                options={'maxiter': 100}
            )

            if -result.fun > best_fidelity:
                best_fidelity = -result.fun
                best_time = result.x[0]

        return best_fidelity, best_time


def optimize_fidelity_parameterized(
    psi: np.ndarray,
    phi: np.ndarray,
    hamiltonians: list,
    t_max: float = 10.0,
    n_trials: int = 10,
    seed: int = None
) -> Tuple[float, np.ndarray, float]:
    """
    Optimize over both time t and Hamiltonian parameters λ.

    Finds: max_{λ, t} |⟨φ|exp(-i(Σ λ_k H_k)t)|ψ⟩|²

    Args:
        psi: Initial state
        phi: Target state
        hamiltonians: List of basis Hamiltonians
        t_max: Maximum time
        n_trials: Number of random λ initializations
        seed: Random seed

    Returns:
        Tuple of (max_fidelity, optimal_lambdas, optimal_time)
    """
    K = len(hamiltonians)
    rng = np.random.RandomState(seed)

    best_fidelity = 0.0
    best_lambdas = None
    best_time = 0.0

    def neg_fidelity(params):
        """Negative fidelity over (λ, t)."""
        lambdas = params[:-1]
        t = params[-1]

        # Build H(λ)
        H = sum(lam * H_k for lam, H_k in zip(lambdas, hamiltonians))

        return -compute_fidelity(psi, phi, H, t)

    # Try multiple random initializations
    for trial in range(n_trials):
        # Initialize λ ~ N(0, 1/√K), t ~ U(0, t_max)
        lambdas_0 = rng.randn(K) / np.sqrt(K)
        t_0 = rng.uniform(0, t_max)
        x0 = np.concatenate([lambdas_0, [t_0]])

        # Bounds: λ ∈ [-2, 2], t ∈ [0, t_max]
        bounds = [(-2, 2)] * K + [(0, t_max)]

        result = minimize(
            neg_fidelity,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 200}
        )

        if -result.fun > best_fidelity:
            best_fidelity = -result.fun
            best_lambdas = result.x[:-1]
            best_time = result.x[-1]

    return best_fidelity, best_lambdas, best_time


def compare_static_vs_floquet(
    psi: np.ndarray,
    phi: np.ndarray,
    static_hams: list,
    lambdas: np.ndarray,
    driving_functions: list,
    T: float,
    t_max: float = 10.0,
    floquet_order: int = 2
) -> dict:
    """
    Direct comparison of static vs Floquet effective Hamiltonians.

    This is the KEY EXPERIMENT for demonstrating Floquet's advantage!

    Args:
        psi: Initial state
        phi: Target state
        static_hams: List of static basis Hamiltonians
        lambdas: Coupling coefficients
        driving_functions: Time-periodic driving functions
        T: Period
        t_max: Maximum evolution time
        floquet_order: Magnus expansion order

    Returns:
        Dictionary with comparison metrics
    """
    from reach import floquet

    # Convert to numpy if needed
    if not isinstance(static_hams[0], np.ndarray):
        static_hams = floquet.hamiltonians_to_numpy(static_hams)

    # Static Hamiltonian: H = Σ λ_k H_k
    H_static = sum(lam * H_k for lam, H_k in zip(lambdas, static_hams))

    # Floquet effective Hamiltonian: H_F = H_F^(1) + H_F^(2)
    H_floquet = floquet.compute_floquet_hamiltonian(
        static_hams, lambdas, driving_functions, T,
        order=floquet_order, n_fourier_terms=10
    )

    # Optimize fidelity for each
    fid_static, t_static = optimize_fidelity(psi, phi, H_static, t_max)
    fid_floquet, t_floquet = optimize_fidelity(psi, phi, H_floquet, t_max)

    # Compute improvement
    improvement = (fid_floquet - fid_static) / max(fid_static, 1e-10)

    # Norms for diagnostics
    norm_static = np.linalg.norm(H_static)
    norm_floquet = np.linalg.norm(H_floquet)

    # Check if commutators are significant
    H_F1 = floquet.compute_floquet_hamiltonian_order1(static_hams, lambdas, driving_functions, T)
    H_F2 = floquet.compute_floquet_hamiltonian_order2(static_hams, lambdas, driving_functions, T)

    return {
        'fidelity_static': fid_static,
        'fidelity_floquet': fid_floquet,
        'time_static': t_static,
        'time_floquet': t_floquet,
        'improvement': improvement,
        'improvement_percent': improvement * 100,
        'norm_static': norm_static,
        'norm_floquet': norm_floquet,
        'norm_H_F1': np.linalg.norm(H_F1),
        'norm_H_F2': np.linalg.norm(H_F2),
        'ratio_H_F2_H_F1': np.linalg.norm(H_F2) / max(np.linalg.norm(H_F1), 1e-10),
    }


def scan_operator_number(
    psi: np.ndarray,
    phi: np.ndarray,
    generate_hamiltonians_func,
    K_values: list,
    driving_type: str = 'offset_sinusoidal',
    T: float = 1.0,
    t_max: float = 10.0,
    seed: int = 42
) -> dict:
    """
    Scan over number of operators K to find critical threshold.

    This tests the hypothesis: Floquet requires fewer operators (smaller K_c)
    to reach a given fidelity than static Hamiltonians.

    Args:
        psi: Initial state
        phi: Target state
        generate_hamiltonians_func: Function(K, seed) -> list of Hamiltonians
        K_values: List of K values to test
        driving_type: Type of driving function
        T: Period
        t_max: Maximum time
        seed: Random seed

    Returns:
        Dictionary with results for each K
    """
    from reach import floquet

    results = {
        'K_values': K_values,
        'fidelity_static': [],
        'fidelity_floquet_o1': [],
        'fidelity_floquet_o2': [],
        'time_static': [],
        'time_floquet_o1': [],
        'time_floquet_o2': [],
    }

    for K in K_values:
        # Generate Hamiltonians
        hams = generate_hamiltonians_func(K, seed)
        if not isinstance(hams[0], np.ndarray):
            hams = floquet.hamiltonians_to_numpy(hams)

        # Random lambdas
        rng = np.random.RandomState(seed)
        lambdas = rng.randn(K) / np.sqrt(K)

        # Create driving
        driving = floquet.create_driving_functions(K, driving_type, T, seed)

        # Static
        H_static = sum(lam * H_k for lam, H_k in zip(lambdas, hams))
        fid_s, t_s = optimize_fidelity(psi, phi, H_static, t_max)

        # Floquet order 1
        H_F1 = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, T, order=1)
        fid_f1, t_f1 = optimize_fidelity(psi, phi, H_F1, t_max)

        # Floquet order 2
        H_F2 = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, T, order=2)
        fid_f2, t_f2 = optimize_fidelity(psi, phi, H_F2, t_max)

        results['fidelity_static'].append(fid_s)
        results['fidelity_floquet_o1'].append(fid_f1)
        results['fidelity_floquet_o2'].append(fid_f2)
        results['time_static'].append(t_s)
        results['time_floquet_o1'].append(t_f1)
        results['time_floquet_o2'].append(t_f2)

    return results
