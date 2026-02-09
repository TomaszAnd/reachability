"""
Moment-based reachability criteria.

Implements static and Floquet moment criteria for testing unreachability.
These are SUFFICIENT conditions: if criterion succeeds, state is unreachable.

The key question is discriminative power: how often can the criterion
prove unreachability for random state pairs at a given operator count K?

Expected scaling: P(unreachable | K) ~ exp(-α K)
Hypothesis: α_Floquet > α_static (Floquet criterion is stronger)
"""

import numpy as np
from typing import Tuple, Optional, List


def static_moment_criterion(
    psi: np.ndarray,
    phi: np.ndarray,
    hamiltonians: List[np.ndarray],
    x_range: Tuple[float, float] = (-10.0, 10.0),
    n_x_points: int = 1000,
    tol: float = 1e-10
) -> Tuple[bool, Optional[float], np.ndarray]:
    """
    Static moment criterion for unreachability.

    Tests if there exists x such that Q + x L L^T is positive definite,
    where:
        L[k] = ⟨H_k⟩_φ - ⟨H_k⟩_ψ
        Q[k,m] = ⟨{H_k, H_m}/2⟩_φ - ⟨{H_k, H_m}/2⟩_ψ

    This is λ-INDEPENDENT - uses operators directly.

    Args:
        psi: Initial state vector
        phi: Target state vector
        hamiltonians: List of K Hamiltonian operators (as numpy arrays)
        x_range: Range of x values to search
        n_x_points: Number of x values to test
        tol: Tolerance for positive definiteness

    Returns:
        Tuple of:
        - unreachable: True if criterion proves unreachability
        - x_opt: Value of x where criterion succeeds (None if fails)
        - eigenvalues: Eigenvalues of Q + x_opt L L^T (for diagnostics)
    """
    K = len(hamiltonians)

    # Compute L vector: ⟨H_k⟩_φ - ⟨H_k⟩_ψ
    L = np.zeros(K)
    for k in range(K):
        H_k = hamiltonians[k]
        exp_val_phi = np.real(phi.conj() @ H_k @ phi)
        exp_val_psi = np.real(psi.conj() @ H_k @ psi)
        L[k] = exp_val_phi - exp_val_psi

    # Compute Q matrix: ⟨{H_k, H_m}/2⟩_φ - ⟨{H_k, H_m}/2⟩_ψ
    Q = np.zeros((K, K))
    for k in range(K):
        for m in range(K):
            # Anticommutator: {A, B} = AB + BA
            anticomm = (hamiltonians[k] @ hamiltonians[m] +
                       hamiltonians[m] @ hamiltonians[k]) / 2

            exp_val_phi = np.real(phi.conj() @ anticomm @ phi)
            exp_val_psi = np.real(psi.conj() @ anticomm @ psi)
            Q[k, m] = exp_val_phi - exp_val_psi

    # Search for x such that Q + x L L^T is positive definite
    L_outer = np.outer(L, L)

    x_values = np.linspace(x_range[0], x_range[1], n_x_points)

    for x in x_values:
        M = Q + x * L_outer

        # Check positive definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(M)

        if np.all(eigvals > tol):
            # SUCCESS: Found x where matrix is positive definite
            # This proves the state is unreachable
            return True, x, eigvals

    # FAIL: No x found that makes matrix positive definite
    # Criterion is inconclusive (state might be reachable or unreachable)
    return False, None, np.array([])


def floquet_moment_criterion(
    psi: np.ndarray,
    phi: np.ndarray,
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List,
    period: float,
    order: int = 2,
    x_range: Tuple[float, float] = (-10.0, 10.0),
    n_x_points: int = 1000,
    tol: float = 1e-10
) -> Tuple[bool, Optional[float], np.ndarray]:
    """
    Floquet moment criterion for unreachability (λ-DEPENDENT).

    Tests if there exists x such that Q_F + x L_F L_F^T is positive definite,
    where:
        L_F[k] = ⟨∂H_F/∂λ_k⟩_φ - ⟨∂H_F/∂λ_k⟩_ψ

    and ∂H_F/∂λ_k depends on λ through:
        ∂H_F/∂λ_k = λ̄_k H_k + Σ_{j≠k} λ_j F_jk [H_j, H_k] / (2i)

    This is λ-DEPENDENT - different λ give different L_F and Q_F!

    Args:
        psi: Initial state vector
        phi: Target state vector
        hamiltonians: List of K Hamiltonian operators
        lambdas: Coupling coefficients (affects ∂H_F/∂λ_k!)
        driving_functions: Time-periodic driving f_k(t)
        period: Driving period T
        order: Magnus expansion order (1 or 2)
        x_range: Range of x values to search
        n_x_points: Number of x values to test
        tol: Tolerance for positive definiteness

    Returns:
        Tuple of:
        - unreachable: True if criterion proves unreachability
        - x_opt: Value of x where criterion succeeds (None if fails)
        - eigenvalues: Eigenvalues of Q_F + x_opt L_F L_F^T
    """
    from reach import floquet

    K = len(hamiltonians)

    # Compute derivatives ∂H_F/∂λ_k
    # These are λ-DEPENDENT!
    dH_F_dlambda = []

    for k in range(K):
        # First-order contribution: λ̄_k H_k
        lambda_bar_k = floquet.compute_time_average(driving_functions[k], period)
        derivative = lambda_bar_k * hamiltonians[k]

        # Second-order contribution: Σ_{j≠k} λ_j F_jk [H_j, H_k] / (2i)
        if order >= 2:
            for j in range(K):
                if j != k:
                    F_jk = floquet.compute_fourier_overlap(
                        driving_functions[j], driving_functions[k], period
                    )
                    # Commutator: [H_j, H_k] = H_j H_k - H_k H_j
                    commutator = hamiltonians[j] @ hamiltonians[k] - hamiltonians[k] @ hamiltonians[j]

                    # Add: λ_j F_jk [H_j, H_k] / (2i)
                    derivative += lambdas[j] * F_jk * commutator / (2 * 1j)

        # Ensure Hermitian (remove numerical errors)
        derivative = (derivative + derivative.conj().T) / 2
        dH_F_dlambda.append(derivative)

    # Compute L_F vector: ⟨∂H_F/∂λ_k⟩_φ - ⟨∂H_F/∂λ_k⟩_ψ
    L_F = np.zeros(K)
    for k in range(K):
        exp_val_phi = np.real(phi.conj() @ dH_F_dlambda[k] @ phi)
        exp_val_psi = np.real(psi.conj() @ dH_F_dlambda[k] @ psi)
        L_F[k] = exp_val_phi - exp_val_psi

    # Compute Q_F matrix: ⟨{∂H_F/∂λ_k, ∂H_F/∂λ_m}/2⟩_φ - ⟨...⟩_ψ
    Q_F = np.zeros((K, K))
    for k in range(K):
        for m in range(K):
            # Anticommutator
            anticomm = (dH_F_dlambda[k] @ dH_F_dlambda[m] +
                       dH_F_dlambda[m] @ dH_F_dlambda[k]) / 2

            exp_val_phi = np.real(phi.conj() @ anticomm @ phi)
            exp_val_psi = np.real(psi.conj() @ anticomm @ psi)
            Q_F[k, m] = exp_val_phi - exp_val_psi

    # Search for x such that Q_F + x L_F L_F^T is positive definite
    L_F_outer = np.outer(L_F, L_F)

    x_values = np.linspace(x_range[0], x_range[1], n_x_points)

    for x in x_values:
        M = Q_F + x * L_F_outer

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(M)

        if np.all(eigvals > tol):
            # SUCCESS: Floquet criterion proves unreachability with these λ
            return True, x, eigvals

    # FAIL: No x found for these λ
    return False, None, np.array([])


def floquet_moment_criterion_optimized(
    psi: np.ndarray,
    phi: np.ndarray,
    hamiltonians: List[np.ndarray],
    driving_functions: List,
    period: float,
    order: int = 2,
    n_lambda_trials: int = 100,
    x_range: Tuple[float, float] = (-10.0, 10.0),
    n_x_points: int = 1000,
    tol: float = 1e-10,
    seed: Optional[int] = None
) -> Tuple[bool, Optional[np.ndarray], Optional[float], np.ndarray]:
    """
    Floquet moment criterion with λ OPTIMIZATION.

    KEY DIFFERENCE from static: Floquet criterion is λ-dependent.
    We search for λ such that the criterion proves unreachability.

    This tests: "Does there EXIST a λ such that Floquet criterion succeeds?"

    Args:
        psi: Initial state vector
        phi: Target state vector
        hamiltonians: List of K Hamiltonian operators
        driving_functions: Time-periodic driving f_k(t)
        period: Driving period T
        order: Magnus expansion order
        n_lambda_trials: Number of random λ to try
        x_range: Range of x values to search
        n_x_points: Number of x values to test
        tol: Tolerance for positive definiteness
        seed: Random seed for λ generation

    Returns:
        Tuple of:
        - unreachable: True if ANY λ proves unreachability
        - lambdas_opt: λ that succeeded (None if all failed)
        - x_opt: Value of x where criterion succeeds
        - eigenvalues: Eigenvalues of Q_F + x_opt L_F L_F^T
    """
    K = len(hamiltonians)
    rng = np.random.RandomState(seed)

    # Try multiple random λ values
    for trial in range(n_lambda_trials):
        # Generate random coupling coefficients
        lambdas = rng.randn(K) / np.sqrt(K)

        # Test criterion with these λ
        unreachable, x_opt, eigvals = floquet_moment_criterion(
            psi, phi, hamiltonians, lambdas, driving_functions, period,
            order=order, x_range=x_range, n_x_points=n_x_points, tol=tol
        )

        if unreachable:
            # SUCCESS: Found λ that proves unreachability!
            return True, lambdas, x_opt, eigvals

    # FAIL: No λ found among all trials
    return False, None, None, np.array([])


def compare_criterion_strength(
    psi: np.ndarray,
    phi: np.ndarray,
    hamiltonians: List[np.ndarray],
    driving_functions: List,
    period: float,
    n_lambda_trials: int = 100,
    seed: Optional[int] = None
) -> dict:
    """
    Compare static vs Floquet criterion for a single state pair.

    Returns dictionary with:
    - 'static_unreachable': bool
    - 'floquet_o1_unreachable': bool
    - 'floquet_o2_unreachable': bool
    - 'static_x': x value for static (if succeeded)
    - 'floquet_o1_x': x value for Floquet O1 (if succeeded)
    - 'floquet_o2_x': x value for Floquet O2 (if succeeded)
    """
    results = {}

    # Static criterion (λ-independent)
    static_unreachable, static_x, _ = static_moment_criterion(psi, phi, hamiltonians)
    results['static_unreachable'] = static_unreachable
    results['static_x'] = static_x

    # Floquet O1 (time-averaged only)
    floquet_o1_unreachable, lambdas_o1, x_o1, _ = floquet_moment_criterion_optimized(
        psi, phi, hamiltonians, driving_functions, period,
        order=1, n_lambda_trials=n_lambda_trials, seed=seed
    )
    results['floquet_o1_unreachable'] = floquet_o1_unreachable
    results['floquet_o1_x'] = x_o1
    results['floquet_o1_lambdas'] = lambdas_o1

    # Floquet O2 (time-averaged + commutators)
    floquet_o2_unreachable, lambdas_o2, x_o2, _ = floquet_moment_criterion_optimized(
        psi, phi, hamiltonians, driving_functions, period,
        order=2, n_lambda_trials=n_lambda_trials, seed=seed
    )
    results['floquet_o2_unreachable'] = floquet_o2_unreachable
    results['floquet_o2_x'] = x_o2
    results['floquet_o2_lambdas'] = lambdas_o2

    return results
