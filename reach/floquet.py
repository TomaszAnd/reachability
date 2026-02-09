"""
Floquet engineering utilities for time-periodic Hamiltonians.

This module implements the Magnus expansion to compute effective Floquet
Hamiltonians for time-periodic systems. The key insight is that higher-order
Magnus terms include commutators that make the moment criterion λ-dependent.

Mathematical Background:
------------------------
For a time-periodic Hamiltonian H(t) = Σ_k λ_k f_k(t) H_k with period T,
the time-evolution operator can be written as:

    U(T) = exp(-i H_F T)

where H_F is the effective Floquet Hamiltonian given by the Magnus expansion:

    H_F = H_F^(1) + H_F^(2) + H_F^(3) + ...

First order (time-averaged):
    H_F^(1) = (1/T) ∫_0^T H(t) dt = Σ_k λ̄_k H_k

where λ̄_k = (1/T) ∫_0^T λ_k f_k(t) dt is the time-averaged coefficient.

Second order (commutator corrections):
    H_F^(2) = (1/2iT) ∫_0^T dt ∫_0^t dt' [H(t), H(t')]

For periodic driving, this can be written as:
    H_F^(2) ≈ Σ_{j<k} λ_j λ_k F_{jk} [H_j, H_k] / (2i)

where F_{jk} is a Fourier overlap coefficient depending on f_j(t) and f_k(t).

Key Insight for Moment Criterion:
----------------------------------
The regular moment criterion uses ⟨H_k⟩ (λ-independent).
The Floquet moment criterion uses ⟨∂H_F/∂λ_k⟩ which includes:

    ∂H_F/∂λ_k = λ̄_k H_k + Σ_{j≠k} λ_j F_{jk} [H_j, H_k] / (2i)

The second term is λ-DEPENDENT, which should make the Floquet moment criterion
more discriminative (similar to Spectral and Krylov criteria).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import qutip
import scipy.integrate as integrate


# Type alias for driving functions
DrivingFunction = Callable[[float], float]


def compute_time_average(f: DrivingFunction, T: float, n_points: int = 1000) -> float:
    """
    Compute time average of a periodic function: (1/T) ∫_0^T f(t) dt

    Args:
        f: Time-periodic function
        T: Period
        n_points: Number of integration points

    Returns:
        Time-averaged value
    """
    result, _ = integrate.quad(f, 0, T, limit=n_points)
    return result / T


def compute_fourier_overlap(
    f1: DrivingFunction,
    f2: DrivingFunction,
    T: float,
    n_terms: int = 10,
    n_points: int = 1000
) -> float:
    """
    Compute Fourier overlap coefficient for two driving functions.

    This coefficient F_{jk} appears in the second-order Magnus expansion:
        H_F^(2) ∝ Σ_{j,k} λ_j λ_k F_{jk} [H_j, H_k]

    The overlap is computed as:
        F_{jk} = Σ_n (a_{j,n} b_{k,n} - b_{j,n} a_{k,n}) / (n ω)

    where a_{j,n}, b_{j,n} are Fourier coefficients and ω = 2π/T.

    Args:
        f1: First driving function
        f2: Second driving function
        T: Period
        n_terms: Number of Fourier terms to include
        n_points: Integration points per term

    Returns:
        Fourier overlap coefficient
    """
    omega = 2 * np.pi / T
    overlap = 0.0

    for n in range(1, n_terms + 1):
        # Fourier coefficients for f1
        def f1_cos(t): return f1(t) * np.cos(n * omega * t)
        def f1_sin(t): return f1(t) * np.sin(n * omega * t)

        a1n, _ = integrate.quad(f1_cos, 0, T, limit=n_points)
        b1n, _ = integrate.quad(f1_sin, 0, T, limit=n_points)
        a1n *= 2 / T
        b1n *= 2 / T

        # Fourier coefficients for f2
        def f2_cos(t): return f2(t) * np.cos(n * omega * t)
        def f2_sin(t): return f2(t) * np.sin(n * omega * t)

        a2n, _ = integrate.quad(f2_cos, 0, T, limit=n_points)
        b2n, _ = integrate.quad(f2_sin, 0, T, limit=n_points)
        a2n *= 2 / T
        b2n *= 2 / T

        # Contribution to overlap
        overlap += (a1n * b2n - b1n * a2n) / (n * omega)

    return overlap


def compute_floquet_hamiltonian_order1(
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List[DrivingFunction],
    T: float
) -> np.ndarray:
    """
    Compute first-order effective Floquet Hamiltonian (time-averaged).

    H_F^(1) = Σ_k λ_k λ̄_k H_k

    where λ̄_k = (1/T) ∫_0^T f_k(t) dt is the time-averaged driving amplitude.

    Args:
        hamiltonians: List of K basis Hamiltonians {H_k}
        lambdas: Coupling coefficients λ_k (array of length K)
        driving_functions: List of K time-periodic functions f_k(t)
        T: Period

    Returns:
        First-order effective Hamiltonian H_F^(1)
    """
    K = len(hamiltonians)
    d = hamiltonians[0].shape[0]
    H_F1 = np.zeros((d, d), dtype=complex)

    for k in range(K):
        # Compute time-averaged coefficient
        lambda_bar_k = compute_time_average(driving_functions[k], T)
        H_F1 += lambdas[k] * lambda_bar_k * hamiltonians[k]

    return H_F1


def compute_floquet_hamiltonian_order2(
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List[DrivingFunction],
    T: float,
    n_fourier_terms: int = 10
) -> np.ndarray:
    """
    Compute second-order effective Floquet Hamiltonian (commutator corrections).

    H_F^(2) = Σ_{j,k} λ_j λ_k F_{jk} [H_j, H_k] / (2i)

    where F_{jk} is the Fourier overlap coefficient between f_j(t) and f_k(t).

    Args:
        hamiltonians: List of K basis Hamiltonians {H_k}
        lambdas: Coupling coefficients λ_k (array of length K)
        driving_functions: List of K time-periodic functions f_k(t)
        T: Period
        n_fourier_terms: Number of Fourier terms for overlap computation

    Returns:
        Second-order effective Hamiltonian H_F^(2)
    """
    K = len(hamiltonians)
    d = hamiltonians[0].shape[0]
    H_F2 = np.zeros((d, d), dtype=complex)

    # Compute all pairwise commutators and overlaps
    for j in range(K):
        for k in range(j + 1, K):  # Only j < k to avoid double counting
            # Compute Fourier overlap
            F_jk = compute_fourier_overlap(
                driving_functions[j],
                driving_functions[k],
                T,
                n_terms=n_fourier_terms
            )

            # Commutator [H_j, H_k]
            commutator = hamiltonians[j] @ hamiltonians[k] - hamiltonians[k] @ hamiltonians[j]

            # Add contribution (factor of 2 because j<k symmetrizes the sum)
            # Note: / (2 * 1j) = * (-0.5j) converts anti-Hermitian to Hermitian
            H_F2 += lambdas[j] * lambdas[k] * F_jk * commutator / (2 * 1j)

    return H_F2


def compute_floquet_hamiltonian(
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List[DrivingFunction],
    T: float,
    order: int = 2,
    n_fourier_terms: int = 10
) -> np.ndarray:
    """
    Compute effective Floquet Hamiltonian via Magnus expansion.

    H_F = H_F^(1) + H_F^(2) + ... (up to specified order)

    Args:
        hamiltonians: List of K basis Hamiltonians {H_k}
        lambdas: Coupling coefficients λ_k (array of length K)
        driving_functions: List of K time-periodic functions f_k(t)
        T: Period
        order: Magnus expansion order (1 or 2)
        n_fourier_terms: Number of Fourier terms for order-2 computation

    Returns:
        Effective Floquet Hamiltonian H_F
    """
    if len(hamiltonians) != len(driving_functions):
        raise ValueError("Number of Hamiltonians must match number of driving functions")
    if len(lambdas) != len(hamiltonians):
        raise ValueError("Number of lambdas must match number of Hamiltonians")

    # First order
    H_F = compute_floquet_hamiltonian_order1(hamiltonians, lambdas, driving_functions, T)

    # Second order (if requested)
    if order >= 2:
        H_F += compute_floquet_hamiltonian_order2(
            hamiltonians, lambdas, driving_functions, T, n_fourier_terms
        )

    return H_F


def compute_floquet_hamiltonian_derivative(
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List[DrivingFunction],
    T: float,
    order: int = 2,
    n_fourier_terms: int = 10
) -> List[np.ndarray]:
    """
    Compute derivatives ∂H_F/∂λ_k for all k.

    For order 1:
        ∂H_F^(1)/∂λ_k = λ̄_k H_k

    For order 2:
        ∂H_F/∂λ_k = λ̄_k H_k + Σ_{j≠k} λ_j F_{jk} [H_j, H_k] / (2i)

    The second term is λ-DEPENDENT, making the Floquet moment criterion
    discriminative (unlike regular moment which is λ-independent).

    Args:
        hamiltonians: List of K basis Hamiltonians {H_k}
        lambdas: Coupling coefficients λ_k (array of length K)
        driving_functions: List of K time-periodic functions f_k(t)
        T: Period
        order: Magnus expansion order (1 or 2)
        n_fourier_terms: Number of Fourier terms for overlap computation

    Returns:
        List of derivatives [∂H_F/∂λ_1, ∂H_F/∂λ_2, ..., ∂H_F/∂λ_K]
    """
    K = len(hamiltonians)
    d = hamiltonians[0].shape[0]
    derivatives = []

    for k in range(K):
        # First order contribution
        lambda_bar_k = compute_time_average(driving_functions[k], T)
        dH_k = lambda_bar_k * hamiltonians[k]

        # Second order contribution (if requested)
        if order >= 2:
            for j in range(K):
                if j != k:
                    # Compute Fourier overlap F_{jk}
                    F_jk = compute_fourier_overlap(
                        driving_functions[j],
                        driving_functions[k],
                        T,
                        n_terms=n_fourier_terms
                    )

                    # Commutator [H_j, H_k]
                    commutator = (hamiltonians[j] @ hamiltonians[k] -
                                hamiltonians[k] @ hamiltonians[j])

                    # Add λ-DEPENDENT term (this makes Floquet moment discriminative!)
                    dH_k += lambdas[j] * F_jk * commutator / (2 * 1j)

        derivatives.append(dH_k)

    return derivatives


# ============================================================================
# Standard Driving Functions
# ============================================================================

def sinusoidal_drive(omega: float, phi: float = 0.0) -> DrivingFunction:
    """
    Create sinusoidal driving function: f(t) = cos(ωt + φ)

    Args:
        omega: Angular frequency
        phi: Phase offset (default 0)

    Returns:
        Driving function f(t)
    """
    return lambda t: np.cos(omega * t + phi)


def square_wave_drive(omega: float) -> DrivingFunction:
    """
    Create square wave driving function: f(t) = sign(cos(ωt))

    Args:
        omega: Angular frequency

    Returns:
        Driving function f(t)
    """
    return lambda t: np.sign(np.cos(omega * t))


def multi_frequency_drive(omega_0: float, N: int = 3) -> DrivingFunction:
    """
    Create multi-frequency driving (GKP-like): f(t) = 2 + 4 Σ_{n=1}^N cos(4n ω_0 t)

    Args:
        omega_0: Base frequency
        N: Number of harmonics (default 3)

    Returns:
        Driving function f(t)
    """
    return lambda t: 2 + 4 * sum(np.cos(4*n*omega_0*t) for n in range(1, N+1))


def constant_drive() -> DrivingFunction:
    """
    Create constant driving function: f(t) = 1

    Returns:
        Driving function f(t) = 1
    """
    return lambda t: 1.0


def offset_sinusoidal_drive(
    omega: float,
    phi: float = 0.0,
    offset: float = 1.0,
    amplitude: float = 0.5
) -> DrivingFunction:
    """
    Create offset sinusoidal driving: f(t) = offset + amplitude * cos(ωt + φ)

    This has NON-ZERO time average ⟨f⟩ = offset, which means H_F^(1) ≠ 0!
    This is crucial for Floquet enhancement.

    Args:
        omega: Angular frequency
        phi: Phase offset (default 0)
        offset: DC offset (default 1.0)
        amplitude: Oscillation amplitude (default 0.5)

    Returns:
        Driving function f(t) with ⟨f⟩ = offset
    """
    return lambda t: offset + amplitude * np.cos(omega * t + phi)


def bichromatic_drive(
    omega1: float,
    omega2: float,
    phi1: float = 0.0,
    phi2: float = 0.0,
    offset: float = 1.0
) -> DrivingFunction:
    """
    Create bichromatic driving: f(t) = offset + cos(ω₁t + φ₁) + cos(ω₂t + φ₂)

    Two incommensurate frequencies create richer Fourier structure,
    enhancing second-order Magnus terms.

    Args:
        omega1: First frequency
        omega2: Second frequency (should be incommensurate with omega1)
        phi1: Phase for first component
        phi2: Phase for second component
        offset: DC offset (default 1.0)

    Returns:
        Driving function with two-frequency modulation
    """
    return lambda t: offset + np.cos(omega1 * t + phi1) + np.cos(omega2 * t + phi2)


def create_driving_functions(
    K: int,
    drive_type: str = 'offset_sinusoidal',
    T: float = 1.0,
    seed: int = None
) -> List[DrivingFunction]:
    """
    Create K driving functions of specified type.

    Args:
        K: Number of driving functions to create
        drive_type: Type of driving:
            - 'sinusoidal': cos(ωt) [ZERO DC, H_F^(1) = 0]
            - 'offset_sinusoidal': 1 + 0.5*cos(ωt) [NON-ZERO DC, H_F^(1) ≠ 0] **RECOMMENDED**
            - 'bichromatic': 1 + cos(ω₁t) + cos(ω₂t) [Richer Fourier structure]
            - 'square': sign(cos(ωt)) [Sharp transitions]
            - 'multi_freq': GKP-like multi-harmonic
            - 'constant': f(t) = 1 [Static case]
        T: Period (used to set frequencies)
        seed: Random seed for phase randomization

    Returns:
        List of K driving functions
    """
    rng = np.random.RandomState(seed)
    omega_base = 2 * np.pi / T

    functions = []

    if drive_type == 'sinusoidal':
        # Random phases for each operator [ZERO DC!]
        phases = rng.uniform(0, 2*np.pi, K)
        for phi in phases:
            functions.append(sinusoidal_drive(omega_base, phi))

    elif drive_type == 'offset_sinusoidal':
        # Random phases, WITH DC offset [RECOMMENDED]
        phases = rng.uniform(0, 2*np.pi, K)
        for phi in phases:
            functions.append(offset_sinusoidal_drive(omega_base, phi, offset=1.0, amplitude=0.5))

    elif drive_type == 'bichromatic':
        # Two incommensurate frequencies per operator
        for k in range(K):
            omega1 = omega_base * (1 + k * 0.1)  # Slightly detuned
            omega2 = omega_base * np.sqrt(2) * (1 + k * 0.1)  # Incommensurate
            phi1 = rng.uniform(0, 2*np.pi)
            phi2 = rng.uniform(0, 2*np.pi)
            functions.append(bichromatic_drive(omega1, omega2, phi1, phi2, offset=1.0))

    elif drive_type == 'square':
        # Different frequencies for each operator
        for k in range(K):
            omega_k = omega_base * (k + 1)
            functions.append(square_wave_drive(omega_k))

    elif drive_type == 'multi_freq':
        # Multi-frequency with different base frequencies
        for k in range(K):
            omega_k = omega_base * (k + 1) / 2
            functions.append(multi_frequency_drive(omega_k, N=3))

    elif drive_type == 'constant':
        # All constant (no driving, reduces to static case)
        functions = [constant_drive() for _ in range(K)]

    else:
        raise ValueError(f"Unknown drive_type: {drive_type}")

    return functions


# ============================================================================
# Conversion utilities for QuTiP compatibility
# ============================================================================

def qutip_to_numpy(H: qutip.Qobj) -> np.ndarray:
    """Convert QuTiP operator to numpy array."""
    return H.full()


def numpy_to_qutip(H: np.ndarray) -> qutip.Qobj:
    """Convert numpy array to QuTiP operator."""
    d = H.shape[0]
    return qutip.Qobj(H, dims=[[d], [d]])


def hamiltonians_to_numpy(hamiltonians: List) -> List[np.ndarray]:
    """
    Convert list of Hamiltonians (possibly QuTiP) to numpy arrays.

    Args:
        hamiltonians: List of operators (numpy or QuTiP)

    Returns:
        List of numpy arrays
    """
    result = []
    for H in hamiltonians:
        if isinstance(H, qutip.Qobj):
            result.append(qutip_to_numpy(H))
        else:
            result.append(np.asarray(H))
    return result


# ============================================================================
# Floquet Moment Criterion
# ============================================================================

def floquet_moment_criterion(
    psi: np.ndarray,
    phi: np.ndarray,
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List[DrivingFunction],
    T: float = 1.0,
    order: int = 2,
    n_fourier_terms: int = 10,
    tol: float = 1e-10
) -> Tuple[bool, Optional[float], np.ndarray]:
    """
    Compute Floquet moment criterion for reachability.

    Unlike the regular moment criterion (which uses ⟨H_k⟩ and is λ-independent),
    the Floquet moment criterion uses ⟨∂H_F/∂λ_k⟩ which includes commutator
    terms from the second-order Magnus expansion, making it λ-DEPENDENT.

    Mathematical formulation:
    -------------------------
    L_F = [⟨∂H_F/∂λ_1⟩_φ - ⟨∂H_F/∂λ_1⟩_ψ, ..., ⟨∂H_F/∂λ_K⟩_φ - ⟨∂H_F/∂λ_K⟩_ψ]

    Q_F = K×K matrix with elements:
        Q_F[k,m] = ⟨{∂H_F/∂λ_k, ∂H_F/∂λ_m}/2⟩_φ - ⟨{∂H_F/∂λ_k, ∂H_F/∂λ_m}/2⟩_ψ

    where {A,B} = AB + BA is the anticommutator.

    The target |φ⟩ is UNREACHABLE from |ψ⟩ if there exists x such that:
        Q_F + x L_F L_F^T  is positive definite

    Args:
        psi: Initial state vector (numpy array)
        phi: Target state vector (numpy array)
        hamiltonians: List of basis Hamiltonians {H_k}
        lambdas: Coupling coefficients λ_k
        driving_functions: List of time-periodic driving functions
        T: Period of driving
        order: Magnus expansion order (1 or 2)
        n_fourier_terms: Number of Fourier terms for overlaps
        tol: Tolerance for eigenvalue positivity check

    Returns:
        Tuple of:
        - definite: True if UNREACHABLE (Q_F + x L_F L_F^T is positive definite)
        - x_opt: The value of x that makes the matrix positive definite (or None)
        - eigenvalues: Eigenvalues of Q_F (for diagnostics)
    """
    # Compute derivatives ∂H_F/∂λ_k
    dH_F_dlambda = compute_floquet_hamiltonian_derivative(
        hamiltonians, lambdas, driving_functions, T, order, n_fourier_terms
    )

    K = len(dH_F_dlambda)

    # Compute L_F vector
    L_F = np.zeros(K)
    for k in range(K):
        dH_k = dH_F_dlambda[k]
        L_F[k] = np.real(phi.conj() @ dH_k @ phi - psi.conj() @ dH_k @ psi)

    # Compute Q_F matrix (anticommutators)
    Q_F = np.zeros((K, K))
    for k in range(K):
        for m in range(K):
            dH_k = dH_F_dlambda[k]
            dH_m = dH_F_dlambda[m]

            # Anticommutator {∂H_F/∂λ_k, ∂H_F/∂λ_m} / 2
            anticomm = (dH_k @ dH_m + dH_m @ dH_k) / 2

            Q_F[k, m] = np.real(phi.conj() @ anticomm @ phi -
                                psi.conj() @ anticomm @ psi)

    # Compute eigenvalues of Q_F for diagnostics
    Q_F_eigenvalues = np.linalg.eigvalsh(Q_F)

    # Check if Q_F + x L_F L_F^T is positive definite for some x
    L_F_outer = np.outer(L_F, L_F)

    # Test range of x values
    x_values = np.linspace(-10, 10, 1000)

    for x in x_values:
        M = Q_F + x * L_F_outer
        eigvals = np.linalg.eigvalsh(M)

        if np.all(eigvals > tol):  # Positive definite
            return True, x, Q_F_eigenvalues  # UNREACHABLE

    return False, None, Q_F_eigenvalues  # Inconclusive


def floquet_moment_criterion_probability(
    hamiltonians: List[np.ndarray],
    lambdas: np.ndarray,
    driving_functions: List[DrivingFunction],
    T: float = 1.0,
    order: int = 2,
    n_trials: int = 100,
    dim: int = None,
    seed: int = None,
    **kwargs
) -> float:
    """
    Monte Carlo estimate of P(unreachable) using Floquet moment criterion.

    Args:
        hamiltonians: List of basis Hamiltonians
        lambdas: Coupling coefficients
        driving_functions: Time-periodic driving functions
        T: Period
        order: Magnus expansion order
        n_trials: Number of random state pairs to test
        dim: Hilbert space dimension (inferred from hamiltonians if None)
        seed: Random seed
        **kwargs: Additional arguments passed to floquet_moment_criterion

    Returns:
        Probability that random state pairs are unreachable
    """
    if dim is None:
        dim = hamiltonians[0].shape[0]

    rng = np.random.RandomState(seed)
    unreachable_count = 0

    for trial in range(n_trials):
        # Generate random initial and target states (Haar measure)
        psi = rng.randn(dim) + 1j * rng.randn(dim)
        psi = psi / np.linalg.norm(psi)

        phi = rng.randn(dim) + 1j * rng.randn(dim)
        phi = phi / np.linalg.norm(phi)

        # Test Floquet moment criterion
        definite, _, _ = floquet_moment_criterion(
            psi, phi, hamiltonians, lambdas, driving_functions, T, order, **kwargs
        )

        if definite:
            unreachable_count += 1

    return unreachable_count / n_trials
