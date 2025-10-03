"""
Core mathematical utilities for quantum reachability analysis.

Pipeline Role:
This module implements the mathematical core of time-free reachability analysis.
It bridges between parameterized Hamiltonians H(λ) and the spectral overlap
function S(λ) that determines reachability.

Key Equations:

1. **Parameterized Hamiltonian**:
   H(λ) = Σᵢ₌₁ᴷ λᵢ Hᵢ  where λ ∈ [-1,1]ᴷ

2. **Hermitian Eigendecomposition** (via scipy.linalg.eigh):
   H(λ) = U(λ) diag(E₁, ..., Eₐ) U†(λ)

   We use scipy.linalg.eigh instead of eig because:
   - Hermitian matrices guarantee real eigenvalues
   - eigh is numerically more stable (uses specialized algorithms)
   - Faster for Hermitian case

3. **State Projections onto Eigenbasis**:
   ψₙ(λ) = ⟨n(λ)|ψ⟩  (initial state projection)
   φₙ(λ) = ⟨n(λ)|φ⟩  (target state projection)

   where |n(λ)⟩ is the nth eigenstate of H(λ)

4. **Spectral Overlap** (time-free criterion):
   S(λ) = Σₙ₌₁ᵈ |φₙ*(λ) ψₙ(λ)| ∈ [0,1]

   Interpretation: Measures alignment between initial and target states
   in the eigenbasis of H(λ). Higher overlap → more reachable.

5. **Binomial Standard Error**:
   SEM(p) = √(p(1-p)/N)

   Used for error bars on probability estimates from Monte Carlo sampling.

All functions include numerical stability checks and graceful error handling.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import numpy as np
import qutip
from scipy.linalg import eigh

from . import settings

logger = logging.getLogger(__name__)


def validate_hermitian(
    H: Union[qutip.Qobj, np.ndarray], tolerance: float = settings.OVERLAP_TOLERANCE
) -> bool:
    """
    Check if matrix is Hermitian within specified tolerance.

    Args:
        H: Matrix to check (QuTiP Qobj or numpy array)
        tolerance: Maximum allowed deviation from Hermiticity

    Returns:
        True if matrix is Hermitian within tolerance
    """
    if isinstance(H, qutip.Qobj):
        matrix = H.full()
    else:
        matrix = H

    hermitian_diff = np.linalg.norm(matrix - matrix.conj().T)
    return hermitian_diff < tolerance


def validate_bounds(bounds: List[Tuple[float, float]], k: int) -> None:
    """
    Validate optimization bounds for k-dimensional parameter space.

    Args:
        bounds: List of (lower, upper) bound tuples
        k: Expected dimension of parameter space

    Raises:
        ValueError: If bounds are malformed or inconsistent
    """
    if len(bounds) == 1 and k > 1:
        # Broadcast single bound to all dimensions
        return

    if len(bounds) != k:
        raise ValueError(f"Bounds length {len(bounds)} != parameter dimension k={k}")

    for i, (lower, upper) in enumerate(bounds):
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise ValueError(f"Bound {i} contains non-numeric values: ({lower}, {upper})")
        if lower >= upper:
            raise ValueError(f"Bound {i} invalid: lower={lower} >= upper={upper}")


def eigendecompose(H: qutip.Qobj, validate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe eigendecomposition with validation and error handling.

    Performs eigendecomposition H = U Λ U† where:
    - Λ = diag(E₁, E₂, ..., Eₑ) are real eigenvalues
    - U are corresponding eigenvectors (columns)

    Args:
        H: Hermitian operator to diagonalize
        validate: Whether to check Hermiticity before decomposition

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors are column-wise

    Raises:
        ValueError: If H is not Hermitian (when validate=True)
        RuntimeError: If eigendecomposition fails
    """
    if not isinstance(H, qutip.Qobj):
        raise TypeError(f"Expected qutip.Qobj, got {type(H)}")

    matrix = H.full()

    # Validate Hermiticity if requested
    if validate and not validate_hermitian(matrix):
        raise ValueError("Matrix is not Hermitian within tolerance")

    try:
        # Use scipy.linalg.eigh for Hermitian matrices (more stable than eig)
        eigenvalues, eigenvectors = eigh(matrix)

        # Check for numerical issues
        if np.any(~np.isfinite(eigenvalues)):
            raise RuntimeError("Eigendecomposition produced non-finite eigenvalues")

        if np.any(~np.isfinite(eigenvectors)):
            raise RuntimeError("Eigendecomposition produced non-finite eigenvectors")

        # Check condition number for stability warning
        if eigenvalues.size > 1:
            condition_number = (
                np.abs(eigenvalues.max() / eigenvalues.min()) if eigenvalues.min() != 0 else np.inf
            )
            if condition_number > 1.0 / settings.MIN_CONDITION_NUMBER:
                logger.warning(
                    f"Ill-conditioned eigendecomposition: condition number = {condition_number:.2e}"
                )

        return eigenvalues, eigenvectors

    except Exception as e:
        error_msg = f"Eigendecomposition failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def spectral_overlap(
    lambdas: np.ndarray, psi: qutip.Qobj, phi: qutip.Qobj, hams: List[qutip.Qobj]
) -> float:
    """
    Compute time-free spectral overlap criterion.

    The spectral overlap is defined as:
    S(λ) = Σₙ |⟨φₙ(λ)|ψₙ(λ)⟩|

    where |ψₙ(λ)⟩ and |φₙ(λ)⟩ are the projections of initial and target
    states onto the nth eigenstate of H(λ) = Σₖ λₖ Hₖ.

    Mathematical steps:
    1. Construct H(λ) = Σₖ λₖ Hₖ
    2. Diagonalize: H(λ) = U(λ) diag(E₁,...,Eₑ) U†(λ)
    3. Project states: ψₙ(λ) = ⟨uₙ(λ)|ψ⟩, φₙ(λ) = ⟨uₙ(λ)|φ⟩
    4. Compute: S(λ) = Σₙ |φₙ*(λ) ψₙ(λ)|

    Args:
        lambdas: Parameter vector λ = (λ₁, λ₂, ..., λₖ)
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of k Hamiltonian operators [H₁, H₂, ..., Hₖ]

    Returns:
        Spectral overlap S(λ) ∈ [0, 1]

    Raises:
        ValueError: If parameter dimensions don't match
        RuntimeError: If eigendecomposition fails
    """
    k = len(hams)
    if len(lambdas) != k:
        raise ValueError(f"Parameter dimension {len(lambdas)} != number of Hamiltonians {k}")

    # Construct H(λ) = Σₖ λₖ Hₖ
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))

    # Safe eigendecomposition
    try:
        eigenvalues, eigenvectors = eigendecompose(H_lambda, validate=True)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Eigendecomposition failed for λ={lambdas}: {e}")
        return 0.0

    # Extract state vectors
    psi_vec = psi.full().flatten()
    phi_vec = phi.full().flatten()

    # Project onto eigenbasis: ψₙ = ⟨uₙ|ψ⟩, φₙ = ⟨uₙ|φ⟩
    psi_coeffs = eigenvectors.conj().T @ psi_vec  # Shape: (d,)
    phi_coeffs = eigenvectors.conj().T @ phi_vec  # Shape: (d,)

    # Spectral overlap: S(λ) = Σₙ |φₙ*(λ) ψₙ(λ)|
    overlap_terms = np.abs(phi_coeffs.conj() * psi_coeffs)
    spectral_overlap_value = np.sum(overlap_terms)

    # Ensure real and validate bounds
    spectral_overlap_value = float(np.real(spectral_overlap_value))

    # Check bounds with tolerance
    if not (
        -settings.OVERLAP_TOLERANCE <= spectral_overlap_value <= 1.0 + settings.OVERLAP_TOLERANCE
    ):
        logger.warning(f"Spectral overlap {spectral_overlap_value:.6f} outside [0,1]")
        spectral_overlap_value = np.clip(spectral_overlap_value, 0.0, 1.0)

    return spectral_overlap_value


def compute_binomial_sem(p: float, n: int) -> float:
    """
    Compute standard error of the mean for binomial proportion.

    For a binomial random variable X ~ Binomial(n, p), the sample proportion
    p̂ = X/n has variance Var(p̂) = p(1-p)/n, so SEM = √(p(1-p)/n).

    Args:
        p: Observed proportion (success rate)
        n: Number of trials

    Returns:
        Standard error of the mean

    Raises:
        ValueError: If parameters are invalid
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Proportion p={p} must be in [0,1]")
    if n <= 0:
        raise ValueError(f"Number of trials n={n} must be positive")

    # Handle edge cases
    if p == 0.0 or p == 1.0 or n == 1:
        return 0.0

    return np.sqrt(p * (1 - p) / n)


def clip_to_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clip parameter vector to specified bounds.

    Args:
        x: Parameter vector to clip
        bounds: List of (lower, upper) bounds

    Returns:
        Clipped parameter vector

    Raises:
        ValueError: If bounds are malformed
    """
    x = np.asarray(x)

    # Handle single bound broadcast to all dimensions
    if len(bounds) == 1:
        bounds = bounds * len(x)

    validate_bounds(bounds, len(x))

    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    return np.clip(x, lower_bounds, upper_bounds)


def construct_hamiltonian(lambdas: np.ndarray, hams: List[qutip.Qobj]) -> qutip.Qobj:
    """
    Construct parameterized Hamiltonian H(λ) = Σₖ λₖ Hₖ.

    Args:
        lambdas: Parameter vector λ = (λ₁, λ₂, ..., λₖ)
        hams: List of k Hamiltonian operators

    Returns:
        Parameterized Hamiltonian H(λ)

    Raises:
        ValueError: If dimensions don't match
    """
    if len(lambdas) != len(hams):
        raise ValueError(
            f"Parameter dimension {len(lambdas)} != number of Hamiltonians {len(hams)}"
        )

    return sum(lam * H for lam, H in zip(lambdas, hams))


def validate_quantum_state(state: qutip.Qobj, tolerance: float = 1e-10) -> bool:
    """
    Validate that object is a normalized quantum state.

    Args:
        state: State to validate
        tolerance: Tolerance for normalization check

    Returns:
        True if state is valid and normalized
    """
    if not isinstance(state, qutip.Qobj):
        return False

    if not state.isket:
        return False

    # Check normalization
    norm = state.norm()
    return abs(norm - 1.0) < tolerance
