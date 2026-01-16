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
from typing import List, Optional, Tuple, Union

import numpy as np
import qutip
from scipy.linalg import eigh

from . import settings

logger = logging.getLogger(__name__)


# ================================================================================
# SHARED UTILITIES
# ================================================================================
# General mathematical utilities used across all reachability criteria


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


# ================================================================================
# SPECTRAL CRITERION FUNCTIONS (τ-based overlap maximization)
# ================================================================================
# Core principle: max_λ S(λ) < τ → unreachable
#
# The spectral criterion uses eigendecomposition to compute the spectral overlap
# S(λ) = Σₙ |⟨n(λ)|φ⟩* ⟨n(λ)|ψ⟩|, where |n(λ)⟩ are eigenstates of H(λ).
# Unreachability is determined by comparing max_λ S(λ) against threshold τ.
#
# Key function:
# - spectral_overlap(): Compute S(λ) for given parameters
#
# Dependencies: optimize.py provides the maximization over λ


# Note: spectral_overlap() is defined earlier at line 166
# (placed in utilities section for historical reasons, logically belongs here)


# ================================================================================
# KRYLOV CRITERION FUNCTIONS (subspace projection-residual)
# ================================================================================
# Core principle: ‖|φ⟩ - |φ_proj⟩‖ > ε → unreachable
#
# The Krylov criterion tests whether the target state |φ⟩ lies in the Krylov
# subspace K_m(H, |ψ⟩) = span{|ψ⟩, H|ψ⟩, H²|ψ⟩, ..., H^(m-1)|ψ⟩}.
# Uses Arnoldi iteration (modified Gram-Schmidt) to construct orthonormal basis.
#
# Functions:
# - krylov_basis(): Construct orthonormal basis of K_m via Arnoldi iteration
# - is_unreachable_krylov(): Apply projection-residual test
#
# No optimization required (analytical test on subspace membership)


def krylov_basis(
    H: Union[qutip.Qobj, np.ndarray],
    psi: Union[qutip.Qobj, np.ndarray],
    m: int,
    tol: float = settings.KRYLOV_BREAKDOWN_TOL,
) -> np.ndarray:
    """
    Compute orthonormal basis of Krylov subspace via Arnoldi iteration.

    Computes basis of span{ψ, Hψ, H²ψ, ..., H^(m-1)ψ} using modified
    Gram-Schmidt orthogonalization.

    PROVENANCE: Copied from jupyter_Project_Reachability/reach_bib.py::compute_Krylov_basis
    with minimal adaptation (parameter renaming, input normalization, QuTiP compatibility).

    Algorithm:
    1. Normalize ψ → v₀
    2. For i = 1, ..., m-1:
       - Compute w = H @ v_{i-1}
       - Orthogonalize w against {v₀, ..., v_{i-1}} (complex inner products)
       - Normalize w → v_i
       - If ||w|| < tol, Krylov space degenerates (rank < m), break early

    Args:
        H: Hamiltonian operator (d×d matrix, Hermitian)
        psi: Initial state vector (d-dimensional ket)
        m: Krylov subspace dimension (1 ≤ m ≤ d)
        tol: Breakdown tolerance for rank detection (default: 1e-14)

    Returns:
        V: (d, m) matrix with orthonormal columns spanning Krylov subspace.
           If breakdown occurs at iteration k < m, columns k:m are zero-padded.

    Raises:
        ValueError: If psi is zero vector or m out of range
    """
    # Extract numpy arrays from inputs (handle both np.ndarray and qutip.Qobj)
    if isinstance(H, qutip.Qobj):
        H_matrix = H.full()
    else:
        H_matrix = np.asarray(H)

    if isinstance(psi, qutip.Qobj):
        psi_vec = psi.full().flatten()
    else:
        psi_vec = np.asarray(psi).flatten()

    # Validate inputs
    d = H_matrix.shape[0]
    if H_matrix.shape != (d, d):
        raise ValueError(f"H must be square, got shape {H_matrix.shape}")

    if len(psi_vec) != d:
        raise ValueError(f"psi dimension {len(psi_vec)} != H dimension {d}")

    psi_norm = np.linalg.norm(psi_vec)
    if psi_norm == 0:
        raise ValueError("Input vector psi must be non-zero.")

    if not (1 <= m <= d):
        raise ValueError(f"Krylov rank m={m} must be in range [1, {d}]")

    # Initialize result matrix
    result = np.empty((d, m), dtype=np.complex128)

    # Normalize initial vector
    result[:, 0] = psi_vec / psi_norm

    # Arnoldi iteration (copied from jupyter_Project_Reachability/reach_bib.py)
    for index in range(1, m):
        # Multiply the previous basis vector with H
        w = H_matrix @ result[:, index - 1]

        # Orthogonalize against previous basis vectors
        h = result[:, :index].conj().T @ w  # complex inner products
        w = w - result[:, :index] @ h

        htilde = np.linalg.norm(w)
        # Handle near-zero vectors (Krylov space degenerates)
        if htilde < tol:
            # Subspace has degenerated; return the basis so far
            logger.debug(f"Krylov breakdown at iteration {index}, m={m}, rank={index}")
            result[:, index:] = 0  # fill remaining with zeros
            break
        result[:, index] = w / htilde

    # QR compression for numerical stability
    Q, _ = np.linalg.qr(result, mode="reduced")
    return Q


def krylov_score(
    lambdas: np.ndarray,
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    m: Optional[int] = None,
) -> float:
    """
    Compute continuous Krylov reachability score.

    Evaluates R_Krylov(λ) = ‖P_Kₘ(H(λ))|φ⟩‖² where P_Kₘ is the
    projection operator onto the Krylov subspace.

    Args:
        lambdas: Parameter vector λ ∈ ℝᴷ
        psi: Initial state |ψ⟩
        phi: Target state |φ⟩
        hams: List of K Hamiltonian operators [H₁, H₂, ..., Hₖ]
        m: Krylov rank (default: full dimension d)

    Returns:
        Krylov score R(λ) ∈ [0,1]
        - R ≈ 1: target state lies in Krylov subspace (reachable)
        - R ≈ 0: target state outside Krylov subspace (unreachable)

    Mathematical formula:
        R(λ) = Σₙ₌₀^(m-1) |⟨Kₙ(λ)|φ⟩|²
             = ‖V(V†|φ⟩)‖²
        where V = [|K₀⟩, |K₁⟩, ..., |Kₘ₋₁⟩] is the Krylov basis matrix

    Equivalently:
        R(λ) = 1 - ε²_res(λ)
        where ε_res is the residual norm from projection test
    """
    # Validate parameter dimensions
    k = len(hams)
    if len(lambdas) != k:
        raise ValueError(f"Parameter dimension {len(lambdas)} != number of Hamiltonians {k}")

    # Construct H(λ) = Σₖ λₖ Hₖ
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))

    # Get Hilbert space dimension
    d = H_lambda.shape[0]

    # Set default Krylov rank to full dimension if not specified
    if m is None:
        m = d

    # Validate m
    if not (1 <= m <= d):
        logger.warning(f"Invalid Krylov rank m={m} for dimension d={d}, using m=d")
        m = d

    # Build Krylov basis via Arnoldi iteration
    try:
        V = krylov_basis(H_lambda, psi, m)
    except ValueError as e:
        logger.warning(f"Krylov basis computation failed for λ={lambdas}: {e}")
        return 0.0  # Return maximally unreachable if computation fails

    # Extract phi as column vector
    if isinstance(phi, qutip.Qobj):
        phi_vec = phi.full().flatten()
    else:
        phi_vec = np.asarray(phi).flatten()

    # Compute projection coefficients: c = V†|φ⟩
    coeffs = V.conj().T @ phi_vec

    # Compute Krylov score: R(λ) = ‖c‖² = Σ|cₙ|²
    score = float(np.real(np.vdot(coeffs, coeffs)))

    # Ensure score is in valid range [0, 1] with tolerance
    if not (-settings.OVERLAP_TOLERANCE <= score <= 1.0 + settings.OVERLAP_TOLERANCE):
        logger.warning(f"Krylov score {score:.6f} outside [0,1], clipping")
        score = np.clip(score, 0.0, 1.0)

    return score


def is_unreachable_krylov(
    H: Union[qutip.Qobj, np.ndarray],
    psi: Union[qutip.Qobj, np.ndarray],
    phi: Union[qutip.Qobj, np.ndarray],
    m: int,
    rank_tol: float = settings.KRYLOV_RANK_TOL,
    proj_tol: float = 1e-10,
) -> bool:
    """
    Check unreachability via Krylov projection-residual criterion.

    Mathematical criterion:
    - Krylov subspace K_m(H, ψ) = span{ψ, Hψ, ..., H^(m-1)ψ}
    - φ is reachable from ψ iff φ ∈ K_m(H, ψ)
    - Primary test: ‖φ - V(V†φ)‖ ≤ proj_tol (projection-residual)
    - Fallback test: rank(V) == rank([V | φ]) (rank comparison)


    The projection-residual test is numerically stable and geometrically direct:
    it measures the distance from φ to the Krylov subspace. Small residual
    (≪ 1) indicates membership; residual ≈ 1 indicates non-membership.

    Rank comparison is kept as a fallback for edge cases near the tolerance
    boundary, but the projection test is the primary decision criterion.

    Args:
        H: Hamiltonian operator (d×d)
        psi: Initial state (d-dim)
        phi: Target state (d-dim)
        m: Krylov rank to test (1 ≤ m ≤ d)
        rank_tol: Numerical tolerance for matrix_rank fallback (default: 1e-8)
        proj_tol: Projection residual tolerance (default: 1e-10)

    Returns:
        True if φ is UNREACHABLE (φ ∉ K_m)
        False if φ is REACHABLE (φ ∈ K_m)

    Raises:
        ValueError: If inputs are malformed
    """
    # Compute Krylov basis
    try:
        V = krylov_basis(H, psi, m)
    except ValueError as e:
        logger.warning(f"Krylov basis computation failed: {e}")
        return False  # Conservative: assume reachable if computation fails

    # Extract phi as column vector
    if isinstance(phi, qutip.Qobj):
        phi_vec = phi.full().flatten()
    else:
        phi_vec = np.asarray(phi).flatten()

    # PRIMARY TEST: Projection-residual criterion
    # Compute orthogonal projection of φ onto K_m: proj = V(V†φ)
    proj = V @ (V.conj().T @ phi_vec)

    # Compute residual: r = φ - proj
    resid = phi_vec - proj
    resid_norm = np.linalg.norm(resid)

    # If residual is large, φ is outside K_m (unreachable)
    if resid_norm > proj_tol:
        return True  # Unreachable: φ ∉ K_m

    # FALLBACK TEST: Rank comparison (for edge cases near tolerance)
    # This is kept for numerical safety but should rarely trigger
    phi_col = phi_vec.reshape(-1, 1)
    V_aug = np.concatenate([V, phi_col], axis=1)

    rank_V = np.linalg.matrix_rank(V, tol=rank_tol)
    rank_V_aug = np.linalg.matrix_rank(V_aug, tol=rank_tol)

    # Return True if unreachable (rank increased)
    return rank_V < rank_V_aug
