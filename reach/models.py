"""
Random matrix ensembles and quantum state generation for reachability analysis.

Pipeline Role:
This module provides the foundation for all Monte Carlo experiments by generating
random Hamiltonian ensembles {H₁, H₂, ..., Hₖ} and random target states |φ⟩.
All generation functions use explicit seeding for full reproducibility.

Random Matrix Ensembles:
- **GOE (Gaussian Orthogonal Ensemble)**: Real symmetric matrices from time-reversal
  invariant systems. Generated as H = (A + A^T) / √2 where A ~ N(0,1).

- **GUE (Gaussian Unitary Ensemble)**: Complex Hermitian matrices without symmetry
  constraints. Generated as H = (A + A†) / √2 where A ~ N(0,1) + iN(0,1).

Both ensembles are normalized to unit variance per entry and satisfy:
    H† = H (Hermiticity)

These form the basis for parameterized Hamiltonians:
    H(λ) = Σᵢ₌₁ᴷ λᵢ Hᵢ

where λ ∈ [-1,1]ᴷ are the optimization parameters.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import qutip

from . import settings


def setup_rng(seed: int = settings.SEED) -> np.random.RandomState:
    """
    Create a reproducible random number generator.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configured RandomState instance
    """
    return np.random.RandomState(seed)


def setup_environment(seed: int = settings.SEED) -> None:
    """
    Configure global environment for deterministic quantum computations.

    Args:
        seed: Global random seed
    """
    np.random.seed(seed)
    qutip.settings.auto_tidyup = settings.AUTO_TIDYUP
    warnings.filterwarnings("ignore", category=UserWarning)


def validate_ensemble_params(dim: int, k: int) -> None:
    """
    Validate parameters for Hamiltonian ensemble generation.

    Args:
        dim: Hilbert space dimension
        k: Number of Hamiltonians

    Raises:
        ValueError: If parameters are invalid
    """
    if dim < 2:
        raise ValueError(f"Dimension must be ≥ 2, got {dim}")
    if k < 2:
        raise ValueError(f"Number of Hamiltonians k must be ≥ 2, got {k}")
    if k >= dim:
        raise ValueError(f"k must be < dim, got k={k}, dim={dim}")


def _random_gaussian_matrix(dim: int, real: bool, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate random Gaussian matrix (internal helper).

    Args:
        dim: Matrix dimension
        real: If True, generate real matrix; if False, complex
        rng: Random number generator

    Returns:
        Random Gaussian matrix
    """
    if real:
        return rng.randn(dim, dim)
    else:
        return rng.randn(dim, dim) + 1j * rng.randn(dim, dim)


def random_hermitian_matrix(dim: int, real: bool = True, seed: Optional[int] = None) -> qutip.Qobj:
    """
    Generate random Hermitian matrix from GOE (real=True) or GUE (real=False).

    For GOE: H = (A + A^T) / √2 where A ~ N(0,1)^(d×d)
    For GUE: H = (A + A†) / √2 where A ~ N(0,1)^(d×d) + iN(0,1)^(d×d)

    Args:
        dim: Hilbert space dimension
        real: If True, generate GOE; if False, generate GUE
        seed: Random seed (uses settings.SEED if None)

    Returns:
        Random Hermitian QuTiP operator
    """
    validate_ensemble_params(dim, 2)  # Minimal validation

    if seed is None:
        seed = settings.SEED
    rng = setup_rng(seed)

    # Generate random matrix
    A = _random_gaussian_matrix(dim, real, rng)
    qobj = qutip.Qobj(A)

    # Make Hermitian and normalize
    if real:
        H = (qobj + qobj.trans()) / np.sqrt(2)
    else:
        H = (qobj + qobj.trans().conj()) / np.sqrt(2)

    return H


def random_hamiltonian_ensemble(
    dim: int, k: int, ensemble: str, seed: Optional[int] = None
) -> List[qutip.Qobj]:
    """
    Generate k random Hamiltonians from specified ensemble.

    Args:
        dim: Hilbert space dimension
        k: Number of Hamiltonians to generate
        ensemble: Either "GOE" or "GUE"
        seed: Random seed (uses settings.SEED if None)

    Returns:
        List of k random Hermitian operators

    Raises:
        ValueError: If ensemble is not "GOE" or "GUE"
    """
    validate_ensemble_params(dim, k)

    if ensemble not in ["GOE", "GUE"]:
        raise ValueError(f"Ensemble must be 'GOE' or 'GUE', got '{ensemble}'")

    if seed is None:
        seed = settings.SEED
    rng = setup_rng(seed)

    # Generate with different seeds for each Hamiltonian
    hamiltonians = []
    real_valued = ensemble == "GOE"

    for i in range(k):
        h_seed = rng.randint(0, 2**31 - 1)  # Generate seed for this Hamiltonian
        H = random_hermitian_matrix(dim, real=real_valued, seed=h_seed)
        hamiltonians.append(H)

    return hamiltonians


def random_states(n: int, dim: int, seed: Optional[int] = None) -> List[qutip.Qobj]:
    """
    Generate n random quantum states using controlled seeding.

    Uses QuTiP's rand_ket with explicit numpy seeding for reproducibility.

    Args:
        n: Number of states to generate
        dim: Hilbert space dimension
        seed: Random seed (uses settings.SEED if None)

    Returns:
        List of n random quantum states
    """
    if dim < 2:
        raise ValueError(f"Dimension must be ≥ 2, got {dim}")
    if n < 1:
        raise ValueError(f"Number of states n must be ≥ 1, got {n}")

    if seed is None:
        seed = settings.SEED
    rng = setup_rng(seed)

    states = []
    for i in range(n):
        # Set numpy seed before each qutip call for reproducibility
        state_seed = rng.randint(0, 2**31 - 1)
        np.random.seed(state_seed)
        state = qutip.rand_ket(dim)
        states.append(state)

    # Restore original seeding
    np.random.seed(seed)
    return states


def fock_state(dim: int, n: int = 0) -> qutip.Qobj:
    """
    Generate computational basis state |n⟩.

    Args:
        dim: Hilbert space dimension
        n: Basis state index (default: |0⟩)

    Returns:
        Fock state |n⟩
    """
    if not (0 <= n < dim):
        raise ValueError(f"Basis index n={n} must be in range [0, {dim})")
    return qutip.fock(dim, n)


# Legacy compatibility functions for existing code
def random_k_goes(dim: int, k: int, seed: Optional[int] = None) -> List[qutip.Qobj]:
    """Legacy: Generate k random GOE matrices (deprecated - use random_hamiltonian_ensemble)."""
    warnings.warn(
        "random_k_goes is deprecated, use random_hamiltonian_ensemble(..., ensemble='GOE')",
        DeprecationWarning,
        stacklevel=2,
    )
    return random_hamiltonian_ensemble(dim, k, "GOE", seed)


def random_k_gues(dim: int, k: int, seed: Optional[int] = None) -> List[qutip.Qobj]:
    """Legacy: Generate k random GUE matrices (deprecated - use random_hamiltonian_ensemble)."""
    warnings.warn(
        "random_k_gues is deprecated, use random_hamiltonian_ensemble(..., ensemble='GUE')",
        DeprecationWarning,
        stacklevel=2,
    )
    return random_hamiltonian_ensemble(dim, k, "GUE", seed)
