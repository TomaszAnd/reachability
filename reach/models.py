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
from typing import List, Optional, Tuple

import numpy as np
import qutip
from scipy.sparse import csr_matrix, eye as speye, kron as spkron

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

    Note: K >= d is now allowed for density sweeps (uses m = min(K, d) for Krylov).

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


class CanonicalBasis:
    """
    Canonical basis for d×d Hermitian matrices.

    Basis consists of:
    - X_jk = |j⟩⟨k| + |k⟩⟨j| for j < k (Pauli-X like, symmetric)
    - Y_jk = -i(|j⟩⟨k| - |k⟩⟨j|) for j < k (Pauli-Y like, antisymmetric)
    - Z_j = |j⟩⟨j| - |j+1⟩⟨j+1| for j < d-1 (Pauli-Z like, diagonal)
    - I = Identity matrix (optional, to complete the basis to d² operators)

    Total operators: d(d-1)/2 + d(d-1)/2 + (d-1) + 1 = d² operators

    This provides a deterministic, structured basis for parameterized Hamiltonians,
    in contrast to random ensembles like GOE/GUE.

    Args:
        dim: Hilbert space dimension
        include_identity: If True, include identity matrix as d²-th basis operator
    """

    def __init__(self, dim: int, cdinclude_identity: bool = True):
        if dim < 2:
            raise ValueError(f"Dimension must be ≥ 2, got {dim}")

        self.dim = dim
        self.include_identity = include_identity

        # Build canonical basis operators
        self.operators = self._build_canonical_basis()
        self.L = len(self.operators)

        # Validate operator count
        expected_L = dim * dim if include_identity else dim * dim - 1
        assert self.L == expected_L, (
            f"Operator count mismatch: got {self.L}, expected {expected_L}"
        )

    def _build_canonical_basis(self) -> List[qutip.Qobj]:
        """
        Build canonical basis operators {X_jk, Y_jk, Z_j, I}.

        Returns:
            List of d² Hermitian operators forming a complete basis
        """
        d = self.dim
        operators = []

        # X_jk operators: |j⟩⟨k| + |k⟩⟨j| for j < k
        for j in range(d):
            for k in range(j + 1, d):
                # Create matrix with 1 at (j,k) and (k,j)
                mat = np.zeros((d, d), dtype=complex)
                mat[j, k] = 1.0
                mat[k, j] = 1.0
                operators.append(qutip.Qobj(mat))

        # Y_jk operators: -i(|j⟩⟨k| - |k⟩⟨j|) for j < k
        for j in range(d):
            for k in range(j + 1, d):
                # Create matrix with -i at (j,k) and +i at (k,j)
                mat = np.zeros((d, d), dtype=complex)
                mat[j, k] = -1j
                mat[k, j] = 1j
                operators.append(qutip.Qobj(mat))

        # Z_j operators: |j⟩⟨j| - |j+1⟩⟨j+1| for j < d-1
        for j in range(d - 1):
            # Create diagonal matrix with +1 at position j and -1 at position j+1
            mat = np.zeros((d, d), dtype=complex)
            mat[j, j] = 1.0
            mat[j + 1, j + 1] = -1.0
            operators.append(qutip.Qobj(mat))

        # Identity operator (optional)
        if self.include_identity:
            operators.append(qutip.qeye(d))

        return operators

    def sample_k_operators(self, k: int, rng: np.random.RandomState) -> List[qutip.Qobj]:
        """
        Randomly sample k operators from the canonical basis without replacement.

        Args:
            k: Number of operators to sample
            rng: Random number generator

        Returns:
            List of k canonical basis operators

        Raises:
            ValueError: If k > number of available operators
        """
        if k > self.L:
            raise ValueError(
                f"Cannot sample {k} operators from canonical basis of size {self.L}"
            )
        if k < 2:
            raise ValueError(f"Need at least 2 operators, got k={k}")

        # Sample indices without replacement
        indices = rng.choice(self.L, size=k, replace=False)

        # Return corresponding operators
        return [self.operators[i] for i in indices]


class GeometricTwoLocal:
    """
    Gaussian Geo-Local (GEO2) ensemble on a rectangular lattice.

    Basis: P_2(G) contains all 1-local {X,Y,Z}_i and 2-local {X,Y,Z}_i⊗{X,Y,Z}_j
    Pauli terms on lattice sites and nearest-neighbor edges.

    Hamiltonian: H = (1/√L) Σ_a g_a P_a where g_a ~ N(0,1), L = |P_2(G)|.

    Formula: L = 3n + 9|E(G)| where n = nx * ny sites, |E(G)| = number of edges.

    Args:
        nx: Lattice width (number of sites in x direction)
        ny: Lattice height (number of sites in y direction)
        periodic: Use periodic boundary conditions
        backend: "sparse" (default) or "dense" operator construction
    """

    def __init__(self, nx: int, ny: int, periodic: bool = False, backend: str = "sparse"):
        if nx < 1 or ny < 1:
            raise ValueError(f"Lattice dimensions must be ≥ 1, got nx={nx}, ny={ny}")

        self.nx = nx
        self.ny = ny
        self.periodic = periodic
        self.backend = backend
        self.n_sites = nx * ny
        self.dim = 2 ** self.n_sites

        # Build sparse Pauli basis
        self.Hs = self._build_pauli_basis()
        self.L = len(self.Hs)

        # Validate operator count: L = 3n + 9|E|
        edges = self._build_lattice_edges()
        expected_L = 3 * self.n_sites + 9 * len(edges)
        assert self.L == expected_L, (
            f"Operator count mismatch: got {self.L}, expected {expected_L} "
            f"(3n={3*self.n_sites} + 9|E|={9*len(edges)})"
        )

    def _build_lattice_edges(self) -> List[Tuple[int, int]]:
        """Build edge list for rectangular lattice with nearest-neighbor connectivity."""
        edges = []
        for y in range(self.ny):
            for x in range(self.nx):
                site = y * self.nx + x

                # Right neighbor (x+1)
                if x + 1 < self.nx:
                    neighbor = y * self.nx + (x + 1)
                    edges.append((site, neighbor))
                elif self.periodic and self.nx > 1:
                    neighbor = y * self.nx + 0
                    edges.append((site, neighbor))

                # Down neighbor (y+1)
                if y + 1 < self.ny:
                    neighbor = (y + 1) * self.nx + x
                    edges.append((site, neighbor))
                elif self.periodic and self.ny > 1:
                    neighbor = 0 * self.nx + x
                    edges.append((site, neighbor))

        return edges

    def _build_pauli_basis(self) -> List[qutip.Qobj]:
        """Build sparse Pauli basis P_2(G) for the lattice."""
        if self.backend == "sparse":
            return self._build_sparse_pauli_basis()
        else:
            return self._build_dense_pauli_basis()

    def _build_sparse_pauli_basis(self) -> List[qutip.Qobj]:
        """Build Pauli basis using sparse matrix operations."""
        # Pauli matrices as sparse matrices
        pauli_x = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
        pauli_y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
        pauli_z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        identity = speye(2, dtype=complex, format='csr')

        paulis = [pauli_x, pauli_y, pauli_z]
        basis = []

        # 1-local terms: X_i, Y_i, Z_i for each site
        for site in range(self.n_sites):
            for pauli in paulis:
                # Build I ⊗ ... ⊗ I ⊗ Pauli_site ⊗ I ⊗ ... ⊗ I
                op = identity
                for s in range(self.n_sites):
                    if s == 0:
                        op = pauli if s == site else identity
                    else:
                        op = spkron(op, pauli if s == site else identity, format='csr')

                basis.append(qutip.Qobj(op, dims=[[self.dim], [self.dim]]))

        # 2-local terms: Pauli_i ⊗ Pauli_j for each edge
        edges = self._build_lattice_edges()
        for site_i, site_j in edges:
            for pauli_i in paulis:
                for pauli_j in paulis:
                    # Build tensor product with Pauli operators on sites i and j
                    op = identity
                    for s in range(self.n_sites):
                        if s == 0:
                            if s == site_i:
                                op = pauli_i
                            elif s == site_j:
                                op = pauli_j
                            else:
                                op = identity
                        else:
                            if s == site_i:
                                op = spkron(op, pauli_i, format='csr')
                            elif s == site_j:
                                op = spkron(op, pauli_j, format='csr')
                            else:
                                op = spkron(op, identity, format='csr')

                    basis.append(qutip.Qobj(op, dims=[[self.dim], [self.dim]]))

        return basis

    def _build_dense_pauli_basis(self) -> List[qutip.Qobj]:
        """Build Pauli basis using dense qutip tensor products (for small systems)."""
        # Pauli matrices
        pauli_ops = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
        identity = qutip.qeye(2)

        basis = []

        # 1-local terms
        for site in range(self.n_sites):
            for pauli in pauli_ops:
                ops = [identity] * self.n_sites
                ops[site] = pauli
                term = qutip.tensor(ops)
                # Flatten dims to match pipeline expectations
                term.dims = [[self.dim], [self.dim]]
                basis.append(term)

        # 2-local terms
        edges = self._build_lattice_edges()
        for site_i, site_j in edges:
            for pauli_i in pauli_ops:
                for pauli_j in pauli_ops:
                    ops = [identity] * self.n_sites
                    ops[site_i] = pauli_i
                    ops[site_j] = pauli_j
                    term = qutip.tensor(ops)
                    # Flatten dims to match pipeline expectations
                    term.dims = [[self.dim], [self.dim]]
                    basis.append(term)

        return basis

    def sample_lambda(self, rng) -> np.ndarray:
        """
        Sample Gaussian coefficients for GEO2 Hamiltonian.

        Returns: λ = g/√L where g ~ N(0, I_L), so E[λ_a^2] = 1/L.

        Args:
            rng: numpy random Generator or RandomState
        """
        # Support both old and new numpy random APIs
        if hasattr(rng, 'standard_normal'):
            return rng.standard_normal(self.L) / np.sqrt(self.L)
        else:
            return rng.randn(self.L) / np.sqrt(self.L)

    def sample_hamiltonian(self, rng: np.random.RandomState) -> qutip.Qobj:
        """Generate one GEO2 Hamiltonian instance: H = Σ_a λ_a H_a."""
        lambdas = self.sample_lambda(rng)
        H = sum(lam * H for lam, H in zip(lambdas, self.Hs))
        return H


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


# Ensemble registry for factory pattern
ENSEMBLES = {
    "GOE": "GOE",
    "GUE": "GUE",
    "GEO2": "GEO2",
    "canonical": "canonical",
}


def random_hamiltonian_ensemble(
    dim: int, k: int, ensemble: str, seed: Optional[int] = None, **kwargs
) -> List[qutip.Qobj]:
    """
    Generate k random Hamiltonians from specified ensemble.

    Args:
        dim: Hilbert space dimension
        k: Number of Hamiltonians to generate
        ensemble: "GOE", "GUE", "GEO2", or "canonical"
        seed: Random seed (uses settings.SEED if None)
        **kwargs: Ensemble-specific parameters
            - For GEO2: nx, ny, periodic
            - For canonical: include_identity (default True)

    Returns:
        List of k random Hermitian operators

    Raises:
        ValueError: If ensemble is not recognized or parameters are invalid
    """
    validate_ensemble_params(dim, k)

    if ensemble not in ENSEMBLES:
        raise ValueError(
            f"Ensemble must be one of {list(ENSEMBLES.keys())}, got '{ensemble}'"
        )

    if seed is None:
        seed = settings.SEED
    rng = setup_rng(seed)

    hamiltonians = []

    if ensemble == "GEO2":
        # Extract lattice parameters
        nx = kwargs.get("nx")
        ny = kwargs.get("ny")
        periodic = kwargs.get("periodic", False)
        geo2_optimize_weights = kwargs.get("geo2_optimize_weights", False)

        if nx is None or ny is None:
            raise ValueError("GEO2 ensemble requires 'nx' and 'ny' parameters")

        # Validate dimension matches lattice size
        n_sites = nx * ny
        expected_dim = 2 ** n_sites
        if dim != expected_dim:
            raise ValueError(
                f"Dimension {dim} does not match lattice size {nx}×{ny} = {n_sites} sites "
                f"(expected dimension 2^{n_sites} = {expected_dim})"
            )

        # Create GEO2 instance (builds basis once)
        geo2 = GeometricTwoLocal(nx, ny, periodic, backend="sparse")

        if geo2_optimize_weights:
            # Approach 1: Sample K basis operators (weights optimized by maximize_* functions)
            # Similar to canonical ensemble - select without replacement
            if k > geo2.L:
                raise ValueError(
                    f"Cannot sample {k} operators from GEO2 basis of size {geo2.L} "
                    f"(lattice {nx}×{ny}). Maximum k = {geo2.L}."
                )
            indices = rng.choice(geo2.L, size=k, replace=False)
            hamiltonians = [geo2.Hs[i] for i in indices]
        else:
            # Approach 2a: Sample k Hamiltonians with fixed random weights (default, arXiv definition)
            for i in range(k):
                h_seed = rng.randint(0, 2**31 - 1)
                h_rng = setup_rng(h_seed)
                H = geo2.sample_hamiltonian(h_rng)
                hamiltonians.append(H)

    elif ensemble == "canonical":
        # Canonical basis: sample k operators from {X_jk, Y_jk, Z_j, I}
        include_identity = kwargs.get("include_identity", True)

        # Create canonical basis instance (builds all d² operators)
        canonical = CanonicalBasis(dim, include_identity=include_identity)

        # Validate k doesn't exceed basis size
        if k > canonical.L:
            raise ValueError(
                f"Cannot sample {k} operators from canonical basis of size {canonical.L} "
                f"(dimension {dim}). Maximum k = {canonical.L}."
            )

        # Sample k operators without replacement
        hamiltonians = canonical.sample_k_operators(k, rng)

    else:
        # GOE or GUE
        real_valued = ensemble == "GOE"
        for i in range(k):
            h_seed = rng.randint(0, 2**31 - 1)
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
        state = qutip.rand_ket(dim) # <- is that sampled haar uniformly
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
