"""
Hamiltonian families for quantum reachability analysis.

A Hamiltonian family defines an operator basis {B_1, ..., B_L}.
We select K operators to form a control family {H_1, ..., H_K}.
The combined Hamiltonian is H(lambda) = sum_k lambda_k H_k, where lambda in [-1, 1]^K.

Two families are supported:
- CanonicalModel: generalized Pauli basis {X_jk, Y_jk, Z_j, I} for arbitrary dimension d
- GeometricTwoLocalModel: Pauli chains on a rectangular qubit grid

Design principles:
- K is the primary parameter (not rho = K/d^2)
- sample_submodel(K) returns a new QuantumModel, not a list
- numpy arrays throughout (no qutip.Qobj in interfaces)
- np.random.SeedSequence with .spawn() for parallel-safe RNG
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, eye as speye, kron as spkron


@dataclass
class ModelMetadata:
    """Describes how a submodel was derived."""
    parent_basis_size: int = 0
    selected_indices: Optional[np.ndarray] = None
    description: str = ""


class QuantumModel(ABC):
    """
    Abstract base class for Hamiltonian families.

    A Hamiltonian family defines an operator basis {B_1, ..., B_L}.
    We select K operators to form a control family {H_1, ..., H_K}.
    The combined Hamiltonian is H(lambda) = sum_k lambda_k H_k,
    where lambda in [-1, 1]^K.
    """

    def __init__(self, dim: int, seed: Optional[int] = None):
        if dim < 2:
            raise ValueError(f"Dimension must be >= 2, got {dim}")
        self.dim = dim
        self._seed_seq = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(self._seed_seq)
        self._basis: Optional[List[np.ndarray]] = None
        self._metadata = ModelMetadata()

    @abstractmethod
    def _build_basis(self) -> List[np.ndarray]:
        """Build the complete operator basis as numpy arrays."""
        pass

    @property
    def basis(self) -> List[np.ndarray]:
        """Lazily-built operator basis as list of (d, d) numpy arrays."""
        if self._basis is None:
            self._basis = self._build_basis()
            self._metadata.parent_basis_size = len(self._basis)
        return self._basis

    @property
    def K(self) -> int:
        """Number of control operators in this model."""
        return len(self.basis)

    @property
    def basis_size(self) -> int:
        """Total number of operators in the full basis (L)."""
        return len(self.basis)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @abstractmethod
    def sample_submodel(self, k: int) -> 'QuantumModel':
        """
        Create a new QuantumModel with k operators sampled from this basis.

        Args:
            k: Number of operators to select

        Returns:
            A new QuantumModel with k operators as its basis
        """
        pass

    def spawn_rng(self) -> np.random.Generator:
        """Create an independent child RNG for parallel/nested sampling."""
        child_seq = self._seed_seq.spawn(1)[0]
        return np.random.default_rng(child_seq)

    def init_state(self) -> np.ndarray:
        """Default initial state |0> (or |00...0> for qubit systems)."""
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0
        return state

    def random_state(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate a Haar-random pure state as a (d,) numpy array."""
        if rng is None:
            rng = self._rng
        vec = rng.standard_normal(self.dim) + 1j * rng.standard_normal(self.dim)
        return vec / np.linalg.norm(vec)

    def construct_hamiltonian(self, lambdas: np.ndarray) -> np.ndarray:
        """Build H(lambda) = sum_k lambda_k H_k."""
        return sum(l * h for l, h in zip(lambdas, self.basis))

    @classmethod
    def create(cls, family_type: str, dim: int, **kwargs) -> 'QuantumModel':
        """
        Factory method.

        Args:
            family_type: 'canonical' or 'geometric_local' (aliases: 'geo2', 'qubit_grid')
            dim: Hilbert space dimension
            **kwargs: Passed to the specific model constructor

        Returns:
            QuantumModel instance
        """
        ft = family_type.lower()
        if ft == "canonical":
            return CanonicalModel(dim, **kwargs)
        elif ft in ("geometric_local", "geo2", "qubit_grid"):
            return GeometricTwoLocalModel(dim, **kwargs)
        raise ValueError(f"Unknown family type: {family_type!r}")


class _SubModel(QuantumModel):
    """A submodel created by selecting operators from a parent model."""

    def __init__(self, dim: int, operators: List[np.ndarray],
                 metadata: ModelMetadata, seed: Optional[int] = None):
        super().__init__(dim, seed)
        self._basis = operators
        self._metadata = metadata

    def _build_basis(self) -> List[np.ndarray]:
        return self._basis

    def sample_submodel(self, k: int) -> 'QuantumModel':
        if k > self.K:
            raise ValueError(
                f"Cannot sample {k} operators from submodel with {self.K} operators")
        if k < 2:
            raise ValueError(f"Need at least 2 operators, got k={k}")
        indices = self._rng.choice(self.K, size=k, replace=False)
        ops = [self._basis[i] for i in indices]
        meta = ModelMetadata(
            parent_basis_size=self.K,
            selected_indices=indices,
            description=f"Sub-submodel: {k} of {self.K} operators",
        )
        child_seed = self._seed_seq.spawn(1)[0].entropy
        return _SubModel(self.dim, ops, meta, seed=child_seed)


class CanonicalModel(QuantumModel):
    """
    Canonical basis for d x d Hermitian matrices.

    Basis: {X_jk, Y_jk, Z_j, I}, total d^2 operators.
    - X_jk = |j><k| + |k><j| for j < k
    - Y_jk = -i(|j><k| - |k><j|) for j < k
    - Z_j = |j><j| - |j+1><j+1| for j < d-1
    - I = identity (optional)
    """

    def __init__(self, dim: int, include_identity: bool = True,
                 seed: Optional[int] = None):
        self.include_identity = include_identity
        super().__init__(dim, seed)

    def _build_basis(self) -> List[np.ndarray]:
        d = self.dim
        operators = []

        # X_jk operators
        for j in range(d):
            for k in range(j + 1, d):
                mat = np.zeros((d, d), dtype=np.complex128)
                mat[j, k] = 1.0
                mat[k, j] = 1.0
                operators.append(mat)

        # Y_jk operators
        for j in range(d):
            for k in range(j + 1, d):
                mat = np.zeros((d, d), dtype=np.complex128)
                mat[j, k] = -1j
                mat[k, j] = 1j
                operators.append(mat)

        # Z_j operators
        for j in range(d - 1):
            mat = np.zeros((d, d), dtype=np.complex128)
            mat[j, j] = 1.0
            mat[j + 1, j + 1] = -1.0
            operators.append(mat)

        # Identity
        if self.include_identity:
            operators.append(np.eye(d, dtype=np.complex128))

        expected_L = d * d if self.include_identity else d * d - 1
        assert len(operators) == expected_L, (
            f"Operator count mismatch: got {len(operators)}, expected {expected_L}")

        return operators

    def sample_submodel(self, k: int) -> QuantumModel:
        """Sample k operators from the d^2 canonical basis without replacement."""
        L = self.basis_size
        if k > L:
            raise ValueError(
                f"Cannot sample {k} operators from canonical basis of size {L}")
        if k < 2:
            raise ValueError(f"Need at least 2 operators, got k={k}")

        indices = self._rng.choice(L, size=k, replace=False)
        ops = [self.basis[i] for i in indices]

        meta = ModelMetadata(
            parent_basis_size=L,
            selected_indices=indices,
            description=f"Canonical submodel: {k} of {L} operators (d={self.dim})",
        )
        child_seed = self._seed_seq.spawn(1)[0].entropy
        return _SubModel(self.dim, ops, meta, seed=child_seed)


# GEO2 lattice configs: dimension -> (nx, ny)
GEO2_LATTICE_CONFIGS = {
    8: (1, 3),
    16: (2, 2),
    32: (1, 5),
    64: (2, 3),
}


class GeometricTwoLocalModel(QuantumModel):
    """
    Geometric two-local Hamiltonian family on a rectangular qubit lattice.

    Basis: P_2(G) = {X_i, Y_i, Z_i} (1-local) + {sigma_i x sigma_j} (2-local)
    Total operators: L = 3n + 9|E(G)| where n = nx * ny, |E| = edges.

    For each site, there are 3 single-qubit Pauli operators.
    For each edge, there are 9 two-qubit Pauli tensor products.

    Dimension: d = 2^(nx * ny).
    """

    def __init__(self, dim: int, nx: Optional[int] = None,
                 ny: Optional[int] = None, periodic: bool = False,
                 seed: Optional[int] = None):
        # Infer lattice dimensions if not provided
        if nx is None or ny is None:
            if dim in GEO2_LATTICE_CONFIGS:
                nx, ny = GEO2_LATTICE_CONFIGS[dim]
            else:
                raise ValueError(
                    f"Cannot infer lattice for d={dim}. "
                    f"Provide nx, ny explicitly. Known dims: {list(GEO2_LATTICE_CONFIGS.keys())}")

        self.nx = nx
        self.ny = ny
        self.periodic = periodic
        self.n_sites = nx * ny

        expected_dim = 2 ** self.n_sites
        if dim != expected_dim:
            raise ValueError(
                f"Dimension {dim} does not match lattice {nx}x{ny} = {self.n_sites} sites "
                f"(expected 2^{self.n_sites} = {expected_dim})")

        super().__init__(dim, seed)

    def _build_lattice_edges(self) -> List[Tuple[int, int]]:
        """Build edge list for rectangular lattice."""
        edges = []
        for y in range(self.ny):
            for x in range(self.nx):
                site = y * self.nx + x
                if x + 1 < self.nx:
                    edges.append((site, y * self.nx + (x + 1)))
                elif self.periodic and self.nx > 1:
                    edges.append((site, y * self.nx + 0))
                if y + 1 < self.ny:
                    edges.append((site, (y + 1) * self.nx + x))
                elif self.periodic and self.ny > 1:
                    edges.append((site, 0 * self.nx + x))
        return edges

    def _build_basis(self) -> List[np.ndarray]:
        """Build Pauli basis P_2(G) using sparse matrix operations."""
        pauli_x = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
        pauli_y = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
        pauli_z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        identity = speye(2, dtype=complex, format='csr')
        paulis = [pauli_x, pauli_y, pauli_z]
        basis = []

        def _build_operator(site_ops):
            """Build tensor product operator given {site: pauli} mapping."""
            op = None
            for s in range(self.n_sites):
                p = site_ops.get(s, identity)
                if op is None:
                    op = p
                else:
                    op = spkron(op, p, format='csr')
            return op.toarray()

        # 1-local terms
        for site in range(self.n_sites):
            for pauli in paulis:
                basis.append(_build_operator({site: pauli}))

        # 2-local terms
        for site_i, site_j in self._build_lattice_edges():
            for pauli_i in paulis:
                for pauli_j in paulis:
                    basis.append(_build_operator({site_i: pauli_i, site_j: pauli_j}))

        # Validate
        edges = self._build_lattice_edges()
        expected_L = 3 * self.n_sites + 9 * len(edges)
        assert len(basis) == expected_L, (
            f"Operator count mismatch: got {len(basis)}, expected {expected_L}")

        return basis

    def sample_submodel(self, k: int) -> QuantumModel:
        """
        Create a submodel by sampling k random Hamiltonians.

        Each sampled Hamiltonian is a random weighted sum of all L basis operators:
        H_i = (1/sqrt(L)) sum_a g_a^(i) P_a, where g_a ~ N(0,1).

        This matches the GEO2 definition from arXiv:2510.06321.
        """
        L = self.basis_size
        if k < 2:
            raise ValueError(f"Need at least 2 operators, got k={k}")

        child_rngs = [
            np.random.default_rng(s) for s in self._seed_seq.spawn(k)
        ]

        ops = []
        for rng in child_rngs:
            coeffs = rng.standard_normal(L) / np.sqrt(L)
            H = sum(c * B for c, B in zip(coeffs, self.basis))
            ops.append(H)

        meta = ModelMetadata(
            parent_basis_size=L,
            description=f"GEO2 submodel: {k} random Hamiltonians "
                        f"(lattice {self.nx}x{self.ny}, d={self.dim})",
        )
        child_seed = self._seed_seq.spawn(1)[0].entropy
        return _SubModel(self.dim, ops, meta, seed=child_seed)


# ============================================================================
# Legacy compatibility layer — re-exported from reach.legacy
# ============================================================================
# These functions are kept here so that `from reach import models; models.fock_state()`
# continues to work for existing scripts.

from .legacy import setup_rng, random_hamiltonian_ensemble, random_states, fock_state
