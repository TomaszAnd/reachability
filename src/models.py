"""
Hamiltonian families for quantum reachability analysis.

A Hamiltonian family defines an operator basis {B_1, ..., B_L}.
We select K operators to form a control family {H_1, ..., H_K}.
The combined Hamiltonian is H(lambda) = sum_k lambda_k H_k, where lambda in [-1, 1]^K.

Two families are supported:
- CanonicalQuditModel: generalized Pauli basis {X_jk, Y_jk, Z_j, I} for arbitrary dimension d
- QubitGridModel: Pauli chains on a rectangular qubit grid

Design principles:
- K is the primary parameter (not rho = K/d^2)
- sample_submodel(K) returns a new QuantumModel, not a list
- numpy arrays throughout (no qutip.Qobj in interfaces)
- np.random.SeedSequence with .spawn() for parallel-safe RNG
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        self._basis_sparse: Optional[List[csr_matrix]] = None
        self._metadata = ModelMetadata()

    @abstractmethod
    def _build_basis(self) -> List[np.ndarray]:
        """Build the complete operator basis as numpy arrays."""
        pass

    def _build_basis_sparse(self) -> Optional[List[csr_matrix]]:
        """Build sparse operator basis. Override in subclasses with native sparse support."""
        return None

    @property
    def basis(self) -> List[np.ndarray]:
        """Lazily-built operator basis as list of (d, d) numpy arrays."""
        if self._basis is None:
            # Try sparse first, convert to dense
            if self._basis_sparse is not None:
                self._basis = [op.toarray() for op in self._basis_sparse]
            else:
                sparse = self._build_basis_sparse()
                if sparse is not None:
                    self._basis_sparse = sparse
                    self._basis = [op.toarray() for op in sparse]
                    self._metadata.parent_basis_size = len(self._basis)
                else:
                    self._basis = self._build_basis()
                    self._metadata.parent_basis_size = len(self._basis)
        return self._basis

    @property
    def basis_sparse(self) -> Optional[List[csr_matrix]]:
        """Sparse operator basis, if available. None for dense-only models."""
        if self._basis_sparse is None:
            self._basis_sparse = self._build_basis_sparse()
            if self._basis_sparse is not None:
                self._metadata.parent_basis_size = len(self._basis_sparse)
        return self._basis_sparse

    @property
    def has_sparse(self) -> bool:
        """Whether this model has a sparse basis representation."""
        return self.basis_sparse is not None

    @property
    def K(self) -> int:
        """Number of control operators in this model."""
        if self._basis is not None:
            return len(self._basis)
        if self._basis_sparse is not None:
            return len(self._basis_sparse)
        return len(self.basis)

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @abstractmethod
    def sample_submodel(self, k: int, seed: Optional[int] = None) -> 'QuantumModel':
        """
        Create a new QuantumModel with k operators sampled from this basis.

        Args:
            k: Number of operators to select
            seed: If provided, create a fresh model with this seed for
                  reproducible sampling. If None, use internal RNG.

        Returns:
            A new QuantumModel with k operators as its basis
        """
        pass

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
            family_type: 'canonical' or 'qubit_grid' (alias: 'geometric_local')
            dim: Hilbert space dimension
            **kwargs: Passed to the specific model constructor

        Returns:
            QuantumModel instance
        """
        ft = family_type.lower()
        if ft == "canonical":
            return CanonicalQuditModel(dim, **kwargs)
        elif ft in ("geometric_local", "qubit_grid"):
            return QubitGridModel(dim, **kwargs)
        raise ValueError(f"Unknown family type: {family_type!r}")


class _SubModel(QuantumModel):
    """A submodel created by selecting operators from a parent model."""

    def __init__(self, dim: int, operators: List[np.ndarray],
                 metadata: ModelMetadata, seed: Optional[int] = None,
                 operators_sparse: Optional[List[csr_matrix]] = None):
        super().__init__(dim, seed)
        self._basis = operators
        self._basis_sparse = operators_sparse
        self._metadata = metadata

    def _build_basis(self) -> List[np.ndarray]:
        return self._basis

    def _build_basis_sparse(self) -> Optional[List[csr_matrix]]:
        return self._basis_sparse

    def sample_submodel(self, k: int, seed: Optional[int] = None) -> 'QuantumModel':
        if k > self.K:
            raise ValueError(
                f"Cannot sample {k} operators from submodel with {self.K} operators")
        if k < 2:
            raise ValueError(f"Need at least 2 operators, got k={k}")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        indices = rng.choice(self.K, size=k, replace=False)
        ops = [self._basis[i] for i in indices] if self._basis is not None else None
        ops_sparse = ([self._basis_sparse[i] for i in indices]
                      if self._basis_sparse is not None else None)
        meta = ModelMetadata(
            parent_basis_size=self.K,
            selected_indices=indices,
            description=f"Sub-submodel: {k} of {self.K} operators",
        )
        child_seed = int(self._seed_seq.spawn(1)[0].generate_state(1)[0])
        return _SubModel(self.dim, ops, meta, seed=child_seed,
                         operators_sparse=ops_sparse)


class CanonicalQuditModel(QuantumModel):
    """
    Canonical basis for d x d Hermitian matrices.

    Basis: {X_jk, Y_jk, Z_j, I}, total d^2 operators.
    - X_jk = |j><k| + |k><j| for j < k
    - Y_jk = -i(|j><k| - |k><j|) for j < k
    - Z_j = |j><j| - |j+1><j+1| for j < d-1
    - I = identity (optional)

    Memory warning: At d=256, the full basis contains d^2=65536 dense
    (256x256) complex128 matrices, requiring ~68GB RAM. This is infeasible
    on machines with <=16GB. Use QubitGridModel for d>=256 experiments.
    """

    def __init__(self, dim: int, include_identity: bool = True,
                 seed: Optional[int] = None, K_max: Optional[int] = None):
        self.include_identity = include_identity
        self.K_max = K_max
        super().__init__(dim, seed)

    @staticmethod
    def _build_operator_by_index(d: int, idx: int) -> np.ndarray:
        """Build a single Gell-Mann basis operator by its global index.

        Index layout over d^2 operators:
          [0, n_off):           X_jk = |j><k| + |k><j|   (n_off = d(d-1)/2)
          [n_off, 2*n_off):     Y_jk = -i|j><k| + i|k><j|
          [2*n_off, 2*n_off+d-1): Z_j = |j><j| - |j+1><j+1|
          [d^2-1]:              Identity
        """
        n_off = d * (d - 1) // 2
        mat = np.zeros((d, d), dtype=np.complex128)

        if idx < n_off:
            # X_jk: map linear index to (j, k) pair with j < k
            remaining = idx
            for j in range(d - 1):
                count_j = d - 1 - j
                if remaining < count_j:
                    k = j + 1 + remaining
                    mat[j, k] = 1.0
                    mat[k, j] = 1.0
                    return mat
                remaining -= count_j
        elif idx < 2 * n_off:
            # Y_jk
            remaining = idx - n_off
            for j in range(d - 1):
                count_j = d - 1 - j
                if remaining < count_j:
                    k = j + 1 + remaining
                    mat[j, k] = -1j
                    mat[k, j] = 1j
                    return mat
                remaining -= count_j
        elif idx < 2 * n_off + d - 1:
            # Z_j
            j = idx - 2 * n_off
            mat[j, j] = 1.0
            mat[j + 1, j + 1] = -1.0
            return mat
        else:
            # Identity
            return np.eye(d, dtype=np.complex128)

        return mat  # Should not reach here

    def _build_basis(self) -> List[np.ndarray]:
        d = self.dim
        total_L = d * d if self.include_identity else d * d - 1

        if self.K_max is not None and self.K_max < total_L:
            # Random sampling: pick K_max indices uniformly from all d^2
            # operator types (X, Y, Z, I). This ensures a balanced mix
            # across ALL row/column indices, avoiding the structural bias
            # of sequential truncation (which produces only X-type operators
            # concentrated in the first few rows).
            indices = self._rng.choice(total_L, size=self.K_max, replace=False)
            return [self._build_operator_by_index(d, int(i)) for i in indices]

        # Full basis: build all operators sequentially
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

        assert len(operators) == total_L, (
            f"Operator count mismatch: got {len(operators)}, expected {total_L}")

        return operators

    def sample_submodel(self, k: int, seed: Optional[int] = None) -> QuantumModel:
        """Sample k operators from the canonical basis without replacement."""
        if seed is not None:
            fresh = CanonicalQuditModel(
                self.dim, include_identity=self.include_identity, seed=seed,
                K_max=self.K_max)
            return fresh.sample_submodel(k)

        L = self.K
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
        child_seed = int(self._seed_seq.spawn(1)[0].generate_state(1)[0])
        return _SubModel(self.dim, ops, meta, seed=child_seed)


# Default lattice configs: dimension -> (nx, ny)
LATTICE_CONFIGS = {
    8: (1, 3),
    16: (2, 2),
    32: (1, 5),
    64: (2, 3),
    128: (1, 7),   # 7 qubits
    256: (2, 4),   # 8 qubits
}


class QubitGridModel(QuantumModel):
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
            if dim in LATTICE_CONFIGS:
                nx, ny = LATTICE_CONFIGS[dim]
            else:
                raise ValueError(
                    f"Cannot infer lattice for d={dim}. "
                    f"Provide nx, ny explicitly. Known dims: {list(LATTICE_CONFIGS.keys())}")

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

    def _build_basis_sparse(self) -> List[csr_matrix]:
        """Build Pauli basis P_2(G) as sparse matrices (native format)."""
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
            return op

        # Cache edges to avoid redundant computation
        edges = self._build_lattice_edges()

        # 1-local terms
        for site in range(self.n_sites):
            for pauli in paulis:
                basis.append(_build_operator({site: pauli}))

        # 2-local terms
        for site_i, site_j in edges:
            for pauli_i in paulis:
                for pauli_j in paulis:
                    basis.append(_build_operator({site_i: pauli_i, site_j: pauli_j}))

        # Validate
        expected_L = 3 * self.n_sites + 9 * len(edges)
        assert len(basis) == expected_L, (
            f"Operator count mismatch: got {len(basis)}, expected {expected_L}")

        return basis

    def _build_basis(self) -> List[np.ndarray]:
        """Build dense Pauli basis (converts from sparse)."""
        sparse = self._build_basis_sparse()
        self._basis_sparse = sparse
        return [op.toarray() for op in sparse]

    def sample_submodel(self, k: int, seed: Optional[int] = None) -> QuantumModel:
        """
        Sample k operators from the Pauli basis P_2(G) without replacement.

        Each H_k is a single Pauli operator (1-local or 2-local term),
        matching the paper Sec. IV.B approach.
        """
        if seed is not None:
            fresh = QubitGridModel(
                self.dim, nx=self.nx, ny=self.ny,
                periodic=self.periodic, seed=seed)
            return fresh.sample_submodel(k)

        L = self.K
        if k > L:
            raise ValueError(
                f"Cannot sample {k} operators from QubitGrid basis of size {L}")
        if k < 2:
            raise ValueError(f"Need at least 2 operators, got k={k}")

        indices = self._rng.choice(L, size=k, replace=False)

        # Select sparse operators (primary); dense built lazily when needed
        sparse_basis = self.basis_sparse
        ops_sparse = [sparse_basis[i] for i in indices]
        ops = None  # Dense basis built lazily by _SubModel.basis property

        meta = ModelMetadata(
            parent_basis_size=L,
            selected_indices=indices,
            description=f"QubitGrid submodel: {k} of {L} operators "
                        f"(lattice {self.nx}x{self.ny}, d={self.dim})",
        )
        child_seed = int(self._seed_seq.spawn(1)[0].generate_state(1)[0])
        return _SubModel(self.dim, ops, meta, seed=child_seed,
                         operators_sparse=ops_sparse)
