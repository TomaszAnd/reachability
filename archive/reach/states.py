"""
State generation for stabilizer codes and quantum error correction experiments.

This module provides functions to create initial and target states relevant to
quantum error correction, including GHZ states, W states, cluster states, and
various product states.

All states are returned as numpy arrays representing state vectors in the
computational basis.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import qutip
from scipy.linalg import kron


def pauli_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return single-qubit Pauli matrices and identity.

    Returns:
        Tuple of (I, X, Y, Z) as 2×2 numpy arrays
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def tensor_product(ops: list) -> np.ndarray:
    """
    Compute tensor product of a list of operators or state vectors.

    Args:
        ops: List of numpy arrays (1D for states, 2D for operators)

    Returns:
        Tensor product as a numpy array
    """
    # Check if we're dealing with vectors (1D) or operators (2D)
    is_vector = ops[0].ndim == 1

    if is_vector:
        # For vectors, use np.kron directly
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    else:
        # For operators, use scipy.linalg.kron
        result = ops[0]
        for op in ops[1:]:
            result = kron(result, op)
        return result


def computational_basis(n_qubits: int, bitstring: str) -> np.ndarray:
    """
    Create computational basis state from bitstring.

    Args:
        n_qubits: Number of qubits
        bitstring: Binary string like "0101" (length must match n_qubits)

    Returns:
        State vector of dimension 2^n_qubits

    Example:
        >>> computational_basis(2, "01")  # |01⟩ state
    """
    if len(bitstring) != n_qubits:
        raise ValueError(f"Bitstring length {len(bitstring)} != n_qubits {n_qubits}")

    d = 2**n_qubits
    idx = int(bitstring, 2)
    state = np.zeros(d, dtype=complex)
    state[idx] = 1.0
    return state


def create_initial_states(n_qubits: int = 4) -> Dict[str, np.ndarray]:
    """
    Create dictionary of initial states for experiments.

    Args:
        n_qubits: Number of qubits (default 4 for 2×2 lattice)

    Returns:
        Dictionary mapping state names to state vectors

    States:
        - product_0: |000...0⟩ (all qubits in |0⟩)
        - product_+: |+++...+⟩ (all qubits in (|0⟩+|1⟩)/√2)
        - neel: |0101...⟩ (alternating pattern)
        - domain_wall: |00...11...⟩ (half 0s, half 1s)
    """
    states = {}

    # |000...0⟩
    states['product_0'] = computational_basis(n_qubits, '0' * n_qubits)

    # |+++...+⟩
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    states['product_+'] = tensor_product([plus] * n_qubits)

    # |0101...⟩ (Néel-like pattern)
    pattern = '01' * (n_qubits // 2)
    if n_qubits % 2 == 1:
        pattern += '0'
    states['neel'] = computational_basis(n_qubits, pattern)

    # |00...11...⟩ (domain wall)
    n_half = n_qubits // 2
    pattern = '0' * n_half + '1' * (n_qubits - n_half)
    states['domain_wall'] = computational_basis(n_qubits, pattern)

    return states


def create_target_states(n_qubits: int = 4) -> Dict[str, np.ndarray]:
    """
    Create dictionary of target states relevant to stabilizer codes.

    Args:
        n_qubits: Number of qubits (default 4 for 2×2 lattice)

    Returns:
        Dictionary mapping state names to state vectors

    States:
        - ghz: (|00...0⟩ + |11...1⟩)/√2 (4-qubit code logical |0_L⟩)
        - ghz_minus: (|00...0⟩ - |11...1⟩)/√2 (4-qubit code logical |1_L⟩)
        - w_state: (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n
        - cluster: Linear cluster state (CZ gates applied to |+++...+⟩)
    """
    states = {}
    d = 2**n_qubits

    # GHZ state: (|00...0⟩ + |11...1⟩)/√2
    state_0 = computational_basis(n_qubits, '0' * n_qubits)
    state_1 = computational_basis(n_qubits, '1' * n_qubits)
    states['ghz'] = (state_0 + state_1) / np.sqrt(2)
    states['ghz_minus'] = (state_0 - state_1) / np.sqrt(2)

    # W state: (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n
    w = np.zeros(d, dtype=complex)
    for i in range(n_qubits):
        # Create bitstring with single 1 at position i
        bitstring = '0' * i + '1' + '0' * (n_qubits - i - 1)
        idx = int(bitstring, 2)
        w[idx] = 1.0
    states['w_state'] = w / np.linalg.norm(w)

    # Cluster state: CZ_{i,i+1} gates on |+++...+⟩
    # CZ on computational basis: |ij⟩ → (-1)^{i·j} |ij⟩
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    cluster = tensor_product([plus] * n_qubits)

    # Apply CZ gates between neighboring qubits
    for i in range(n_qubits - 1):
        # CZ gate on qubits i and i+1
        CZ = np.eye(d, dtype=complex)
        for j in range(d):
            bits = format(j, f'0{n_qubits}b')
            # CZ flips sign if both qubits are |1⟩
            if bits[i] == '1' and bits[i+1] == '1':
                CZ[j, j] = -1
        cluster = CZ @ cluster
    states['cluster'] = cluster

    return states


def create_stabilizer_eigenstate(n_qubits: int = 4) -> np.ndarray:
    """
    Create +1 eigenstate of stabilizer generators for 4-qubit code.

    For a 4-qubit code with stabilizers S1 = XXXX and S2 = ZZZZ,
    the +1 eigenstate is the GHZ state: (|0000⟩ + |1111⟩)/√2

    Args:
        n_qubits: Number of qubits (only 4 currently supported)

    Returns:
        State vector in the +1 eigenspace of both stabilizers
    """
    if n_qubits != 4:
        raise NotImplementedError("Only 4-qubit stabilizer code implemented")

    # For XXXX and ZZZZ stabilizers, the code space is spanned by GHZ states
    states = create_target_states(n_qubits)
    return states['ghz']


def random_state(dim: int, seed: int = None) -> np.ndarray:
    """
    Generate a random normalized state vector using Haar measure.

    Args:
        dim: Hilbert space dimension
        seed: Random seed for reproducibility

    Returns:
        Random state vector of dimension dim, normalized
    """
    rng = np.random.RandomState(seed)
    # Ginibre ensemble: complex Gaussian entries
    state = rng.randn(dim) + 1j * rng.randn(dim)
    return state / np.linalg.norm(state)


def state_to_qutip(state: np.ndarray) -> qutip.Qobj:
    """
    Convert numpy state vector to QuTiP Qobj.

    Args:
        state: Numpy array representing state vector

    Returns:
        QuTiP Qobj with appropriate dimensions
    """
    d = len(state)
    return qutip.Qobj(state, dims=[[d], [1]])


def qutip_to_state(qobj: qutip.Qobj) -> np.ndarray:
    """
    Convert QuTiP Qobj to numpy state vector.

    Args:
        qobj: QuTiP quantum object

    Returns:
        Numpy array representing state vector
    """
    return qobj.full().flatten()


# Convenience function to get all states at once
def get_all_states(n_qubits: int = 4) -> Dict[str, np.ndarray]:
    """
    Get both initial and target states in a single dictionary.

    Args:
        n_qubits: Number of qubits

    Returns:
        Dictionary with all initial and target states
    """
    states = {}
    states.update(create_initial_states(n_qubits))
    states.update(create_target_states(n_qubits))
    return states