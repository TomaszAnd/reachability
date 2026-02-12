"""
Legacy compatibility functions (qutip-based interface).

These functions are used by older scripts (confusion_matrix_benchmark.py, audit
scripts, integrability scripts, etc.) that were written before the class-based
API. New code should use CanonicalModel, GeometricTwoLocalModel, and the
criteria classes directly.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import qutip

from . import settings as _settings
from .models import CanonicalModel, GeometricTwoLocalModel


def setup_rng(seed: int = _settings.SEED) -> np.random.RandomState:
    """Create a reproducible random number generator (legacy)."""
    return np.random.RandomState(seed)


def random_hamiltonian_ensemble(
    dim: int, k: int, ensemble: str, seed: Optional[int] = None, **kwargs
) -> List[qutip.Qobj]:
    """
    Generate k random Hamiltonians from specified ensemble (legacy interface).

    Returns qutip.Qobj list for backward compatibility.
    """
    if seed is None:
        seed = _settings.SEED
    rng = setup_rng(seed)

    if ensemble == "canonical":
        include_identity = kwargs.get("include_identity", True)
        model = CanonicalModel(dim, include_identity=include_identity, seed=seed)
        sub = model.sample_submodel(k)
        return [qutip.Qobj(op) for op in sub.basis]

    elif ensemble == "GEO2":
        nx = kwargs.get("nx")
        ny = kwargs.get("ny")
        periodic = kwargs.get("periodic", False)
        if nx is None or ny is None:
            raise ValueError("GEO2 ensemble requires 'nx' and 'ny' parameters")
        model = GeometricTwoLocalModel(dim, nx=nx, ny=ny, periodic=periodic, seed=seed)
        sub = model.sample_submodel(k)
        return [qutip.Qobj(op) for op in sub.basis]

    elif ensemble in ("GOE", "GUE"):
        real_valued = (ensemble == "GOE")
        hamiltonians = []
        for i in range(k):
            h_seed = rng.randint(0, 2**31 - 1)
            h_rng = setup_rng(h_seed)
            A = h_rng.randn(dim, dim)
            if not real_valued:
                A = A + 1j * h_rng.randn(dim, dim)
            H = (A + A.conj().T) / np.sqrt(2)
            hamiltonians.append(qutip.Qobj(H))
        return hamiltonians

    else:
        raise ValueError(f"Unknown ensemble: {ensemble!r}")


def random_states(n: int, dim: int, seed: Optional[int] = None) -> List[qutip.Qobj]:
    """Generate n random quantum states (legacy interface, returns qutip.Qobj)."""
    if seed is None:
        seed = _settings.SEED
    rng = setup_rng(seed)
    states = []
    for i in range(n):
        state_seed = rng.randint(0, 2**31 - 1)
        state = qutip.rand_ket(dim, seed=state_seed)
        states.append(state)
    return states


def fock_state(dim: int, n: int = 0) -> qutip.Qobj:
    """Generate computational basis state |n> (legacy interface)."""
    if not (0 <= n < dim):
        raise ValueError(f"Basis index n={n} must be in range [0, {dim})")
    return qutip.fock(dim, n)
