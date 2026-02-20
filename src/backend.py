"""
Backend abstraction for numpy vs JAX.

Default: numpy (CPU). Set REACHABILITY_BACKEND=jax for GPU acceleration.

Usage:
    # Default CPU
    from src.backend import xp, eigh, jit_if_available

    # Force JAX
    import os
    os.environ['REACHABILITY_BACKEND'] = 'jax'
    from src.backend import xp, eigh, jit_if_available
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# Backend selection
_BACKEND_NAME = os.environ.get('REACHABILITY_BACKEND', 'numpy').lower()
_JAX_AVAILABLE = False

if _BACKEND_NAME == 'jax':
    try:
        import jax
        import jax.numpy as jnp
        from jax.numpy.linalg import eigh as jax_eigh
        _JAX_AVAILABLE = True
        logger.info("JAX backend active")
    except ImportError:
        logger.warning("JAX requested but not installed, falling back to numpy")
        _BACKEND_NAME = 'numpy'


def get_backend() -> str:
    """Return current backend name: 'numpy' or 'jax'."""
    return _BACKEND_NAME


def is_jax() -> bool:
    """Whether JAX backend is active."""
    return _BACKEND_NAME == 'jax' and _JAX_AVAILABLE


# Export appropriate array module
if is_jax():
    import jax.numpy as xp
    from jax.numpy.linalg import eigh
    from jax import jit as _jit

    def jit_if_available(f):
        """Apply JIT compilation if JAX is available."""
        return _jit(f)
else:
    import numpy as xp
    from numpy.linalg import eigh

    def jit_if_available(f):
        """No-op decorator when JAX is not available."""
        return f
