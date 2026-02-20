"""
JAX-accelerated criterion implementations for GPU execution.

These are drop-in replacements for SpectralCriterion and KrylovCriterion
that use JAX for JIT compilation and GPU acceleration.

Usage:
    import os
    os.environ['REACHABILITY_BACKEND'] = 'jax'
    from src.criteria_jax import SpectralCriterionJAX, KrylovCriterionJAX

Requires: jax, jaxlib
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.debug("JAX not available, criteria_jax will not function")

if JAX_AVAILABLE:
    jax.config.update("jax_enable_x64", True)

    from .criteria import (
        OptimizableCriterion, ReachabilityResult, Verdict,
        DEFAULT_BOUNDS, DEFAULT_METHOD, DEFAULT_RESTARTS, DEFAULT_MAXITER, DEFAULT_FTOL,
    )
    from .math_utils import clip_to_bounds, KRYLOV_BREAKDOWN_TOL

    @jit
    def _spectral_score_jax(lambdas, hams_array, phi, psi):
        """JIT-compiled spectral score."""
        H = jnp.tensordot(lambdas, hams_array, axes=(0, 0))
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        phi_coeffs = eigenvectors.conj().T @ phi
        psi_coeffs = eigenvectors.conj().T @ psi
        c = psi_coeffs.conj() * phi_coeffs
        S = jnp.sum(jnp.abs(c))
        return jnp.clip(S, 0.0, 1.0)

    _spectral_grad_jax = jit(grad(_spectral_score_jax, argnums=0))

    from functools import partial

    @partial(jax.jit, static_argnums=(4,))
    def _krylov_score_jax(lambdas, hams_array, phi, psi, m):
        """JIT-compiled Krylov score using Lanczos."""
        H = jnp.tensordot(lambdas, hams_array, axes=(0, 0))
        d = H.shape[0]

        phi_norm = jnp.linalg.norm(phi)
        v = phi / phi_norm

        # Fixed-size Lanczos (no early termination for JIT compatibility)
        V = jnp.zeros((d, m), dtype=jnp.complex128)
        V = V.at[:, 0].set(v)
        beta_prev = 0.0
        v_prev = jnp.zeros(d, dtype=jnp.complex128)

        def lanczos_step(carry, j):
            V, beta_prev, v_prev = carry
            v_j = V[:, j]
            w = H @ v_j
            alpha = jnp.real(jnp.vdot(v_j, w))
            w = w - alpha * v_j - beta_prev * v_prev
            beta = jnp.linalg.norm(w)
            # Use safe division (beta=0 means breakdown but we continue with zeros)
            v_next = jnp.where(beta > 1e-14, w / beta, jnp.zeros_like(w))
            V = V.at[:, j + 1].set(v_next)
            return (V, beta, v_j), None

        (V, _, _), _ = jax.lax.scan(lanczos_step, (V, 0.0, jnp.zeros(d, dtype=jnp.complex128)),
                                      jnp.arange(m - 1))

        # QR for numerical stability
        Q, _ = jnp.linalg.qr(V)
        c = Q.conj().T @ psi
        R = jnp.real(jnp.vdot(c, c))
        return jnp.clip(R, 0.0, 1.0)

    @partial(jax.jit, static_argnums=(4,))
    def _krylov_grad_jax(lambdas, hams_array, phi, psi, m):
        return grad(_krylov_score_jax, argnums=0)(lambdas, hams_array, phi, psi, m)

    class SpectralCriterionJAX(OptimizableCriterion):
        """JAX-accelerated Spectral criterion for GPU execution."""

        def __init__(self, model_or_hams, phi, psi, tau=0.99):
            super().__init__(model_or_hams, phi, psi, tau)
            # Convert to JAX arrays
            self._hams_jax = jnp.array(self._hams_array)
            self._phi_jax = jnp.array(self.phi)
            self._psi_jax = jnp.array(self.psi)

        def evaluate(self, lambdas, return_gradient=False):
            lambdas_jax = jnp.array(lambdas)

            S = float(_spectral_score_jax(lambdas_jax, self._hams_jax,
                                           self._phi_jax, self._psi_jax))
            if not return_gradient:
                return S

            grad_val = np.array(_spectral_grad_jax(lambdas_jax, self._hams_jax,
                                                    self._phi_jax, self._psi_jax))
            return S, grad_val

    class KrylovCriterionJAX(OptimizableCriterion):
        """JAX-accelerated Krylov criterion for GPU execution."""

        def __init__(self, model_or_hams, phi, psi, tau=0.99, m=None):
            super().__init__(model_or_hams, phi, psi, tau)
            self.m = m if m is not None else self.dim
            self._hams_jax = jnp.array(self._hams_array)
            self._phi_jax = jnp.array(self.phi)
            self._psi_jax = jnp.array(self.psi)

        def evaluate(self, lambdas, return_gradient=False):
            lambdas_jax = jnp.array(lambdas)
            m = min(self.m, self.dim)

            R = float(_krylov_score_jax(lambdas_jax, self._hams_jax,
                                         self._phi_jax, self._psi_jax, m))
            if not return_gradient:
                return R

            grad_val = np.array(_krylov_grad_jax(lambdas_jax, self._hams_jax,
                                                  self._phi_jax, self._psi_jax, m))
            return R, grad_val
