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



import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad
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
        # Smooth abs: sqrt(|c|^2 + eps) avoids NaN gradient when c=0
        S = jnp.sum(jnp.sqrt(jnp.real(c * jnp.conj(c)) + 1e-30))
        return jnp.clip(S, 0.0, 1.0)

    _spectral_grad_jax = jit(grad(_spectral_score_jax, argnums=0))

    from functools import partial

    @partial(jax.jit, static_argnums=(4,))
    def _krylov_score_jax(lambdas, hams_array, phi, psi, m):
        """JIT-compiled Krylov score with masked breakdown handling.

        Uses jax.lax.scan (fixed iteration count) but tracks column validity.
        After Krylov breakdown (beta < tol), subsequent columns are zeroed
        and masked out of the projection.
        """
        H = jnp.tensordot(lambdas, hams_array, axes=(0, 0))
        d = H.shape[0]

        phi_norm = jnp.linalg.norm(phi)
        v = phi / phi_norm

        V = jnp.zeros((d, m), dtype=jnp.complex128)
        V = V.at[:, 0].set(v)

        # valid_mask tracks which columns are meaningful Krylov vectors
        valid_mask = jnp.zeros(m, dtype=jnp.float64)
        valid_mask = valid_mask.at[0].set(1.0)

        def lanczos_step(carry, j):
            V, beta_prev, v_prev, valid_mask = carry
            is_prev_valid = valid_mask[j]
            v_j = V[:, j] * is_prev_valid
            w = H @ v_j
            alpha = jnp.real(jnp.vdot(v_j, w))
            w = w - alpha * v_j - beta_prev * v_prev * is_prev_valid

            # Safe norm: epsilon inside sqrt prevents NaN gradient
            beta_sq = jnp.real(jnp.vdot(w, w))
            beta = jnp.sqrt(beta_sq + 1e-300)
            is_valid = beta_sq > 1e-28

            v_next = (w / beta) * jnp.where(is_valid & (is_prev_valid > 0.5), 1.0, 0.0)
            V = V.at[:, j + 1].set(v_next)

            col_valid = jnp.where(is_valid & (is_prev_valid > 0.5), 1.0, 0.0)
            valid_mask = valid_mask.at[j + 1].set(col_valid)

            return (V, beta, v_j, valid_mask), None

        (V, _, _, valid_mask), _ = jax.lax.scan(
            lanczos_step,
            (V, 0.0, jnp.zeros(d, dtype=jnp.complex128), valid_mask),
            jnp.arange(m - 1),
        )

        # Project psi onto Krylov basis (no QR — matches NumPy gradient path)
        # QR is not used because autodiff through QR fails with zero columns
        c = V.conj().T @ psi  # (m,)
        c_masked = c * valid_mask
        R = jnp.real(jnp.vdot(c_masked, c_masked))
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
