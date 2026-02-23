"""
Reachability criteria for quantum control systems.

Three criteria for testing whether target state |psi> is reachable
from initial state |phi> under H(lambda) = sum_k lambda_k H_k:

- SpectralCriterion: Spectral overlap S(lambda) = sum_n |<u_n|psi>* <u_n|phi>|
- KrylovCriterion: Krylov projection R(lambda) = ||P_K |psi>||^2
- MomentCriterion: Positive definiteness of moment matrix Q + x L L^T

Class hierarchy:
    ReachabilityCriterion (ABC)
    ├── OptimizableCriterion (ABC) — multi-restart L-BFGS-B optimizer
    │   ├── SpectralCriterion
    │   └── KrylovCriterion
    └── MomentCriterion — no optimization, checks γ=±1000

Verdict system:
- SpectralCriterion and KrylovCriterion return REACHABLE or UNREACHABLE
- MomentCriterion returns UNREACHABLE or INCONCLUSIVE (never REACHABLE)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import issparse, csr_matrix, coo_matrix

from .math_utils import (
    KRYLOV_BREAKDOWN_TOL,
    eigendecompose, clip_to_bounds,
)

logger = logging.getLogger(__name__)


class Verdict(Enum):
    """Three-valued reachability verdict."""
    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ReachabilityResult:
    """Result of a reachability test."""
    verdict: Verdict
    score: float
    optimal_params: Optional[np.ndarray] = None
    certificate: Optional[str] = None
    raw_data: Optional[dict] = field(default=None, repr=False)


# Default optimization parameters
DEFAULT_BOUNDS = [(-1.0, 1.0)]
DEFAULT_METHOD = "L-BFGS-B"
DEFAULT_RESTARTS = 2
DEFAULT_MAXITER = 200
DEFAULT_FTOL = 1e-6


# ============================================================================
# Abstract base classes
# ============================================================================

class ReachabilityCriterion(ABC):
    """
    Abstract base class for reachability criteria.

    Accepts either a QuantumModel or a list of Hamiltonian numpy arrays.
    """

    def __init__(
        self,
        model_or_hams,
        phi: np.ndarray,
        psi: np.ndarray,
        tau: float = 0.99,
    ):
        from .models import QuantumModel
        if isinstance(model_or_hams, QuantumModel):
            self.model = model_or_hams
            self.hams_sparse = model_or_hams.basis_sparse
            # Avoid building dense basis when sparse is available (saves ~119MB at d=256)
            if self.hams_sparse is not None:
                self.hams = None  # Dense basis built lazily via _hams_array
                self.K = len(self.hams_sparse)
                self.dim = self.hams_sparse[0].shape[0]
            else:
                self.hams = model_or_hams.basis
                self.K = len(self.hams)
                self.dim = self.hams[0].shape[0]
        else:
            self.model = None
            # Check if list of sparse matrices was passed
            if model_or_hams and issparse(model_or_hams[0]):
                self.hams_sparse = model_or_hams
                self.hams = None
                self.K = len(self.hams_sparse)
                self.dim = self.hams_sparse[0].shape[0]
            else:
                self.hams_sparse = None
                self.hams = model_or_hams
                self.K = len(self.hams)
                self.dim = self.hams[0].shape[0]
        self.phi = phi.flatten()
        self.psi = psi.flatten()
        self.tau = tau
        self._hams_array_cache = None  # Lazy; built on first access
        self._use_sparse = self.hams_sparse is not None
        # Pre-extract COO data for fast H(lambda) assembly (9x vs CSR loop)
        if self._use_sparse:
            rows_l, cols_l, data_l, nnz_l = [], [], [], []
            for op in self.hams_sparse:
                c = op.tocoo()
                rows_l.append(c.row)
                cols_l.append(c.col)
                data_l.append(c.data)
                nnz_l.append(c.nnz)
            self._coo_rows = np.concatenate(rows_l)
            self._coo_cols = np.concatenate(cols_l)
            self._coo_data = np.concatenate(data_l)
            self._coo_nnz = np.array(nnz_l)
        else:
            self._coo_rows = None
            self._coo_cols = None
            self._coo_data = None
            self._coo_nnz = None

    @property
    def _hams_array(self) -> np.ndarray:
        """Lazily-constructed stacked Hamiltonians array (K, d, d)."""
        if self._hams_array_cache is None:
            if self.hams is not None:
                self._hams_array_cache = np.stack(self.hams)
            elif self.hams_sparse is not None:
                self._hams_array_cache = np.stack(
                    [op.toarray() for op in self.hams_sparse])
            else:
                raise RuntimeError("No Hamiltonian operators available")
        return self._hams_array_cache

    def _construct_H_sparse(self, lambdas: np.ndarray):
        """Build H(lambda) using COO assembly (5-12x faster than CSR loop)."""
        if self._use_sparse:
            scaled = np.repeat(lambdas, self._coo_nnz) * self._coo_data
            return coo_matrix(
                (scaled, (self._coo_rows, self._coo_cols)),
                shape=(self.dim, self.dim),
            ).tocsr()
        return np.tensordot(lambdas, self._hams_array, axes=(0, 0))

    def _construct_H_dense(self, lambdas: np.ndarray) -> np.ndarray:
        """Build H(lambda) as dense array."""
        return np.tensordot(lambdas, self._hams_array, axes=(0, 0))

    def clear_cache(self) -> None:
        """Clear cached arrays to free memory."""
        self._hams_array_cache = None

    @abstractmethod
    def evaluate(
        self, lambdas: np.ndarray, return_gradient: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Evaluate the criterion score at given parameters.

        Args:
            lambdas: Parameter vector (K,)
            return_gradient: If True, also return gradient

        Returns:
            score (float) or (score, gradient) if return_gradient=True
        """
        pass

    @abstractmethod
    def is_reachable(self, **kwargs) -> ReachabilityResult:
        """
        Test reachability.

        Returns:
            ReachabilityResult with verdict, score, and metadata
        """
        pass


class OptimizableCriterion(ReachabilityCriterion):
    """
    Criterion that uses multi-restart optimization to find the maximum score.

    Shared by SpectralCriterion and KrylovCriterion. MomentCriterion does
    NOT inherit from this class since it uses grid search instead.
    """

    def _single_restart(self, x0, method, maxiter, ftol, bounds, use_grad):
        """Run a single optimization restart. Thread-safe."""
        def objective(x):
            if use_grad:
                val, grad = self.evaluate(x, return_gradient=True)
                return -float(val), -grad.astype(np.float64)
            else:
                return -float(self.evaluate(x))

        try:
            options = {"maxiter": maxiter, "ftol": ftol}
            if use_grad:
                result = minimize(objective, x0, method=method,
                                  jac=True, bounds=bounds, options=options)
            else:
                result = minimize(objective, x0, method=method,
                                  bounds=bounds, options=options)
            return -result.fun, result.x.copy(), result.nfev
        except Exception as e:
            logger.debug(f"Optimization restart failed: {e}")
            return 0.0, x0, 0

    def _maximize(
        self,
        method: str = DEFAULT_METHOD,
        restarts: int = DEFAULT_RESTARTS,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        seed: Optional[int] = None,
        parallel: bool = False,
    ) -> Dict:
        """
        Multi-restart optimization to maximize the criterion score.

        Generates `restarts` random starting points in [-1,1]^K and runs
        L-BFGS-B (with analytical gradient) from each. Returns the best
        result across all restarts.

        Args:
            parallel: If True, run restarts in parallel threads.
                      Benefits mainly at large d where BLAS releases the GIL.
        """
        K = self.K
        bounds = DEFAULT_BOUNDS * K
        rng = np.random.RandomState(seed or 42)
        use_grad = (method == "L-BFGS-B")

        # Generate all starting points upfront
        x0_list = [np.array([rng.uniform(lo, hi) for lo, hi in bounds])
                    for _ in range(restarts)]

        start_time = time.time()

        if parallel and restarts > 1:
            # Parallel restarts via ThreadPoolExecutor
            # NumPy/BLAS releases GIL during matrix operations
            best_value = 0.0
            best_x = np.zeros(K)
            total_nfev = 0

            with ThreadPoolExecutor(max_workers=min(restarts, 4)) as executor:
                futures = {
                    executor.submit(self._single_restart, x0, method,
                                    maxiter, ftol, bounds, use_grad): i
                    for i, x0 in enumerate(x0_list)
                }
                for future in as_completed(futures):
                    value, x, nfev = future.result()
                    total_nfev += nfev
                    if value > best_value:
                        best_value = value
                        best_x = x
        else:
            # Sequential restarts (with early termination)
            best_value = 0.0
            best_x = np.zeros(K)
            total_nfev = 0

            for x0 in x0_list:
                value, x, nfev = self._single_restart(
                    x0, method, maxiter, ftol, bounds, use_grad)
                total_nfev += nfev
                if value > best_value:
                    best_value = value
                    best_x = x
                if best_value >= self.tau:
                    break  # Already reachable

        best_x = clip_to_bounds(best_x, bounds)
        runtime = time.time() - start_time

        return {
            "best_value": float(best_value),
            "best_x": best_x,
            "nfev": total_nfev,
            "runtime_s": runtime,
        }

    def is_reachable(
        self,
        method: str = DEFAULT_METHOD,
        restarts: int = DEFAULT_RESTARTS,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        seed: Optional[int] = None,
        parallel: bool = False,
    ) -> ReachabilityResult:
        """Maximize the criterion score and compare to tau threshold."""
        result = self._maximize(
            method=method, restarts=restarts,
            maxiter=maxiter, ftol=ftol, seed=seed,
            parallel=parallel,
        )
        score = result["best_value"]
        verdict = Verdict.REACHABLE if score >= self.tau else Verdict.UNREACHABLE

        return ReachabilityResult(
            verdict=verdict,
            score=score,
            optimal_params=result["best_x"],
            raw_data=result,
        )


# ============================================================================
# Spectral criterion
# ============================================================================

class SpectralCriterion(OptimizableCriterion):
    """
    Spectral overlap criterion.

    S(lambda) = sum_n |<u_n(lambda)|psi>* <u_n(lambda)|phi>|

    where |u_n(lambda)> are eigenstates of H(lambda).

    Verdict:
    - REACHABLE if max_lambda S(lambda) >= tau
    - UNREACHABLE if max_lambda S(lambda) < tau
    """

    def evaluate(
        self, lambdas: np.ndarray, return_gradient: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute spectral overlap S(lambda) and optionally its gradient.

        Algorithm:
            1. Build H(lambda) = sum_k lambda_k H_k
            2. Eigendecompose: H|u_n> = E_n|u_n>
            3. Project states: <u_n|phi> and <u_n|psi>
            4. Overlap coefficients: c_n = <u_n|psi>* <u_n|phi>
            5. Sum: S = sum_n |c_n|
            6. (If gradient) Perturbation theory for dS/dlambda_k
        """
        # Step 1: Build combined Hamiltonian
        if self._use_sparse:
            H_sp = self._construct_H_sparse(lambdas)
            H = H_sp.toarray() if issparse(H_sp) else H_sp
        else:
            H = np.tensordot(lambdas, self._hams_array, axes=(0, 0))

        # Step 2: Eigendecomposition
        try:
            eigenvalues, eigenvectors = eigendecompose(H)
        except RuntimeError:
            if return_gradient:
                return 0.0, np.zeros(self.K)
            return 0.0

        # Step 3: Project states onto eigenbasis
        phi_coeffs = eigenvectors.conj().T @ self.phi  # <u_n|phi>
        psi_coeffs = eigenvectors.conj().T @ self.psi  # <u_n|psi>

        # Step 4: Overlap coefficients
        c = psi_coeffs.conj() * phi_coeffs  # c_n = <u_n|psi>* <u_n|phi>
        abs_c = np.abs(c)

        # Step 5: Spectral overlap
        S = float(np.sum(abs_c))
        S = np.clip(S, 0.0, 1.0)

        if not return_gradient:
            return S

        # Step 6: Analytical gradient via first-order perturbation theory
        # Phase factors: sign(c_n) = c_n / |c_n|
        safe_abs_c = np.where(abs_c > 1e-15, abs_c, 1.0)
        sign_c = np.where(abs_c > 1e-15, c / safe_abs_c, 0.0)

        # Regularized inverse energy differences for degenerate eigenvalues
        E = eigenvalues
        delta_E = E[:, None] - E[None, :]
        eps_reg = 1e-12
        inv_delta_E = delta_E / (delta_E**2 + eps_reg)

        # dS/dlambda_k = sum_n Re[sign(c_n)* dc_n/dlambda_k]
        U = eigenvectors
        sign_c_conj = sign_c.conj()

        # Vectorized gradient: compute all U^T H_k U at once
        # Hk_eig_all has shape (K, d, d)
        Uc = U.conj().T  # (d, d)
        if self._use_sparse:
            HkU_all = np.empty((self.K, self.dim, self.dim), dtype=np.complex128)
            for k in range(self.K):
                r = self.hams_sparse[k] @ U
                HkU_all[k] = r.toarray() if issparse(r) else r
        else:
            HkU_all = self._hams_array @ U  # (K, d, d)
        # Batched U^T @ HkU: (d,d) @ (K,d,d) via broadcasting = (K,d,d)
        Hk_eig_all = Uc @ HkU_all

        # dphi[k,n] = sum_m Hk_eig[k,n,m] * inv_delta_E[n,m] * phi_coeffs[m]
        weighted_inv_phi = inv_delta_E * phi_coeffs[None, :]  # (d, d)
        weighted_inv_psi = inv_delta_E * psi_coeffs[None, :]  # (d, d)
        dphi_all = np.einsum('knm,nm->kn', Hk_eig_all, weighted_inv_phi)
        dpsi_all = np.einsum('knm,nm->kn', Hk_eig_all, weighted_inv_psi)

        dc_all = dpsi_all.conj() * phi_coeffs[None, :] + psi_coeffs.conj()[None, :] * dphi_all
        grad = np.real(np.einsum('n,kn->k', sign_c_conj, dc_all))

        return S, grad


# ============================================================================
# Krylov criterion
# ============================================================================

class KrylovCriterion(OptimizableCriterion):
    """
    Krylov subspace projection criterion.

    R(lambda) = ||P_Km(H(lambda)) |psi>||^2

    where P_Km is the projection onto Krylov subspace K_m(H(lambda), phi).
    Uses Lanczos iteration (3-term recurrence, Hermitian-optimal).

    Args:
        m: Krylov subspace dimension. Default is d (full Hilbert space dimension).
           Smaller values give weaker bounds and may produce trivial results.
    """

    def __init__(
        self,
        hams: List[np.ndarray],
        phi: np.ndarray,
        psi: np.ndarray,
        tau: float = 0.99,
        m: Optional[int] = None,
    ):
        super().__init__(hams, phi, psi, tau)
        self.m = m if m is not None else self.dim

    # -- Internal: Lanczos iteration --

    def _lanczos_basis(self, H) -> np.ndarray:
        """
        Build orthonormal Krylov basis K_m(H, phi) via Lanczos iteration.

        For Hermitian H, Lanczos uses a 3-term recurrence instead of full
        Gram-Schmidt orthogonalization (Arnoldi). This gives equivalent results
        with O(d) work per step instead of O(d*j).

        H can be dense (ndarray) or sparse (csr_matrix). When sparse,
        the matvec H @ v is O(nnz) instead of O(d^2).

        Returns (d, m_actual) matrix with orthonormal columns.
        m_actual <= m due to possible Krylov breakdown.
        Final QR ensures orthogonality despite finite-precision arithmetic.
        """
        d = self.dim
        m = min(self.m, d)
        phi_norm = np.linalg.norm(self.phi)
        if phi_norm == 0:
            return np.zeros((d, 0), dtype=np.complex128)

        V = np.zeros((d, m), dtype=np.complex128)
        V[:, 0] = self.phi / phi_norm

        beta_prev = 0.0
        actual_m = m

        for j in range(m - 1):
            w = H @ V[:, j]
            if issparse(H):
                w = np.asarray(w).ravel()
            alpha = np.real(np.vdot(V[:, j], w))
            w -= alpha * V[:, j]
            if j > 0:
                w -= beta_prev * V[:, j - 1]
            beta = np.linalg.norm(w)
            if beta < KRYLOV_BREAKDOWN_TOL:
                actual_m = j + 1
                break
            V[:, j + 1] = w / beta
            beta_prev = beta

        V = V[:, :actual_m]
        Q, _ = np.linalg.qr(V, mode="reduced")
        return Q

    # -- Public interface --

    def evaluate(
        self, lambdas: np.ndarray, return_gradient: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute Krylov score R(lambda) and optionally its gradient.

        Algorithm:
            1. Build H(lambda) = sum_k lambda_k H_k
            2. Lanczos iteration: build orthonormal Krylov basis V
            3. Project: c = V^dag |psi>
            4. Score: R = ||c||^2
            5. (If gradient) Differentiate through Lanczos 3-term recurrence

        Uses sparse H when available for O(K*nnz*m) vs O(K*d^2*m).

        Memory-optimized: uses two (K,d) buffers for dV instead of (K,d,m)
        array, avoiding strided memory access (16x faster at d=64).
        """
        # Steps 1-4: Score computation via Lanczos
        # Use sparse H for forward pass when available
        H_sparse = self._construct_H_sparse(lambdas) if self._use_sparse else None
        H = H_sparse if H_sparse is not None else self._construct_H_dense(lambdas)

        if not return_gradient:
            V = self._lanczos_basis(H)
            coeffs = V.conj().T @ self.psi
            R = float(np.real(np.vdot(coeffs, coeffs)))
            return np.clip(R, 0.0, 1.0)

        # Step 5: Lanczos iteration with gradient differentiation
        # Uses dense H for batch dV matmul (H(lambda) is too dense for sparse)
        H_dense = H.toarray() if issparse(H) else H

        d = self.dim
        m = min(self.m, d)
        phi_norm = np.linalg.norm(self.phi)
        if phi_norm == 0:
            V = self._lanczos_basis(H)
            coeffs = V.conj().T @ self.psi
            R = float(np.real(np.vdot(coeffs, coeffs)))
            return np.clip(R, 0.0, 1.0), np.zeros(self.K)

        V = np.zeros((d, m), dtype=np.complex128)
        V[:, 0] = self.phi / phi_norm

        # Two-buffer approach for dV: avoids (K,d,m) strided access
        # dV_curr = dV[:, :, j], dV_prev = dV[:, :, j-1]
        dV_curr = np.zeros((self.K, d), dtype=np.complex128)
        dV_prev = np.zeros((self.K, d), dtype=np.complex128)

        # Accumulate gradient incrementally: dc[k,j] = dV[k,:,j]^H @ psi
        # We compute c[j] = V[:,j]^H @ psi at each step, then dc contribution
        psi = self.psi
        c_vec = np.zeros(m, dtype=np.complex128)
        dc_accum = np.zeros(self.K, dtype=np.complex128)  # sum of c[j]* dc[k,j]

        beta_prev = 0.0
        dbeta_prev = np.zeros(self.K)
        actual_m = m

        for j in range(m - 1):
            v_j = V[:, j]
            c_vec[j] = np.vdot(v_j, psi)

            # Accumulate dc for step j: dc[k,j] = dV_curr[k]^H @ psi
            dc_j = dV_curr.conj() @ psi  # (K,)
            dc_accum += c_vec[j].conj() * dc_j

            w = H_dense @ v_j

            # dw[k] = H_k @ v_j + H @ dV_curr[k]
            if self._use_sparse:
                hk_vj = np.empty((self.K, d), dtype=np.complex128)
                for k in range(self.K):
                    r = self.hams_sparse[k] @ v_j
                    hk_vj[k] = np.asarray(r).ravel()
                dw = hk_vj + (H_dense @ dV_curr.T).T
            else:
                dw = self._hams_array @ v_j + (H_dense @ dV_curr.T).T

            # Lanczos coefficients and their derivatives
            alpha = np.real(np.vdot(v_j, w))
            dalpha = np.real(dV_curr.conj() @ w + dw @ v_j.conj())

            # 3-term recurrence
            w_tilde = w - alpha * v_j
            dw_tilde = dw - dalpha[:, None] * v_j - alpha * dV_curr
            if j > 0:
                v_prev = V[:, j - 1]
                w_tilde -= beta_prev * v_prev
                dw_tilde -= (dbeta_prev[:, None] * v_prev
                             + beta_prev * dV_prev)

            beta = np.linalg.norm(w_tilde)
            if beta < KRYLOV_BREAKDOWN_TOL:
                actual_m = j + 1
                break

            dbeta = np.real(dw_tilde @ w_tilde.conj()) / beta

            V[:, j + 1] = w_tilde / beta
            dV_next = (dw_tilde - dbeta[:, None] * V[:, j + 1]) / beta

            # Rotate buffers
            dV_prev = dV_curr
            dV_curr = dV_next

            beta_prev = beta
            dbeta_prev = dbeta

        # Final step: accumulate dc for the last Lanczos vector
        # Only needed when loop completed normally (no breakdown).
        # On breakdown, the loop already accumulated dc for the last vector
        # before the break statement.
        if actual_m == m:
            last_j = actual_m - 1
            c_vec[last_j] = np.vdot(V[:, last_j], psi)
            dc_j = dV_curr.conj() @ psi
            dc_accum += c_vec[last_j].conj() * dc_j

        c = c_vec[:actual_m]
        R = float(np.real(np.vdot(c, c)))
        R = np.clip(R, 0.0, 1.0)

        # Gradient: dR/dlambda_k = 2 Re(sum_j c[j]* dc[k,j])
        grad = 2.0 * np.real(dc_accum)

        return R, grad


# ============================================================================
# Moment criterion
# ============================================================================

class MomentCriterion(ReachabilityCriterion):
    """
    Moment-based criterion (lambda-independent).

    Tests whether Q + gamma L L^T is positive definite for some gamma,
    where:
        L[k] = <H_k>_psi - <H_k>_phi
        Q[k,m] = <{H_k, H_m}>_psi - <{H_k, H_m}>_phi  ({A,B} = AB+BA)

    For large |gamma|, gamma*LL^T dominates except on ker(L), so
    M(gamma) > 0 effectively checks Q's positive definiteness on ker(L).

    Verdict:
    - UNREACHABLE if certificate found (Q + gamma*LL^T > 0)
    - INCONCLUSIVE if no certificate found (does NOT mean reachable)
    - Never returns REACHABLE
    """

    def __init__(
        self,
        hams: List[np.ndarray],
        phi: np.ndarray,
        psi: np.ndarray,
        tau: float = 0.99,
    ):
        super().__init__(hams, phi, psi, tau)

    def evaluate(
        self, lambdas: np.ndarray = None, return_gradient: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Evaluate the moment criterion (lambda-independent).

        Returns 1.0 if unreachable (certificate found), 0.0 otherwise.
        """
        unreachable, _, _ = self._check()
        score = 1.0 if unreachable else 0.0
        if return_gradient:
            return score, np.zeros(self.K)
        return score

    def _check(self) -> Tuple[bool, Optional[float], np.ndarray]:
        """
        Check if Q + gamma*LL^T certifies unreachability.

        Builds the first-moment difference vector L and second-moment
        difference matrix Q, then tests positive definiteness at
        gamma = +-1000 (large values that probe Q's definiteness on ker(L)).
        """
        phi = self.phi
        psi = self.psi

        # Compute H_k @ psi and H_k @ phi for all k
        if self._use_sparse:
            Hpsi = np.empty((self.K, self.dim), dtype=np.complex128)
            Hphi = np.empty((self.K, self.dim), dtype=np.complex128)
            for k in range(self.K):
                r = self.hams_sparse[k] @ psi
                Hpsi[k] = np.asarray(r).ravel()
                r = self.hams_sparse[k] @ phi
                Hphi[k] = np.asarray(r).ravel()
        else:
            hams = self._hams_array  # (K, d, d)
            Hpsi = np.einsum('kij,j->ki', hams, psi)   # (K, d)
            Hphi = np.einsum('kij,j->ki', hams, phi)   # (K, d)

        # L[k] = <H_k>_psi - <H_k>_phi (first moment difference)
        L = np.real(np.einsum('d,kd->k', psi.conj(), Hpsi)
                    - np.einsum('d,kd->k', phi.conj(), Hphi))

        # Q[k,m] = <{H_k, H_m}>_psi - <{H_k, H_m}>_phi
        # For Hermitian operators: <{H_k,H_m}> = 2*Re(<H_k H_m>)
        Q = 2 * (np.real(Hpsi.conj() @ Hpsi.T) - np.real(Hphi.conj() @ Hphi.T))

        L_outer = np.outer(L, L)
        tol = 1e-10

        for gamma in [1000.0, -1000.0]:
            M = Q + gamma * L_outer
            eigvals = np.linalg.eigvalsh(M)
            if np.all(eigvals > tol) or np.all(eigvals < -tol):
                return True, gamma, eigvals

        return False, None, np.array([])

    def is_reachable(self, **kwargs) -> ReachabilityResult:
        """
        Check moment criterion.

        Returns UNREACHABLE if certificate found, INCONCLUSIVE otherwise.
        """
        unreachable, gamma, eigvals = self._check()

        if unreachable:
            sign = ">" if eigvals[0] > 0 else "<"
            return ReachabilityResult(
                verdict=Verdict.UNREACHABLE,
                score=1.0,
                certificate=f"Q + {gamma:.4f} LL^T {sign} 0",
                raw_data={"gamma": gamma, "eigvals": eigvals},
            )
        else:
            return ReachabilityResult(
                verdict=Verdict.INCONCLUSIVE,
                score=0.0,
                certificate=None,
            )
