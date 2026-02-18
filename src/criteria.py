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
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

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
            self.hams = model_or_hams.basis
        else:
            self.model = None
            self.hams = model_or_hams
        self.phi = phi.flatten()
        self.psi = psi.flatten()
        self.tau = tau
        self.K = len(self.hams)
        self.dim = self.hams[0].shape[0]
        self._hams_array = np.stack(self.hams)  # (K, d, d) for vectorized ops

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

    def _maximize(
        self,
        method: str = DEFAULT_METHOD,
        restarts: int = DEFAULT_RESTARTS,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Multi-restart optimization to maximize the criterion score.

        Generates `restarts` random starting points in [-1,1]^K and runs
        L-BFGS-B (with analytical gradient) from each. Returns the best
        result across all restarts.
        """
        K = self.K
        bounds = DEFAULT_BOUNDS * K
        rng = np.random.RandomState(seed or 42)
        use_grad = (method == "L-BFGS-B")

        def objective(x):
            if use_grad:
                val, grad = self.evaluate(x, return_gradient=True)
                return -float(val), -grad.astype(np.float64)
            else:
                return -float(self.evaluate(x))

        best_value = 0.0
        best_x = np.zeros(K)
        total_nfev = 0
        start_time = time.time()

        for _ in range(restarts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            try:
                options = {"maxiter": maxiter, "ftol": ftol}
                if use_grad:
                    result = minimize(objective, x0, method=method,
                                      jac=True, bounds=bounds, options=options)
                else:
                    result = minimize(objective, x0, method=method,
                                      bounds=bounds, options=options)
                total_nfev += result.nfev
                current = -result.fun
                if current > best_value:
                    best_value = current
                    best_x = result.x.copy()
                if best_value >= self.tau:
                    break  # Already reachable, skip remaining restarts
            except Exception as e:
                logger.debug(f"Optimization restart failed: {e}")
                continue

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
    ) -> ReachabilityResult:
        """Maximize the criterion score and compare to tau threshold."""
        result = self._maximize(
            method=method, restarts=restarts,
            maxiter=maxiter, ftol=ftol, seed=seed,
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
        grad = np.zeros(self.K)
        U = eigenvectors
        weighted_inv_phi = inv_delta_E * phi_coeffs[None, :]  # (d, d)
        weighted_inv_psi = inv_delta_E * psi_coeffs[None, :]  # (d, d)
        sign_c_conj = sign_c.conj()

        for k in range(self.K):
            Hk_eig = U.conj().T @ self.hams[k] @ U
            dphi = np.sum(Hk_eig * weighted_inv_phi, axis=1)
            dpsi = np.sum(Hk_eig * weighted_inv_psi, axis=1)
            dc = dpsi.conj() * phi_coeffs + psi_coeffs.conj() * dphi
            grad[k] = np.real(np.sum(sign_c_conj * dc))

        return S, grad


# ============================================================================
# Krylov criterion
# ============================================================================

class KrylovCriterion(OptimizableCriterion):
    """
    Krylov subspace projection criterion.

    R(lambda) = ||P_Km(H(lambda)) |psi>||^2

    where P_Km is the projection onto Krylov subspace K_m(H(lambda), phi).

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

    def _lanczos_basis(self, H: np.ndarray) -> np.ndarray:
        """
        Build orthonormal Krylov basis K_m(H, phi) via Lanczos iteration.

        For Hermitian H, Lanczos uses a 3-term recurrence instead of full
        Gram-Schmidt orthogonalization (Arnoldi). This gives equivalent results
        with O(d) work per step instead of O(d*j).

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

    def _krylov_basis(self, H: np.ndarray) -> np.ndarray:
        """
        Build orthonormal Krylov basis via Arnoldi iteration.

        Deprecated: Use _lanczos_basis for Hermitian H (faster, equivalent).
        Kept for reference and potential non-Hermitian extensions.
        """
        d = H.shape[0]
        m = min(self.m, d)
        phi_norm = np.linalg.norm(self.phi)
        if phi_norm == 0:
            return np.zeros((d, 0), dtype=np.complex128)

        result = np.empty((d, m), dtype=np.complex128)
        result[:, 0] = self.phi / phi_norm

        actual_m = m
        for i in range(1, m):
            w = H @ result[:, i - 1]
            h = result[:, :i].conj().T @ w
            w = w - result[:, :i] @ h
            norm_w = np.linalg.norm(w)
            if norm_w < KRYLOV_BREAKDOWN_TOL:
                actual_m = i
                break
            result[:, i] = w / norm_w

        result = result[:, :actual_m]
        Q, _ = np.linalg.qr(result, mode="reduced")
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
        """
        # Steps 1-4: Score computation via Lanczos
        H = np.tensordot(lambdas, self._hams_array, axes=(0, 0))

        if not return_gradient:
            V = self._lanczos_basis(H)
            coeffs = V.conj().T @ self.psi
            R = float(np.real(np.vdot(coeffs, coeffs)))
            return np.clip(R, 0.0, 1.0)

        # Step 5: Lanczos iteration with gradient differentiation
        #
        # Lanczos 3-term recurrence for Hermitian H:
        #   w = H v_j - alpha_j v_j - beta_{j-1} v_{j-1}
        # where alpha_j = Re(v_j† H v_j), beta_j = ||w||.
        #
        # Differentiating w.r.t. lambda_k (denoting d/dlambda_k as d prefix):
        #   dw_k  = H_k v_j + H dv_j_k
        #   dalpha_k = Re(dv_j_k† w + v_j† dw_k)
        #   dwtilde_k = dw_k - dalpha_k v_j - alpha_j dv_j_k
        #                     - dbeta_{j-1}_k v_{j-1} - beta_{j-1} dv_{j-1}_k
        #   dbeta_k = Re(wtilde† dwtilde_k) / beta_j
        #   dv_{j+1}_k = (dwtilde_k - dbeta_k v_{j+1}) / beta_j
        #
        # Each step is O(K*d) instead of Arnoldi's O(K*d*j), giving
        # O(K*d*m) total gradient cost vs Arnoldi's O(K*d*m^2).

        d = H.shape[0]
        m = min(self.m, d)
        phi_norm = np.linalg.norm(self.phi)
        if phi_norm == 0:
            V = self._lanczos_basis(H)
            coeffs = V.conj().T @ self.psi
            R = float(np.real(np.vdot(coeffs, coeffs)))
            return np.clip(R, 0.0, 1.0), np.zeros(self.K)

        V = np.zeros((d, m), dtype=np.complex128)
        dV = np.zeros((self.K, d, m), dtype=np.complex128)
        V[:, 0] = self.phi / phi_norm
        # dV[:, :, 0] = 0 (phi is constant w.r.t. lambda)

        beta_prev = 0.0
        dbeta_prev = np.zeros(self.K)
        actual_m = m

        for j in range(m - 1):
            v_j = V[:, j]
            w = H @ v_j
            # dw[k] = H_k @ v_j + H @ dV[k, :, j]
            dw = self._hams_array @ v_j + (H @ dV[:, :, j].T).T

            # Lanczos coefficients and their derivatives
            alpha = np.real(np.vdot(v_j, w))
            dalpha = np.real(dV[:, :, j].conj() @ w + dw @ v_j.conj())  # (K,)

            # 3-term recurrence
            w_tilde = w - alpha * v_j
            dw_tilde = dw - dalpha[:, None] * v_j - alpha * dV[:, :, j]
            if j > 0:
                v_prev = V[:, j - 1]
                w_tilde -= beta_prev * v_prev
                dw_tilde -= (dbeta_prev[:, None] * v_prev
                             + beta_prev * dV[:, :, j - 1])

            beta = np.linalg.norm(w_tilde)
            if beta < KRYLOV_BREAKDOWN_TOL:
                actual_m = j + 1
                break

            # Derivative of beta = ||w_tilde||
            dbeta = np.real(dw_tilde @ w_tilde.conj()) / beta  # (K,)

            V[:, j + 1] = w_tilde / beta
            dV[:, :, j + 1] = (dw_tilde - dbeta[:, None] * V[:, j + 1]) / beta

            beta_prev = beta
            dbeta_prev = dbeta

        V = V[:, :actual_m]
        dV = dV[:, :, :actual_m]

        # Score: R = ||V† psi||^2
        c = V.conj().T @ self.psi  # (actual_m,)
        R = float(np.real(np.vdot(c, c)))
        R = np.clip(R, 0.0, 1.0)

        # Gradient: dR/dlambda_k = 2 Re(c† dc_k)
        dc_all = dV.conj().transpose(0, 2, 1) @ self.psi  # (K, actual_m)
        grad = 2.0 * np.real(np.einsum('i,ki->k', c.conj(), dc_all))

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
        hams = self._hams_array  # (K, d, d)

        # L[k] = <H_k>_psi - <H_k>_phi (first moment difference)
        # Vectorized: L[k] = Re(psi^H @ H_k @ psi - phi^H @ H_k @ phi)
        Hpsi = np.einsum('kij,j->ki', hams, psi)   # (K, d)
        Hphi = np.einsum('kij,j->ki', hams, phi)   # (K, d)
        L = np.real(np.einsum('d,kd->k', psi.conj(), Hpsi)
                    - np.einsum('d,kd->k', phi.conj(), Hphi))

        # Q[k,m] = <{H_k, H_m}>_psi - <{H_k, H_m}>_phi
        # where {A,B} = AB + BA (paper Eq. 6, no 1/2 factor)
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
        unreachable, x_opt, eigvals = self._check()

        if unreachable:
            sign = ">" if eigvals[0] > 0 else "<"
            return ReachabilityResult(
                verdict=Verdict.UNREACHABLE,
                score=1.0,
                certificate=f"Q + {x_opt:.4f} LL^T {sign} 0",
                raw_data={"x_opt": x_opt, "eigvals": eigvals},
            )
        else:
            return ReachabilityResult(
                verdict=Verdict.INCONCLUSIVE,
                score=0.0,
                certificate=None,
            )
