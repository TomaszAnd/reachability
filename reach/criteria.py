"""
Reachability criteria for quantum control systems.

Three criteria for testing whether target state |phi> is reachable
from initial state |psi> under H(lambda) = sum_k lambda_k H_k:

- SpectralCriterion: Spectral overlap S(lambda) = sum_n |<u_n|phi>* <u_n|psi>|
- KrylovCriterion: Krylov projection R(lambda) = ||P_K |phi>||^2
- MomentCriterion: Positive definiteness of moment matrix Q + x L L^T

Verdict system:
- SpectralCriterion and KrylovCriterion return:
  - REACHABLE if optimized score >= tau
  - UNREACHABLE if optimized score < tau

- MomentCriterion returns:
  - UNREACHABLE if positive definiteness certificate found
  - INCONCLUSIVE if no certificate found (does NOT imply reachable)

Note: INCONCLUSIVE is only used by MomentCriterion. Spectral and Krylov
always return a definite verdict based on the threshold comparison.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from .math_utils import (
    OVERLAP_TOLERANCE, KRYLOV_BREAKDOWN_TOL,
    eigendecompose, clip_to_bounds, construct_hamiltonian,
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
DEFAULT_FTOL = 1e-8


class ReachabilityCriterion(ABC):
    """
    Abstract base class for reachability criteria.

    Accepts either a QuantumModel or a list of Hamiltonian numpy arrays.
    """

    def __init__(
        self,
        model_or_hams,
        psi: np.ndarray,
        phi: np.ndarray,
        tau: float = 0.99,
    ):
        from .models import QuantumModel
        if isinstance(model_or_hams, QuantumModel):
            self.model = model_or_hams
            self.hams = model_or_hams.basis
        else:
            self.model = None
            self.hams = model_or_hams
        self.psi = psi.flatten()
        self.phi = phi.flatten()
        self.tau = tau
        self.K = len(self.hams)
        self.dim = self.hams[0].shape[0]

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
        Test reachability by optimizing the criterion.

        Returns:
            ReachabilityResult with verdict, score, and metadata
        """
        pass


class SpectralCriterion(ReachabilityCriterion):
    """
    Spectral overlap criterion.

    S(lambda) = sum_n |<u_n(lambda)|phi>* <u_n(lambda)|psi>|

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

        S(lambda) = sum_n |<u_n(lambda)|phi>* <u_n(lambda)|psi>|

        where |u_n(lambda)> are eigenstates of H(lambda) = sum_k lambda_k H_k.
        """
        # Build combined Hamiltonian H(lambda)
        H = construct_hamiltonian(lambdas, self.hams)

        # Eigendecomposition: H |u_n> = E_n |u_n>
        try:
            eigenvalues, eigenvectors = eigendecompose(H)
        except RuntimeError:
            if return_gradient:
                return 0.0, np.zeros(self.K)
            return 0.0

        # Project states onto eigenbasis: <u_n|psi> and <u_n|phi>
        psi_coeffs = eigenvectors.conj().T @ self.psi
        phi_coeffs = eigenvectors.conj().T @ self.phi

        # Overlap coefficients: c_n = <u_n|phi>* <u_n|psi>
        c = phi_coeffs.conj() * psi_coeffs
        abs_c = np.abs(c)
        # Spectral overlap: S = sum_n |c_n|
        S = float(np.sum(abs_c))
        S = np.clip(S, 0.0, 1.0)

        if not return_gradient:
            return S

        # Analytical gradient via first-order perturbation theory.
        # Phase factors: sign(c_n) = c_n / |c_n| (zero for vanishing terms)
        safe_abs_c = np.where(abs_c > 1e-15, abs_c, 1.0)
        sign_c = np.where(abs_c > 1e-15, c / safe_abs_c, 0.0)

        # Energy differences: regularized inverse Delta/(Delta^2 + eps^2)
        # smoothly handles degenerate eigenvalues
        E = eigenvalues
        delta_E = E[:, None] - E[None, :]
        eps_reg = 1e-12
        inv_delta_E = delta_E / (delta_E**2 + eps_reg)

        # dS/dlambda_k = sum_n Re[sign(c_n)* dc_n/dlambda_k]
        grad = np.zeros(self.K)
        for k in range(self.K):
            # Transform H_k to eigenbasis
            Hk_eig = eigenvectors.conj().T @ self.hams[k] @ eigenvectors
            # Perturbation theory: d<u_n|psi>/dlambda_k
            dpsi = np.sum(Hk_eig * inv_delta_E * psi_coeffs[None, :], axis=1)
            dphi = np.sum(Hk_eig * inv_delta_E * phi_coeffs[None, :], axis=1)
            dc = dphi.conj() * psi_coeffs + phi_coeffs.conj() * dpsi
            grad[k] = np.real(np.sum(sign_c.conj() * dc))

        return S, grad

    def is_reachable(
        self,
        method: str = DEFAULT_METHOD,
        restarts: int = DEFAULT_RESTARTS,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        seed: Optional[int] = None,
    ) -> ReachabilityResult:
        """Maximize S(lambda) over [-1,1]^K and compare to tau."""
        result = _maximize_criterion(
            self, method=method, restarts=restarts,
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


class KrylovCriterion(ReachabilityCriterion):
    """
    Krylov subspace projection criterion.

    R(lambda) = ||P_Km(H(lambda)) |phi>||^2

    where P_Km is the projection onto Krylov subspace K_m(H(lambda), psi).

    Default: m = dim (full dimension). Use m = min(K, d) for
    genuine phase transitions in density sweeps.
    """

    def __init__(
        self,
        hams: List[np.ndarray],
        psi: np.ndarray,
        phi: np.ndarray,
        tau: float = 0.99,
        m: Optional[int] = None,
    ):
        super().__init__(hams, psi, phi, tau)
        self.m = m if m is not None else self.dim

    def _krylov_basis(self, H: np.ndarray) -> np.ndarray:
        """
        Build orthonormal Krylov basis K_m(H, psi) via Arnoldi iteration.

        Returns (d, m) matrix with orthonormal columns, QR-compressed.
        """
        d = H.shape[0]
        m = min(self.m, d)
        psi_norm = np.linalg.norm(self.psi)
        if psi_norm == 0:
            return np.zeros((d, 0), dtype=np.complex128)

        result = np.empty((d, m), dtype=np.complex128)
        result[:, 0] = self.psi / psi_norm

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

    def evaluate(
        self, lambdas: np.ndarray, return_gradient: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Compute Krylov score R(lambda) and optionally its gradient.

        R(lambda) = ||P_Km(H(lambda)) |phi>||^2

        where P_Km is projection onto Krylov subspace K_m(H(lambda), psi).
        """
        H = construct_hamiltonian(lambdas, self.hams)
        V = self._krylov_basis(H)

        coeffs = V.conj().T @ self.phi
        R = float(np.real(np.vdot(coeffs, coeffs)))
        R = np.clip(R, 0.0, 1.0)

        if not return_gradient:
            return R

        # Gradient via differentiating Arnoldi iteration
        d = H.shape[0]
        m = min(self.m, d)
        psi_norm = np.linalg.norm(self.psi)
        if psi_norm == 0:
            return R, np.zeros(self.K)

        # Re-run Arnoldi with derivative tracking
        V_arn = np.zeros((d, m), dtype=np.complex128)
        dV = np.zeros((self.K, d, m), dtype=np.complex128)
        V_arn[:, 0] = self.psi / psi_norm

        actual_m = m
        for i in range(1, m):
            w = H @ V_arn[:, i - 1]
            dw = np.zeros((self.K, d), dtype=np.complex128)
            for k in range(self.K):
                dw[k] = self.hams[k] @ V_arn[:, i - 1] + H @ dV[k, :, i - 1]

            h = V_arn[:, :i].conj().T @ w
            dh = np.zeros((self.K, i), dtype=np.complex128)
            for k in range(self.K):
                dh[k] = V_arn[:, :i].conj().T @ dw[k] + dV[k, :, :i].conj().T @ w

            w_tilde = w - V_arn[:, :i] @ h
            dw_tilde = np.zeros((self.K, d), dtype=np.complex128)
            for k in range(self.K):
                dw_tilde[k] = dw[k] - V_arn[:, :i] @ dh[k] - dV[k, :, :i] @ h

            norm_w = np.linalg.norm(w_tilde)
            if norm_w < KRYLOV_BREAKDOWN_TOL:
                actual_m = i
                break

            V_arn[:, i] = w_tilde / norm_w
            v_i = V_arn[:, i]
            for k in range(self.K):
                proj = np.real(np.vdot(v_i, dw_tilde[k]))
                dV[k, :, i] = (dw_tilde[k] - proj * v_i) / norm_w

        V_arn = V_arn[:, :actual_m]
        dV = dV[:, :, :actual_m]

        c = V_arn.conj().T @ self.phi
        grad = np.zeros(self.K)
        for k in range(self.K):
            dc = dV[k].conj().T @ self.phi
            grad[k] = 2.0 * np.real(np.vdot(c, dc))

        return R, grad

    def is_reachable(
        self,
        method: str = DEFAULT_METHOD,
        restarts: int = DEFAULT_RESTARTS,
        maxiter: int = DEFAULT_MAXITER,
        ftol: float = DEFAULT_FTOL,
        seed: Optional[int] = None,
    ) -> ReachabilityResult:
        """Maximize R(lambda) over [-1,1]^K and compare to tau."""
        result = _maximize_criterion(
            self, method=method, restarts=restarts,
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


class MomentCriterion(ReachabilityCriterion):
    """
    Moment-based criterion (lambda-independent).

    Tests whether Q + x L L^T is positive definite for some x,
    where:
        L[k] = <H_k>_phi - <H_k>_psi
        Q[k,m] = <{H_k, H_m}/2>_phi - <{H_k, H_m}/2>_psi

    Verdict:
    - UNREACHABLE if certificate found (Q + xLL^T > 0)
    - INCONCLUSIVE if no certificate found (does NOT mean reachable)
    - Never returns REACHABLE (moment criterion can only prove unreachability)
    """

    def __init__(
        self,
        hams: List[np.ndarray],
        psi: np.ndarray,
        phi: np.ndarray,
        tau: float = 0.99,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        n_x_points: int = 1000,
    ):
        super().__init__(hams, psi, phi, tau)
        self.x_range = x_range
        self.n_x_points = n_x_points

    def evaluate(
        self, lambdas: np.ndarray = None, return_gradient: bool = False
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """
        Evaluate the moment criterion.

        Note: lambdas is ignored (moment criterion is lambda-independent).
        Returns 1.0 if unreachable (certificate found), 0.0 otherwise.
        """
        unreachable, _, _ = self._check()
        score = 1.0 if unreachable else 0.0
        if return_gradient:
            return score, np.zeros(self.K)
        return score

    def _check(self) -> Tuple[bool, Optional[float], np.ndarray]:
        """Run the moment criterion check."""
        K = self.K
        psi = self.psi
        phi = self.phi

        # Compute L vector
        L = np.zeros(K)
        for k in range(K):
            L[k] = np.real(phi.conj() @ self.hams[k] @ phi
                          - psi.conj() @ self.hams[k] @ psi)

        # Compute Q matrix
        Q = np.zeros((K, K))
        for k in range(K):
            for m_idx in range(K):
                anticomm = (self.hams[k] @ self.hams[m_idx]
                           + self.hams[m_idx] @ self.hams[k]) / 2
                Q[k, m_idx] = np.real(
                    phi.conj() @ anticomm @ phi
                    - psi.conj() @ anticomm @ psi)

        # Search for x where Q + x L L^T is positive definite
        L_outer = np.outer(L, L)
        tol = 1e-10

        for x in np.linspace(self.x_range[0], self.x_range[1], self.n_x_points):
            M = Q + x * L_outer
            eigvals = np.linalg.eigvalsh(M)
            if np.all(eigvals > tol):
                return True, x, eigvals

        return False, None, np.array([])

    def is_reachable(self, **kwargs) -> ReachabilityResult:
        """
        Check moment criterion.

        Returns UNREACHABLE if certificate found, INCONCLUSIVE otherwise.
        """
        unreachable, x_opt, eigvals = self._check()

        if unreachable:
            return ReachabilityResult(
                verdict=Verdict.UNREACHABLE,
                score=1.0,
                certificate=f"Q + {x_opt:.4f} LL^T > 0",
                raw_data={"x_opt": x_opt, "eigvals": eigvals},
            )
        else:
            return ReachabilityResult(
                verdict=Verdict.INCONCLUSIVE,
                score=0.0,
                certificate=None,
            )


# ============================================================================
# Multi-restart optimizer (shared by Spectral and Krylov)
# ============================================================================

def _maximize_criterion(
    criterion: ReachabilityCriterion,
    method: str = DEFAULT_METHOD,
    restarts: int = DEFAULT_RESTARTS,
    maxiter: int = DEFAULT_MAXITER,
    ftol: float = DEFAULT_FTOL,
    seed: Optional[int] = None,
) -> Dict:
    """
    Multi-restart optimization to maximize a criterion score.

    Uses analytical gradient when available (L-BFGS-B).
    """
    K = criterion.K
    bounds = DEFAULT_BOUNDS * K
    rng = np.random.RandomState(seed or 42)

    use_grad = (method == "L-BFGS-B")

    def objective(x):
        if use_grad:
            val, grad = criterion.evaluate(x, return_gradient=True)
            return -float(val), -grad.astype(np.float64)
        else:
            return -float(criterion.evaluate(x))

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
