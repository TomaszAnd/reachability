"""
Optimization utilities for maximizing spectral overlap (TIME-FREE).

Pipeline Role:
This module solves the core optimization problem in time-free reachability:

    S* = max_{λ ∈ [-1,1]ᴷ} S(λ)

where S(λ) is the spectral overlap function. The value S* determines whether
a target state |φ⟩ is reachable from initial state |ψ⟩ under Hamiltonian H(λ).

Optimization Problem:
- **Objective**: Maximize S(λ) = Σₙ |φₙ*(λ) ψₙ(λ)|
- **Variables**: λ = (λ₁, ..., λₖ) ∈ ℝᴷ
- **Constraints**: λᵢ ∈ [-1, 1] for all i (box constraints)

We minimize -S(λ) using scipy.optimize.minimize with multiple methods:
- **L-BFGS-B** (recommended): Handles box constraints natively
- **TNC, SLSQP**: Also support native bounds
- **CG, Powell, Nelder-Mead**: Unconstrained methods with clipping in objective

Multi-restart strategy improves global convergence by testing different
initial points. Returns best result across all restarts.

Bounds Handling:
- Methods with native bound support (L-BFGS-B, TNC, SLSQP): Pass bounds directly
- Unconstrained methods (CG, Powell, Nelder-Mead): Clip parameters in objective function

Module Relationship:
    - optimize.py (THIS FILE): Time-FREE spectral overlap S(λ) and Krylov score R(λ)
    - optimization.py: Time-dependent fidelity via U(t) = exp(-iHt) for Floquet validation
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import qutip
from scipy.optimize import minimize

from . import mathematics, settings

logger = logging.getLogger(__name__)


def get_optimizer_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get registry of available optimization methods with their configurations.

    Returns:
        Dictionary mapping method names to configuration dictionaries
        containing 'supports_bounds' and 'description' keys
    """
    return {
        "L-BFGS-B": {
            "supports_bounds": True,
            "description": "Limited-memory BFGS with bounds",
            "recommended": True,
        },
        "TNC": {
            "supports_bounds": True,
            "description": "Truncated Newton with bounds",
            "recommended": False,
        },
        "SLSQP": {
            "supports_bounds": True,
            "description": "Sequential Least Squares Programming",
            "recommended": False,
        },
        "CG": {
            "supports_bounds": False,
            "description": "Conjugate Gradient (unconstrained)",
            "recommended": False,
        },
        "Powell": {
            "supports_bounds": False,
            "description": "Powell's method (unconstrained)",
            "recommended": False,
        },
        "Nelder-Mead": {
            "supports_bounds": False,
            "description": "Nelder-Mead simplex (unconstrained)",
            "recommended": False,
        },
    }


def clip_to_bounds(x: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clip parameter vector to specified bounds.

    Args:
        x: Parameter vector to clip
        bounds: List of (lower, upper) bounds

    Returns:
        Clipped parameter vector

    Raises:
        ValueError: If bounds are malformed
    """
    return mathematics.clip_to_bounds(x, bounds)


def create_objective_function(
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    bounds: List[Tuple[float, float]],
    method: str,
) -> Callable[[np.ndarray], float]:
    """
    Create objective function for optimization.

    The objective is the negative spectral overlap (for minimization):
    f(λ) = -S(λ) = -Σₙ |⟨φₙ(λ)|ψₙ(λ)⟩|

    For methods that don't support bounds natively, parameters are clipped
    inside the objective function to ensure constraints are respected.

    Args:
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of Hamiltonian operators
        bounds: Parameter bounds [(λ₁_min, λ₁_max), ...]
        method: Optimization method name

    Returns:
        Objective function f(λ) → ℝ
    """
    registry = get_optimizer_registry()
    supports_bounds = registry.get(method, {}).get("supports_bounds", False)

    def objective(x: np.ndarray) -> float:
        """Objective function: negative spectral overlap."""
        # Clip to bounds for methods that don't support them natively
        if not supports_bounds:
            x = clip_to_bounds(x, bounds)

        # Compute spectral overlap
        try:
            overlap = mathematics.spectral_overlap(x, psi, phi, hams)
            return -float(overlap)  # Negative for minimization
        except Exception as e:
            logger.debug(f"Objective evaluation failed at λ={x}: {e}")
            return 0.0  # Return worst case (S=0 → f=0)

    return objective


def _maximize_criterion(
    criterion_func: Callable[..., float],
    criterion_name: str,
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = settings.DEFAULT_METHOD,
    restarts: int = settings.DEFAULT_RESTARTS,
    maxiter: int = settings.DEFAULT_MAXITER,
    ftol: float = settings.DEFAULT_FTOL,
    seed: Optional[int] = None,
    **criterion_kwargs,
) -> Dict[str, Any]:
    """
    Generic multi-restart optimizer for maximizing a criterion function.

    This is the common implementation used by both maximize_spectral_overlap()
    and maximize_krylov_score(). It handles input validation, multi-restart
    optimization, and result construction.

    Args:
        criterion_func: Function to maximize (e.g., mathematics.spectral_overlap)
        criterion_name: Name for logging (e.g., "S" for spectral, "R" for Krylov)
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of k Hamiltonian operators [H₁, H₂, ..., Hₖ]
        bounds: Parameter bounds (default: [-1,1]^k)
        method: Optimization method (see get_optimizer_registry())
        restarts: Number of random restarts
        maxiter: Maximum iterations per restart
        ftol: Function tolerance for convergence
        seed: Random seed for initial points
        **criterion_kwargs: Additional keyword arguments passed to criterion_func

    Returns:
        Dictionary containing:
        - best_value: Maximum criterion value
        - best_x: Optimal parameters λ*
        - nfev: Total function evaluations
        - success: Whether optimization succeeded
        - runtime_s: Total runtime in seconds
        - method: Optimization method used

    Raises:
        ValueError: If input parameters are invalid
        KeyError: If optimization method is not recognized
    """
    # Validate inputs
    k = len(hams)
    if k < 2:
        raise ValueError(f"Need at least 2 Hamiltonians, got {k}")

    if not mathematics.validate_quantum_state(psi):
        raise ValueError("psi is not a valid normalized quantum state")
    if not mathematics.validate_quantum_state(phi):
        raise ValueError("phi is not a valid normalized quantum state")

    # Set default bounds
    if bounds is None:
        bounds = [settings.DEFAULT_BOUNDS[0]] * k
    elif len(bounds) == 1:
        bounds = bounds * k

    mathematics.validate_bounds(bounds, k)

    # Check method availability
    registry = get_optimizer_registry()
    if method not in registry:
        available = list(registry.keys())
        raise KeyError(f"Unknown method '{method}'. Available: {available}")

    # Setup random number generator
    if seed is None:
        seed = settings.SEED
    rng = np.random.RandomState(seed)

    # Create objective function (negative criterion for minimization)
    supports_bounds = registry[method]["supports_bounds"]

    def objective(x: np.ndarray) -> float:
        """Objective function: negative criterion value."""
        if not supports_bounds:
            x = clip_to_bounds(x, bounds)
        try:
            value = criterion_func(x, psi, phi, hams, **criterion_kwargs)
            return -float(value)
        except Exception as e:
            logger.debug(f"Objective evaluation failed at λ={x}: {e}")
            return 0.0

    # Initialize tracking variables
    best_value = 0.0
    best_x = np.zeros(k)
    total_nfev = 0
    success = False
    start_time = time.time()

    logger.debug(
        f"Starting {criterion_name} optimization: method={method}, restarts={restarts}, maxiter={maxiter}"
    )

    # Multi-restart optimization
    for restart_idx in range(restarts):
        x0 = np.array([rng.uniform(low, high) for low, high in bounds])
        logger.debug(f"Restart {restart_idx + 1}/{restarts}: x0={x0}")

        try:
            options = {"maxiter": maxiter, "ftol": ftol}

            if supports_bounds:
                result = minimize(objective, x0, method=method, bounds=bounds, options=options)
            else:
                result = minimize(objective, x0, method=method, options=options)

            total_nfev += result.nfev
            current_value = -result.fun

            if current_value > best_value:
                best_value = current_value
                best_x = result.x.copy()
                success = result.success

            logger.debug(
                f"Restart {restart_idx + 1}: {criterion_name}={current_value:.6f}, success={result.success}"
            )

        except Exception as e:
            logger.warning(f"Restart {restart_idx + 1} failed with {method}: {e}")
            continue

    # Final clipping and recomputation
    best_x = clip_to_bounds(best_x, bounds)
    try:
        final_value = criterion_func(best_x, psi, phi, hams, **criterion_kwargs)
        best_value = max(best_value, final_value)
    except Exception as e:
        logger.warning(f"Final evaluation failed: {e}")

    runtime_s = time.time() - start_time

    result_dict = {
        "best_value": float(best_value),
        "best_x": best_x,
        "nfev": total_nfev,
        "success": success,
        "runtime_s": runtime_s,
        "method": method,
    }

    logger.debug(
        f"{criterion_name} optimization complete: {criterion_name}*={best_value:.6f}, "
        f"nfev={total_nfev}, time={runtime_s:.3f}s"
    )

    return result_dict


def maximize_spectral_overlap(
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = settings.DEFAULT_METHOD,
    restarts: int = settings.DEFAULT_RESTARTS,
    maxiter: int = settings.DEFAULT_MAXITER,
    ftol: float = settings.DEFAULT_FTOL,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Maximize spectral overlap S(λ) over parameter space.

    Solves the optimization problem:
    λ* = argmax S(λ) = argmax Σₙ |⟨φₙ(λ)|ψₙ(λ)⟩|
         λ∈bounds

    where H(λ) = Σₖ λₖ Hₖ and {|ψₙ(λ)⟩} are eigenstates of H(λ).

    Args:
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of k Hamiltonian operators [H₁, H₂, ..., Hₖ]
        bounds: Parameter bounds (default: [-1,1]^k)
        method: Optimization method (see get_optimizer_registry())
        restarts: Number of random restarts
        maxiter: Maximum iterations per restart
        ftol: Function tolerance for convergence
        seed: Random seed for initial points

    Returns:
        Dictionary containing:
        - best_value: Maximum spectral overlap S*
        - best_x: Optimal parameters λ*
        - nfev: Total function evaluations
        - success: Whether optimization succeeded
        - runtime_s: Total runtime in seconds
        - method: Optimization method used

    Raises:
        ValueError: If input parameters are invalid
        KeyError: If optimization method is not recognized
    """
    return _maximize_criterion(
        criterion_func=mathematics.spectral_overlap,
        criterion_name="S",
        psi=psi,
        phi=phi,
        hams=hams,
        bounds=bounds,
        method=method,
        restarts=restarts,
        maxiter=maxiter,
        ftol=ftol,
        seed=seed,
    )


def maximize_krylov_score(
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    m: Optional[int] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    method: str = settings.DEFAULT_METHOD,
    restarts: int = settings.DEFAULT_RESTARTS,
    maxiter: int = settings.DEFAULT_MAXITER,
    ftol: float = settings.DEFAULT_FTOL,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Maximize Krylov score R(λ) over parameter space.

    Solves the optimization problem:
    λ* = argmax R(λ) = argmax ‖P_Kₘ(H(λ))|φ⟩‖²
         λ∈bounds

    where H(λ) = Σₖ λₖ Hₖ and P_Kₘ is the projection onto the Krylov subspace
    K_m(H(λ), |ψ⟩) = span{|ψ⟩, H(λ)|ψ⟩, ..., H(λ)^(m-1)|ψ⟩}.

    Args:
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of k Hamiltonian operators [H₁, H₂, ..., Hₖ]
        m: Krylov rank (default: full dimension d)
        bounds: Parameter bounds (default: [-1,1]^k)
        method: Optimization method (see get_optimizer_registry())
        restarts: Number of random restarts
        maxiter: Maximum iterations per restart
        ftol: Function tolerance for convergence
        seed: Random seed for initial points

    Returns:
        Dictionary containing:
        - best_value: Maximum Krylov score R*
        - best_x: Optimal parameters λ*
        - nfev: Total function evaluations
        - success: Whether optimization succeeded
        - runtime_s: Total runtime in seconds
        - method: Optimization method used

    Raises:
        ValueError: If input parameters are invalid
        KeyError: If optimization method is not recognized
    """
    return _maximize_criterion(
        criterion_func=mathematics.krylov_score,
        criterion_name="R",
        psi=psi,
        phi=phi,
        hams=hams,
        bounds=bounds,
        method=method,
        restarts=restarts,
        maxiter=maxiter,
        ftol=ftol,
        seed=seed,
        m=m,  # Krylov-specific parameter
    )


def optimize_multiple_methods(
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    methods: Optional[List[str]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare optimization across multiple methods.

    Args:
        psi: Initial quantum state
        phi: Target quantum state
        hams: List of Hamiltonians
        methods: List of method names (default: all available)
        bounds: Parameter bounds
        **kwargs: Additional arguments passed to maximize_spectral_overlap

    Returns:
        Dictionary mapping method names to optimization results
    """
    if methods is None:
        methods = list(get_optimizer_registry().keys())

    results = {}
    for method in methods:
        try:
            result = maximize_spectral_overlap(
                psi, phi, hams, bounds=bounds, method=method, **kwargs
            )
            results[method] = result
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            results[method] = {
                "best_value": 0.0,
                "best_x": np.zeros(len(hams)),
                "nfev": 0,
                "success": False,
                "runtime_s": 0.0,
                "method": method,
                "error": str(e),
            }

    return results
