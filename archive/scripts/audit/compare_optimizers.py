#!/usr/bin/env python3
"""
Audit Test 3: Compare optimization methods and configurations.

Tests:
1. L-BFGS-B vs Nelder-Mead vs Powell (derivative-free alternatives)
2. Effect of maxiter: 20 vs 50 vs 200 vs 500
3. Effect of restarts: 1 vs 2 vs 5 vs 10
4. GEO2 vs Canonical at same K
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from reach import models, optimize, mathematics, settings
from scipy.optimize import minimize


def compare_methods_single_problem(psi, phi, hams, methods=None):
    """Compare optimization methods on a single problem instance."""
    if methods is None:
        methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']

    K = len(hams)
    results = {}

    # Same starting point for all methods
    rng = np.random.RandomState(42)
    x0 = rng.uniform(-1, 1, K)
    bounds = [(-1.0, 1.0)] * K

    def neg_spectral(x):
        return -mathematics.spectral_overlap(x, psi, phi, hams)

    for method in methods:
        t0 = time.time()
        try:
            if method in ['L-BFGS-B', 'TNC', 'SLSQP']:
                res = minimize(neg_spectral, x0, method=method, bounds=bounds,
                             options={'maxiter': 200})
            else:
                res = minimize(neg_spectral, x0, method=method,
                             options={'maxiter': 500})
                res.x = np.clip(res.x, -1, 1)

            elapsed = time.time() - t0
            results[method] = {
                'score': -res.fun,
                'success': res.success,
                'nfev': res.nfev,
                'time': elapsed,
            }
        except Exception as e:
            results[method] = {
                'score': 0.0,
                'success': False,
                'nfev': 0,
                'time': time.time() - t0,
                'error': str(e),
            }

    return results


def test_maxiter_effect(ensemble, d, K, maxiters=(10, 20, 50, 100, 200), n_trials=10, **params):
    """Test effect of maxiter on optimization quality."""
    rng = np.random.RandomState(42)
    scores_by_maxiter = {m: [] for m in maxiters}

    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed + 1, **params)

        for maxiter in maxiters:
            result = optimize.maximize_spectral_overlap(
                psi, phi, hams,
                restarts=1,
                maxiter=maxiter,
                seed=seed + 2,
            )
            scores_by_maxiter[maxiter].append(result['best_value'])

    return {m: np.array(v) for m, v in scores_by_maxiter.items()}


def test_restart_effect(ensemble, d, K, restart_counts=(1, 2, 5, 10), n_trials=10, **params):
    """Test effect of restart count on optimization quality."""
    rng = np.random.RandomState(42)
    scores_by_restarts = {r: [] for r in restart_counts}

    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed + 1, **params)

        for restarts in restart_counts:
            result = optimize.maximize_spectral_overlap(
                psi, phi, hams,
                restarts=restarts,
                maxiter=settings.GEO2_MAXITER if ensemble == 'GEO2' else settings.DEFAULT_MAXITER,
                seed=seed + 2,
            )
            scores_by_restarts[restarts].append(result['best_value'])

    return {r: np.array(v) for r, v in scores_by_restarts.items()}


def main():
    print("=" * 70)
    print("AUDIT TEST 3: Optimizer Configuration Comparison")
    print("=" * 70)

    rho = 0.05

    # === Part A: Method Comparison ===
    print("\n" + "=" * 70)
    print("Part A: Method Comparison (GEO2 d=16)")
    print("=" * 70)

    d = 16
    K = max(2, round(rho * d**2))
    rng = np.random.RandomState(42)

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=42)[0]
    hams = models.random_hamiltonian_ensemble(d, K, "GEO2", seed=43, nx=2, ny=2)

    results = compare_methods_single_problem(psi, phi, hams)
    print(f"\n  d={d}, K={K}:")
    for method, res in results.items():
        print(f"    {method:12s}: score={res['score']:.4f}, nfev={res['nfev']}, "
              f"time={res['time']:.3f}s, success={res['success']}")

    # === Part B: maxiter Effect ===
    print("\n" + "=" * 70)
    print("Part B: Effect of maxiter")
    print("=" * 70)

    for label, ensemble, d, params in [
        ("GEO2 d=16", "GEO2", 16, {'nx': 2, 'ny': 2}),
        ("GEO2 d=32", "GEO2", 32, {'nx': 1, 'ny': 5}),
        ("Canonical d=16", "canonical", 16, {}),
    ]:
        K = max(2, round(rho * d**2))
        print(f"\n  {label} (K={K}):")

        scores = test_maxiter_effect(ensemble, d, K, n_trials=8, **params)
        for maxiter in sorted(scores.keys()):
            arr = scores[maxiter]
            print(f"    maxiter={maxiter:4d}: mean S*={np.mean(arr):.4f} +/- {np.std(arr):.4f}")

    # === Part C: Restart Effect ===
    print("\n" + "=" * 70)
    print("Part C: Effect of restarts")
    print("=" * 70)

    for label, ensemble, d, params in [
        ("GEO2 d=16", "GEO2", 16, {'nx': 2, 'ny': 2}),
        ("GEO2 d=32", "GEO2", 32, {'nx': 1, 'ny': 5}),
    ]:
        K = max(2, round(rho * d**2))
        print(f"\n  {label} (K={K}):")

        scores = test_restart_effect(ensemble, d, K, n_trials=8, **params)
        for restarts in sorted(scores.keys()):
            arr = scores[restarts]
            print(f"    restarts={restarts:2d}: mean S*={np.mean(arr):.4f} +/- {np.std(arr):.4f}")


if __name__ == "__main__":
    main()
