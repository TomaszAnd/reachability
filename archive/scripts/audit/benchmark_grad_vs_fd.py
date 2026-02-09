#!/usr/bin/env python3
"""
Benchmark: L-BFGS-B with analytical gradient vs finite-difference.
This directly measures the speedup from the gradient implementation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from scipy.optimize import minimize
from reach import models, mathematics, settings


def run_benchmark(d, K, ensemble, n_trials=10, maxiter=100, restarts=2, **model_params):
    """Compare L-BFGS-B with analytical grad vs finite-diff."""

    times_grad = []
    times_fd = []
    scores_grad = []
    scores_fd = []
    nfev_grad = []
    nfev_fd = []

    for trial in range(n_trials):
        seed = 5000 + trial
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed + 500)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **model_params)
        hams_np = [H.full() for H in hams]
        bounds = [(-1.0, 1.0)] * K
        rng = np.random.RandomState(seed + 100)

        # --- Analytical gradient ---
        def obj_grad(x):
            S, g = mathematics.spectral_overlap_with_grad(x, psi, phi, hams, hams_np=hams_np)
            return -S, -g.astype(np.float64)

        best_val_grad = 0.0
        total_nfev_g = 0
        t0 = time.time()
        for r in range(restarts):
            x0 = rng.uniform(-1, 1, K)
            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-6})
            total_nfev_g += res.nfev
            val = -res.fun
            if val > best_val_grad:
                best_val_grad = val
        times_grad.append(time.time() - t0)
        scores_grad.append(best_val_grad)
        nfev_grad.append(total_nfev_g)

        # --- Finite-difference gradient (old behavior) ---
        def obj_fd(x):
            S = mathematics.spectral_overlap(x, psi, phi, hams)
            return -S

        rng2 = np.random.RandomState(seed + 100)  # Same starting points
        best_val_fd = 0.0
        total_nfev_f = 0
        t0 = time.time()
        for r in range(restarts):
            x0 = rng2.uniform(-1, 1, K)
            res = minimize(obj_fd, x0, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-6})
            total_nfev_f += res.nfev
            val = -res.fun
            if val > best_val_fd:
                best_val_fd = val
        times_fd.append(time.time() - t0)
        scores_fd.append(best_val_fd)
        nfev_fd.append(total_nfev_f)

    return {
        'grad_time': np.mean(times_grad),
        'fd_time': np.mean(times_fd),
        'grad_score': np.mean(scores_grad),
        'fd_score': np.mean(scores_fd),
        'grad_nfev': np.mean(nfev_grad),
        'fd_nfev': np.mean(nfev_fd),
        'speedup': np.mean(times_fd) / np.mean(times_grad),
    }


def main():
    print("=" * 70)
    print("BENCHMARK: L-BFGS-B Analytical Gradient vs Finite-Difference")
    print("Both use L-BFGS-B with same maxiter=100, restarts=2")
    print("=" * 70)

    configs = [
        ("GEO2 d=16 (K=13)", 16, 13, "GEO2", {'nx': 2, 'ny': 2}),
        ("GEO2 d=32 (K=51)", 32, 51, "GEO2", {'nx': 1, 'ny': 5}),
    ]

    for label, d, K, ensemble, params in configs:
        print(f"\n{label}:")
        r = run_benchmark(d, K, ensemble, n_trials=10, maxiter=100, restarts=2, **params)
        print(f"  Analytical grad: {r['grad_time']*1000:.0f}ms/trial, "
              f"S*={r['grad_score']:.4f}, nfev={r['grad_nfev']:.0f}")
        print(f"  Finite-diff:     {r['fd_time']*1000:.0f}ms/trial, "
              f"S*={r['fd_score']:.4f}, nfev={r['fd_nfev']:.0f}")
        print(f"  Speedup: {r['speedup']:.1f}×")
        print(f"  Score difference: {r['grad_score'] - r['fd_score']:.6f}")


if __name__ == "__main__":
    main()
