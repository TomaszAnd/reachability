#!/usr/bin/env python3
"""
Verify analytical gradient of spectral_overlap matches finite-difference.
Also benchmark analytical gradient vs finite-difference optimization speed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from reach import models, mathematics, optimize


def test_gradient_accuracy():
    """Compare analytical gradient to finite-difference gradient."""
    print("=" * 70)
    print("TEST 1: Gradient Accuracy (analytical vs finite-difference)")
    print("=" * 70)

    configs = [
        ("d=8, K=5", 8, 5),
        ("d=16, K=13", 16, 13),
        ("d=32, K=20", 32, 20),
    ]

    for label, d, K in configs:
        print(f"\n{label}:")
        rng = np.random.RandomState(42)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=123)[0]
        hams = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)
        lambdas = rng.uniform(-1, 1, K)

        # Analytical gradient
        S, grad_analytical = mathematics.spectral_overlap_with_grad(lambdas, psi, phi, hams)

        # Finite-difference gradient
        eps = 1e-6
        grad_fd = np.zeros(K)
        for k in range(K):
            lam_plus = lambdas.copy()
            lam_plus[k] += eps
            lam_minus = lambdas.copy()
            lam_minus[k] -= eps
            S_plus = mathematics.spectral_overlap(lam_plus, psi, phi, hams)
            S_minus = mathematics.spectral_overlap(lam_minus, psi, phi, hams)
            grad_fd[k] = (S_plus - S_minus) / (2 * eps)

        # Compare
        max_diff = np.max(np.abs(grad_analytical - grad_fd))
        rel_err = max_diff / (np.max(np.abs(grad_fd)) + 1e-15)
        print(f"  S(λ) = {S:.6f}")
        print(f"  max |grad_analytical - grad_fd| = {max_diff:.2e}")
        print(f"  relative error = {rel_err:.2e}")
        print(f"  grad_analytical[:5] = {grad_analytical[:5]}")
        print(f"  grad_fd[:5]         = {grad_fd[:5]}")

        if rel_err < 1e-4:
            print(f"  PASS (relative error < 1e-4)")
        else:
            print(f"  FAIL (relative error >= 1e-4)")


def test_gradient_speed():
    """Benchmark analytical gradient computation speed."""
    print("\n" + "=" * 70)
    print("TEST 2: Gradient Computation Speed")
    print("=" * 70)

    configs = [
        ("d=16, K=13", 16, 13, "GEO2", {'nx': 2, 'ny': 2}),
        ("d=32, K=51", 32, 51, "GEO2", {'nx': 1, 'ny': 5}),
    ]

    for label, d, K, ensemble, params in configs:
        print(f"\n{label}:")
        rng = np.random.RandomState(42)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=123)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=42, **params)
        lambdas = rng.uniform(-1, 1, K)

        # Time analytical gradient (single evaluation)
        n_rep = 20
        t0 = time.time()
        for _ in range(n_rep):
            S, grad = mathematics.spectral_overlap_with_grad(lambdas, psi, phi, hams)
        t_analytical = (time.time() - t0) / n_rep

        # Time finite-difference gradient (K+1 evaluations)
        t0 = time.time()
        for _ in range(n_rep):
            S0 = mathematics.spectral_overlap(lambdas, psi, phi, hams)
            eps = 1e-6
            for k_idx in range(K):
                lam_plus = lambdas.copy()
                lam_plus[k_idx] += eps
                mathematics.spectral_overlap(lam_plus, psi, phi, hams)
        t_fd = (time.time() - t0) / n_rep

        speedup = t_fd / t_analytical
        print(f"  Analytical: {t_analytical*1000:.1f}ms")
        print(f"  Finite-diff ({K+1} evals): {t_fd*1000:.1f}ms")
        print(f"  Speedup: {speedup:.1f}×")


def test_optimizer_with_gradient():
    """Compare optimization with and without analytical gradient."""
    print("\n" + "=" * 70)
    print("TEST 3: Optimization with Analytical Gradient vs Finite-Diff")
    print("=" * 70)

    configs = [
        ("d=16, K=13", 16, 13, "GEO2", {'nx': 2, 'ny': 2}),
        ("d=32, K=51", 32, 51, "GEO2", {'nx': 1, 'ny': 5}),
    ]

    for label, d, K, ensemble, params in configs:
        print(f"\n{label}:")
        n_trials = 10
        rng = np.random.RandomState(42)

        for trial in range(min(3, n_trials)):
            seed = rng.randint(0, 2**31)
            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed + 500)[0]
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)

            # With analytical gradient (default now)
            t0 = time.time()
            res_grad = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=100, restarts=2, ftol=1e-6, seed=seed,
            )
            t_grad = time.time() - t0

            # Without analytical gradient — use Nelder-Mead (no gradient needed)
            t0 = time.time()
            res_nm = optimize.maximize_spectral_overlap(
                psi, phi, hams, method='Nelder-Mead',
                maxiter=100, restarts=2, ftol=1e-6, seed=seed,
            )
            t_nm = time.time() - t0

            print(f"  Trial {trial}: "
                  f"L-BFGS-B+grad: S*={res_grad['best_value']:.4f} "
                  f"({t_grad*1000:.0f}ms, nfev={res_grad['nfev']}), "
                  f"Nelder-Mead: S*={res_nm['best_value']:.4f} "
                  f"({t_nm*1000:.0f}ms, nfev={res_nm['nfev']})")

        # Full timing comparison
        scores_grad = []
        scores_nm = []
        t_grad_total = 0
        t_nm_total = 0

        for trial in range(n_trials):
            seed = 7000 + trial
            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed + 500)[0]
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)

            t0 = time.time()
            res = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=100, restarts=2, ftol=1e-6, seed=seed,
            )
            t_grad_total += time.time() - t0
            scores_grad.append(res['best_value'])

            t0 = time.time()
            res = optimize.maximize_spectral_overlap(
                psi, phi, hams, method='Nelder-Mead',
                maxiter=100, restarts=2, ftol=1e-6, seed=seed,
            )
            t_nm_total += time.time() - t0
            scores_nm.append(res['best_value'])

        print(f"\n  Summary ({n_trials} trials):")
        print(f"    L-BFGS-B+grad: mean S*={np.mean(scores_grad):.4f}, "
              f"time/trial={t_grad_total/n_trials*1000:.0f}ms")
        print(f"    Nelder-Mead:    mean S*={np.mean(scores_nm):.4f}, "
              f"time/trial={t_nm_total/n_trials*1000:.0f}ms")
        print(f"    Speedup: {t_nm_total/t_grad_total:.1f}×")


if __name__ == "__main__":
    test_gradient_accuracy()
    test_gradient_speed()
    test_optimizer_with_gradient()
