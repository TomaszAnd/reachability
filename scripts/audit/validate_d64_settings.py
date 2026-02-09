#!/usr/bin/env python3
"""
Validate optimizer settings at d=64 (2x3 lattice, 6 qubits).

Tests:
1. Provably reachable targets: false-unreachable rate must be ≤5%
2. Maxiter sweep: find minimum maxiter for convergence at K=205 (ρ=0.05)
3. Gradient quality: verify analytical gradient works at high K

d=64 is the critical dimension because K = round(ρ * d²) = 205 at ρ=0.05,
making this the most demanding optimization configuration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from scipy.optimize import minimize
from reach import models, mathematics, settings


def test_provably_reachable(d, K, ensemble, nx, ny, n_trials=20,
                            maxiter=100, restarts=3, tau=0.99):
    """Test false-unreachable rate on provably reachable targets."""
    print(f"\n{'='*60}")
    print(f"TEST 1: Provably Reachable Targets")
    print(f"  d={d}, K={K}, {ensemble} ({nx}x{ny}), maxiter={maxiter}, restarts={restarts}")
    print(f"{'='*60}")

    false_unreachable = 0
    scores = []
    times = []

    for trial in range(n_trials):
        seed = 9000 + trial
        t0 = time.time()

        # Generate system
        psi = models.fock_state(d, 0)
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, nx=nx, ny=ny)
        hams_np = [H.full() for H in hams]

        # Create provably reachable target: phi = exp(-iH(lambda*)t) |psi>
        rng = np.random.RandomState(seed + 300)
        lam_true = rng.uniform(-1, 1, K)
        H_true = sum(lam_true[k] * hams_np[k] for k in range(K))
        t_evol = 1.0
        U = _matrix_exp(-1j * H_true * t_evol)
        phi_vec = U @ psi.full().flatten()
        import qutip
        phi = qutip.Qobj(phi_vec.reshape(-1, 1))

        # Optimize spectral overlap
        bounds = [(-1.0, 1.0)] * K
        best_S = 0.0

        for r in range(restarts):
            x0 = np.random.RandomState(seed + 100 + r).uniform(-1, 1, K)

            def obj_grad(x):
                S, g = mathematics.spectral_overlap_with_grad(
                    x, psi, phi, hams, hams_np=hams_np)
                return -S, -g.astype(np.float64)

            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                          bounds=bounds,
                          options={'maxiter': maxiter, 'ftol': 1e-6})
            val = -res.fun
            if val > best_S:
                best_S = val

        elapsed = time.time() - t0
        times.append(elapsed)
        scores.append(best_S)

        is_false_unreach = best_S < tau
        if is_false_unreach:
            false_unreachable += 1

        status = "FALSE-UNREACH" if is_false_unreach else "OK"
        print(f"  Trial {trial:2d}: S*={best_S:.4f} [{status}] ({elapsed:.1f}s)")

    rate = false_unreachable / n_trials
    print(f"\n  RESULT: {false_unreachable}/{n_trials} false-unreachable = {rate:.1%}")
    print(f"  Mean S*: {np.mean(scores):.4f} (min={min(scores):.4f}, max={max(scores):.4f})")
    print(f"  Mean time: {np.mean(times):.1f}s per trial")
    return rate, scores, times


def test_maxiter_sweep(d, K, ensemble, nx, ny, n_trials=10):
    """Sweep maxiter to find minimum for convergence at d=64."""
    print(f"\n{'='*60}")
    print(f"TEST 2: Maxiter Sweep (d={d}, K={K})")
    print(f"{'='*60}")

    maxiter_values = [50, 75, 100, 150, 200]

    for maxiter in maxiter_values:
        scores = []
        t0 = time.time()

        for trial in range(n_trials):
            seed = 9500 + trial
            psi = models.fock_state(d, 0)
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, nx=nx, ny=ny)
            hams_np = [H.full() for H in hams]

            # Provably reachable target
            rng = np.random.RandomState(seed + 300)
            lam_true = rng.uniform(-1, 1, K)
            H_true = sum(lam_true[k] * hams_np[k] for k in range(K))
            U = _matrix_exp(-1j * H_true * 1.0)
            phi_vec = U @ psi.full().flatten()
            import qutip
            phi = qutip.Qobj(phi_vec.reshape(-1, 1))

            bounds = [(-1.0, 1.0)] * K
            best_S = 0.0

            for r in range(3):
                x0 = np.random.RandomState(seed + 100 + r).uniform(-1, 1, K)

                def obj_grad(x):
                    S, g = mathematics.spectral_overlap_with_grad(
                        x, psi, phi, hams, hams_np=hams_np)
                    return -S, -g.astype(np.float64)

                res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                              bounds=bounds,
                              options={'maxiter': maxiter, 'ftol': 1e-6})
                val = -res.fun
                if val > best_S:
                    best_S = val

            scores.append(best_S)

        elapsed = time.time() - t0
        false_unreach = sum(1 for s in scores if s < 0.99)
        print(f"  maxiter={maxiter:3d}: mean S*={np.mean(scores):.4f}, "
              f"min={min(scores):.4f}, false_unreach={false_unreach}/{n_trials}, "
              f"time={elapsed:.1f}s")


def test_gradient_quality(d, K, ensemble, nx, ny):
    """Verify gradient accuracy at d=64 by comparing with finite differences."""
    print(f"\n{'='*60}")
    print(f"TEST 3: Gradient Quality (d={d}, K={K})")
    print(f"{'='*60}")

    seed = 9900
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed + 500)[0]
    hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, nx=nx, ny=ny)
    hams_np = [H.full() for H in hams]

    rng = np.random.RandomState(seed)
    lam = rng.uniform(-1, 1, K)

    # Analytical gradient
    t0 = time.time()
    S_anal, grad_anal = mathematics.spectral_overlap_with_grad(
        lam, psi, phi, hams, hams_np=hams_np)
    t_anal = time.time() - t0

    # Finite-difference gradient (sample 10 components)
    eps = 1e-6
    n_check = min(10, K)
    indices = np.random.RandomState(42).choice(K, n_check, replace=False)

    t0 = time.time()
    grad_fd = np.zeros(n_check)
    for i, idx in enumerate(indices):
        lam_p = lam.copy()
        lam_m = lam.copy()
        lam_p[idx] += eps
        lam_m[idx] -= eps
        Sp = mathematics.spectral_overlap(lam_p, psi, phi, hams)
        Sm = mathematics.spectral_overlap(lam_m, psi, phi, hams)
        grad_fd[i] = (Sp - Sm) / (2 * eps)
    t_fd = time.time() - t0

    grad_anal_subset = grad_anal[indices]
    rel_errors = np.abs(grad_anal_subset - grad_fd) / (np.abs(grad_fd) + 1e-15)
    max_rel_error = np.max(rel_errors)

    print(f"  S(λ) = {S_anal:.6f}")
    print(f"  Analytical gradient time: {t_anal:.3f}s")
    print(f"  Finite-diff time ({n_check} components): {t_fd:.3f}s")
    print(f"  Extrapolated FD full gradient: {t_fd * K / n_check:.1f}s")
    print(f"  Speedup estimate: {(t_fd * K / n_check) / t_anal:.1f}×")
    print(f"  Max relative error: {max_rel_error:.2e}")
    print(f"  Mean relative error: {np.mean(rel_errors):.2e}")

    if max_rel_error < 1e-4:
        print(f"  PASS: Gradient accurate at d={d}")
    else:
        print(f"  WARN: Gradient may be inaccurate (max rel error={max_rel_error:.2e})")


def _matrix_exp(M):
    """Matrix exponential via eigendecomposition."""
    from scipy.linalg import eigh
    # M should be -i*H*t where H is Hermitian => M is anti-Hermitian
    # Use eigh on i*M = H*t
    H_eff = 1j * M  # This is H*t, which is Hermitian
    E, V = eigh(H_eff)
    return V @ np.diag(np.exp(-1j * E)) @ V.conj().T


def main():
    print("=" * 60)
    print("d=64 OPTIMIZER SETTINGS VALIDATION")
    print("Lattice: 2×3 (6 qubits), d=64")
    print("=" * 60)

    d = 64
    nx, ny = 2, 3
    ensemble = "GEO2"
    K_rho005 = round(0.05 * d**2)  # K=205

    print(f"\nConfiguration:")
    print(f"  d = {d}")
    print(f"  Lattice: {nx}×{ny} ({nx*ny} qubits)")
    print(f"  K at ρ=0.05: {K_rho005}")
    print(f"  Current GEO2 settings: maxiter={settings.GEO2_MAXITER}, "
          f"restarts={settings.GEO2_RESTARTS}")

    # Test 3: Gradient quality (fast, do first)
    test_gradient_quality(d, K_rho005, ensemble, nx, ny)

    # Test 2: Maxiter sweep (medium)
    test_maxiter_sweep(d, K_rho005, ensemble, nx, ny, n_trials=5)

    # Test 1: Provably reachable targets (slowest)
    rate, scores, times = test_provably_reachable(
        d, K_rho005, ensemble, nx, ny,
        n_trials=10, maxiter=150, restarts=3, tau=0.99
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if rate <= 0.05:
        print(f"  PASS: False-unreachable rate = {rate:.1%} (≤5% threshold)")
    else:
        print(f"  FAIL: False-unreachable rate = {rate:.1%} (>5% threshold)")
        print(f"  ACTION: Increase maxiter or restarts for d=64")

    print(f"\n  Recommended d=64 settings:")
    print(f"    maxiter = 150 (or 200 for extra margin)")
    print(f"    restarts = 3")
    print(f"    Estimated time per trial: {np.mean(times):.1f}s")


if __name__ == "__main__":
    main()
