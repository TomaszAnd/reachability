#!/usr/bin/env python3
"""
Micro-benchmark to isolate timing of each component.
Tests: eigendecomposition, Hamiltonian construction, single criterion evaluation,
full is_reachable, and adaptive restart overhead.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.math_utils import eigendecompose
from src.models import CanonicalQuditModel, QubitGridModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion, Verdict


def time_eigendecomposition(d, n_trials=20):
    """Time eigendecomposition for dimension d."""
    rng = np.random.default_rng(42)
    A = rng.random((d, d)) + 1j * rng.random((d, d))
    H = (A + A.conj().T) / 2

    # Warmup
    eigendecompose(H.copy())

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        eigendecompose(H.copy())
        times.append(time.perf_counter() - t0)

    return np.mean(times), np.std(times)


def time_hamiltonian_construction(d, K, sparse=False, n_trials=50):
    """Time H(lambda) construction."""
    if sparse:
        model = QubitGridModel(dim=d, seed=42)
        sub = model.sample_submodel(min(K, model.K), seed=42)
    else:
        model = CanonicalQuditModel(dim=d, seed=42)
        sub = model.sample_submodel(K, seed=42)

    phi = model.init_state()
    psi = sub.random_state()
    crit = SpectralCriterion(sub, phi, psi, tau=0.99)
    lambdas = np.random.randn(crit.K) * 0.5

    # Warmup
    crit._construct_H(lambdas)

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        crit._construct_H(lambdas)
        times.append(time.perf_counter() - t0)

    return np.mean(times), np.std(times)


def time_single_evaluate(d, K, criterion='spectral', n_trials=20):
    """Time a single evaluate() call (with gradient)."""
    model = CanonicalQuditModel(dim=d, seed=42)
    sub = model.sample_submodel(K, seed=42)
    phi = model.init_state()
    psi = sub.random_state()

    if criterion == 'spectral':
        crit = SpectralCriterion(sub, phi, psi, tau=0.99)
    else:
        crit = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)

    lambdas = np.random.randn(K) * 0.5

    # Warmup
    crit.evaluate(lambdas, return_gradient=True)

    times = []
    for _ in range(n_trials):
        lambdas = np.random.randn(K) * 0.5
        t0 = time.perf_counter()
        crit.evaluate(lambdas, return_gradient=True)
        times.append(time.perf_counter() - t0)

    return np.mean(times), np.std(times)


def time_full_is_reachable(d, K, criterion='spectral', maxiter=50, restarts=3, n_trials=5):
    """Time full is_reachable() including optimization."""
    model = CanonicalQuditModel(dim=d, seed=42)

    times = []
    for trial in range(n_trials):
        sub = model.sample_submodel(K, seed=trial + 100)
        phi = model.init_state()
        psi = sub.random_state()

        if criterion == 'spectral':
            crit = SpectralCriterion(sub, phi, psi, tau=0.99)
        else:
            crit = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)

        t0 = time.perf_counter()
        result = crit.is_reachable(maxiter=maxiter, restarts=restarts)
        times.append(time.perf_counter() - t0)

    return np.mean(times), np.std(times)


def show_effective_restarts(d, restarts=3):
    """Show what effective_restarts would be at various K values."""
    rho_c_est = 0.12 * np.sqrt(8 / d)
    K_values = [2, 5, int(0.5 * rho_c_est * d**2),
                int(rho_c_est * d**2), int(2 * rho_c_est * d**2)]
    K_values = [max(2, k) for k in K_values]

    print(f"  d={d}, base restarts={restarts}:")
    for K in K_values:
        eff = restarts
        if K < 10:
            eff = max(eff, 18 - K)
        rho = K / d**2
        if 0.04 < rho < 0.30:
            eff = max(eff, 12)
        if d >= 32:
            dim_factor = 1 + (d - 32) // 32
            eff = max(eff, restarts + 3 * dim_factor)
        eff = max(eff, 5)
        print(f"    K={K:4d} (rho={rho:.4f}): effective_restarts={eff}")


if __name__ == "__main__":
    print("=" * 70)
    print("MICRO-BENCHMARK: Component Timing")
    print("=" * 70)

    # 1. Eigendecomposition
    print("\n1. EIGENDECOMPOSITION")
    print("-" * 50)
    for d in [16, 32, 64, 128, 256]:
        mean_t, std_t = time_eigendecomposition(d)
        print(f"  d={d:3d}: {mean_t*1000:8.2f} +/- {std_t*1000:.2f} ms")

    # 2. Hamiltonian construction
    print("\n2. HAMILTONIAN CONSTRUCTION (dense tensordot)")
    print("-" * 50)
    for d in [16, 32, 64]:
        K = max(2, int(0.1 * d**2))
        mean_t, std_t = time_hamiltonian_construction(d, K, sparse=False)
        print(f"  d={d:2d}, K={K:3d}: {mean_t*1000:8.3f} +/- {std_t*1000:.3f} ms")

    # 3. Single evaluate with gradient
    print("\n3. SINGLE EVALUATE (with gradient)")
    print("-" * 50)
    for d in [16, 32, 64]:
        K = max(2, int(0.1 * d**2))
        for crit in ['spectral', 'krylov']:
            mean_t, std_t = time_single_evaluate(d, K, crit)
            print(f"  d={d:2d}, K={K:3d}, {crit:8s}: {mean_t*1000:8.2f} +/- {std_t*1000:.2f} ms")

    # 4. Full is_reachable at different settings
    print("\n4. FULL is_reachable (optimization)")
    print("-" * 50)
    for maxiter in [50, 100]:
        for restarts_base in [3, 5]:
            print(f"\n  maxiter={maxiter}, restarts={restarts_base}:")
            for d in [16, 32, 64]:
                K = max(2, int(0.1 * d**2))
                for crit in ['spectral', 'krylov']:
                    mean_t, std_t = time_full_is_reachable(
                        d, K, crit, maxiter=maxiter, restarts=restarts_base, n_trials=3)
                    print(f"    d={d:2d}, K={K:3d}, {crit:8s}: "
                          f"{mean_t:7.3f} +/- {std_t:.3f} s")

    # 5. Adaptive restart overhead
    print("\n5. ADAPTIVE RESTART ANALYSIS")
    print("-" * 50)
    for d in [16, 32, 64, 128]:
        show_effective_restarts(d, restarts=3)

    # 6. Sweep time estimates
    print("\n6. SWEEP TIME ESTIMATES (15 K values)")
    print("-" * 50)
    # Use maxiter=50, restarts=3 timings
    for d in [16, 32, 64, 128]:
        K = max(2, int(0.1 * d**2))
        t_spec, _ = time_full_is_reachable(d, K, 'spectral', 50, 3, n_trials=3)
        t_kry, _ = time_full_is_reachable(d, K, 'krylov', 50, 3, n_trials=3)
        t_total = t_spec + t_kry  # per trial (both criteria)

        for n_trials_est, label in [(500, "500 trials"), (1000, "1000 trials")]:
            hours = 15 * n_trials_est * t_total / 3600
            print(f"  d={d:3d}: {t_total:.3f}s/trial -> {label} x 15 K = {hours:.1f}h")
