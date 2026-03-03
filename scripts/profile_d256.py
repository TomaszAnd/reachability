#!/usr/bin/env python3
"""
Timing profile for d=128 (Canonical) and d=256 (QubitGrid).

Measures key operations and extrapolates Canonical d=256 timings
using known algorithmic complexity:
  - Spectral: O(d^3) dominated by eigh
  - Moment: O(K*d^2) dominated by Hpsi/Hphi computation
  - Krylov: O(K*d*m) dominated by matvec

QubitGrid d=256 is directly measured (feasible with ~114 operators).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np

from src.models import CanonicalQuditModel, QubitGridModel, LATTICE_CONFIGS
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion


def time_operation(func, n_repeats=3):
    """Time a function, return median of n_repeats runs."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def profile_model(model_cls, d, model_kwargs, label, K_test=None):
    """Profile all operations for a given model and dimension."""
    print(f"\n{'='*60}")
    print(f"{label} d={d}")
    print(f"{'='*60}")

    # Model creation
    t_create = time_operation(lambda: model_cls(dim=d, seed=42, **model_kwargs))
    print(f"  Model creation:       {t_create*1000:8.1f} ms")

    model = model_cls(dim=d, seed=42, **model_kwargs)
    if K_test is None:
        K_test = min(20, model.K)

    sub = model.sample_submodel(K_test, seed=42)
    phi = sub.init_state()
    psi = sub.random_state()
    lambdas = np.random.RandomState(42).uniform(-1, 1, K_test)

    results = {'d': d, 'K': K_test, 'label': label}

    # SpectralCriterion
    sc = SpectralCriterion(sub, phi, psi)
    t_spec_create = time_operation(lambda: SpectralCriterion(sub, phi, psi))
    t_spec_eval = time_operation(lambda: sc.evaluate(lambdas))
    t_spec_grad = time_operation(lambda: sc.evaluate(lambdas, return_gradient=True))
    print(f"  Spectral create:      {t_spec_create*1000:8.1f} ms")
    print(f"  Spectral evaluate:    {t_spec_eval*1000:8.1f} ms")
    print(f"  Spectral eval+grad:   {t_spec_grad*1000:8.1f} ms")
    results['spectral_create'] = t_spec_create
    results['spectral_eval'] = t_spec_eval
    results['spectral_grad'] = t_spec_grad

    # MomentCriterion
    mc = MomentCriterion(sub, phi, psi)
    t_moment = time_operation(lambda: mc.is_reachable())
    print(f"  Moment is_reachable:  {t_moment*1000:8.1f} ms")
    results['moment_is_reachable'] = t_moment

    # KrylovCriterion
    kc = KrylovCriterion(sub, phi, psi, m=d)
    t_krylov_eval = time_operation(lambda: kc.evaluate(lambdas))
    t_krylov_grad = time_operation(lambda: kc.evaluate(lambdas, return_gradient=True))
    t_krylov_random = time_operation(
        lambda: kc.is_reachable(method='random', restarts=3))
    print(f"  Krylov evaluate:      {t_krylov_eval*1000:8.1f} ms")
    print(f"  Krylov eval+grad:     {t_krylov_grad*1000:8.1f} ms")
    print(f"  Krylov random search: {t_krylov_random*1000:8.1f} ms")
    results['krylov_eval'] = t_krylov_eval
    results['krylov_grad'] = t_krylov_grad
    results['krylov_random'] = t_krylov_random

    return results


def main():
    print("d=256 Timing Profile")
    print("=" * 60)

    # Canonical d=128
    canonical_128 = profile_model(
        CanonicalQuditModel, 128, {}, "Canonical", K_test=20)

    # QubitGrid d=256
    if 256 in LATTICE_CONFIGS:
        nx, ny = LATTICE_CONFIGS[256]
        qubitgrid_256 = profile_model(
            QubitGridModel, 256, {'nx': nx, 'ny': ny}, "QubitGrid", K_test=20)
    else:
        print("\nQubitGrid d=256: no lattice config, skipping")
        qubitgrid_256 = None

    # Extrapolate Canonical d=256 from d=128
    print(f"\n{'='*60}")
    print("EXTRAPOLATION: Canonical d=256 from d=128")
    print("=" * 60)
    scale_d3 = (256 / 128) ** 3  # = 8x for O(d^3) operations
    scale_d2 = (256 / 128) ** 2  # = 4x for O(d^2) operations

    print(f"  Scaling factors: O(d^3) = {scale_d3:.0f}x, O(d^2) = {scale_d2:.0f}x")
    print()

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Operation':<25} {'d=128 (ms)':>12} {'d=256 proj (ms)':>16}")
    print(f"{'-'*25} {'-'*12} {'-'*16}")

    ops = [
        ('Spectral eval', 'spectral_eval', scale_d3),
        ('Spectral eval+grad', 'spectral_grad', scale_d3),
        ('Moment is_reachable', 'moment_is_reachable', scale_d2),
        ('Krylov eval', 'krylov_eval', scale_d2),
        ('Krylov eval+grad', 'krylov_grad', scale_d2),
        ('Krylov random', 'krylov_random', scale_d2),
    ]

    for name, key, scale in ops:
        t128 = canonical_128[key] * 1000
        t256_proj = t128 * scale
        print(f"  {name:<25} {t128:>10.1f} {t256_proj:>14.1f}")

    if qubitgrid_256:
        print(f"\n{'Operation':<25} {'d=256 actual (ms)':>18}")
        print(f"{'-'*25} {'-'*18}")
        for name, key, _ in ops:
            t256 = qubitgrid_256[key] * 1000
            print(f"  {name:<25} {t256:>16.1f}")

    print(f"\nNote: Canonical d=256 requires ~68GB RAM (d^2=65536 dense matrices).")
    print(f"      Only QubitGrid is feasible at d=256 on 16GB machines.")


if __name__ == '__main__':
    main()
