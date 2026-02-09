#!/usr/bin/env python3
"""Benchmark QObj vs numpy path for all criterion functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
import numpy as np
from reach import mathematics, models

def bench(func, args_qobj, args_np, n_calls=50, label=""):
    """Time function with QObj args vs numpy args."""
    # Warmup
    func(*args_qobj); func(*args_np)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        func(*args_qobj)
    t_qobj = (time.perf_counter() - t0) / n_calls

    t0 = time.perf_counter()
    for _ in range(n_calls):
        func(*args_np)
    t_np = (time.perf_counter() - t0) / n_calls

    speedup = t_qobj / t_np if t_np > 0 else float('inf')
    print(f"  {label:40s}  QObj={t_qobj*1e3:8.2f}ms  numpy={t_np*1e3:8.2f}ms  speedup={speedup:.2f}x")
    return t_qobj, t_np, speedup

print("=" * 90)
print("TIMING: QObj vs numpy path")
print("=" * 90)

results = []
for d in [8, 16, 32]:
    K = max(2, d)
    hams = models.random_hamiltonian_ensemble(d, K, 'canonical', seed=42)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=99)[0]
    lam = np.random.RandomState(42).uniform(-1, 1, K)
    hams_np = [H.full() for H in hams]
    m = min(K, d)

    n_calls = 200 if d <= 16 else 50

    print(f"\n--- d={d}, K={K} ({n_calls} calls each) ---")

    # spectral_overlap: QObj vs numpy path
    def spec_qobj():
        return mathematics.spectral_overlap(lam, psi, phi, hams)
    def spec_np():
        return mathematics.spectral_overlap(lam, psi, phi, hams, hams_np=hams_np)

    spec_qobj(); spec_np()
    t0 = time.perf_counter()
    for _ in range(n_calls): spec_qobj()
    t_sq = (time.perf_counter() - t0) / n_calls
    t0 = time.perf_counter()
    for _ in range(n_calls): spec_np()
    t_sn = (time.perf_counter() - t0) / n_calls
    sp = t_sq / t_sn if t_sn > 0 else 0
    print(f"  {'spectral_overlap d=%d' % d:40s}  QObj={t_sq*1e3:8.2f}ms  numpy={t_sn*1e3:8.2f}ms  speedup={sp:.2f}x")

    bench(lambda *a: mathematics.spectral_overlap_with_grad(*a),
          (lam, psi, phi, hams),
          (lam, psi, phi, hams, hams_np),
          n_calls, f"spectral_overlap_with_grad d={d}")

    bench(lambda: mathematics.krylov_score(lam, psi, phi, hams, m=m),
          (),
          (),
          n_calls, f"krylov_score d={d} m={m} (QObj baseline)")

    # krylov_score: QObj vs numpy path
    def krylov_qobj():
        return mathematics.krylov_score(lam, psi, phi, hams, m=m)
    def krylov_np():
        return mathematics.krylov_score(lam, psi, phi, hams, m=m, hams_np=hams_np)

    # Warmup
    krylov_qobj(); krylov_np()
    t0 = time.perf_counter()
    for _ in range(n_calls): krylov_qobj()
    t_kq = (time.perf_counter() - t0) / n_calls
    t0 = time.perf_counter()
    for _ in range(n_calls): krylov_np()
    t_kn = (time.perf_counter() - t0) / n_calls
    sp = t_kq / t_kn if t_kn > 0 else 0
    print(f"  {'krylov_score d=%d m=%d' % (d,m):40s}  QObj={t_kq*1e3:8.2f}ms  numpy={t_kn*1e3:8.2f}ms  speedup={sp:.2f}x")

    # krylov_score_with_grad: QObj vs numpy path
    def krylov_grad_qobj():
        return mathematics.krylov_score_with_grad(lam, psi, phi, hams, m=m)
    def krylov_grad_np():
        return mathematics.krylov_score_with_grad(lam, psi, phi, hams, m=m, hams_np=hams_np)

    krylov_grad_qobj(); krylov_grad_np()
    t0 = time.perf_counter()
    for _ in range(n_calls): krylov_grad_qobj()
    t_kgq = (time.perf_counter() - t0) / n_calls
    t0 = time.perf_counter()
    for _ in range(n_calls): krylov_grad_np()
    t_kgn = (time.perf_counter() - t0) / n_calls
    sp = t_kgq / t_kgn if t_kgn > 0 else 0
    print(f"  {'krylov_score_with_grad d=%d m=%d' % (d,m):40s}  QObj={t_kgq*1e3:8.2f}ms  numpy={t_kgn*1e3:8.2f}ms  speedup={sp:.2f}x")

print("\nTiming complete")
