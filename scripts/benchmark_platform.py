#!/usr/bin/env python3
"""Platform-specific eigh benchmark.

Run this script to determine the optimal eigendecomposition backend
for your hardware. The result tells you whether the default
eigendecompose() settings in math_utils.py are optimal for your platform.

Usage:
    python scripts/benchmark_platform.py
"""
import platform
import time

import numpy as np
from scipy.linalg import eigh as scipy_eigh


def determine_optimal_backend():
    """Benchmark to determine optimal eigh backend for this platform."""
    d = 256
    rng = np.random.default_rng(42)
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (H + H.conj().T) / 2

    n_trials = 20

    # Warmup
    np.linalg.eigh(H.copy())
    scipy_eigh(H.copy())
    scipy_eigh(H.copy(), driver="evd")

    # numpy (heevd)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        np.linalg.eigh(H.copy())
    numpy_time = (time.perf_counter() - t0) / n_trials * 1000

    # scipy default (heevr)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        scipy_eigh(H.copy(), overwrite_a=True, check_finite=False)
    scipy_heevr_time = (time.perf_counter() - t0) / n_trials * 1000

    # scipy evd (heevd)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        scipy_eigh(H.copy(), driver="evd", overwrite_a=True, check_finite=False)
    scipy_heevd_time = (time.perf_counter() - t0) / n_trials * 1000

    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"\nBenchmark results (d={d}, complex128, {n_trials} trials):")
    print(f"  numpy.linalg.eigh (heevd):     {numpy_time:.2f} ms")
    print(f"  scipy eigh default (heevr):    {scipy_heevr_time:.2f} ms")
    print(f"  scipy eigh driver=evd (heevd): {scipy_heevd_time:.2f} ms")

    best = min(numpy_time, scipy_heevr_time, scipy_heevd_time)
    if scipy_heevr_time == best:
        print(f"\nOptimal for this platform: scipy.linalg.eigh (heevr default)")
        print("Current eigendecompose() settings are optimal.")
    elif scipy_heevd_time == best:
        print(f"\nOptimal for this platform: scipy.linalg.eigh(driver='evd')")
        print("Consider updating eigendecompose() to use driver='evd' at d>=256.")
    else:
        print(f"\nOptimal for this platform: numpy.linalg.eigh")
        print("Consider updating eigendecompose() to use numpy at all dimensions.")

    # Ratio info
    speedup = numpy_time / scipy_heevr_time
    if speedup > 1:
        print(f"\nscipy heevr is {speedup:.1f}x faster than numpy heevd at d={d}")
    else:
        print(f"\nnumpy heevd is {1/speedup:.1f}x faster than scipy heevr at d={d}")


if __name__ == "__main__":
    determine_optimal_backend()
