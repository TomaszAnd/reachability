#!/usr/bin/env python3
"""Exhaustive eigh benchmark for complex Hermitian matrices.

Compares all available LAPACK drivers via numpy and scipy to determine
the optimal eigendecomposition backend for this platform.

Usage:
    python scripts/benchmark_eigh_exhaustive.py
    python scripts/benchmark_eigh_exhaustive.py --dims 64 128 256 512
"""
import argparse
import platform
import time

import numpy as np
from scipy.linalg import eigh as scipy_eigh


def benchmark_all(d: int, n_trials: int = 10) -> dict:
    """Test all available eigh implementations at dimension d."""
    rng = np.random.default_rng(42)
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (H + H.conj().T) / 2

    results = {}

    configs = [
        ("numpy (heevd)", lambda h: np.linalg.eigh(h)),
        ("scipy default (heevr)", lambda h: scipy_eigh(h)),
        ("scipy optimized (heevr)", lambda h: scipy_eigh(h, overwrite_a=True, check_finite=False)),
        ("scipy driver=evd (heevd)", lambda h: scipy_eigh(h, driver="evd")),
        ("scipy driver=ev (heev)", lambda h: scipy_eigh(h, driver="ev")),
    ]

    for name, func in configs:
        # Warmup
        func(H.copy())

        t0 = time.perf_counter()
        for _ in range(n_trials):
            func(H.copy())
        results[name] = (time.perf_counter() - t0) / n_trials * 1000

    # Print results
    print(f"\n=== d={d} ===")
    fastest = min(results.values())
    for name, t in sorted(results.items(), key=lambda x: x[1]):
        ratio = t / fastest
        marker = " <-- fastest" if t == fastest else ""
        print(f"  {name:30s}: {t:8.2f} ms  ({ratio:.2f}x){marker}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exhaustive eigh benchmark")
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 384, 512])
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    print("Complex Hermitian eigh benchmark")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print("=" * 60)

    all_results = {}
    for d in args.dims:
        all_results[d] = benchmark_all(d, n_trials=args.trials)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Fastest method per dimension")
    print("=" * 60)
    for d, results in all_results.items():
        best_name = min(results, key=results.get)
        best_time = results[best_name]
        print(f"  d={d:4d}: {best_name:30s} ({best_time:.2f} ms)")


if __name__ == "__main__":
    main()
