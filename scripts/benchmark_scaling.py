#!/usr/bin/env python3
"""Analyze how each component scales with Hilbert space dimension.

Measures eigendecomposition, Spectral, Krylov, and Moment times across
dimensions to understand scaling behavior and bottleneck distribution.

Usage:
    python scripts/benchmark_scaling.py
    python scripts/benchmark_scaling.py --dims 32 64 128
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.models import QubitGridModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion
from src.math_utils import eigendecompose


def scaling_analysis(dims, n_eigh=10):
    """Measure component times across dimensions."""
    results = {
        "d": [], "K": [],
        "eigh_ms": [], "spectral_ms": [], "krylov_ms": [], "moment_ms": [],
    }

    for d in dims:
        K = min(d // 2, 60)  # Cap K to avoid exceeding basis size
        print(f"Testing d={d}, K={K}...")

        # Eigendecomposition
        rng = np.random.default_rng(42)
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H = (H + H.conj().T) / 2
        eigendecompose(H.copy())  # warmup
        t0 = time.perf_counter()
        for _ in range(n_eigh):
            eigendecompose(H.copy())
        eigh_time = (time.perf_counter() - t0) / n_eigh * 1000

        # Full criteria (QubitGrid)
        model = QubitGridModel(dim=d, seed=42)
        sub = model.sample_submodel(K, seed=0)
        phi = sub.init_state()
        psi = sub.random_state()

        # Moment
        mc = MomentCriterion(sub, phi, psi, tau=0.99)
        mc.is_reachable()  # warmup
        t0 = time.perf_counter()
        for _ in range(n_eigh):
            mc.is_reachable()
        moment_time = (time.perf_counter() - t0) / n_eigh * 1000

        # Krylov (single restart)
        kc = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)
        t0 = time.perf_counter()
        kc.is_reachable(maxiter=30, restarts=1)
        krylov_time = (time.perf_counter() - t0) * 1000

        # Spectral (single restart)
        sc = SpectralCriterion(sub, phi, psi, tau=0.99)
        t0 = time.perf_counter()
        sc.is_reachable(maxiter=30, restarts=1)
        spectral_time = (time.perf_counter() - t0) * 1000

        results["d"].append(d)
        results["K"].append(K)
        results["eigh_ms"].append(eigh_time)
        results["spectral_ms"].append(spectral_time)
        results["krylov_ms"].append(krylov_time)
        results["moment_ms"].append(moment_time)

    # Print results
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS (QubitGrid)")
    print(f"{'='*70}")
    print(f"{'d':>4} {'K':>4} | {'eigh':>10} | {'Spectral':>12} | {'Krylov':>12} | {'Moment':>10}")
    print("-" * 70)
    for i, d in enumerate(results["d"]):
        print(
            f"{d:>4} {results['K'][i]:>4} | "
            f"{results['eigh_ms'][i]:>8.2f}ms | "
            f"{results['spectral_ms'][i]:>10.1f}ms | "
            f"{results['krylov_ms'][i]:>10.1f}ms | "
            f"{results['moment_ms'][i]:>8.2f}ms"
        )

    # Scaling ratios
    print("\nScaling ratios (d -> 2d):")
    for i in range(len(dims) - 1):
        ratio_eigh = results["eigh_ms"][i + 1] / results["eigh_ms"][i]
        ratio_spectral = results["spectral_ms"][i + 1] / results["spectral_ms"][i]
        ratio_krylov = results["krylov_ms"][i + 1] / results["krylov_ms"][i]
        print(
            f"  d={dims[i]}->{dims[i+1]}: "
            f"eigh {ratio_eigh:.1f}x, "
            f"Spectral {ratio_spectral:.1f}x, "
            f"Krylov {ratio_krylov:.1f}x"
        )


def main():
    parser = argparse.ArgumentParser(description="Scaling analysis benchmark")
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()
    scaling_analysis(args.dims)


if __name__ == "__main__":
    main()
