"""
Performance profiling for reachability criteria.

Identifies bottlenecks and measures optimization impact.

Usage:
    python scripts/tests/profile_performance.py
"""

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

import sys
sys.path.insert(0, '../..')
from src.models import CanonicalQuditModel, QubitGridModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion
from src.sampling import DensitySweep, SweepConfig


def profile_single_criterion(CriterionClass, model, K, n_trials=100):
    """Profile a single criterion evaluation."""
    sub = model.sample_submodel(K)
    phi = model.init_state()

    times = []
    for _ in range(n_trials):
        psi = sub.random_state()
        crit = CriterionClass(sub, phi, psi, tau=0.99)

        start = time.perf_counter()
        if hasattr(crit, '_maximize'):
            crit.is_reachable(maxiter=50, restarts=2)
        else:
            crit.is_reachable()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times)


def profile_sweep(dim, K_values, n_trials=50):
    """Profile a full density sweep and return cProfile output."""
    config = SweepConfig(
        n_hamiltonians=5,
        n_targets=n_trials // 5,
        maxiter=50,
        restarts=2,
    )

    model = CanonicalQuditModel(dim=dim, seed=42)
    sweep = DensitySweep(model, config)

    profiler = cProfile.Profile()
    profiler.enable()

    sweep.run(K_values, criteria=['moment', 'spectral', 'krylov'], verbose=False)

    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)

    return stream.getvalue()


def benchmark_components():
    """Benchmark individual components across dimensions."""
    print("=" * 60)
    print("COMPONENT BENCHMARKS")
    print("=" * 60)

    dims = [8, 16, 32]
    K_fracs = [0.1, 0.2, 0.3]

    for d in dims:
        model = CanonicalQuditModel(dim=d, seed=42)
        print(f"\n--- d={d} ---")

        for frac in K_fracs:
            K = max(2, int(frac * d * d))
            if K > model.K:
                continue

            print(f"\nK={K} (rho={K/d**2:.2f}):")

            for name, Crit in [('Moment', MomentCriterion),
                               ('Spectral', SpectralCriterion),
                               ('Krylov', KrylovCriterion)]:
                mean_t, std_t = profile_single_criterion(Crit, model, K, n_trials=20)
                print(f"  {name:10s}: {mean_t*1000:8.2f} +/- {std_t*1000:5.2f} ms")


def identify_bottlenecks():
    """Identify the slowest parts of the code."""
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS (d=16, K=25)")
    print("=" * 60)

    profile_output = profile_sweep(dim=16, K_values=[10, 15, 20, 25], n_trials=50)
    print(profile_output)


def suggest_optimizations():
    """Print optimization suggestions based on profiling."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUGGESTIONS")
    print("=" * 60)

    print("""
    1. EIGENDECOMPOSITION (SpectralCriterion)
       - Currently uses scipy.linalg.eigh
       - For d>64, consider iterative methods (scipy.sparse.linalg.eigsh)

    2. ARNOLDI ITERATION (KrylovCriterion)
       - Current: Python loop with numpy operations
       - For large d: Cython/Numba for the inner loop

    3. L-BFGS-B OPTIMIZER
       - Early termination when score >= tau (already implemented)
       - Reduce restarts for production runs (2-3 sufficient)

    4. MOMENT Q MATRIX
       - Already vectorized (v47)
       - For very large K, consider sparse representations

    5. PARALLEL PROCESSING
       - Trials are independent -> embarrassingly parallel
       - Use joblib or multiprocessing for n_hamiltonians loop
       - Potential 4-8x speedup on modern CPUs
    """)


if __name__ == '__main__':
    benchmark_components()
    identify_bottlenecks()
    suggest_optimizations()
