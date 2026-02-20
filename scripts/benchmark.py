#!/usr/bin/env python3
"""
Benchmark suite for performance comparison.

Measures per-evaluation and per-is_reachable timings across dimensions
for both Canonical and QubitGrid models.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --dims 16 32 64 128
    python scripts/benchmark.py --quick
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import numpy as np

from src.models import CanonicalQuditModel, QubitGridModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion


def benchmark_single_eval(model_cls, d, K, n_trials=20):
    """Benchmark single criterion.evaluate() call."""
    model = model_cls(dim=d, seed=42)
    sub = model.sample_submodel(k=min(K, model.K), seed=0)
    phi = sub.init_state()
    psi = sub.random_state()
    lambdas = np.random.RandomState(42).uniform(-1, 1, sub.K)

    results = {}

    # Spectral
    sc = SpectralCriterion(sub, phi, psi, tau=0.99)
    sc.evaluate(lambdas, return_gradient=True)  # warmup
    t0 = time.time()
    for _ in range(n_trials):
        sc.evaluate(lambdas, return_gradient=True)
    results['spectral_eval_ms'] = (time.time() - t0) / n_trials * 1000

    # Krylov
    kc = KrylovCriterion(sub, phi, psi, tau=0.99)
    kc.evaluate(lambdas, return_gradient=True)  # warmup
    t0 = time.time()
    for _ in range(n_trials):
        kc.evaluate(lambdas, return_gradient=True)
    results['krylov_eval_ms'] = (time.time() - t0) / n_trials * 1000

    # Moment
    mc = MomentCriterion(sub, phi, psi, tau=0.99)
    mc.evaluate()  # warmup
    t0 = time.time()
    for _ in range(n_trials):
        mc.evaluate()
    results['moment_eval_ms'] = (time.time() - t0) / n_trials * 1000

    return results


def benchmark_is_reachable(model_cls, d, K, n_trials=5, maxiter=50, restarts=2):
    """Benchmark full is_reachable() call."""
    model = model_cls(dim=d, seed=42)
    sub = model.sample_submodel(k=min(K, model.K), seed=0)
    phi = sub.init_state()
    psi = sub.random_state()

    results = {}

    # Spectral
    sc = SpectralCriterion(sub, phi, psi, tau=0.99)
    sc.is_reachable(maxiter=maxiter, restarts=restarts)  # warmup
    t0 = time.time()
    for _ in range(n_trials):
        sc.is_reachable(maxiter=maxiter, restarts=restarts)
    results['spectral_reach_ms'] = (time.time() - t0) / n_trials * 1000

    # Krylov
    kc = KrylovCriterion(sub, phi, psi, tau=0.99)
    kc.is_reachable(maxiter=maxiter, restarts=restarts)  # warmup
    t0 = time.time()
    for _ in range(n_trials):
        kc.is_reachable(maxiter=maxiter, restarts=restarts)
    results['krylov_reach_ms'] = (time.time() - t0) / n_trials * 1000

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark suite')
    parser.add_argument('--dims', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--quick', action='store_true', help='Fewer trials')
    parser.add_argument('--model', choices=['canonical', 'qubitgrid', 'both'],
                        default='both')
    args = parser.parse_args()

    n_eval = 5 if args.quick else 20
    n_reach = 2 if args.quick else 5

    models_to_test = []
    if args.model in ('canonical', 'both'):
        models_to_test.append(('Canonical', CanonicalQuditModel))
    if args.model in ('qubitgrid', 'both'):
        models_to_test.append(('QubitGrid', QubitGridModel))

    print("=" * 80)
    print("REACHABILITY BENCHMARK")
    print("=" * 80)

    for model_name, model_cls in models_to_test:
        print(f"\n--- {model_name} ---")
        print(f"{'d':>6} {'K':>6} {'Spec eval':>12} {'Krylov eval':>12} "
              f"{'Moment eval':>12} {'Spec reach':>12} {'Krylov reach':>12}")
        print("-" * 80)

        for d in args.dims:
            if model_name == 'QubitGrid':
                from src.models import LATTICE_CONFIGS
                if d not in LATTICE_CONFIGS:
                    print(f"{d:>6}   (no lattice config)")
                    continue

            K = d // 2
            try:
                eval_res = benchmark_single_eval(model_cls, d, K, n_eval)
                reach_res = benchmark_is_reachable(model_cls, d, K, n_reach)

                print(f"{d:>6} {K:>6} "
                      f"{eval_res['spectral_eval_ms']:>10.1f}ms "
                      f"{eval_res['krylov_eval_ms']:>10.1f}ms "
                      f"{eval_res['moment_eval_ms']:>10.1f}ms "
                      f"{reach_res['spectral_reach_ms']:>10.0f}ms "
                      f"{reach_res['krylov_reach_ms']:>10.0f}ms")
            except Exception as e:
                print(f"{d:>6}   ERROR: {e}")

    # Sparsity report for QubitGrid
    if args.model in ('qubitgrid', 'both'):
        print("\n--- QubitGrid Sparsity ---")
        from src.models import LATTICE_CONFIGS
        for d in args.dims:
            if d in LATTICE_CONFIGS:
                model = QubitGridModel(dim=d, seed=42)
                sp = model.basis_sparse
                if sp:
                    nnz = sp[0].nnz
                    print(f"  d={d}: nnz={nnz}/{d*d} ({nnz/(d*d)*100:.1f}% nonzero)")


if __name__ == '__main__':
    main()
