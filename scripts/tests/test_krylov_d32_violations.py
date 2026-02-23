#!/usr/bin/env python3
"""Investigate Krylov d=32 non-monotonicity via paired tests.

At d=32, Krylov shows non-monotonic P(K) in quickstart output.
This script counts paired violations (K unreachable but K+1 reachable)
to distinguish optimizer convergence failures from real behavior.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import CanonicalQuditModel
from src.criteria import KrylovCriterion, Verdict


def test_krylov_paired_d32(d=32, K_values=None, n_hamiltonians=50,
                            n_targets=10, restarts=3, maxiter=100, seed=42):
    """Paired test: same operators, increasing K (superset) at d=32."""
    if K_values is None:
        # Focus on rho range near the transition
        K_values = [4, 8, 12, 16, 20, 24, 28, 32]

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    n_trials = n_hamiltonians * n_targets

    K_values = [k for k in K_values if k <= model.K]

    violations = {K: 0 for K in K_values[:-1]}
    counts = {K: 0 for K in K_values}

    rng = np.random.default_rng(seed)

    print(f"Krylov Paired Test: d={d}, {n_trials} trials, "
          f"restarts={restarts}, maxiter={maxiter}")

    for h in range(n_hamiltonians):
        K_max = max(K_values)
        sub = model.sample_submodel(K_max, seed=int(rng.integers(0, 2**31)))

        for t in range(n_targets):
            psi = sub.random_state(rng=rng)

            verdicts = {}
            for K in K_values:
                hams_K = sub.basis[:K]
                kc = KrylovCriterion(hams_K, phi, psi, tau=0.99, m=d)
                result = kc.is_reachable(maxiter=maxiter, restarts=restarts)
                is_unreach = result.verdict == Verdict.UNREACHABLE
                verdicts[K] = is_unreach
                if is_unreach:
                    counts[K] += 1

            for i, K in enumerate(K_values[:-1]):
                K_next = K_values[i + 1]
                if verdicts[K] and not verdicts[K_next]:
                    violations[K] += 1

        if (h + 1) % 10 == 0:
            print(f"  {h+1}/{n_hamiltonians} hamiltonians done")

    print(f"\n{'K':>4} {'P(unreach)':>12} {'violations':>12} {'violation%':>12}")
    print("-" * 44)
    for K in K_values:
        P = counts[K] / n_trials
        v = violations.get(K, '-')
        pct = f"{violations[K]/n_trials*100:.1f}%" if K in violations else '-'
        print(f"{K:4d} {P:12.3f} {str(v):>12} {pct:>12}")

    total = sum(violations.values())
    print(f"\nTotal violations: {total}/{n_trials} ({total/n_trials*100:.1f}%)")
    return total


def test_krylov_restarts_comparison(d=32, K_values=None, n_hamiltonians=30,
                                     n_targets=10, seed=42):
    """Compare violation counts with different restart/maxiter settings."""
    if K_values is None:
        K_values = [8, 12, 16, 20, 24]

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    n_trials = n_hamiltonians * n_targets

    K_values = [k for k in K_values if k <= model.K]

    configs = [
        ("default (r=3, m=100)", 3, 100),
        ("more restarts (r=8, m=100)", 8, 100),
        ("more maxiter (r=3, m=300)", 3, 300),
        ("both (r=8, m=300)", 8, 300),
    ]

    print(f"\nRestart/Maxiter Comparison: d={d}, {n_trials} trials")
    print(f"{'Config':>35}  {'violations':>10}  {'violation%':>10}")
    print("-" * 60)

    rng_base = np.random.default_rng(seed)
    # Pre-generate seeds for reproducibility across configs
    ham_seeds = [int(rng_base.integers(0, 2**31)) for _ in range(n_hamiltonians)]

    for label, restarts, maxiter in configs:
        total_violations = 0

        for h in range(n_hamiltonians):
            rng = np.random.default_rng(ham_seeds[h])
            K_max = max(K_values)
            sub = model.sample_submodel(K_max, seed=ham_seeds[h])

            for t in range(n_targets):
                psi = sub.random_state(rng=rng)
                verdicts = {}

                for K in K_values:
                    hams_K = sub.basis[:K]
                    kc = KrylovCriterion(hams_K, phi, psi, tau=0.99, m=d)
                    result = kc.is_reachable(maxiter=maxiter, restarts=restarts)
                    verdicts[K] = result.verdict == Verdict.UNREACHABLE

                for i, K in enumerate(K_values[:-1]):
                    K_next = K_values[i + 1]
                    if verdicts[K] and not verdicts[K_next]:
                        total_violations += 1

        pct = total_violations / n_trials * 100
        print(f"{label:>35}  {total_violations:>10}  {pct:>9.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        test_krylov_paired_d32(n_hamiltonians=20, n_targets=5)
        test_krylov_restarts_comparison(n_hamiltonians=10, n_targets=5)
    else:
        test_krylov_paired_d32()
        test_krylov_restarts_comparison()
