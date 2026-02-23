#!/usr/bin/env python3
"""Investigate Krylov criterion at small K for QubitGrid.

At K=2-3, the optimization landscape has very few parameters.
This script tests whether the observed P(K=2) < P(K=3) fluctuation
is statistical noise or a real effect.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import QubitGridModel
from src.criteria import KrylovCriterion, Verdict


def test_krylov_small_K(d=16, K_values=None, n_trials=500,
                        restarts=5, maxiter=100, seed=42):
    """Test Krylov at small K with more trials and restarts."""
    if K_values is None:
        K_values = [2, 3, 4, 5, 6, 8, 10]

    model = QubitGridModel(dim=d, seed=seed)
    phi = model.init_state()

    K_values = [k for k in K_values if k <= model.K]

    print(f"Krylov Small-K Test: d={d}, {n_trials} trials, "
          f"restarts={restarts}, maxiter={maxiter}")
    print(f"{'K':>4} {'P(unreach)':>12} {'SEM':>8} {'mean_score':>12} {'std_score':>12}")
    print("-" * 52)

    rng = np.random.default_rng(seed)

    for K in K_values:
        count = 0
        scores = []

        for trial in range(n_trials):
            sub = model.sample_submodel(K, seed=int(rng.integers(0, 2**31)))
            psi = sub.random_state(rng=rng)

            kc = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)
            result = kc.is_reachable(maxiter=maxiter, restarts=restarts)

            if result.verdict == Verdict.UNREACHABLE:
                count += 1
            scores.append(result.score)

        P = count / n_trials
        sem = np.sqrt(P * (1 - P) / n_trials) if 0 < P < 1 else 0
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        print(f"{K:4d} {P:12.3f} {sem:8.3f} {mean_s:12.4f} {std_s:12.4f}")


def test_krylov_paired(d=16, K_values=None, n_hamiltonians=100,
                       n_targets=5, restarts=5, maxiter=100, seed=42):
    """Paired test: same operators, increasing K (superset)."""
    if K_values is None:
        K_values = [2, 3, 4, 5, 6]

    model = QubitGridModel(dim=d, seed=seed)
    phi = model.init_state()
    n_trials = n_hamiltonians * n_targets

    violations = {K: 0 for K in K_values[:-1]}
    counts = {K: 0 for K in K_values}

    rng = np.random.default_rng(seed)

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

    print(f"\nKrylov Paired Test: d={d}, {n_trials} trials, "
          f"restarts={restarts}")
    print(f"{'K':>4} {'P(unreach)':>12} {'violations':>12}")
    print("-" * 32)
    for K in K_values:
        P = counts[K] / n_trials
        v = violations.get(K, '-')
        print(f"{K:4d} {P:12.3f} {str(v):>12}")

    total = sum(violations.values())
    if total > 0:
        print(f"\n{total} paired violations — likely optimizer convergence issue")
    else:
        print(f"\nNo paired violations — Krylov is monotonic (as expected)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 100

    test_krylov_small_K(d=args.dim, n_trials=args.n_trials)
    test_krylov_paired(d=args.dim, n_hamiltonians=min(args.n_trials // 5, 50),
                       n_targets=5)
