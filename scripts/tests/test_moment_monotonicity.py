#!/usr/bin/env python3
"""Investigate Moment criterion non-monotonicity in K.

The Moment criterion tests a SUFFICIENT condition for unreachability:
whether Q + gamma*LL^T is positive (or negative) definite.

Unlike Spectral/Krylov (which maximize over lambda and are guaranteed
monotonic in K), the Moment criterion's power is NOT guaranteed monotonic
because adding operators changes the structure of Q and L.

This script verifies this by:
1. Running many trials at consecutive K values
2. Checking paired trials (same Hamiltonians/targets, different K)
3. Counting cases where smaller K finds unreachable but larger K misses
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import CanonicalQuditModel
from src.criteria import MomentCriterion, Verdict


def test_moment_paired(d=16, K_values=None, n_hamiltonians=100,
                       n_targets=10, seed=42):
    """Test Moment monotonicity using paired trials.

    For each Hamiltonian sample, we test with K and K+delta operators
    using the SAME operators (K+delta is a superset of K).
    """
    if K_values is None:
        K_values = list(range(2, min(20, d*d)))

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()

    n_trials = n_hamiltonians * n_targets
    counts = {K: 0 for K in K_values}
    # Track paired violations: K finds unreachable but K+1 misses
    violations = {K: 0 for K in K_values[:-1]}

    rng = np.random.default_rng(seed)

    for h in range(n_hamiltonians):
        K_max = max(K_values)
        sub = model.sample_submodel(K_max, seed=int(rng.integers(0, 2**31)))

        for t in range(n_targets):
            psi = sub.random_state(rng=rng)

            verdicts = {}
            for K in K_values:
                hams_K = sub.basis[:K]
                mc = MomentCriterion(hams_K, phi, psi, tau=0.99)
                result = mc.is_reachable()
                is_unreach = result.verdict == Verdict.UNREACHABLE
                verdicts[K] = is_unreach
                if is_unreach:
                    counts[K] += 1

            # Check paired violations
            for i, K in enumerate(K_values[:-1]):
                K_next = K_values[i + 1]
                if verdicts[K] and not verdicts[K_next]:
                    violations[K] += 1

    print(f"Moment Monotonicity Test: d={d}, {n_trials} trials")
    print(f"{'K':>4} {'P(unreach)':>12} {'SEM':>8} {'violations':>12}")
    print("-" * 40)
    for K in K_values:
        P = counts[K] / n_trials
        sem = np.sqrt(P * (1 - P) / n_trials) if 0 < P < 1 else 0
        v = violations.get(K, '-')
        print(f"{K:4d} {P:12.3f} {sem:8.3f} {str(v):>12}")

    total_violations = sum(violations.values())
    print(f"\nTotal paired violations (K finds, K+1 misses): {total_violations}")
    if total_violations > 0:
        print("=> Moment criterion is NOT monotonic in K (confirmed)")
    else:
        print("=> No violations found (monotonicity holds for this sample)")

    return counts, violations


def test_moment_independent(d=16, K_values=None, n_hamiltonians=50,
                            n_targets=20, seed=42):
    """Test Moment with independent submodel sampling (like DensitySweep).

    This matches the notebook methodology where each K uses independently
    sampled submodels.
    """
    if K_values is None:
        K_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    n_trials = n_hamiltonians * n_targets

    print(f"\nIndependent Sampling Test: d={d}, {n_trials} trials per K")
    print(f"{'K':>4} {'P(unreach)':>12} {'SEM':>8}")
    print("-" * 28)

    for K in K_values:
        if K > model.K:
            continue
        count = 0
        rng = np.random.default_rng(seed + K)
        for h in range(n_hamiltonians):
            sub = model.sample_submodel(K, seed=int(rng.integers(0, 2**31)))
            for t in range(n_targets):
                psi = sub.random_state(rng=rng)
                mc = MomentCriterion(sub, phi, psi, tau=0.99)
                result = mc.is_reachable()
                if result.verdict == Verdict.UNREACHABLE:
                    count += 1
        P = count / n_trials
        sem = np.sqrt(P * (1 - P) / n_trials) if 0 < P < 1 else 0
        print(f"{K:4d} {P:12.3f} {sem:8.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--n-hamiltonians", type=int, default=100)
    parser.add_argument("--n-targets", type=int, default=10)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (fewer trials)")
    args = parser.parse_args()

    if args.quick:
        args.n_hamiltonians = 30
        args.n_targets = 5

    test_moment_paired(d=args.dim, n_hamiltonians=args.n_hamiltonians,
                       n_targets=args.n_targets)
    test_moment_independent(d=args.dim,
                            n_hamiltonians=args.n_hamiltonians // 2,
                            n_targets=args.n_targets * 2)
