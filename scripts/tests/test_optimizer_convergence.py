#!/usr/bin/env python3
"""Test that Spectral and Krylov paired scores are monotonic in K.

Mathematics guarantees: max_{lambda in R^{K+1}} score(lambda) >= max_{lambda in R^K} score(lambda)
because setting lambda_{K+1}=0 recovers the K-dimensional solution.

Any paired violation (score at K > score at K+1) indicates optimizer convergence failure.
This test verifies the optimizer finds reliable maxima across dimensions.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import CanonicalQuditModel
from src.criteria import SpectralCriterion, KrylovCriterion, Verdict


def test_monotonicity_paired(criterion_cls, d=16, K_values=None,
                              n_hamiltonians=20, n_targets=5,
                              restarts=None, maxiter=200, seed=42):
    """
    Paired test: for each trial, use the SAME operators and target.
    Score at K must be <= score at K+4 (mathematically required).

    Returns: number of score violations (should be 0 for correct optimizer)
    """
    if K_values is None:
        K_max = min(d * d // 4, 60)
        K_values = list(range(4, K_max, 4))

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    rng = np.random.default_rng(seed)

    # If restarts not specified, let the adaptive logic in _maximize handle it
    if restarts is None:
        restarts = 3  # Low base value; adaptive logic will increase as needed

    violations = 0
    violation_details = []
    total_pairs = 0

    for h in range(n_hamiltonians):
        K_max_val = max(K_values)
        if K_max_val > model.K:
            K_max_val = model.K
        sub = model.sample_submodel(K_max_val, seed=int(rng.integers(0, 2**31)))

        for t in range(n_targets):
            psi = sub.random_state(rng=rng)

            scores = {}
            for K in K_values:
                if K > len(sub.basis):
                    continue
                hams_K = sub.basis[:K]
                kwargs = {'tau': 0.99}
                if criterion_cls == KrylovCriterion:
                    kwargs['m'] = d
                crit = criterion_cls(hams_K, phi, psi, **kwargs)
                result = crit.is_reachable(maxiter=maxiter, restarts=restarts)
                scores[K] = result.score

            sorted_Ks = sorted(scores.keys())
            for i in range(len(sorted_Ks) - 1):
                K = sorted_Ks[i]
                K_next = sorted_Ks[i + 1]
                total_pairs += 1
                # Score must not decrease (allowing tiny numerical tolerance)
                if scores[K] > scores[K_next] + 1e-6:
                    violations += 1
                    violation_details.append({
                        'K': K, 'K_next': K_next,
                        'score_K': scores[K], 'score_Knext': scores[K_next],
                        'diff': scores[K] - scores[K_next],
                    })

    name = criterion_cls.__name__
    print(f"\n{name} d={d}: {violations} score violations in {total_pairs} pairs")
    if violations > 0:
        print(f"  Violation rate: {violations/total_pairs*100:.1f}%")
        print("  Sample violations:")
        for v in violation_details[:5]:
            print(f"    K={v['K']}→{v['K_next']}: "
                  f"score {v['score_K']:.4f} > {v['score_Knext']:.4f} "
                  f"(diff={v['diff']:.6f})")
    else:
        print("  No violations — optimizer is monotonic ✓")
    return violations


def test_verdict_monotonicity(criterion_cls, d=16, K_values=None,
                               n_hamiltonians=20, n_targets=5,
                               restarts=None, maxiter=200, seed=42):
    """
    Paired verdict test: K=UNREACHABLE but K+4=REACHABLE is a violation.

    This is the user-visible failure mode: adding operators flips the verdict
    from UNREACHABLE to REACHABLE.
    """
    if K_values is None:
        K_max = min(d * d // 4, 60)
        K_values = list(range(4, K_max, 4))

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    rng = np.random.default_rng(seed)

    if restarts is None:
        restarts = 3

    verdict_violations = 0
    total_pairs = 0

    for h in range(n_hamiltonians):
        K_max_val = max(K_values)
        if K_max_val > model.K:
            K_max_val = model.K
        sub = model.sample_submodel(K_max_val, seed=int(rng.integers(0, 2**31)))

        for t in range(n_targets):
            psi = sub.random_state(rng=rng)

            verdicts = {}
            for K in K_values:
                if K > len(sub.basis):
                    continue
                hams_K = sub.basis[:K]
                kwargs = {'tau': 0.99}
                if criterion_cls == KrylovCriterion:
                    kwargs['m'] = d
                crit = criterion_cls(hams_K, phi, psi, **kwargs)
                result = crit.is_reachable(maxiter=maxiter, restarts=restarts)
                verdicts[K] = result.verdict

            sorted_Ks = sorted(verdicts.keys())
            for i in range(len(sorted_Ks) - 1):
                K = sorted_Ks[i]
                K_next = sorted_Ks[i + 1]
                total_pairs += 1
                # UNREACHABLE at K but REACHABLE at K+step is a verdict violation
                if (verdicts[K] == Verdict.UNREACHABLE and
                        verdicts[K_next] == Verdict.REACHABLE):
                    verdict_violations += 1

    name = criterion_cls.__name__
    print(f"\n{name} d={d}: {verdict_violations} verdict violations in {total_pairs} pairs")
    if verdict_violations > 0:
        print(f"  Verdict violation rate: {verdict_violations/total_pairs*100:.1f}%")
    else:
        print("  No verdict violations — monotonicity preserved ✓")
    return verdict_violations


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: fewer trials")
    parser.add_argument("--dims", type=int, nargs="+", default=[8, 16, 32],
                        help="Dimensions to test")
    args = parser.parse_args()

    all_pass = True

    for d in args.dims:
        print(f"\n{'='*60}")
        print(f"Testing d={d}")
        print(f"{'='*60}")

        if args.quick:
            n_h, n_t = (10, 3) if d <= 16 else (5, 3)
        else:
            n_h, n_t = (20, 5) if d <= 16 else (10, 5)

        K_max = min(d * d // 4, 60)
        K_values = list(range(4, K_max, 4))

        for cls in [SpectralCriterion, KrylovCriterion]:
            sv = test_monotonicity_paired(
                cls, d=d, K_values=K_values,
                n_hamiltonians=n_h, n_targets=n_t, maxiter=200)

            vv = test_verdict_monotonicity(
                cls, d=d, K_values=K_values,
                n_hamiltonians=n_h, n_targets=n_t, maxiter=200)

            if sv > 0 or vv > 0:
                all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL CONVERGENCE TESTS PASSED ✓")
    else:
        print("*** CONVERGENCE ISSUES DETECTED ***")
    print(f"{'='*60}")
