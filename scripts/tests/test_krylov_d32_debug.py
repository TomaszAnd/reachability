#!/usr/bin/env python3
"""Debug high Krylov violation rate at d=32.

85% paired violations at d=32 with r=8, m=300 suggests this may NOT be
optimizer convergence. Investigate:
1. Are violations due to score being very close to tau?
2. Does the score actually decrease when adding operators?
3. What are the actual score differences?
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import CanonicalQuditModel
from src.criteria import KrylovCriterion, Verdict


def debug_violations(d=32, K_values=None, n_hamiltonians=20,
                     n_targets=5, restarts=8, maxiter=300, seed=42):
    """Track actual scores at K and K+1 for each violation."""
    if K_values is None:
        K_values = [8, 12, 16, 20, 24, 28]

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()

    K_values = [k for k in K_values if k <= model.K]

    rng = np.random.default_rng(seed)
    tau = 0.99

    violations = []
    total_pairs = 0

    for h in range(n_hamiltonians):
        K_max = max(K_values) + 4  # Need extra for K+4
        if K_max > model.K:
            K_max = model.K
        sub = model.sample_submodel(K_max, seed=int(rng.integers(0, 2**31)))

        for t in range(n_targets):
            psi = sub.random_state(rng=rng)

            scores = {}
            for K in K_values:
                hams_K = sub.basis[:K]
                kc = KrylovCriterion(hams_K, phi, psi, tau=tau, m=d)
                result = kc.is_reachable(maxiter=maxiter, restarts=restarts)
                scores[K] = result.score

            for i, K in enumerate(K_values[:-1]):
                K_next = K_values[i + 1]
                total_pairs += 1
                is_unreach_K = scores[K] < tau
                is_unreach_Kn = scores[K_next] < tau
                if is_unreach_K and not is_unreach_Kn:
                    violations.append({
                        'K': K, 'K_next': K_next,
                        'score_K': scores[K], 'score_Knext': scores[K_next],
                        'delta': scores[K_next] - scores[K],
                    })

    print(f"Debug Krylov d={d}: {len(violations)}/{total_pairs} violations "
          f"({len(violations)/total_pairs*100:.1f}%)")
    print(f"restarts={restarts}, maxiter={maxiter}\n")

    if violations:
        deltas = [v['delta'] for v in violations]
        scores_K = [v['score_K'] for v in violations]
        scores_Kn = [v['score_Knext'] for v in violations]
        print(f"Score at K (violations only):")
        print(f"  min={min(scores_K):.4f}, max={max(scores_K):.4f}, "
              f"mean={np.mean(scores_K):.4f}")
        print(f"Score at K+1 (violations only):")
        print(f"  min={min(scores_Kn):.4f}, max={max(scores_Kn):.4f}, "
              f"mean={np.mean(scores_Kn):.4f}")
        print(f"Score delta (K+1 - K):")
        print(f"  min={min(deltas):.4f}, max={max(deltas):.4f}, "
              f"mean={np.mean(deltas):.4f}")

        # How many violations have score_K very close to tau?
        near_tau = sum(1 for v in violations if v['score_K'] > 0.95)
        print(f"\nViolations with score_K > 0.95: {near_tau}/{len(violations)}")

        # Distribution of score_K
        bins = [0, 0.5, 0.8, 0.9, 0.95, 0.99]
        print(f"\nScore_K distribution in violations:")
        for i in range(len(bins) - 1):
            count = sum(1 for v in violations if bins[i] <= v['score_K'] < bins[i+1])
            print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count}")

        # Show first 10 violations
        print(f"\nFirst 10 violations:")
        for v in violations[:10]:
            print(f"  K={v['K']:3d}→{v['K_next']:3d}: "
                  f"score {v['score_K']:.4f}→{v['score_Knext']:.4f} "
                  f"(delta={v['delta']:+.4f})")


if __name__ == "__main__":
    debug_violations()
