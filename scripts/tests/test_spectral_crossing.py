#!/usr/bin/env python3
"""Investigate Spectral d=12 vs d=16 crossing at fixed K.

From quickstart output, d=16 shows HIGHER P(unreachable) than d=12
at K=25. This script tests whether the crossing is statistical noise
or a real effect by running more trials at fixed K values.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import CanonicalQuditModel
from src.criteria import SpectralCriterion, Verdict


def compare_at_fixed_K(K, dims, n_trials=300, maxiter=100, restarts=5, seed=42):
    """Compare Spectral P at fixed K across dimensions."""
    print(f"\nSpectral at K={K}, {n_trials} trials, "
          f"maxiter={maxiter}, restarts={restarts}")
    print(f"{'d':>4} {'rho':>8} {'P':>8} {'SEM':>8} "
          f"{'mean_score':>12} {'std_score':>12}")
    print("-" * 58)

    for d in dims:
        model = CanonicalQuditModel(dim=d, seed=seed)
        if K > model.K:
            print(f"{d:4d}  (K={K} > K_max={model.K}, skipped)")
            continue
        phi = model.init_state()
        rng = np.random.default_rng(seed)

        count = 0
        scores = []

        for trial in range(n_trials):
            sub = model.sample_submodel(K, seed=int(rng.integers(0, 2**31)))
            psi = sub.random_state(rng=rng)

            sc = SpectralCriterion(sub, phi, psi, tau=0.99)
            result = sc.is_reachable(maxiter=maxiter, restarts=restarts)

            if result.verdict == Verdict.UNREACHABLE:
                count += 1
            scores.append(result.score)

        P = count / n_trials
        sem = np.sqrt(P * (1 - P) / n_trials) if 0 < P < 1 else 0
        rho = K / d**2
        print(f"{d:4d} {rho:8.4f} {P:8.3f} {sem:8.3f} "
              f"{np.mean(scores):12.4f} {np.std(scores):12.4f}")


def rho_matched_comparison(dims, rho_targets, n_trials=300,
                           maxiter=100, restarts=5, seed=42):
    """Compare Spectral P at matched rho values across dimensions."""
    print(f"\nRho-matched comparison, {n_trials} trials, "
          f"maxiter={maxiter}, restarts={restarts}")

    for d in dims:
        model = CanonicalQuditModel(dim=d, seed=seed)
        phi = model.init_state()

        print(f"\n  d={d}:")
        print(f"  {'K':>4} {'rho':>8} {'P':>8} {'SEM':>8}")
        print(f"  " + "-" * 32)

        for rho in rho_targets:
            K = max(2, int(rho * d**2))
            if K > model.K:
                continue
            rng = np.random.default_rng(seed)
            count = 0

            for trial in range(n_trials):
                sub = model.sample_submodel(K, seed=int(rng.integers(0, 2**31)))
                psi = sub.random_state(rng=rng)
                sc = SpectralCriterion(sub, phi, psi, tau=0.99)
                result = sc.is_reachable(maxiter=maxiter, restarts=restarts)
                if result.verdict == Verdict.UNREACHABLE:
                    count += 1

            P = count / n_trials
            sem = np.sqrt(P * (1 - P) / n_trials) if 0 < P < 1 else 0
            actual_rho = K / d**2
            print(f"  {K:4d} {actual_rho:8.4f} {P:8.3f} {sem:8.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n = 100 if args.quick else 300
    dims = [8, 12, 16]

    # Fixed-K comparison at the crossing point
    compare_at_fixed_K(K=25, dims=dims, n_trials=n, restarts=5)
    compare_at_fixed_K(K=35, dims=dims, n_trials=n, restarts=5)

    # Rho-matched comparison
    rho_matched_comparison(
        dims=dims,
        rho_targets=[0.05, 0.10, 0.15, 0.20, 0.25],
        n_trials=n,
        restarts=5,
    )
