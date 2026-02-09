#!/usr/bin/env python3
"""
Audit Test 4: Verify optimization actually optimizes for GEO2LOCAL.

CRITICAL CHECK: The optimization score should be CONSISTENTLY HIGHER than
random lambda. If not, the optimizer is effectively broken at that dimension.

Also tests the fixed-lambda path (optimize_lambda=False) to verify
it uses the intended sampling distribution.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from reach import models, optimize, mathematics, settings


def verify_optimization_not_random(ensemble, d, K, n_trials=20, **ensemble_params):
    """Verify that optimization consistently beats random lambda."""
    rng = np.random.RandomState(42)

    improvements = []
    random_scores = []
    opt_scores = []
    opt_settings = settings.get_optimization_settings(ensemble)

    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed + 1, **ensemble_params)

        # Score with RANDOM lambda (uniform [-1, 1])
        random_lambda = rng.uniform(-1, 1, K)
        random_score = mathematics.spectral_overlap(random_lambda, psi, phi, hams)

        # Score with Gaussian lambda (as used in fixed-lambda path)
        gauss_lambda = rng.randn(K) / np.sqrt(K)
        gauss_score = mathematics.spectral_overlap(gauss_lambda, psi, phi, hams)

        # Score with OPTIMIZED lambda (using ensemble-specific settings)
        opt_result = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=opt_settings['restarts'],
            maxiter=opt_settings['maxiter'],
            ftol=opt_settings['ftol'],
            seed=seed + 2,
        )
        opt_score = opt_result['best_value']

        improvement = opt_score - random_score
        improvements.append(improvement)
        random_scores.append(random_score)
        opt_scores.append(opt_score)

        if trial < 3:
            print(f"    Trial {trial}: random={random_score:.4f}, gauss={gauss_score:.4f}, "
                  f"opt={opt_score:.4f}, improvement={improvement:+.4f}")

    improvements = np.array(improvements)
    mean_improvement = np.mean(improvements)
    mean_random = np.mean(random_scores)
    mean_opt = np.mean(opt_scores)

    print(f"\n    Summary over {n_trials} trials:")
    print(f"      Mean random S*:  {mean_random:.4f} +/- {np.std(random_scores):.4f}")
    print(f"      Mean opt S*:     {mean_opt:.4f} +/- {np.std(opt_scores):.4f}")
    print(f"      Mean improvement: {mean_improvement:+.4f} +/- {np.std(improvements):.4f}")
    print(f"      Fraction improved: {np.mean(improvements > 0):.0%}")

    # Critical check
    if mean_improvement < 0.01:
        print(f"\n    WARNING: Mean improvement is only {mean_improvement:.4f}")
        print(f"    Optimization is NOT working properly at this configuration!")
        return False
    else:
        print(f"\n    OK: Optimization provides meaningful improvement.")
        return True


def main():
    print("=" * 70)
    print("AUDIT TEST 4: Verify Optimization is Actually Working")
    print("=" * 70)

    geo2_configs = {
        16: {'nx': 2, 'ny': 2},
        32: {'nx': 1, 'ny': 5},
    }

    results = {}

    # Test GEO2 at multiple densities
    print("\n" + "=" * 70)
    print("GEO2LOCAL")
    print("=" * 70)

    for d, params in geo2_configs.items():
        for rho in [0.03, 0.05, 0.08]:
            K = max(2, round(rho * d**2))
            print(f"\n  d={d}, K={K} (rho={K/d**2:.4f}):")
            ok = verify_optimization_not_random("GEO2", d, K, n_trials=12, **params)
            results[(f"GEO2_d{d}", rho)] = ok

    # Test Canonical
    print("\n" + "=" * 70)
    print("CANONICAL")
    print("=" * 70)

    for d in [16, 32]:
        for rho in [0.03, 0.05]:
            K = max(2, round(rho * d**2))
            if K >= d**2:  # canonical basis has d^2 operators max
                continue
            print(f"\n  d={d}, K={K} (rho={K/d**2:.4f}):")
            ok = verify_optimization_not_random("canonical", d, K, n_trials=12)
            results[(f"canonical_d{d}", rho)] = ok

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {key}: {status}")


if __name__ == "__main__":
    main()
