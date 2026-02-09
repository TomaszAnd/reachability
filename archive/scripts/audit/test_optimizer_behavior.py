#!/usr/bin/env python3
"""
Audit Test 1: Optimizer Behavior for GEO2LOCAL vs Canonical ensembles.

Investigates:
1. Does optimization actually improve scores over random lambda?
2. How severe is the under-optimization at high dimensions?
3. Does GEO2's aggressive settings (maxiter=20, 1 restart) hurt?

Key hypothesis: GEO2 anomaly (lower d drops first) is caused by
GEO2_MAXITER=20 being catastrophically insufficient for K=O(d^2) parameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from reach import models, optimize, mathematics, settings


def test_optimization_improvement(ensemble, d, K, n_trials=15, **ensemble_params):
    """Test if optimization actually improves over random sampling."""
    rng = np.random.RandomState(42)

    random_scores = []
    opt_scores_default = []
    opt_scores_boosted = []

    for trial in range(n_trials):
        seed_base = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed_base)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed_base + 1, **ensemble_params)

        # Random lambda score
        random_lambda = rng.uniform(-1, 1, K)
        random_score = mathematics.spectral_overlap(random_lambda, psi, phi, hams)
        random_scores.append(random_score)

        # Optimized with ensemble-specific settings (GEO2: maxiter=20, restarts=1)
        opt_settings = settings.get_optimization_settings(ensemble)
        result_default = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=opt_settings['restarts'],
            maxiter=opt_settings['maxiter'],
            ftol=opt_settings['ftol'],
            seed=seed_base + 2,
        )
        opt_scores_default.append(result_default['best_value'])

        # Optimized with BOOSTED settings (maxiter=200, restarts=5)
        result_boosted = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=5,
            maxiter=200,
            ftol=1e-8,
            seed=seed_base + 3,
        )
        opt_scores_boosted.append(result_boosted['best_value'])

    return {
        'random': np.array(random_scores),
        'default': np.array(opt_scores_default),
        'boosted': np.array(opt_scores_boosted),
    }


def main():
    print("=" * 70)
    print("AUDIT TEST 1: Optimizer Improvement Analysis")
    print("=" * 70)
    print(f"\nGEO2 settings: maxiter={settings.GEO2_MAXITER}, restarts={settings.GEO2_RESTARTS}, ftol={settings.GEO2_FTOL}")
    print(f"Default settings: maxiter={settings.DEFAULT_MAXITER}, restarts={settings.DEFAULT_RESTARTS}, ftol={settings.DEFAULT_FTOL}")

    # GEO2 lattice configs matching production
    geo2_configs = {
        16: {'nx': 2, 'ny': 2},
        32: {'nx': 1, 'ny': 5},
    }

    rho = 0.05  # Test at fixed density

    for ensemble_label, ensemble_name, configs in [
        ("GEO2LOCAL", "GEO2", geo2_configs),
        ("Canonical", "canonical", {16: {}, 32: {}}),
    ]:
        print(f"\n{'=' * 70}")
        print(f"ENSEMBLE: {ensemble_label}")
        print(f"{'=' * 70}")

        for d in sorted(configs.keys()):
            K = max(2, round(rho * d**2))
            params = configs[d]

            print(f"\n  d={d}, K={K} (rho={K/d**2:.4f}):")

            t0 = time.time()
            results = test_optimization_improvement(
                ensemble_name, d, K, n_trials=12, **params
            )
            elapsed = time.time() - t0

            rand_mean = np.mean(results['random'])
            default_mean = np.mean(results['default'])
            boosted_mean = np.mean(results['boosted'])

            improvement_default = default_mean - rand_mean
            improvement_boosted = boosted_mean - rand_mean
            gap = boosted_mean - default_mean

            print(f"    Random lambda:     {rand_mean:.4f} +/- {np.std(results['random']):.4f}")
            print(f"    Default optimizer: {default_mean:.4f} +/- {np.std(results['default']):.4f}  (improvement: {improvement_default:+.4f})")
            print(f"    Boosted optimizer: {boosted_mean:.4f} +/- {np.std(results['boosted']):.4f}  (improvement: {improvement_boosted:+.4f})")
            print(f"    Gap (boosted - default): {gap:+.4f}")
            print(f"    Time: {elapsed:.1f}s")

            if gap > 0.05:
                print(f"    >>> SIGNIFICANT GAP: Default optimizer is under-performing! <<<")
            elif gap > 0.01:
                print(f"    >>> MODERATE GAP: Default optimizer may be suboptimal.")
            else:
                print(f"    >>> GAP IS SMALL: Default optimizer is adequate.")


if __name__ == "__main__":
    main()
