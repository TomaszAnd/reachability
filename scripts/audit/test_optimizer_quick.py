#!/usr/bin/env python3
"""
Quick optimizer audit - d=16 only, fewer trials.
Tests default vs boosted optimization for GEO2 and Canonical.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from reach import models, optimize, mathematics, settings


def test_improvement(ensemble, d, K, n_trials=8, **params):
    """Compare random, default-optimized, and boosted-optimized scores."""
    rng = np.random.RandomState(42)
    random_s, default_s, boosted_s = [], [], []
    opt_settings = settings.get_optimization_settings(ensemble)

    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed + 1, **params)

        # Random
        lam = rng.uniform(-1, 1, K)
        random_s.append(mathematics.spectral_overlap(lam, psi, phi, hams))

        # Default settings
        r1 = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=opt_settings['restarts'],
            maxiter=opt_settings['maxiter'],
            seed=seed + 2,
        )
        default_s.append(r1['best_value'])

        # Boosted (3 restarts, 100 iterations)
        r2 = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=3,
            maxiter=100,
            seed=seed + 3,
        )
        boosted_s.append(r2['best_value'])

    return np.array(random_s), np.array(default_s), np.array(boosted_s)


def main():
    print("=" * 70)
    print("QUICK OPTIMIZER AUDIT")
    print(f"GEO2 settings: maxiter={settings.GEO2_MAXITER}, restarts={settings.GEO2_RESTARTS}")
    print(f"Default settings: maxiter={settings.DEFAULT_MAXITER}, restarts={settings.DEFAULT_RESTARTS}")
    print("=" * 70)

    configs = [
        ("GEO2 d=16", "GEO2", 16, 0.05, {'nx': 2, 'ny': 2}),
        ("GEO2 d=16", "GEO2", 16, 0.10, {'nx': 2, 'ny': 2}),
        ("Canonical d=16", "canonical", 16, 0.05, {}),
        ("Canonical d=16", "canonical", 16, 0.10, {}),
    ]

    for label, ensemble, d, rho, params in configs:
        K = max(2, round(rho * d**2))
        print(f"\n{label}, K={K} (rho={K/d**2:.4f}):")

        t0 = time.time()
        rand, default, boosted = test_improvement(ensemble, d, K, n_trials=8, **params)
        elapsed = time.time() - t0

        gap = np.mean(boosted) - np.mean(default)
        print(f"  Random:  {np.mean(rand):.4f} +/- {np.std(rand):.4f}")
        print(f"  Default: {np.mean(default):.4f} +/- {np.std(default):.4f}")
        print(f"  Boosted: {np.mean(boosted):.4f} +/- {np.std(boosted):.4f}")
        print(f"  Gap (boosted-default): {gap:+.4f}  [{elapsed:.1f}s]")
        if gap > 0.05:
            print(f"  >>> SIGNIFICANT: default optimizer is under-performing!")


if __name__ == "__main__":
    main()
