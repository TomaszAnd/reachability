#!/usr/bin/env python3
"""
Audit Test 2: Validate criteria against PROVABLY REACHABLE targets.

Method: Generate target = exp(-i H(lambda) t) |psi> for known lambda.
This target MUST be reachable. If criteria say unreachable, there is a bug.

Tests both GEO2 and Canonical ensembles.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.linalg import expm
import qutip
from reach import models, optimize, mathematics, settings


def generate_reachable_target(psi, hams, t=1.0, lambdas=None):
    """Generate target state reachable by construction.

    target = exp(-i H(lambda) t) |psi>
    """
    K = len(hams)
    if lambdas is None:
        lambdas = np.random.uniform(-1, 1, K)

    # Build H(lambda) = sum lambda_k H_k
    H_lambda = sum(l * H for l, H in zip(lambdas, hams))
    H_matrix = H_lambda.full()

    # Evolve
    psi_arr = psi.full().flatten()
    U = expm(-1j * H_matrix * t)
    target_arr = U @ psi_arr

    # Convert to Qobj and normalize
    target = qutip.Qobj(target_arr.reshape(-1, 1))
    target = target.unit()

    return target, lambdas


def validate_criteria_on_reachable(ensemble, d, K, n_trials=30, tau=0.99, **ensemble_params):
    """Test that criteria correctly identify provably reachable states."""
    rng = np.random.RandomState(42)

    false_unreachable = {'spectral_default': 0, 'spectral_boosted': 0, 'krylov_default': 0, 'krylov_boosted': 0}
    spectral_scores_default = []
    spectral_scores_boosted = []
    krylov_scores_default = []
    krylov_scores_boosted = []

    opt_settings = settings.get_optimization_settings(ensemble)

    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **ensemble_params)

        # Generate PROVABLY REACHABLE target
        np.random.seed(seed + 100)
        phi, true_lambda = generate_reachable_target(psi, hams, t=1.0)

        # Test Spectral with default settings
        spec_default = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=opt_settings['restarts'],
            maxiter=opt_settings['maxiter'],
            seed=seed + 200,
        )
        spectral_scores_default.append(spec_default['best_value'])
        if spec_default['best_value'] < tau:
            false_unreachable['spectral_default'] += 1

        # Test Spectral with boosted settings
        spec_boosted = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=5,
            maxiter=200,
            seed=seed + 300,
        )
        spectral_scores_boosted.append(spec_boosted['best_value'])
        if spec_boosted['best_value'] < tau:
            false_unreachable['spectral_boosted'] += 1

        # Test Krylov with default settings
        m = min(K, d)
        kry_default = optimize.maximize_krylov_score(
            psi, phi, hams, m=m,
            restarts=opt_settings['restarts'],
            maxiter=opt_settings['maxiter'],
            seed=seed + 400,
        )
        krylov_scores_default.append(kry_default['best_value'])
        if kry_default['best_value'] < tau:
            false_unreachable['krylov_default'] += 1

        # Test Krylov with boosted settings
        kry_boosted = optimize.maximize_krylov_score(
            psi, phi, hams, m=m,
            restarts=5,
            maxiter=200,
            seed=seed + 500,
        )
        krylov_scores_boosted.append(kry_boosted['best_value'])
        if kry_boosted['best_value'] < tau:
            false_unreachable['krylov_boosted'] += 1

        if trial < 3:
            print(f"    Trial {trial}: S*_default={spec_default['best_value']:.4f}, "
                  f"S*_boosted={spec_boosted['best_value']:.4f}, "
                  f"R*_default={kry_default['best_value']:.4f}, "
                  f"R*_boosted={kry_boosted['best_value']:.4f}")

    print(f"\n  False unreachable rates (tau={tau}):")
    for key, count in false_unreachable.items():
        rate = count / n_trials
        scores = (spectral_scores_default if 'spectral' in key else krylov_scores_default) if 'default' in key else (spectral_scores_boosted if 'spectral' in key else krylov_scores_boosted)
        print(f"    {key:25s}: {count}/{n_trials} ({100*rate:.1f}%)  mean_score={np.mean(scores):.4f}")

    return false_unreachable


def main():
    print("=" * 70)
    print("AUDIT TEST 2: Validation on PROVABLY REACHABLE targets")
    print("Any false unreachable = potential bug or optimizer failure")
    print("=" * 70)

    geo2_configs = {
        16: {'nx': 2, 'ny': 2},
    }

    rho = 0.05

    # Test GEO2
    for d, params in geo2_configs.items():
        K = max(2, round(rho * d**2))
        print(f"\n--- GEO2 d={d} K={K} ---")
        validate_criteria_on_reachable("GEO2", d, K, n_trials=20, **params)

    # Test Canonical
    for d in [16]:
        K = max(2, round(rho * d**2))
        print(f"\n--- Canonical d={d} K={K} ---")
        validate_criteria_on_reachable("canonical", d, K, n_trials=20)


if __name__ == "__main__":
    main()
