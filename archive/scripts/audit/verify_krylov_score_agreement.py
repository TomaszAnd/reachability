#!/usr/bin/env python3
"""
Verify krylov_score_with_grad returns same scores as krylov_score.

Critical: The gradient function must NOT change the score computation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from reach import models, mathematics

def test_score_agreement():
    """Verify R from krylov_score_with_grad matches krylov_score exactly."""
    print("=" * 60)
    print("KRYLOV SCORE AGREEMENT TEST")
    print("=" * 60)

    configs = [
        {'d': 8, 'K': 4, 'ensemble': 'canonical', 'n_tests': 50},
        {'d': 8, 'K': 4, 'ensemble': 'GUE', 'n_tests': 50},
        {'d': 16, 'K': 8, 'ensemble': 'canonical', 'n_tests': 30},
        {'d': 16, 'K': 8, 'ensemble': 'GUE', 'n_tests': 30},
        {'d': 32, 'K': 16, 'ensemble': 'GEO2', 'n_tests': 20},
    ]

    all_passed = True

    for config in configs:
        d, K, ensemble = config['d'], config['K'], config['ensemble']
        n_tests = config['n_tests']

        max_diff = 0.0

        for seed in range(n_tests):
            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed)[0]

            if ensemble == 'GEO2':
                # Determine lattice params
                if d == 8:
                    lattice = {'nx': 1, 'ny': 3}
                elif d == 16:
                    lattice = {'nx': 2, 'ny': 2}
                elif d == 32:
                    lattice = {'nx': 1, 'ny': 5}
                hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **lattice)
            else:
                hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

            # Random lambda
            rng = np.random.RandomState(seed + 1000)
            lambdas = rng.uniform(-1, 1, K)

            # Original function
            R_orig = mathematics.krylov_score(lambdas, psi, phi, hams)

            # New function with gradient
            R_new, _ = mathematics.krylov_score_with_grad(lambdas, psi, phi, hams)

            diff = abs(R_orig - R_new)
            max_diff = max(max_diff, diff)

        passed = max_diff < 1e-10
        status = "PASS" if passed else "FAIL"
        print(f"  {ensemble} d={d}, K={K}: max diff = {max_diff:.2e} [{status}]")

        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - Scores are identical")
    else:
        print("SOME TESTS FAILED - Score discrepancy detected!")
    print("=" * 60)

    return all_passed

if __name__ == '__main__':
    test_score_agreement()
