#!/usr/bin/env python3
"""
Audit Test 5: Dimension scaling of optimizer effectiveness.

This is the KEY test. For GEO2 at fixed rho=0.05:
  d=16 -> K=13, d=32 -> K=51

Compare GEO2 default settings (maxiter=20) vs boosted (maxiter=200)
to see if the dimension-ordering anomaly is caused by under-optimization.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from reach import models, optimize, mathematics, settings


def measure_unreachability(ensemble, d, K, n_trials=30, maxiter=20, restarts=1, tau=0.99, **params):
    """Estimate P(unreachable) at given optimizer settings."""
    rng = np.random.RandomState(42)
    unreachable_count = 0
    scores = []

    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed + 1, **params)

        result = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed + 2,
        )
        scores.append(result['best_value'])
        if result['best_value'] < tau:
            unreachable_count += 1

    p = unreachable_count / n_trials
    return p, np.array(scores)


def main():
    print("=" * 70)
    print("AUDIT TEST 5: Dimension Scaling of GEO2 Optimizer")
    print("=" * 70)
    print("\nIf anomaly is optimizer-related, boosted settings should:")
    print("  - Reduce P(unreachable) more at higher d")
    print("  - Potentially reverse the dimension ordering")

    rho = 0.05
    tau = 0.99
    n_trials = 25

    geo2_configs = [
        (16, {'nx': 2, 'ny': 2}),
        (32, {'nx': 1, 'ny': 5}),
    ]

    print(f"\nFixed rho={rho}, tau={tau}, n_trials={n_trials}")
    print(f"\nGEO2 default: maxiter={settings.GEO2_MAXITER}, restarts={settings.GEO2_RESTARTS}")

    # Compare optimizer configurations
    configs = [
        ("Default (maxiter=20, r=1)", 20, 1),
        ("Medium  (maxiter=50, r=2)", 50, 2),
        ("Boosted (maxiter=200, r=3)", 200, 3),
    ]

    print(f"\n{'Config':<35s} {'d=16 P(unreach)':<20s} {'d=32 P(unreach)':<20s} {'Order?'}")
    print("-" * 95)

    for config_name, maxiter, restarts in configs:
        p_values = {}
        mean_scores = {}
        for d, params in geo2_configs:
            K = max(2, round(rho * d**2))
            t0 = time.time()
            p, scores = measure_unreachability(
                "GEO2", d, K, n_trials=n_trials,
                maxiter=maxiter, restarts=restarts, tau=tau, **params
            )
            elapsed = time.time() - t0
            p_values[d] = p
            mean_scores[d] = np.mean(scores)
            print(f"  ({config_name}, d={d}, K={K}): P={p:.2f}, mean S*={np.mean(scores):.4f}  [{elapsed:.0f}s]", flush=True)

        # Check ordering
        if p_values[16] < p_values[32]:
            order = "d=16 drops first (ANOMALOUS)"
        elif p_values[16] > p_values[32]:
            order = "d=32 drops first (EXPECTED)"
        else:
            order = "TIED"

        print(f"  {config_name:<35s} P16={p_values[16]:.2f}           P32={p_values[32]:.2f}           {order}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  If order reverses with boosted settings -> optimizer is the cause")
    print("  If order persists -> may be genuine physics of GEO2 ensemble")
    print("=" * 70)


if __name__ == "__main__":
    main()
