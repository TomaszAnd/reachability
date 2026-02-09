#!/usr/bin/env python3
"""
Find optimal (maxiter, restarts) settings for GEO2 that balance accuracy and speed.

Target: ≤1% false-unreachable rate on provably reachable targets.
Metric: Time per problem instance at each (maxiter, restarts) combination.

Test matrix:
- maxiter: [10, 20, 30, 40, 50, 75, 100, 150]
- restarts: [1, 2, 3]
- dimensions: d=16 (K=13), d=32 (K=51) at rho=0.05
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from scipy.linalg import expm
import qutip
from reach import models, optimize, mathematics


def generate_reachable_target(psi, hams, rng, t=1.0):
    """Generate target = exp(-iH(lambda)t)|psi> (provably reachable)."""
    K = len(hams)
    lambdas = rng.uniform(-1, 1, K)
    H_lambda = sum(l * H for l, H in zip(lambdas, hams))
    H_matrix = H_lambda.full()
    psi_arr = psi.full().flatten()
    U = expm(-1j * H_matrix * t)
    target_arr = U @ psi_arr
    target = qutip.Qobj(target_arr.reshape(-1, 1)).unit()
    return target, lambdas


def sweep_settings(ensemble, d, K, n_trials=30, tau=0.99, **params):
    """Sweep (maxiter, restarts) and measure false-unreachable rate + timing."""
    maxiter_values = [10, 20, 30, 40, 50, 75, 100, 150]
    restart_values = [1, 2, 3]

    # Pre-generate test problems (reachable targets)
    rng = np.random.RandomState(42)
    problems = []
    for trial in range(n_trials):
        seed = rng.randint(0, 2**31 - 1)
        psi = models.fock_state(d, 0)
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)
        phi, true_lam = generate_reachable_target(psi, hams, rng)
        problems.append((psi, phi, hams, seed))

    results = []

    for restarts in restart_values:
        for maxiter in maxiter_values:
            false_unreach = 0
            scores = []
            t0 = time.time()

            for psi, phi, hams, seed in problems:
                result = optimize.maximize_spectral_overlap(
                    psi, phi, hams,
                    restarts=restarts,
                    maxiter=maxiter,
                    ftol=1e-6,
                    seed=seed + 10,
                )
                scores.append(result['best_value'])
                if result['best_value'] < tau:
                    false_unreach += 1

            elapsed = time.time() - t0
            time_per_trial = elapsed / n_trials

            rate = false_unreach / n_trials
            mean_score = np.mean(scores)

            results.append({
                'maxiter': maxiter,
                'restarts': restarts,
                'false_unreach_rate': rate,
                'false_unreach_count': false_unreach,
                'mean_score': mean_score,
                'time_per_trial': time_per_trial,
                'total_time': elapsed,
            })

            marker = "***" if rate <= 0.01 else "   "
            print(f"  {marker} maxiter={maxiter:4d}, restarts={restarts}: "
                  f"false_unreach={rate:5.1%} ({false_unreach}/{n_trials}), "
                  f"mean_S*={mean_score:.4f}, "
                  f"time/trial={time_per_trial*1000:.0f}ms", flush=True)

    return results


def main():
    print("=" * 70)
    print("OPTIMIZER SETTINGS SWEEP")
    print("Target: ≤1% false-unreachable rate on provably reachable targets")
    print("=" * 70)

    configs = [
        ("GEO2 d=16 (K=13)", "GEO2", 16, 13, {'nx': 2, 'ny': 2}),
        ("GEO2 d=32 (K=51)", "GEO2", 32, 51, {'nx': 1, 'ny': 5}),
    ]

    all_results = {}

    for label, ensemble, d, K, params in configs:
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"{'=' * 70}")
        results = sweep_settings(ensemble, d, K, n_trials=25, **params)
        all_results[label] = results

    # Find Pareto-optimal configurations
    print("\n" + "=" * 70)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("(≤1% false-unreachable rate, sorted by time)")
    print("=" * 70)

    for label, results in all_results.items():
        print(f"\n{label}:")
        accurate = [r for r in results if r['false_unreach_rate'] <= 0.01]
        accurate.sort(key=lambda r: r['time_per_trial'])

        if not accurate:
            print("  No configuration achieved ≤1% false-unreachable rate!")
            # Show best available
            best = min(results, key=lambda r: r['false_unreach_rate'])
            print(f"  Best: maxiter={best['maxiter']}, restarts={best['restarts']}, "
                  f"rate={best['false_unreach_rate']:.1%}, "
                  f"time={best['time_per_trial']*1000:.0f}ms")
        else:
            for r in accurate[:5]:
                print(f"  maxiter={r['maxiter']:4d}, restarts={r['restarts']}: "
                      f"rate={r['false_unreach_rate']:.1%}, "
                      f"mean_S*={r['mean_score']:.4f}, "
                      f"time={r['time_per_trial']*1000:.0f}ms")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find fastest config with ≤1% rate across both dimensions
    for label, results in all_results.items():
        accurate = [r for r in results if r['false_unreach_rate'] <= 0.01]
        if accurate:
            fastest = min(accurate, key=lambda r: r['time_per_trial'])
            print(f"  {label}: maxiter={fastest['maxiter']}, restarts={fastest['restarts']} "
                  f"({fastest['time_per_trial']*1000:.0f}ms/trial)")


if __name__ == "__main__":
    main()
