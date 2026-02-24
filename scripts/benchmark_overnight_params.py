#!/usr/bin/env python3
"""
Benchmark to determine optimal parameters for overnight sweep.
Tests different maxiter, restarts, and K ranges at each dimension.

Outputs per-trial timing and score quality to inform v88 configuration.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.models import CanonicalQuditModel, QubitGridModel
from src.criteria import SpectralCriterion, KrylovCriterion, Verdict


def benchmark_dimension(d, n_trials=5):
    """Benchmark various settings for one dimension."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING d={d}")
    print(f"{'='*60}")

    model = CanonicalQuditModel(dim=d, seed=42)

    # Estimate K values in transition region
    rho_c_est = 0.12 * np.sqrt(8 / d)
    K_trans = max(2, int(rho_c_est * d**2))
    K_values = [max(2, K_trans // 2), K_trans, min(K_trans * 2, model.K)]

    print(f"Estimated rho_c: {rho_c_est:.4f}, K_trans: {K_trans}")
    print(f"Testing K values: {K_values}")

    results = []

    for K in K_values:
        if K > model.K:
            continue
        print(f"\n  K={K} (rho={K/d**2:.4f}):")

        for criterion_name, CritCls in [('spectral', SpectralCriterion),
                                         ('krylov', KrylovCriterion)]:
            for maxiter in [50, 100]:
                for restarts in [3, 5]:
                    times = []
                    scores = []
                    verdicts_unreach = 0
                    for trial in range(n_trials):
                        sub = model.sample_submodel(K, seed=trial + 100)
                        phi = model.init_state()
                        psi = sub.random_state()

                        kwargs = {}
                        if criterion_name == 'krylov':
                            kwargs['m'] = model.dim
                        crit = CritCls(sub, phi, psi, tau=0.99, **kwargs)
                        t0 = time.time()
                        result = crit.is_reachable(maxiter=maxiter, restarts=restarts)
                        times.append(time.time() - t0)
                        scores.append(result.score)
                        if result.verdict == Verdict.UNREACHABLE:
                            verdicts_unreach += 1

                    avg_time = np.mean(times)
                    avg_score = np.mean(scores)
                    results.append({
                        'd': d, 'K': K, 'rho': K / d**2,
                        'criterion': criterion_name,
                        'maxiter': maxiter, 'restarts': restarts,
                        'avg_time_s': avg_time, 'avg_score': avg_score,
                        'P_unreach': verdicts_unreach / n_trials,
                    })
                    print(f"    {criterion_name} m={maxiter} r={restarts}: "
                          f"{avg_time:.3f}s/trial, score={avg_score:.3f}, "
                          f"P_unreach={verdicts_unreach/n_trials:.2f}")

    return results


def estimate_sweep_time(n_K, n_trials, time_per_trial_spectral, time_per_trial_krylov):
    """Estimate total sweep time including moment (fast)."""
    # Moment is ~0 cost relative to Spectral/Krylov
    total_s = n_K * n_trials * (time_per_trial_spectral + time_per_trial_krylov)
    return total_s / 3600


if __name__ == "__main__":
    all_results = []

    for d in [16, 32, 64, 128]:
        n_trials = 5 if d <= 64 else 3
        results = benchmark_dimension(d, n_trials=n_trials)
        all_results.extend(results)

    print("\n" + "=" * 60)
    print("FULL RESULTS")
    print("=" * 60)

    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))

    # Sweep time estimates
    print("\n" + "=" * 60)
    print("SWEEP TIME ESTIMATES")
    print("=" * 60)

    for maxiter in [50, 100]:
        for restarts in [3, 5]:
            print(f"\n--- maxiter={maxiter}, restarts={restarts} ---")
            for d in [16, 32, 64, 128, 256]:
                subset = df[(df['d'] == d) &
                            (df['maxiter'] == maxiter) &
                            (df['restarts'] == restarts)]
                if len(subset) > 0:
                    t_spec = subset[subset['criterion'] == 'spectral']['avg_time_s'].mean()
                    t_kry = subset[subset['criterion'] == 'krylov']['avg_time_s'].mean()
                else:
                    # Extrapolate from d=128 using d^2 scaling
                    base = df[(df['d'] == 128) &
                              (df['maxiter'] == maxiter) &
                              (df['restarts'] == restarts)]
                    if len(base) > 0:
                        scale = (d / 128) ** 2
                        t_spec = base[base['criterion'] == 'spectral']['avg_time_s'].mean() * scale
                        t_kry = base[base['criterion'] == 'krylov']['avg_time_s'].mean() * scale
                    else:
                        continue

                # Assume optimized K range: ~15 K values
                n_K = 15
                # Trials based on dimension
                if d <= 32:
                    n_trials_est = 1200
                elif d <= 64:
                    n_trials_est = 600
                elif d <= 128:
                    n_trials_est = 300
                else:
                    n_trials_est = 160

                hours = estimate_sweep_time(n_K, n_trials_est, t_spec, t_kry)
                print(f"  d={d:3d}: spec={t_spec:.3f}s kry={t_kry:.3f}s "
                      f"-> {n_trials_est} trials x {n_K} K -> {hours:.1f}h")
