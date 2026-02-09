#!/usr/bin/env python3
"""
Corrected Krylov experiment for canonical ensemble.

Key difference from overnight_canonical.py:
- Krylov ONLY (no spectral, no moment — those data already exist)
- m = min(K, d) instead of m = d
- This creates a genuine Krylov phase transition

Config: d in {8, 16, 32, 64}, tau = 0.99, 200 trials/point
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from reach import models, mathematics, optimize, settings

# -- Configuration ----------------------------------------------------------
ENSEMBLE = 'canonical'
DIMS = [8, 16, 32, 64]
TAU = 0.99
N_TRIALS = 200
SEED_BASE = 300000

# rho points per dimension (dense near expected transitions)
RHO_POINTS = {
    8: np.concatenate([
        np.linspace(0.02, 0.10, 5),
        np.linspace(0.10, 0.25, 20),
        np.linspace(0.28, 0.40, 4),
    ]),
    16: np.concatenate([
        np.linspace(0.02, 0.06, 4),
        np.linspace(0.06, 0.18, 20),
        np.linspace(0.20, 0.30, 4),
    ]),
    32: np.concatenate([
        np.linspace(0.01, 0.03, 4),
        np.linspace(0.03, 0.12, 20),
        np.linspace(0.14, 0.20, 4),
    ]),
    64: np.concatenate([
        np.linspace(0.005, 0.02, 4),
        np.linspace(0.02, 0.07, 20),
        np.linspace(0.08, 0.12, 3),
    ]),
}

OPTIMIZER = {
    8:  {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    16: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    32: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    64: {'maxiter': 150, 'restarts': 3, 'ftol': 1e-6},
}


def evaluate_single_krylov(d, K, tau, seed, opt_settings):
    """Evaluate Krylov criterion for one (psi, phi, hams) triple with m=min(K,d)."""
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed + 1000)[0]
    hams = models.random_hamiltonian_ensemble(d, K, ENSEMBLE, seed=seed)

    m = min(K, d)  # CRITICAL: corrected Krylov rank

    res = optimize.maximize_krylov_score(
        psi, phi, hams, m=m,
        maxiter=opt_settings['maxiter'],
        restarts=opt_settings['restarts'],
        ftol=opt_settings['ftol'],
        seed=seed,
    )
    return {
        'krylov_score': res['best_value'],
        'krylov_unreachable': res['best_value'] < tau,
        'runtime_s': res['runtime_s'],
    }


def run():
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'krylov_corrected'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"CORRECTED KRYLOV -- CANONICAL -- {timestamp}")
    print(f"m = min(K, d)  |  tau = {TAU}  |  trials = {N_TRIALS}")
    print("=" * 70)

    all_results = []
    total_points = sum(len(np.unique(RHO_POINTS[d])) for d in DIMS)
    completed = 0
    start_time = time.time()

    for d in DIMS:
        rho_points = np.unique(RHO_POINTS[d])
        opt = OPTIMIZER[d]

        print(f"\n--- d={d}: {len(rho_points)} rho points ---")

        for rho in rho_points:
            K = max(2, int(round(rho * d**2)))
            rho_actual = K / d**2
            m = min(K, d)

            t0 = time.time()

            # Sequential evaluation (parallel would add overhead for single-criterion)
            unreachable_count = 0
            scores = []
            for trial in range(N_TRIALS):
                seed = SEED_BASE + d * 10000 + int(rho * 10000) * 100 + trial
                result = evaluate_single_krylov(d, K, TAU, seed, opt)
                scores.append(result['krylov_score'])
                if result['krylov_unreachable']:
                    unreachable_count += 1

            krylov_P = unreachable_count / N_TRIALS
            elapsed = time.time() - t0
            completed += 1

            row = {
                'ensemble': ENSEMBLE,
                'd': d,
                'K': K,
                'rho': rho_actual,
                'm': m,
                'tau': TAU,
                'n_trials': N_TRIALS,
                'krylov_P': krylov_P,
                'krylov_mean': np.mean(scores),
                'krylov_std': np.std(scores),
            }
            all_results.append(row)

            print(f"  K={K:3d} (rho={rho_actual:.4f}) m={m:2d}: "
                  f"P={krylov_P:.3f}  mean_R={np.mean(scores):.4f}  "
                  f"[{elapsed:.1f}s] ({completed}/{total_points})")

            # Checkpoint every 5 points
            if completed % 5 == 0:
                df = pd.DataFrame(all_results)
                df.to_csv(output_dir / f'canonical_krylov_checkpoint_{timestamp}.csv', index=False)

    # Save final
    df = pd.DataFrame(all_results)
    output_file = output_dir / f'canonical_krylov_corrected_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    print(f"   Saved to: {output_file}")
    print(f"   Total data points: {len(all_results)}")


if __name__ == '__main__':
    run()
