#!/usr/bin/env python3
"""
Corrected Krylov experiment for GEO2 ensemble.

Key difference from overnight_geo2.py:
- Krylov ONLY (spectral + moment data already exist)
- m = min(K, d) instead of m = d
- Previous GEO2 Krylov was TRIVIALLY P=0 everywhere because m=d
  caused the Krylov subspace to span all of R^d
- With m=min(K,d), expect genuine phase transition

Config: d in {8, 16, 32, 64}, tau = 0.99
        200 trials/point (100 for d=64)
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
ENSEMBLE = 'GEO2'
TAU = 0.99
SEED_BASE = 400000

# GEO2 lattice -> dimension mapping
LATTICE_CONFIGS = {
    8:  {'nx': 3, 'ny': 1},   # 3 qubits -> d=8
    16: {'nx': 2, 'ny': 2},   # 4 qubits -> d=16
    32: {'nx': 5, 'ny': 1},   # 5 qubits -> d=32
    64: {'nx': 3, 'ny': 2},   # 6 qubits -> d=64
}

TRIALS_PER_DIM = {8: 200, 16: 200, 32: 200, 64: 100}

# rho points per dimension
RHO_POINTS = {
    8: np.concatenate([
        np.linspace(0.02, 0.08, 4),
        np.linspace(0.08, 0.25, 20),
        np.linspace(0.28, 0.40, 4),
    ]),
    16: np.concatenate([
        np.linspace(0.01, 0.04, 4),
        np.linspace(0.04, 0.15, 20),
        np.linspace(0.17, 0.25, 4),
    ]),
    32: np.concatenate([
        np.linspace(0.005, 0.02, 4),
        np.linspace(0.02, 0.08, 15),
        np.linspace(0.09, 0.15, 4),
    ]),
    64: np.concatenate([
        np.linspace(0.005, 0.015, 3),
        np.linspace(0.015, 0.05, 15),
        np.linspace(0.055, 0.08, 3),
    ]),
}

OPTIMIZER = {
    8:  {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    16: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    32: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    64: {'maxiter': 150, 'restarts': 3, 'ftol': 1e-6},
}


def evaluate_single_krylov_geo2(d, K, tau, seed, opt_settings, lattice_params):
    """Evaluate Krylov criterion for one GEO2 trial with m=min(K,d)."""
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed + 1000)[0]
    hams = models.random_hamiltonian_ensemble(d, K, ENSEMBLE, seed=seed, **lattice_params)

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
    print(f"CORRECTED KRYLOV -- GEO2 -- {timestamp}")
    print(f"m = min(K, d)  |  tau = {TAU}")
    print("=" * 70)

    all_results = []
    total_points = sum(len(np.unique(RHO_POINTS[d])) for d in LATTICE_CONFIGS)
    completed = 0
    start_time = time.time()

    for d, lattice in LATTICE_CONFIGS.items():
        rho_points = np.unique(RHO_POINTS[d])
        opt = OPTIMIZER[d]
        n_trials = TRIALS_PER_DIM[d]

        print(f"\n--- d={d} ({lattice['nx']}x{lattice['ny']} lattice): "
              f"{len(rho_points)} rho points, {n_trials} trials ---")

        for rho in rho_points:
            K = max(2, int(round(rho * d**2)))
            rho_actual = K / d**2
            m = min(K, d)

            t0 = time.time()

            unreachable_count = 0
            scores = []
            for trial in range(n_trials):
                seed = SEED_BASE + d * 10000 + int(rho * 10000) * 100 + trial
                result = evaluate_single_krylov_geo2(d, K, TAU, seed, opt, lattice)
                scores.append(result['krylov_score'])
                if result['krylov_unreachable']:
                    unreachable_count += 1

            krylov_P = unreachable_count / n_trials
            elapsed = time.time() - t0
            completed += 1

            row = {
                'ensemble': ENSEMBLE,
                'd': d,
                'K': K,
                'rho': rho_actual,
                'm': m,
                'tau': TAU,
                'n_trials': n_trials,
                'krylov_P': krylov_P,
                'krylov_mean': np.mean(scores),
                'krylov_std': np.std(scores),
                'nx': lattice['nx'],
                'ny': lattice['ny'],
            }
            all_results.append(row)

            print(f"  K={K:3d} (rho={rho_actual:.4f}) m={m:2d}: "
                  f"P={krylov_P:.3f}  mean_R={np.mean(scores):.4f}  "
                  f"[{elapsed:.1f}s] ({completed}/{total_points})")

            if completed % 5 == 0:
                df = pd.DataFrame(all_results)
                df.to_csv(output_dir / f'geo2_krylov_checkpoint_{timestamp}.csv', index=False)

    df = pd.DataFrame(all_results)
    output_file = output_dir / f'geo2_krylov_corrected_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.1f} minutes")
    print(f"   Saved to: {output_file}")
    print(f"   Total data points: {len(all_results)}")


if __name__ == '__main__':
    run()
