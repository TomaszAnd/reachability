#!/usr/bin/env python3
"""
Overnight production run for GEO2 ensemble.

Includes d=64 with optimized settings.
Generates publication-quality data with:
- 4 dimensions (d=8, 16, 32, 64)
- tau=0.99 only
- 200 trials (100 for d=64)
- Dense sampling near transitions

Estimated runtime: 10-15 hours (d=64 is slow)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from reach.parallel import parallel_evaluate

CONFIG = {
    'ensemble': 'GEO2',
    'dimensions': [8, 16, 32, 64],
    'tau_values': [0.99],  # Only tau=0.99 for GEO2
    'n_trials': {
        8: 200,
        16: 200,
        32: 200,
        64: 100,  # Reduced for d=64
    },
    'seed_base': 200000,
    'n_jobs': 1,  # Sequential (faster than multiprocess due to QuTiP serialization overhead)

    # Lattice configurations
    'lattice_params': {
        8: {'nx': 1, 'ny': 3},   # 3 qubits
        16: {'nx': 2, 'ny': 2},  # 4 qubits
        32: {'nx': 1, 'ny': 5},  # 5 qubits
        64: {'nx': 2, 'ny': 3},  # 6 qubits
    },

    # Dimension-specific rho ranges (GEO2 transitions earlier)
    'rho_points': {
        8: np.concatenate([
            np.linspace(0.01, 0.04, 4),
            np.linspace(0.04, 0.12, 20),
            np.linspace(0.14, 0.20, 4),
        ]),
        16: np.concatenate([
            np.linspace(0.005, 0.02, 4),
            np.linspace(0.02, 0.06, 20),
            np.linspace(0.08, 0.12, 4),
        ]),
        32: np.concatenate([
            np.linspace(0.005, 0.015, 4),
            np.linspace(0.015, 0.04, 20),
            np.linspace(0.05, 0.08, 4),
        ]),
        64: np.concatenate([
            np.array([0.005, 0.008, 0.01]),
            np.linspace(0.01, 0.03, 15),
            np.array([0.035, 0.04, 0.05]),
        ]),
    },

    # Criteria (skip Krylov for GEO2 d=64 - uninformative and slow)
    'criteria': {
        8: ['spectral', 'krylov', 'moment'],
        16: ['spectral', 'krylov', 'moment'],
        32: ['spectral', 'krylov', 'moment'],
        64: ['spectral', 'moment'],
    },

    'optimizer': {
        8: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
        16: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
        32: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
        64: {'maxiter': 150, 'restarts': 3, 'ftol': 1e-6},
    },
}


def run_overnight():
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'overnight'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(__file__).parent.parent.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"OVERNIGHT GEO2 RUN - {timestamp}")
    print("=" * 70)
    print(f"Dimensions: {CONFIG['dimensions']}")
    print(f"Tau values: {CONFIG['tau_values']}")

    all_results = []
    total_points = sum(
        len(np.unique(CONFIG['rho_points'][d]))
        for d in CONFIG['dimensions']
    ) * len(CONFIG['tau_values'])
    completed = 0

    start_time = time.time()

    for d in CONFIG['dimensions']:
        rho_points = np.unique(CONFIG['rho_points'][d])
        n_trials = CONFIG['n_trials'][d]
        criteria = CONFIG['criteria'][d]
        optimizer = CONFIG['optimizer'][d]
        lattice = CONFIG['lattice_params'][d]

        print(f"\n{'='*60}")
        print(f"Dimension d={d}: {len(rho_points)} rho points, {n_trials} trials")
        print(f"Criteria: {criteria}")
        print(f"Lattice: {lattice}")
        print(f"{'='*60}")

        for tau in CONFIG['tau_values']:
            for rho in rho_points:
                K = max(2, int(round(rho * d**2)))
                rho_actual = K / d**2

                t0 = time.time()

                result = parallel_evaluate(
                    ensemble=CONFIG['ensemble'],
                    d=d, K=K, tau=tau,
                    n_trials=n_trials,
                    seed_base=CONFIG['seed_base'] + d * 10000 + int(rho * 10000),
                    criteria=criteria,
                    optimizer_settings=optimizer,
                    lattice_params=lattice,
                    n_jobs=CONFIG['n_jobs'],
                )
                result['tau'] = tau

                elapsed = time.time() - t0
                completed += 1

                all_results.append(result)

                # Format output based on available criteria
                parts = [f"d={d}, K={K:3d}"]
                if 'spectral_P' in result:
                    parts.append(f"S={result['spectral_P']:.2f}")
                if 'krylov_P' in result:
                    parts.append(f"K={result['krylov_P']:.2f}")
                if 'moment_P' in result:
                    parts.append(f"M={result['moment_P']:.2f}")
                parts.append(f"[{elapsed:.1f}s] ({completed}/{total_points})")
                print(f"  {', '.join(parts)}")

                # Checkpoint
                if completed % 5 == 0:
                    df = pd.DataFrame(all_results)
                    df.to_csv(output_dir / f'geo2_checkpoint_{timestamp}.csv', index=False)
                    elapsed_total = time.time() - start_time
                    rate = elapsed_total / completed
                    remaining = rate * (total_points - completed)
                    print(f"    Checkpoint saved. ETA: {remaining/3600:.1f}h")

    # Save final
    df = pd.DataFrame(all_results)
    output_file = output_dir / f'geo2_overnight_{timestamp}.csv'
    df.to_csv(output_file, index=False)

    total_time = time.time() - start_time
    print(f"\n\nCompleted in {total_time/3600:.1f} hours")
    print(f"Saved to: {output_file}")
    print(f"Total data points: {len(all_results)}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for d in CONFIG['dimensions']:
        df_d = df[df['d'] == d]
        if len(df_d) > 0:
            print(f"  d={d}: {len(df_d)} points, "
                  f"spectral P range [{df_d['spectral_P'].min():.2f}, {df_d['spectral_P'].max():.2f}]")


if __name__ == '__main__':
    run_overnight()
