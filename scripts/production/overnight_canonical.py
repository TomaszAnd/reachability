#!/usr/bin/env python3
"""
Overnight production run for canonical ensemble.

Generates publication-quality data with:
- 4 dimensions (d=8, 16, 32, 64)
- 3 tau values (0.99, 0.95, 0.90)
- 200 trials per point
- Dense sampling near transitions

Estimated runtime: 8-12 hours
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

# Configuration
CONFIG = {
    'ensemble': 'canonical',
    'dimensions': [8, 16, 32, 64],
    'tau_values': [0.99, 0.95, 0.90],
    'n_trials': 200,
    'criteria': ['spectral', 'krylov', 'moment'],
    'seed_base': 100000,
    'n_jobs': 1,  # Sequential (faster than multiprocess due to QuTiP serialization overhead)

    # Dimension-specific rho ranges (based on transition estimates)
    'rho_points': {
        8: np.concatenate([
            np.linspace(0.02, 0.10, 5),   # Pre-transition
            np.linspace(0.10, 0.22, 20),   # Transition (dense)
            np.linspace(0.25, 0.35, 4),    # Post-transition
        ]),
        16: np.concatenate([
            np.linspace(0.02, 0.06, 4),
            np.linspace(0.06, 0.16, 20),
            np.linspace(0.18, 0.25, 4),
        ]),
        32: np.concatenate([
            np.linspace(0.01, 0.03, 4),
            np.linspace(0.03, 0.10, 20),
            np.linspace(0.12, 0.18, 4),
        ]),
        64: np.concatenate([
            np.linspace(0.005, 0.02, 4),
            np.linspace(0.02, 0.06, 20),
            np.linspace(0.07, 0.10, 4),
        ]),
    },

    # Optimizer settings
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
    print(f"OVERNIGHT CANONICAL RUN - {timestamp}")
    print("=" * 70)
    print(f"Dimensions: {CONFIG['dimensions']}")
    print(f"Tau values: {CONFIG['tau_values']}")
    print(f"Trials per point: {CONFIG['n_trials']}")

    all_results = []
    total_points = sum(
        len(np.unique(CONFIG['rho_points'][d]))
        for d in CONFIG['dimensions']
    ) * len(CONFIG['tau_values'])
    completed = 0

    start_time = time.time()

    for d in CONFIG['dimensions']:
        rho_points = np.unique(CONFIG['rho_points'][d])
        optimizer = CONFIG['optimizer'][d]

        print(f"\n{'='*60}")
        print(f"Dimension d={d}: {len(rho_points)} rho points")
        print(f"{'='*60}")

        for tau in CONFIG['tau_values']:
            for rho in rho_points:
                K = max(2, int(round(rho * d**2)))
                rho_actual = K / d**2

                t0 = time.time()

                result = parallel_evaluate(
                    ensemble=CONFIG['ensemble'],
                    d=d, K=K, tau=tau,
                    n_trials=CONFIG['n_trials'],
                    seed_base=CONFIG['seed_base'] + d * 10000 + int(rho * 10000),
                    criteria=CONFIG['criteria'],
                    optimizer_settings=optimizer,
                    n_jobs=CONFIG['n_jobs'],
                )
                result['tau'] = tau

                elapsed = time.time() - t0
                completed += 1

                all_results.append(result)

                print(f"  d={d}, K={K:3d}, tau={tau}: "
                      f"S={result['spectral_P']:.2f}, "
                      f"K={result['krylov_P']:.2f}, "
                      f"M={result['moment_P']:.2f} "
                      f"[{elapsed:.1f}s] ({completed}/{total_points})")

                # Save checkpoint every 10 points
                if completed % 10 == 0:
                    df = pd.DataFrame(all_results)
                    df.to_csv(output_dir / f'canonical_checkpoint_{timestamp}.csv', index=False)
                    elapsed_total = time.time() - start_time
                    rate = elapsed_total / completed
                    remaining = rate * (total_points - completed)
                    print(f"    Checkpoint saved. ETA: {remaining/3600:.1f}h")

    # Save final results
    df = pd.DataFrame(all_results)
    output_file = output_dir / f'canonical_overnight_{timestamp}.csv'
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
        print(f"  d={d}: {len(df_d)} points, "
              f"spectral P range [{df_d['spectral_P'].min():.2f}, {df_d['spectral_P'].max():.2f}]")


if __name__ == '__main__':
    run_overnight()
