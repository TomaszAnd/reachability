#!/usr/bin/env python3
"""
Overnight production run for canonical ensemble.

Generates publication-quality data with:
- 4 dimensions (d=8, 16, 32, 64)
- 3 tau values (0.99, 0.95, 0.90)
- 200 trials per point (n_hamiltonians * n_targets)
- Dense sampling near transitions

Estimated runtime: 8-12 hours
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from pathlib import Path
from datetime import datetime

from src.models import CanonicalQuditModel
from src.sampling import DensitySweep, SweepConfig

# Configuration
CONFIG = {
    'dimensions': [8, 16, 32, 64],
    'tau_values': [0.99, 0.95, 0.90],
    'n_hamiltonians': 20,
    'n_targets': 10,
    'criteria': ['spectral', 'krylov', 'moment'],
    'seed_base': 100000,

    # Dimension-specific K ranges (based on transition estimates)
    'K_values': {
        8: sorted(set(max(2, int(round(rho * 64)))
                       for rho in np.concatenate([
                           np.linspace(0.02, 0.10, 5),
                           np.linspace(0.10, 0.22, 20),
                           np.linspace(0.25, 0.35, 4),
                       ]))),
        16: sorted(set(max(2, int(round(rho * 256)))
                        for rho in np.concatenate([
                            np.linspace(0.02, 0.06, 4),
                            np.linspace(0.06, 0.16, 20),
                            np.linspace(0.18, 0.25, 4),
                        ]))),
        32: sorted(set(max(2, int(round(rho * 1024)))
                        for rho in np.concatenate([
                            np.linspace(0.01, 0.03, 4),
                            np.linspace(0.03, 0.10, 20),
                            np.linspace(0.12, 0.18, 4),
                        ]))),
        64: sorted(set(max(2, int(round(rho * 4096)))
                        for rho in np.concatenate([
                            np.linspace(0.005, 0.02, 4),
                            np.linspace(0.02, 0.06, 20),
                            np.linspace(0.07, 0.10, 4),
                        ]))),
    },

    # Optimizer settings
    'optimizer': {
        8: {'maxiter': 100, 'restarts': 3},
        16: {'maxiter': 100, 'restarts': 3},
        32: {'maxiter': 100, 'restarts': 3},
        64: {'maxiter': 150, 'restarts': 3},
    },
}


def run_overnight():
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'overnight'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"OVERNIGHT CANONICAL RUN - {timestamp}")
    print("=" * 70)
    print(f"Dimensions: {CONFIG['dimensions']}")
    print(f"Tau values: {CONFIG['tau_values']}")

    start_time = time.time()

    for d in CONFIG['dimensions']:
        K_values = CONFIG['K_values'][d]
        opt = CONFIG['optimizer'][d]

        print(f"\n{'='*60}")
        print(f"Dimension d={d}: {len(K_values)} K values")
        print(f"{'='*60}")

        for tau in CONFIG['tau_values']:
            print(f"\n  tau={tau}")

            config = SweepConfig(
                n_hamiltonians=CONFIG['n_hamiltonians'],
                n_targets=CONFIG['n_targets'],
                tau=tau,
                maxiter=opt['maxiter'],
                restarts=opt['restarts'],
            )

            model = CanonicalQuditModel(
                dim=d, seed=CONFIG['seed_base'] + d * 1000)
            sweep = DensitySweep(model, config)
            df = sweep.run(
                K_values=K_values,
                criteria=CONFIG['criteria'],
                verbose=True,
            )

            # Save checkpoint per (d, tau) combination
            checkpoint_file = output_dir / f'canonical_d{d}_tau{tau}_{timestamp}.csv'
            sweep.save(checkpoint_file)
            print(f"  Saved: {checkpoint_file}")

    total_time = time.time() - start_time
    print(f"\n\nCompleted in {total_time/3600:.1f} hours")


if __name__ == '__main__':
    run_overnight()
