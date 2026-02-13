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
import time
from pathlib import Path
from datetime import datetime

from src.models import QubitGridModel
from src.sampling import DensitySweep, SweepConfig

CONFIG = {
    'dimensions': [8, 16, 32, 64],
    'tau_values': [0.99],
    'seed_base': 200000,

    # Trials per dimension (n_hamiltonians * n_targets)
    'trials': {
        8: (20, 10),    # 200 total
        16: (20, 10),
        32: (20, 10),
        64: (10, 10),   # 100 total (reduced for d=64)
    },

    # Lattice configurations
    'lattice_params': {
        8: {'nx': 1, 'ny': 3},
        16: {'nx': 2, 'ny': 2},
        32: {'nx': 1, 'ny': 5},
        64: {'nx': 2, 'ny': 3},
    },

    # Dimension-specific K ranges (GEO2 transitions earlier)
    'K_values': {
        8: sorted(set(max(2, int(round(rho * 64)))
                       for rho in np.concatenate([
                           np.linspace(0.01, 0.04, 4),
                           np.linspace(0.04, 0.12, 20),
                           np.linspace(0.14, 0.20, 4),
                       ]))),
        16: sorted(set(max(2, int(round(rho * 256)))
                        for rho in np.concatenate([
                            np.linspace(0.005, 0.02, 4),
                            np.linspace(0.02, 0.06, 20),
                            np.linspace(0.08, 0.12, 4),
                        ]))),
        32: sorted(set(max(2, int(round(rho * 1024)))
                        for rho in np.concatenate([
                            np.linspace(0.005, 0.015, 4),
                            np.linspace(0.015, 0.04, 20),
                            np.linspace(0.05, 0.08, 4),
                        ]))),
        64: sorted(set(max(2, int(round(rho * 4096)))
                        for rho in np.concatenate([
                            np.array([0.005, 0.008, 0.01]),
                            np.linspace(0.01, 0.03, 15),
                            np.array([0.035, 0.04, 0.05]),
                        ]))),
    },

    # Criteria (skip Krylov for GEO2 d=64 - uninformative and slow)
    'criteria': {
        8: ['spectral', 'krylov', 'moment'],
        16: ['spectral', 'krylov', 'moment'],
        32: ['spectral', 'krylov', 'moment'],
        64: ['spectral', 'moment'],
    },

    'optimizer': {
        8: {'maxiter': 100, 'restarts': 3},
        16: {'maxiter': 100, 'restarts': 3},
        32: {'maxiter': 100, 'restarts': 3},
        64: {'maxiter': 150, 'restarts': 3},
    },
}


def run_overnight():
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'qubitgrid'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"OVERNIGHT QUBITGRID RUN - {timestamp}")
    print("=" * 70)
    print(f"Dimensions: {CONFIG['dimensions']}")
    print(f"Tau values: {CONFIG['tau_values']}")

    start_time = time.time()

    for d in CONFIG['dimensions']:
        K_values = CONFIG['K_values'][d]
        lattice = CONFIG['lattice_params'][d]
        n_hams, n_tgt = CONFIG['trials'][d]
        criteria = CONFIG['criteria'][d]
        opt = CONFIG['optimizer'][d]

        print(f"\n{'='*60}")
        print(f"Dimension d={d}: {len(K_values)} K values, lattice {lattice}")
        print(f"Criteria: {criteria}")
        print(f"{'='*60}")

        for tau in CONFIG['tau_values']:
            config = SweepConfig(
                n_hamiltonians=n_hams,
                n_targets=n_tgt,
                tau=tau,
                maxiter=opt['maxiter'],
                restarts=opt['restarts'],
            )

            model = QubitGridModel(
                dim=d, seed=CONFIG['seed_base'] + d * 1000,
                **lattice)
            sweep = DensitySweep(model, config)
            df = sweep.run(
                K_values=K_values,
                criteria=criteria,
                verbose=True,
            )

            checkpoint_file = output_dir / f'qubitgrid_d{d}_tau{tau}_{timestamp}.csv'
            sweep.save(checkpoint_file)
            print(f"  Saved: {checkpoint_file}")

    total_time = time.time() - start_time
    print(f"\n\nCompleted in {total_time/3600:.1f} hours")


if __name__ == '__main__':
    run_overnight()
