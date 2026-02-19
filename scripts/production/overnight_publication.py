#!/usr/bin/env python3
"""
Publication-quality production runs.

Models: CanonicalQuditModel, QubitGridModel
Dimensions: 16, 32, 64
Criteria:
  - Canonical: moment, spectral, krylov
  - QubitGrid: moment, spectral (skip krylov — always P=0)

Estimated runtime: ~12 hours

Usage:
    python scripts/production/overnight_publication.py
    python scripts/production/overnight_publication.py --canonical-only
    python scripts/production/overnight_publication.py --qubitgrid-only
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
import argparse
from pathlib import Path
from datetime import datetime

from src.models import CanonicalQuditModel, QubitGridModel
from src.sampling import DensitySweep, SweepConfig

# =============================================================================
# CONFIGURATION (matching overnight scripts)
# =============================================================================

CANONICAL_CONFIG = {
    'dimensions': [16, 32, 64],
    'tau': 0.99,
    'seed_base': 100000,
    'criteria': ['moment', 'spectral', 'krylov'],

    'trials': {
        16: (20, 10),  # n_hamiltonians, n_targets = 200 total
        32: (20, 10),
        64: (10, 10),  # Reduced = 100 total
    },

    'optimizer': {
        16: {'maxiter': 100, 'restarts': 3},
        32: {'maxiter': 100, 'restarts': 3},
        64: {'maxiter': 150, 'restarts': 3},
    },

    'K_values': {
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
}

QUBITGRID_CONFIG = {
    'dimensions': [16, 32, 64],
    'tau': 0.99,
    'seed_base': 200000,
    'criteria': ['moment', 'spectral'],  # Skip krylov (always P=0)

    'lattice_params': {
        16: {'nx': 2, 'ny': 2},
        32: {'nx': 1, 'ny': 5},
        64: {'nx': 2, 'ny': 3},
    },

    'trials': {
        16: (20, 10),
        32: (20, 10),
        64: (10, 10),
    },

    'optimizer': {
        16: {'maxiter': 100, 'restarts': 3},
        32: {'maxiter': 100, 'restarts': 3},
        64: {'maxiter': 150, 'restarts': 3},
    },

    'K_values': {
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
}


# =============================================================================
# PLOTTING (quick diagnostic — use plot_overnight_v7style.py for publication)
# =============================================================================

DIM_COLORS = {16: '#ff7f0e', 32: '#2ca02c', 64: '#d62728'}
DIM_MARKERS = {16: 's', 32: '^', 64: 'D'}


def plot_quick_summary(canonical_results, qubitgrid_results, output_dir):
    """Generate quick diagnostic plots (not publication quality)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Canonical 3-panel
    if canonical_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, crit in zip(axes, ['moment', 'spectral', 'krylov']):
            for d, df in sorted(canonical_results.items()):
                if len(df) == 0 or f'{crit}_P' not in df.columns:
                    continue
                color = DIM_COLORS.get(d, 'gray')
                marker = DIM_MARKERS.get(d, 'o')
                ax.errorbar(
                    df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                    fmt=f'{marker}-', color=color, label=f'd={d}',
                    capsize=3, markersize=5,
                )
            ax.set_xlabel(r'$\rho = K/d^2$')
            ax.set_ylabel('P(unreachable)')
            ax.set_title(f'Canonical - {crit.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / 'quick_canonical.png', dpi=150)
        plt.close()
        print(f"Saved quick_canonical.png")

    # QubitGrid 2-panel
    if qubitgrid_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, crit in zip(axes, ['moment', 'spectral']):
            for d, df in sorted(qubitgrid_results.items()):
                if len(df) == 0 or f'{crit}_P' not in df.columns:
                    continue
                color = DIM_COLORS.get(d, 'gray')
                marker = DIM_MARKERS.get(d, 'o')
                ax.errorbar(
                    df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                    fmt=f'{marker}-', color=color, label=f'd={d}',
                    capsize=3, markersize=5,
                )
            ax.set_xlabel(r'$\rho = K/d^2$')
            ax.set_ylabel('P(unreachable)')
            ax.set_title(f'QubitGrid - {crit.capitalize()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / 'quick_qubitgrid.png', dpi=150)
        plt.close()
        print(f"Saved quick_qubitgrid.png")


# =============================================================================
# RUN FUNCTIONS
# =============================================================================

def run_canonical(output_dir):
    """Run CanonicalQuditModel sweeps for d=16,32,64."""
    config = CANONICAL_CONFIG
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {}

    for d in config['dimensions']:
        print(f"\n{'='*60}")
        print(f"CanonicalQuditModel d={d}")
        print(f"{'='*60}")

        n_hams, n_tgt = config['trials'][d]
        opt = config['optimizer'][d]
        K_values = config['K_values'][d]

        print(f"K values: {len(K_values)} points from {K_values[0]} to {K_values[-1]}")
        print(f"Trials per K: {n_hams * n_tgt}")

        sweep_config = SweepConfig(
            n_hamiltonians=n_hams,
            n_targets=n_tgt,
            tau=config['tau'],
            maxiter=opt['maxiter'],
            restarts=opt['restarts'],
        )

        model = CanonicalQuditModel(
            dim=d, seed=config['seed_base'] + d * 1000)
        sweep = DensitySweep(model, sweep_config)

        t0 = time.perf_counter()
        df = sweep.run(
            K_values=K_values,
            criteria=config['criteria'],
            early_stop_zeros=5,
            verbose=True,
        )
        elapsed = time.perf_counter() - t0

        results[d] = df

        checkpoint_file = output_dir / f'canonical_d{d}_tau{config["tau"]}_{timestamp}.csv'
        sweep.save(checkpoint_file)
        print(f"Saved: {checkpoint_file} ({elapsed/60:.1f} min)")

    return results


def run_qubitgrid(output_dir):
    """Run QubitGridModel sweeps for d=16,32,64 (skip Krylov)."""
    config = QUBITGRID_CONFIG
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {}

    for d in config['dimensions']:
        print(f"\n{'='*60}")
        print(f"QubitGridModel d={d}")
        print(f"{'='*60}")

        n_hams, n_tgt = config['trials'][d]
        opt = config['optimizer'][d]
        lattice = config['lattice_params'][d]
        K_values = config['K_values'][d]

        print(f"Lattice: {lattice}")
        print(f"K values: {len(K_values)} points from {K_values[0]} to {K_values[-1]}")
        print(f"Trials per K: {n_hams * n_tgt}")
        print(f"Criteria: {config['criteria']} (skipping Krylov)")

        sweep_config = SweepConfig(
            n_hamiltonians=n_hams,
            n_targets=n_tgt,
            tau=config['tau'],
            maxiter=opt['maxiter'],
            restarts=opt['restarts'],
        )

        model = QubitGridModel(
            dim=d, seed=config['seed_base'] + d * 1000,
            **lattice)
        sweep = DensitySweep(model, sweep_config)

        t0 = time.perf_counter()
        df = sweep.run(
            K_values=K_values,
            criteria=config['criteria'],
            early_stop_zeros=5,
            verbose=True,
        )
        elapsed = time.perf_counter() - t0

        results[d] = df

        checkpoint_file = output_dir / f'qubitgrid_d{d}_tau{config["tau"]}_{timestamp}.csv'
        sweep.save(checkpoint_file)
        print(f"Saved: {checkpoint_file} ({elapsed/60:.1f} min)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Publication-quality production runs')
    parser.add_argument('--canonical-only', action='store_true',
                        help='Run only CanonicalQuditModel')
    parser.add_argument('--qubitgrid-only', action='store_true',
                        help='Run only QubitGridModel')
    args = parser.parse_args()

    data_root = Path(__file__).parent.parent.parent / 'data'
    can_dir = data_root / 'canonical'
    geo_dir = data_root / 'qubitgrid'
    can_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print("=" * 70)
    print(f"PUBLICATION PRODUCTION RUN - {start_time}")
    print("=" * 70)
    print(f"Dimensions: 16, 32, 64")
    print(f"Data output: {data_root}")

    canonical_results = {}
    qubitgrid_results = {}

    if not args.qubitgrid_only:
        print("\n" + "=" * 70)
        print("CANONICAL QUDIT MODEL")
        print("=" * 70)
        canonical_results = run_canonical(can_dir)

    if not args.canonical_only:
        print("\n" + "=" * 70)
        print("QUBITGRID MODEL (Krylov skipped)")
        print("=" * 70)
        qubitgrid_results = run_qubitgrid(geo_dir)

    # Quick diagnostic plots
    if canonical_results or qubitgrid_results:
        fig_dir = Path(__file__).parent.parent.parent / 'fig'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_quick_summary(canonical_results, qubitgrid_results, fig_dir)

    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETED at {end_time}")
    print(f"Total runtime: {elapsed}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
