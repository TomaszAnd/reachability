#!/usr/bin/env python3
"""
Dense Krylov Resampling - Match overnight (d, K) grid exactly.

Problem: The corrected Krylov runs used DIFFERENT rho grids than overnight spectral.
Result: Krylov has ~6 markers vs ~25 Spectral markers in combined plots.

Solution: Extract exact (d, K) pairs from overnight CSVs, run Krylov at those
SAME K values. Also add extra dense points near known Krylov rho_c.

Usage:
    python krylov_dense_resample.py --ensemble canonical
    python krylov_dense_resample.py --ensemble geo2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime

from reach import models, optimize

# =============================================================================
# CONFIGURATION
# =============================================================================

SEED_BASE = 500000  # New seed to avoid collision with previous runs
TAU = 0.99

# GEO2 lattice configurations
GEO2_LATTICE = {
    8:  {'nx': 3, 'ny': 1},
    16: {'nx': 2, 'ny': 2},
    32: {'nx': 5, 'ny': 1},
    64: {'nx': 3, 'ny': 2},
}

# Optimizer settings (dimension-dependent)
OPTIMIZER = {
    8:  {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    16: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    32: {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6},
    64: {'maxiter': 150, 'restarts': 3, 'ftol': 1e-6},
}

# Trials per dimension
TRIALS = {
    'canonical': {8: 200, 16: 200, 32: 200, 64: 200},
    'geo2': {8: 200, 16: 200, 32: 200, 64: 100},
}

# Known Krylov rho_c values for dense sampling (from v23 fits)
KRYLOV_RHO_C = {
    'canonical': {
        8:  {'rho_c': 0.158, 'delta': 0.030},
        16: {'rho_c': 0.086, 'delta': 0.023},
        32: {'rho_c': 0.050, 'delta': 0.017},
        64: {'rho_c': 0.027, 'delta': 0.010},
    },
    'geo2': {
        8:  {'rho_c': 0.078, 'delta': 0.003},
        16: {'rho_c': 0.038, 'delta': 0.0004},
        32: {'rho_c': 0.019, 'delta': 0.0003},
        64: {'rho_c': 0.007, 'delta': 0.0002},
    },
}

# Data file paths
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
OVERNIGHT_FILES = {
    'canonical': DATA_DIR / 'overnight' / 'canonical_overnight_20260128_224236.csv',
    'geo2': DATA_DIR / 'overnight' / 'geo2_overnight_20260128_224613.csv',
}
CORRECTED_FILES = {
    'canonical': DATA_DIR / 'krylov_corrected' / 'canonical_krylov_corrected_20260204_222726.csv',
    'geo2': DATA_DIR / 'krylov_corrected' / 'geo2_krylov_corrected_20260204_222747.csv',
}


def get_overnight_dk_pairs(ensemble):
    """Extract unique (d, K) pairs from overnight CSV at tau=0.99."""
    csv_file = OVERNIGHT_FILES[ensemble]
    if not csv_file.exists():
        raise FileNotFoundError(f"Overnight file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df_tau = df[df['tau'] == TAU]

    # Get unique (d, K) pairs
    pairs = df_tau[['d', 'K']].drop_duplicates().values.tolist()
    return [(int(d), int(K)) for d, K in pairs]


def get_existing_dk_pairs(ensemble):
    """Get (d, K) pairs already in corrected Krylov CSV."""
    csv_file = CORRECTED_FILES[ensemble]
    if not csv_file.exists():
        return set()

    df = pd.read_csv(csv_file)
    pairs = df[['d', 'K']].drop_duplicates().values.tolist()
    return {(int(d), int(K)) for d, K in pairs}


def get_extra_transition_points(ensemble):
    """Generate extra (d, K) points near Krylov transition for dense sampling."""
    extra_points = []

    for d, params in KRYLOV_RHO_C[ensemble].items():
        rho_c = params['rho_c']
        delta = params['delta']

        # 10 extra points centered on rho_c
        rho_extra = np.linspace(rho_c - 3*delta, rho_c + 3*delta, 10)

        for rho in rho_extra:
            if rho > 0:
                K = max(2, int(round(rho * d**2)))
                extra_points.append((d, K))

    return extra_points


def evaluate_single_krylov(d, K, tau, seed, opt_settings, ensemble):
    """Evaluate Krylov criterion for one trial with m=min(K,d)."""
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed + 1000)[0]

    if ensemble == 'geo2':
        lattice = GEO2_LATTICE[d]
        hams = models.random_hamiltonian_ensemble(d, K, 'GEO2', seed=seed, **lattice)
    else:
        hams = models.random_hamiltonian_ensemble(d, K, 'canonical', seed=seed)

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


def run_krylov_for_points(dk_pairs, ensemble, timestamp):
    """Run Krylov evaluation for list of (d, K) pairs."""
    output_dir = DATA_DIR / 'krylov_corrected'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by d, then K for organized processing
    dk_pairs = sorted(set(dk_pairs), key=lambda x: (x[0], x[1]))

    all_results = []
    total = len(dk_pairs)
    start_time = time.time()

    print(f"\nRunning Krylov for {total} (d, K) pairs")
    print("-" * 60)

    current_d = None
    for i, (d, K) in enumerate(dk_pairs):
        if d != current_d:
            current_d = d
            print(f"\n--- d={d} ---")

        opt = OPTIMIZER[d]
        n_trials = TRIALS[ensemble][d]
        m = min(K, d)
        rho = K / d**2

        t0 = time.time()

        unreachable_count = 0
        scores = []
        for trial in range(n_trials):
            seed = SEED_BASE + d * 10000 + K * 100 + trial
            result = evaluate_single_krylov(d, K, TAU, seed, opt, ensemble)
            scores.append(result['krylov_score'])
            if result['krylov_unreachable']:
                unreachable_count += 1

        krylov_P = unreachable_count / n_trials
        elapsed = time.time() - t0

        row = {
            'ensemble': ensemble.upper() if ensemble == 'geo2' else ensemble,
            'd': d,
            'K': K,
            'rho': rho,
            'm': m,
            'tau': TAU,
            'n_trials': n_trials,
            'krylov_P': krylov_P,
            'krylov_mean': np.mean(scores),
            'krylov_std': np.std(scores),
        }

        # Add lattice params for GEO2
        if ensemble == 'geo2':
            lattice = GEO2_LATTICE[d]
            row['nx'] = lattice['nx']
            row['ny'] = lattice['ny']

        all_results.append(row)

        print(f"  K={K:3d} (rho={rho:.4f}) m={m:2d}: "
              f"P={krylov_P:.3f}  mean_R={np.mean(scores):.4f}  "
              f"[{elapsed:.1f}s] ({i+1}/{total})")

        # Checkpoint every 10 points
        if (i + 1) % 10 == 0:
            df = pd.DataFrame(all_results)
            checkpoint_file = output_dir / f'{ensemble}_krylov_dense_checkpoint_{timestamp}.csv'
            df.to_csv(checkpoint_file, index=False)

    return all_results


def merge_with_existing(new_results, ensemble, timestamp):
    """Merge new results with existing corrected Krylov data."""
    output_dir = DATA_DIR / 'krylov_corrected'

    # Load existing data
    existing_file = CORRECTED_FILES[ensemble]
    if existing_file.exists():
        df_existing = pd.read_csv(existing_file)
        print(f"\nLoaded {len(df_existing)} existing rows from {existing_file.name}")
    else:
        df_existing = pd.DataFrame()
        print(f"\nNo existing file found, creating new")

    # Convert new results to DataFrame
    df_new = pd.DataFrame(new_results)
    print(f"New results: {len(df_new)} rows")

    # Concatenate
    if len(df_existing) > 0:
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_merged = df_new

    # Deduplicate on (d, K) - keep last (new) values
    df_merged = df_merged.drop_duplicates(subset=['d', 'K'], keep='last')
    df_merged = df_merged.sort_values(['d', 'K']).reset_index(drop=True)

    # Save merged file
    ensemble_name = ensemble if ensemble == 'canonical' else 'geo2'
    output_file = output_dir / f'{ensemble_name}_krylov_dense_{timestamp}.csv'
    df_merged.to_csv(output_file, index=False)

    print(f"Merged: {len(df_merged)} total rows")
    print(f"Saved to: {output_file}")

    return df_merged, output_file


def main():
    parser = argparse.ArgumentParser(description='Dense Krylov resampling')
    parser.add_argument('--ensemble', choices=['canonical', 'geo2'], required=True,
                        help='Ensemble type')
    args = parser.parse_args()

    ensemble = args.ensemble
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 70)
    print(f"DENSE KRYLOV RESAMPLING -- {ensemble.upper()} -- {timestamp}")
    print(f"m = min(K, d)  |  tau = {TAU}  |  seed_base = {SEED_BASE}")
    print("=" * 70)

    # Step 1: Get overnight (d, K) pairs
    overnight_pairs = get_overnight_dk_pairs(ensemble)
    print(f"\nOvernight (d, K) pairs: {len(overnight_pairs)}")

    # Step 2: Get existing corrected pairs
    existing_pairs = get_existing_dk_pairs(ensemble)
    print(f"Existing corrected pairs: {len(existing_pairs)}")

    # Step 3: Find missing pairs
    missing_pairs = [(d, K) for d, K in overnight_pairs if (d, K) not in existing_pairs]
    print(f"Missing pairs (overnight - existing): {len(missing_pairs)}")

    # Step 4: Add extra transition points
    extra_pairs = get_extra_transition_points(ensemble)
    extra_new = [(d, K) for d, K in extra_pairs if (d, K) not in existing_pairs]
    print(f"Extra transition points (new): {len(extra_new)}")

    # Step 5: Combine all points to run
    all_new_pairs = list(set(missing_pairs + extra_new))
    print(f"Total new (d, K) pairs to run: {len(all_new_pairs)}")

    if len(all_new_pairs) == 0:
        print("\nNo new points to compute. All overnight points already covered.")
        return

    # Estimate runtime
    estimated_time = 0
    for d, K in all_new_pairs:
        # Rough estimate: 0.1s per trial per K parameter
        n_trials = TRIALS[ensemble][d]
        estimated_time += n_trials * K * 0.05 / 60  # minutes
    print(f"\nEstimated runtime: ~{estimated_time:.0f} minutes")

    # Step 6: Run Krylov evaluations
    start_time = time.time()
    new_results = run_krylov_for_points(all_new_pairs, ensemble, timestamp)

    # Step 7: Merge with existing
    df_merged, output_file = merge_with_existing(new_results, ensemble, timestamp)

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETED in {total_time/60:.1f} minutes")
    print(f"Output: {output_file}")
    print(f"Total rows: {len(df_merged)}")

    # Summary by dimension
    print("\nSummary by dimension:")
    for d in sorted(df_merged['d'].unique()):
        sub = df_merged[df_merged['d'] == d]
        print(f"  d={d}: {len(sub)} K values, K range [{sub['K'].min()}, {sub['K'].max()}]")


if __name__ == '__main__':
    main()
