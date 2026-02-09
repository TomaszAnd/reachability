#!/usr/bin/env python3
"""
Production Data Generation for GEO2 with Corrected Settings.

Regenerates GEO2 production data using validated optimizer settings
(maxiter=100-150, restarts=3, analytical gradient).

Configurations:
- Lattices: 2x2 (d=16), 1x5 (d=32), 2x3 (d=64)
- Approach: Optimized lambda only (the scientifically meaningful one)
- Criteria: Spectral, Krylov, Moment (all three)
- tau = 0.99

The old data (geo2_production_complete_*.pkl) used maxiter=20, restarts=1
which produced 70% false-unreachable rate and anomalous dimension ordering.

Expected runtime with analytical gradient:
- d=16: ~5 min
- d=32: ~20 min
- d=64: ~2-4 hours
"""
import sys
sys.stdout = sys.stderr  # Unbuffered

import pickle
from datetime import datetime
from pathlib import Path
import traceback

from reach import analysis, settings

OUTPUT_DIR = Path("data/raw_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Production configurations - uses corrected optimizer settings
# analysis.py reads from settings.get_optimization_settings('GEO2')
# which now returns maxiter=100, restarts=3, ftol=1e-6
LATTICE_CONFIGS = [
    # (nx, ny, d, rho_max, rho_step, nks, nst, k_cap)
    # d=16: 2x2 lattice, L=48 operators
    # 15 density points x 30 hamiltonians x 20 states = 9000 trials
    (2, 2, 16, 0.15, 0.01, 30, 20, 50),

    # d=32: 1x5 lattice, L=51 operators
    # 12 density points x 20 hamiltonians x 20 states = 4800 trials
    (1, 5, 32, 0.12, 0.01, 20, 20, 150),

    # d=64: 2x3 lattice, L=??? operators
    # 10 density points x 15 hamiltonians x 10 states = 1500 trials
    (2, 3, 64, 0.10, 0.01, 15, 10, 500),
]

TAU = 0.99


def run_experiment(nx, ny, d, rho_max, rho_step, nks, nst, k_cap, timestamp):
    """Run a single production experiment."""
    n_points = int(rho_max / rho_step)
    total_trials = n_points * nks * nst

    # Get the settings that will actually be used
    opt = settings.get_optimization_settings('GEO2')

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT: d={d} ({nx}x{ny} lattice)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Density: rho=0..{rho_max} (step {rho_step}), {n_points} points", flush=True)
    print(f"  Sampling: {nks} hamiltonians x {nst} states = {nks*nst} trials/point", flush=True)
    print(f"  Total trials: {total_trials:,}", flush=True)
    print(f"  Optimizer: maxiter={opt['maxiter']}, restarts={opt['restarts']}, "
          f"ftol={opt['ftol']}", flush=True)
    print(f"  K cap: {k_cap}", flush=True)
    print(f"  Start: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print("", flush=True)

    start_time = datetime.now()

    result = analysis.monte_carlo_unreachability_vs_density(
        dims=[d],
        rho_max=rho_max,
        rho_step=rho_step,
        taus=[TAU],
        ensemble="GEO2",
        k_cap=k_cap,
        nks=nks,
        nst=nst,
        seed=42,
        optimize_lambda=True,
        nx=nx,
        ny=ny,
        periodic=False,
    )

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    time_per_trial = elapsed * 60 / max(total_trials, 1) * 1000  # ms

    print(f"\nCompleted d={d} in {elapsed:.1f} minutes", flush=True)
    print(f"  Time per trial: {time_per_trial:.1f} ms", flush=True)

    return {
        'data': result,
        'd': d,
        'nx': nx,
        'ny': ny,
        'rho_max': rho_max,
        'rho_step': rho_step,
        'nks': nks,
        'nst': nst,
        'optimizer_settings': opt,
        'runtime_minutes': elapsed,
        'time_per_trial_ms': time_per_trial,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70, flush=True)
    print("GEO2 PRODUCTION DATA GENERATION (CORRECTED SETTINGS)", flush=True)
    print("=" * 70, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Lattices: {len(LATTICE_CONFIGS)} configurations", flush=True)
    print(f"Tau: {TAU}", flush=True)

    # Show settings
    opt = settings.get_optimization_settings('GEO2')
    print(f"GEO2 settings: maxiter={opt['maxiter']}, restarts={opt['restarts']}, "
          f"ftol={opt['ftol']}", flush=True)
    print(f"Analytical gradient: enabled (spectral_overlap_with_grad)", flush=True)
    print("", flush=True)

    all_results = {}
    checkpoint_file = OUTPUT_DIR / f"geo2_corrected_checkpoint_{timestamp}.pkl"

    for nx, ny, d, rho_max, rho_step, nks, nst, k_cap in LATTICE_CONFIGS:
        try:
            result = run_experiment(
                nx, ny, d, rho_max, rho_step, nks, nst, k_cap, timestamp
            )
            all_results[d] = result

            # Save checkpoint after each dimension
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'timestamp': timestamp,
                    'tau': TAU,
                    'settings': opt,
                    'phase': f'completed_d{d}',
                    'results': all_results,
                }, f)
            print(f"  Checkpoint saved: {checkpoint_file}", flush=True)

        except Exception as e:
            print(f"\nERROR d={d}: {e}", flush=True)
            traceback.print_exc()
            continue

    # Save final results
    final_file = OUTPUT_DIR / f"geo2_corrected_production_{timestamp}.pkl"
    with open(final_file, 'wb') as f:
        pickle.dump({
            'timestamp': timestamp,
            'tau': TAU,
            'lattice_configs': LATTICE_CONFIGS,
            'settings': opt,
            'note': 'Generated with corrected optimizer settings (audit 2026-01-27)',
            'results': all_results,
        }, f)

    print(f"\n{'='*70}", flush=True)
    print(f"PRODUCTION RUN COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Output: {final_file}", flush=True)
    print(f"Dimensions completed: {sorted(all_results.keys())}", flush=True)

    # Print summary table
    print(f"\n{'d':>4} {'Runtime':>10} {'ms/trial':>10} {'Points':>8}", flush=True)
    print(f"{'-'*4:>4} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}", flush=True)
    for d in sorted(all_results.keys()):
        r = all_results[d]
        n_pts = int(r['rho_max'] / r['rho_step'])
        print(f"{d:4d} {r['runtime_minutes']:8.1f}m {r['time_per_trial_ms']:8.1f}ms "
              f"{n_pts:8d}", flush=True)


if __name__ == "__main__":
    main()
