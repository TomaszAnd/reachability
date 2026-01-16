#!/usr/bin/env python3
"""
GEO2LOCAL Production Experiments - Publication Quality

Generates data for plots similar to canonical ensemble:
- 3 lattices: 2×2 (d=16), 1×5 (d=32), 2×3 (d=64)
- 2 approaches: Fixed λ, Optimized λ
- 3 criteria: Moment, Spectral, Krylov

Expected runtime: ~5-6 hours total
"""
import sys
sys.stdout = sys.stderr  # Unbuffered

import pickle
from datetime import datetime
from pathlib import Path
import traceback

from reach import analysis

OUTPUT_DIR = Path("data/raw_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Production configurations - balanced for quality and runtime
# Based on benchmark: d=16 ~105ms/trial, d=32 ~332ms/trial optimized
LATTICE_CONFIGS = [
    # (nx, ny, d, rho_max, rho_step, nks, nst)
    # d=16: 16 pts × 600 trials = 9,600 total (~17 min opt, ~2 min fixed)
    (2, 2, 16, 0.15, 0.01, 30, 20),

    # d=32: 12 pts × 400 trials = 4,800 total (~27 min opt, ~4 min fixed)
    (1, 5, 32, 0.12, 0.01, 20, 20),

    # d=64: 10 pts × 200 trials = 2,000 total (~est 1-2 hours opt, ~15 min fixed)
    (2, 3, 64, 0.10, 0.01, 20, 10),
]

TAU = 0.99

def run_single_experiment(nx, ny, d, rho_max, rho_step, nks, nst, optimize_lambda, timestamp):
    """Run a single experiment and return results."""

    approach = "optimized" if optimize_lambda else "fixed"
    n_points = int(rho_max / rho_step) + 1
    total_trials = n_points * nks * nst

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT: d={d} ({nx}×{ny}), {approach} λ", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Density points: {n_points}", flush=True)
    print(f"  Trials per point: {nks} × {nst} = {nks*nst}", flush=True)
    print(f"  Total trials: {total_trials:,}", flush=True)
    print(f"  Start: {datetime.now().strftime('%H:%M:%S')}", flush=True)
    print("", flush=True)

    start_time = datetime.now()

    result = analysis.monte_carlo_unreachability_vs_density(
        dims=[d],
        rho_max=rho_max,
        rho_step=rho_step,
        taus=[TAU],
        ensemble="GEO2",
        k_cap=int(rho_max * d**2) + 20,
        nks=nks,
        nst=nst,
        seed=12345 if optimize_lambda else 54321,
        optimize_lambda=optimize_lambda,
        nx=nx,
        ny=ny,
        periodic=False,
    )

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    time_per_trial = elapsed * 60 / total_trials * 1000  # ms

    print(f"\n✓ Completed d={d} {approach} in {elapsed:.1f} minutes", flush=True)
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
        'optimize_lambda': optimize_lambda,
        'runtime_minutes': elapsed,
        'time_per_trial_ms': time_per_trial,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*70, flush=True)
    print("GEO2LOCAL PRODUCTION EXPERIMENTS", flush=True)
    print("="*70, flush=True)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Lattices: {len(LATTICE_CONFIGS)} configurations", flush=True)
    print(f"Approaches: Fixed λ, Optimized λ", flush=True)
    print(f"Expected runtime: ~5-6 hours", flush=True)
    print("", flush=True)

    all_results = {'fixed': {}, 'optimized': {}}
    checkpoint_file = OUTPUT_DIR / f"geo2_production_checkpoint_{timestamp}.pkl"

    # PHASE 1: Fixed λ (faster - run first)
    print("\n" + "="*70, flush=True)
    print("PHASE 1: FIXED λ EXPERIMENTS", flush=True)
    print("="*70, flush=True)

    for nx, ny, d, rho_max, rho_step, nks, nst in LATTICE_CONFIGS:
        try:
            result = run_single_experiment(
                nx, ny, d, rho_max, rho_step, nks, nst,
                optimize_lambda=False, timestamp=timestamp
            )
            all_results['fixed'][d] = result

            # Save checkpoint
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'timestamp': timestamp,
                    'tau': TAU,
                    'phase': 'fixed_complete' if d == 64 else f'fixed_d{d}',
                    'results': all_results,
                }, f)
            print(f"  Checkpoint saved", flush=True)

        except Exception as e:
            print(f"\n⚠️ ERROR d={d} fixed: {e}", flush=True)
            traceback.print_exc()
            continue

    # PHASE 2: Optimized λ (slower)
    print("\n" + "="*70, flush=True)
    print("PHASE 2: OPTIMIZED λ EXPERIMENTS", flush=True)
    print("="*70, flush=True)

    for nx, ny, d, rho_max, rho_step, nks, nst in LATTICE_CONFIGS:
        try:
            result = run_single_experiment(
                nx, ny, d, rho_max, rho_step, nks, nst,
                optimize_lambda=True, timestamp=timestamp
            )
            all_results['optimized'][d] = result

            # Save checkpoint
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'timestamp': timestamp,
                    'tau': TAU,
                    'phase': 'optimized_complete' if d == 64 else f'optimized_d{d}',
                    'results': all_results,
                }, f)
            print(f"  Checkpoint saved", flush=True)

        except Exception as e:
            print(f"\n⚠️ ERROR d={d} optimized: {e}", flush=True)
            traceback.print_exc()
            continue

    # Save final results
    final_file = OUTPUT_DIR / f"geo2_production_complete_{timestamp}.pkl"
    with open(final_file, 'wb') as f:
        pickle.dump({
            'timestamp': timestamp,
            'tau': TAU,
            'lattice_configs': LATTICE_CONFIGS,
            'results': all_results,
        }, f)

    # Summary
    print("\n" + "="*70, flush=True)
    print("EXPERIMENTS COMPLETE", flush=True)
    print("="*70, flush=True)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Results: {final_file}", flush=True)

    total_fixed = sum(r.get('runtime_minutes', 0) for r in all_results['fixed'].values())
    total_opt = sum(r.get('runtime_minutes', 0) for r in all_results['optimized'].values())

    print(f"\nRuntime Summary:", flush=True)
    print(f"  Fixed λ:", flush=True)
    for d, r in sorted(all_results['fixed'].items()):
        print(f"    d={d}: {r['runtime_minutes']:.1f} min ({r['time_per_trial_ms']:.1f} ms/trial)", flush=True)
    print(f"  Total fixed: {total_fixed:.1f} min", flush=True)

    print(f"\n  Optimized λ:", flush=True)
    for d, r in sorted(all_results['optimized'].items()):
        print(f"    d={d}: {r['runtime_minutes']:.1f} min ({r['time_per_trial_ms']:.1f} ms/trial)", flush=True)
    print(f"  Total optimized: {total_opt:.1f} min", flush=True)

    print(f"\n  Grand total: {(total_fixed + total_opt):.1f} min ({(total_fixed + total_opt)/60:.1f} hours)", flush=True)


if __name__ == "__main__":
    main()
