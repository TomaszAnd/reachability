"""
Benchmark quickstart-style sweep before and after optimizations.
Produces timing data and plots for comparison.

Usage:
    python scripts/benchmark_before_after.py                          # Sequential
    python scripts/benchmark_before_after.py --parallel               # Parallel
    python scripts/benchmark_before_after.py --dimensions 8 12 16     # Custom dims
"""
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import CanonicalQuditModel, QubitGridModel
from src.sampling import DensitySweep, SweepConfig


def run_benchmark(name: str, use_parallel: bool = False, dimensions: list = None):
    """
    Run quickstart-style benchmark and return timing + results.

    Args:
        name: Identifier for this benchmark run
        use_parallel: Use run_parallel() instead of run()
        dimensions: List of dimensions to test (default: [8, 12, 16])

    Returns:
        dict with timing and DataFrame results
    """
    if dimensions is None:
        dimensions = [8, 12, 16]

    results = {}

    for d in dimensions:
        print(f"\n{'='*60}")
        print(f"Dimension d={d}")
        print(f"{'='*60}")

        config = SweepConfig(
            n_hamiltonians=20,
            n_targets=10,
            maxiter=50,
            restarts=3,
            tau=0.99,
        )

        K_max = d * d
        K_values = sorted(set([
            max(2, int(0.05 * K_max)),
            max(2, int(0.10 * K_max)),
            max(2, int(0.15 * K_max)),
            max(2, int(0.20 * K_max)),
            max(2, int(0.25 * K_max)),
            max(2, int(0.30 * K_max)),
            max(2, int(0.35 * K_max)),
            max(2, int(0.40 * K_max)),
        ]))

        # CanonicalQuditModel
        print(f"\nCanonicalQuditModel d={d}:")
        model = CanonicalQuditModel(dim=d, seed=42)
        sweep = DensitySweep(model, config)

        t0 = time.perf_counter()
        if use_parallel:
            df = sweep.run_parallel(K_values, verbose=True)
        else:
            df = sweep.run(K_values, verbose=True)
        elapsed = time.perf_counter() - t0

        results[f'canonical_d{d}'] = {
            'time_seconds': elapsed,
            'n_K_values': len(K_values),
            'n_trials_per_K': config.n_hamiltonians * config.n_targets,
            'data': df.to_dict('records'),
        }
        print(f"  Time: {elapsed:.1f}s")

        # QubitGridModel (only for valid lattice dimensions)
        if d in [8, 16, 32, 64]:
            print(f"\nQubitGridModel d={d}:")
            model = QubitGridModel(dim=d, seed=42)
            sweep = DensitySweep(model, config)

            K_max_qg = model.K
            K_values_qg = sorted(set([
                max(2, int(0.10 * K_max_qg)),
                max(2, int(0.20 * K_max_qg)),
                max(2, int(0.30 * K_max_qg)),
                max(2, int(0.40 * K_max_qg)),
                max(2, int(0.50 * K_max_qg)),
                max(2, int(0.60 * K_max_qg)),
            ]))

            t0 = time.perf_counter()
            if use_parallel:
                df = sweep.run_parallel(K_values_qg, verbose=True)
            else:
                df = sweep.run(K_values_qg, verbose=True)
            elapsed = time.perf_counter() - t0

            results[f'qubitgrid_d{d}'] = {
                'time_seconds': elapsed,
                'n_K_values': len(K_values_qg),
                'n_trials_per_K': config.n_hamiltonians * config.n_targets,
                'data': df.to_dict('records'),
            }
            print(f"  Time: {elapsed:.1f}s")

    return results


def plot_results(results: dict, output_dir: Path, title_suffix: str = ""):
    """Generate comparison plots from benchmark results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crit_colors = {'moment': 'blue', 'spectral': 'red', 'krylov': 'green'}

    for key, data in results.items():
        if 'data' not in data or not data['data']:
            continue

        df = pd.DataFrame(data['data'])

        fig, ax = plt.subplots(figsize=(8, 5))

        for crit, color in crit_colors.items():
            col = f'{crit}_P'
            sem_col = f'{crit}_sem'
            if col in df.columns:
                ax.errorbar(df['rho'], df[col], yerr=df.get(sem_col, 0),
                           label=crit.capitalize(), color=color,
                           marker='o', capsize=3)

        ax.set_xlabel('Density ρ = K/d²')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(f'{key} {title_suffix}\n(time: {data["time_seconds"]:.1f}s)')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{key}.png', dpi=150)
        plt.close()

    print(f"\nPlots saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true', help='Use parallel execution')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[8, 12, 16])
    parser.add_argument('--output', type=str, default='benchmark_results')
    args = parser.parse_args()

    print(f"Running benchmark (parallel={args.parallel}, dimensions={args.dimensions})")

    results = run_benchmark(
        name='benchmark',
        use_parallel=args.parallel,
        dimensions=args.dimensions,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                'time_seconds': v['time_seconds'],
                'n_K_values': v['n_K_values'],
                'n_trials_per_K': v['n_trials_per_K'],
            }
        json.dump(json_results, f, indent=2)

    suffix = "(parallel)" if args.parallel else "(sequential)"
    plot_results(results, output_dir / 'plots', title_suffix=suffix)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_time = sum(r['time_seconds'] for r in results.values())
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    for key, data in results.items():
        print(f"  {key}: {data['time_seconds']:.1f}s")


if __name__ == '__main__':
    main()
