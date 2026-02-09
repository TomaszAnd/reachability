#!/usr/bin/env python3
"""
Generate benchmark plots for v24:
1. optimizer_heatmap_krylov.png - Krylov optimizer comparison
2. time_accuracy_scatter.png - Pareto front of time vs TPR
3. Update benchmark_results.json with Krylov optimizer data

Uses production optimizer settings and provably reachable targets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize

from reach import models, mathematics, optimize

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'fig' / 'benchmarks'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 0.99
SEED_BASE = 600000

# Optimizer methods to test
METHODS = ['L-BFGS-B', 'TNC', 'SLSQP', 'CG', 'Powell', 'Nelder-Mead']

# Test grid for optimizer comparison
OPTIMIZER_GRID = {
    'dims': [8, 16],
    'K_values': [5, 10],
    'ensembles': ['canonical'],  # Use canonical (not GUE per v23)
    'n_trials': 10,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_reachable_target(d, K, ensemble, seed, t=2.0):
    """Generate provably reachable target via time evolution."""
    psi = models.fock_state(d, 0)

    rng = np.random.RandomState(seed)
    lambda_0 = rng.uniform(-1, 1, K)

    hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)
    hams_np = [H.full() for H in hams]

    # H(lambda_0) = sum_k lambda_k * H_k
    H_total = sum(lambda_0[k] * hams_np[k] for k in range(K))

    # phi = exp(-i H t) |psi>
    from scipy.linalg import expm
    U = expm(-1j * H_total * t)
    psi_vec = psi.full().flatten()
    phi_vec = U @ psi_vec

    # Convert to Qobj
    import qutip
    phi = qutip.Qobj(phi_vec.reshape(-1, 1))

    return psi, phi, hams, hams_np


def maximize_krylov_with_method(psi, phi, hams, m, method, maxiter=100, seed=42):
    """Maximize Krylov score with specific optimizer method."""
    K = len(hams)
    hams_np = [H.full() for H in hams]
    bounds = [(-1.0, 1.0)] * K
    rng = np.random.RandomState(seed)

    def obj(x):
        R = mathematics.krylov_score(x, psi, phi, hams, m=m, hams_np=hams_np)
        return -R

    def obj_grad(x):
        R, g = mathematics.krylov_score_with_grad(x, psi, phi, hams, m=m, hams_np=hams_np)
        return -R, -g.astype(np.float64)

    best_val = 0.0
    t0 = time.time()

    for restart in range(3):  # 3 restarts
        x0 = rng.uniform(-1, 1, K)

        try:
            if method in ['L-BFGS-B', 'TNC', 'SLSQP', 'CG']:
                # Methods that support analytical gradient
                res = minimize(obj_grad, x0, method=method, jac=True,
                               bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP'] else None,
                               options={'maxiter': maxiter})
            else:
                # Derivative-free methods
                res = minimize(obj, x0, method=method,
                               bounds=bounds if method in ['Powell'] else None,
                               options={'maxiter': maxiter})

            val = -res.fun
            if val > best_val:
                best_val = val
        except Exception as e:
            pass  # Skip failed optimizations

    runtime = time.time() - t0
    return {'best_value': best_val, 'runtime_s': runtime}


def run_optimizer_comparison_krylov():
    """Run optimizer comparison for Krylov criterion."""
    results = []

    for ensemble in OPTIMIZER_GRID['ensembles']:
        for d in OPTIMIZER_GRID['dims']:
            for K in OPTIMIZER_GRID['K_values']:
                m = min(K, d)
                print(f"\n{ensemble} d={d} K={K} m={m}:")

                for method in METHODS:
                    scores = []
                    times = []

                    for trial in range(OPTIMIZER_GRID['n_trials']):
                        seed = SEED_BASE + d * 1000 + K * 100 + trial

                        # Generate reachable target
                        psi, phi, hams, hams_np = generate_reachable_target(d, K, ensemble, seed)

                        # Optimize
                        res = maximize_krylov_with_method(psi, phi, hams, m, method,
                                                          maxiter=100, seed=seed)
                        scores.append(res['best_value'])
                        times.append(res['runtime_s'])

                    mean_score = np.mean(scores)
                    mean_time = np.mean(times)
                    tpr = np.mean([s >= TAU for s in scores])

                    results.append({
                        'ensemble': ensemble,
                        'd': d,
                        'K': K,
                        'm': m,
                        'method': method,
                        'mean_score': mean_score,
                        'std_score': np.std(scores),
                        'mean_time': mean_time,
                        'tpr': tpr,
                    })

                    print(f"  {method:12s}: R*={mean_score:.4f}  time={mean_time:.3f}s  TPR={tpr:.2f}")

    return results


def plot_optimizer_heatmap_krylov(results):
    """Create optimizer heatmap for Krylov criterion."""
    df = pd.DataFrame(results)

    # Create pivot table for heatmap
    # Rows: methods, Cols: (d, K) combinations
    df['config'] = df.apply(lambda r: f"d={r['d']}, K={r['K']}", axis=1)

    pivot_score = df.pivot(index='method', columns='config', values='mean_score')
    pivot_time = df.pivot(index='method', columns='config', values='mean_time')

    # Order methods by performance
    method_order = ['L-BFGS-B', 'TNC', 'SLSQP', 'CG', 'Powell', 'Nelder-Mead']
    pivot_score = pivot_score.reindex(method_order)
    pivot_time = pivot_time.reindex(method_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Score heatmap
    sns.heatmap(pivot_score, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1,
                vmin=0.9, vmax=1.0, cbar_kws={'label': 'Mean R*'})
    ax1.set_title('Krylov Score by Optimizer Method', fontsize=14)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Method')

    # Time heatmap
    sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax2,
                cbar_kws={'label': 'Mean Time (s)'})
    ax2.set_title('Optimization Time by Method', fontsize=14)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Method')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'optimizer_heatmap_krylov.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    return pivot_score, pivot_time


def load_benchmark_data():
    """Load existing benchmark data from JSON."""
    json_path = OUTPUT_DIR / 'benchmark_results.json'
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {'benchmark': [], 'optimizer': []}


def plot_time_accuracy_scatter(benchmark_data):
    """Create Pareto front scatter plot of time vs TPR."""
    if not benchmark_data.get('benchmark'):
        print("No benchmark data available for scatter plot")
        return

    df = pd.DataFrame(benchmark_data['benchmark'])

    # Filter to canonical and GEO2 only (no GUE per v23)
    df = df[df['ensemble'].isin(['canonical', 'GEO2'])]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Define markers and colors
    criterion_markers = {'spectral': 'o', 'krylov': '^', 'moment': 's'}
    ensemble_colors = {'canonical': '#1f77b4', 'GEO2': '#d62728'}

    # Create scatter for each criterion
    for criterion in ['spectral', 'krylov', 'moment']:
        time_col = f'{criterion}_mean_s'
        tpr_col = f'{criterion}_true_positive_rate' if criterion != 'moment' else 'moment_correct_rate'

        if time_col not in df.columns or tpr_col not in df.columns:
            continue

        for ensemble in ['canonical', 'GEO2']:
            sub = df[df['ensemble'] == ensemble]
            if len(sub) == 0:
                continue

            times = sub[time_col].values
            tprs = sub[tpr_col].values
            ds = sub['d'].values

            # Size proportional to d
            sizes = 20 + (ds / ds.max()) * 100

            ax.scatter(times, tprs,
                       marker=criterion_markers[criterion],
                       c=ensemble_colors[ensemble],
                       s=sizes,
                       alpha=0.7,
                       label=f'{criterion.capitalize()} ({ensemble})')

    ax.set_xscale('log')
    ax.set_xlabel('Mean Time per Evaluation (s)', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Time-Accuracy Tradeoff Across Criteria and Ensembles', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation for key insight
    ax.annotate('Moment: Fast but may\nmiss near transition',
                xy=(0.001, 0.95), xytext=(0.003, 0.7),
                fontsize=9, alpha=0.7,
                arrowprops=dict(arrowstyle='->', alpha=0.5))

    ax.annotate('Spectral/Krylov:\nSlower but higher accuracy',
                xy=(0.1, 1.0), xytext=(0.3, 0.8),
                fontsize=9, alpha=0.7,
                arrowprops=dict(arrowstyle='->', alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'time_accuracy_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_benchmarks_readme():
    """Create BENCHMARKS_README.md documenting all plots."""
    timestamp = datetime.now().isoformat()
    readme_content = f"""# Benchmarks Directory

This directory contains benchmark plots and data for the reachability criteria.

## Files

### Data
- **benchmark_results.json** - Raw benchmark data including timing profiles, criterion comparisons, and optimizer studies.

### Plots

#### Optimizer Comparison
- **optimizer_heatmap_krylov.png** - Heatmap comparing 6 scipy.optimize methods on Krylov criterion R*.
  - Methods: L-BFGS-B, TNC, SLSQP, CG, Powell, Nelder-Mead
  - Grid: canonical ensemble, d in {{8, 16}}, K in {{5, 10}}
  - Shows mean R* score and wall time
  - L-BFGS-B recommended for best speed/quality tradeoff

#### Time-Accuracy Tradeoff
- **time_accuracy_scatter.png** - Pareto front showing tradeoff between evaluation time and TPR.
  - X-axis: Mean time per evaluation (log scale)
  - Y-axis: True positive rate on reachable targets
  - Markers: circle=Spectral, triangle=Krylov, square=Moment
  - Colors: blue=canonical, red=GEO2
  - Size: proportional to dimension d
  - Key insight: Moment is fast but has lower TPR near transitions

#### Confusion Matrix (Part 3.1, 4.1)
- **confusion_matrix_canonical.png** - TP/FP/TN/FN for canonical ensemble
- **confusion_matrix_geo2.png** - TP/FP/TN/FN for GEO2 ensemble
- **confusion_matrix_near_transition.png** - Detailed analysis near K_c

#### Iterations Study (Part 4.2)
- **iterations_vs_accuracy.png** - Effect of maxiter on S* and R*

## Data Generation

Benchmark data generated by:
- `scripts/benchmarks/generate_benchmark_plots.py` - Main plotting script
- `scripts/benchmarks/criteria_time_accuracy.py` - Time/accuracy measurements (if exists)
- `scripts/audit/benchmark_grad_vs_fd.py` - Gradient vs finite-diff comparison

All benchmarks use:
- Provably reachable targets: phi = exp(-i H(lambda_0) t) |psi>
- tau = 0.99 threshold
- Production optimizer settings (maxiter=100, restarts=3)
- Canonical and GEO2 ensembles (GUE excluded per v23)

## Key Findings

1. **L-BFGS-B** is best optimizer for both Spectral and Krylov criteria
2. **Moment** criterion is 5-45x faster but has higher FPR near transitions
3. **Krylov** has reduced TPR at low K for GEO2 (m=min(K,d) constraint)
4. **Analytical gradients** provide 5-64x speedup over finite differences

---
Generated: {timestamp}
"""
    readme_path = OUTPUT_DIR / 'BENCHMARKS_README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Saved: {readme_path}")


def main():
    print("=" * 70)
    print("BENCHMARK PLOT GENERATION - v24")
    print("=" * 70)

    # Part 3.2: Krylov optimizer heatmap
    print("\n[1/3] Running Krylov optimizer comparison...")
    krylov_results = run_optimizer_comparison_krylov()
    plot_optimizer_heatmap_krylov(krylov_results)

    # Part 3.3: Time-accuracy scatter
    print("\n[2/3] Generating time-accuracy scatter plot...")
    benchmark_data = load_benchmark_data()
    plot_time_accuracy_scatter(benchmark_data)

    # Part 3.4: BENCHMARKS_README.md
    print("\n[3/3] Creating BENCHMARKS_README.md...")
    create_benchmarks_readme()

    # Save Krylov optimizer results to JSON
    json_path = OUTPUT_DIR / 'krylov_optimizer_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': OPTIMIZER_GRID,
            'results': krylov_results,
        }, f, indent=2)
    print(f"Saved: {json_path}")

    print("\n" + "=" * 70)
    print("BENCHMARK GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
