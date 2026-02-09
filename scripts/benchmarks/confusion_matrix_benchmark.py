#!/usr/bin/env python3
"""
Confusion Matrix Benchmark: Moment Criterion as Ground Truth.

Uses the moment criterion to establish ground truth for evaluating
spectral and Krylov classification performance.

- Reachable targets: constructed via |phi> = exp(-iH(lam)t)|psi>
- Unreachable targets: certified by moment criterion (provable certificate)

Produces:
  - Confusion matrix plots (fig/benchmarks/confusion_matrix_{ensemble}.png)
  - Accuracy vs time scatter (fig/benchmarks/accuracy_vs_time.png)
  - Iterations vs accuracy (fig/benchmarks/iterations_vs_accuracy.png)
  - LaTeX confusion matrix table (tex/confusion_matrix_table.tex)

Usage:
    python scripts/benchmarks/confusion_matrix_benchmark.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from reach import models, mathematics, optimize
import qutip

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'fig' / 'benchmarks'
TEX_DIR = Path(__file__).parent.parent.parent / 'tex'

GEO2_LATTICE = {
    8: {'nx': 1, 'ny': 3},
    16: {'nx': 2, 'ny': 2},
    32: {'nx': 1, 'ny': 5},
}

TAU = 0.99
N_REACHABLE = 100
N_UNREACHABLE = 100
MAXITER = 50
RESTARTS = 3
SEED_BASE = 123456


# =============================================================================
# Ground-Truth Generation
# =============================================================================

def generate_reachable_targets(d, K, ensemble, n_targets=N_REACHABLE, seed=SEED_BASE):
    """Generate targets that are reachable BY CONSTRUCTION."""
    results = []
    rng = np.random.RandomState(seed)

    for i in range(n_targets):
        trial_seed = seed + i
        psi = models.fock_state(d, 0)

        if ensemble == 'GEO2' and d in GEO2_LATTICE:
            hams = models.random_hamiltonian_ensemble(
                d, K, ensemble, seed=trial_seed, **GEO2_LATTICE[d])
        else:
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=trial_seed)

        # Random control parameters and evolution time
        lambda_0 = rng.uniform(-1, 1, K)
        t = rng.uniform(0.5, 5.0)

        # Evolve to get target (reachable by construction)
        H = sum(l * Hk for l, Hk in zip(lambda_0, hams))
        U = (-1j * H * t).expm()
        phi = U * psi

        results.append({
            'psi': psi, 'phi': phi, 'hams': hams,
            'K': K, 'ground_truth': 'reachable',
        })

    return results


def generate_unreachable_targets(d, K_small, ensemble, n_targets=N_UNREACHABLE,
                                  seed=SEED_BASE + 100000):
    """Generate targets that moment criterion certifies as UNREACHABLE."""
    from scipy.linalg import null_space

    results = []
    attempts = 0

    while len(results) < n_targets and attempts < 10 * n_targets:
        attempts += 1
        trial_seed = seed + attempts

        psi = models.fock_state(d, 0)

        if ensemble == 'GEO2' and d in GEO2_LATTICE:
            hams = models.random_hamiltonian_ensemble(
                d, K_small, ensemble, seed=trial_seed, **GEO2_LATTICE[d])
        else:
            hams = models.random_hamiltonian_ensemble(
                d, K_small, ensemble, seed=trial_seed)

        # Random target (not constructed via evolution)
        phi = models.random_states(1, d, seed=trial_seed + 50000)[0]

        # Check moment criterion
        K = len(hams)
        L = np.array([qutip.expect(Hk, phi) - qutip.expect(Hk, psi) for Hk in hams])
        kernel = null_space(L.reshape(1, -1))

        if kernel.size == 0:
            continue

        Q = np.zeros((K, K))
        for ii in range(K):
            for jj in range(K):
                anticomm = (hams[ii] * hams[jj] + hams[jj] * hams[ii]) / 2
                Q[ii, jj] = qutip.expect(anticomm, phi) - qutip.expect(anticomm, psi)

        Q_proj = kernel.T @ Q @ kernel
        eigvals = np.linalg.eigvalsh(Q_proj)
        unreachable = bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10))

        if unreachable:
            results.append({
                'psi': psi, 'phi': phi, 'hams': hams,
                'K': K_small, 'ground_truth': 'unreachable',
            })

    print(f"    Generated {len(results)}/{n_targets} unreachable targets "
          f"in {attempts} attempts")
    return results


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_targets(targets, tau=TAU, maxiter=MAXITER, restarts=RESTARTS):
    """Evaluate spectral and krylov on a set of targets. Returns list of dicts."""
    results = []

    for i, tgt in enumerate(targets):
        psi, phi, hams = tgt['psi'], tgt['phi'], tgt['hams']
        K = len(hams)
        d = psi.shape[0]
        m = min(K, d)

        # Spectral
        t0 = time.perf_counter()
        spec_res = optimize.maximize_spectral_overlap(
            psi, phi, hams, maxiter=maxiter, restarts=restarts,
            ftol=1e-6, seed=42 + i)
        spec_time = time.perf_counter() - t0

        # Krylov
        t0 = time.perf_counter()
        kryl_res = optimize.maximize_krylov_score(
            psi, phi, hams, m=m, maxiter=maxiter, restarts=restarts,
            ftol=1e-6, seed=42 + i)
        kryl_time = time.perf_counter() - t0

        results.append({
            'ground_truth': tgt['ground_truth'],
            'spectral_score': spec_res['best_value'],
            'spectral_reachable': spec_res['best_value'] >= tau,
            'spectral_time': spec_time,
            'krylov_score': kryl_res['best_value'],
            'krylov_reachable': kryl_res['best_value'] >= tau,
            'krylov_time': kryl_time,
            'd': d, 'K': K,
        })

    return results


def compute_confusion_matrix(results):
    """Compute TP/FP/TN/FN for spectral and krylov."""
    cm = {}
    for crit in ['spectral', 'krylov']:
        tp = sum(1 for r in results
                 if r['ground_truth'] == 'reachable' and r[f'{crit}_reachable'])
        fn = sum(1 for r in results
                 if r['ground_truth'] == 'reachable' and not r[f'{crit}_reachable'])
        fp = sum(1 for r in results
                 if r['ground_truth'] == 'unreachable' and r[f'{crit}_reachable'])
        tn = sum(1 for r in results
                 if r['ground_truth'] == 'unreachable' and not r[f'{crit}_reachable'])

        total = tp + fn + fp + tn
        cm[crit] = {
            'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'accuracy': (tp + tn) / total if total > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        }
    return cm


# =============================================================================
# Plotting: Confusion Matrices
# =============================================================================

def plot_confusion_matrices(all_cms, ensemble, output_path):
    """Plot confusion matrices: 2 rows (Spectral, Krylov) x N cols (dimensions)."""
    dims = sorted(all_cms.keys())
    n_dims = len(dims)

    fig, axes = plt.subplots(2, n_dims, figsize=(4 * n_dims, 7))
    if n_dims == 1:
        axes = axes.reshape(2, 1)

    for col, d in enumerate(dims):
        for row, crit in enumerate(['spectral', 'krylov']):
            ax = axes[row, col]
            cm = all_cms[d][crit]
            matrix = np.array([[cm['TP'], cm['FN']],
                               [cm['FP'], cm['TN']]])

            # Color map: green for correct, red for errors
            cmap = plt.cm.RdYlGn
            im = ax.imshow(matrix, cmap=cmap, vmin=0,
                           vmax=max(N_REACHABLE, N_UNREACHABLE))

            # Annotate cells
            labels = [['TP', 'FN'], ['FP', 'TN']]
            for i in range(2):
                for j in range(2):
                    total = matrix.sum()
                    pct = 100 * matrix[i, j] / total if total > 0 else 0
                    ax.text(j, i, f'{labels[i][j]}\n{matrix[i,j]}\n({pct:.0f}%)',
                            ha='center', va='center', fontsize=10,
                            fontweight='bold')

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pred\nReach', 'Pred\nUnreach'], fontsize=9)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Actual\nReach', 'Actual\nUnreach'], fontsize=9)

            acc = cm['accuracy']
            title = f'{crit.title()} d={d}\nAcc={acc:.0%}'
            ax.set_title(title, fontsize=11, fontweight='bold')

    fig.suptitle(f'{ensemble} Confusion Matrices (tau={TAU})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")


# =============================================================================
# Plotting: Accuracy vs Time
# =============================================================================

def plot_accuracy_vs_time(all_results, output_path):
    """Scatter: x=mean time (log), y=accuracy, by ensemble/criterion/dimension."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ens_colors = {'canonical': '#1f77b4', 'GEO2': '#d62728'}
    dim_sizes = {8: 60, 16: 100, 32: 160}

    for ens_name, ens_data in all_results.items():
        color = ens_colors.get(ens_name, 'gray')
        for d, d_data in sorted(ens_data.items()):
            size = dim_sizes.get(d, 80)
            cm = d_data['cm']

            # Spectral
            spec_time = np.mean([r['spectral_time'] for r in d_data['raw']])
            spec_acc = cm['spectral']['accuracy']
            ax.scatter(spec_time, spec_acc, marker='o', c=color, s=size,
                       alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'{ens_name} d={d} Spec' if d == min(dim_sizes) else '')

            # Krylov
            kryl_time = np.mean([r['krylov_time'] for r in d_data['raw']])
            kryl_acc = cm['krylov']['accuracy']
            ax.scatter(kryl_time, kryl_acc, marker='^', c=color, s=size,
                       alpha=0.8, edgecolors='black', linewidth=0.5,
                       label=f'{ens_name} d={d} Kryl' if d == min(dim_sizes) else '')

    ax.set_xscale('log')
    ax.set_xlabel('Mean evaluation time (s)', fontsize=13)
    ax.set_ylabel('Classification accuracy', fontsize=13)
    ax.set_ylim([0.75, 1.02])
    ax.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Spectral'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, label='Krylov'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4',
               markersize=8, label='Canonical'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
               markersize=8, label='GEO2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='d=8'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=9, label='d=16'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=12, label='d=32'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")


# =============================================================================
# Plotting: Iterations vs Accuracy (Part 6)
# =============================================================================

def run_iterations_sweep(d, K, ensemble, maxiters, n_trials=20, seed=77777):
    """Run spectral and krylov for varying maxiter values."""
    psi = models.fock_state(d, 0)

    if ensemble == 'GEO2' and d in GEO2_LATTICE:
        hams = models.random_hamiltonian_ensemble(
            d, K, ensemble, seed=seed, **GEO2_LATTICE[d])
    else:
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

    m = min(K, d)
    rng = np.random.RandomState(seed)

    # Generate reachable targets
    targets = []
    for i in range(n_trials):
        lam0 = rng.uniform(-1, 1, K)
        t = rng.uniform(0.5, 5.0)
        H = sum(l * Hk for l, Hk in zip(lam0, hams))
        U = (-1j * H * t).expm()
        phi = U * psi
        targets.append(phi)

    results = {'spectral': {}, 'krylov': {}}
    for mi in maxiters:
        spec_scores = []
        kryl_scores = []
        for i, phi in enumerate(targets):
            sr = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=mi, restarts=1, ftol=1e-6, seed=42 + i)
            spec_scores.append(sr['best_value'])

            kr = optimize.maximize_krylov_score(
                psi, phi, hams, m=m, maxiter=mi, restarts=1, ftol=1e-6, seed=42 + i)
            kryl_scores.append(kr['best_value'])

        results['spectral'][mi] = np.mean(spec_scores)
        results['krylov'][mi] = np.mean(kryl_scores)

    return results


def plot_iterations_vs_accuracy(iter_results, output_path):
    """2x2 plot: rows=Spectral/Krylov, cols=Canonical/GEO2."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    dim_colors = {8: '#1f77b4', 16: '#ff7f0e', 32: '#2ca02c'}
    dim_markers = {8: 'o', 16: 's', 32: '^'}

    for col, ensemble in enumerate(['canonical', 'GEO2']):
        if ensemble not in iter_results:
            continue
        for row, crit in enumerate(['spectral', 'krylov']):
            ax = axes[row, col]
            for d in sorted(iter_results[ensemble].keys()):
                data = iter_results[ensemble][d]
                maxiters = sorted(data[crit].keys())
                scores = [data[crit][mi] for mi in maxiters]
                color = dim_colors.get(d, 'gray')
                marker = dim_markers.get(d, 'o')
                ax.plot(maxiters, scores, color=color, marker=marker,
                        linewidth=2, markersize=6, label=f'd={d}')

            ax.set_ylabel(f'{crit.title()} score', fontsize=12)
            ax.set_ylim([0.8, 1.02])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            if row == 0:
                ax.set_title(ensemble, fontsize=13, fontweight='bold')
            if row == 1:
                ax.set_xlabel('maxiter', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")


# =============================================================================
# LaTeX Table Generation
# =============================================================================

def generate_confusion_table(all_ensemble_cms, output_path):
    """Generate LaTeX confusion matrix summary table."""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Classification performance using moment criterion as ground truth ($\tau = 0.99$).}',
        r'\label{tab:confusion_summary}',
        r'\begin{tabular}{ll|rrrr|rrrr}',
        r'\toprule',
        r' & & \multicolumn{4}{c|}{Canonical} & \multicolumn{4}{c}{GEO2} \\',
        r' & $d$ & TP & FP & TN & FN & TP & FP & TN & FN \\',
        r'\midrule',
    ]

    for crit in ['Spectral', 'Krylov']:
        crit_key = crit.lower()
        dims = sorted(set(
            list(all_ensemble_cms.get('canonical', {}).keys()) +
            list(all_ensemble_cms.get('GEO2', {}).keys())
        ))
        first = True
        for d in dims:
            prefix = f'\\multirow{{{len(dims)}}}{{*}}{{{crit}}}' if first else ''
            first = False

            can = all_ensemble_cms.get('canonical', {}).get(d, {}).get(crit_key, {})
            geo = all_ensemble_cms.get('GEO2', {}).get(d, {}).get(crit_key, {})

            can_str = (f"{can.get('TP', '-')} & {can.get('FP', '-')} & "
                       f"{can.get('TN', '-')} & {can.get('FN', '-')}")
            geo_str = (f"{geo.get('TP', '-')} & {geo.get('FP', '-')} & "
                       f"{geo.get('TN', '-')} & {geo.get('FN', '-')}")

            lines.append(f'{prefix} & $d={d}$ & {can_str} & {geo_str} \\\\')

        lines.append(r'\midrule')

    # Remove last \midrule and replace with \bottomrule
    lines[-1] = r'\bottomrule'

    lines.extend([
        r'\end{tabular}',
        r'\end{table}',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved {output_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONFUSION MATRIX BENCHMARK")
    print("=" * 70)

    all_ensemble_results = {}
    all_ensemble_cms = {}

    # ---------------------------------------------------------------
    # Part 3-4: Confusion matrices
    # ---------------------------------------------------------------
    for ensemble in ['canonical', 'GEO2']:
        print(f"\n{'='*50}")
        print(f"Ensemble: {ensemble}")
        print(f"{'='*50}")

        ensemble_cms = {}
        ensemble_raw = {}

        for d in [8, 16, 32]:
            K_reach = max(10, d)  # enough Hamiltonians for reachability
            K_unreach = 3  # minimal control for unreachability

            print(f"\n  d={d}: generating targets...")

            # Reachable targets
            reach_targets = generate_reachable_targets(
                d, K_reach, ensemble, n_targets=N_REACHABLE)
            print(f"    {len(reach_targets)} reachable targets (K={K_reach})")

            # Unreachable targets
            unreach_targets = generate_unreachable_targets(
                d, K_unreach, ensemble, n_targets=N_UNREACHABLE)

            if len(unreach_targets) < 5:
                print(f"    WARNING: Only {len(unreach_targets)} unreachable targets. "
                      f"Trying K={K_unreach-1}...")
                unreach_targets = generate_unreachable_targets(
                    d, max(2, K_unreach - 1), ensemble, n_targets=N_UNREACHABLE,
                    seed=SEED_BASE + 200000)

            all_targets = reach_targets + unreach_targets
            print(f"    Evaluating {len(all_targets)} targets...")

            # Evaluate
            raw_results = evaluate_targets(all_targets, maxiter=MAXITER, restarts=RESTARTS)

            # Compute confusion matrix
            cm = compute_confusion_matrix(raw_results)
            ensemble_cms[d] = cm
            ensemble_raw[d] = {'cm': cm, 'raw': raw_results}

            for crit in ['spectral', 'krylov']:
                c = cm[crit]
                print(f"    {crit.title()}: TP={c['TP']} FP={c['FP']} "
                      f"TN={c['TN']} FN={c['FN']} "
                      f"Acc={c['accuracy']:.1%}")

        all_ensemble_cms[ensemble] = ensemble_cms
        all_ensemble_results[ensemble] = ensemble_raw

        # Plot confusion matrices
        plot_confusion_matrices(
            ensemble_cms, ensemble,
            OUTPUT_DIR / f'confusion_matrix_{ensemble.lower()}.png')

    # ---------------------------------------------------------------
    # Part 4: LaTeX table
    # ---------------------------------------------------------------
    print("\nGenerating LaTeX table...")
    generate_confusion_table(all_ensemble_cms, TEX_DIR / 'confusion_matrix_table.tex')

    # ---------------------------------------------------------------
    # Part 5: Accuracy vs Time
    # ---------------------------------------------------------------
    print("\nPlotting accuracy vs time...")
    plot_accuracy_vs_time(all_ensemble_results, OUTPUT_DIR / 'accuracy_vs_time.png')

    # ---------------------------------------------------------------
    # Part 6: Iterations vs Accuracy
    # ---------------------------------------------------------------
    print("\nRunning iterations sweep...")
    maxiters = [5, 10, 20, 50, 100, 200]
    iter_results = {}

    for ensemble in ['canonical', 'GEO2']:
        print(f"  {ensemble}:")
        iter_results[ensemble] = {}
        for d in [8, 16, 32]:
            print(f"    d={d}...", end=' ', flush=True)
            data = run_iterations_sweep(d, K=10, ensemble=ensemble,
                                         maxiters=maxiters, n_trials=20)
            iter_results[ensemble][d] = data
            print("done")

    plot_iterations_vs_accuracy(iter_results, OUTPUT_DIR / 'iterations_vs_accuracy.png')

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
