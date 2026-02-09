#!/usr/bin/env python3
"""
Confusion Matrix Analysis for Reachability Criteria

Part 3.1 + 4.1: Generate confusion matrix plots showing TP/FP/TN/FN.
Part 4.2: Iterations vs accuracy tradeoff study.

Ground truth construction:
- REACHABLE: phi = exp(-i H(lambda_0) t) |psi> for known lambda_0
- UNREACHABLE: K=2, d=16+ random phi (control algebra too small)
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
from scipy.linalg import expm

from reach import models, mathematics, optimize

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'fig' / 'benchmarks'
TEX_DIR = Path(__file__).parent.parent.parent / 'tex'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 0.99
SEED_BASE = 700000

# Confusion matrix test grid
CM_GRID = {
    'dims': [8, 16, 32],
    'n_reachable': 30,
    'n_unreachable': 30,
    'K_unreachable': 2,  # K=2 for unreachable targets
    'K_reachable': 15,   # High K for reachable targets
}

# GEO2 lattice configs
GEO2_LATTICE = {
    8:  {'nx': 3, 'ny': 1},
    16: {'nx': 2, 'ny': 2},
    32: {'nx': 5, 'ny': 1},
}

# Iterations study grid
ITER_GRID = {
    'd': 16,
    'K': 10,
    'maxiters': [5, 10, 20, 50, 100, 200],
    'n_trials': 20,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_reachable_target(d, K, ensemble, seed, t=2.0):
    """Generate provably reachable target via time evolution."""
    import qutip
    psi = models.fock_state(d, 0)
    rng = np.random.RandomState(seed)
    lambda_0 = rng.uniform(-1, 1, K)

    if ensemble == 'GEO2':
        hams = models.random_hamiltonian_ensemble(d, K, 'GEO2', seed=seed, **GEO2_LATTICE[d])
    else:
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

    hams_np = [H.full() for H in hams]
    H_total = sum(lambda_0[k] * hams_np[k] for k in range(K))
    U = expm(-1j * H_total * t)
    psi_vec = psi.full().flatten()
    phi_vec = U @ psi_vec
    phi = qutip.Qobj(phi_vec.reshape(-1, 1))

    return psi, phi, hams


def generate_unreachable_target(d, K, ensemble, seed):
    """Generate likely unreachable target (small K, random phi)."""
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed)[0]

    if ensemble == 'GEO2':
        hams = models.random_hamiltonian_ensemble(d, K, 'GEO2', seed=seed, **GEO2_LATTICE[d])
    else:
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

    return psi, phi, hams


def evaluate_criteria(psi, phi, hams, tau=0.99, maxiter=100):
    """Evaluate all three criteria on a single target."""
    K = len(hams)
    m = min(K, hams[0].shape[0])

    # Spectral
    t0 = time.time()
    try:
        spec_res = optimize.maximize_spectral_overlap(psi, phi, hams, maxiter=maxiter, restarts=3)
        spec_score = spec_res['best_value']
        spec_reachable = spec_score >= tau
    except Exception:
        spec_score = 0.0
        spec_reachable = False
    spec_time = time.time() - t0

    # Krylov
    t0 = time.time()
    try:
        krylov_res = optimize.maximize_krylov_score(psi, phi, hams, m=m, maxiter=maxiter, restarts=3)
        krylov_score = krylov_res['best_value']
        krylov_reachable = krylov_score >= tau
    except Exception:
        krylov_score = 0.0
        krylov_reachable = False
    krylov_time = time.time() - t0

    # Moment (analytical, no optimization)
    t0 = time.time()
    try:
        moment_result = mathematics.moment_criterion(hams, psi, phi)
        moment_reachable = not moment_result  # moment_criterion returns True if UNREACHABLE
    except Exception:
        moment_reachable = True  # Assume reachable if criterion fails
    moment_time = time.time() - t0

    return {
        'spectral_score': spec_score,
        'spectral_reachable': spec_reachable,
        'spectral_time': spec_time,
        'krylov_score': krylov_score,
        'krylov_reachable': krylov_reachable,
        'krylov_time': krylov_time,
        'moment_reachable': moment_reachable,
        'moment_time': moment_time,
    }


def compute_confusion_matrix(ensemble):
    """Compute confusion matrix for one ensemble."""
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX - {ensemble}")
    print(f"{'='*60}")

    results = []

    for d in CM_GRID['dims']:
        if d not in GEO2_LATTICE and ensemble == 'GEO2':
            continue

        print(f"\n--- d={d} ---")

        # Generate reachable targets
        print(f"  Generating {CM_GRID['n_reachable']} reachable targets (K={CM_GRID['K_reachable']})...")
        for i in range(CM_GRID['n_reachable']):
            seed = SEED_BASE + d * 10000 + i
            psi, phi, hams = generate_reachable_target(d, CM_GRID['K_reachable'], ensemble, seed)
            eval_result = evaluate_criteria(psi, phi, hams, tau=TAU)
            eval_result['d'] = d
            eval_result['actual_reachable'] = True
            eval_result['ensemble'] = ensemble
            results.append(eval_result)

        # Generate unreachable targets
        print(f"  Generating {CM_GRID['n_unreachable']} unreachable targets (K={CM_GRID['K_unreachable']})...")
        for i in range(CM_GRID['n_unreachable']):
            seed = SEED_BASE + d * 10000 + 5000 + i
            psi, phi, hams = generate_unreachable_target(d, CM_GRID['K_unreachable'], ensemble, seed)
            eval_result = evaluate_criteria(psi, phi, hams, tau=TAU)
            eval_result['d'] = d
            eval_result['actual_reachable'] = False
            eval_result['ensemble'] = ensemble
            results.append(eval_result)

    return pd.DataFrame(results)


def compute_metrics(df, criterion):
    """Compute TP, FP, TN, FN and derived metrics for a criterion."""
    pred_col = f'{criterion}_reachable'
    actual = df['actual_reachable'].values
    predicted = df[pred_col].values

    TP = np.sum(actual & predicted)
    FP = np.sum(~actual & predicted)
    TN = np.sum(~actual & ~predicted)
    FN = np.sum(actual & ~predicted)

    total = len(df)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Sensitivity/Recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) > 0 else 0
    Accuracy = (TP + TN) / total if total > 0 else 0

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'TPR': TPR, 'FPR': FPR, 'TNR': TNR, 'FNR': FNR,
        'Precision': Precision, 'F1': F1, 'Accuracy': Accuracy,
        'Total': total,
    }


def plot_confusion_matrix_single(df, ensemble, output_path):
    """Create 3x1 confusion matrix subplot for one ensemble."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    criteria = ['spectral', 'krylov', 'moment']
    titles = ['Spectral (S*)', 'Krylov (R*)', 'Moment']

    for ax, criterion, title in zip(axes, criteria, titles):
        metrics = compute_metrics(df, criterion)
        cm = np.array([[metrics['TN'], metrics['FP']],
                       [metrics['FN'], metrics['TP']]])

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred Unreach.', 'Pred Reach.'],
                    yticklabels=['Actual Unreach.', 'Actual Reach.'],
                    cbar=False)

        # Add percentages
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / metrics['Total'] * 100
                ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                        ha='center', va='center', fontsize=10, color='gray')

        ax.set_title(f'{title}\nTPR={metrics["TPR"]:.2f}, F1={metrics["F1"]:.2f}', fontsize=12)

    plt.suptitle(f'Confusion Matrices - {ensemble} (tau={TAU})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_confusion_matrix_table(df_canonical, df_geo2):
    """Generate LaTeX table for confusion matrix results."""
    rows = []

    for ensemble, df in [('canonical', df_canonical), ('GEO2', df_geo2)]:
        if df is None or len(df) == 0:
            continue
        for criterion in ['spectral', 'krylov', 'moment']:
            metrics = compute_metrics(df, criterion)
            rows.append({
                'Ensemble': ensemble,
                'Criterion': criterion.capitalize(),
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'TN': metrics['TN'],
                'FN': metrics['FN'],
                'TPR': f"{metrics['TPR']:.3f}",
                'FPR': f"{metrics['FPR']:.3f}",
                'F1': f"{metrics['F1']:.3f}",
            })

    df_table = pd.DataFrame(rows)

    # Generate LaTeX
    latex = r"""\begin{table}[htbp]
\centering
\caption{Confusion matrix metrics for reachability criteria ($\tau = 0.99$). TPR = True Positive Rate, FPR = False Positive Rate.}
\label{tab:confusion_matrix}
\begin{tabular}{llrrrrrrrr}
\toprule
Ensemble & Criterion & TP & FP & TN & FN & TPR & FPR & F1 \\
\midrule
"""
    for _, row in df_table.iterrows():
        latex += f"  {row['Ensemble']} & {row['Criterion']} & {row['TP']} & {row['FP']} & {row['TN']} & {row['FN']} & {row['TPR']} & {row['FPR']} & {row['F1']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    tex_path = TEX_DIR / 'confusion_matrix_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {tex_path}")

    return df_table


# =============================================================================
# PART 4.2: ITERATIONS VS ACCURACY
# =============================================================================

def run_iterations_study():
    """Study effect of maxiter on optimization quality."""
    print(f"\n{'='*60}")
    print("ITERATIONS VS ACCURACY STUDY")
    print(f"{'='*60}")

    d = ITER_GRID['d']
    K = ITER_GRID['K']
    m = min(K, d)

    results = []

    for ensemble in ['canonical', 'GEO2']:
        print(f"\n--- {ensemble} d={d} K={K} ---")

        for maxiter in ITER_GRID['maxiters']:
            spectral_scores = []
            krylov_scores = []
            spectral_times = []
            krylov_times = []

            for trial in range(ITER_GRID['n_trials']):
                seed = SEED_BASE + 100000 + maxiter * 1000 + trial

                # Generate reachable target
                if ensemble == 'GEO2':
                    psi, phi, hams = generate_reachable_target(d, K, 'GEO2', seed)
                else:
                    psi, phi, hams = generate_reachable_target(d, K, 'canonical', seed)

                # Spectral
                t0 = time.time()
                res = optimize.maximize_spectral_overlap(psi, phi, hams, maxiter=maxiter, restarts=3)
                spectral_scores.append(res['best_value'])
                spectral_times.append(time.time() - t0)

                # Krylov
                t0 = time.time()
                res = optimize.maximize_krylov_score(psi, phi, hams, m=m, maxiter=maxiter, restarts=3)
                krylov_scores.append(res['best_value'])
                krylov_times.append(time.time() - t0)

            results.append({
                'ensemble': ensemble,
                'maxiter': maxiter,
                'spectral_mean': np.mean(spectral_scores),
                'spectral_std': np.std(spectral_scores),
                'spectral_time': np.mean(spectral_times),
                'krylov_mean': np.mean(krylov_scores),
                'krylov_std': np.std(krylov_scores),
                'krylov_time': np.mean(krylov_times),
            })

            print(f"  maxiter={maxiter:3d}: S*={np.mean(spectral_scores):.4f} ({np.mean(spectral_times):.2f}s) | "
                  f"R*={np.mean(krylov_scores):.4f} ({np.mean(krylov_times):.2f}s)")

    return pd.DataFrame(results)


def plot_iterations_vs_accuracy(df):
    """Plot iterations vs accuracy/time tradeoff."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'canonical': '#1f77b4', 'GEO2': '#d62728'}
    linestyles = {'spectral': '-', 'krylov': '--'}

    for ax, criterion in zip(axes, ['spectral', 'krylov']):
        for ensemble in ['canonical', 'GEO2']:
            sub = df[df['ensemble'] == ensemble]

            # Score vs maxiter
            ax.errorbar(sub['maxiter'], sub[f'{criterion}_mean'],
                        yerr=sub[f'{criterion}_std'],
                        color=colors[ensemble],
                        marker='o', capsize=3,
                        label=f'{ensemble}')

        ax.set_xlabel('maxiter', fontsize=12)
        ax.set_ylabel(f'Mean {"S*" if criterion == "spectral" else "R*"}', fontsize=12)
        ax.set_title(f'{"Spectral" if criterion == "spectral" else "Krylov"} Score vs Iterations', fontsize=14)
        ax.legend()
        ax.set_ylim(0.95, 1.005)
        ax.grid(True, alpha=0.3)
        ax.axhline(TAU, color='gray', linestyle=':', alpha=0.5, label=f'tau={TAU}')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'iterations_vs_accuracy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_iterations_table(df):
    """Generate LaTeX table for iterations study."""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Effect of maxiter on optimizer convergence (d=16, K=10, 20 trials). Shows mean score and wall time.}
\label{tab:optimizer_iterations}
\begin{tabular}{llrrrr}
\toprule
Ensemble & maxiter & Spectral S* & Time (s) & Krylov R* & Time (s) \\
\midrule
"""
    for _, row in df.iterrows():
        latex += f"  {row['ensemble']} & {row['maxiter']} & {row['spectral_mean']:.4f} & {row['spectral_time']:.2f} & {row['krylov_mean']:.4f} & {row['krylov_time']:.2f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    tex_path = TEX_DIR / 'optimizer_iterations_table.tex'
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"Saved: {tex_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CONFUSION MATRIX & ITERATIONS ANALYSIS - v24")
    print("=" * 70)

    # Part 3.1 + 4.1: Confusion matrices
    print("\n[1/4] Computing confusion matrix for canonical...")
    df_canonical = compute_confusion_matrix('canonical')

    print("\n[2/4] Computing confusion matrix for GEO2...")
    df_geo2 = compute_confusion_matrix('GEO2')

    # Plot confusion matrices
    print("\n[3/4] Generating confusion matrix plots...")
    plot_confusion_matrix_single(df_canonical, 'canonical', OUTPUT_DIR / 'confusion_matrix_canonical.png')
    plot_confusion_matrix_single(df_geo2, 'GEO2', OUTPUT_DIR / 'confusion_matrix_geo2.png')

    # Generate LaTeX table
    generate_confusion_matrix_table(df_canonical, df_geo2)

    # Part 4.2: Iterations study
    print("\n[4/4] Running iterations vs accuracy study...")
    df_iter = run_iterations_study()
    plot_iterations_vs_accuracy(df_iter)
    generate_iterations_table(df_iter)

    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'confusion_canonical': df_canonical.to_dict('records'),
        'confusion_geo2': df_geo2.to_dict('records'),
        'iterations': df_iter.to_dict('records'),
    }
    json_path = OUTPUT_DIR / 'confusion_matrix_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all results: {json_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
