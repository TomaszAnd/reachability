#!/usr/bin/env python3
"""
Analyze Fixed vs Optimized λ gap for Spectral vs Krylov criteria.

KEY QUESTION: Why is Krylov nearly λ-independent while Spectral strongly depends on λ?

ANALYSIS:
1. Extract Fixed and Optimized scores from GEO2 data
2. Compute gap: S*(optimized) - S*(fixed) and R*(optimized) - R*(fixed)
3. Correlate with dimension and operator density

HYPOTHESIS:
- Krylov's projection-based measure depends only on subspace spanned by {H^k|ψ⟩}
- This subspace is geometrically constrained, doesn't change much with λ
- Spectral's eigenbasis alignment can vary dramatically with λ

Usage:
    python scripts/analysis/fixed_vs_optimized.py

Author: Claude Code (research exploration)
Date: 2026-01-13
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DATA LOADING
# =============================================================================

def load_geo2_data() -> Optional[Dict]:
    """Load GEO2 production data with both Fixed and Optimized results."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    files = sorted(data_dir.glob("geo2_production_complete_*.pkl"))
    if not files:
        print("Warning: No GEO2 data found")
        return None

    with open(files[-1], 'rb') as f:
        return pickle.load(f)


def extract_scores_vs_K(data: Dict, d: int, criterion: str, approach: str, tau: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract score values vs K for given configuration.

    Returns:
        K values and corresponding scores (mean over trials)
    """
    if approach not in data['results']:
        return np.array([]), np.array([])

    if d not in data['results'][approach]:
        return np.array([]), np.array([])

    key = (d, tau, criterion)
    if key not in data['results'][approach][d]['data']:
        return np.array([]), np.array([])

    crit_data = data['results'][approach][d]['data'][key]
    K = np.array(crit_data['K'])

    # Get scores if available
    if 'scores' in crit_data:
        scores = np.array(crit_data['scores'])  # Mean scores
        return K, scores

    # Otherwise return probability data
    P = np.array(crit_data['p'])
    return K, P


# =============================================================================
# GAP ANALYSIS
# =============================================================================

def compute_lambda_gap(data: Dict) -> Dict:
    """
    Compute gap between Optimized and Fixed λ for both criteria.

    Gap = Score(optimized) - Score(fixed) at each K
    """
    results = {
        "dims": [],
        "spectral": {"gap_mean": [], "gap_std": [], "gap_at_transition": []},
        "krylov": {"gap_mean": [], "gap_std": [], "gap_at_transition": []},
    }

    print("\n" + "=" * 70)
    print("FIXED vs OPTIMIZED λ GAP ANALYSIS")
    print("=" * 70)

    for d in [16, 32, 64]:
        results["dims"].append(d)

        print(f"\n--- d = {d} ---")

        for criterion in ['spectral', 'krylov']:
            K_fixed, P_fixed = extract_scores_vs_K(data, d, criterion, 'fixed')
            K_opt, P_opt = extract_scores_vs_K(data, d, criterion, 'optimized')

            if len(K_fixed) == 0 or len(K_opt) == 0:
                results[criterion]["gap_mean"].append(np.nan)
                results[criterion]["gap_std"].append(np.nan)
                results[criterion]["gap_at_transition"].append(np.nan)
                continue

            # Align K values
            K_common = np.intersect1d(K_fixed, K_opt)
            if len(K_common) == 0:
                results[criterion]["gap_mean"].append(np.nan)
                results[criterion]["gap_std"].append(np.nan)
                results[criterion]["gap_at_transition"].append(np.nan)
                continue

            # Get P values for common K
            P_f = P_fixed[np.isin(K_fixed, K_common)]
            P_o = P_opt[np.isin(K_opt, K_common)]

            # Gap = P(unreachable|fixed) - P(unreachable|optimized)
            # If optimized helps, this is POSITIVE (higher unreachable at fixed)
            gap = P_f - P_o

            results[criterion]["gap_mean"].append(np.mean(gap))
            results[criterion]["gap_std"].append(np.std(gap))

            # Gap at transition (where P ≈ 0.5)
            trans_idx = np.argmin(np.abs(P_o - 0.5))
            results[criterion]["gap_at_transition"].append(gap[trans_idx] if trans_idx < len(gap) else np.nan)

            print(f"  {criterion.capitalize()}:")
            print(f"    Mean gap: {np.mean(gap):.4f} ± {np.std(gap):.4f}")
            print(f"    Gap at transition: {gap[trans_idx]:.4f}" if trans_idx < len(gap) else "    Gap at transition: N/A")
            print(f"    Max gap: {np.max(gap):.4f} at K={K_common[np.argmax(gap)]}")

    return results


def analyze_gap_scaling(gap_results: Dict) -> Dict:
    """Analyze how the gap scales with dimension."""
    dims = np.array(gap_results["dims"])

    analysis = {}

    print("\n" + "=" * 70)
    print("GAP SCALING ANALYSIS")
    print("=" * 70)

    for criterion in ['spectral', 'krylov']:
        gap_mean = np.array(gap_results[criterion]["gap_mean"])
        gap_trans = np.array(gap_results[criterion]["gap_at_transition"])

        # Filter valid values
        valid = ~np.isnan(gap_mean)
        if np.sum(valid) < 2:
            print(f"\n{criterion.capitalize()}: Insufficient data for scaling analysis")
            continue

        # Correlation with dimension
        r_mean, p_mean = pearsonr(dims[valid], gap_mean[valid])
        r_trans, p_trans = pearsonr(dims[valid], gap_trans[valid]) if np.sum(~np.isnan(gap_trans)) >= 2 else (np.nan, np.nan)

        analysis[criterion] = {
            "r_mean_vs_d": r_mean,
            "p_mean_vs_d": p_mean,
            "r_trans_vs_d": r_trans,
            "p_trans_vs_d": p_trans,
        }

        print(f"\n{criterion.capitalize()}:")
        print(f"  Gap-mean vs d: r={r_mean:.3f}, p={p_mean:.3f}")
        print(f"  Gap-trans vs d: r={r_trans:.3f}, p={p_trans:.3f}" if not np.isnan(r_trans) else "  Gap-trans vs d: N/A")

        # Interpretation
        if np.abs(r_mean) > 0.8:
            trend = "increases" if r_mean > 0 else "decreases"
            print(f"  → Strong trend: Gap {trend} with dimension")
        elif np.abs(r_mean) < 0.3:
            print(f"  → Weak trend: Gap nearly dimension-independent")

    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_gap_analysis(data: Dict, output_dir: Path):
    """Create comprehensive gap analysis figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, d in enumerate([16, 32, 64]):
        # Top row: P(unreachable) curves
        ax = axes[0, idx]
        ax.set_title(f"d={d}: P(unreachable) vs K", fontsize=10)

        for criterion, color in [('spectral', 'C0'), ('krylov', 'C1')]:
            K_fixed, P_fixed = extract_scores_vs_K(data, d, criterion, 'fixed')
            K_opt, P_opt = extract_scores_vs_K(data, d, criterion, 'optimized')

            if len(K_fixed) > 0:
                ax.plot(K_fixed, P_fixed, 'o--', color=color, alpha=0.5,
                       label=f'{criterion.capitalize()} Fixed')
            if len(K_opt) > 0:
                ax.plot(K_opt, P_opt, 's-', color=color,
                       label=f'{criterion.capitalize()} Optimized')

        ax.set_xlabel('K')
        ax.set_ylabel('P(unreachable)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        # Bottom row: Gap curves
        ax = axes[1, idx]
        ax.set_title(f"d={d}: Gap (Fixed - Optimized)", fontsize=10)

        for criterion, color in [('spectral', 'C0'), ('krylov', 'C1')]:
            K_fixed, P_fixed = extract_scores_vs_K(data, d, criterion, 'fixed')
            K_opt, P_opt = extract_scores_vs_K(data, d, criterion, 'optimized')

            if len(K_fixed) == 0 or len(K_opt) == 0:
                continue

            K_common = np.intersect1d(K_fixed, K_opt)
            if len(K_common) == 0:
                continue

            P_f = P_fixed[np.isin(K_fixed, K_common)]
            P_o = P_opt[np.isin(K_opt, K_common)]
            gap = P_f - P_o

            ax.plot(K_common, gap, 'o-', color=color, label=f'{criterion.capitalize()}')

        ax.set_xlabel('K')
        ax.set_ylabel('Gap = P(fixed) - P(optimized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Fixed vs Optimized λ: Gap Analysis for GEO2', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "fixed_vs_optimized_gap.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def plot_gap_summary(gap_results: Dict, output_dir: Path):
    """Plot gap summary across dimensions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dims = np.array(gap_results["dims"])

    # Left: Mean gap vs d
    ax = axes[0]
    for criterion, color, marker in [('spectral', 'C0', 'o'), ('krylov', 'C1', 's')]:
        gap_mean = np.array(gap_results[criterion]["gap_mean"])
        gap_std = np.array(gap_results[criterion]["gap_std"])

        valid = ~np.isnan(gap_mean)
        ax.errorbar(dims[valid], gap_mean[valid], yerr=gap_std[valid],
                   fmt=f'{marker}-', color=color, label=criterion.capitalize(),
                   capsize=5, markersize=10)

    ax.set_xlabel('Dimension d', fontsize=12)
    ax.set_ylabel('Mean Gap (Fixed - Optimized)', fontsize=12)
    ax.set_title('Mean λ-Optimization Gap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    # Right: Gap at transition vs d
    ax = axes[1]
    for criterion, color, marker in [('spectral', 'C0', 'o'), ('krylov', 'C1', 's')]:
        gap_trans = np.array(gap_results[criterion]["gap_at_transition"])

        valid = ~np.isnan(gap_trans)
        ax.plot(dims[valid], gap_trans[valid], f'{marker}-', color=color,
               label=criterion.capitalize(), markersize=10)

    ax.set_xlabel('Dimension d', fontsize=12)
    ax.set_ylabel('Gap at Transition (P=0.5)', fontsize=12)
    ax.set_title('λ-Optimization Gap at Transition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('GEO2: Fixed vs Optimized λ Gap Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "gap_scaling_summary.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIXED vs OPTIMIZED λ ANALYSIS")
    print("=" * 70)

    # Load data
    data = load_geo2_data()

    if data is None:
        print("\nNo data available. Run GEO2 production sweep first.")
        exit(1)

    # Compute gap
    gap_results = compute_lambda_gap(data)

    # Analyze scaling
    scaling_analysis = analyze_gap_scaling(gap_results)

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"fixed_vs_optimized_analysis_{timestamp}.pkl"

    results = {
        "gap_results": gap_results,
        "scaling_analysis": scaling_analysis,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results: {output_file}")

    # Generate plots
    fig_dir = Path(__file__).parent.parent.parent / "fig" / "analysis"
    plot_gap_analysis(data, fig_dir)
    plot_gap_summary(gap_results, fig_dir)

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
INTERPRETATION:
- Gap > 0: Optimization helps (Fixed has higher P(unreachable))
- Gap ≈ 0: Criterion is λ-independent

EXPECTED PATTERN:
- Spectral: Large gap (eigenbasis alignment varies with λ)
- Krylov: Small gap (Krylov subspace depends weakly on λ direction)

WHY KRYLOV IS λ-INDEPENDENT:
The Krylov subspace K_m(H(λ),|ψ⟩) = span{|ψ⟩, H|ψ⟩, ..., H^(m-1)|ψ⟩}
depends on the *direction* of H(λ), not its magnitude.
Since λ scales coefficients, it changes magnitude but not the
algebraic structure that determines subspace geometry.
""")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
