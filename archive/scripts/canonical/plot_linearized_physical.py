#!/usr/bin/env python3
"""
Physically rigorous linearized fits with NO artificial clipping.

Philosophy:
-----------
Values P=0 and P=1 are REAL physical outcomes (all reachable or all unreachable),
not numerical errors. We treat them statistically using Wilson score intervals,
which provide proper uncertainty bounds even for boundary cases.

Methodology:
------------
1. Recover trial counts from P and SEM
2. Compute Wilson confidence intervals for all points
3. Classify points: boundary (k=0 or k=N) vs transition (0<k<N)
4. Fit only transition points (informative for slope)
5. Show all points with appropriate uncertainty bars
6. Report explicit data quality metrics (N_transition, fraction)

No arbitrary ε parameter. No artificial clipping. Just physics and statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.stats import linregress, norm
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from plot_config import DIMENSION_COLORS

# Configuration
DATA_FILE = 'data/raw_logs/comprehensive_reachability_20251209_153938.pkl'
TAU_DISPLAY = 0.99
ALPHA_CI = 0.05  # 95% confidence interval

# Quality thresholds
QUALITY_THRESHOLDS = {
    'good': {'n_trans_min': 10, 'frac_trans_min': 0.20},
    'marginal': {'n_trans_min': 5, 'frac_trans_min': 0.10},
}


def wilson_interval(k, n, alpha=ALPHA_CI):
    """
    Wilson score confidence interval for binomial proportion.

    Handles k=0 and k=N gracefully - interval doesn't collapse to point.

    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        (center, lower, upper): Center and bounds of interval
    """
    if n == 0:
        return 0.5, 0.0, 1.0

    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator

    sqrt_term = np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    margin = z * sqrt_term / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return center, lower, upper


def recover_trial_count(P, sem):
    """
    Recover trial count N from probability P and standard error.

    From binomial: SEM = sqrt(P(1-P)/N)
    Therefore: N = P(1-P) / SEM²

    For boundary cases (P=0 or P=1): SEM≈0, so we can't recover N directly.
    In this case, estimate from nearby points or use a reasonable default.

    Returns:
        N: Estimated number of trials
        confidence: 'high', 'medium', 'low' based on how reliably we estimated N
    """
    if sem == 0 or P == 0 or P == 1:
        # Boundary case - can't recover N from SEM
        # For this dataset, trials = 150 (from metadata)
        return None, 'boundary'

    # Standard case
    N_estimate = P * (1 - P) / (sem**2)

    # Check if estimate is reasonable (should be integer near expected value)
    N_rounded = int(np.round(N_estimate))

    if abs(N_estimate - N_rounded) < 0.1:  # Close to integer
        confidence = 'high'
    else:
        confidence = 'medium'

    return N_rounded, confidence


def get_trial_count_from_metadata(data, d, criterion, tau=None):
    """
    Get trial count from dataset metadata or reasonable default.

    For this dataset: typically 150 trials per (d, K, tau) combination.
    """
    # Try to extract from metadata
    if 'metadata' in data and 'trials' in data['metadata']:
        return data['metadata']['trials']

    # Otherwise, estimate from non-boundary points
    if tau is None:
        # Moment criterion (no tau)
        P_arr = data['results'][criterion][d]['P']
        sem_arr = data['results'][criterion][d]['sem']
    else:
        # Spectral or Krylov (with tau)
        P_arr = data['results'][criterion][d][f'P_{tau}']
        sem_arr = data['results'][criterion][d][f'sem_{tau}']

    # Find non-boundary points
    non_boundary = (P_arr > 0.01) & (P_arr < 0.99)

    if np.sum(non_boundary) > 0:
        # Estimate from non-boundary points
        P_mid = P_arr[non_boundary]
        sem_mid = sem_arr[non_boundary]

        N_estimates = P_mid * (1 - P_mid) / (sem_mid**2 + 1e-12)
        N_median = int(np.median(N_estimates))
        return N_median

    # Fallback: common value for this dataset
    return 150


def classify_points(P, N):
    """
    Classify data points as boundary or transition.

    Args:
        P: Probability values
        N: Number of trials

    Returns:
        boundary_mask: Boolean array (True for k=0 or k=N)
        transition_mask: Boolean array (True for 0<k<N)
        k: Recovered success counts
    """
    k = np.round(P * N).astype(int)

    # Ensure k is within valid range
    k = np.clip(k, 0, N)

    boundary_mask = (k == 0) | (k == N)
    transition_mask = ~boundary_mask

    return boundary_mask, transition_mask, k


def fit_exponential_physical(K, P, N):
    """
    Fit exponential model using transition region only.

    For moment criterion: log(P) = -α·K + α·K_c

    Returns:
        fit_result: dict with parameters, quality metrics
        classification: dict with point categories
    """
    boundary_mask, transition_mask, k = classify_points(P, N)

    n_trans = np.sum(transition_mask)
    n_boundary = np.sum(boundary_mask)

    if n_trans < 3:
        return None, {
            'n_total': len(P),
            'n_trans': n_trans,
            'n_boundary': n_boundary,
            'frac_trans': n_trans / len(P),
            'quality': 'insufficient',
        }

    # Fit only transition region
    K_trans = K[transition_mask]
    P_trans = P[transition_mask]

    # Transform (no clipping needed - all 0 < P < 1 by construction)
    log_P = np.log(P_trans)

    # Linear regression
    result = linregress(K_trans, log_P)

    slope = result.slope
    intercept = result.intercept
    R2 = result.rvalue**2

    # Extract physical parameters
    alpha = -slope
    K_c = -intercept / slope

    # Quality assessment
    frac_trans = n_trans / len(P)
    if (n_trans >= QUALITY_THRESHOLDS['good']['n_trans_min'] and
            frac_trans >= QUALITY_THRESHOLDS['good']['frac_trans_min']):
        quality = 'good'
    elif (n_trans >= QUALITY_THRESHOLDS['marginal']['n_trans_min'] and
          frac_trans >= QUALITY_THRESHOLDS['marginal']['frac_trans_min']):
        quality = 'marginal'
    else:
        quality = 'insufficient'

    return {
        'K_c': K_c,
        'alpha': alpha,
        'slope': slope,
        'intercept': intercept,
        'R2': R2,
        'K_range': (K_trans.min(), K_trans.max()),
        'stderr_slope': result.stderr,
        'stderr_intercept': result.intercept_stderr,
    }, {
        'n_total': len(P),
        'n_trans': n_trans,
        'n_boundary': n_boundary,
        'frac_trans': frac_trans,
        'quality': quality,
        'boundary_mask': boundary_mask,
        'transition_mask': transition_mask,
        'k': k,
    }


def fit_fermi_dirac_physical(K, P, N):
    """
    Fit Fermi-Dirac model using transition region only.

    logit(P) = -K/Δ + K_c/Δ

    Returns:
        fit_result: dict with parameters, quality metrics
        classification: dict with point categories
    """
    boundary_mask, transition_mask, k = classify_points(P, N)

    n_trans = np.sum(transition_mask)
    n_boundary = np.sum(boundary_mask)

    if n_trans < 3:
        return None, {
            'n_total': len(P),
            'n_trans': n_trans,
            'n_boundary': n_boundary,
            'frac_trans': n_trans / len(P),
            'quality': 'insufficient',
        }

    # Fit only transition region
    K_trans = K[transition_mask]
    P_trans = P[transition_mask]

    # Transform (no clipping needed)
    logit_P = np.log(P_trans / (1 - P_trans))

    # Linear regression
    result = linregress(K_trans, logit_P)

    slope = result.slope
    intercept = result.intercept
    R2 = result.rvalue**2

    # Extract physical parameters
    Delta = -1.0 / slope
    K_c = -intercept / slope

    # Quality assessment
    frac_trans = n_trans / len(P)
    if (n_trans >= QUALITY_THRESHOLDS['good']['n_trans_min'] and
            frac_trans >= QUALITY_THRESHOLDS['good']['frac_trans_min']):
        quality = 'good'
    elif (n_trans >= QUALITY_THRESHOLDS['marginal']['n_trans_min'] and
          frac_trans >= QUALITY_THRESHOLDS['marginal']['frac_trans_min']):
        quality = 'marginal'
    else:
        quality = 'insufficient'

    return {
        'K_c': K_c,
        'Delta': Delta,
        'slope': slope,
        'intercept': intercept,
        'R2': R2,
        'K_range': (K_trans.min(), K_trans.max()),
        'stderr_slope': result.stderr,
        'stderr_intercept': result.intercept_stderr,
    }, {
        'n_total': len(P),
        'n_trans': n_trans,
        'n_boundary': n_boundary,
        'frac_trans': frac_trans,
        'quality': quality,
        'boundary_mask': boundary_mask,
        'transition_mask': transition_mask,
        'k': k,
    }


def create_physical_linearized_plots():
    """Generate physically rigorous linearized fits plot."""

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    dims = data['metadata']['dims']

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Storage for quality assessment
    quality_report = {
        'moment': {},
        'spectral': {},
        'krylov': {},
    }

    # =========================================================================
    # PANEL (a): MOMENT - log(P) vs K (Exponential)
    # =========================================================================

    ax = axes[0]

    print("\n" + "="*80)
    print("PANEL (a): MOMENT - Exponential Linearization (Physical)")
    print("="*80)
    print(f"{'d':<6} {'N_tot':<8} {'N_trans':<10} {'Frac%':<8} {'K_c':<10} {'α':<10} {'R²':<10} {'Quality'}")
    print("-"*80)

    for d in dims:
        K = data['results']['moment'][d]['K']
        P = data['results']['moment'][d]['P']
        sem = data['results']['moment'][d]['sem']
        color = DIMENSION_COLORS[d]

        # Get trial count
        N = get_trial_count_from_metadata(data, d, 'moment')

        # Fit
        fit_result, classification = fit_exponential_physical(K, P, N)

        if fit_result is None:
            print(f"{d:<6} {classification['n_total']:<8} {classification['n_trans']:<10} "
                  f"{classification['frac_trans']*100:<7.1f}% - INSUFFICIENT DATA")
            continue

        # Compute Wilson intervals for ALL points
        k_arr = classification['k']
        P_wilson = np.zeros_like(P)
        P_err_lower = np.zeros_like(P)
        P_err_upper = np.zeros_like(P)

        for i in range(len(P)):
            center, lower, upper = wilson_interval(k_arr[i], N)
            P_wilson[i] = center
            P_err_lower[i] = center - lower
            P_err_upper[i] = upper - center

        # Transform to log scale
        log_P_wilson = np.log(P_wilson)
        # Asymmetric error bars in log space (approximate)
        log_P_err_lower = log_P_wilson - np.log(P_wilson - P_err_lower)
        log_P_err_upper = np.log(P_wilson + P_err_upper) - log_P_wilson

        # Plot boundary points (faded)
        boundary_mask = classification['boundary_mask']
        if np.sum(boundary_mask) > 0:
            ax.errorbar(K[boundary_mask], log_P_wilson[boundary_mask],
                       yerr=[log_P_err_lower[boundary_mask], log_P_err_upper[boundary_mask]],
                       fmt='o', color=color, markersize=4, alpha=0.25, capsize=2)

        # Plot transition points (prominent)
        transition_mask = classification['transition_mask']
        ax.errorbar(K[transition_mask], log_P_wilson[transition_mask],
                   yerr=[log_P_err_lower[transition_mask], log_P_err_upper[transition_mask]],
                   fmt='o', color=color, markersize=6, alpha=0.9, capsize=3, label=f'd={d}')

        # Plot fit line (through transition region)
        K_fit = np.linspace(fit_result['K_range'][0] - 2, fit_result['K_range'][1] + 2, 200)
        log_P_fit = fit_result['slope'] * K_fit + fit_result['intercept']
        ax.plot(K_fit, log_P_fit, '-', color=color, linewidth=2.5, alpha=0.8)

        # Store quality assessment
        quality_report['moment'][d] = {
            **fit_result,
            **classification,
        }

        # Print results
        print(f"{d:<6} {classification['n_total']:<8} {classification['n_trans']:<10} "
              f"{classification['frac_trans']*100:<7.1f}% {fit_result['K_c']:<9.3f} "
              f"{fit_result['alpha']:<9.4f} {fit_result['R2']:<9.4f} {classification['quality']}")

    ax.set_xlabel('K', fontsize=14, fontweight='bold')
    ax.set_ylabel('log(P)', fontsize=14, fontweight='bold')
    ax.set_title('(a) Moment: Exponential Linearization (Physical)\nlog(P) = -α·K + α·K_c',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(np.log(0.5), color='gray', ls=':', lw=1.5, alpha=0.5)

    # =========================================================================
    # PANEL (b): SPECTRAL - logit(P) vs K (Fermi-Dirac)
    # =========================================================================

    ax = axes[1]

    print("\n" + "="*80)
    print(f"PANEL (b): SPECTRAL (τ={TAU_DISPLAY}) - Fermi-Dirac Linearization (Physical)")
    print("="*80)
    print(f"{'d':<6} {'N_tot':<8} {'N_trans':<10} {'Frac%':<8} {'K_c':<10} {'Δ':<10} {'R²':<10} {'Quality'}")
    print("-"*80)

    for d in dims:
        K = data['results']['spectral'][d]['K']
        P = data['results']['spectral'][d][f'P_{TAU_DISPLAY}']
        sem = data['results']['spectral'][d][f'sem_{TAU_DISPLAY}']
        color = DIMENSION_COLORS[d]

        # Get trial count
        N = get_trial_count_from_metadata(data, d, 'spectral', TAU_DISPLAY)

        # Fit
        fit_result, classification = fit_fermi_dirac_physical(K, P, N)

        if fit_result is None:
            print(f"{d:<6} {classification['n_total']:<8} {classification['n_trans']:<10} "
                  f"{classification['frac_trans']*100:<7.1f}% - INSUFFICIENT DATA")
            continue

        # Compute Wilson intervals
        k_arr = classification['k']
        P_wilson = np.zeros_like(P)
        P_err_lower = np.zeros_like(P)
        P_err_upper = np.zeros_like(P)

        for i in range(len(P)):
            center, lower, upper = wilson_interval(k_arr[i], N)
            P_wilson[i] = center
            P_err_lower[i] = center - lower
            P_err_upper[i] = upper - center

        # Transform to logit scale
        logit_P_wilson = np.log(P_wilson / (1 - P_wilson))
        # Asymmetric error bars
        logit_P_err_lower = logit_P_wilson - np.log((P_wilson - P_err_lower) / (1 - (P_wilson - P_err_lower)))
        logit_P_err_upper = np.log((P_wilson + P_err_upper) / (1 - (P_wilson + P_err_upper))) - logit_P_wilson

        # Plot boundary points
        boundary_mask = classification['boundary_mask']
        if np.sum(boundary_mask) > 0:
            ax.errorbar(K[boundary_mask], logit_P_wilson[boundary_mask],
                       yerr=[logit_P_err_lower[boundary_mask], logit_P_err_upper[boundary_mask]],
                       fmt='s', color=color, markersize=4, alpha=0.25, capsize=2)

        # Plot transition points
        transition_mask = classification['transition_mask']
        ax.errorbar(K[transition_mask], logit_P_wilson[transition_mask],
                   yerr=[logit_P_err_lower[transition_mask], logit_P_err_upper[transition_mask]],
                   fmt='s', color=color, markersize=6, alpha=0.9, capsize=3, label=f'd={d}')

        # Plot fit line
        K_fit = np.linspace(fit_result['K_range'][0] - 2, fit_result['K_range'][1] + 2, 200)
        logit_P_fit = fit_result['slope'] * K_fit + fit_result['intercept']
        ax.plot(K_fit, logit_P_fit, '-', color=color, linewidth=2.5, alpha=0.8)

        # Store quality assessment
        quality_report['spectral'][d] = {
            **fit_result,
            **classification,
        }

        # Print results
        print(f"{d:<6} {classification['n_total']:<8} {classification['n_trans']:<10} "
              f"{classification['frac_trans']*100:<7.1f}% {fit_result['K_c']:<9.3f} "
              f"{fit_result['Delta']:<9.4f} {fit_result['R2']:<9.4f} {classification['quality']}")

    ax.set_xlabel('K', fontsize=14, fontweight='bold')
    ax.set_ylabel('logit(P) = log(P/(1-P))', fontsize=14, fontweight='bold')
    ax.set_title(f'(b) Spectral (τ={TAU_DISPLAY}): Fermi-Dirac Linearization (Physical)\nlogit(P) = -K/Δ + K_c/Δ',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls=':', lw=1.5, alpha=0.5)

    # =========================================================================
    # PANEL (c): KRYLOV - logit(P) vs K (Step-like Fermi-Dirac)
    # =========================================================================

    ax = axes[2]

    print("\n" + "="*80)
    print(f"PANEL (c): KRYLOV (τ={TAU_DISPLAY}) - Step-like Fermi-Dirac (Physical)")
    print("="*80)
    print(f"{'d':<6} {'N_tot':<8} {'N_trans':<10} {'Frac%':<8} {'K_c':<10} {'Δ':<10} {'R²':<10} {'Quality'}")
    print("-"*80)

    for d in dims:
        K = data['results']['krylov'][d]['K']
        P = data['results']['krylov'][d][f'P_{TAU_DISPLAY}']
        sem = data['results']['krylov'][d][f'sem_{TAU_DISPLAY}']
        color = DIMENSION_COLORS[d]

        # Get trial count
        N = get_trial_count_from_metadata(data, d, 'krylov', TAU_DISPLAY)

        # Fit
        fit_result, classification = fit_fermi_dirac_physical(K, P, N)

        if fit_result is None:
            print(f"{d:<6} {classification['n_total']:<8} {classification['n_trans']:<10} "
                  f"{classification['frac_trans']*100:<7.1f}% - INSUFFICIENT DATA ⚠️")
            quality_report['krylov'][d] = classification
            continue

        # Compute Wilson intervals
        k_arr = classification['k']
        P_wilson = np.zeros_like(P)
        P_err_lower = np.zeros_like(P)
        P_err_upper = np.zeros_like(P)

        for i in range(len(P)):
            center, lower, upper = wilson_interval(k_arr[i], N)
            P_wilson[i] = center
            P_err_lower[i] = center - lower
            P_err_upper[i] = upper - center

        # Transform to logit scale
        logit_P_wilson = np.log(P_wilson / (1 - P_wilson))
        logit_P_err_lower = logit_P_wilson - np.log((P_wilson - P_err_lower) / (1 - (P_wilson - P_err_lower)))
        logit_P_err_upper = np.log((P_wilson + P_err_upper) / (1 - (P_wilson + P_err_upper))) - logit_P_wilson

        # Plot boundary points
        boundary_mask = classification['boundary_mask']
        if np.sum(boundary_mask) > 0:
            ax.errorbar(K[boundary_mask], logit_P_wilson[boundary_mask],
                       yerr=[logit_P_err_lower[boundary_mask], logit_P_err_upper[boundary_mask]],
                       fmt='^', color=color, markersize=4, alpha=0.25, capsize=2)

        # Plot transition points
        transition_mask = classification['transition_mask']
        ax.errorbar(K[transition_mask], logit_P_wilson[transition_mask],
                   yerr=[logit_P_err_lower[transition_mask], logit_P_err_upper[transition_mask]],
                   fmt='^', color=color, markersize=6, alpha=0.9, capsize=3, label=f'd={d}')

        # Plot fit line
        K_fit = np.linspace(fit_result['K_range'][0] - 2, fit_result['K_range'][1] + 2, 200)
        logit_P_fit = fit_result['slope'] * K_fit + fit_result['intercept']
        ax.plot(K_fit, logit_P_fit, '-', color=color, linewidth=2.5, alpha=0.8)

        # Store quality assessment
        quality_report['krylov'][d] = {
            **fit_result,
            **classification,
        }

        # Print results
        note = "⚠️" if classification['quality'] == 'insufficient' else ""
        print(f"{d:<6} {classification['n_total']:<8} {classification['n_trans']:<10} "
              f"{classification['frac_trans']*100:<7.1f}% {fit_result['K_c']:<9.3f} "
              f"{fit_result['Delta']:<9.4f} {fit_result['R2']:<9.4f} {classification['quality']} {note}")

    ax.set_xlabel('K', fontsize=14, fontweight='bold')
    ax.set_ylabel('logit(P) = log(P/(1-P))', fontsize=14, fontweight='bold')
    ax.set_title(f'(c) Krylov (τ={TAU_DISPLAY}): Step-like Fermi-Dirac (Physical)\nlogit(P) = -K/Δ + K_c/Δ',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', ls=':', lw=1.5, alpha=0.5)

    # =========================================================================
    # FINALIZE
    # =========================================================================

    plt.suptitle('Physical Linearized Fits: Wilson Intervals, No Artificial Clipping\n'
                 'Faded points: k=0 or k=N (boundary). Prominent points: 0<k<N (transition)',
                 fontsize=15, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = Path('fig/publication')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_base = output_dir / f'linearized_fits_physical_tau{TAU_DISPLAY:.2f}'.replace('.', '')
    plt.savefig(str(output_base) + '.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(str(output_base) + '.png', dpi=200, bbox_inches='tight')

    print("\n" + "="*80)
    print("OUTPUT")
    print("="*80)
    print(f"\n✅ Physical linearized fits generated:")
    print(f"   {output_base}.pdf")
    print(f"   {output_base}.png")

    plt.close()

    # Save quality report
    quality_file = Path('data/analysis/data_quality_assessment.json')
    quality_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    quality_report_serializable = {}
    for criterion, dims_data in quality_report.items():
        quality_report_serializable[criterion] = {}
        for d, metrics in dims_data.items():
            quality_report_serializable[criterion][d] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    quality_report_serializable[criterion][d][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    quality_report_serializable[criterion][d][key] = float(value)
                else:
                    quality_report_serializable[criterion][d][key] = value

    with open(quality_file, 'w') as f:
        json.dump(quality_report_serializable, f, indent=2)

    print(f"\n✅ Quality assessment saved:")
    print(f"   {quality_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("QUALITY THRESHOLDS")
    print("="*80)
    print(f"Good:         N_trans ≥ {QUALITY_THRESHOLDS['good']['n_trans_min']} AND "
          f"Frac_trans ≥ {QUALITY_THRESHOLDS['good']['frac_trans_min']*100:.0f}%")
    print(f"Marginal:     N_trans ≥ {QUALITY_THRESHOLDS['marginal']['n_trans_min']} AND "
          f"Frac_trans ≥ {QUALITY_THRESHOLDS['marginal']['frac_trans_min']*100:.0f}%")
    print(f"Insufficient: Below marginal thresholds")

    return quality_report


if __name__ == '__main__':
    quality_report = create_physical_linearized_plots()
