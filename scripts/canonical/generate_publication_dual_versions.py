#!/usr/bin/env python3
"""
Publication Figures v7 FINAL - Simple Exponential for Moment

Functional forms:
- MOMENT:   P = exp(-ρ/λ)             [Simple exponential, P(0)=1]
- SPECTRAL: P = 1/(1+exp((ρ-ρ_c)/Δ))  [Fermi-Dirac]
- KRYLOV:   P = 1/(1+exp((ρ-ρ_c)/Δ))  [Fermi-Dirac]

K_c definitions:
- MOMENT:   K_c = d² × λ × ln(2)      [From P(K_c/d²) = 0.5]
- SPECTRAL: K_c = d² × ρ_c            [From P(ρ_c) = 0.5]
- KRYLOV:   K_c = d² × ρ_c            [From P(ρ_c) = 0.5]

Generates 4 final publication plots:
1. final_summary_3panel_v7.png - Decay curves for all dimensions
2. combined_criteria_d26_v7.png - All criteria at d=26
3. Kc_vs_d_v7.png - K_c scaling with dimension
4. Kc_vs_tau_v7.png - K_c vs threshold τ
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import logit
import warnings
from pathlib import Path

# Consistent plot styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 6
})

COLORS = {
    10: '#1f77b4',  # Blue
    12: '#ff7f0e',  # Orange
    14: '#2ca02c',  # Green
    18: '#d62728',  # Red
    22: '#9467bd',  # Purple
    26: '#8c564b',  # Brown
}

# ============================================================================
# FIT FUNCTIONS
# ============================================================================

def simple_exponential(rho, lam):
    """Simple exponential: P = exp(-ρ/λ)"""
    return np.exp(-rho / lam)

def moment_half_life(rho, rho_c):
    """
    Moment criterion with half-life parameterization.
    P(ρ) = 2^(-ρ/ρ_c)

    - At ρ = 0: P = 1 (fully reachable)
    - At ρ = ρ_c: P = 0.5 (half reachable)
    - K_c = d² × ρ_c (direct, matches Spectral/Krylov)
    """
    return np.power(2, -rho / rho_c)

def moment_physical(rho, alpha, rho_c, d):
    """
    Moment criterion with physically motivated exponential.
    P(ρ) = exp(-α d² (ρ - ρ_c))

    Parameters:
    - alpha: Universal decay rate (dimension-independent)
    - rho_c: Critical density (onset threshold where P starts to decay from 1)
    - d: Hilbert space dimension (fixed per fit)
    - K_c = d² ρ_c + ln(2)/α (at P = 0.5)
    """
    exponent = -alpha * d**2 * (rho - rho_c)
    return np.clip(np.exp(exponent), 0, 1)

def fermi_dirac(rho, rho_c, delta):
    """Fermi-Dirac: P = 1/(1 + exp((ρ-ρ_c)/Δ))"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))

# ============================================================================
# DATA LOADING
# ============================================================================

def load_moment_data():
    """Load and merge Moment data from comprehensive + extension files."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw_logs'

    # Load comprehensive data
    comp_file = data_dir / 'comprehensive_reachability_20251209_153938.pkl'
    with open(comp_file, 'rb') as f:
        comp_data = pickle.load(f)

    # Load extension data
    ext_file = data_dir / 'moment_extension_all_dims_20251215_160333.pkl'
    with open(ext_file, 'rb') as f:
        ext_data = pickle.load(f)

    # Merge data
    moment_data = {}
    for d in [10, 14, 18, 22, 26]:
        if d in comp_data['results']['moment']:
            K_comp = np.array(comp_data['results']['moment'][d]['K'])
            P_comp = np.array(comp_data['results']['moment'][d]['P'])
            sem_comp = np.array(comp_data['results']['moment'][d].get('sem', np.zeros_like(P_comp)))

            if d in ext_data['results']:
                K_ext = np.array(ext_data['results'][d]['K'])
                P_ext = np.array(ext_data['results'][d]['P'])
                sem_ext = np.array(ext_data['results'][d].get('sem', np.zeros_like(P_ext)))

                K_all = np.concatenate([K_comp, K_ext])
                P_all = np.concatenate([P_comp, P_ext])
                sem_all = np.concatenate([sem_comp, sem_ext])
            else:
                K_all = K_comp
                P_all = P_comp
                sem_all = sem_comp

            rho_all = K_all / d**2

            # Sort and remove duplicates
            sort_idx = np.argsort(rho_all)
            rho_sorted = rho_all[sort_idx]
            P_sorted = P_all[sort_idx]
            sem_sorted = sem_all[sort_idx]

            unique_rho, unique_idx = np.unique(rho_sorted, return_index=True)

            moment_data[d] = {
                'K': unique_rho * d**2,
                'rho': unique_rho,
                'P': P_sorted[unique_idx],
                'sem': sem_sorted[unique_idx]
            }

    return moment_data

def load_spectral_data(tau=0.99):
    """Load Spectral data for given tau (fixed - tau not used, always 0.99)."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw_logs'
    spectral_file = data_dir / 'spectral_complete_merged_20251216_153002.pkl'

    with open(spectral_file, 'rb') as f:
        data = pickle.load(f)

    spectral_data = {}
    # Access: data['spectral'][d]
    if 'spectral' in data:
        for d in [10, 14, 18, 22, 26]:
            if d in data['spectral']:
                K = np.array(data['spectral'][d]['K'])
                P = np.array(data['spectral'][d]['P'])
                sem = np.array(data['spectral'][d].get('sem', np.zeros_like(P)))
                rho = K / d**2
                spectral_data[d] = {'K': K, 'rho': rho, 'P': P, 'sem': sem}

    return spectral_data

def load_krylov_data(tau=0.99):
    """Load Krylov data for given tau (fixed - tau not used, always 0.99)."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw_logs'

    # Load fixed krylov file
    krylov_fixed_file = data_dir / 'krylov_spectral_canonical_20251215_154634.pkl'
    with open(krylov_fixed_file, 'rb') as f:
        fixed_data = pickle.load(f)

    # Load dense krylov file
    krylov_dense_file = data_dir / 'krylov_dense_20251216_112335.pkl'
    with open(krylov_dense_file, 'rb') as f:
        dense_data = pickle.load(f)

    krylov_data = {}
    for d in [10, 14, 18, 22, 26]:
        K_list = []
        P_list = []
        sem_list = []

        # From fixed: results[d]['K'], results[d]['krylov']['P'], results[d]['krylov']['sem']
        if 'results' in fixed_data and d in fixed_data['results']:
            K_fixed = np.array(fixed_data['results'][d]['K'])
            P_fixed = np.array(fixed_data['results'][d]['krylov']['P'])
            sem_fixed = np.array(fixed_data['results'][d]['krylov'].get('sem', np.zeros_like(P_fixed)))
            K_list.extend(K_fixed.tolist())
            P_list.extend(P_fixed.tolist())
            sem_list.extend(sem_fixed.tolist())

        # From dense: krylov[d]['K'], krylov[d]['P'], krylov[d]['sem'] (avoid duplicates)
        if 'krylov' in dense_data and d in dense_data['krylov']:
            K_dense = np.array(dense_data['krylov'][d]['K'])
            P_dense = np.array(dense_data['krylov'][d]['P'])
            sem_dense = np.array(dense_data['krylov'][d].get('sem', np.zeros_like(P_dense)))
            for i, k in enumerate(K_dense):
                if k not in K_list:
                    K_list.append(k)
                    P_list.append(P_dense[i])
                    sem_list.append(sem_dense[i])

        if K_list:
            K = np.array(K_list)
            P = np.array(P_list)
            sem = np.array(sem_list)
            idx = np.argsort(K)
            rho = K[idx] / d**2
            krylov_data[d] = {'K': K[idx], 'rho': rho, 'P': P[idx], 'sem': sem[idx]}

    return krylov_data

# ============================================================================
# FITTING
# ============================================================================

def fit_moment_half_life(moment_data):
    """
    Fit half-life exponential to Moment data.
    P(ρ) = 2^(-ρ/ρ_c)

    - At ρ = 0: P = 1
    - At ρ = ρ_c: P = 0.5
    - K_c = d² × ρ_c (DIRECT, matches Spectral/Krylov!)
    """
    results = {}

    for d in sorted(moment_data.keys()):
        rho = moment_data[d]['rho']
        P = moment_data[d]['P']

        # Fit range: P in [0.01, 0.99]
        valid = (P > 0.01) & (P < 0.99)

        if np.sum(valid) < 5:
            continue

        try:
            # Fit: P = 2^(-ρ/ρ_c)
            popt, _ = curve_fit(moment_half_life, rho[valid], P[valid],
                               p0=[0.02], bounds=([1e-4], [0.5]))
            rho_c = popt[0]

            # K_c = d² × ρ_c (DIRECT!)
            K_c = d**2 * rho_c

            # Compute R²
            P_fit = moment_half_life(rho, rho_c)
            ss_res = np.sum((P - P_fit)**2)
            ss_tot = np.sum((P - np.mean(P))**2)
            r2 = 1 - ss_res / ss_tot

            results[d] = {
                'rho_c': rho_c,
                'K_c': K_c,
                'R2': r2,
                'd': d
            }
        except Exception as e:
            print(f"Failed to fit Moment half-life for d={d}: {str(e)}")
            continue

    return results

def fit_moment_physical(moment_data):
    """
    Fit physical exponential to Moment data.
    P(ρ) = exp(-α d² (ρ - ρ_c))

    - Two parameters: α (universal decay rate), ρ_c (onset threshold)
    - K_c = d² ρ_c + ln(2)/α (at P = 0.5)
    """
    results = {}

    for d in sorted(moment_data.keys()):
        rho = moment_data[d]['rho']
        P = moment_data[d]['P']

        # Fit range: P in [0.01, 0.99]
        valid = (P > 0.01) & (P < 0.99)

        if np.sum(valid) < 5:
            continue

        try:
            # Wrapper for curve_fit (fixes d)
            def moment_physical_d(rho, alpha, rho_c):
                return moment_physical(rho, alpha, rho_c, d)

            # Fit: P = exp(-α d² (ρ - ρ_c))
            # Initial guess: α=0.15, ρ_c=0.01
            popt, _ = curve_fit(moment_physical_d, rho[valid], P[valid],
                               p0=[0.15, 0.01],
                               bounds=([0.01, 0.0], [1.0, 0.1]))
            alpha, rho_c = popt[0], popt[1]

            # K_c = d² ρ_c + ln(2)/α
            K_c = d**2 * rho_c + np.log(2) / alpha

            # Compute R²
            P_fit = moment_physical(rho, alpha, rho_c, d)
            ss_res = np.sum((P - P_fit)**2)
            ss_tot = np.sum((P - np.mean(P))**2)
            r2 = 1 - ss_res / ss_tot

            results[d] = {
                'alpha': alpha,
                'rho_c': rho_c,
                'K_c': K_c,
                'R2': r2,
                'd': d
            }
        except Exception as e:
            print(f"Failed to fit Moment physical for d={d}: {str(e)}")
            continue

    return results

def fit_fermi_dirac_criterion(data, criterion_name):
    """Fit Fermi-Dirac to Spectral or Krylov data."""
    results = {}

    for d in sorted(data.keys()):
        rho = data[d]['rho']
        P = data[d]['P']

        # Fit range: P in [0.01, 0.99]
        valid = (P > 0.01) & (P < 0.99)

        # Require at least 2 points for Krylov (very sharp transitions), 3 for others
        min_points = 2 if criterion_name.lower() == 'krylov' else 3
        if np.sum(valid) < min_points:
            print(f"Skipping {criterion_name} d={d}: only {np.sum(valid)} points in (0.01, 0.99)")
            continue

        # Initial guess
        rho_mid = np.median(rho[valid])

        try:
            popt, _ = curve_fit(fermi_dirac, rho[valid], P[valid],
                               p0=[rho_mid, 0.01],
                               bounds=([0, 1e-6], [1, 0.5]),
                               maxfev=5000)  # Increase max iterations for difficult fits
            rho_c, delta = popt

            # Compute K_c = d² × ρ_c
            K_c = d**2 * rho_c

            # Compute R²
            P_fit = fermi_dirac(rho, rho_c, delta)
            ss_res = np.sum((P - P_fit)**2)
            ss_tot = np.sum((P - np.mean(P))**2)
            r2 = 1 - ss_res / ss_tot

            results[d] = {
                'rho_c': rho_c,
                'delta': delta,
                'K_c': K_c,
                'R2': r2
            }
        except Exception as e:
            print(f"Failed to fit {criterion_name} for d={d}: {str(e)}")
            continue

    return results

# ============================================================================
# PLOT 1: 3-PANEL DECAY CURVES
# ============================================================================

def plot_3panel_decay(moment_data, spectral_data, krylov_data,
                     moment_fits, spectral_fits, krylov_fits, output_dir, version='pow2'):
    """Generate 3-panel decay curves showing all dimensions (v6 styling)."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Moment (Half-Life Exponential)
    ax = axes[0]
    for d in sorted(moment_data.keys()):
        rho = moment_data[d]['rho']
        P = moment_data[d]['P']
        sem = moment_data[d]['sem']
        ax.errorbar(rho, P, yerr=sem, fmt='o', color=COLORS[d], label=f'd={d}',
                   alpha=0.6, markersize=4, capsize=2)

        if d in moment_fits:
            rho_fit = np.linspace(0, rho.max(), 200)
            if version == 'exp':
                alpha = moment_fits[d]['alpha']
                rho_c = moment_fits[d]['rho_c']
                P_fit = moment_physical(rho_fit, alpha, rho_c, d)
            else:
                rho_c = moment_fits[d]['rho_c']
                P_fit = moment_half_life(rho_fit, rho_c)
            ax.plot(rho_fit, P_fit, '-', color=COLORS[d], alpha=0.8)

    ax.set_xlabel('ρ = K/d²', fontsize=12)
    ax.set_ylabel('P(unreachable)', fontsize=12)
    ax.set_title('Moment Criterion', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, title='Dimension')
    ax.set_ylim(-0.05, 1.05)

    # Add TOP RIGHT equation box for Moment (below dimension legend)
    if moment_fits:
        # Compute average rho_c
        rho_c_mean = np.mean([moment_fits[d]['rho_c'] for d in moment_fits.keys()])
        # Linear fit of K_c vs d
        d_arr = np.array(list(moment_fits.keys()))
        Kc_arr = np.array([moment_fits[d]['K_c'] for d in d_arr])
        coeffs = np.polyfit(d_arr, Kc_arr, 1)
        a, b = coeffs[0], coeffs[1]
        # Average R²
        r2_mean = np.mean([moment_fits[d]['R2'] for d in moment_fits.keys()])

        if version == 'exp':
            alpha_mean = np.mean([moment_fits[d]['alpha'] for d in moment_fits.keys()])
            info = (r"$P = e^{-\alpha d^2(\rho - \rho_c)}$" + "\n"
                    f"$\\alpha \\approx {alpha_mean:.2f}, \\rho_c \\approx {rho_c_mean:.4f}$\n"
                    f"$K_c = d^2\\rho_c + \\ln(2)/\\alpha$\n"
                    f"$K_c \\approx {a:.2f}d {b:+.1f}$\n"
                    f"$R^2 = {r2_mean:.3f}$")
        else:
            info = (r"$P = 2^{-\rho/\rho_c}$" + "\n"
                    f"$\\rho_c \\approx {rho_c_mean:.4f}$\n"
                    f"$K_c = d^2 \\rho_c$\n"
                    f"$K_c \\approx {a:.2f}d {b:+.1f}$\n"
                    f"$R^2 = {r2_mean:.3f}$")
        ax.text(0.97, 0.40, info, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # Panel 2: Spectral (Fermi-Dirac)
    ax = axes[1]
    for d in sorted(spectral_data.keys()):
        rho = spectral_data[d]['rho']
        P = spectral_data[d]['P']
        sem = spectral_data[d]['sem']
        ax.errorbar(rho, P, yerr=sem, fmt='o', color=COLORS[d], label=f'd={d}',
                   alpha=0.6, markersize=4, capsize=2)

        if d in spectral_fits:
            rho_c = spectral_fits[d]['rho_c']
            delta = spectral_fits[d]['delta']
            rho_fit = np.linspace(0, rho.max(), 200)
            P_fit = fermi_dirac(rho_fit, rho_c, delta)
            ax.plot(rho_fit, P_fit, '-', color=COLORS[d], alpha=0.8)

    ax.set_xlabel('ρ = K/d²', fontsize=12)
    ax.set_ylabel('P(unreachable)', fontsize=12)
    ax.set_title('Spectral Criterion (τ=0.99)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, title='Dimension')
    ax.set_ylim(-0.05, 1.05)

    # Add bottom-left equation box for Spectral
    if spectral_fits:
        d_arr = np.array(list(spectral_fits.keys()))
        Kc_arr = np.array([spectral_fits[d]['K_c'] for d in d_arr])
        coeffs = np.polyfit(d_arr, Kc_arr, 1)
        a, b = coeffs[0], coeffs[1]
        r2_mean = np.mean([spectral_fits[d]['R2'] for d in spectral_fits.keys()])

        info = (r"$P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$" + "\n"
                f"$K_c \\approx {a:.2f}d {b:+.1f}$\n"
                f"$R^2 = {r2_mean:.3f}$")
        ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # Panel 3: Krylov (Fermi-Dirac)
    ax = axes[2]
    for d in sorted(krylov_data.keys()):
        rho = krylov_data[d]['rho']
        P = krylov_data[d]['P']
        sem = krylov_data[d]['sem']
        ax.errorbar(rho, P, yerr=sem, fmt='o', color=COLORS[d], label=f'd={d}',
                   alpha=0.6, markersize=4, capsize=2)

        if d in krylov_fits:
            rho_c = krylov_fits[d]['rho_c']
            delta = krylov_fits[d]['delta']
            rho_fit = np.linspace(0, rho.max(), 200)
            P_fit = fermi_dirac(rho_fit, rho_c, delta)
            ax.plot(rho_fit, P_fit, '-', color=COLORS[d], alpha=0.8)

    ax.set_xlabel('ρ = K/d²', fontsize=12)
    ax.set_ylabel('P(unreachable)', fontsize=12)
    ax.set_title('Krylov Criterion (τ=0.99)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, title='Dimension')
    ax.set_ylim(-0.05, 1.05)

    # Add bottom-left equation box for Krylov
    if krylov_fits:
        d_arr = np.array(list(krylov_fits.keys()))
        Kc_arr = np.array([krylov_fits[d]['K_c'] for d in d_arr])
        coeffs = np.polyfit(d_arr, Kc_arr, 1)
        a, b = coeffs[0], coeffs[1]
        r2_mean = np.mean([krylov_fits[d]['R2'] for d in krylov_fits.keys()])

        info = (r"$P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$" + "\n"
                f"$K_c \\approx {a:.2f}d {b:+.1f}$\n"
                f"$R^2 = {r2_mean:.3f}$")
        ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    plt.tight_layout()
    suffix = '_v7_exp' if version == 'exp' else '_v7_pow2'
    plt.savefig(output_dir / f'final_summary_3panel{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: final_summary_3panel_v7.png")

# ============================================================================
# PLOT 2: ALL CRITERIA AT d=26
# ============================================================================

def plot_combined_criteria_d26(moment_data, spectral_data, krylov_data,
                               moment_fits, spectral_fits, krylov_fits, output_dir, version='pow2'):
    """Compare all three criteria at d=26 with v6 styling."""

    fig, ax = plt.subplots(figsize=(10, 6))
    d = 26

    # Store fit info for equation box
    fit_info = {}

    # Moment
    if d in moment_data and d in moment_fits:
        rho = moment_data[d]['rho']
        P = moment_data[d]['P']
        sem = moment_data[d]['sem']
        rho_c = moment_fits[d]['rho_c']
        K_c = moment_fits[d]['K_c']

        ax.errorbar(rho, P, yerr=sem, fmt='o', color='blue', label='Moment',
                   alpha=0.7, markersize=6, capsize=2)
        rho_fit = np.linspace(0, rho.max(), 200)
        if version == 'exp':
            alpha = moment_fits[d]['alpha']
            P_fit = moment_physical(rho_fit, alpha, rho_c, d)
        else:
            P_fit = moment_half_life(rho_fit, rho_c)
        ax.plot(rho_fit, P_fit, '-', color='blue', alpha=0.8, linewidth=2)

        fit_info['moment'] = {'rho_c': rho_c, 'K_c': K_c}

    # Spectral
    if d in spectral_data and d in spectral_fits:
        rho = spectral_data[d]['rho']
        P = spectral_data[d]['P']
        sem = spectral_data[d]['sem']
        rho_c = spectral_fits[d]['rho_c']
        delta = spectral_fits[d]['delta']
        K_c = spectral_fits[d]['K_c']

        ax.errorbar(rho, P, yerr=sem, fmt='s', color='red', label='Spectral',
                   alpha=0.7, markersize=6, capsize=2)
        rho_fit = np.linspace(0, rho.max(), 200)
        P_fit = fermi_dirac(rho_fit, rho_c, delta)
        ax.plot(rho_fit, P_fit, '-', color='red', alpha=0.8, linewidth=2)

        fit_info['spectral'] = {'rho_c': rho_c, 'delta': delta, 'K_c': K_c}

    # Krylov
    if d in krylov_data and d in krylov_fits:
        rho = krylov_data[d]['rho']
        P = krylov_data[d]['P']
        sem = krylov_data[d]['sem']
        rho_c = krylov_fits[d]['rho_c']
        delta = krylov_fits[d]['delta']
        K_c = krylov_fits[d]['K_c']

        ax.errorbar(rho, P, yerr=sem, fmt='^', color='green', label='Krylov',
                   alpha=0.7, markersize=6, capsize=2)
        rho_fit = np.linspace(0, rho.max(), 200)
        P_fit = fermi_dirac(rho_fit, rho_c, delta)
        ax.plot(rho_fit, P_fit, '-', color='green', alpha=0.8, linewidth=2)

        fit_info['krylov'] = {'rho_c': rho_c, 'delta': delta, 'K_c': K_c}

    # Add equation box (TOP RIGHT, below data legend) matching user's request
    eq_text = ""
    if 'moment' in fit_info:
        rho_c = fit_info['moment']['rho_c']
        K_c = fit_info['moment']['K_c']
        eq_text += r"$P_{\mathrm{moment}} = 2^{-\rho/\rho_c}$" + "\n"
        eq_text += f"  ρ_c={rho_c:.4f}, K_c={K_c:.1f}\n\n"

    if 'spectral' in fit_info or 'krylov' in fit_info:
        eq_text += r"$P_{\mathrm{spec/kryl}} = \frac{1}{1 + e^{(\rho - \rho_c)/\Delta}}$" + "\n"

        if 'spectral' in fit_info:
            rho_c = fit_info['spectral']['rho_c']
            delta = fit_info['spectral']['delta']
            K_c = fit_info['spectral']['K_c']
            eq_text += f"Spectral: ρ_c={rho_c:.3f}, K_c={K_c:.1f}, Δ={delta:.4f}\n"

        if 'krylov' in fit_info:
            rho_c = fit_info['krylov']['rho_c']
            delta = fit_info['krylov']['delta']
            K_c = fit_info['krylov']['K_c']
            eq_text += f"Krylov: ρ_c={rho_c:.3f}, K_c={K_c:.1f}, Δ={delta:.4f}\n"

    ax.text(0.98, 0.65, eq_text.strip(), transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           family='monospace')

    ax.set_xlabel('ρ = K/d²', fontsize=13)
    ax.set_ylabel('P(unreachable)', fontsize=13)
    ax.set_title(f'All Criteria Comparison (d={d}, τ=0.99)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    suffix = '_v7_exp' if version == 'exp' else '_v7_pow2'
    plt.savefig(output_dir / f'combined_criteria_d26{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: combined_criteria_d26_v7.png")

# ============================================================================
# PLOT 3: K_c vs d SCALING
# ============================================================================

def plot_Kc_vs_d(moment_fits, spectral_fits, krylov_fits, output_dir, version='pow2'):
    """Plot K_c vs d for all three criteria with linear fits (fixed legend placement)."""

    fig, ax = plt.subplots(figsize=(10, 7))

    # Moment
    d_vals = sorted(moment_fits.keys())
    K_c_vals = [moment_fits[d]['K_c'] for d in d_vals]
    ax.plot(d_vals, K_c_vals, 'o-', color='blue', label='Moment', markersize=8, linewidth=2)

    # Linear fit
    coeffs = np.polyfit(d_vals, K_c_vals, 1)
    K_c_fit = np.polyval(coeffs, d_vals)
    ax.plot(d_vals, K_c_fit, '--', color='blue', alpha=0.5)
    ax.text(0.05, 0.70, f'Moment: K_c = {coeffs[0]:.2f}d + {coeffs[1]:.1f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Spectral
    if len(spectral_fits) > 0:
        d_vals = sorted(spectral_fits.keys())
        K_c_vals = [spectral_fits[d]['K_c'] for d in d_vals]
        ax.plot(d_vals, K_c_vals, 's-', color='red', label='Spectral (τ=0.99)', markersize=8, linewidth=2)

        coeffs = np.polyfit(d_vals, K_c_vals, 1)
        K_c_fit = np.polyval(coeffs, d_vals)
        ax.plot(d_vals, K_c_fit, '--', color='red', alpha=0.5)
        ax.text(0.05, 0.63, f'Spectral: K_c = {coeffs[0]:.2f}d + {coeffs[1]:.1f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Krylov
    if len(krylov_fits) > 0:
        d_vals = sorted(krylov_fits.keys())
        K_c_vals = [krylov_fits[d]['K_c'] for d in d_vals]
        ax.plot(d_vals, K_c_vals, '^-', color='green', label='Krylov (τ=0.99)', markersize=8, linewidth=2)

        coeffs = np.polyfit(d_vals, K_c_vals, 1)
        K_c_fit = np.polyval(coeffs, d_vals)
        ax.plot(d_vals, K_c_fit, '--', color='green', alpha=0.5)
        ax.text(0.05, 0.56, f'Krylov: K_c = {coeffs[0]:.2f}d + {coeffs[1]:.1f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Dimension d', fontsize=13)
    ax.set_ylabel('K_c (Critical # Hamiltonians)', fontsize=13)
    ax.set_title('Critical K_c Scaling with Dimension', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    suffix = '_v7_exp' if version == 'exp' else '_v7_pow2'
    plt.savefig(output_dir / f'Kc_vs_d{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: Kc_vs_d_v7.png")

# ============================================================================
# PLOT 4: LINEARIZED FITS
# ============================================================================

def plot_linearized_fits(moment_data, spectral_data, krylov_data,
                         moment_fits, spectral_fits, krylov_fits, output_dir, version='pow2'):
    """Generate linearized fits: ln(P) for Moment, logit(P) for Spectral/Krylov."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    criteria = ['moment', 'spectral', 'krylov']
    data_sources = [moment_data, spectral_data, krylov_data]
    fit_sources = [moment_fits, spectral_fits, krylov_fits]

    for ax, criterion, data, fits in zip(axes, criteria, data_sources, fit_sources):
        for d in sorted(data.keys()):
            if d not in data:
                continue

            rho = data[d]['rho']
            P = data[d]['P']

            # Filter: 0.01 < P < 0.99 for linearization
            mask = (P > 0.01) & (P < 0.99)
            if np.sum(mask) < 2:
                continue

            rho_m = rho[mask]
            P_m = P[mask]

            # Different transformations for different criteria
            if criterion == 'moment':
                if version == 'exp':
                    y_transform = np.log(P_m)  # ln(P) for exp version
                    ylabel = r'$\ln(P)$'
                    title = 'Moment: Linearized as ln(P)'
                else:
                    y_transform = np.log2(P_m)  # log2(P) for pow2 version
                    ylabel = r'$\log_2(P)$'
                    title = 'Moment: Linearized as log₂(P)'
            else:
                y_transform = np.log(P_m / (1 - P_m))  # logit(P) for Fermi-Dirac
                ylabel = r'$\mathrm{logit}(P) = \ln\frac{P}{1-P}$'
                if criterion == 'spectral':
                    title = 'Spectral: Linearized as logit(P)'
                else:
                    title = 'Krylov: Linearized as logit(P)'

            ax.scatter(rho_m, y_transform, s=30, alpha=0.7, label=f'd={d}', color=COLORS.get(d, 'gray'))

            # Plot fit line if available
            if d in fits:
                if criterion == 'moment':
                    rho_fit = np.linspace(rho_m.min(), rho_m.max(), 100)
                    if version == 'exp':
                        alpha = fits[d]['alpha']
                        rho_c = fits[d]['rho_c']
                        P_fit = moment_physical(rho_fit, alpha, rho_c, d)
                        y_fit = np.log(P_fit)
                    else:
                        rho_c = fits[d]['rho_c']
                        P_fit = moment_half_life(rho_fit, rho_c)
                        y_fit = np.log2(P_fit)
                else:
                    rho_c = fits[d]['rho_c']
                    delta = fits[d]['delta']
                    rho_fit = np.linspace(rho_m.min(), rho_m.max(), 100)
                    P_fit = fermi_dirac(rho_fit, rho_c, delta)
                    # Avoid log(0) issues
                    P_fit = np.clip(P_fit, 0.01, 0.99)
                    y_fit = np.log(P_fit / (1 - P_fit))

                ax.plot(rho_fit, y_fit, '-', alpha=0.8, color=COLORS.get(d, 'gray'), linewidth=1.5)

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    suffix = '_v7_exp' if version == 'exp' else '_v7_pow2'
    plt.savefig(output_dir / f'linearized_fits{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: linearized_fits_v7.png")

# ============================================================================
# PLOT 5: K_c vs τ
# ============================================================================

def plot_Kc_vs_tau(output_dir):
    """Plot K_c vs threshold τ for Spectral and Krylov."""

    print("  Skipping Kc_vs_tau plot (no Spectral/Krylov data)")
    return  # Skip this plot for now since we don't have Spectral/Krylov data loaded

    # Load data for multiple tau values
    tau_values = np.arange(0.80, 1.00, 0.01)
    dimensions = [10, 14, 18]

    spectral_Kc = {d: [] for d in dimensions}
    krylov_Kc = {d: [] for d in dimensions}

    for tau in tau_values:
        tau_round = round(tau, 2)

        # Load Spectral
        spectral_data = load_spectral_data(tau_round)
        spectral_fits = fit_fermi_dirac_criterion(spectral_data, 'Spectral')

        for d in dimensions:
            if d in spectral_fits:
                spectral_Kc[d].append(spectral_fits[d]['K_c'])
            else:
                spectral_Kc[d].append(np.nan)

        # Load Krylov
        krylov_data = load_krylov_data(tau_round)
        krylov_fits = fit_fermi_dirac_criterion(krylov_data, 'Krylov')

        for d in dimensions:
            if d in krylov_fits:
                krylov_Kc[d].append(krylov_fits[d]['K_c'])
            else:
                krylov_Kc[d].append(np.nan)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Spectral
    ax = axes[0]
    for d in dimensions:
        Kc_vals = np.array(spectral_Kc[d])
        valid = ~np.isnan(Kc_vals)
        ax.plot(tau_values[valid], Kc_vals[valid], 'o-', label=f'd={d}', markersize=6)

    ax.set_xlabel('Threshold τ', fontsize=12)
    ax.set_ylabel('K_c (Critical # Hamiltonians)', fontsize=12)
    ax.set_title('Spectral: K_c vs τ', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: Krylov
    ax = axes[1]
    for d in dimensions:
        Kc_vals = np.array(krylov_Kc[d])
        valid = ~np.isnan(Kc_vals)
        ax.plot(tau_values[valid], Kc_vals[valid], 'o-', label=f'd={d}', markersize=6)

    ax.set_xlabel('Threshold τ', fontsize=12)
    ax.set_ylabel('K_c (Critical # Hamiltonians)', fontsize=12)
    ax.set_title('Krylov: K_c vs τ', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'Kc_vs_tau_v7.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: Kc_vs_tau_v7.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate BOTH versions of v7 publication figures (exp + pow2)."""

    output_dir = Path(__file__).parent.parent / 'fig' / 'publication'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("DUAL-VERSION PUBLICATION FIGURES (exp + pow2)")
    print("="*70)

    # Load data once
    print("\n[1/3] Loading data...")
    moment_data = load_moment_data()
    spectral_data = load_spectral_data(tau=0.99)
    krylov_data = load_krylov_data(tau=0.99)
    print(f"  Moment: {len(moment_data)} dimensions")
    print(f"  Spectral: {len(spectral_data)} dimensions")
    print(f"  Krylov: {len(krylov_data)} dimensions")

    # Fit Spectral/Krylov once (same for both versions)
    spectral_fits = fit_fermi_dirac_criterion(spectral_data, 'Spectral')
    krylov_fits = fit_fermi_dirac_criterion(krylov_data, 'Krylov')

    # Generate BOTH versions
    for version in ['exp', 'pow2']:
        print(f"\n[2/3] Generating {version.upper()} version...")

        if version == 'exp':
            moment_fits = fit_moment_physical(moment_data)
        else:
            moment_fits = fit_moment_half_life(moment_data)

        plot_3panel_decay(moment_data, spectral_data, krylov_data,
                         moment_fits, spectral_fits, krylov_fits, output_dir, version)
        plot_combined_criteria_d26(moment_data, spectral_data, krylov_data,
                                   moment_fits, spectral_fits, krylov_fits, output_dir, version)
        plot_Kc_vs_d(moment_fits, spectral_fits, krylov_fits, output_dir, version)
        plot_linearized_fits(moment_data, spectral_data, krylov_data,
                            moment_fits, spectral_fits, krylov_fits, output_dir, version)

    print("\n[3/3] Done! Generated 8 files.")
    print("\nGenerated files:")
    for version_suffix in ['_v7_exp', '_v7_pow2']:
        for basename in ['final_summary_3panel', 'combined_criteria_d26', 'Kc_vs_d', 'linearized_fits']:
            fname = f"{basename}{version_suffix}.png"
            fpath = output_dir / fname
            if fpath.exists():
                size_kb = fpath.stat().st_size / 1024
                print(f"  ✓ {fname} ({size_kb:.0f} KB)")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
