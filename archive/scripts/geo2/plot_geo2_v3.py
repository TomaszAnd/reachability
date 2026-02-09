#!/usr/bin/env python3
"""
GEO2LOCAL Publication Plots v3 - Canonical v7 Style with Equation Text Boxes

Key changes from v2:
1. Equation text boxes (wheat-colored) with analytical form + parameters
2. Scaling plot shows LINEAR FIT ONLY: ρ_c = 0.0455 + 0.00220×d
3. Simplified legends (just dimension labels)

Fit equations:
- MOMENT:   P = exp(-ρ/λ)
- SPECTRAL: P = 1/(1+exp((ρ-ρc)/Δ))  [Fermi-Dirac]
- KRYLOV:   P = 1/(1+exp((ρ-ρc)/Δ))  [Fermi-Dirac]

Linear scaling (FIXED): ρ_c = 0.0455 + 0.00220×d
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import warnings

OUTPUT_DIR = Path("fig/geo2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# Color scheme
DIM_COLORS = {16: '#1f77b4', 32: '#2ca02c', 64: '#d62728'}
DIM_MARKERS = {16: 'o', 32: 's', 64: '^'}
CRIT_COLORS = {'moment': '#2E86AB', 'spectral': '#A23B72', 'krylov': '#F18F01'}

# LINEAR SCALING (FIXED - use throughout)
LINEAR_A = 0.0455
LINEAR_B = 0.00220


# ============================================================================
# FIT FUNCTIONS
# ============================================================================

def simple_exponential(rho, lam):
    """P = exp(-ρ/λ)"""
    return np.exp(-rho / lam)

def fermi_dirac(rho, rho_c, delta):
    """P = 1/(1 + exp((ρ-ρc)/Δ))"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))

def linear_scaling(d):
    """ρ_c = 0.0455 + 0.00220×d (FIXED)"""
    return LINEAR_A + LINEAR_B * d


# ============================================================================
# DATA & FITTING
# ============================================================================

def load_data():
    files = sorted(Path("data/raw_logs").glob("geo2_production_complete_*.pkl"))
    if not files:
        raise FileNotFoundError("No GEO2 data found")
    with open(files[-1], 'rb') as f:
        return pickle.load(f)


def extract_curve(results, d, criterion):
    key = (d, 0.99, criterion)
    if key not in results[d]['data']:
        return None, None, None
    crit_data = results[d]['data'][key]
    rho = np.array(crit_data['rho'])
    P = np.array(crit_data['p'])
    err = np.array(crit_data.get('err', np.zeros_like(P)))
    idx = np.argsort(rho)
    return rho[idx], P[idx], err[idx]


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_fermi_dirac(rho, P):
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                           p0=[np.median(rho[mask]), 0.02],
                           bounds=([0, 0.001], [1.0, 0.5]), maxfev=10000)
        r2 = compute_r2(P[mask], fermi_dirac(rho[mask], *popt))
        return {'rho_c': popt[0], 'delta': popt[1], 'R2': r2}
    except:
        return None


def fit_exponential(rho, P):
    try:
        mask = (P > 0.01) & (P < 0.99)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, _ = curve_fit(simple_exponential, rho[mask], P[mask],
                           p0=[0.01], bounds=([0.0001], [1.0]), maxfev=10000)
        r2 = compute_r2(P[mask], simple_exponential(rho[mask], *popt))
        return {'lambda': popt[0], 'R2': r2}
    except:
        return None


# ============================================================================
# PLOT 1: MAIN 1×3 WITH EQUATION TEXT BOXES
# ============================================================================

def plot_main_v3(fixed, optimized):
    """Main 1×3 figure with equation text boxes (canonical v7 style)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    criteria = ['moment', 'spectral', 'krylov']
    titles = ['Moment Criterion', 'Spectral Criterion (τ=0.99)', 'Krylov Criterion (τ=0.99)']

    for ax, criterion, title in zip(axes, criteria, titles):
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Collect fits for text box
        fits_for_box = {}

        for d in [16, 32, 64]:
            color = DIM_COLORS[d]
            marker = DIM_MARKERS[d]

            # Fixed λ
            rho_f, P_f, err_f = extract_curve(fixed, d, criterion)
            if rho_f is not None:
                ax.errorbar(rho_f, P_f, yerr=err_f, marker=marker, color=color,
                           linestyle='-', linewidth=2, markersize=5,
                           markerfacecolor=color, capsize=2, alpha=0.9,
                           label=f'd={d} Fixed')

            # Optimized λ
            rho_o, P_o, err_o = extract_curve(optimized, d, criterion)
            if rho_o is not None:
                ax.errorbar(rho_o, P_o, yerr=err_o, marker=marker, color=color,
                           linestyle='--', linewidth=2, markersize=5,
                           markerfacecolor='white', markeredgecolor=color,
                           markeredgewidth=1.5, capsize=2, alpha=0.9,
                           label=f'd={d} Opt')

                # Fit curve
                if criterion == 'moment':
                    fit = fit_exponential(rho_o, P_o)
                    if fit and fit['R2'] > 0.3:
                        rho_fine = np.linspace(rho_o.min(), rho_o.max(), 100)
                        ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                               ':', color=color, alpha=0.5, linewidth=1.5)
                        fits_for_box[d] = fit
                else:
                    fit = fit_fermi_dirac(rho_o, P_o)
                    if fit and fit['R2'] > 0.3:
                        rho_fine = np.linspace(rho_o.min(), rho_o.max(), 100)
                        ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                               ':', color=color, alpha=0.5, linewidth=1.5)
                        fits_for_box[d] = fit

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel('P(unreachable)', fontsize=12)
        ax.set_xlim(0, None)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=7, ncol=2)

        # ADD EQUATION TEXT BOX (canonical v7 style)
        if fits_for_box:
            if criterion == 'moment':
                # Moment equation box
                lam_mean = np.mean([fits_for_box[d]['lambda'] for d in fits_for_box])
                r2_mean = np.mean([fits_for_box[d]['R2'] for d in fits_for_box])

                info = (r"$P = e^{-\rho/\lambda}$" + "\n"
                        + "─" * 20 + "\n")
                for d in sorted(fits_for_box.keys()):
                    info += f"d={d}: λ={fits_for_box[d]['lambda']:.4f}, R²={fits_for_box[d]['R2']:.2f}\n"
                info += "─" * 20 + "\n"
                info += f"Mean: λ≈{lam_mean:.4f}, R²≈{r2_mean:.2f}"

            else:
                # Spectral/Krylov equation box
                info = (r"$P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$" + "\n"
                        + "─" * 20 + "\n")
                for d in sorted(fits_for_box.keys()):
                    f = fits_for_box[d]
                    info += f"d={d}: ρc={f['rho_c']:.3f}, Δ={f['delta']:.3f}, R²={f['R2']:.2f}\n"

            ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                   family='monospace')

    fig.suptitle('GEO2LOCAL: Unreachability Phase Transitions',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'geo2_main_v3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: geo2_main_v3.png")


# ============================================================================
# PLOT 2: SCALING - LINEAR FIT ONLY
# ============================================================================

def plot_scaling_v3(optimized):
    """Scaling plot with LINEAR FIT ONLY."""
    fig, ax = plt.subplots(figsize=(9, 7))

    dims = np.array([16, 32, 64])
    rho_c_vals = []
    rho_c_errs = []
    valid_dims = []

    for d in dims:
        rho, P, _ = extract_curve(optimized, d, 'spectral')
        if rho is None:
            continue
        fit = fit_fermi_dirac(rho, P)
        if fit:
            rho_c_vals.append(fit['rho_c'])
            rho_c_errs.append(0.01)  # Default error
            valid_dims.append(d)

    valid_dims = np.array(valid_dims)
    rho_c_vals = np.array(rho_c_vals)
    rho_c_errs = np.array(rho_c_errs)

    # Data points
    ax.errorbar(valid_dims, rho_c_vals, yerr=rho_c_errs,
               marker='o', color='blue', markersize=14,
               capsize=6, linewidth=0, elinewidth=2,
               label='Data (Spectral, Optimized λ)', zorder=10)

    # LINEAR FIT ONLY
    d_fine = np.linspace(10, 80, 100)
    rho_c_pred = linear_scaling(valid_dims)
    r2 = compute_r2(rho_c_vals, rho_c_pred)

    ax.plot(d_fine, linear_scaling(d_fine), '--', color='green', linewidth=2.5,
           label=rf'$\rho_c = {LINEAR_A:.4f} + {LINEAR_B:.5f} \times d$ (R²={r2:.3f})')

    # Labels
    for d, rho_c in zip(valid_dims, rho_c_vals):
        ax.annotate(f'ρ_c={rho_c:.3f}', (d, rho_c),
                   textcoords="offset points", xytext=(12, 5),
                   fontsize=11, fontweight='bold')

    # EQUATION TEXT BOX
    info = (r"$\rho_c = a + b \times d$" + "\n"
            + "─" * 25 + "\n"
            f"$a = {LINEAR_A:.4f}$\n"
            f"$b = {LINEAR_B:.5f}$\n"
            + "─" * 25 + "\n"
            f"$R^2 = {r2:.3f}$")
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_xlabel('Dimension d', fontsize=14)
    ax.set_ylabel('Critical Density ρ_c', fontsize=14)
    ax.set_title('GEO2LOCAL: Critical Density Scaling\n(Spectral, Optimized λ)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'geo2_scaling_v3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: geo2_scaling_v3.png")


# ============================================================================
# PLOT 3-5: DIMENSION SUMMARY WITH EQUATION BOXES
# ============================================================================

def plot_dimension_summary_v3(fixed, optimized, d, filename):
    """Summary plot for single dimension with equation text box."""
    fig, ax = plt.subplots(figsize=(10, 7))

    criteria = ['moment', 'spectral', 'krylov']
    fits_for_box = {}

    for criterion in criteria:
        color = CRIT_COLORS[criterion]

        # Fixed
        rho_f, P_f, err_f = extract_curve(fixed, d, criterion)
        if rho_f is not None:
            ax.errorbar(rho_f, P_f, yerr=err_f, marker='o', color=color,
                       linestyle='-', linewidth=2, markersize=6,
                       markerfacecolor=color, capsize=2, alpha=0.85,
                       label=f'{criterion.title()} Fixed')

        # Optimized
        rho_o, P_o, err_o = extract_curve(optimized, d, criterion)
        if rho_o is not None:
            ax.errorbar(rho_o, P_o, yerr=err_o, marker='s', color=color,
                       linestyle='--', linewidth=2, markersize=6,
                       markerfacecolor='white', markeredgecolor=color,
                       markeredgewidth=1.5, capsize=2, alpha=0.85,
                       label=f'{criterion.title()} Opt')

            # Fit
            if criterion == 'moment':
                fit = fit_exponential(rho_o, P_o)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho_o.min(), rho_o.max(), 100)
                    ax.plot(rho_fine, simple_exponential(rho_fine, fit['lambda']),
                           ':', color=color, alpha=0.4, linewidth=1.5)
                    fits_for_box[criterion] = fit
            else:
                fit = fit_fermi_dirac(rho_o, P_o)
                if fit and fit['R2'] > 0.3:
                    rho_fine = np.linspace(rho_o.min(), rho_o.max(), 100)
                    ax.plot(rho_fine, fermi_dirac(rho_fine, fit['rho_c'], fit['delta']),
                           ':', color=color, alpha=0.4, linewidth=1.5)
                    fits_for_box[criterion] = fit

    nx = fixed[d].get('nx', '?')
    ny = fixed[d].get('ny', '?')

    ax.set_xlabel('ρ = K/d²', fontsize=13)
    ax.set_ylabel('P(unreachable)', fontsize=13)
    ax.set_title(f'GEO2LOCAL d={d} ({nx}×{ny} lattice): All Criteria',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, None)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    # EQUATION TEXT BOX
    info = (r"Moment: $P = e^{-\rho/\lambda}$" + "\n"
            r"Spectral/Krylov: $P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$" + "\n"
            + "─" * 35 + "\n")

    for crit in ['moment', 'spectral', 'krylov']:
        if crit in fits_for_box:
            f = fits_for_box[crit]
            if crit == 'moment':
                info += f"{crit.title()}: λ={f['lambda']:.4f}, R²={f['R2']:.2f}\n"
            else:
                info += f"{crit.title()}: ρc={f['rho_c']:.3f}, Δ={f['delta']:.3f}, R²={f['R2']:.2f}\n"

    ax.text(0.03, 0.03, info, transform=ax.transAxes, fontsize=8,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
           family='monospace')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {filename}")


# ============================================================================
# PLOT 6: LINEARIZED FITS
# ============================================================================

def plot_linearized_v3(fixed, optimized):
    """Linearized plots with equation annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    criteria = ['moment', 'spectral', 'krylov']

    for ax, criterion in zip(axes, criteria):
        if criterion == 'moment':
            ylabel = r'$\ln(P)$'
            title = (r'Moment: $\ln(P)$ vs $\rho$' + '\n' +
                    r'[$P = e^{-\rho/\lambda}$ → $\ln(P) = -\rho/\lambda$]')
        else:
            ylabel = r'$\mathrm{logit}(P) = \ln\frac{P}{1-P}$'
            eq_note = r'$P = \frac{1}{1+e^{(\rho-\rho_c)/\Delta}}$'
            if criterion == 'spectral':
                title = f'Spectral: logit(P) vs ρ\n[{eq_note} → logit = -(ρ-ρc)/Δ]'
            else:
                title = f'Krylov: logit(P) vs ρ\n[{eq_note} → logit = -(ρ-ρc)/Δ]'

        for d in [16, 32, 64]:
            color = DIM_COLORS[d]
            rho, P, _ = extract_curve(optimized, d, criterion)
            if rho is None:
                continue

            mask = (P > 0.01) & (P < 0.99)
            if np.sum(mask) < 2:
                continue

            rho_m, P_m = rho[mask], P[mask]

            if criterion == 'moment':
                y = np.log(P_m)
            else:
                y = np.log(P_m / (1 - P_m))

            ax.scatter(rho_m, y, s=50, alpha=0.7, label=f'd={d}',
                      color=color, marker=DIM_MARKERS[d])

            # Fit line
            if criterion == 'moment':
                fit = fit_exponential(rho, P)
                if fit:
                    rho_fit = np.linspace(rho_m.min(), rho_m.max(), 100)
                    ax.plot(rho_fit, -rho_fit / fit['lambda'], '-',
                           alpha=0.6, color=color, linewidth=1.5)
            else:
                fit = fit_fermi_dirac(rho, P)
                if fit:
                    rho_fit = np.linspace(rho_m.min(), rho_m.max(), 100)
                    ax.plot(rho_fit, -(rho_fit - fit['rho_c']) / fit['delta'], '-',
                           alpha=0.6, color=color, linewidth=1.5)

        ax.set_xlabel('ρ = K/d²', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    fig.suptitle('GEO2LOCAL: Linearized Fit Verification (Optimized λ)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'geo2_linearized_v3.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: geo2_linearized_v3.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("GEO2LOCAL PUBLICATION PLOTS v3")
    print("Canonical v7 Style with Equation Text Boxes")
    print("="*70)

    data = load_data()
    fixed = data['results']['fixed']
    optimized = data['results']['optimized']

    print(f"\nLinear scaling: ρ_c = {LINEAR_A} + {LINEAR_B}×d")

    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    print("\n[1/6] Main 1×3 with equation text boxes...")
    plot_main_v3(fixed, optimized)

    print("\n[2/6] Scaling with LINEAR FIT ONLY...")
    plot_scaling_v3(optimized)

    print("\n[3/6] d=16 summary...")
    plot_dimension_summary_v3(fixed, optimized, 16, 'geo2_d16_summary_v3.png')

    print("\n[4/6] d=32 summary...")
    plot_dimension_summary_v3(fixed, optimized, 32, 'geo2_d32_summary_v3.png')

    print("\n[5/6] d=64 summary...")
    plot_dimension_summary_v3(fixed, optimized, 64, 'geo2_d64_summary_v3.png')

    print("\n[6/6] Linearized fits...")
    plot_linearized_v3(fixed, optimized)

    print("\n" + "="*70)
    print(f"Generated 6 plots in {OUTPUT_DIR}/")
    print("="*70)

    # List outputs
    print("\nOutput files:")
    for f in ['geo2_main_v3.png', 'geo2_scaling_v3.png',
              'geo2_d16_summary_v3.png', 'geo2_d32_summary_v3.png',
              'geo2_d64_summary_v3.png', 'geo2_linearized_v3.png']:
        fpath = OUTPUT_DIR / f
        if fpath.exists():
            size = fpath.stat().st_size / 1024
            print(f"  {f} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
