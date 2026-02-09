#!/usr/bin/env python3
"""
Create publication-ready figures for criterion ordering analysis.

This script generates clean, single-purpose figures suitable for publication:
1. Main figure: Criterion gap ratio vs dimension (ratio_vs_dimension.png)
2. K_c scaling: Critical count vs dimension with power-law fits (kc_vs_dimension.png)
3. Lambda explanation: Why Krylov is λ-independent (lambda_explanation.png)

Style: Publication-ready with serif fonts, appropriate sizing, colorblind-friendly palette.

Usage:
    python scripts/analysis/create_publication_figures.py

Author: Claude Code (research exploration)
Date: 2026-01-14
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# PUBLICATION STYLE SETTINGS
# =============================================================================

def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'axes.linewidth': 1.2,
        'axes.grid': False,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# Colorblind-friendly palette (updated for better contrast)
# Using IBM design colorblind-safe palette
COLORS = {
    'geo2': '#648FFF',       # Blue (for GEO2 ensemble)
    'canonical': '#785EF0',  # Purple (for Canonical ensemble)
    'spectral': '#DC267F',   # Magenta/Pink (for Spectral criterion)
    'krylov': '#006E00',     # Green (for Krylov criterion) - distinct from red/pink
    'moment': '#FE6100',     # Orange (for Moment criterion)
    'reference': '#666666',  # Gray
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_geo2_data() -> Optional[Dict]:
    """Load GEO2 production data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    files = sorted(data_dir.glob("geo2_production_complete_*.pkl"))
    if not files:
        return None
    with open(files[-1], 'rb') as f:
        return pickle.load(f)


def load_canonical_data() -> Optional[Dict]:
    """Load Canonical basis data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"

    spectral_file = data_dir / "spectral_complete_merged_20251216_153002.pkl"
    krylov_file = data_dir / "krylov_spectral_canonical_20251215_154634.pkl"

    data = {"spectral": {}, "krylov": {}}

    if spectral_file.exists():
        with open(spectral_file, 'rb') as f:
            sdata = pickle.load(f)
            if 'spectral' in sdata:
                data['spectral'] = sdata['spectral']

    if krylov_file.exists():
        with open(krylov_file, 'rb') as f:
            kdata = pickle.load(f)
            if 'results' in kdata:
                for d in kdata['results']:
                    if 'krylov' in kdata['results'][d]:
                        K = np.array(kdata['results'][d]['K'])
                        P = np.array(kdata['results'][d]['krylov']['P'])
                        data['krylov'][d] = {'K': K, 'P': P}

    return data if data['spectral'] or data['krylov'] else None


# =============================================================================
# FITTING UTILITIES
# =============================================================================

def fermi_dirac(rho, rho_c, delta):
    """Fermi-Dirac fit function for P(unreachable)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))


def fit_fermi_dirac(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit Fermi-Dirac to P(rho) data."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, pcov = curve_fit(
            fermi_dirac, rho[mask], P[mask],
            p0=[np.median(rho[mask]), 0.02],
            bounds=([0, 0.001], [1.0, 0.5]),
            maxfev=10000
        )
        # Compute standard errors
        perr = np.sqrt(np.diag(pcov))
        return {"rho_c": popt[0], "delta": popt[1],
                "rho_c_err": perr[0], "delta_err": perr[1]}
    except Exception:
        return None


def extract_critical_densities(geo2_data: Dict, canonical_data: Dict) -> Dict:
    """Extract critical densities from all available data."""
    results = {
        "geo2": {"d": [], "rho_c_S": [], "rho_c_K": [], "K_c_S": [], "K_c_K": []},
        "canonical": {"d": [], "rho_c_S": [], "rho_c_K": [], "K_c_S": [], "K_c_K": []},
    }

    # GEO2 data
    if geo2_data and 'results' in geo2_data:
        for d in [16, 32, 64]:
            if 'optimized' not in geo2_data['results']:
                continue
            if d not in geo2_data['results']['optimized']:
                continue

            rho_c_vals = {}
            for criterion in ['spectral', 'krylov']:
                key = (d, 0.99, criterion)
                if key not in geo2_data['results']['optimized'][d]['data']:
                    continue

                crit_data = geo2_data['results']['optimized'][d]['data'][key]
                K = np.array(crit_data['K'])
                P = np.array(crit_data['p'])
                rho = K / d**2

                fit = fit_fermi_dirac(rho, P)
                if fit:
                    rho_c_vals[criterion] = fit['rho_c']

            if 'spectral' in rho_c_vals and 'krylov' in rho_c_vals:
                results["geo2"]["d"].append(d)
                results["geo2"]["rho_c_S"].append(rho_c_vals['spectral'])
                results["geo2"]["rho_c_K"].append(rho_c_vals['krylov'])
                results["geo2"]["K_c_S"].append(rho_c_vals['spectral'] * d**2)
                results["geo2"]["K_c_K"].append(rho_c_vals['krylov'] * d**2)

    # Canonical data
    if canonical_data:
        for d in [10, 14, 18, 22, 26]:
            rho_c_vals = {}

            if d in canonical_data.get('spectral', {}):
                K = np.array(canonical_data['spectral'][d]['K'])
                P = np.array(canonical_data['spectral'][d]['P'])
                rho = K / d**2
                fit = fit_fermi_dirac(rho, P)
                if fit:
                    rho_c_vals['spectral'] = fit['rho_c']

            if d in canonical_data.get('krylov', {}):
                K = np.array(canonical_data['krylov'][d]['K'])
                P = np.array(canonical_data['krylov'][d]['P'])
                rho = K / d**2
                fit = fit_fermi_dirac(rho, P)
                if fit:
                    rho_c_vals['krylov'] = fit['rho_c']

            if 'spectral' in rho_c_vals and 'krylov' in rho_c_vals:
                results["canonical"]["d"].append(d)
                results["canonical"]["rho_c_S"].append(rho_c_vals['spectral'])
                results["canonical"]["rho_c_K"].append(rho_c_vals['krylov'])
                results["canonical"]["K_c_S"].append(rho_c_vals['spectral'] * d**2)
                results["canonical"]["K_c_K"].append(rho_c_vals['krylov'] * d**2)

    return results


# =============================================================================
# FIGURE 1: MAIN RATIO FIGURE
# =============================================================================

def create_main_ratio_figure(critical_data: Dict, output_dir: Path):
    """
    Create the main publication figure showing criterion gap ratio vs dimension.

    This is THE key result: ratio grows for GEO2, stays stable for Canonical.
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    # GEO2 data
    geo2_d = np.array(critical_data["geo2"]["d"])
    geo2_rho_S = np.array(critical_data["geo2"]["rho_c_S"])
    geo2_rho_K = np.array(critical_data["geo2"]["rho_c_K"])
    geo2_ratio = geo2_rho_S / geo2_rho_K

    # Canonical data
    can_d = np.array(critical_data["canonical"]["d"])
    can_rho_S = np.array(critical_data["canonical"]["rho_c_S"])
    can_rho_K = np.array(critical_data["canonical"]["rho_c_K"])
    can_ratio = can_rho_S / can_rho_K

    # Plot with professional styling
    ax.plot(geo2_d, geo2_ratio, 'o-', color=COLORS['geo2'],
            markersize=10, linewidth=2.5, label='GEO2 (geometric 2-local)')
    ax.plot(can_d, can_ratio, 's-', color=COLORS['canonical'],
            markersize=10, linewidth=2.5, label='Canonical (sparse Pauli)')

    # Reference line at ratio = 1
    ax.axhline(1.0, color=COLORS['reference'], linestyle='--',
               alpha=0.6, linewidth=1.5, label='Equal performance')

    # NOTE: Numerical annotations removed for cleaner publication figure
    # The data points are self-explanatory with the legend and log scale

    # Labels
    ax.set_xlabel(r'Hilbert space dimension $d$', fontsize=12)
    ax.set_ylabel(r'Ratio $\rho_c^{\mathrm{Spectral}} / \rho_c^{\mathrm{Krylov}}$', fontsize=12)
    ax.set_yscale('log')

    # Set axis limits
    ax.set_xlim(5, 70)
    ax.set_ylim(0.8, 20)

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Add interpretation annotation
    ax.text(0.97, 0.03, r'Ratio $> 1 \Rightarrow$ Krylov detects more reachability',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            style='italic', color=COLORS['reference'])

    plt.tight_layout()

    output_file = output_dir / "main_ratio_vs_dimension.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_file}")

    # Also save PDF for publication
    pdf_file = output_dir / "main_ratio_vs_dimension.pdf"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(geo2_d, geo2_ratio, 'o-', color=COLORS['geo2'],
            markersize=10, linewidth=2.5, label='GEO2 (geometric 2-local)')
    ax.plot(can_d, can_ratio, 's-', color=COLORS['canonical'],
            markersize=10, linewidth=2.5, label='Canonical (sparse Pauli)')
    ax.axhline(1.0, color=COLORS['reference'], linestyle='--',
               alpha=0.6, linewidth=1.5, label='Equal performance')
    # NOTE: Numerical annotations removed for cleaner publication figure
    ax.set_xlabel(r'Hilbert space dimension $d$', fontsize=12)
    ax.set_ylabel(r'Ratio $\rho_c^{\mathrm{Spectral}} / \rho_c^{\mathrm{Krylov}}$', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlim(5, 70)
    ax.set_ylim(0.8, 20)
    ax.legend(loc='upper left', fontsize=10)
    ax.text(0.97, 0.03, r'Ratio $> 1 \Rightarrow$ Krylov wins',
            transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
            style='italic', color=COLORS['reference'])
    plt.tight_layout()
    plt.savefig(pdf_file, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {pdf_file}")


# =============================================================================
# FIGURE 2: K_c vs d (CRITICAL COUNT, NOT DENSITY)
# =============================================================================

def create_kc_scaling_figure(critical_data: Dict, output_dir: Path):
    """
    Create figure showing K_c (critical Hamiltonian count) vs dimension.

    This avoids the ρ = K/d² normalization ambiguity.
    Key question: How does K_c scale with d?

    Unified styling:
    - GEO2: Filled markers (circles/squares), solid fit lines
    - Canonical: Open markers (same shapes), dashed fit lines
    - Spectral: Red circles
    - Krylov: Orange squares
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Data extraction
    geo2_d = np.array(critical_data["geo2"]["d"])
    geo2_Kc_S = np.array(critical_data["geo2"]["K_c_S"])
    geo2_Kc_K = np.array(critical_data["geo2"]["K_c_K"])

    can_d = np.array(critical_data["canonical"]["d"])
    can_Kc_S = np.array(critical_data["canonical"]["K_c_S"])
    can_Kc_K = np.array(critical_data["canonical"]["K_c_K"])

    d_fit = np.linspace(8, 80, 100)

    # Fit power law K_c ~ d^α for GEO2
    geo2_fits = {}
    if len(geo2_d) >= 2:
        log_d = np.log(geo2_d)
        slope_K, intercept_K = np.polyfit(log_d, np.log(geo2_Kc_K), 1)
        slope_S, intercept_S = np.polyfit(log_d, np.log(geo2_Kc_S), 1)
        geo2_fits = {'K': (slope_K, intercept_K), 'S': (slope_S, intercept_S)}

    # Fit power law K_c ~ d^α for Canonical
    can_fits = {}
    if len(can_d) >= 2:
        log_d_can = np.log(can_d)
        slope_K_can, intercept_K_can = np.polyfit(log_d_can, np.log(can_Kc_K), 1)
        slope_S_can, intercept_S_can = np.polyfit(log_d_can, np.log(can_Kc_S), 1)
        can_fits = {'K': (slope_K_can, intercept_K_can), 'S': (slope_S_can, intercept_S_can)}

    # --- Plot fit lines first (behind data points) ---

    # GEO2 fit lines (solid)
    if geo2_fits:
        slope_S, intercept_S = geo2_fits['S']
        slope_K, intercept_K = geo2_fits['K']
        ax.plot(d_fit, np.exp(intercept_S) * d_fit**slope_S, '-',
               color=COLORS['spectral'], linewidth=2, alpha=0.6)
        ax.plot(d_fit, np.exp(intercept_K) * d_fit**slope_K, '-',
               color=COLORS['krylov'], linewidth=2, alpha=0.6)

    # Canonical fit lines (dashed)
    if can_fits:
        slope_S_can, intercept_S_can = can_fits['S']
        slope_K_can, intercept_K_can = can_fits['K']
        ax.plot(d_fit, np.exp(intercept_S_can) * d_fit**slope_S_can, '--',
               color=COLORS['spectral'], linewidth=1.5, alpha=0.5)
        ax.plot(d_fit, np.exp(intercept_K_can) * d_fit**slope_K_can, '--',
               color=COLORS['krylov'], linewidth=1.5, alpha=0.5)

    # Reference lines
    ax.plot(d_fit, d_fit, ':', color='gray', alpha=0.4, linewidth=1.5)
    ax.plot(d_fit, d_fit**2 / 50, ':', color='gray', alpha=0.3, linewidth=1.5)
    ax.text(70, 75, r'$\propto d$', fontsize=9, color='gray', alpha=0.6)
    ax.text(70, 90, r'$\propto d^2$', fontsize=9, color='gray', alpha=0.4)

    # --- Plot data points (on top) ---

    # GEO2 data points (filled)
    ax.plot(geo2_d, geo2_Kc_S, 'o', color=COLORS['spectral'], markersize=12,
           markeredgecolor='white', markeredgewidth=1.5,
           label=f'GEO2 Spectral' + (f' ($\\propto d^{{{geo2_fits["S"][0]:.2f}}}$)' if geo2_fits else ''))
    ax.plot(geo2_d, geo2_Kc_K, 's', color=COLORS['krylov'], markersize=12,
           markeredgecolor='white', markeredgewidth=1.5,
           label=f'GEO2 Krylov' + (f' ($\\propto d^{{{geo2_fits["K"][0]:.2f}}}$)' if geo2_fits else ''))

    # Canonical data points (open/hollow)
    ax.plot(can_d, can_Kc_S, 'o', color=COLORS['spectral'], markersize=10,
           markerfacecolor='white', markeredgewidth=2,
           label=f'Canonical Spectral' + (f' ($\\propto d^{{{can_fits["S"][0]:.2f}}}$)' if can_fits else ''))
    ax.plot(can_d, can_Kc_K, 's', color=COLORS['krylov'], markersize=10,
           markerfacecolor='white', markeredgewidth=2,
           label=f'Canonical Krylov' + (f' ($\\propto d^{{{can_fits["K"][0]:.2f}}}$)' if can_fits else ''))

    # Labels and formatting
    ax.set_xlabel(r'Hilbert space dimension $d$', fontsize=12)
    ax.set_ylabel(r'Critical count $K_c$', fontsize=12)
    ax.set_title(r'$K_c$ Scaling: GEO2 vs Canonical', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8, 80)
    ax.set_ylim(5, 1500)

    # Legend with unified style info
    legend = ax.legend(fontsize=9, loc='upper left', framealpha=0.95)

    # Add note about marker styles
    ax.text(0.98, 0.02, 'Filled = GEO2, Open = Canonical',
           transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
           style='italic', color='gray')

    plt.tight_layout()

    output_file = output_dir / "kc_vs_dimension.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_file}")


# =============================================================================
# FIGURE 3: LAMBDA DEPENDENCE EXPLANATION
# =============================================================================

def create_lambda_explanation_figure(output_dir: Path):
    """
    Create conceptual figure explaining why Krylov is λ-independent.

    Key insight: Krylov subspace depends on direction of H, not magnitude.
    """
    set_publication_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: Spectral criterion
    ax = axes[0]
    ax.set_title('Spectral Criterion: λ-Dependent', fontsize=11, fontweight='bold')

    # Create illustration of eigenbasis rotation
    theta_vals = [0, np.pi/6, np.pi/3]
    colors = ['#E94F37', '#F39237', '#1B998B']

    for i, (theta, col) in enumerate(zip(theta_vals, colors)):
        # Draw eigenvectors
        length = 1.5
        ax.arrow(0, 0, length*np.cos(theta), length*np.sin(theta),
                head_width=0.1, head_length=0.1, fc=col, ec=col, linewidth=2)
        ax.arrow(0, 0, length*np.cos(theta+np.pi/2), length*np.sin(theta+np.pi/2),
                head_width=0.1, head_length=0.1, fc=col, ec=col, linewidth=2, alpha=0.5)

    # Add labels
    ax.text(1.7, 0.3, r'$|n(\lambda_1)\rangle$', fontsize=10, color=colors[0])
    ax.text(1.4, 1.0, r'$|n(\lambda_2)\rangle$', fontsize=10, color=colors[1])
    ax.text(0.8, 1.5, r'$|n(\lambda_3)\rangle$', fontsize=10, color=colors[2])

    # Target state
    ax.plot(1.2, 1.2, '*', markersize=20, color='black', label=r'Target $|\phi\rangle$')
    ax.annotate(r'$|\phi\rangle$', (1.2, 1.2), xytext=(1.4, 1.4), fontsize=11)

    ax.set_xlim(-0.5, 2.2)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Explanation text
    ax.text(0.5, -0.3, r'Different $\lambda \Rightarrow$ different eigenbasis',
           fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.45, r'$S(\lambda) = \sum_n |\langle n(\lambda)|\phi\rangle| |\langle n(\lambda)|\psi\rangle|$',
           fontsize=10, ha='center', transform=ax.transAxes, style='italic')

    # Right panel: Krylov criterion
    ax = axes[1]
    ax.set_title('Krylov Criterion: λ-Independent', fontsize=11, fontweight='bold')

    # Draw Krylov subspace as a plane/cone
    from mpl_toolkits.mplot3d import Axes3D

    # 2D representation of subspace
    # Draw vectors showing H^k|ψ⟩ for different k
    ax.arrow(0, 0, 1.5, 0, head_width=0.1, head_length=0.1,
            fc=COLORS['krylov'], ec=COLORS['krylov'], linewidth=2)
    ax.arrow(0, 0, 1.2, 0.8, head_width=0.1, head_length=0.1,
            fc=COLORS['krylov'], ec=COLORS['krylov'], linewidth=2, alpha=0.7)
    ax.arrow(0, 0, 0.8, 1.3, head_width=0.1, head_length=0.1,
            fc=COLORS['krylov'], ec=COLORS['krylov'], linewidth=2, alpha=0.5)

    # Labels
    ax.text(1.6, -0.15, r'$|\psi\rangle$', fontsize=10)
    ax.text(1.3, 0.9, r'$H|\psi\rangle$', fontsize=10)
    ax.text(0.6, 1.4, r'$H^2|\psi\rangle$', fontsize=10)

    # Show subspace span
    from matplotlib.patches import Polygon
    triangle = Polygon([(0, 0), (1.5, 0), (0.8, 1.3)], alpha=0.2,
                       color=COLORS['krylov'], label='Krylov subspace')
    ax.add_patch(triangle)

    # Target projection
    ax.plot(1.0, 0.6, '*', markersize=20, color='black')
    ax.annotate(r'$P_{\mathcal{K}}|\phi\rangle$', (1.0, 0.6), xytext=(1.2, 0.8), fontsize=10)

    ax.set_xlim(-0.3, 2.0)
    ax.set_ylim(-0.5, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Explanation text
    ax.text(0.5, -0.3, r'Scaling $\lambda \rightarrow c\lambda$: $H(c\lambda)^k|\psi\rangle = c^k H(\lambda)^k|\psi\rangle$',
           fontsize=9, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.45, r'Same span! Only ratios $\lambda_i/\lambda_j$ matter.',
           fontsize=10, ha='center', transform=ax.transAxes, style='italic', fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / "lambda_explanation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CREATING PUBLICATION-READY FIGURES")
    print("=" * 70)

    # Load data
    geo2_data = load_geo2_data()
    canonical_data = load_canonical_data()

    print(f"GEO2 data: {'loaded' if geo2_data else 'not found'}")
    print(f"Canonical data: {'loaded' if canonical_data else 'not found'}")

    # Extract critical densities
    critical_data = extract_critical_densities(geo2_data, canonical_data)

    print("\nExtracted critical densities:")
    print(f"  GEO2: d = {critical_data['geo2']['d']}")
    print(f"  Canonical: d = {critical_data['canonical']['d']}")

    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "fig" / "publication"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\n--- Figure 1: Main Ratio Figure ---")
    create_main_ratio_figure(critical_data, output_dir)

    print("\n--- Figure 2: K_c Scaling ---")
    create_kc_scaling_figure(critical_data, output_dir)

    print("\n--- Figure 3: Lambda Explanation ---")
    create_lambda_explanation_figure(output_dir)

    print("\n" + "=" * 70)
    print("PUBLICATION FIGURES COMPLETE")
    print("=" * 70)
    print(f"\nAll figures saved to: {output_dir}")
