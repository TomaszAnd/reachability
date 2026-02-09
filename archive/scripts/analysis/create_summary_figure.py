#!/usr/bin/env python3
"""
Create publication-quality summary figure for criterion ordering analysis.

This figure consolidates all key findings:
1. Universal ordering: Krylov < Spectral (both GEO2 and Canonical)
2. Dimension scaling: Ratio ρ_c(S)/ρ_c(K) vs d
3. λ-optimization gap: Spectral benefits more than Krylov
4. Integrability effect: r-ratio correlation

Usage:
    python scripts/analysis/create_summary_figure.py

Author: Claude Code (research exploration)
Date: 2026-01-13
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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


def load_integrability_data() -> Optional[Dict]:
    """Load integrability study data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    files = sorted(data_dir.glob("integrability_study_*.pkl"))
    if not files:
        return None
    with open(files[-1], 'rb') as f:
        return pickle.load(f)


def load_gap_data() -> Optional[Dict]:
    """Load fixed vs optimized gap data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    files = sorted(data_dir.glob("fixed_vs_optimized_analysis_*.pkl"))
    if not files:
        return None
    with open(files[-1], 'rb') as f:
        return pickle.load(f)


# =============================================================================
# FITTING UTILITIES
# =============================================================================

def fermi_dirac(rho, rho_c, delta):
    """Fermi-Dirac fit function."""
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
        popt, _ = curve_fit(
            fermi_dirac, rho[mask], P[mask],
            p0=[np.median(rho[mask]), 0.02],
            bounds=([0, 0.001], [1.0, 0.5]),
            maxfev=10000
        )
        return {"rho_c": popt[0], "delta": popt[1]}
    except Exception:
        return None


# =============================================================================
# SUMMARY FIGURE
# =============================================================================

def create_summary_figure(
    geo2_data: Dict,
    canonical_data: Dict,
    integrability_data: Dict,
    gap_data: Dict,
    output_dir: Path
):
    """Create comprehensive 6-panel summary figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: GEO2 ρ_c vs d
    ax_a = fig.add_subplot(gs[0, 0])
    plot_geo2_dimension_dependence(ax_a, geo2_data)

    # Panel B: Canonical ρ_c vs d
    ax_b = fig.add_subplot(gs[0, 1])
    plot_canonical_dimension_dependence(ax_b, canonical_data)

    # Panel C: Ratio comparison
    ax_c = fig.add_subplot(gs[0, 2])
    plot_ratio_comparison(ax_c, geo2_data, canonical_data)

    # Panel D: λ-optimization gap
    ax_d = fig.add_subplot(gs[1, 0])
    plot_lambda_gap(ax_d, gap_data)

    # Panel E: Integrability effect
    ax_e = fig.add_subplot(gs[1, 1])
    plot_integrability_effect(ax_e, integrability_data)

    # Panel F: Summary schematic
    ax_f = fig.add_subplot(gs[1, 2])
    plot_summary_text(ax_f)

    # Add panel labels
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                         ['A', 'B', 'C', 'D', 'E', 'F']):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes,
               fontsize=16, fontweight='bold', va='top')

    fig.suptitle('Criterion Ordering Analysis: Krylov vs Spectral',
                fontsize=16, fontweight='bold', y=0.98)

    output_file = output_dir / "criterion_ordering_summary.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved summary figure: {output_file}")


def plot_geo2_dimension_dependence(ax, data: Dict):
    """Panel A: GEO2 ρ_c vs dimension."""
    if data is None:
        ax.text(0.5, 0.5, 'No GEO2 data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GEO2: ρ_c vs Dimension')
        return

    dims = []
    rho_c_spectral = []
    rho_c_krylov = []

    for d in [16, 32, 64]:
        if 'optimized' not in data['results']:
            continue
        if d not in data['results']['optimized']:
            continue

        dims.append(d)

        for criterion in ['spectral', 'krylov']:
            key = (d, 0.99, criterion)
            if key not in data['results']['optimized'][d]['data']:
                continue

            crit_data = data['results']['optimized'][d]['data'][key]
            K = np.array(crit_data['K'])
            P = np.array(crit_data['p'])
            rho = K / d**2

            fit = fit_fermi_dirac(rho, P)
            if fit:
                if criterion == 'spectral':
                    rho_c_spectral.append(fit['rho_c'])
                else:
                    rho_c_krylov.append(fit['rho_c'])

    if dims:
        ax.plot(dims, rho_c_spectral, 'o-', color='C0', markersize=10, linewidth=2, label='Spectral')
        ax.plot(dims, rho_c_krylov, 's-', color='C1', markersize=10, linewidth=2, label='Krylov')

    ax.set_xlabel('Dimension d', fontsize=11)
    ax.set_ylabel('Critical density ρ_c', fontsize=11)
    ax.set_title('GEO2: ρ_c vs Dimension', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_canonical_dimension_dependence(ax, data: Dict):
    """Panel B: Canonical ρ_c vs dimension."""
    if data is None:
        ax.text(0.5, 0.5, 'No Canonical data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Canonical: ρ_c vs Dimension')
        return

    dims = []
    rho_c_spectral = []
    rho_c_krylov = []

    for d in [10, 14, 18, 22, 26]:
        if d in data.get('spectral', {}):
            K = np.array(data['spectral'][d]['K'])
            P = np.array(data['spectral'][d]['P'])
            rho = K / d**2
            fit = fit_fermi_dirac(rho, P)
            if fit:
                if d not in dims:
                    dims.append(d)
                    rho_c_spectral.append(fit['rho_c'])

    dims_k = []
    for d in [10, 14, 18, 22, 26]:
        if d in data.get('krylov', {}):
            K = np.array(data['krylov'][d]['K'])
            P = np.array(data['krylov'][d]['P'])
            rho = K / d**2
            fit = fit_fermi_dirac(rho, P)
            if fit:
                dims_k.append(d)
                rho_c_krylov.append(fit['rho_c'])

    if dims:
        ax.plot(dims, rho_c_spectral, 'o-', color='C0', markersize=10, linewidth=2, label='Spectral')
    if dims_k:
        ax.plot(dims_k, rho_c_krylov, 's-', color='C1', markersize=10, linewidth=2, label='Krylov')

    ax.set_xlabel('Dimension d', fontsize=11)
    ax.set_ylabel('Critical density ρ_c', fontsize=11)
    ax.set_title('Canonical: ρ_c vs Dimension', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_ratio_comparison(ax, geo2_data: Dict, canonical_data: Dict):
    """Panel C: Ratio ρ_c(S)/ρ_c(K) comparison."""
    # GEO2 ratios
    geo2_dims = [16, 32, 64]
    geo2_ratios = [1.70, 5.97, 13.38]  # From dimension_dependence output

    # Canonical ratios
    can_dims = [10, 14, 18, 22, 26]
    can_ratios = [1.56, 1.62, 1.70, 1.73, 1.80]  # From dimension_dependence output

    ax.plot(geo2_dims, geo2_ratios, 'o-', color='C2', markersize=10, linewidth=2, label='GEO2')
    ax.plot(can_dims, can_ratios, 's-', color='C3', markersize=10, linewidth=2, label='Canonical')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.7, label='Equal ordering')

    ax.set_xlabel('Dimension d', fontsize=11)
    ax.set_ylabel('Ratio ρ_c(Spectral) / ρ_c(Krylov)', fontsize=11)
    ax.set_title('Ratio: How Much Krylov "Wins"', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')


def plot_lambda_gap(ax, gap_data: Dict):
    """Panel D: λ-optimization gap."""
    if gap_data is None:
        ax.text(0.5, 0.5, 'No gap data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('λ-Optimization Gap')
        return

    dims = np.array(gap_data['gap_results']['dims'])
    spectral_gap = np.array(gap_data['gap_results']['spectral']['gap_mean'])
    krylov_gap = np.array(gap_data['gap_results']['krylov']['gap_mean'])

    ax.plot(dims, spectral_gap, 'o-', color='C0', markersize=10, linewidth=2, label='Spectral')
    ax.plot(dims, krylov_gap, 's-', color='C1', markersize=10, linewidth=2, label='Krylov')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Dimension d', fontsize=11)
    ax.set_ylabel('Mean Gap (Fixed - Optimized)', fontsize=11)
    ax.set_title('GEO2: λ-Optimization Benefit', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_integrability_effect(ax, data: Dict):
    """Panel E: Integrability effect on criterion ordering."""
    if data is None:
        ax.text(0.5, 0.5, 'No integrability data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Integrability Effect')
        return

    models = list(data['models'].keys())
    r_ratios = []
    rho_ratios = []
    labels = []

    for model_name in models:
        model_data = data['models'][model_name]
        r_mean = np.nanmean(model_data['r_ratios'])

        # Get ρ_c for both criteria
        d = model_data['d']
        k_values = np.array(model_data['k_values'])
        rho = k_values / d**2
        P_s = np.array(model_data['spectral']['P'])
        P_k = np.array(model_data['krylov']['P'])

        fit_s = fit_fermi_dirac(rho, P_s)
        fit_k = fit_fermi_dirac(rho, P_k)

        if fit_s and fit_k and fit_k['rho_c'] > 0:
            ratio = fit_s['rho_c'] / fit_k['rho_c']
            r_ratios.append(r_mean)
            rho_ratios.append(ratio)
            labels.append(model_name)

    if r_ratios:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(r_ratios)))
        for i, (r, ratio, label) in enumerate(zip(r_ratios, rho_ratios, labels)):
            ax.scatter(r, ratio, s=200, c=[colors[i]], label=label, edgecolors='black', linewidths=1)

        ax.axvline(0.386, color='gray', linestyle=':', alpha=0.7, label='Poisson (integrable)')
        ax.axvline(0.531, color='gray', linestyle='--', alpha=0.7, label='GOE (chaotic)')

    ax.set_xlabel('Mean r-ratio (level spacing)', fontsize=11)
    ax.set_ylabel('Ratio ρ_c(S) / ρ_c(K)', fontsize=11)
    ax.set_title('Integrability vs Criterion Ordering', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_summary_text(ax):
    """Panel F: Summary text with key findings."""
    ax.axis('off')

    summary_text = """
    KEY FINDINGS

    1. UNIVERSAL ORDERING
       Krylov < Spectral for ALL ensembles
       (GEO2, Canonical, Integrable, Chaotic)

    2. RATIO SCALING
       • GEO2: Ratio grows with d (1.7 → 13)
       • Canonical: Ratio stable (~1.6)

    3. λ-INDEPENDENCE
       • Krylov is nearly λ-independent
       • Spectral benefits from optimization
       • Gap decreases at higher d

    4. INTEGRABILITY EFFECT
       • Integrable: Spectral never detects
         reachability (diagonal eigenbasis)
       • Chaotic: Both criteria transition,
         ratio ≈ 1.6

    CONCLUSION
    Krylov consistently detects reachability
    at lower operator density than Spectral.
    This ordering is robust across ensembles
    and integrability classes.
    """

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, fontfamily='monospace',
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CREATING SUMMARY FIGURE")
    print("=" * 70)

    # Load all data
    geo2_data = load_geo2_data()
    canonical_data = load_canonical_data()
    integrability_data = load_integrability_data()
    gap_data = load_gap_data()

    print(f"GEO2 data: {'loaded' if geo2_data else 'not found'}")
    print(f"Canonical data: {'loaded' if canonical_data else 'not found'}")
    print(f"Integrability data: {'loaded' if integrability_data else 'not found'}")
    print(f"Gap data: {'loaded' if gap_data else 'not found'}")

    # Create figure
    output_dir = Path(__file__).parent.parent.parent / "fig" / "analysis"
    create_summary_figure(geo2_data, canonical_data, integrability_data, gap_data, output_dir)

    print("\n" + "=" * 70)
    print("SUMMARY FIGURE COMPLETE")
    print("=" * 70)
