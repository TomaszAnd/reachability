#!/usr/bin/env python3
"""
Dimension Dependence Analysis for Criterion Ordering.

KEY QUESTIONS:
1. How does ρ_c(Spectral)/ρ_c(Krylov) ratio depend on dimension d?
2. How do individual ρ_c values scale with d?
3. Can we express results in unified terms (qubits n vs dimension d)?

FINDINGS FROM DATA:
- GEO2: Krylov ρ_c decreases sharply with d, Spectral increases
- Canonical: Both decrease but Krylov faster
- Ratio ρ_c(S)/ρ_c(K) grows with d → Spectral becomes relatively harder

Usage:
    python scripts/analysis/dimension_dependence.py

Author: Claude Code
Date: 2026-01-13
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# FIT FUNCTIONS
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
        popt, pcov = curve_fit(
            fermi_dirac, rho[mask], P[mask],
            p0=[np.median(rho[mask]), 0.02],
            bounds=([0, 0.001], [1.0, 0.5]),
            maxfev=10000
        )
        y_pred = fermi_dirac(rho[mask], *popt)
        ss_res = np.sum((P[mask] - y_pred)**2)
        ss_tot = np.sum((P[mask] - np.mean(P[mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Estimate errors from covariance
        perr = np.sqrt(np.diag(pcov))
        return {"rho_c": popt[0], "delta": popt[1], "R2": r2,
                "rho_c_err": perr[0], "delta_err": perr[1]}
    except Exception:
        return None


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def linear(x, a, b):
    """Linear: y = a + b*x"""
    return a + b * x


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


def load_canonical_data() -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load Canonical spectral and krylov data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"

    spectral_data = None
    krylov_data = None

    spectral_file = data_dir / "spectral_complete_merged_20251216_153002.pkl"
    if spectral_file.exists():
        with open(spectral_file, 'rb') as f:
            spectral_data = pickle.load(f)

    krylov_file = data_dir / "krylov_spectral_canonical_20251215_154634.pkl"
    if krylov_file.exists():
        with open(krylov_file, 'rb') as f:
            krylov_data = pickle.load(f)

    return spectral_data, krylov_data


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def extract_geo2_rho_c(geo2_data: Dict) -> Dict:
    """Extract ρ_c values for all dimensions from GEO2 data."""
    results = {
        "dims": [],
        "n_qubits": [],
        "spectral": {"rho_c": [], "rho_c_err": [], "delta": []},
        "krylov": {"rho_c": [], "rho_c_err": [], "delta": []},
    }

    # GEO2 lattice configurations
    lattice_configs = {
        16: (2, 2),   # 2×2 = 4 qubits
        32: (1, 5),   # 1×5 = 5 qubits
        64: (2, 3),   # 2×3 = 6 qubits
    }

    for d in [16, 32, 64]:
        if d not in geo2_data['results']['optimized']:
            continue

        nx, ny = lattice_configs[d]
        n_qubits = nx * ny

        results["dims"].append(d)
        results["n_qubits"].append(n_qubits)

        for criterion in ['spectral', 'krylov']:
            key = (d, 0.99, criterion)
            if key in geo2_data['results']['optimized'][d]['data']:
                cdata = geo2_data['results']['optimized'][d]['data'][key]
                K = np.array(cdata['K'])
                P = np.array(cdata['p'])
                rho = K / d**2

                fit = fit_fermi_dirac(rho, P)
                if fit:
                    results[criterion]["rho_c"].append(fit['rho_c'])
                    results[criterion]["rho_c_err"].append(fit.get('rho_c_err', 0.01))
                    results[criterion]["delta"].append(fit['delta'])
                else:
                    results[criterion]["rho_c"].append(np.nan)
                    results[criterion]["rho_c_err"].append(np.nan)
                    results[criterion]["delta"].append(np.nan)
            else:
                results[criterion]["rho_c"].append(np.nan)
                results[criterion]["rho_c_err"].append(np.nan)
                results[criterion]["delta"].append(np.nan)

    # Convert to arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
        elif isinstance(results[key], dict):
            for subkey in results[key]:
                results[key][subkey] = np.array(results[key][subkey])

    return results


def extract_canonical_rho_c(spectral_data: Dict, krylov_data: Dict) -> Dict:
    """Extract ρ_c values for all dimensions from Canonical data."""
    results = {
        "dims": [],
        "n_qubits": [],  # log2(d) for effective qubits
        "spectral": {"rho_c": [], "rho_c_err": [], "delta": []},
        "krylov": {"rho_c": [], "rho_c_err": [], "delta": []},
    }

    for d in [10, 14, 18, 22, 26]:
        results["dims"].append(d)
        results["n_qubits"].append(np.log2(d))  # Effective qubits

        # Spectral
        if spectral_data and 'spectral' in spectral_data and d in spectral_data['spectral']:
            K = np.array(spectral_data['spectral'][d]['K'])
            P = np.array(spectral_data['spectral'][d]['P'])
            rho = K / d**2
            fit = fit_fermi_dirac(rho, P)
            if fit:
                results["spectral"]["rho_c"].append(fit['rho_c'])
                results["spectral"]["rho_c_err"].append(fit.get('rho_c_err', 0.01))
                results["spectral"]["delta"].append(fit['delta'])
            else:
                results["spectral"]["rho_c"].append(np.nan)
                results["spectral"]["rho_c_err"].append(np.nan)
                results["spectral"]["delta"].append(np.nan)
        else:
            results["spectral"]["rho_c"].append(np.nan)
            results["spectral"]["rho_c_err"].append(np.nan)
            results["spectral"]["delta"].append(np.nan)

        # Krylov
        if krylov_data and 'results' in krylov_data and d in krylov_data['results']:
            K = np.array(krylov_data['results'][d]['K'])
            P = np.array(krylov_data['results'][d]['krylov']['P'])
            rho = K / d**2
            fit = fit_fermi_dirac(rho, P)
            if fit:
                results["krylov"]["rho_c"].append(fit['rho_c'])
                results["krylov"]["rho_c_err"].append(fit.get('rho_c_err', 0.01))
                results["krylov"]["delta"].append(fit['delta'])
            else:
                results["krylov"]["rho_c"].append(np.nan)
                results["krylov"]["rho_c_err"].append(np.nan)
                results["krylov"]["delta"].append(np.nan)
        else:
            results["krylov"]["rho_c"].append(np.nan)
            results["krylov"]["rho_c_err"].append(np.nan)
            results["krylov"]["delta"].append(np.nan)

    # Convert to arrays
    for key in results:
        if isinstance(results[key], list):
            results[key] = np.array(results[key])
        elif isinstance(results[key], dict):
            for subkey in results[key]:
                results[key][subkey] = np.array(results[key][subkey])

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_dimension_dependence(geo2_results: Dict, canonical_results: Dict, output_dir: Path):
    """Create comprehensive dimension dependence figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # =========================================================================
    # Row 1: GEO2
    # =========================================================================

    # Panel A: ρ_c vs d for GEO2
    ax = axes[0, 0]
    dims = geo2_results["dims"]
    rho_s = geo2_results["spectral"]["rho_c"]
    rho_k = geo2_results["krylov"]["rho_c"]

    ax.semilogy(dims, rho_s, 'o-', label='Spectral', markersize=10, color='C0')
    ax.semilogy(dims, rho_k, 's-', label='Krylov', markersize=10, color='C1')

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('ρ_c = K_c/d²')
    ax.set_title('GEO2: Critical Density vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: K_c vs d for GEO2
    ax = axes[0, 1]
    K_s = rho_s * dims**2
    K_k = rho_k * dims**2

    ax.plot(dims, K_s, 'o-', label='Spectral K_c', markersize=10, color='C0')
    ax.plot(dims, K_k, 's-', label='Krylov K_c', markersize=10, color='C1')

    # Fit power law for K_c
    valid = ~np.isnan(K_s)
    if np.sum(valid) >= 2:
        try:
            popt_s, _ = curve_fit(power_law, dims[valid], K_s[valid], p0=[1, 1])
            popt_k, _ = curve_fit(power_law, dims[valid], K_k[valid], p0=[1, 1])
            d_fine = np.linspace(dims.min(), dims.max(), 100)
            ax.plot(d_fine, power_law(d_fine, *popt_s), '--', color='C0', alpha=0.5,
                   label=f'K_c(S) ~ d^{popt_s[1]:.2f}')
            ax.plot(d_fine, power_law(d_fine, *popt_k), '--', color='C1', alpha=0.5,
                   label=f'K_c(K) ~ d^{popt_k[1]:.2f}')
        except:
            pass

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('K_c (critical number of Hamiltonians)')
    ax.set_title('GEO2: Critical K vs Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Ratio ρ_c(S)/ρ_c(K) for GEO2
    ax = axes[0, 2]
    ratio_geo2 = rho_s / rho_k
    ax.plot(dims, ratio_geo2, 'o-', markersize=10, color='purple')

    for d, r in zip(dims, ratio_geo2):
        ax.annotate(f'{r:.1f}', (d, r), textcoords="offset points",
                   xytext=(5, 5), fontsize=10)

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('ρ_c(Spectral) / ρ_c(Krylov)')
    ax.set_title('GEO2: Criterion Gap Ratio vs Dimension')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Row 2: Canonical
    # =========================================================================

    # Panel D: ρ_c vs d for Canonical
    ax = axes[1, 0]
    dims_c = canonical_results["dims"]
    rho_s_c = canonical_results["spectral"]["rho_c"]
    rho_k_c = canonical_results["krylov"]["rho_c"]

    valid = ~np.isnan(rho_s_c) & ~np.isnan(rho_k_c)

    ax.semilogy(dims_c[valid], rho_s_c[valid], 'o-', label='Spectral', markersize=10, color='C0')
    ax.semilogy(dims_c[valid], rho_k_c[valid], 's-', label='Krylov', markersize=10, color='C1')

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('ρ_c = K_c/d²')
    ax.set_title('Canonical: Critical Density vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel E: K_c vs d for Canonical
    ax = axes[1, 1]
    K_s_c = rho_s_c * dims_c**2
    K_k_c = rho_k_c * dims_c**2

    ax.plot(dims_c[valid], K_s_c[valid], 'o-', label='Spectral K_c', markersize=10, color='C0')
    ax.plot(dims_c[valid], K_k_c[valid], 's-', label='Krylov K_c', markersize=10, color='C1')

    # Fit linear for K_c (canonical tends to be linear in d)
    try:
        popt_s, _ = curve_fit(linear, dims_c[valid], K_s_c[valid])
        popt_k, _ = curve_fit(linear, dims_c[valid], K_k_c[valid])
        d_fine = np.linspace(dims_c.min(), dims_c.max(), 100)
        ax.plot(d_fine, linear(d_fine, *popt_s), '--', color='C0', alpha=0.5,
               label=f'K_c(S) = {popt_s[1]:.2f}d + {popt_s[0]:.1f}')
        ax.plot(d_fine, linear(d_fine, *popt_k), '--', color='C1', alpha=0.5,
               label=f'K_c(K) = {popt_k[1]:.2f}d + {popt_k[0]:.1f}')
    except:
        pass

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('K_c (critical number of Hamiltonians)')
    ax.set_title('Canonical: Critical K vs Dimension')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel F: Ratio comparison
    ax = axes[1, 2]
    ratio_can = rho_s_c / rho_k_c

    ax.plot(dims_c[valid], ratio_can[valid], 'o-', markersize=10, color='purple',
           label='Canonical')
    ax.plot(dims, ratio_geo2, 's--', markersize=10, color='green', alpha=0.7,
           label='GEO2')

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('ρ_c(Spectral) / ρ_c(Krylov)')
    ax.set_title('Criterion Gap Ratio: GEO2 vs Canonical')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Dimension Dependence of Criterion Ordering', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "dimension_dependence_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_unified_qubit_comparison(geo2_results: Dict, canonical_results: Dict, output_dir: Path):
    """Create unified comparison using qubits (or log2(d)) as x-axis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: K_c vs n (qubits)
    ax = axes[0]

    # GEO2
    n_geo2 = geo2_results["n_qubits"]
    K_s_geo2 = geo2_results["spectral"]["rho_c"] * geo2_results["dims"]**2
    K_k_geo2 = geo2_results["krylov"]["rho_c"] * geo2_results["dims"]**2

    ax.semilogy(n_geo2, K_s_geo2, 'o-', label='GEO2 Spectral', markersize=10, color='C0')
    ax.semilogy(n_geo2, K_k_geo2, 's-', label='GEO2 Krylov', markersize=10, color='C1')

    # Canonical (use log2(d) as effective qubits)
    n_can = canonical_results["n_qubits"]
    K_s_can = canonical_results["spectral"]["rho_c"] * canonical_results["dims"]**2
    K_k_can = canonical_results["krylov"]["rho_c"] * canonical_results["dims"]**2

    valid = ~np.isnan(K_s_can) & ~np.isnan(K_k_can)
    ax.semilogy(n_can[valid], K_s_can[valid], 'o--', label='Canonical Spectral',
               markersize=10, color='C0', alpha=0.5)
    ax.semilogy(n_can[valid], K_k_can[valid], 's--', label='Canonical Krylov',
               markersize=10, color='C1', alpha=0.5)

    # Add 2^n and 4^n reference lines
    n_range = np.linspace(3.5, 6.5, 100)
    ax.plot(n_range, 2**n_range, ':', color='gray', alpha=0.5, label='2^n (linear in d)')
    ax.plot(n_range, 4**n_range / 100, ':', color='black', alpha=0.5, label='4^n/100 (quadratic in d)')

    ax.set_xlabel('n (qubits for GEO2, log₂d for Canonical)')
    ax.set_ylabel('K_c (critical Hamiltonians)')
    ax.set_title('Critical K vs Effective Qubits')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: ρ_c vs n
    ax = axes[1]

    ax.semilogy(n_geo2, geo2_results["spectral"]["rho_c"], 'o-',
               label='GEO2 Spectral', markersize=10, color='C0')
    ax.semilogy(n_geo2, geo2_results["krylov"]["rho_c"], 's-',
               label='GEO2 Krylov', markersize=10, color='C1')

    ax.semilogy(n_can[valid], canonical_results["spectral"]["rho_c"][valid], 'o--',
               label='Canonical Spectral', markersize=10, color='C0', alpha=0.5)
    ax.semilogy(n_can[valid], canonical_results["krylov"]["rho_c"][valid], 's--',
               label='Canonical Krylov', markersize=10, color='C1', alpha=0.5)

    ax.set_xlabel('n (qubits for GEO2, log₂d for Canonical)')
    ax.set_ylabel('ρ_c = K_c/d²')
    ax.set_title('Critical Density vs Effective Qubits')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Unified Comparison: GEO2 vs Canonical', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "unified_qubit_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def print_summary_table(geo2_results: Dict, canonical_results: Dict):
    """Print summary table of dimension dependence."""
    print("\n" + "=" * 80)
    print("DIMENSION DEPENDENCE SUMMARY")
    print("=" * 80)

    print("\nGEO2 (Geometric 2-Local):")
    print(f"{'d':>6} {'n':>4} {'ρc(S)':>10} {'ρc(K)':>10} {'Ratio':>8} {'Kc(S)':>8} {'Kc(K)':>8}")
    print("-" * 60)
    for i, d in enumerate(geo2_results["dims"]):
        n = geo2_results["n_qubits"][i]
        rho_s = geo2_results["spectral"]["rho_c"][i]
        rho_k = geo2_results["krylov"]["rho_c"][i]
        ratio = rho_s / rho_k if rho_k > 0 else np.nan
        K_s = rho_s * d**2
        K_k = rho_k * d**2
        print(f"{d:>6} {n:>4} {rho_s:>10.4f} {rho_k:>10.4f} {ratio:>8.2f} {K_s:>8.1f} {K_k:>8.1f}")

    print("\nCanonical (Sparse Pauli Basis):")
    print(f"{'d':>6} {'n':>6} {'ρc(S)':>10} {'ρc(K)':>10} {'Ratio':>8} {'Kc(S)':>8} {'Kc(K)':>8}")
    print("-" * 64)
    for i, d in enumerate(canonical_results["dims"]):
        n = canonical_results["n_qubits"][i]
        rho_s = canonical_results["spectral"]["rho_c"][i]
        rho_k = canonical_results["krylov"]["rho_c"][i]
        if np.isnan(rho_s) or np.isnan(rho_k):
            continue
        ratio = rho_s / rho_k if rho_k > 0 else np.nan
        K_s = rho_s * d**2
        K_k = rho_k * d**2
        print(f"{d:>6} {n:>6.2f} {rho_s:>10.4f} {rho_k:>10.4f} {ratio:>8.2f} {K_s:>8.1f} {K_k:>8.1f}")

    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print("1. Both ensembles show Krylov < Spectral ordering (ρc(K) < ρc(S))")
    print("2. GEO2: Ratio increases dramatically with d (1.7 → 14)")
    print("3. Canonical: Ratio stays relatively stable (~1.5-1.7)")
    print("4. GEO2 Krylov ρc DECREASES with d, Spectral INCREASES")
    print("5. Canonical both decrease, but Krylov faster")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DIMENSION DEPENDENCE ANALYSIS")
    print("=" * 70)

    # Load data
    geo2_data = load_geo2_data()
    spectral_data, krylov_data = load_canonical_data()

    if geo2_data is None:
        print("Warning: No GEO2 data found")
    if spectral_data is None or krylov_data is None:
        print("Warning: Incomplete Canonical data")

    # Extract ρ_c values
    geo2_results = extract_geo2_rho_c(geo2_data) if geo2_data else None
    canonical_results = extract_canonical_rho_c(spectral_data, krylov_data)

    # Print summary
    if geo2_results:
        print_summary_table(geo2_results, canonical_results)

    # Generate plots
    output_dir = Path(__file__).parent.parent.parent / "fig" / "analysis"

    if geo2_results:
        plot_dimension_dependence(geo2_results, canonical_results, output_dir)
        plot_unified_qubit_comparison(geo2_results, canonical_results, output_dir)

    # Save results
    results = {
        "geo2": geo2_results,
        "canonical": canonical_results,
        "timestamp": datetime.now().isoformat()
    }

    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir / f"dimension_dependence_{timestamp}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
