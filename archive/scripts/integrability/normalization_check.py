#!/usr/bin/env python3
"""
Check if K/L normalization (instead of K/d²) resolves criterion ordering difference.

HYPOTHESIS: The apparent difference between GEO2 and Canonical might be an artifact
of comparing K/d² when the available basis sizes L differ drastically.

| Ensemble   | L (operators)      | K/d² interpretation |
|------------|--------------------|--------------------|
| GEO2       | 3n + 9|E| ~ 21n    | Fraction of d² used |
| Canonical  | d²                 | Same fraction |
| GUE        | d² (effectively)   | Same fraction |

If we use K/L instead:
- GEO2: K/L = fraction of available operators used
- Canonical: K/L = K/d² (same)

PREDICTION: If normalization is the issue, replotting with K/L should make
GEO2 look more similar to Canonical.

Usage:
    python scripts/integrability/normalization_check.py

Author: Claude Code (research exploration)
Date: 2026-01-13
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
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
        popt, _ = curve_fit(
            fermi_dirac, rho[mask], P[mask],
            p0=[np.median(rho[mask]), 0.02],
            bounds=([0, 0.001], [1.0, 0.5]),
            maxfev=10000
        )
        y_pred = fermi_dirac(rho[mask], *popt)
        ss_res = np.sum((P[mask] - y_pred)**2)
        ss_tot = np.sum((P[mask] - np.mean(P[mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"rho_c": popt[0], "delta": popt[1], "R2": r2}
    except Exception:
        return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_geo2_data() -> Optional[Dict]:
    """Load GEO2 production data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    files = sorted(data_dir.glob("geo2_production_complete_*.pkl"))
    if not files:
        print("Warning: No GEO2 data found")
        return None

    with open(files[-1], 'rb') as f:
        return pickle.load(f)


def load_canonical_data() -> Optional[Dict]:
    """Load Canonical spectral/krylov data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"

    # Try to load spectral data
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
# NORMALIZATION ANALYSIS
# =============================================================================

def compute_basis_sizes():
    """Compute basis sizes for different ensembles and dimensions."""
    results = {
        "geo2": {},
        "canonical": {},
    }

    # GEO2 lattice configurations
    geo2_configs = [
        (2, 2, 16),   # 2×2 lattice, d=16
        (1, 5, 32),   # 1×5 chain, d=32
        (2, 3, 64),   # 2×3 lattice, d=64
    ]

    for nx, ny, d in geo2_configs:
        n = nx * ny
        # Edges for rectangular lattice (open BC)
        edges_h = (nx - 1) * ny  # horizontal edges
        edges_v = nx * (ny - 1)  # vertical edges
        n_edges = edges_h + edges_v
        L = 3 * n + 9 * n_edges
        results["geo2"][d] = {
            "nx": nx, "ny": ny, "n": n, "edges": n_edges,
            "L": L, "d_squared": d**2, "L_over_d2": L / d**2
        }

    # Canonical basis
    for d in [16, 32, 64, 10, 14, 18, 22, 26]:
        results["canonical"][d] = {
            "L": d**2, "d_squared": d**2, "L_over_d2": 1.0
        }

    return results


def analyze_normalization_effect(geo2_data: Dict, canonical_data: Dict) -> Dict:
    """
    Analyze how K/L vs K/d² normalization affects criterion ordering.

    Returns comparison of ρ_c values under both normalizations.
    """
    basis_sizes = compute_basis_sizes()

    results = {
        "geo2": {},
        "canonical": {},
        "comparison": []
    }

    print("\n" + "=" * 70)
    print("NORMALIZATION ANALYSIS")
    print("=" * 70)

    # Analyze GEO2
    if geo2_data:
        print("\n--- GEO2 ---")
        for approach in ['optimized']:  # Focus on optimized λ
            if approach not in geo2_data['results']:
                continue
            for d in [16, 32, 64]:
                if d not in geo2_data['results'][approach]:
                    continue

                # Get basis size
                L = basis_sizes["geo2"][d]["L"] if d in basis_sizes["geo2"] else d**2

                for criterion in ['spectral', 'krylov']:
                    key = (d, 0.99, criterion)
                    if key not in geo2_data['results'][approach][d]['data']:
                        continue

                    crit_data = geo2_data['results'][approach][d]['data'][key]
                    K = np.array(crit_data['K'])
                    P = np.array(crit_data['p'])

                    # Compute both normalizations
                    rho_d2 = K / d**2       # K/d²
                    rho_L = K / L           # K/L

                    # Fit with K/d²
                    fit_d2 = fit_fermi_dirac(rho_d2, P)
                    # Fit with K/L
                    fit_L = fit_fermi_dirac(rho_L, P)

                    label = f"GEO2 d={d} {criterion}"
                    results["geo2"][label] = {
                        "d": d, "L": L, "criterion": criterion,
                        "rho_c_d2": fit_d2["rho_c"] if fit_d2 else np.nan,
                        "rho_c_L": fit_L["rho_c"] if fit_L else np.nan,
                    }

                    if fit_d2 and fit_L:
                        print(f"  {label}:")
                        print(f"    L={L}, L/d²={L/d**2:.3f}")
                        print(f"    ρ_c(K/d²) = {fit_d2['rho_c']:.4f}")
                        print(f"    ρ_c(K/L)  = {fit_L['rho_c']:.4f}")

    # Analyze Canonical
    if canonical_data:
        print("\n--- Canonical ---")
        for d in [10, 14, 18, 22, 26]:
            L = d**2  # Canonical has d² operators

            for criterion in ['spectral', 'krylov']:
                if criterion in canonical_data and d in canonical_data[criterion]:
                    K = np.array(canonical_data[criterion][d]['K'])
                    P = np.array(canonical_data[criterion][d]['P'])

                    rho_d2 = K / d**2
                    rho_L = K / L  # Same as K/d² for canonical

                    fit_d2 = fit_fermi_dirac(rho_d2, P)

                    label = f"Canonical d={d} {criterion}"
                    results["canonical"][label] = {
                        "d": d, "L": L, "criterion": criterion,
                        "rho_c_d2": fit_d2["rho_c"] if fit_d2 else np.nan,
                        "rho_c_L": fit_d2["rho_c"] if fit_d2 else np.nan,  # Same
                    }

    return results


def plot_normalization_comparison(geo2_data: Dict, output_dir: Path):
    """Create comparison plot showing both normalizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not geo2_data:
        print("No GEO2 data available for plotting")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    basis_sizes = compute_basis_sizes()

    dims = [16, 32, 64]
    for idx, d in enumerate(dims):
        if d not in basis_sizes["geo2"]:
            continue

        L = basis_sizes["geo2"][d]["L"]

        # Get data
        approach = 'optimized'
        if approach not in geo2_data['results'] or d not in geo2_data['results'][approach]:
            continue

        # Top row: K/d² normalization
        ax = axes[0, idx]
        ax.set_title(f"d={d}: K/d² normalization", fontsize=10)

        for criterion, color, marker in [('spectral', 'C0', 'o'), ('krylov', 'C1', 's')]:
            key = (d, 0.99, criterion)
            if key not in geo2_data['results'][approach][d]['data']:
                continue

            crit_data = geo2_data['results'][approach][d]['data'][key]
            K = np.array(crit_data['K'])
            P = np.array(crit_data['p'])
            rho = K / d**2

            ax.plot(rho, P, marker, linestyle='-', label=criterion.capitalize(),
                   color=color, markersize=5)

            fit = fit_fermi_dirac(rho, P)
            if fit:
                ax.axvline(fit['rho_c'], color=color, linestyle='--', alpha=0.5)

        ax.set_xlabel('ρ = K/d²')
        ax.set_ylabel('P(unreachable)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Bottom row: K/L normalization
        ax = axes[1, idx]
        ax.set_title(f"d={d}: K/L normalization (L={L})", fontsize=10)

        for criterion, color, marker in [('spectral', 'C0', 'o'), ('krylov', 'C1', 's')]:
            key = (d, 0.99, criterion)
            if key not in geo2_data['results'][approach][d]['data']:
                continue

            crit_data = geo2_data['results'][approach][d]['data'][key]
            K = np.array(crit_data['K'])
            P = np.array(crit_data['p'])
            rho = K / L  # Different normalization!

            ax.plot(rho, P, marker, linestyle='-', label=criterion.capitalize(),
                   color=color, markersize=5)

            fit = fit_fermi_dirac(rho, P)
            if fit:
                ax.axvline(fit['rho_c'], color=color, linestyle='--', alpha=0.5)

        ax.set_xlabel('K/L (fraction of basis)')
        ax.set_ylabel('P(unreachable)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('GEO2: Effect of Normalization on Criterion Ordering', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "normalization_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def print_summary_table(basis_sizes: Dict):
    """Print summary table of basis sizes."""
    print("\n" + "=" * 70)
    print("BASIS SIZE SUMMARY")
    print("=" * 70)

    print("\nGEO2 Lattices:")
    print(f"{'d':>6} {'nx×ny':>8} {'n':>4} {'|E|':>4} {'L':>6} {'d²':>8} {'L/d²':>8}")
    print("-" * 50)
    for d in sorted(basis_sizes["geo2"].keys()):
        info = basis_sizes["geo2"][d]
        print(f"{d:>6} {info['nx']}×{info['ny']:>6} {info['n']:>4} {info['edges']:>4} "
              f"{info['L']:>6} {info['d_squared']:>8} {info['L_over_d2']:>8.4f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("GEO2: L = 3n + 9|E| << d² as d grows")
    print("This means K/d² vastly underestimates 'density' for GEO2")
    print("K/L may be a fairer comparison between ensembles")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NORMALIZATION ANALYSIS: K/d² vs K/L")
    print("=" * 70)

    # Compute and print basis sizes
    basis_sizes = compute_basis_sizes()
    print_summary_table(basis_sizes)

    # Load data
    geo2_data = load_geo2_data()
    canonical_data = load_canonical_data()

    # Analyze
    if geo2_data:
        results = analyze_normalization_effect(geo2_data, canonical_data)

        # Save results
        output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"normalization_analysis_{timestamp}.pkl"

        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Saved results: {output_file}")

        # Generate plots
        fig_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
        plot_normalization_comparison(geo2_data, fig_dir)
    else:
        print("\nNo data available for detailed analysis.")
        print("Run GEO2 experiments first: python scripts/geo2/run_geo2_production.py")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
