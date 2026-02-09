#!/usr/bin/env python3
"""
Generate combined criteria plots for ALL dimensions (v24).

Part 2 of v24 spec:
- Canonical: d in {8, 16, 32, 64}
- GEO2: d in {8, 16, 32, 64}
- GEO2 vs-K versions for each d

Uses dense Krylov data from krylov_dense_resample.py runs.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path

# Import the plotting functions from plot_overnight_v7style
from plot_overnight_v7style import (
    load_merged_data,
    load_corrected_krylov_data,
    plot_combined_criteria_canonical,
    plot_combined_criteria_geo2,
    plot_combined_criteria_geo2_vs_K,
)

OUTPUT_DIR = Path(__file__).parent.parent.parent / 'fig' / 'publication'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = [8, 16, 32, 64]
TAU = 0.99


def main():
    print("=" * 70)
    print("GENERATING ALL COMBINED CRITERIA PLOTS - v24")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    print("  Canonical:", end=" ")
    df_canonical = load_merged_data('canonical')
    print("  GEO2:", end=" ")
    df_geo2 = load_merged_data('GEO2')

    # Also load Krylov data directly for GEO2 (better fitting)
    print("\nLoading dense Krylov data...")
    df_krylov_canonical = load_corrected_krylov_data('canonical')
    df_krylov_geo2 = load_corrected_krylov_data('GEO2')

    print(f"\n  Canonical Krylov: {len(df_krylov_canonical) if df_krylov_canonical is not None else 0} rows")
    print(f"  GEO2 Krylov: {len(df_krylov_geo2) if df_krylov_geo2 is not None else 0} rows")

    # Generate canonical combined plots
    print("\n" + "-" * 50)
    print("CANONICAL COMBINED PLOTS")
    print("-" * 50)

    canonical_fits = {}
    for d in DIMS:
        output_path = OUTPUT_DIR / f'combined_criteria_canonical_d{d}_tau099.png'
        print(f"\n  d={d}: {output_path.name}")
        try:
            fits = plot_combined_criteria_canonical(df_canonical, output_path, d=d, tau=TAU)
            canonical_fits[d] = fits
            for crit, fit in fits.items():
                if 'rho_c' in fit:
                    print(f"    {crit}: rho_c={fit['rho_c']:.4f}, delta={fit['delta']:.4f}, R²={fit['R2']:.3f}")
                elif 'lambda' in fit:
                    print(f"    {crit}: lambda={fit['lambda']:.4f}, R²={fit['R2']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Generate GEO2 combined plots (vs rho)
    print("\n" + "-" * 50)
    print("GEO2 COMBINED PLOTS (vs rho)")
    print("-" * 50)

    geo2_fits = {}
    for d in DIMS:
        output_path = OUTPUT_DIR / f'combined_criteria_geo2_d{d}_tau099.png'
        print(f"\n  d={d}: {output_path.name}")
        try:
            fits = plot_combined_criteria_geo2(df_geo2, output_path, d=d, tau=TAU)
            geo2_fits[d] = fits
            for crit, fit in fits.items():
                if 'rho_c' in fit:
                    print(f"    {crit}: rho_c={fit['rho_c']:.4f}, delta={fit['delta']:.4f}, R²={fit['R2']:.3f}")
                elif 'lambda' in fit:
                    print(f"    {crit}: lambda={fit['lambda']:.4f}, R²={fit['R2']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Generate GEO2 combined plots (vs K)
    print("\n" + "-" * 50)
    print("GEO2 COMBINED PLOTS (vs K)")
    print("-" * 50)

    for d in DIMS:
        output_path = OUTPUT_DIR / f'geo2_combined_criteria_vs_K_d{d}_tau099.png'
        print(f"\n  d={d}: {output_path.name}")
        try:
            plot_combined_criteria_geo2_vs_K(df_geo2, output_path, d=d, tau=TAU)
            print(f"    Generated successfully")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nGenerated {len(DIMS)} canonical combined plots")
    print(f"Generated {len(DIMS)} GEO2 combined plots (vs rho)")
    print(f"Generated {len(DIMS)} GEO2 combined plots (vs K)")
    print(f"\nTotal: {3 * len(DIMS)} plots in {OUTPUT_DIR}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('combined_criteria_*.png')):
        print(f"  {f.name}")
    for f in sorted(OUTPUT_DIR.glob('geo2_combined_criteria_vs_K_*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
