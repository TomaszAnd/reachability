#!/usr/bin/env python3
"""
Overnight production sweep for publication-quality phase transition data.

Sweeps d = 16, 32, 64, 128, 256 for both Canonical and QubitGrid models.
Produces combined plots and rho_c vs d analysis.

Usage:
    python scripts/overnight_sweep_v87.py              # Full sweep (~12 hours)
    python scripts/overnight_sweep_v87.py --quick      # d=16 only (~10 min)
    python scripts/overnight_sweep_v87.py --dims 16 32 # Custom dimensions
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for overnight runs
import matplotlib.pyplot as plt

from src.models import CanonicalQuditModel, QubitGridModel, LATTICE_CONFIGS
from src.sampling import DensitySweep, SweepConfig
from src.plotting import DIM_COLORS


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("data/overnight_v87")
FIG_DIR = Path("fig/overnight_v87")

# Extended DIM_COLORS for d=128, 256
EXTENDED_DIM_COLORS = {**DIM_COLORS, 128: 'purple', 256: 'brown'}


def get_config_for_dim(d: int) -> SweepConfig:
    """Get sweep configuration optimized for dimension d.

    Balances statistical power (n_trials) against runtime.
    Adaptive restarts in _maximize() handle convergence automatically.
    """
    if d <= 32:
        return SweepConfig(
            n_hamiltonians=100, n_targets=20, tau=0.99,
            maxiter=100, restarts=5,
        )
    elif d <= 64:
        return SweepConfig(
            n_hamiltonians=80, n_targets=15, tau=0.99,
            maxiter=100, restarts=5,
        )
    elif d <= 128:
        return SweepConfig(
            n_hamiltonians=50, n_targets=10, tau=0.99,
            maxiter=150, restarts=5,
        )
    else:  # d=256
        return SweepConfig(
            n_hamiltonians=30, n_targets=8, tau=0.99,
            maxiter=200, restarts=5,
        )


def get_K_values(d: int, model_type: str, model=None) -> list:
    """Generate K values with adaptive sampling near estimated rho_c.

    rho_c scales approximately as 1/sqrt(d) from a baseline of ~0.12 at d=8.
    Dense sampling near the transition, sparse in the tails.
    """
    # Estimate rho_c based on observed scaling
    if model_type == 'canonical':
        rho_c_est = 0.12 * np.sqrt(8 / d)
    else:
        rho_c_est = 0.08 * np.sqrt(8 / d)

    rho_c_est = np.clip(rho_c_est, 0.003, 0.20)

    rho_values = []

    # Very low rho (should be P~1)
    if d >= 64:
        rho_values.extend([0.002, 0.005, 0.008])
    rho_values.extend([0.01, 0.015, 0.02])

    # Dense around transition (12 points)
    rho_trans = np.linspace(0.5 * rho_c_est, 2.5 * rho_c_est, 12)
    rho_values.extend(rho_trans.tolist())

    # Sparse above transition (6 points)
    rho_high = np.linspace(3 * rho_c_est, min(0.4, 8 * rho_c_est), 6)
    rho_values.extend(rho_high.tolist())

    # Convert to K, deduplicate, sort
    K_values = sorted(set(max(2, int(rho * d**2)) for rho in rho_values))

    # Apply K_max limit
    K_max = model.K if model is not None else d**2 - 1
    K_max = min(K_max, 8000)  # Runtime cap

    K_values = [k for k in K_values if 2 <= k <= K_max]
    return K_values


def estimate_rho_c(df, criterion='spectral'):
    """Estimate rho_c as the rho where P crosses 0.5 (linear interpolation)."""
    P = df[f'{criterion}_P'].values
    rho = df['rho'].values

    for i in range(len(P) - 1):
        if P[i] >= 0.5 >= P[i + 1]:
            if P[i] != P[i + 1]:
                frac = (P[i] - 0.5) / (P[i] - P[i + 1])
                return rho[i] + frac * (rho[i + 1] - rho[i])
            return rho[i]

    # Fallback: closest to 0.5
    idx = np.argmin(np.abs(P - 0.5))
    return rho[idx]


# =============================================================================
# Plotting
# =============================================================================

def plot_combined(df, model_name, d, save_path):
    """Create combined 3-panel plot for one dimension."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, crit in zip(axes, ['moment', 'spectral', 'krylov']):
        ax.errorbar(df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                    fmt='o-', color='steelblue', capsize=3, markersize=4)
        ax.set_xlabel(r'$\rho = K/d^2$', fontsize=11)
        ax.set_ylabel('P(unreachable)', fontsize=11)
        ax.set_title(f'{crit.capitalize()}', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Mark estimated rho_c
        try:
            P = df[f'{crit}_P'].values
            rho = df['rho'].values
            idx_half = np.argmin(np.abs(P - 0.5))
            if 0.1 < P[idx_half] < 0.9 and 0 < idx_half < len(P) - 1:
                ax.axvline(rho[idx_half], color='red', linestyle='--', alpha=0.5,
                           label=fr'$\rho_c \approx {rho[idx_half]:.4f}$')
                ax.legend(loc='upper right', fontsize=9)
        except Exception:
            pass

    n_trials = int(df['n_trials'].iloc[0]) if len(df) > 0 else 0
    fig.suptitle(f'{model_name} d={d} ({n_trials} trials/K)', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_transition(all_results, model_name, save_dir):
    """Plot rho_c vs d (log-log) and rho_c vs 1/d."""
    dims = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for crit, color, marker in [('spectral', 'C0', 'o'),
                                 ('krylov', 'C1', 's'),
                                 ('moment', 'C2', '^')]:
        rho_c_values = []
        for d in dims:
            rho_c = estimate_rho_c(all_results[d], crit)
            rho_c_values.append(rho_c)

        axes[0].plot(dims, rho_c_values, f'{marker}-', color=color,
                     label=crit.capitalize(), markersize=8)
        axes[1].plot([1 / d for d in dims], rho_c_values, f'{marker}-',
                     color=color, label=crit.capitalize(), markersize=8)

    axes[0].set_xlabel('Dimension d', fontsize=11)
    axes[0].set_ylabel(r'$\rho_c$ (P=0.5 crossing)', fontsize=11)
    axes[0].set_title(f'{model_name}: Phase Transition vs Dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    axes[1].set_xlabel('1/d', fontsize=11)
    axes[1].set_ylabel(r'$\rho_c$ (P=0.5 crossing)', fontsize=11)
    axes[1].set_title(f'{model_name}: Phase Transition vs 1/d')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name.lower()}_rho_c_scaling.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_dimensions(all_results, model_name, criterion, save_dir):
    """Plot all dimensions overlaid for one criterion."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for d in sorted(all_results.keys()):
        df = all_results[d]
        color = EXTENDED_DIM_COLORS.get(d, None)
        ax.errorbar(df['rho'], df[f'{criterion}_P'], yerr=df[f'{criterion}_sem'],
                    fmt='o-', color=color, label=f'd={d}', capsize=2, markersize=4)

    ax.set_xlabel(r'$\rho = K/d^2$', fontsize=12)
    ax.set_ylabel('P(unreachable)', fontsize=12)
    ax.set_title(f'{model_name} - {criterion.capitalize()} Criterion', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name.lower()}_{criterion}_all_dims.png',
                dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight production sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: d=16 only")
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[16, 32, 64, 128, 256],
                        help="Dimensions to sweep")
    parser.add_argument("--canonical-only", action="store_true",
                        help="Only run Canonical model")
    parser.add_argument("--qubitgrid-only", action="store_true",
                        help="Only run QubitGrid model")
    args = parser.parse_args()

    if args.quick:
        dims = [16]
    else:
        dims = args.dims

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print(f"Starting overnight sweep at {start_time}")
    print(f"Dimensions: {dims}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    timing_log = []
    canonical_results = {}
    qubitgrid_results = {}

    # =========================================================================
    # Canonical Qudit Model
    # =========================================================================
    if not args.qubitgrid_only:
        print("\n" + "=" * 60)
        print("CANONICAL QUDIT MODEL")
        print("=" * 60)

        for d in dims:
            print(f"\n--- Canonical d={d} ---")
            model = CanonicalQuditModel(dim=d, seed=42)
            config = get_config_for_dim(d)
            K_values = get_K_values(d, 'canonical', model)

            n_trials = config.n_hamiltonians * config.n_targets
            print(f"  K values: {len(K_values)} points, range [{K_values[0]}, {K_values[-1]}]")
            print(f"  Config: {config.n_hamiltonians}h x {config.n_targets}t = "
                  f"{n_trials} trials/K, maxiter={config.maxiter}, restarts={config.restarts}")

            sweep = DensitySweep(model, config)
            t0 = time.time()
            df = sweep.run(K_values, criteria=['moment', 'spectral', 'krylov'],
                           early_stop_zeros=5)
            runtime = time.time() - t0

            # Save CSV
            csv_path = OUTPUT_DIR / f"canonical_d{d}.csv"
            df.to_csv(csv_path, index=False)
            canonical_results[d] = df
            timing_log.append({'model': 'canonical', 'd': d,
                               'runtime_min': runtime / 60,
                               'n_K': len(df), 'n_trials': n_trials})

            print(f"\n  Canonical d={d}: {len(df)} K values, {runtime/60:.1f} min")

            # Plot combined
            plot_combined(df, 'Canonical', d, FIG_DIR / f'canonical_d{d}_combined.png')

            # Save timing log incrementally
            pd.DataFrame(timing_log).to_csv(OUTPUT_DIR / 'timing_log.csv', index=False)

    # =========================================================================
    # QubitGrid Model
    # =========================================================================
    if not args.canonical_only:
        print("\n" + "=" * 60)
        print("QUBITGRID MODEL")
        print("=" * 60)

        for d in dims:
            if d not in LATTICE_CONFIGS:
                print(f"\n--- QubitGrid d={d}: No lattice config, skipping ---")
                continue

            nx, ny = LATTICE_CONFIGS[d]
            print(f"\n--- QubitGrid d={d} ({nx}x{ny} lattice) ---")
            model = QubitGridModel(dim=d, nx=nx, ny=ny, seed=42)
            config = get_config_for_dim(d)
            K_values = get_K_values(d, 'qubitgrid', model)

            n_trials = config.n_hamiltonians * config.n_targets
            print(f"  Model K_max: {model.K}")
            print(f"  K values: {len(K_values)} points, range [{K_values[0]}, {K_values[-1]}]")
            print(f"  Config: {config.n_hamiltonians}h x {config.n_targets}t = "
                  f"{n_trials} trials/K, maxiter={config.maxiter}, restarts={config.restarts}")

            sweep = DensitySweep(model, config)
            t0 = time.time()
            df = sweep.run(K_values, criteria=['moment', 'spectral', 'krylov'],
                           early_stop_zeros=5)
            runtime = time.time() - t0

            csv_path = OUTPUT_DIR / f"qubitgrid_d{d}.csv"
            df.to_csv(csv_path, index=False)
            qubitgrid_results[d] = df
            timing_log.append({'model': 'qubitgrid', 'd': d,
                               'runtime_min': runtime / 60,
                               'n_K': len(df), 'n_trials': n_trials})

            print(f"\n  QubitGrid d={d}: {len(df)} K values, {runtime/60:.1f} min")

            plot_combined(df, 'QubitGrid', d, FIG_DIR / f'qubitgrid_d{d}_combined.png')

            pd.DataFrame(timing_log).to_csv(OUTPUT_DIR / 'timing_log.csv', index=False)

    # =========================================================================
    # Summary plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 60)

    if len(canonical_results) > 1:
        plot_phase_transition(canonical_results, 'Canonical', FIG_DIR)
    if len(qubitgrid_results) > 1:
        plot_phase_transition(qubitgrid_results, 'QubitGrid', FIG_DIR)

    for crit in ['spectral', 'krylov', 'moment']:
        if canonical_results:
            plot_all_dimensions(canonical_results, 'Canonical', crit, FIG_DIR)
        if qubitgrid_results:
            plot_all_dimensions(qubitgrid_results, 'QubitGrid', crit, FIG_DIR)

    # =========================================================================
    # Final summary
    # =========================================================================
    timing_df = pd.DataFrame(timing_log)
    timing_df.to_csv(OUTPUT_DIR / 'timing_log.csv', index=False)

    total_hours = (datetime.now() - start_time).total_seconds() / 3600

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_hours:.2f} hours")
    print(f"Data: {OUTPUT_DIR}")
    print(f"Figures: {FIG_DIR}")

    print("\nTiming summary:")
    print(timing_df.to_string(index=False))

    # rho_c estimates
    print("\nPhase transition estimates (rho_c at P=0.5):")
    for model_name, results in [('Canonical', canonical_results),
                                 ('QubitGrid', qubitgrid_results)]:
        if results:
            print(f"\n  {model_name}:")
            for d in sorted(results.keys()):
                df = results[d]
                rho_c_spec = estimate_rho_c(df, 'spectral')
                rho_c_kry = estimate_rho_c(df, 'krylov')
                print(f"    d={d:3d}: Spectral rho_c={rho_c_spec:.4f}, "
                      f"Krylov rho_c={rho_c_kry:.4f}")


if __name__ == "__main__":
    main()
