#!/usr/bin/env python3
"""
Optimized overnight sweep v89 — based on micro-benchmark profiling.

Key decisions based on per-trial timing (maxiter=50, restarts=3, capped adaptive):
  d=16:  0.05s/trial → All criteria, 1000 trials
  d=32:  0.36s/trial → All criteria, 800 trials
  d=64:  3.4s/trial  → All criteria, 300 trials
  d=128: Spectral 14.7s, Krylov 78.8s → Spectral+Moment only, 200 trials
  d=256: Spectral ~200s → Infeasible on this hardware, skip

Estimated total: ~18 hours for d=16-128, both models.

Usage:
    python scripts/overnight_sweep_v89.py              # Full sweep (d=16-128)
    python scripts/overnight_sweep_v89.py --quick      # d=16 only (~5 min)
    python scripts/overnight_sweep_v89.py --dims 16 32 # Custom dimensions
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import CanonicalQuditModel, QubitGridModel, LATTICE_CONFIGS
from src.sampling import DensitySweep, SweepConfig
from src.plotting import DIM_COLORS

OUTPUT_DIR = Path("data/overnight_v89")
FIG_DIR = Path("fig/overnight_v89")

EXTENDED_DIM_COLORS = {**DIM_COLORS, 128: 'purple', 256: 'brown'}

# Krylov is too expensive at d>=128 (78.8s/trial vs 14.7s for Spectral)
KRYLOV_DIM_THRESHOLD = 128


def get_config_for_dim(d: int) -> SweepConfig:
    """Configuration based on micro-benchmark profiling.

    Uses maxiter=50, restarts=3 base. Adaptive restarts in _maximize()
    boost to 8-12 where needed but are now capped at 12.
    """
    if d <= 16:
        return SweepConfig(
            n_hamiltonians=50, n_targets=20, tau=0.99,
            maxiter=50, restarts=3,  # 1000 trials/K
        )
    elif d <= 32:
        return SweepConfig(
            n_hamiltonians=40, n_targets=20, tau=0.99,
            maxiter=50, restarts=3,  # 800 trials/K
        )
    elif d <= 64:
        return SweepConfig(
            n_hamiltonians=30, n_targets=10, tau=0.99,
            maxiter=50, restarts=3,  # 300 trials/K
        )
    else:  # d=128
        return SweepConfig(
            n_hamiltonians=20, n_targets=10, tau=0.99,
            maxiter=50, restarts=3,  # 200 trials/K
        )


def get_criteria_for_dim(d: int) -> list:
    """Select criteria based on dimension feasibility."""
    if d < KRYLOV_DIM_THRESHOLD:
        return ['moment', 'spectral', 'krylov']
    else:
        return ['moment', 'spectral']


def get_optimized_K_values(d: int, model_type: str, model=None) -> list:
    """Generate K values focused on transition region.

    Caps at 8x rho_c — everything beyond is P=0.
    """
    if model_type == 'canonical':
        rho_c = 0.12 * np.sqrt(8 / d)
    else:
        rho_c = 0.08 * np.sqrt(8 / d)

    rho_c = max(rho_c, 0.008)

    rho_values = []

    # Below transition: 3 sparse points
    rho_values.extend([0.2 * rho_c, 0.5 * rho_c, 0.75 * rho_c])

    # Transition region: 8 dense points
    rho_values.extend(np.linspace(0.85 * rho_c, 2.0 * rho_c, 8).tolist())

    # Above transition (confirm P->0): 3 sparse points
    rho_values.extend([2.5 * rho_c, 3.0 * rho_c, 4.0 * rho_c])

    K_values = sorted(set(max(2, int(rho * d**2)) for rho in rho_values))

    # Cap at 5x rho_c (P is always 0 beyond this)
    K_max_physical = int(5 * rho_c * d**2)
    K_max_model = model.K if model else d**2 - 1
    K_hard_cap = 5000 if d >= 256 else 2500
    K_max = min(K_max_physical, K_max_model, K_hard_cap)

    K_values = [k for k in K_values if 2 <= k <= K_max]
    return K_values


def estimate_rho_c(df, criterion='spectral'):
    """Estimate rho_c via linear interpolation at P=0.5."""
    P = df[f'{criterion}_P'].values
    rho = df['rho'].values

    for i in range(len(P) - 1):
        if P[i] >= 0.5 >= P[i + 1]:
            if P[i] != P[i + 1]:
                frac = (P[i] - 0.5) / (P[i] - P[i + 1])
                return rho[i] + frac * (rho[i + 1] - rho[i])
            return rho[i]

    idx = np.argmin(np.abs(P - 0.5))
    return rho[idx]


def plot_combined(df, model_name, d, criteria, save_path):
    """3-panel or 2-panel plot depending on criteria."""
    n_panels = len(criteria)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, crit in zip(axes, criteria):
        ax.errorbar(df['rho'], df[f'{crit}_P'], yerr=df[f'{crit}_sem'],
                    fmt='o-', color='steelblue', capsize=3, markersize=5)
        ax.set_xlabel(r'$\rho = K/d^2$', fontsize=11)
        ax.set_ylabel('P(unreachable)', fontsize=11)
        ax.set_title(f'{crit.capitalize()}', fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        try:
            rho_c = estimate_rho_c(df, crit)
            if 0 < rho_c < df['rho'].max():
                ax.axvline(rho_c, color='red', linestyle='--', alpha=0.5,
                           label=fr'$\rho_c \approx {rho_c:.4f}$')
                ax.legend(loc='upper right', fontsize=9)
        except Exception:
            pass

    n_trials = int(df['n_trials'].iloc[0]) if len(df) > 0 else 0
    crit_str = '+'.join(c[0].upper() for c in criteria)
    fig.suptitle(f'{model_name} d={d} ({n_trials} trials/K, {crit_str})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_transition(all_results, model_name, criteria_by_dim, save_dir):
    """Plot rho_c vs d (log-log)."""
    dims = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for crit, color, marker in [('spectral', 'C0', 'o'),
                                 ('krylov', 'C1', 's'),
                                 ('moment', 'C2', '^')]:
        rho_c_values = []
        valid_dims = []
        for d in dims:
            if crit in criteria_by_dim.get(d, []):
                rho_c = estimate_rho_c(all_results[d], crit)
                rho_c_values.append(rho_c)
                valid_dims.append(d)

        if valid_dims:
            axes[0].plot(valid_dims, rho_c_values, f'{marker}-', color=color,
                         label=crit.capitalize(), markersize=8)
            axes[1].plot([1 / d for d in valid_dims], rho_c_values, f'{marker}-',
                         color=color, label=crit.capitalize(), markersize=8)

    axes[0].set_xlabel('Dimension d', fontsize=11)
    axes[0].set_ylabel(r'$\rho_c$ (P=0.5 crossing)', fontsize=11)
    axes[0].set_title(f'{model_name}: Phase Transition Scaling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    axes[1].set_xlabel('1/d', fontsize=11)
    axes[1].set_ylabel(r'$\rho_c$ (P=0.5 crossing)', fontsize=11)
    axes[1].set_title(fr'{model_name}: $\rho_c$ vs 1/d')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name.lower()}_rho_c_scaling.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_dimensions(all_results, model_name, criterion,
                        criteria_by_dim, save_dir):
    """All dimensions overlaid for one criterion."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for d in sorted(all_results.keys()):
        if criterion not in criteria_by_dim.get(d, []):
            continue
        df = all_results[d]
        color = EXTENDED_DIM_COLORS.get(d)
        ax.errorbar(df['rho'], df[f'{criterion}_P'],
                    yerr=df[f'{criterion}_sem'],
                    fmt='o-', color=color, label=f'd={d}', capsize=2,
                    markersize=4)

    ax.set_xlabel(r'$\rho = K/d^2$', fontsize=12)
    ax.set_ylabel('P(unreachable)', fontsize=12)
    ax.set_title(f'{model_name} - {criterion.capitalize()}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name.lower()}_{criterion}_all_dims.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optimized overnight sweep v89")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: d=16 only")
    parser.add_argument("--dims", type=int, nargs="+",
                        default=[16, 32, 64, 128])
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--qubitgrid-only", action="store_true")
    parser.add_argument("--krylov-threshold", type=int, default=KRYLOV_DIM_THRESHOLD,
                        help=f"Skip Krylov for d >= this (default: {KRYLOV_DIM_THRESHOLD})")
    args = parser.parse_args()

    dims = [16] if args.quick else args.dims
    krylov_threshold = args.krylov_threshold

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    print(f"Starting optimized sweep v89 at {start_time}")
    print(f"Dimensions: {dims}")
    print(f"Krylov threshold: d < {krylov_threshold}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    timing_log = []
    canonical_results = {}
    qubitgrid_results = {}
    canonical_criteria = {}
    qubitgrid_criteria = {}

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
            K_values = get_optimized_K_values(d, 'canonical', model)
            criteria = get_criteria_for_dim(d) if d < krylov_threshold else ['moment', 'spectral']

            n_trials = config.n_hamiltonians * config.n_targets
            print(f"  K values: {len(K_values)} points, "
                  f"range [{K_values[0]}, {K_values[-1]}]")
            print(f"  Config: {config.n_hamiltonians}h x {config.n_targets}t = "
                  f"{n_trials} trials/K, maxiter={config.maxiter}")
            print(f"  Criteria: {', '.join(criteria)}")

            sweep = DensitySweep(model, config)
            t0 = time.time()
            df = sweep.run(K_values, criteria=criteria, early_stop_zeros=3)
            runtime = time.time() - t0

            df.to_csv(OUTPUT_DIR / f"canonical_d{d}.csv", index=False)
            canonical_results[d] = df
            canonical_criteria[d] = criteria
            timing_log.append({
                'model': 'canonical', 'd': d,
                'runtime_min': runtime / 60,
                'n_K': len(df), 'n_trials': n_trials,
                'criteria': '+'.join(criteria),
            })

            print(f"\n  Canonical d={d}: {len(df)} K values, {runtime/60:.1f} min")
            plot_combined(df, 'Canonical', d, criteria,
                          FIG_DIR / f'canonical_d{d}_combined.png')
            pd.DataFrame(timing_log).to_csv(
                OUTPUT_DIR / 'timing_log.csv', index=False)

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
            K_values = get_optimized_K_values(d, 'qubitgrid', model)
            criteria = get_criteria_for_dim(d) if d < krylov_threshold else ['moment', 'spectral']

            n_trials = config.n_hamiltonians * config.n_targets
            print(f"  K_max={model.K}, testing {len(K_values)} K values, "
                  f"range [{K_values[0]}, {K_values[-1]}]")
            print(f"  Config: {config.n_hamiltonians}h x {config.n_targets}t = "
                  f"{n_trials} trials/K, maxiter={config.maxiter}")
            print(f"  Criteria: {', '.join(criteria)}")

            sweep = DensitySweep(model, config)
            t0 = time.time()
            df = sweep.run(K_values, criteria=criteria, early_stop_zeros=3)
            runtime = time.time() - t0

            df.to_csv(OUTPUT_DIR / f"qubitgrid_d{d}.csv", index=False)
            qubitgrid_results[d] = df
            qubitgrid_criteria[d] = criteria
            timing_log.append({
                'model': 'qubitgrid', 'd': d,
                'runtime_min': runtime / 60,
                'n_K': len(df), 'n_trials': n_trials,
                'criteria': '+'.join(criteria),
            })

            print(f"\n  QubitGrid d={d}: {len(df)} K values, {runtime/60:.1f} min")
            plot_combined(df, 'QubitGrid', d, criteria,
                          FIG_DIR / f'qubitgrid_d{d}_combined.png')
            pd.DataFrame(timing_log).to_csv(
                OUTPUT_DIR / 'timing_log.csv', index=False)

    # =========================================================================
    # Summary plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY PLOTS")
    print("=" * 60)

    if len(canonical_results) > 1:
        plot_phase_transition(canonical_results, 'Canonical',
                              canonical_criteria, FIG_DIR)
    if len(qubitgrid_results) > 1:
        plot_phase_transition(qubitgrid_results, 'QubitGrid',
                              qubitgrid_criteria, FIG_DIR)

    for crit in ['spectral', 'krylov', 'moment']:
        if canonical_results:
            plot_all_dimensions(canonical_results, 'Canonical', crit,
                                canonical_criteria, FIG_DIR)
        if qubitgrid_results:
            plot_all_dimensions(qubitgrid_results, 'QubitGrid', crit,
                                qubitgrid_criteria, FIG_DIR)

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

    print("\nPhase transition estimates (rho_c at P=0.5):")
    for model_name, results, crit_map in [
        ('Canonical', canonical_results, canonical_criteria),
        ('QubitGrid', qubitgrid_results, qubitgrid_criteria),
    ]:
        if results:
            print(f"\n  {model_name}:")
            for d in sorted(results.keys()):
                df = results[d]
                parts = []
                for crit in crit_map.get(d, ['spectral']):
                    if crit == 'moment':
                        continue
                    rho_c = estimate_rho_c(df, crit)
                    parts.append(f"{crit.capitalize()}={rho_c:.4f}")
                print(f"    d={d:3d}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
