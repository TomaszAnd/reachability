#!/usr/bin/env python3
"""
Publication-quality overnight sweep v95.

Key design decisions:
  1. Universal 500 trials for all d (comparable statistics)
  2. K range covers all phase transitions (K=2 to 4*K_c)
  3. Krylov uses random search (validated 100% agreement at d>=32)
  4. Spectral uses L-BFGS-B (random search fails)
  5. Krylov skipped at d>=128 (runtime infeasible)
  6. QubitGrid d=128 Spectral: fundamental limitation noted (K_c > K_max)

Timing estimates:
  d=16:  ~1 min (both models)
  d=32:  ~30 min (both models)
  d=64:  ~5 hours (both models)
  d=128: ~17 hours (Canonical + QubitGrid, Spectral+Moment only)
  Total: ~23 hours

Output figures (ONLY these):
  1. P_vs_rho_{model}_d{d}.png -- All 3 criteria per plot (8 figures)
  2. rho_c_vs_inv_d.png -- Single plot, all models/criteria

Usage:
    python scripts/overnight_sweep_v95.py              # Full sweep (~23h)
    python scripts/overnight_sweep_v95.py --quick      # d=16 only (~2 min)
    python scripts/overnight_sweep_v95.py --no-128     # Skip d=128 (~6h)
    python scripts/overnight_sweep_v95.py --dims 16 32 # Custom dims
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

# Output directories
OUTPUT_DIR = Path("data/overnight_v95")
FIG_DIR = Path("fig/overnight_v95")

# Universal configuration
N_HAMILTONIANS = 50
N_TARGETS = 10
N_TRIALS = N_HAMILTONIANS * N_TARGETS  # = 500
TAU = 0.99
MAXITER = 50
RESTARTS = 3

# Spectral K_c estimates from regression: rho_c = a * d^b
# Canonical: a=1.003, b=-0.845
# QubitGrid: a=0.661, b=-1.000
SPECTRAL_KC = {
    'canonical': {16: 25, 32: 55, 64: 122, 128: 273},
    'qubitgrid': {16: 11, 32: 21, 64: 42, 128: 85},
}

# Plot styling — canonical hex colors matching src/plotting.py
CRIT_COLORS = {'moment': '#2ca02c', 'spectral': '#1f77b4', 'krylov': '#ff7f0e'}
CRIT_LABELS = {'moment': 'Moment', 'spectral': 'Spectral', 'krylov': 'Krylov'}
MODEL_LINESTYLES = {'canonical': '-', 'qubitgrid': '--'}
MODEL_MARKERS = {'canonical': 'o', 'qubitgrid': 's'}


def get_config() -> SweepConfig:
    """Universal configuration for all dimensions."""
    return SweepConfig(
        n_hamiltonians=N_HAMILTONIANS,
        n_targets=N_TARGETS,
        tau=TAU,
        maxiter=MAXITER,
        restarts=RESTARTS,
        krylov_method='random',
    )


def get_criteria(d: int) -> list:
    """Krylov skipped at d>=128 (runtime infeasible)."""
    if d >= 128:
        return ['moment', 'spectral']
    return ['moment', 'spectral', 'krylov']


def get_K_values_canonical(d: int, model) -> list:
    """Canonical K values: cover [2, 4*K_c] with ~12 points.

    - Start at K=2 for Moment/Krylov transitions
    - Extra density near Spectral K_c
    - Extend to 4*K_c to capture full transition
    """
    K_c = SPECTRAL_KC['canonical'][d]
    K_max = min(4 * K_c, model.K)

    # Low K for Moment/Krylov transition region
    K_low = [2, 3, 5, 8]

    # Spectral transition: dense sampling from 0.5*K_c to 2*K_c
    K_spectral = np.linspace(0.5 * K_c, 2 * K_c, 6).astype(int).tolist()

    # Above transition: extend to 4*K_c
    K_high = [int(3 * K_c), int(4 * K_c)]

    K_values = sorted(set(K_low + K_spectral + K_high))
    K_values = [k for k in K_values if 2 <= k <= K_max]
    return K_values


def get_K_values_qubitgrid(d: int, model) -> list:
    """QubitGrid K values: span [2, K_max] with ~12 points.

    QubitGrid has small K_max (48-81), so we cover the full range.
    Note: d=128 has K_c ~ 85 > K_max = 75, so Spectral transition
    cannot be captured (fundamental model limitation).
    """
    K_max = model.K
    K_c = SPECTRAL_KC['qubitgrid'][d]

    # Always include K=2,3,5 for Moment/Krylov transitions
    K_low = [2, 3, 5]

    # If K_c < K_max, add dense sampling around K_c
    if K_c < K_max:
        K_spectral = np.linspace(
            max(5, int(0.5 * K_c)),
            min(int(2 * K_c), K_max),
            6
        ).astype(int).tolist()
    else:
        # K_c > K_max: sample uniformly across available range
        K_spectral = np.linspace(10, K_max, 6).astype(int).tolist()

    # Fill to K_max
    K_high = [int(0.7 * K_max), K_max]

    K_values = sorted(set(K_low + K_spectral + K_high))
    K_values = [k for k in K_values if 2 <= k <= K_max]
    return K_values


def estimate_rho_c(df: pd.DataFrame, criterion: str) -> float:
    """Estimate rho_c via linear interpolation at P=0.5."""
    col = f'{criterion}_P'
    if col not in df.columns:
        return np.nan

    P = df[col].values
    rho = df['rho'].values

    if P.max() < 0.5 or P.min() > 0.5:
        return np.nan

    for i in range(len(P) - 1):
        if P[i] >= 0.5 >= P[i + 1] and P[i] != P[i + 1]:
            frac = (P[i] - 0.5) / (P[i] - P[i + 1])
            return rho[i] + frac * (rho[i + 1] - rho[i])

    return np.nan


def plot_P_vs_rho(df: pd.DataFrame, model: str, d: int, criteria: list,
                  save_path: Path) -> None:
    """Plot P vs rho with all criteria on same plot. No title."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for crit in criteria:
        col_P = f'{crit}_P'
        col_sem = f'{crit}_sem'
        if col_P not in df.columns:
            continue

        ax.errorbar(
            df['rho'], df[col_P], yerr=df[col_sem],
            fmt='o-', color=CRIT_COLORS[crit], label=CRIT_LABELS[crit],
            capsize=3, markersize=5, linewidth=1.5
        )

    ax.set_xlabel(r'$\rho = K/d^2$', fontsize=12)
    ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rho_c_vs_inv_d(results: dict, save_path: Path) -> None:
    """Single plot: rho_c vs 1/d for all models and criteria. No title."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for model in ['canonical', 'qubitgrid']:
        if model not in results:
            continue

        for crit in ['moment', 'spectral', 'krylov']:
            inv_d_vals = []
            rho_c_vals = []

            for d in sorted(results[model].keys()):
                df = results[model][d]
                rho_c = estimate_rho_c(df, crit)
                if np.isfinite(rho_c):
                    inv_d_vals.append(1.0 / d)
                    rho_c_vals.append(rho_c)

            if not inv_d_vals:
                continue

            label = f'{CRIT_LABELS[crit]} ({model.capitalize()})'
            ax.plot(
                inv_d_vals, rho_c_vals,
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                color=CRIT_COLORS[crit],
                label=label,
                markersize=7, linewidth=2
            )

    ax.set_xlabel(r'$1/d$', fontsize=12)
    ax.set_ylabel(r'$\rho_c$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_sweep(model, model_name: str, d: int, config: SweepConfig,
              K_values: list, criteria: list) -> pd.DataFrame:
    """Run sweep and return DataFrame."""
    print(f"  K values: {len(K_values)}, range [{K_values[0]}, {K_values[-1]}]")
    print(f"  Trials: {N_TRIALS} (={N_HAMILTONIANS}h x {N_TARGETS}t)")
    print(f"  Criteria: {', '.join(criteria)}")

    sweep = DensitySweep(model, config)
    df = sweep.run(K_values, criteria=criteria, early_stop_zeros=3)
    return df


def main():
    parser = argparse.ArgumentParser(description="Overnight sweep v95")
    parser.add_argument("--quick", action="store_true", help="d=16 only")
    parser.add_argument("--no-128", action="store_true", help="Skip d=128")
    parser.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--qubitgrid-only", action="store_true")
    args = parser.parse_args()

    dims = [16] if args.quick else args.dims
    if args.no_128:
        dims = [d for d in dims if d != 128]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    print(f"Starting v95 sweep at {start}")
    print(f"Dimensions: {dims}")
    print(f"Universal config: {N_TRIALS} trials, tau={TAU}")
    print(f"Krylov=random, Spectral=L-BFGS-B")
    print("=" * 60)

    timing_log = []
    results = {'canonical': {}, 'qubitgrid': {}}
    criteria_map = {'canonical': {}, 'qubitgrid': {}}

    config = get_config()

    # Canonical sweeps
    if not args.qubitgrid_only:
        print("\n" + "=" * 60)
        print("CANONICAL QUDIT MODEL")
        print("=" * 60)
        for d in dims:
            print(f"\n--- Canonical d={d} ---")
            model = CanonicalQuditModel(dim=d, seed=42)
            K_values = get_K_values_canonical(d, model)
            criteria = get_criteria(d)

            t0 = time.time()
            df = run_sweep(model, 'Canonical', d, config, K_values, criteria)
            runtime = time.time() - t0

            df.to_csv(OUTPUT_DIR / f"canonical_d{d}.csv", index=False)
            results['canonical'][d] = df
            criteria_map['canonical'][d] = criteria

            timing_log.append({
                'model': 'canonical', 'd': d, 'runtime_min': runtime / 60,
                'n_K': len(df), 'n_trials': N_TRIALS, 'criteria': '+'.join(criteria)
            })
            print(f"\n  Done: {len(df)} K values, {runtime/60:.1f} min")

            plot_P_vs_rho(df, 'canonical', d, criteria,
                          FIG_DIR / f"P_vs_rho_canonical_d{d}.png")
            print(f"  Saved: P_vs_rho_canonical_d{d}.png")

            # Save timing incrementally
            pd.DataFrame(timing_log).to_csv(
                OUTPUT_DIR / 'timing_log.csv', index=False)

    # QubitGrid sweeps
    if not args.canonical_only:
        print("\n" + "=" * 60)
        print("QUBITGRID MODEL")
        print("=" * 60)
        for d in dims:
            if d not in LATTICE_CONFIGS:
                print(f"\n--- QubitGrid d={d}: No lattice config, skipping ---")
                continue

            nx, ny = LATTICE_CONFIGS[d]
            print(f"\n--- QubitGrid d={d} ({nx}x{ny}) ---")
            model = QubitGridModel(dim=d, nx=nx, ny=ny, seed=42)

            K_c = SPECTRAL_KC['qubitgrid'][d]
            if K_c > model.K:
                print(f"  Note: Spectral K_c={K_c} > K_max={model.K}")
                print(f"     Spectral transition CANNOT be captured (model limitation)")

            K_values = get_K_values_qubitgrid(d, model)
            criteria = get_criteria(d)

            t0 = time.time()
            df = run_sweep(model, 'QubitGrid', d, config, K_values, criteria)
            runtime = time.time() - t0

            df.to_csv(OUTPUT_DIR / f"qubitgrid_d{d}.csv", index=False)
            results['qubitgrid'][d] = df
            criteria_map['qubitgrid'][d] = criteria

            timing_log.append({
                'model': 'qubitgrid', 'd': d, 'runtime_min': runtime / 60,
                'n_K': len(df), 'n_trials': N_TRIALS, 'criteria': '+'.join(criteria)
            })
            print(f"\n  Done: {len(df)} K values, {runtime/60:.1f} min")

            plot_P_vs_rho(df, 'qubitgrid', d, criteria,
                          FIG_DIR / f"P_vs_rho_qubitgrid_d{d}.png")
            print(f"  Saved: P_vs_rho_qubitgrid_d{d}.png")

            pd.DataFrame(timing_log).to_csv(
                OUTPUT_DIR / 'timing_log.csv', index=False)

    # Save final timing log
    timing_df = pd.DataFrame(timing_log)
    timing_df.to_csv(OUTPUT_DIR / 'timing_log.csv', index=False)

    # Generate rho_c vs 1/d plot
    print("\n" + "=" * 60)
    print("GENERATING rho_c vs 1/d PLOT")
    print("=" * 60)
    plot_rho_c_vs_inv_d(results, FIG_DIR / 'rho_c_vs_inv_d.png')
    print("Saved: rho_c_vs_inv_d.png")

    # Print phase transitions
    print("\n" + "=" * 60)
    print("PHASE TRANSITIONS (rho_c at P=0.5)")
    print("=" * 60)
    for model in ['canonical', 'qubitgrid']:
        if model not in results or not results[model]:
            continue
        print(f"\n{model.upper()}:")
        for d in sorted(results[model].keys()):
            df = results[model][d]
            parts = []
            for crit in criteria_map[model].get(d, []):
                rho_c = estimate_rho_c(df, crit)
                if np.isfinite(rho_c):
                    parts.append(f"{crit}={rho_c:.4f}")
                else:
                    parts.append(f"{crit}=--")
            print(f"  d={d:3d}: {', '.join(parts)}")

    # Final summary
    total_h = (datetime.now() - start).total_seconds() / 3600
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_h:.2f} hours")
    print(timing_df.to_string(index=False))

    # Write README
    readme_path = OUTPUT_DIR / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"# Overnight Sweep v95 Data\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Total runtime**: {total_h:.2f} hours\n")
        f.write(f"- **Trials per K**: {N_TRIALS} "
                f"(={N_HAMILTONIANS} hamiltonians x {N_TARGETS} targets)\n")
        f.write(f"- **tau**: {TAU}\n")
        f.write(f"- **Spectral optimizer**: L-BFGS-B "
                f"(maxiter={MAXITER}, restarts={RESTARTS})\n")
        f.write(f"- **Krylov optimizer**: Random search (50-75 samples)\n")
        f.write(f"- **Moment**: No optimization (lambda-independent)\n\n")
        f.write(f"## Dimensions\n{', '.join(f'd={d}' for d in dims)}\n\n")
        f.write(f"## Known Limitations\n")
        f.write(f"- QubitGrid d=128: Spectral transition K_c ~ 85 > K_max = 75\n")
        f.write(f"  (fundamental model limitation, transition cannot be captured)\n")
        f.write(f"- Krylov skipped at d>=128 (runtime infeasible)\n\n")
        f.write(f"## Files\n")
        f.write(f"- `canonical_d{{d}}.csv`: Canonical model data\n")
        f.write(f"- `qubitgrid_d{{d}}.csv`: QubitGrid model data\n")
        f.write(f"- `timing_log.csv`: Runtime per sweep\n")
        f.write(f"- `README.md`: This file\n\n")
        f.write(f"## Figures (in fig/overnight_v95/)\n")
        f.write(f"- `P_vs_rho_{{model}}_d{{d}}.png`: "
                f"P(unreachable) vs rho, all criteria\n")
        f.write(f"- `rho_c_vs_inv_d.png`: rho_c vs 1/d, all models and criteria\n")
    print(f"\nWrote: {readme_path}")


if __name__ == "__main__":
    main()
