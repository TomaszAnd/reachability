#!/usr/bin/env python3
"""
Publication-quality overnight sweep v96.

Key improvements over v95:
  1. Two-pass K selection: coarse log-spaced scan -> dense linear around transitions
  2. AdaptiveSweep with min_hamiltonians=10, target_sem=0.02
  3. Krylov uses random search at all dimensions, skipped at d>=128
  4. Publication-quality plots (REVTeX4-2 single column, Computer Modern fonts)
  5. Bootstrap rho_c confidence intervals
  6. Power-law rho_c vs 1/d fits

Output:
  data/overnight_v96/  — CSVs per (model, d)
  fig/overnight_v96/   — Publication figures

Usage:
    python scripts/production/overnight_sweep_v96.py              # Full (~24h)
    python scripts/production/overnight_sweep_v96.py --quick      # d=16 only
    python scripts/production/overnight_sweep_v96.py --no-256     # Skip d=256
    python scripts/production/overnight_sweep_v96.py --dims 16 32 # Custom dims
    python scripts/production/overnight_sweep_v96.py --timing     # Estimate only
    python scripts/production/overnight_sweep_v96.py --canonical-only
    python scripts/production/overnight_sweep_v96.py --qubitgrid-only
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import CanonicalQuditModel, QubitGridModel, LATTICE_CONFIGS
from src.sampling import (
    DensitySweep, SweepConfig, AdaptiveSweep, AdaptiveSweepConfig,
    estimate_rho_c, bootstrap_rho_c,
)
from src.plotting import CRIT_COLORS, CRIT_MARKERS, CRIT_LABELS
from src.math_utils import compute_binomial_sem

# Output directories
OUTPUT_DIR = Path("data/overnight_v96")
FIG_DIR = Path("fig/overnight_v96")

# Config
TAU = 0.99
MAXITER = 50
RESTARTS = 3

# Spectral K_c estimates (from v95 regression)
SPECTRAL_KC = {
    'canonical': {16: 25, 32: 55, 64: 122, 128: 273},
    'qubitgrid': {16: 11, 32: 21, 64: 42, 128: 85},
}

MODEL_LINESTYLES = {'canonical': '-', 'qubitgrid': '--'}


# ============================================================================
# Two-pass K selection
# ============================================================================

def two_pass_K_values(model, model_name, d, config):
    """Two-pass K selection: coarse scan -> dense around transitions.

    Pass 1: 8 log-spaced K values to find approximate transitions.
    Pass 2: 12 linearly-spaced K values around each criterion's transition.
    """
    K_max = model.K

    # Pass 1: coarse log-spaced scan
    K_min = 2
    K_coarse = sorted(set(
        [K_min] +
        [max(K_min, int(k)) for k in np.logspace(
            np.log10(K_min), np.log10(K_max), 8)] +
        [K_max]
    ))
    K_coarse = [k for k in K_coarse if K_min <= k <= K_max]

    print(f"  Pass 1: coarse scan with {len(K_coarse)} K values...")
    sweep = DensitySweep(model, SweepConfig(
        n_hamiltonians=10, n_targets=5, tau=TAU,
        maxiter=MAXITER, restarts=2, spectral_restarts=3,
        krylov_method='random',
    ))
    criteria = _get_criteria(d)
    df_coarse = sweep.run(K_coarse, criteria=criteria, verbose=False)

    if len(df_coarse) == 0:
        return K_coarse

    # Find transition regions for each criterion
    K_dense = set(K_coarse)
    for crit in criteria:
        col = f'{crit}_P'
        if col not in df_coarse.columns:
            continue
        P = df_coarse[col].values
        K_vals = df_coarse['K'].values

        # Find K range where 0.05 < P < 0.95
        trans_mask = (P > 0.05) & (P < 0.95)
        if trans_mask.any():
            K_lo = max(K_min, K_vals[trans_mask].min() - 2)
            K_hi = min(K_max, K_vals[trans_mask].max() + 2)
        else:
            # No transition found; add points around estimated K_c
            kc_est = SPECTRAL_KC.get(model_name, {}).get(d, K_max // 3)
            K_lo = max(K_min, int(0.5 * kc_est))
            K_hi = min(K_max, int(2 * kc_est))

        K_dense_crit = [max(K_min, int(k))
                        for k in np.linspace(K_lo, K_hi, 12)]
        K_dense.update(K_dense_crit)

    K_all = sorted(k for k in K_dense if K_min <= k <= K_max)
    print(f"  Pass 2: {len(K_all)} K values (added {len(K_all) - len(K_coarse)} around transitions)")
    return K_all


def _get_criteria(d, model_name=None):
    """Select criteria based on dimension and model.

    - Krylov skipped at d>=128 for Canonical (too slow with dense ops)
    - Spectral skipped at d>=256 (41s/trial, infeasible for overnight)
    - QubitGrid d=256 runs Moment+Krylov only (~11ms/trial Krylov)
    """
    if d >= 256:
        return ['moment', 'krylov']
    if d >= 128:
        return ['moment', 'spectral']
    return ['moment', 'spectral', 'krylov']


# ============================================================================
# Plotting
# ============================================================================

def _setup_latex():
    """Configure LaTeX fonts with fallback."""
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.latex.preamble': r'\usepackage{amsmath}',
        })
        # Test if LaTeX works
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.set_title(r'$\frac{1}{2}$')
        fig.savefig('/dev/null', format='png')
        plt.close(fig)
        return True
    except Exception:
        plt.rcParams.update({
            'text.usetex': False,
            'font.family': 'serif',
        })
        return False


def plot_P_vs_rho(df, model_name, d, criteria, save_path, use_latex=False):
    """Publication-quality P vs rho plot (REVTeX4-2 single column)."""
    fig, ax = plt.subplots(figsize=(3.375, 2.5))

    for crit in criteria:
        col_P = f'{crit}_P'
        col_sem = f'{crit}_sem'
        if col_P not in df.columns:
            continue

        ax.errorbar(
            df['rho'], df[col_P], yerr=df.get(col_sem, None),
            fmt=CRIT_MARKERS[crit] + '-',
            color=CRIT_COLORS[crit],
            label=CRIT_LABELS[crit],
            capsize=2, markersize=4, linewidth=1.2,
        )

    # Auto-trim x-axis
    max_rho = 0
    for crit in criteria:
        col_P = f'{crit}_P'
        if col_P in df.columns:
            mask = df[col_P] > 0.02
            if mask.any():
                max_rho = max(max_rho, df.loc[mask, 'rho'].max())
    if max_rho > 0:
        ax.set_xlim(0, 1.15 * max_rho)

    if use_latex:
        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel(r'$P(\mathrm{unreachable})$')
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([r'$0$', r'$\frac{1}{2}$', r'$1$'])
    else:
        ax.set_xlabel('rho = K/d^2')
        ax.set_ylabel('P(unreachable)')
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '1/2', '1'])

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rho_c_vs_inv_d(all_results, save_path, use_latex=False):
    """rho_c vs 1/d with bootstrap error bars and power-law fits."""
    fig, ax = plt.subplots(figsize=(3.375, 2.5))

    for model_name in ['canonical', 'qubitgrid']:
        if model_name not in all_results:
            continue

        for crit in ['moment', 'spectral', 'krylov']:
            inv_d_vals = []
            rho_c_vals = []
            err_lo = []
            err_hi = []

            for d in sorted(all_results[model_name].keys()):
                df = all_results[model_name][d]
                rc = estimate_rho_c(df, crit)
                if not np.isfinite(rc):
                    continue

                median, lo, hi = bootstrap_rho_c(df, crit)
                inv_d_vals.append(1.0 / d)
                rho_c_vals.append(rc)
                if np.isfinite(median):
                    err_lo.append(rc - lo)
                    err_hi.append(hi - rc)
                else:
                    err_lo.append(0)
                    err_hi.append(0)

            if not inv_d_vals:
                continue

            label = f'{CRIT_LABELS[crit]} ({model_name.capitalize()})'
            ax.errorbar(
                inv_d_vals, rho_c_vals,
                yerr=[err_lo, err_hi],
                fmt=CRIT_MARKERS[crit],
                linestyle=MODEL_LINESTYLES[model_name],
                color=CRIT_COLORS[crit],
                label=label,
                markersize=5, linewidth=1.2, capsize=2,
            )

            # Power-law fit if >= 3 points
            if len(inv_d_vals) >= 3:
                try:
                    x = np.array(inv_d_vals)
                    y = np.array(rho_c_vals)
                    log_x = np.log(x)
                    log_y = np.log(y)
                    b, log_a = np.polyfit(log_x, log_y, 1)
                    a = np.exp(log_a)
                    x_fit = np.linspace(min(x) * 0.8, max(x) * 1.2, 50)
                    y_fit = a * x_fit ** b
                    ax.plot(x_fit, y_fit,
                            linestyle=MODEL_LINESTYLES[model_name],
                            color=CRIT_COLORS[crit], alpha=0.3, linewidth=1)
                except Exception:
                    pass

    if use_latex:
        ax.set_xlabel(r'$1/d$')
        ax.set_ylabel(r'$\rho_c$')
    else:
        ax.set_xlabel('1/d')
        ax.set_ylabel('rho_c')

    # x-axis ticks as 2^{-n}
    d_vals = [16, 32, 64, 128, 256]
    tick_vals = [1.0 / d for d in d_vals]
    if use_latex:
        tick_labels = [rf'$2^{{-{int(np.log2(d))}}}$' for d in d_vals]
    else:
        tick_labels = [f'1/{d}' for d in d_vals]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=5, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main sweep
# ============================================================================

def run_model_sweep(model_cls, model_name, d, model_kwargs=None):
    """Run two-pass adaptive sweep for one (model, d) combination."""
    model_kwargs = model_kwargs or {}
    model = model_cls(dim=d, seed=42, **model_kwargs)

    criteria = _get_criteria(d)
    K_values = two_pass_K_values(model, model_name, d,
                                 AdaptiveSweepConfig())

    spectral_restarts = 5 if d <= 32 else 10
    config = AdaptiveSweepConfig(
        min_hamiltonians=10,
        max_hamiltonians=150,
        n_targets=10,
        tau=TAU,
        maxiter=MAXITER,
        restarts=RESTARTS,
        spectral_restarts=spectral_restarts,
        krylov_method='random',
        target_sem=0.02,
        batch_size=10,
    )

    sweep = AdaptiveSweep(model, config)
    df = sweep.run(K_values, criteria=criteria, early_stop_zeros=3)
    return df


def main():
    parser = argparse.ArgumentParser(description="Overnight sweep v96")
    parser.add_argument("--quick", action="store_true", help="d=16 only")
    parser.add_argument("--no-256", action="store_true", help="Skip d=256")
    parser.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--timing", action="store_true",
                        help="Print time estimates only, do not run")
    parser.add_argument("--canonical-only", action="store_true")
    parser.add_argument("--qubitgrid-only", action="store_true")
    args = parser.parse_args()

    dims = [16] if args.quick else args.dims
    # QubitGrid includes d=256 by default (Moment+Krylov only, ~1h)
    if args.no_256:
        qubitgrid_dims = [d for d in dims if d != 256]
    else:
        qubitgrid_dims = sorted(set(dims) | {256})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.timing:
        print("TIME ESTIMATES (approximate):")
        print(f"  d=16:   ~2 min (both models)")
        print(f"  d=32:   ~30 min (both models)")
        print(f"  d=64:   ~5 hours (both models)")
        print(f"  d=128:  ~17 hours (Spectral+Moment only)")
        print(f"  d=256:  QubitGrid only, ~1 hour (Moment+Krylov, Spectral infeasible)")
        return

    use_latex = _setup_latex()
    if use_latex:
        print("LaTeX fonts: enabled (Computer Modern)")
    else:
        print("LaTeX fonts: unavailable, using serif fallback")

    start = datetime.now()
    print(f"Starting v96 sweep at {start}")
    print(f"Canonical dims: {dims}")
    print(f"QubitGrid dims: {qubitgrid_dims}")
    print(f"Two-pass K selection, AdaptiveSweep, target_sem=0.02")
    print("=" * 60)

    all_results = {'canonical': {}, 'qubitgrid': {}}
    timing_log = []

    # Canonical sweeps
    if not args.qubitgrid_only:
        print("\n" + "=" * 60)
        print("CANONICAL QUDIT MODEL")
        print("=" * 60)
        for d in dims:
            if d >= 256:
                print(f"\n--- Canonical d={d}: SKIPPED (requires ~68GB RAM) ---")
                continue
            print(f"\n--- Canonical d={d} ---")
            t0 = time.time()
            df = run_model_sweep(CanonicalQuditModel, 'canonical', d)
            runtime = time.time() - t0

            df.to_csv(OUTPUT_DIR / f"canonical_d{d}.csv", index=False)
            all_results['canonical'][d] = df

            timing_log.append({
                'model': 'canonical', 'd': d, 'runtime_min': runtime / 60,
                'n_K': len(df),
            })
            print(f"  Done: {len(df)} K values, {runtime/60:.1f} min")

            plot_P_vs_rho(df, 'canonical', d, _get_criteria(d),
                          FIG_DIR / f"P_vs_rho_canonical_d{d}.png", use_latex)

    # QubitGrid sweeps
    if not args.canonical_only:
        print("\n" + "=" * 60)
        print("QUBITGRID MODEL")
        print("=" * 60)
        for d in qubitgrid_dims:
            if d not in LATTICE_CONFIGS:
                print(f"\n--- QubitGrid d={d}: No lattice config, skipping ---")
                continue
            nx, ny = LATTICE_CONFIGS[d]
            print(f"\n--- QubitGrid d={d} ({nx}x{ny}) ---")

            t0 = time.time()
            df = run_model_sweep(
                QubitGridModel, 'qubitgrid', d,
                model_kwargs={'nx': nx, 'ny': ny})
            runtime = time.time() - t0

            df.to_csv(OUTPUT_DIR / f"qubitgrid_d{d}.csv", index=False)
            all_results['qubitgrid'][d] = df

            timing_log.append({
                'model': 'qubitgrid', 'd': d, 'runtime_min': runtime / 60,
                'n_K': len(df),
            })
            print(f"  Done: {len(df)} K values, {runtime/60:.1f} min")

            plot_P_vs_rho(df, 'qubitgrid', d, _get_criteria(d),
                          FIG_DIR / f"P_vs_rho_qubitgrid_d{d}.png", use_latex)

    # Save timing log
    pd.DataFrame(timing_log).to_csv(OUTPUT_DIR / 'timing_log.csv', index=False)

    # rho_c vs 1/d plot
    print("\n" + "=" * 60)
    print("GENERATING rho_c vs 1/d PLOT")
    print("=" * 60)
    plot_rho_c_vs_inv_d(all_results, FIG_DIR / 'rho_c_vs_inv_d.png', use_latex)
    print("Saved: rho_c_vs_inv_d.png")

    # Phase transition summary and rho_c data export
    print("\n" + "=" * 60)
    print("PHASE TRANSITIONS (rho_c with 95% CI)")
    print("=" * 60)
    rho_c_rows = []
    for model_name in ['canonical', 'qubitgrid']:
        if not all_results.get(model_name):
            continue
        print(f"\n{model_name.upper()}:")
        for d in sorted(all_results[model_name].keys()):
            df = all_results[model_name][d]
            parts = []
            for crit in _get_criteria(d):
                rc = estimate_rho_c(df, crit)
                med, lo, hi = bootstrap_rho_c(df, crit)
                n_K = len(df)
                col_P = f'{crit}_P'
                n_trans = int(((df[col_P] > 0.1) & (df[col_P] < 0.9)).sum()) if col_P in df.columns else 0
                rho_c_rows.append({
                    'model': model_name, 'd': d, 'criterion': crit,
                    'rho_c': rc,
                    'rho_c_median': med, 'rho_c_lo': lo, 'rho_c_hi': hi,
                    'n_K_values': n_K, 'n_transitions': n_trans,
                })
                if np.isfinite(rc):
                    if np.isfinite(med):
                        parts.append(f"{crit}={rc:.4f} [{lo:.4f}, {hi:.4f}]")
                    else:
                        parts.append(f"{crit}={rc:.4f}")
                else:
                    parts.append(f"{crit}=--")
            print(f"  d={d:3d}: {', '.join(parts)}")

    # Save rho_c summary as CSV and JSON
    if rho_c_rows:
        import json
        df_rc = pd.DataFrame(rho_c_rows)
        df_rc.to_csv(OUTPUT_DIR / 'rho_c_summary.csv', index=False)
        # JSON with NaN -> null conversion
        rc_json = df_rc.where(df_rc.notna(), None).to_dict(orient='records')
        with open(OUTPUT_DIR / 'rho_c_summary.json', 'w') as f:
            json.dump(rc_json, f, indent=2, default=str)
        print(f"\nSaved: rho_c_summary.csv and rho_c_summary.json ({len(rho_c_rows)} entries)")

    total_h = (datetime.now() - start).total_seconds() / 3600
    print(f"\nTotal runtime: {total_h:.2f} hours")


if __name__ == "__main__":
    main()
