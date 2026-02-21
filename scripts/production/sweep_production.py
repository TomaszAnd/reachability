#!/usr/bin/env python3
"""
Production sweep script with automatic backend selection.

Uses JAX JIT for Krylov at d<=128 (12-97x speedup over NumPy),
falls back to NumPy for d>=256 (JAX autodiff unstable through long Lanczos).

Supports:
- Both CanonicalQudit and QubitGrid models
- Krylov-only mode for large dimensions (Spectral is O(d^3))
- Adaptive sampling for efficient resource use
- Checkpoint saving per (d, tau) combination

Usage:
    python scripts/production/sweep_production.py --model canonical --dims 16 32 64
    python scripts/production/sweep_production.py --model qubitgrid --dims 64 128 --krylov-only
    python scripts/production/sweep_production.py --model qubitgrid --dims 256 --krylov-only --n-hams 50
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from src.models import CanonicalQuditModel, QubitGridModel
from src.sampling import DensitySweep, SweepConfig, AdaptiveSweep, AdaptiveSweepConfig


def get_k_values(model_type: str, d: int) -> list:
    """Generate K values with dense sampling near expected transitions."""
    if model_type == 'canonical':
        d2 = d * d
        # Transition region depends on dimension
        rho_ranges = {
            8:   (0.02, 0.35),
            16:  (0.02, 0.25),
            32:  (0.01, 0.18),
            64:  (0.005, 0.10),
            128: (0.003, 0.06),
        }
        rho_min, rho_max = rho_ranges.get(d, (0.002, 0.05))
        rho_vals = np.concatenate([
            np.linspace(rho_min, rho_min + (rho_max - rho_min) * 0.3, 4),
            np.linspace(rho_min + (rho_max - rho_min) * 0.3,
                        rho_min + (rho_max - rho_min) * 0.8, 16),
            np.linspace(rho_min + (rho_max - rho_min) * 0.8, rho_max, 4),
        ])
        K_vals = sorted(set(max(2, int(round(rho * d2))) for rho in rho_vals))
        return [k for k in K_vals if k <= d2]

    elif model_type == 'qubitgrid':
        # QubitGrid has K ~ 3*n_qubits + couplings, much smaller than d^2
        # Use linear K spacing since K_max is small
        from src.models import LATTICE_CONFIGS
        if d not in LATTICE_CONFIGS:
            raise ValueError(f"No lattice config for d={d}. Available: {sorted(LATTICE_CONFIGS.keys())}")
        # Create model to get K_max
        model = QubitGridModel(dim=d, seed=0)
        K_max = model.K
        # Sample ~20 K values from 2 to K_max
        n_points = min(20, K_max - 1)
        K_vals = sorted(set([max(2, int(k)) for k in np.linspace(2, K_max, n_points)]))
        return K_vals

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def select_criteria(krylov_only: bool, d: int) -> list:
    """Select criteria based on mode and dimension."""
    if krylov_only:
        return ['krylov']
    if d >= 128:
        # Spectral is O(d^3) eigh — very slow at d>=128
        print(f"  Note: d={d}, Spectral will be slow (eigh dominates at 97%+ of time)")
    return ['moment', 'spectral', 'krylov']


def use_jax_krylov(d: int) -> bool:
    """Decide whether to use JAX for Krylov based on dimension."""
    if d > 128:
        return False  # Autodiff unstable through long Lanczos recurrence
    try:
        import jax
        return True
    except ImportError:
        return False


def create_jax_sweep(model, config, criteria):
    """Create a DensitySweep that uses JAX Krylov criterion.

    Monkey-patches the sweep's _evaluate_submodel to use JAX Krylov
    while keeping Moment/Spectral on NumPy.
    """
    from src.criteria_jax import KrylovCriterionJAX
    from src.criteria import MomentCriterion, SpectralCriterion, Verdict

    sweep = DensitySweep(model, config)

    # Override _evaluate_submodel to use JAX Krylov
    original_evaluate = sweep._evaluate_submodel

    def _jax_evaluate_submodel(sub_seed, K, phi, criteria_list):
        cfg = sweep.config
        d = sweep.model.dim
        sub_model = sweep.model.sample_submodel(K, seed=sub_seed)
        m = cfg.krylov_m if cfg.krylov_m is not None else d

        counts = {c: 0 for c in criteria_list}
        scores = {c: [] for c in criteria_list}

        for _ in range(cfg.n_targets):
            psi = sub_model.random_state()

            if 'moment' in criteria_list:
                mc = MomentCriterion(sub_model, phi, psi, tau=cfg.tau)
                result = mc.is_reachable()
                if result.verdict == Verdict.UNREACHABLE:
                    counts['moment'] += 1
                scores['moment'].append(result.score)

            if 'spectral' in criteria_list:
                sc = SpectralCriterion(sub_model, phi, psi, tau=cfg.tau)
                result = sc.is_reachable(
                    maxiter=cfg.maxiter, restarts=cfg.restarts,
                    method=cfg.method)
                if result.verdict == Verdict.UNREACHABLE:
                    counts['spectral'] += 1
                scores['spectral'].append(result.score)

            if 'krylov' in criteria_list:
                kc = KrylovCriterionJAX(sub_model, phi, psi, tau=cfg.tau, m=m)
                result = kc.is_reachable(
                    maxiter=cfg.maxiter, restarts=cfg.restarts,
                    method=cfg.method)
                if result.verdict == Verdict.UNREACHABLE:
                    counts['krylov'] += 1
                scores['krylov'].append(result.score)

        return counts, scores

    sweep._evaluate_submodel = _jax_evaluate_submodel
    return sweep


def run_sweep(args):
    output_dir = Path(__file__).parent.parent.parent / 'data' / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'jax' if any(use_jax_krylov(d) for d in args.dims) else 'numpy'

    print("=" * 70)
    print(f"PRODUCTION SWEEP — {args.model} — {timestamp}")
    print("=" * 70)
    print(f"Dimensions: {args.dims}")
    print(f"Tau: {args.tau}")
    print(f"Trials: {args.n_hams} hamiltonians x {args.n_targets} targets = {args.n_hams * args.n_targets}")
    if args.krylov_only:
        print("Mode: Krylov-only")
    if args.adaptive:
        print("Sampling: Adaptive")

    total_start = time.time()

    for d in args.dims:
        criteria = select_criteria(args.krylov_only, d)
        K_values = get_k_values(args.model, d)
        jax_mode = use_jax_krylov(d) and 'krylov' in criteria

        print(f"\n{'='*60}")
        print(f"d={d}: {len(K_values)} K values, criteria={criteria}")
        print(f"Backend: {'JAX' if jax_mode else 'NumPy'} Krylov")
        print(f"{'='*60}")

        # Create model
        ModelClass = CanonicalQuditModel if args.model == 'canonical' else QubitGridModel
        model = ModelClass(dim=d, seed=args.seed + d * 1000)

        # Check feasibility
        if K_values[-1] > model.K:
            K_values = [k for k in K_values if k <= model.K]
            print(f"  Trimmed to K_max={model.K}: {len(K_values)} K values")

        for tau in [args.tau]:
            print(f"\n  tau={tau}")
            dim_start = time.time()

            # Build config
            if args.adaptive:
                config = AdaptiveSweepConfig(
                    min_hamiltonians=max(10, args.n_hams // 5),
                    max_hamiltonians=args.n_hams,
                    n_targets=args.n_targets,
                    tau=tau,
                    maxiter=args.maxiter,
                    restarts=args.restarts,
                    target_sem=0.02,
                    batch_size=max(5, args.n_hams // 10),
                )
                if jax_mode:
                    sweep = create_jax_sweep(model, config, criteria)
                    # For adaptive, we need AdaptiveSweep behavior
                    # Re-create as AdaptiveSweep with JAX patching
                    adaptive_sweep = AdaptiveSweep(model, config)
                    adaptive_sweep._evaluate_submodel = sweep._evaluate_submodel
                    sweep = adaptive_sweep
                else:
                    sweep = AdaptiveSweep(model, config)
            else:
                config = SweepConfig(
                    n_hamiltonians=args.n_hams,
                    n_targets=args.n_targets,
                    tau=tau,
                    maxiter=args.maxiter,
                    restarts=args.restarts,
                )
                if jax_mode:
                    sweep = create_jax_sweep(model, config, criteria)
                else:
                    sweep = DensitySweep(model, config)

            df = sweep.run(
                K_values=K_values,
                criteria=criteria,
                verbose=True,
                early_stop_zeros=args.early_stop,
            )

            dim_time = time.time() - dim_start

            # Save checkpoint
            backend_tag = 'jax' if jax_mode else 'np'
            filename = f'{args.model}_d{d}_tau{tau}_{backend_tag}_{timestamp}.csv'
            filepath = output_dir / filename
            sweep.save(filepath)
            print(f"  Saved: {filepath}")
            print(f"  Time: {dim_time:.1f}s ({dim_time/60:.1f} min)")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"COMPLETED in {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} h)")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Production reachability sweep with automatic backend selection')
    parser.add_argument('--model', choices=['canonical', 'qubitgrid'], required=True,
                        help='Hamiltonian model family')
    parser.add_argument('--dims', type=int, nargs='+', required=True,
                        help='Dimensions to sweep (e.g., 16 32 64 128)')
    parser.add_argument('--tau', type=float, default=0.99,
                        help='Reachability threshold (default: 0.99)')
    parser.add_argument('--n-hams', type=int, default=100,
                        help='Number of Hamiltonian samples per K (default: 100)')
    parser.add_argument('--n-targets', type=int, default=10,
                        help='Number of target states per Hamiltonian (default: 10)')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='L-BFGS-B max iterations (default: 100)')
    parser.add_argument('--restarts', type=int, default=3,
                        help='Optimization restarts (default: 3)')
    parser.add_argument('--seed', type=int, default=100000,
                        help='Base seed (default: 100000)')
    parser.add_argument('--krylov-only', action='store_true',
                        help='Only run Krylov criterion (fast, recommended for d>=128)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive sampling (fewer trials in flat regions)')
    parser.add_argument('--early-stop', type=int, default=3,
                        help='Stop after N consecutive all-zero K values (default: 3, 0=disabled)')

    args = parser.parse_args()
    run_sweep(args)


if __name__ == '__main__':
    main()
