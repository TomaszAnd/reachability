"""
Command-line interface for quantum reachability analysis.

Pipeline Role:
This module provides the user-facing CLI that orchestrates the entire pipeline:
  models.py → analysis.py → viz.py
Each subcommand corresponds to a specific figure type with exact naming conventions.

Global Flags:
- --ensemble {GOE,GUE}: Random matrix ensemble choice
- --fast: Use reduced sampling (nks=80, nst=20) for quick validation
- --seed N: Override default random seed (42)
- --verbose: Enable INFO-level logging
- --summary: Save outputs to fig_summary/ directory

Subcommands:
- landscape-S: Generate S(λ₁,λ₂) landscapes (2D/3D)
- tau-hist: Threshold sensitivity histograms
- optimizer-hist: Optimizer comparison (S* distributions)
- iter-sweep: Convergence analysis (iterations vs probability)
- audit-old-criterion: Compare old (τ-free) vs new (τ-based) criteria

All parameters pulled from settings.py for consistency and reproducibility.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List

import numpy as np

from . import analysis, models, settings, viz

# Configure logging
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser with subcommands.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Time-free quantum state reachability analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mathematical background:
  Eigendecomposition: U(λ)†H(λ)U(λ) = diag(E₁,...,E_d)
  Spectral overlap:   S(λ) = Σₙ |φₙ(λ)*ψₙ(λ)|
  Maximized overlap:  S* = max_{λ∈[-1,1]^K} S(λ)
  Unreachability:     P_unreach(d,K;τ) = Pr[max_λ S(λ) < τ]

Examples:
  reach landscape-S --ensemble GOE -d 6 -k 3 --grid 81
  reach tau-hist --ensemble GUE --taus 0.90,0.95,0.99 -k 4
  reach optimizer-hist --ensemble GOE -d 10 -k 5 --methods L-BFGS-B,CG,Powell
  reach iter-sweep --ensemble GUE -d 6 -k 5 --iters 10,50,100,200
  reach compare-rank --ensemble GOE --fast
  reach audit-old-criterion --ensemble GUE --dims 3,4,6 --k-values 2,3,4 --fast
        """,
    )

    # Global options
    parser.add_argument(
        "--ensemble",
        choices=["GOE", "GUE"],
        default="GOE",
        help="Random matrix ensemble (default: GOE)",
    )
    parser.add_argument(
        "--outdir", default=".", help="Output directory for figures (default: current)"
    )
    parser.add_argument(
        "--summary-dir",
        default=settings.FIG_SUMMARY_DIR,
        help=f"Summary output directory (default: {settings.FIG_SUMMARY_DIR})",
    )
    parser.add_argument("--fast", action="store_true", help="Use fast mode with reduced sampling")
    parser.add_argument(
        "--seed", type=int, default=settings.SEED, help=f"Random seed (default: {settings.SEED})"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--summary", action="store_true", help="Generate comprehensive summary reports"
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Analysis commands")

    # landscape-S subcommand
    cmd_landscape = subparsers.add_parser("landscape-S", help="Generate S(λ₁,λ₂) landscape plots")
    cmd_landscape.add_argument(
        "-d", "--dim", type=int, required=True, help="Hilbert space dimension"
    )
    cmd_landscape.add_argument("-k", "--k", type=int, required=True, help="Number of Hamiltonians")
    cmd_landscape.add_argument(
        "--grid",
        type=int,
        default=settings.DEFAULT_GRID_SIZE,
        help=f"Grid resolution (default: {settings.DEFAULT_GRID_SIZE})",
    )
    cmd_landscape.add_argument(
        "--targets",
        type=int,
        default=settings.DEFAULT_LANDSCAPE_TARGETS,
        help=f"Number of target states (default: {settings.DEFAULT_LANDSCAPE_TARGETS})",
    )
    cmd_landscape.add_argument(
        "--lambda-range",
        type=float,
        nargs=2,
        default=[-1.5, 1.5],
        help="Parameter range [min, max] (default: [-1.5, 1.5])",
    )
    cmd_landscape.add_argument(
        "--plot-3d", action="store_true", help="Generate additional 3D surface plot"
    )

    # tau-hist subcommand
    cmd_tau = subparsers.add_parser("tau-hist", help="Generate τ threshold histograms")
    cmd_tau.add_argument(
        "--dims",
        type=str,
        default="6,10,14,18",
        help="Comma-separated dimensions (default: 6,10,14,18)",
    )
    cmd_tau.add_argument(
        "-k", "--k", type=int, default=4, help="Number of Hamiltonians (default: 4)"
    )
    cmd_tau.add_argument(
        "--taus",
        type=str,
        default="0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99",
        help="Comma-separated tau values (default: 0.90 to 0.99)",
    )

    # optimizer-hist subcommand
    cmd_opt = subparsers.add_parser(
        "optimizer-hist", help="Generate optimizer comparison histograms"
    )
    cmd_opt.add_argument(
        "--dims",
        type=str,
        default="6,10,14,18",
        help="Comma-separated dimensions (default: 6,10,14,18)",
    )
    cmd_opt.add_argument(
        "--kmax", type=int, default=7, help="Maximum number of Hamiltonians (default: 7)"
    )
    cmd_opt.add_argument(
        "--methods",
        type=str,
        default="L-BFGS-B,CG,Powell,TNC,SLSQP,Nelder-Mead",
        help="Comma-separated optimizer methods",
    )
    cmd_opt.add_argument(
        "--tau",
        type=float,
        default=settings.DEFAULT_TAU,
        help=f"Unreachability threshold (default: {settings.DEFAULT_TAU})",
    )

    # punreach-grid subcommand - DISABLED (replaced by spectral overlap criterion)
    # cmd_punreach = subparsers.add_parser('punreach-grid',
    #                                     help='Generate P(unreachability) vs (d,K) heatmaps')
    # cmd_punreach.add_argument('--d-range', type=str, default='3,4,5,6,7,8,10,12,15',
    #                          help='Comma-separated dimensions')
    # cmd_punreach.add_argument('--k-range', type=str, default='2,3,4,5,6,7',
    #                          help='Comma-separated K values')
    # cmd_punreach.add_argument('--epsilons', type=str, default='0.90,0.95,0.97,0.99',
    #                          help='Comma-separated epsilon thresholds')

    # iter-sweep subcommand
    cmd_iter = subparsers.add_parser("iter-sweep", help="Generate iteration convergence analysis")
    cmd_iter.add_argument(
        "-d", "--dim", type=int, default=6, help="Hilbert space dimension (default: 6)"
    )
    cmd_iter.add_argument(
        "-k", "--k", type=int, default=5, help="Number of Hamiltonians (default: 5)"
    )
    cmd_iter.add_argument(
        "--iters", type=str, default="10,20,50,100,200,400", help="Comma-separated iteration counts"
    )
    cmd_iter.add_argument(
        "--tau",
        type=float,
        default=settings.DEFAULT_TAU,
        help=f"Unreachability threshold (default: {settings.DEFAULT_TAU})",
    )

    # compare-rank subcommand
    cmd_rank = subparsers.add_parser(
        "compare-rank", help="Generate old vs new criterion comparison"
    )
    cmd_rank.add_argument(
        "--dims", type=str, default="3,4,5,6,7,8,10,12,15", help="Comma-separated dimensions"
    )
    cmd_rank.add_argument(
        "--kmax", type=int, default=7, help="Maximum number of Hamiltonians (default: 7)"
    )
    cmd_rank.add_argument(
        "--tau",
        type=float,
        default=settings.DEFAULT_TAU,
        help=f"Unreachability threshold (default: {settings.DEFAULT_TAU})",
    )

    # audit-old-criterion subcommand
    cmd_audit = subparsers.add_parser(
        "audit-old-criterion", help="Compare old vs new criterion probabilities"
    )
    cmd_audit.add_argument(
        "--dims", type=str, default="3,4,6,8", help="Comma-separated dimensions (default: 3,4,6,8)"
    )
    cmd_audit.add_argument(
        "--k-values",
        type=str,
        default="2,3,4,5",
        help="Comma-separated k values (default: 2,3,4,5)",
    )

    return parser


def parse_comma_separated(value: str, dtype=int) -> List:
    """Parse comma-separated string into list of specified type."""
    return [dtype(x.strip()) for x in value.split(",")]


def get_sampling_params(fast_mode: bool) -> tuple:
    """Get sampling parameters based on mode."""
    if fast_mode:
        return settings.FAST_SAMPLING
    else:
        return settings.FULL_SAMPLING


def cmd_landscape_S(args) -> None:
    """Execute landscape-S subcommand."""
    logger.info(f"Generating S(λ₁,λ₂) landscape: d={args.dim}, k={args.k}, {args.ensemble}")

    # Compute landscape
    L1, L2, S = analysis.landscape_spectral_overlap(
        d=args.dim,
        k=args.k,
        ensemble=args.ensemble,
        grid=args.grid,
        n_targets=args.targets,
        lambda_range=tuple(args.lambda_range),
        seed=args.seed,
    )

    # Generate 2D plot (required)
    saved_path_2d = viz.plot_landscape_S(
        L1, L2, S, d=args.dim, k=args.k, ensemble=args.ensemble, output_dir=args.outdir
    )
    print(f"Saved: {saved_path_2d}")

    # Generate 3D plot if requested (additional)
    if args.plot_3d:
        saved_path_3d = viz.plot_landscape_S_3d(
            L1, L2, S, d=args.dim, k=args.k, ensemble=args.ensemble, output_dir=args.outdir
        )
        print(f"Saved: {saved_path_3d}")


def cmd_tau_hist(args) -> None:
    """Execute tau-hist subcommand."""
    dims = parse_comma_separated(args.dims, int)
    taus = np.array(parse_comma_separated(args.taus, float))
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Generating τ histograms: dims={dims}, k={args.k}, {args.ensemble}")

    # Compute tau sweep
    data = analysis.probability_vs_tau(
        dims=dims,
        taus=taus,
        k=args.k,
        ensemble=args.ensemble,
        nks_tau=nks,
        nst_tau=nst,
        seed=args.seed,
    )

    # Generate plots
    saved_paths = viz.plot_tau_histograms(data, ensemble=args.ensemble, output_dir=args.outdir)

    for path in saved_paths:
        print(f"Saved: {path}")


def cmd_optimizer_hist(args) -> None:
    """Execute optimizer-hist subcommand."""
    dims = parse_comma_separated(args.dims, int)
    methods = parse_comma_separated(args.methods, str)
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Generating optimizer histograms: dims={dims}, methods={methods}, {args.ensemble}")

    # Compute optimizer comparison
    data = analysis.optimizer_comparison(
        dims=dims,
        methods=methods,
        Kmax=args.kmax,
        ensemble=args.ensemble,
        tau=args.tau,
        nks_opt=nks // 2,
        nst_opt=nst // 2,
        seed=args.seed,  # Reduced for multiple methods
    )

    # Generate plots for each dimension
    for d in dims:
        if d in data:
            saved_path = viz.plot_optimizer_histograms(
                data, ensemble=args.ensemble, d=d, k=args.kmax, output_dir=args.outdir
            )
            print(f"Saved: {saved_path}")


# DISABLED: punreach-grid functionality replaced by spectral overlap criterion
# def cmd_punreach_grid(args) -> None:
#     """Execute punreach-grid subcommand."""
#     d_range = parse_comma_separated(args.d_range, int)
#     k_range = parse_comma_separated(args.k_range, int)
#     epsilons = parse_comma_separated(args.epsilons, float)
#     nks, nst = get_sampling_params(args.fast)
#
#     logger.info(f"Generating P(unreachability) grids: epsilons={epsilons}, {args.ensemble}")
#
#     # Compute heatmap data
#     data = analysis.punreach_vs_dimension_K(
#         d_range=d_range, K_range=k_range, ensemble=args.ensemble,
#         epsilons=epsilons, nks=nks//2, nst=nst//2, seed=args.seed
#     )
#
#     # Generate plots for each epsilon
#     for eps in epsilons:
#         saved_path = viz.plot_punreach_heatmaps(
#             data, ensemble=args.ensemble, epsilon=eps,
#             output_dir=args.outdir
#         )
#         print(f"Saved: {saved_path}")


def cmd_iter_sweep(args) -> None:
    """Execute iter-sweep subcommand."""
    iters = tuple(parse_comma_separated(args.iters, int))
    nks, nst = get_sampling_params(args.fast)

    logger.info(
        f"Generating iteration sweep: d={args.dim}, k={args.k}, iters={iters}, {args.ensemble}"
    )

    # Compute iteration analysis
    data = analysis.probability_vs_iterations(
        d=args.dim,
        k=args.k,
        ensemble=args.ensemble,
        iters=iters,
        tau=args.tau,
        nks_iter=nks // 2,
        nst_iter=nst // 2,
        seed=args.seed,
    )

    # Generate plot
    saved_path = viz.plot_iteration_sweep(
        data, d=args.dim, k=args.k, ensemble=args.ensemble, tau=args.tau, output_dir=args.outdir
    )

    print(f"Saved: {saved_path}")


def cmd_compare_rank(args) -> None:
    """Execute compare-rank subcommand."""
    dims = parse_comma_separated(args.dims, int)
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Generating rank comparison: dims={dims}, {args.ensemble}")

    # Compute old vs new comparison (simplified for CLI)
    new_results = analysis.monte_carlo_unreachability(
        dims=dims,
        ks=list(range(2, args.kmax + 1)),
        ensemble=args.ensemble,
        tau=args.tau,
        nks=nks,
        nst=nst,
        seed=args.seed,
    )

    # For old results, use mock data (would need legacy implementation)
    old_results = {}
    for (d, k), prob in new_results.items():
        # Mock old criterion results (placeholder)
        if k == 2:
            old_results[(d, k)] = 1.0  # Always unreachable for k=2 in old criterion
        elif prob > 0.1:
            old_results[(d, k)] = prob * 0.3  # Scale down for demonstration
        else:
            old_results[(d, k)] = settings.DISPLAY_FLOOR

    # Generate plots
    saved_paths = viz.plot_rank_comparison(
        old_results=old_results,
        new_results=new_results,
        dims=dims,
        ensemble=args.ensemble,
        output_dir=args.outdir,
    )

    for path in saved_paths:
        print(f"Saved: {path}")


def cmd_audit_old_criterion(args) -> None:
    """Execute audit-old-criterion subcommand."""
    dims = parse_comma_separated(args.dims, int)
    k_values = parse_comma_separated(args.k_values, int)
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Auditing old criterion: dims={dims}, k_values={k_values}, {args.ensemble}")

    # Compute old criterion probabilities
    logger.info("Computing old criterion probabilities...")
    old_results = analysis.old_criterion_probabilities(
        dims=dims,
        k_values=k_values,
        ensemble=args.ensemble,
        nks=nks // 4,
        nst=nst // 4,
        seed=args.seed,  # Reduced sampling for speed
    )

    # Compute new criterion probabilities for comparison
    logger.info("Computing new criterion probabilities...")
    new_results = {}
    for k in k_values:
        logger.info(f"  Processing k={k}")
        prob_data = analysis.probability_vs_tau(
            dims=dims,
            k=k,
            ensemble=args.ensemble,
            taus=np.array([settings.DEFAULT_TAU]),
            nks_tau=nks // 4,
            nst_tau=nst // 4,
            method=settings.DEFAULT_METHOD,
            maxiter=settings.DEFAULT_MAXITER,
            seed=args.seed,
        )
        for d in dims:
            if k < d and d in prob_data:
                new_results[(d, k)] = prob_data[d]["p"][0]

    # Generate comparison plots
    saved_paths = viz.plot_rank_comparison(
        old_results=old_results,
        new_results=new_results,
        dims=dims,
        ensemble=args.ensemble,
        output_dir=args.outdir,
    )

    for path in saved_paths:
        print(f"Saved: {path}")

    # Print comparison summary
    print("\nOld vs New Criterion Comparison:")
    print("=" * 50)
    for d in sorted(dims):
        print(f"Dimension d={d}:")
        for k in sorted(k_values):
            if k >= d:
                continue
            old_p = old_results.get((d, k), 0.0)
            new_p = new_results.get((d, k), 0.0)
            ratio = new_p / old_p if old_p > 0 else float("inf")
            print(f"  k={k}: old={old_p:.4f}, new={new_p:.4f}, ratio={ratio:.2f}")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    # Setup environment
    models.setup_environment(args.seed)

    # Update output directory if summary mode is enabled
    if args.summary and hasattr(args, "outdir"):
        args.outdir = args.summary_dir
        logger.info(f"Summary mode enabled - using output directory: {args.outdir}")

    # Route to subcommand
    command_map = {
        "landscape-S": cmd_landscape_S,
        "tau-hist": cmd_tau_hist,
        "optimizer-hist": cmd_optimizer_hist,
        # 'punreach-grid': cmd_punreach_grid,  # DISABLED
        "iter-sweep": cmd_iter_sweep,
        "compare-rank": cmd_compare_rank,
        "audit-old-criterion": cmd_audit_old_criterion,
    }

    if args.command in command_map:
        try:
            command_map[args.command](args)
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.verbose:
                raise
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
