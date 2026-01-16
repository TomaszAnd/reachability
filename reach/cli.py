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
- audit-moment-criterion: Compare moment (τ-free) vs spectral (τ-based) criteria

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
  reach audit-moment-criterion --ensemble GUE --dims 3,4,6 --k-values 2,3,4 --fast
        """,
    )

    # Global options
    parser.add_argument(
        "--ensemble",
        choices=["GOE", "GUE", "GEO2"],
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

    # GEO2-specific lattice parameters
    parser.add_argument(
        "--nx", type=int, help="GEO2: Lattice width (number of sites in x direction)"
    )
    parser.add_argument(
        "--ny", type=int, help="GEO2: Lattice height (number of sites in y direction)"
    )
    parser.add_argument(
        "--periodic", action="store_true", help="GEO2: Use periodic boundary conditions"
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
    cmd_landscape.add_argument(
        "--no-smooth",
        action="store_true",
        help="Skip Gaussian smoothing and interpolation (raw grid)",
    )
    cmd_landscape.add_argument(
        "--oversample-axes",
        action="store_true",
        help="Compute values exactly along λ₁=0 and λ₂=0 (not yet implemented)",
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

    # compare-rank subcommand (rescaled, no inset)
    cmd_rank = subparsers.add_parser(
        "compare-rank", help="Generate moment vs spectral criterion comparison (rescaled, no inset)"
    )
    cmd_rank.add_argument(
        "--dims", type=str, default="6,8,10,12,14,16,18,20,24,30", help="Comma-separated dimensions (default: 6,8,10,12,14,16,18,20,24,30)"
    )
    cmd_rank.add_argument(
        "--kmax", type=int, default=14, help="Maximum number of Hamiltonians (default: 14)"
    )
    cmd_rank.add_argument(
        "--taus",
        type=str,
        required=True,
        help="Comma-separated tau values (e.g., '0.99,0.999')",
    )
    cmd_rank.add_argument(
        "--eps-floor",
        type=float,
        default=1e-9,
        help="Epsilon floor for log10 display (default: 1e-9)",
    )
    cmd_rank.add_argument(
        "--legend-loc",
        type=str,
        default="lower left",
        help="Legend location (default: 'lower left')",
    )
    cmd_rank.add_argument(
        "--hide-floored",
        action="store_true",
        default=True,
        help="Hide points with p <= eps_floor to avoid cliff artifacts (default: True)",
    )

    # audit-moment-criterion subcommand
    cmd_audit = subparsers.add_parser(
        "audit-moment-criterion", help="Compare moment vs spectral criterion probabilities"
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

    # single-d-vs-k subcommand
    cmd_single_d = subparsers.add_parser(
        "single-d-vs-k", help="Plot P(unreachability) vs K for a single fixed dimension"
    )
    cmd_single_d.add_argument(
        "--d", type=int, required=True, help="Hilbert space dimension (fixed)"
    )
    cmd_single_d.add_argument(
        "--k-max", type=int, required=True, help="Maximum K value (sweep from 1 to k-max)"
    )
    cmd_single_d.add_argument(
        "--tau",
        type=float,
        default=settings.DEFAULT_TAU,
        help=f"Unreachability threshold (default: {settings.DEFAULT_TAU})",
    )

    # rank-compare-zoom subcommand
    cmd_rank_zoom = subparsers.add_parser(
        "rank-compare-zoom", help="Generate moment vs spectral criterion comparison with zoomed inset"
    )
    cmd_rank_zoom.add_argument(
        "--dims", type=str, default="6,8,10,12", help="Comma-separated dimensions (default: 6,8,10,12)"
    )
    cmd_rank_zoom.add_argument(
        "--kmax", type=int, default=7, help="Maximum number of Hamiltonians (default: 7)"
    )
    cmd_rank_zoom.add_argument(
        "--taus",
        type=str,
        required=True,
        help="Comma-separated tau values (e.g., '0.99,0.999')",
    )
    cmd_rank_zoom.add_argument(
        "--yfloor",
        type=float,
        default=1e-8,
        help="Display floor for log scale (default: 1e-8)",
    )

    # three-criteria-vs-m subcommand
    cmd_3crit_m = subparsers.add_parser(
        "three-criteria-vs-m",
        help="Compare 3 criteria (spectral, old, Krylov) vs Krylov rank m",
    )
    cmd_3crit_m.add_argument(
        "--ensemble", choices=["GOE", "GUE"], required=True, help="Random matrix ensemble"
    )
    cmd_3crit_m.add_argument("-d", "--dim", type=int, required=True, help="Hilbert space dimension")
    cmd_3crit_m.add_argument(
        "-K", "--k", type=int, required=True, help="Number of Hamiltonians (fixed)"
    )
    cmd_3crit_m.add_argument(
        "--m-values",
        type=str,
        required=True,
        help="Comma-separated Krylov ranks (e.g., '1,2,3,4,5')",
    )
    cmd_3crit_m.add_argument(
        "--tau",
        type=float,
        default=settings.DEFAULT_TAU,
        help=f"Threshold for spectral (default: {settings.DEFAULT_TAU})",
    )
    cmd_3crit_m.add_argument(
        "--trials", type=int, default=150, help="Number of total trials (nks * nst, default: 150)"
    )
    cmd_3crit_m.add_argument(
        "--criteria",
        type=str,
        default="krylov,spectral,old",
        help="Comma-separated criteria (default: krylov,spectral,old)",
    )
    cmd_3crit_m.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file path to log results (one row per m per criterion)",
    )

    # three-criteria-vs-K subcommand
    cmd_3crit_K = subparsers.add_parser(
        "three-criteria-vs-K", help="Compare 3 criteria vs K (number of Hamiltonians)"
    )
    cmd_3crit_K.add_argument(
        "--ensemble", choices=["GOE", "GUE", "GEO2", "canonical"], required=True, help="Random matrix ensemble"
    )
    cmd_3crit_K.add_argument(
        "-d", "--dim", type=int, required=True, help="Hilbert space dimension (fixed)"
    )
    cmd_3crit_K.add_argument(
        "--k-values", type=str, required=True, help="Comma-separated K values (e.g., '1,2,3,4,5')"
    )
    cmd_3crit_K.add_argument(
        "--tau",
        type=float,
        default=settings.DEFAULT_TAU,
        help=f"Threshold for spectral (default: {settings.DEFAULT_TAU})",
    )
    cmd_3crit_K.add_argument(
        "--krylov-m",
        type=str,
        default="K",
        help="Krylov m strategy: 'K' (dynamic) or fixed int (default: 'K')",
    )
    cmd_3crit_K.add_argument(
        "--trials", type=int, default=150, help="Number of total trials (default: 150)"
    )
    cmd_3crit_K.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file path to log results (one row per K per criterion)",
    )

    # three-criteria-vs-density subcommand
    cmd_3crit_dens = subparsers.add_parser(
        "three-criteria-vs-density",
        help="Compare 3 criteria vs density (K/d²) for multiple dimensions",
    )
    cmd_3crit_dens.add_argument(
        "--ensemble", choices=["GOE", "GUE", "GEO2", "canonical"], required=True, help="Random matrix ensemble"
    )
    cmd_3crit_dens.add_argument(
        "--dims",
        type=str,
        required=True,
        help="Comma-separated dimensions (e.g., '20,30,40,50')",
    )
    cmd_3crit_dens.add_argument(
        "--rho-max",
        type=float,
        required=True,
        help="Maximum density value K/d² (e.g., 0.15)",
    )
    cmd_3crit_dens.add_argument(
        "--rho-step",
        type=float,
        required=True,
        help="Density step size (e.g., 0.01)",
    )
    cmd_3crit_dens.add_argument(
        "--taus",
        type=str,
        required=True,
        help="Comma-separated tau values for spectral (e.g., '0.90,0.95,0.99')",
    )
    cmd_3crit_dens.add_argument(
        "--trials", type=int, default=150, help="Number of total trials (default: 150)"
    )
    cmd_3crit_dens.add_argument(
        "--k-cap",
        type=int,
        default=200,
        help="Maximum K value cap (default: 200)",
    )
    cmd_3crit_dens.add_argument(
        "--y",
        type=str,
        choices=["reachable", "unreachable"],
        default="unreachable",
        help="Y-axis quantity: 'unreachable' (default) or 'reachable'",
    )
    cmd_3crit_dens.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file path to log results",
    )
    cmd_3crit_dens.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush CSV to disk every N data points (default: 10, enables streaming/resumable runs)",
    )
    cmd_3crit_dens.add_argument(
        "--log-data",
        action="store_true",
        default=False,
        help="Enable enhanced data logging (saves raw scores to pickle for post-hoc analysis)",
    )
    cmd_3crit_dens.add_argument(
        "--log-data-dir",
        type=str,
        default="data/raw_logs",
        help="Directory for enhanced data logs (default: data/raw_logs)",
    )

    # three-criteria-vs-K-multi-tau subcommand
    cmd_3crit_K_multitau = subparsers.add_parser(
        "three-criteria-vs-K-multi-tau",
        help="K-sweep with multiple τ for spectral (shows gradient)",
    )
    cmd_3crit_K_multitau.add_argument(
        "--ensemble", choices=["GOE", "GUE", "GEO2", "canonical"], required=True, help="Random matrix ensemble"
    )
    cmd_3crit_K_multitau.add_argument(
        "-d", "--dim", type=int, required=True, help="Hilbert space dimension (fixed)"
    )
    cmd_3crit_K_multitau.add_argument(
        "--k-max",
        type=int,
        required=True,
        help="Maximum K value (sweep from 2 to k-max)",
    )
    cmd_3crit_K_multitau.add_argument(
        "--taus",
        type=str,
        required=True,
        help="Comma-separated tau values for spectral (e.g., '0.90,0.95,0.99')",
    )
    cmd_3crit_K_multitau.add_argument(
        "--trials", type=int, default=300, help="Number of total trials (default: 300)"
    )
    cmd_3crit_K_multitau.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file path to log results",
    )
    cmd_3crit_K_multitau.add_argument(
        "--y",
        type=str,
        choices=["reachable", "unreachable"],
        default="unreachable",
        help="Plot P(reachable) or P(unreachable) (default: unreachable)",
    )
    cmd_3crit_K_multitau.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush CSV to disk every N data points (default: 10, enables streaming/resumable runs)",
    )

    # plot-from-csv subcommand
    cmd_plot_csv = subparsers.add_parser(
        "plot-from-csv",
        help="Generate plots from existing CSV files (useful for partial/incremental results)",
    )
    cmd_plot_csv.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (e.g., fig_summary/density_gue.csv)",
    )
    cmd_plot_csv.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["density", "k-multi-tau"],
        help="Plot type: 'density' or 'k-multi-tau'",
    )
    cmd_plot_csv.add_argument(
        "--ensemble",
        type=str,
        choices=["GOE", "GUE", "GEO2"],
        default="GUE",
        help="Random matrix ensemble (default: GUE)",
    )
    cmd_plot_csv.add_argument(
        "--y",
        type=str,
        choices=["unreachable", "reachable"],
        default="unreachable",
        help="Y-axis quantity (default: unreachable)",
    )
    cmd_plot_csv.add_argument(
        "--outdir",
        type=str,
        default="fig_summary/",
        help="Output directory for plots (default: fig_summary/)",
    )
    cmd_plot_csv.add_argument(
        "--taus",
        type=str,
        default=None,
        help="Optional: filter to specific tau values (comma-separated, e.g., '0.90,0.95,0.99')",
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


def validate_geo2_params(args) -> dict:
    """
    Validate and extract GEO2 lattice parameters from command-line args.

    For GEO2 ensemble:
    - If --dim is provided, it must be a power of 2; infer nx, ny if not given
    - If --nx and --ny are provided, validate dim = 2^(nx*ny)
    - Fail fast with actionable error messages

    Args:
        args: Parsed command-line arguments

    Returns:
        dict with 'nx', 'ny', 'periodic' for GEO2; empty dict for GOE/GUE

    Raises:
        ValueError: If GEO2 parameters are inconsistent or missing
    """
    if args.ensemble != "GEO2":
        return {}

    nx = args.nx
    ny = args.ny
    periodic = args.periodic

    # Case 1: Both nx and ny provided
    if nx is not None and ny is not None:
        n_sites = nx * ny
        expected_dim = 2 ** n_sites

        # If dim is also provided, validate consistency
        if hasattr(args, 'dim') and args.dim is not None:
            if args.dim != expected_dim:
                raise ValueError(
                    f"GEO2: Dimension mismatch. Lattice {nx}×{ny} = {n_sites} sites requires "
                    f"dimension 2^{n_sites} = {expected_dim}, but --dim={args.dim} was given."
                )

        return {"nx": nx, "ny": ny, "periodic": periodic}

    # Case 2: Only dim provided (for GEO2, try to infer square lattice)
    if hasattr(args, 'dim') and args.dim is not None:
        dim = args.dim

        # Check if dim is a power of 2
        if dim <= 0 or (dim & (dim - 1)) != 0:
            raise ValueError(
                f"GEO2: Dimension must be a power of 2 (got {dim}). "
                f"Please provide --nx and --ny explicitly (e.g., --nx 3 --ny 3 for d=64)."
            )

        # Infer number of qubits
        n_sites = int(np.log2(dim))

        # If nx, ny not provided, fail with helpful message
        if nx is None or ny is None:
            # Suggest square lattice if possible
            sqrt_n = int(np.sqrt(n_sites))
            if sqrt_n * sqrt_n == n_sites:
                raise ValueError(
                    f"GEO2: For dimension {dim} = 2^{n_sites}, please specify lattice shape. "
                    f"Suggestion: --nx {sqrt_n} --ny {sqrt_n} (square lattice)"
                )
            else:
                raise ValueError(
                    f"GEO2: For dimension {dim} = 2^{n_sites}, please specify lattice shape "
                    f"with --nx and --ny (e.g., --nx {n_sites} --ny 1 for a chain)"
                )

    # Case 3: Missing parameters
    raise ValueError(
        "GEO2 ensemble requires either (1) --nx and --ny, or (2) --dim with --nx and --ny. "
        "Example: --ensemble GEO2 --nx 3 --ny 3 --periodic"
    )


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
        no_smooth=args.no_smooth,
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
    """Execute compare-rank subcommand (rescaled, no inset)."""
    dims = parse_comma_separated(args.dims, int)
    taus = parse_comma_separated(args.taus, float)
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Rank comparison (rescaled): dims={dims}, taus={taus}, {args.ensemble}")

    # Compute moment criterion probabilities (τ-free, so compute once)
    logger.info("Computing moment criterion probabilities (τ-free)...")
    k_values = list(range(2, args.kmax + 1))
    moment_results = analysis.moment_criterion_probabilities(
        dims=dims,
        k_values=k_values,
        ensemble=args.ensemble,
        nks=nks // 4,
        nst=nst // 4,
        seed=args.seed,
    )

    # For each tau, compute new criterion and generate rescaled plot
    for tau in taus:
        logger.info(f"Processing tau={tau}...")

        # Compute spectral criterion probabilities for this tau
        logger.info(f"  Computing spectral criterion probabilities for τ={tau}...")
        spectral_results = {}
        for k in k_values:
            logger.info(f"    Processing k={k}")
            prob_data = analysis.probability_vs_tau(
                dims=dims,
                k=k,
                ensemble=args.ensemble,
                taus=np.array([tau]),
                nks_tau=nks // 4,
                nst_tau=nst // 4,
                method=settings.DEFAULT_METHOD,
                maxiter=settings.DEFAULT_MAXITER,
                seed=args.seed,
            )
            for d in dims:
                if k < d and d in prob_data:
                    spectral_results[(d, k)] = prob_data[d]["p"][0]

        # Generate rescaled plot (no inset)
        saved_path = viz.plot_rank_comparison_rescaled(
            moment_results=moment_results,
            spectral_results=spectral_results,
            dims=dims,
            ensemble=args.ensemble,
            tau=tau,
            output_dir=args.outdir,
            eps_floor=args.eps_floor,
            legend_loc=args.legend_loc,
            hide_floored=args.hide_floored,
        )

        print(f"Saved: {saved_path}")


def cmd_audit_moment_criterion(args) -> None:
    """Execute audit-moment-criterion subcommand."""
    dims = parse_comma_separated(args.dims, int)
    k_values = parse_comma_separated(args.k_values, int)
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Auditing moment criterion: dims={dims}, k_values={k_values}, {args.ensemble}")

    # Compute moment criterion probabilities
    logger.info("Computing moment criterion probabilities...")
    moment_results = analysis.moment_criterion_probabilities(
        dims=dims,
        k_values=k_values,
        ensemble=args.ensemble,
        nks=nks // 4,
        nst=nst // 4,
        seed=args.seed,  # Reduced sampling for speed
    )

    # Compute spectral criterion probabilities for comparison
    logger.info("Computing spectral criterion probabilities...")
    spectral_results = {}
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
                spectral_results[(d, k)] = prob_data[d]["p"][0]

    # Generate comparison plots
    saved_path = viz.plot_rank_comparison(
        moment_results=moment_results,
        spectral_results=spectral_results,
        dims=dims,
        ensemble=args.ensemble,
        tau=settings.DEFAULT_TAU,
        output_dir=args.outdir,
    )

    print(f"Saved: {saved_path}")

    # Print comparison summary
    print("\nMoment vs Spectral Criterion Comparison:")
    print("=" * 50)
    for d in sorted(dims):
        print(f"Dimension d={d}:")
        for k in sorted(k_values):
            if k >= d:
                continue
            moment_p = moment_results.get((d, k), 0.0)
            spectral_p = spectral_results.get((d, k), 0.0)
            ratio = spectral_p / moment_p if moment_p > 0 else float("inf")
            print(f"  k={k}: moment={moment_p:.4f}, spectral={spectral_p:.4f}, ratio={ratio:.2f}")


def cmd_single_d_vs_k(args) -> None:
    """Execute single-d-vs-k subcommand."""
    nks, nst = get_sampling_params(args.fast)
    ks = list(range(1, args.k_max + 1))

    logger.info(f"Single-d K sweep: d={args.d}, K=1..{args.k_max}, tau={args.tau}")

    # Compute probabilities vs K for fixed d
    data = analysis.probability_vs_k_single_d(
        d=args.d,
        ks=ks,
        ensemble=args.ensemble,
        tau=args.tau,
        nks=nks,
        nst=nst,
        method=settings.DEFAULT_METHOD,
        maxiter=settings.DEFAULT_MAXITER,
        seed=args.seed,
    )

    # Generate plot
    saved_path = viz.plot_unreach_vs_k_single_d(
        data=data,
        d=args.d,
        ensemble=args.ensemble,
        tau=args.tau,
        output_dir=args.outdir,
    )

    print(f"Saved: {saved_path}")


def cmd_rank_compare_zoom(args) -> None:
    """Execute rank-compare-zoom subcommand."""
    dims = parse_comma_separated(args.dims, int)
    taus = parse_comma_separated(args.taus, float)
    nks, nst = get_sampling_params(args.fast)

    logger.info(f"Rank comparison with zoom: dims={dims}, taus={taus}, {args.ensemble}")

    # Compute moment criterion probabilities (τ-free, so compute once)
    logger.info("Computing moment criterion probabilities (τ-free)...")
    k_values = list(range(2, args.kmax + 1))
    moment_results = analysis.moment_criterion_probabilities(
        dims=dims,
        k_values=k_values,
        ensemble=args.ensemble,
        nks=nks // 4,
        nst=nst // 4,
        seed=args.seed,
    )

    # For each tau, compute new criterion and generate plot with inset
    for tau in taus:
        logger.info(f"Processing tau={tau}...")

        # Compute spectral criterion probabilities for this tau
        logger.info(f"  Computing spectral criterion probabilities for τ={tau}...")
        spectral_results = {}
        for k in k_values:
            logger.info(f"    Processing k={k}")
            prob_data = analysis.probability_vs_tau(
                dims=dims,
                k=k,
                ensemble=args.ensemble,
                taus=np.array([tau]),
                nks_tau=nks // 4,
                nst_tau=nst // 4,
                method=settings.DEFAULT_METHOD,
                maxiter=settings.DEFAULT_MAXITER,
                seed=args.seed,
            )
            for d in dims:
                if k < d and d in prob_data:
                    spectral_results[(d, k)] = prob_data[d]["p"][0]

        # Generate plot with inset
        saved_path = viz.plot_rank_comparison_with_inset(
            moment_results=moment_results,
            spectral_results=spectral_results,
            dims=dims,
            ensemble=args.ensemble,
            tau=tau,
            output_dir=args.outdir,
            y_floor=args.yfloor,
        )

        print(f"Saved: {saved_path}")


def cmd_three_criteria_vs_m(args) -> None:
    """Execute three-criteria-vs-m subcommand."""
    from datetime import datetime
    from . import logging_utils

    m_vals = parse_comma_separated(args.m_values, int)
    crit_list = tuple(args.criteria.split(","))

    # Compute sampling from trials (split between nks and nst)
    # Use approximate sqrt split: nks * nst ≈ trials
    nks = int(np.sqrt(args.trials))
    nst = args.trials // nks

    logger.info(
        f"Three-criteria m-sweep: d={args.dim}, K={args.k}, {args.ensemble}, "
        f"m_values={m_vals}, tau={args.tau}, trials={args.trials} (nks={nks}, nst={nst})"
    )

    # Compute
    data = analysis.monte_carlo_unreachability_vs_m(
        d=args.dim,
        m_values=m_vals,
        K=args.k,
        ensemble=args.ensemble,
        criteria=crit_list,
        tau=args.tau,
        nks=nks,
        nst=nst,
        seed=args.seed,
    )

    # CSV logging (if requested)
    if args.csv:
        import uuid

        run_id = f"m_sweep_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        rows = []

        for criterion in crit_list:
            p_key = f"p_{criterion}"
            err_key = f"err_{criterion}"

            if p_key not in data:
                continue

            for i, m in enumerate(m_vals):
                p_unreach = float(data[p_key][i])
                log10_p = float(np.log10(max(p_unreach, settings.DISPLAY_FLOOR)))
                successes = int(p_unreach * args.trials)

                # For spectral: include overlap statistics
                if criterion == "spectral" and "mean_best_overlap_spectral" in data:
                    mean_overlap = float(data["mean_best_overlap_spectral"][i])
                    sem_overlap = float(data["sem_best_overlap_spectral"][i])
                else:
                    mean_overlap = ""
                    sem_overlap = ""

                row = {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "ensemble": args.ensemble,
                    "criterion": criterion,
                    "tau": args.tau if criterion == "spectral" else "",
                    "d": args.dim,
                    "K": args.k,
                    "m": m,
                    "rho_K_over_d2": args.k / (args.dim**2),
                    "trials": args.trials,
                    "successes_unreach": successes,
                    "p_unreach": p_unreach,
                    "log10_p_unreach": log10_p,
                    "mean_best_overlap": mean_overlap,
                    "sem_best_overlap": sem_overlap,
                }
                rows.append(row)

        logging_utils.append_rows_csv(args.csv, rows, logging_utils.REACHABILITY_CSV_FIELDS)
        logger.info(f"Logged {len(rows)} rows to CSV: {args.csv}")

    # Plot
    outdir = args.summary_dir if args.summary else args.outdir
    filepath = viz.plot_unreachability_three_criteria_vs_m(
        data=data,
        ensemble=args.ensemble,
        d=args.dim,
        K=args.k,
        tau=args.tau,
        outdir=outdir,
        trials=args.trials,
    )

    print(f"Saved: {filepath}")


def cmd_three_criteria_vs_K(args) -> None:
    """Execute three-criteria-vs-K subcommand."""
    from datetime import datetime
    from . import logging_utils

    k_vals = parse_comma_separated(args.k_values, int)

    # Parse Krylov m strategy
    if args.krylov_m == "K":
        strategy = "K"
        fixed = None
    else:
        strategy = "fixed"
        fixed = int(args.krylov_m)

    # Compute sampling
    nks = int(np.sqrt(args.trials))
    nst = args.trials // nks

    logger.info(
        f"Three-criteria K-sweep: d={args.dim}, {args.ensemble}, k_values={k_vals}, "
        f"tau={args.tau}, Krylov m={args.krylov_m}, trials={args.trials} (nks={nks}, nst={nst})"
    )

    # Compute
    data = analysis.monte_carlo_unreachability_vs_K_three(
        d=args.dim,
        k_values=k_vals,
        ensemble=args.ensemble,
        tau=args.tau,
        krylov_m_strategy=strategy,
        krylov_m_fixed=fixed,
        nks=nks,
        nst=nst,
        seed=args.seed,
    )

    # CSV logging (if requested)
    if args.csv:
        import uuid

        run_id = f"K_sweep_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()
        rows = []

        criteria = ["krylov", "spectral", "moment"]
        for criterion in criteria:
            p_key = f"p_{criterion}"
            if p_key not in data:
                continue

            for i, K in enumerate(k_vals):
                p_unreach = float(data[p_key][i])
                log10_p = float(np.log10(max(p_unreach, settings.DISPLAY_FLOOR)))
                successes = int(p_unreach * args.trials)

                # Determine m for Krylov (for CSV logging consistency)
                if strategy == "K":
                    m_value = K
                else:
                    m_value = fixed if fixed is not None else K
                m_value = max(1, min(m_value, args.dim))

                # For spectral: include overlap statistics
                if criterion == "spectral" and "mean_best_overlap_spectral" in data:
                    mean_overlap = float(data["mean_best_overlap_spectral"][i])
                    sem_overlap = float(data["sem_best_overlap_spectral"][i])
                else:
                    mean_overlap = ""
                    sem_overlap = ""

                row = {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "ensemble": args.ensemble,
                    "criterion": criterion,
                    "tau": args.tau if criterion == "spectral" else "",
                    "d": args.dim,
                    "K": K,
                    "m": m_value if criterion == "krylov" else "",
                    "rho_K_over_d2": K / (args.dim**2),
                    "trials": args.trials,
                    "successes_unreach": successes,
                    "p_unreach": p_unreach,
                    "log10_p_unreach": log10_p,
                    "mean_best_overlap": mean_overlap,
                    "sem_best_overlap": sem_overlap,
                }
                rows.append(row)

        logging_utils.append_rows_csv(args.csv, rows, logging_utils.REACHABILITY_CSV_FIELDS)
        logger.info(f"Logged {len(rows)} rows to CSV: {args.csv}")

    # Plot
    outdir = args.summary_dir if args.summary else args.outdir
    filepath = viz.plot_unreachability_three_criteria_vs_K(
        data=data,
        ensemble=args.ensemble,
        d=args.dim,
        tau=args.tau,
        outdir=outdir,
        trials=args.trials,
    )

    print(f"Saved: {filepath}")


def cmd_three_criteria_vs_density(args) -> None:
    """Execute three-criteria-vs-density subcommand."""
    from datetime import datetime
    from . import logging_utils

    dims = parse_comma_separated(args.dims, int)
    taus = parse_comma_separated(args.taus, float)

    # HARD ASSERTION: Density sweeps must use exact dims for GOE/GUE (publication standard)
    # For GEO2 and canonical, allow flexible dimensions
    if args.ensemble in ["GOE", "GUE"]:
        REQUIRED_DIMS = {20, 30, 40, 50}
        if set(dims) != REQUIRED_DIMS:
            raise ValueError(
                f"Density sweep for {args.ensemble} requires EXACTLY dims={sorted(REQUIRED_DIMS)}, "
                f"got dims={sorted(set(dims))}. This ensures publication-ready comparisons."
            )
    elif args.ensemble == "canonical":
        # Canonical basis: typical dims are 10,12,14 (basis size = d²)
        logger.info(f"Canonical ensemble: using dims={dims} (basis size = d² for each)")

    # Validate and extract ensemble parameters (for GEO2)
    ensemble_params = validate_geo2_params(args)

    # Compute sampling
    nks = int(np.sqrt(args.trials))
    nst = args.trials // nks

    logger.info(
        f"Three-criteria density sweep: ensemble={args.ensemble}, dims={dims}, "
        f"rho_max={args.rho_max}, rho_step={args.rho_step}, k_cap={args.k_cap}, "
        f"taus={taus}, trials={args.trials} (nks={nks}, nst={nst}), y={args.y}"
    )
    if ensemble_params:
        logger.info(f"  GEO2 lattice: nx={ensemble_params['nx']}, ny={ensemble_params['ny']}, periodic={ensemble_params['periodic']}")
    if args.log_data:
        logger.info(f"  Enhanced data logging enabled: {args.log_data_dir}")

    # Setup enhanced data logger if requested
    data_logger = None
    if args.log_data:
        metadata = {
            'rho_max': args.rho_max,
            'rho_step': args.rho_step,
            'k_cap': args.k_cap,
            'taus': taus,
            'nks': nks,
            'nst': nst,
            'seed': args.seed,
            'method': settings.DEFAULT_METHOD,
            'maxiter': settings.DEFAULT_MAXITER,
            **ensemble_params,
        }
        data_logger = logging_utils.EnhancedDataLogger(
            output_dir=args.log_data_dir,
            ensemble=args.ensemble,
            dims=dims,
            metadata=metadata,
            enable_logging=True,
        )

    # Compute
    try:
        data = analysis.monte_carlo_unreachability_vs_density(
            dims=dims,
            rho_max=args.rho_max,
            rho_step=args.rho_step,
            taus=taus,
            ensemble=args.ensemble,
            k_cap=args.k_cap,
            nks=nks,
            nst=nst,
            seed=args.seed,
            data_logger=data_logger,
            **ensemble_params,
        )
    finally:
        # Save enhanced data logger if it was used
        if data_logger is not None:
            data_logger.save()

    # CSV logging (if requested) - using streaming writer
    if args.csv:
        import uuid

        run_id = f"density_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        with logging_utils.StreamingCSVWriter(
            args.csv, flush_every=args.flush_every
        ) as csv_writer:
            row_count = 0
            for tau in taus:
                for criterion in ["spectral", "moment", "krylov"]:
                    for d in dims:
                        key = (d, tau, criterion)
                        if key not in data:
                            continue

                        result = data[key]
                        K_vals = result["K"]
                        rho_vals = result["rho"]
                        p_vals = result["p"]

                        for i, K in enumerate(K_vals):
                            p_unreach = float(p_vals[i])
                            log10_p = float(np.log10(max(p_unreach, settings.DISPLAY_FLOOR)))
                            successes = int(p_unreach * args.trials)

                            # For spectral: include overlap statistics
                            if criterion == "spectral" and "mean_overlap" in result:
                                mean_overlap = float(result["mean_overlap"][i])
                                sem_overlap = float(result["sem_overlap"][i])
                            else:
                                mean_overlap = ""
                                sem_overlap = ""

                            row = {
                                "run_id": run_id,
                                "timestamp": timestamp,
                                "ensemble": args.ensemble,
                                "criterion": criterion,
                                "tau": tau if criterion == "spectral" else "",
                                "d": d,
                                "K": int(K),
                                "m": "",  # Not applicable for density plots
                                "rho_K_over_d2": float(rho_vals[i]),
                                "trials": args.trials,
                                "successes_unreach": successes,
                                "p_unreach": p_unreach,
                                "log10_p_unreach": log10_p,
                                "mean_best_overlap": mean_overlap,
                                "sem_best_overlap": sem_overlap,
                            }
                            csv_writer.write_row(row)
                            row_count += 1

        logger.info(f"Logged {row_count} rows to CSV: {args.csv}")

    # Plot
    outdir = args.summary_dir if args.summary else args.outdir
    filepaths = viz.plot_unreachability_three_criteria_vs_density(
        data=data,
        ensemble=args.ensemble,
        outdir=outdir,
        trials=args.trials,
        y_axis=args.y,
    )

    for filepath in filepaths:
        print(f"Saved: {filepath}")


def cmd_three_criteria_vs_K_multi_tau(args) -> None:
    """Execute three-criteria-vs-K-multi-tau subcommand."""
    from datetime import datetime
    from . import logging_utils

    taus = parse_comma_separated(args.taus, float)

    # Validate and extract ensemble parameters (for GEO2)
    ensemble_params = validate_geo2_params(args)

    # Compute sampling
    nks = int(np.sqrt(args.trials))
    nst = args.trials // nks

    logger.info(
        f"K-sweep multi-tau: d={args.dim}, k_max={args.k_max}, {args.ensemble}, "
        f"taus={taus}, trials={args.trials} (nks={nks}, nst={nst}), y={args.y}"
    )
    if ensemble_params:
        logger.info(f"  GEO2 lattice: nx={ensemble_params['nx']}, ny={ensemble_params['ny']}, periodic={ensemble_params['periodic']}")

    # Compute
    data = analysis.monte_carlo_unreachability_vs_K_multi_tau(
        d=args.dim,
        k_max=args.k_max,
        taus=taus,
        ensemble=args.ensemble,
        nks=nks,
        nst=nst,
        seed=args.seed,
        **ensemble_params,
    )

    # CSV logging (if requested) - using streaming writer
    if args.csv:
        import uuid

        run_id = f"Kmulti_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.now().isoformat()

        with logging_utils.StreamingCSVWriter(
            args.csv, flush_every=args.flush_every
        ) as csv_writer:
            row_count = 0
            k_values = data["k"]

            # Log spectral for each tau
            for tau in taus:
                result = data[(tau, "spectral")]
                p_vals = result["p"]
                mean_overlaps = result["mean_overlap"]
                sem_overlaps = result["sem_overlap"]

                for i, K in enumerate(k_values):
                    p_unreach = float(p_vals[i])
                    log10_p = float(np.log10(max(p_unreach, settings.DISPLAY_FLOOR)))
                    successes = int(p_unreach * args.trials)

                    row = {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "ensemble": args.ensemble,
                        "criterion": "spectral",
                        "tau": tau,
                        "d": args.dim,
                        "K": int(K),
                        "m": "",
                        "rho_K_over_d2": K / (args.dim**2),
                        "trials": args.trials,
                        "successes_unreach": successes,
                        "p_unreach": p_unreach,
                        "log10_p_unreach": log10_p,
                        "mean_best_overlap": float(mean_overlaps[i]),
                        "sem_best_overlap": float(sem_overlaps[i]),
                    }
                    csv_writer.write_row(row)
                    row_count += 1

            # Log moment and krylov (tau-independent)
            for criterion in ["moment", "krylov"]:
                result = data[criterion]
                p_vals = result["p"]

                for i, K in enumerate(k_values):
                    p_unreach = float(p_vals[i])
                    log10_p = float(np.log10(max(p_unreach, settings.DISPLAY_FLOOR)))
                    successes = int(p_unreach * args.trials)

                    # Determine m for Krylov
                    m_value = min(int(K), args.dim) if criterion == "krylov" else ""

                    row = {
                        "run_id": run_id,
                        "timestamp": timestamp,
                        "ensemble": args.ensemble,
                        "criterion": criterion,
                        "tau": "",
                        "d": args.dim,
                        "K": int(K),
                        "m": m_value,
                        "rho_K_over_d2": K / (args.dim**2),
                        "trials": args.trials,
                        "successes_unreach": successes,
                        "p_unreach": p_unreach,
                        "log10_p_unreach": log10_p,
                        "mean_best_overlap": "",
                        "sem_best_overlap": "",
                    }
                    csv_writer.write_row(row)
                    row_count += 1

        logger.info(f"Logged {row_count} rows to CSV: {args.csv}")

    # Plot
    outdir = args.summary_dir if args.summary else args.outdir
    filepath = viz.plot_unreachability_K_multi_tau(
        data=data,
        ensemble=args.ensemble,
        outdir=outdir,
        trials=args.trials,
        y_type=args.y,
    )

    print(f"Saved: {filepath}")


def cmd_plot_from_csv(args) -> None:
    """Execute plot-from-csv subcommand."""
    logger.info(
        f"Plotting from CSV: {args.csv}, type={args.type}, "
        f"ensemble={args.ensemble}, y={args.y}"
    )

    # Parse optional tau filter
    taus = None
    if args.taus:
        taus = parse_comma_separated(args.taus, float)

    # Route to appropriate plotting function
    if args.type == "density":
        filepaths = viz.plot_density_from_csv(
            csv_path=args.csv,
            ensemble=args.ensemble,
            y_axis=args.y,
            outdir=args.outdir,
            taus=taus,
        )
    elif args.type == "k-multi-tau":
        filepaths = viz.plot_k_multi_tau_from_csv(
            csv_path=args.csv,
            ensemble=args.ensemble,
            y_type=args.y,
            outdir=args.outdir,
            taus=taus,
        )
    else:
        raise ValueError(f"Unknown plot type: {args.type}")

    for filepath in filepaths:
        print(f"Saved: {filepath}")


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
        "audit-moment-criterion": cmd_audit_moment_criterion,
        "single-d-vs-k": cmd_single_d_vs_k,
        "rank-compare-zoom": cmd_rank_compare_zoom,
        "three-criteria-vs-m": cmd_three_criteria_vs_m,
        "three-criteria-vs-K": cmd_three_criteria_vs_K,
        "three-criteria-vs-density": cmd_three_criteria_vs_density,
        "three-criteria-vs-K-multi-tau": cmd_three_criteria_vs_K_multi_tau,
        "plot-from-csv": cmd_plot_from_csv,
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
