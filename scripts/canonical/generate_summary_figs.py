"""
Generate all summary figures for reach package.

This script generates the complete set of summary figures according to the
specification, with multi-tau rank plots and comprehensive parameter sweeps.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add reach package to path (parent.parent gets us to the dir containing reach/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from reach import analysis, models, settings, viz

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_rank_comparisons_multi_tau(ensembles=["GOE", "GUE"]):
    """
    Generate rank comparison plots for multiple tau values (6 files total).

    Creates 3 tau plots × 2 ensembles = 6 files:
    - unreachability_vs_rank_old_vs_new_{GOE,GUE}_tau0.90.png
    - unreachability_vs_rank_old_vs_new_{GOE,GUE}_tau0.95.png
    - unreachability_vs_rank_old_vs_new_{GOE,GUE}_tau0.99.png
    """
    logger.info("=" * 60)
    logger.info("GENERATING RANK COMPARISONS (MULTI-TAU)")
    logger.info("=" * 60)

    dims = settings.RANK_DIMS
    k_values = list(range(2, 8))  # k = 2..7 for rank plots

    # Reduced sampling for rank plots (performance guardrail)
    rank_nks = settings.BIG_NKS // 2
    rank_nst = settings.BIG_NST

    logger.info(f"Using reduced sampling for rank plots: nks={rank_nks}, nst={rank_nst}")

    for ensemble in ensembles:
        logger.info(f"\n{ensemble} ensemble...")

        # Compute moment criterion once (tau-free)
        logger.info("  Computing moment criterion (τ-free)...")
        moment_results = analysis.moment_criterion_probabilities(
            dims=dims,
            k_values=k_values,
            ensemble=ensemble,
            nks=rank_nks,
            nst=rank_nst,
            seed=settings.SEED,
        )

        # For each tau, compute spectral criterion and generate plot
        for tau in settings.RANK_TAUS_3:
            logger.info(f"  Computing spectral criterion for τ={tau}...")
            spectral_results = analysis.monte_carlo_unreachability(
                dims=dims,
                ks=k_values,
                ensemble=ensemble,
                tau=tau,
                nks=rank_nks,
                nst=rank_nst,
                seed=settings.SEED,
            )

            # Generate plot
            path = viz.plot_rank_comparison(
                moment_results=moment_results,
                spectral_results=spectral_results,
                dims=dims,
                ensemble=ensemble,
                tau=tau,
                output_dir=settings.FIG_SPECTRAL_DIR,
            )
            logger.info(f"  ✓ {path}")


def generate_optimizer_comparison(ensembles=["GOE", "GUE"]):
    """Generate optimizer comparison figures (2 files)."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING OPTIMIZER COMPARISON")
    logger.info("=" * 60)

    dims = settings.BIG_DIMS_5
    methods = ["L-BFGS-B", "CG", "Powell"]
    k = 4  # Fixed k for optimizer comparison

    for ensemble in ensembles:
        logger.info(f"\n{ensemble} ensemble...")

        data = analysis.optimizer_Sstar_comparison(
            dims=dims,
            methods=methods,
            k=k,
            ensemble=ensemble,
            nks_opt=settings.BIG_NKS,
            nst_opt=settings.BIG_NST,
            seed=settings.SEED,
        )

        # Print verification table
        logger.info("\n  S* Statistics Table:")
        logger.info("  " + "-" * 50)
        for method in methods:
            for d in dims:
                if d in data[method]:
                    mean_S = data[method][d]["mean_S"]
                    sem_S = data[method][d]["sem_S"]
                    logger.info(f"  {method:12} d={d:2}: S* = {mean_S:.4f} ± {sem_S:.4f}")

        path = viz.plot_optimizer_comparison(
            data, ensemble=ensemble, output_dir=settings.FIG_SPECTRAL_DIR
        )
        logger.info(f"  ✓ {path}")


def generate_tau_histograms(ensembles=["GOE", "GUE"]):
    """Generate tau histogram figures (2 files)."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING TAU HISTOGRAMS (THRESHOLD SENSITIVITY)")
    logger.info("=" * 60)

    dims = settings.BIG_DIMS_5
    taus = np.array(settings.TAUS_5)
    k = 4  # Fixed k

    for ensemble in ensembles:
        logger.info(f"\n{ensemble} ensemble...")

        data = analysis.probability_vs_tau(
            dims=dims,
            taus=taus,
            k=k,
            ensemble=ensemble,
            nks_tau=settings.BIG_NKS,
            nst_tau=settings.BIG_NST,
            seed=settings.SEED,
        )

        paths = viz.plot_tau_histograms(
            data, ensemble=ensemble, output_dir=settings.FIG_SPECTRAL_DIR
        )

        for path in paths:
            logger.info(f"  ✓ {path}")


def generate_iteration_sweeps(ensembles=["GOE", "GUE"]):
    """Generate iteration sweep figures with 5 dimensions overlaid (2 files)."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING ITERATION SWEEPS (MULTI-D)")
    logger.info("=" * 60)

    dims = settings.BIG_DIMS_5
    k = 4  # Fixed k
    iters = tuple(settings.ITER_SWEEP_ITERS)
    tau = settings.DEFAULT_TAU

    for ensemble in ensembles:
        logger.info(f"\n{ensemble} ensemble...")

        data_dict = {}
        for d in dims:
            logger.info(f"  Computing d={d}...")
            data = analysis.probability_vs_iterations(
                d=d,
                k=k,
                ensemble=ensemble,
                iters=iters,
                tau=tau,
                nks_iter=settings.BIG_NKS,
                nst_iter=settings.BIG_NST,
                seed=settings.SEED,
            )
            data_dict[(d, k)] = data

        path = viz.plot_iteration_sweep_multi_dk(
            data_dict, ensemble=ensemble, tau=tau, output_dir=settings.FIG_SPECTRAL_DIR
        )
        logger.info(f"  ✓ {path}")


def generate_landscapes(ensembles=["GOE", "GUE"]):
    """Generate landscape figures 2D+3D (4 files)."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING LANDSCAPES")
    logger.info("=" * 60)

    d, k = 10, 3  # Fixed config
    grid = 41

    for ensemble in ensembles:
        logger.info(f"\n{ensemble} ensemble, d={d}, k={k}...")

        # Compute landscape
        L1, L2, S = analysis.landscape_spectral_overlap(
            d=d, k=k, ensemble=ensemble, grid=grid, n_targets=30, seed=settings.SEED
        )

        # Generate 2D
        path_2d = viz.plot_landscape_S2D(
            L1, L2, S, d=d, k=k, ensemble=ensemble, output_dir=settings.FIG_SPECTRAL_DIR
        )
        logger.info(f"  ✓ {path_2d}")

        # Generate 3D
        path_3d = viz.plot_landscape_S3D(
            L1, L2, S, d=d, k=k, ensemble=ensemble, output_dir=settings.FIG_SPECTRAL_DIR
        )
        logger.info(f"  ✓ {path_3d}")


def generate_overlap_hist_pdfs(ensembles=["GOE", "GUE"]):
    """Generate overlap histogram PDFs (2 files)."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING OVERLAP HISTOGRAM PDFs")
    logger.info("=" * 60)

    dims = settings.BIG_DIMS_5  # [12,16,20,24,30]
    bins = np.arange(0.90, 1.00 + 1e-9, 0.01)

    for ensemble in ensembles:
        logger.info(f"\n{ensemble} ensemble...")

        data = analysis.collect_Sstar_for_dims(
            dims=dims,
            ensemble=ensemble,
            k=4,
            method="L-BFGS-B",
            nks=settings.BIG_NKS,
            nst=settings.BIG_NST,
            maxiter=settings.DEFAULT_MAXITER,
            seed=settings.SEED,
        )

        path = viz.plot_overlap_hist_pdf(
            Sstar_by_d=data, ensemble=ensemble, bins=bins, output_dir=settings.FIG_SPECTRAL_DIR
        )
        logger.info(f"  ✓ {path}")


def generate_comparison_plots(ensembles=["GOE", "GUE"]):
    """
    Generate 3-criteria comparison plots (4 files total).

    Creates comparison plots with all dimensions overlaid for each ensemble:
    - 2 plots per ensemble × 2 ensembles = 4 files
    - Dimensions: GOE/GUE use [14,16,18,24], GEO2 uses [16,32,64]
    - All 3 criteria (spectral, moment, krylov) overlaid on each plot
    - Two x-axes: K/d² (density)
    - Two y-axes: reachability and unreachability

    Output files:
      - {reachability,unreachability}_vs_k_over_d2_{ensemble}_tau0.95.png (6 files)
      - TODO: K x-axis plots (6 files) - requires multi-d K-sweep infrastructure
    """
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING COMPARISON PLOTS (3 CRITERIA)")
    logger.info("=" * 60)

    # Dimensions per ensemble for Phase 3
    dims_map = {
        "GOE": [14, 16, 18, 24],
        "GUE": [14, 16, 18, 24],
        "GEO2": [16, 32, 64],
    }

    # Parameters
    tau = 0.95  # Fixed tau for comparison plots
    rho_max = 0.15  # Max density K/d²
    rho_step = 0.01
    trials_per_point = 50  # Reduced for faster testing (increase for production)

    for ensemble in ensembles:
        dims = dims_map[ensemble]
        logger.info(f"\n{ensemble} ensemble (dims={dims})...")

        # Run Monte Carlo for density sweep (K/d² x-axis)
        # This computes all 3 criteria for all dimensions
        logger.info(f"  Running Monte Carlo density sweep...")
        logger.info(f"    ρ_max={rho_max}, step={rho_step}, τ={tau}")
        logger.info(f"    Trials: {trials_per_point} per (d, ρ) point")

        data = analysis.monte_carlo_unreachability_vs_density(
            dims=dims,
            rho_max=rho_max,
            rho_step=rho_step,
            taus=[tau],  # Single tau for comparison plots
            ensemble=ensemble,
            nks=trials_per_point // 3,
            nst=3,
            seed=settings.SEED,
        )

        # Generate K/d² plots (2 plots per ensemble)
        logger.info(f"  Generating comparison plots (K/d² x-axis)...")

        # Plot 1: Unreachability vs K/d²
        paths = viz.plot_unreachability_three_criteria_vs_density(
            data=data,
            ensemble=ensemble,
            outdir=settings.FIG_COMPARISON_DIR,
            trials=trials_per_point,
            y_axis="unreachable",
        )
        for path in paths:
            logger.info(f"    ✓ {path}")

        # Plot 2: Reachability vs K/d²
        paths = viz.plot_unreachability_three_criteria_vs_density(
            data=data,
            ensemble=ensemble,
            outdir=settings.FIG_COMPARISON_DIR,
            trials=trials_per_point,
            y_axis="reachable",
        )
        for path in paths:
            logger.info(f"    ✓ {path}")

        #  TODO: K x-axis plots (reachability_vs_k and unreachability_vs_k)
        # These require a multi-dimension K-sweep function that doesn't exist yet.
        # For now, these can be generated via CLI:
        #   python -m reach.cli three-criteria-vs-K-multi-tau --ensemble {ensemble} ...
        logger.info(f"  ⚠️  K x-axis plots not yet implemented (use CLI for now)")


def print_summary():
    """Print summary of generated files."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATED FILES SUMMARY")
    logger.info("=" * 60)


    fig_dir = Path(settings.FIG_SPECTRAL_DIR)
    if not fig_dir.exists():
        logger.warning(f"Directory {fig_dir} not found")
        return

    files = sorted(fig_dir.glob("*.png"))

    if not files:
        logger.warning("No PNG files found")
        return

    logger.info(f"\n{'Filename':<60} {'Size (KB)':>12}")
    logger.info("-" * 72)

    for filepath in files:
        size_kb = filepath.stat().st_size / 1024
        logger.info(f"{filepath.name:<60} {size_kb:>12.1f}")

    logger.info("-" * 72)
    logger.info(f"Total: {len(files)} files\n")


def main():
    """Generate all figures."""
    models.setup_environment(settings.SEED)

    logger.info("\n" + "=" * 60)
    logger.info("REACH PACKAGE FIGURE GENERATION - PHASE 3")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Rank dimensions: {settings.RANK_DIMS}")
    logger.info(f"  Rank taus: {settings.RANK_TAUS_3}")
    logger.info(f"  Big dimensions (5): {settings.BIG_DIMS_5}")
    logger.info(f"  Threshold taus (5): {settings.TAUS_5}")
    logger.info(f"  Iteration sweep: {settings.ITER_SWEEP_ITERS}")
    logger.info(f"  Sampling: nks={settings.BIG_NKS}, nst={settings.BIG_NST}")
    logger.info(f"  Spectral output: {settings.FIG_SPECTRAL_DIR}/")
    logger.info(f"  Comparison output: {settings.FIG_COMPARISON_DIR}/")
    logger.info("=" * 60)

    try:
        # Create output directories
        Path(settings.FIG_SPECTRAL_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.FIG_COMPARISON_DIR).mkdir(parents=True, exist_ok=True)
        logger.info("✓ Output directories created")

        # Generate spectral criterion plots (12 files: 6 per ensemble × 2 ensembles)
        logger.info("\n[PHASE 3A: SPECTRAL CRITERION PLOTS]")
        generate_rank_comparisons_multi_tau()
        generate_optimizer_comparison()
        generate_tau_histograms()
        generate_iteration_sweeps()
        generate_overlap_hist_pdfs()
        generate_landscapes()

        # Generate 3-criteria comparison plots (4 files: 2 per ensemble × 2 ensembles)
        # NOTE: Only K/d² x-axis plots for now (K x-axis plots require additional infrastructure)
        # NOTE: GEO2 skipped - moment criterion doesn't support GEO2 yet
        logger.info("\n[PHASE 3B: COMPARISON PLOTS]")
        generate_comparison_plots()

        # Print summary table
        print_summary()

        logger.info("\n" + "=" * 60)
        logger.info("FIGURE GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Spectral plots: {settings.FIG_SPECTRAL_DIR}/")
        logger.info(f"✓ Comparison plots: {settings.FIG_COMPARISON_DIR}/")
        logger.info(f"  Total: ~16 files (12 spectral + 4 comparison) for GOE & GUE")
        logger.info(f"  NOTE: GEO2 skipped (moment criterion not yet implemented)")
        logger.info(f"  NOTE: K x-axis comparison plots require CLI generation")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
