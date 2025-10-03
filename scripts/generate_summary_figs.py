"""
Generate all summary figures for reach package.

This script generates the complete set of summary figures according to the
specification, with multi-tau rank plots and comprehensive parameter sweeps.
"""

import logging
import sys
from pathlib import Path

import numpy as np

# Add reach package to path
sys.path.insert(0, str(Path(__file__).parent))

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

        # Compute old criterion once (tau-free)
        logger.info("  Computing old criterion (τ-free)...")
        old_results = analysis.old_criterion_probabilities(
            dims=dims,
            k_values=k_values,
            ensemble=ensemble,
            nks=rank_nks,
            nst=rank_nst,
            seed=settings.SEED,
        )

        # For each tau, compute new criterion and generate plot
        for tau in settings.RANK_TAUS_3:
            logger.info(f"  Computing new criterion for τ={tau}...")
            new_results = analysis.monte_carlo_unreachability(
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
                old_results=old_results,
                new_results=new_results,
                dims=dims,
                ensemble=ensemble,
                tau=tau,
                output_dir=settings.FIG_SUMMARY_DIR,
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
            data, ensemble=ensemble, output_dir=settings.FIG_SUMMARY_DIR
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
            data, ensemble=ensemble, output_dir=settings.FIG_SUMMARY_DIR
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
            data_dict, ensemble=ensemble, tau=tau, output_dir=settings.FIG_SUMMARY_DIR
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
            L1, L2, S, d=d, k=k, ensemble=ensemble, output_dir=settings.FIG_SUMMARY_DIR
        )
        logger.info(f"  ✓ {path_2d}")

        # Generate 3D
        path_3d = viz.plot_landscape_S3D(
            L1, L2, S, d=d, k=k, ensemble=ensemble, output_dir=settings.FIG_SUMMARY_DIR
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
            Sstar_by_d=data, ensemble=ensemble, bins=bins, output_dir=settings.FIG_SUMMARY_DIR
        )
        logger.info(f"  ✓ {path}")


def print_summary():
    """Print summary of generated files."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATED FILES SUMMARY")
    logger.info("=" * 60)


    fig_dir = Path(settings.FIG_SUMMARY_DIR)
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
    logger.info("REACH PACKAGE FIGURE GENERATION")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Rank dimensions: {settings.RANK_DIMS}")
    logger.info(f"  Rank taus: {settings.RANK_TAUS_3}")
    logger.info(f"  Big dimensions (5): {settings.BIG_DIMS_5}")
    logger.info(f"  Threshold taus (5): {settings.TAUS_5}")
    logger.info(f"  Iteration sweep: {settings.ITER_SWEEP_ITERS}")
    logger.info(f"  Sampling: nks={settings.BIG_NKS}, nst={settings.BIG_NST}")
    logger.info(f"  Output: {settings.FIG_SUMMARY_DIR}/")
    logger.info("=" * 60)

    try:
        # Generate all figure types
        generate_rank_comparisons_multi_tau()
        generate_optimizer_comparison()
        generate_tau_histograms()
        generate_iteration_sweeps()
        generate_overlap_hist_pdfs()
        generate_landscapes()

        # Print summary table
        print_summary()

        logger.info("=" * 60)
        logger.info("ALL FIGURES GENERATED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
