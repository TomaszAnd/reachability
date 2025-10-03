"""
Generate overlap histogram PDF plots only.
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


def main():
    """Generate overlap histogram PDFs."""
    models.setup_environment(settings.SEED)

    logger.info("\n" + "=" * 60)
    logger.info("GENERATING OVERLAP HISTOGRAM PDFs")
    logger.info("=" * 60)

    dims = settings.BIG_DIMS_5  # [12,16,20,24,30]
    bins = np.arange(0.90, 1.00 + 1e-9, 0.01)
    ensembles = ["GOE", "GUE"]

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

    logger.info("\n" + "=" * 60)
    logger.info("GENERATED FILES:")
    logger.info("=" * 60)

    fig_dir = Path(settings.FIG_SUMMARY_DIR)
    for filename in ["overlap_hist_pdf_GOE.png", "overlap_hist_pdf_GUE.png"]:
        filepath = fig_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            logger.info(f"{filename:<50} {size_kb:>12.1f} KB")
        else:
            logger.warning(f"{filename} not found")


if __name__ == "__main__":
    main()
