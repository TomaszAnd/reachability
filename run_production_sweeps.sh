#!/bin/bash
# Production-ready reachability analysis sweeps
# Run time: ~2-3 hours total for all sweeps
#
# This script generates publication-ready comparison plots for GUE ensembles
# with proper floor-aware rendering, CSV logging, streaming mode, and both P(unreachable) and P(reachable) versions.
#
# NEW: Streaming mode with --flush-every ensures CSV is written incrementally
# for resumable runs and mid-run plotting.

set -e  # Exit on error

echo "========================================="
echo "REACHABILITY ANALYSIS - PRODUCTION SWEEPS"
echo "========================================="
echo ""
echo "This will generate:"
echo "  - 6 density plots (3 τ × 2 versions)"
echo "  - 2 K-sweep plots (d=30, 2 versions)"
echo "  - 2 CSV files with complete data (streamed incrementally)"
echo ""
echo "Estimated time: 2-3 hours"
echo "========================================="
echo ""

# Create output directory
mkdir -p fig/comparison

# Configuration
TRIALS_DENSITY=150  # For density sweeps (4 dimensions × ~16 ρ values × 3 τ)
TRIALS_K=300        # For K-sweep (d=30, K=2-14, 3 τ)
FLUSH_EVERY=10      # Flush CSV every N points (enables streaming/resumable runs)

echo "[1/4] Running density sweep (unreachable)..."
echo "      d ∈ {20, 30, 40, 50}, ρ = 0-0.15 step 0.01, τ ∈ {0.90, 0.95, 0.99}"
echo "      This will take ~50-60 minutes..."
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE \
  --dims 20,30,40,50 \
  --rho-max 0.15 \
  --rho-step 0.01 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_DENSITY \
  --y unreachable \
  --csv fig/comparison/density_gue.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable

echo ""
echo "[2/4] Running density sweep (reachable)..."
echo "      Re-running with y=reachable for complementary view"
echo "      This will take ~50-60 minutes..."
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE \
  --dims 20,30,40,50 \
  --rho-max 0.15 \
  --rho-step 0.01 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_DENSITY \
  --y reachable \
  --csv fig/comparison/density_gue.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y reachable

echo ""
echo "[3/4] Running K-sweep for d=30 (unreachable)..."
echo "      d=30, K=2-14, τ ∈ {0.90, 0.95, 0.99}"
echo "      This will take ~10-15 minutes..."
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE \
  -d 30 \
  --k-max 14 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_K \
  --y unreachable \
  --csv fig/comparison/k30_gue.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/k30_gue.csv \
  --type k-multi-tau \
  --ensemble GUE \
  --y unreachable

echo ""
echo "[4/4] Running K-sweep for d=30 (reachable)..."
echo "      Re-running with y=reachable"
echo "      This will take ~10-15 minutes..."
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE \
  -d 30 \
  --k-max 14 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_K \
  --y reachable \
  --csv fig/comparison/k30_gue.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/k30_gue.csv \
  --type k-multi-tau \
  --ensemble GUE \
  --y reachable

echo ""
echo "========================================="
echo "PRODUCTION SWEEPS COMPLETE!"
echo "========================================="
echo ""
echo "Generated files (fig/comparison/):"
echo ""
echo "Density plots (6 files):"
echo "  - three_criteria_vs_density_GUE_tau0.90_unreachable.png"
echo "  - three_criteria_vs_density_GUE_tau0.90_reachable.png"
echo "  - three_criteria_vs_density_GUE_tau0.95_unreachable.png"
echo "  - three_criteria_vs_density_GUE_tau0.95_reachable.png"
echo "  - three_criteria_vs_density_GUE_tau0.99_unreachable.png"
echo "  - three_criteria_vs_density_GUE_tau0.99_reachable.png"
echo ""
echo "K-sweep plots (2 files):"
echo "  - K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_unreachable.png"
echo "  - K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_reachable.png"
echo ""
echo "CSV data (2 files):"
echo "  - density_gue.csv  (all density sweep data)"
echo "  - k30_gue.csv      (all K-sweep data)"
echo ""
echo "All plots are publication-ready:"
echo "  ✓ 14×10 inches, 200 DPI"
echo "  ✓ Floor-aware rendering (no vertical cliffs)"
echo "  ✓ Proper error bars (Wilson intervals)"
echo "  ✓ Enhanced typography"
echo ""
echo "Deliverables checklist:"
echo "  ✓ 6 density figures (3 τ × {unreachable, reachable})"
echo "  ✓ 2 K-sweep d=30 figures ({unreachable, reachable})"
echo "  ✓ 2 CSVs with complete schema"
echo "  ✓ Only dimensions {20,30,40,50} on density plots"
echo "  ✓ No vertical cliff segments"
echo "  ✓ Krylov uses m = min(K, d)"
echo "========================================="
