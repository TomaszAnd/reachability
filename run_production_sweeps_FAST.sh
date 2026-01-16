#!/bin/bash
# FAST DEMONSTRATION VERSION - Production-ready reachability analysis
# This version uses reduced parameters for demonstration purposes
# For full publication quality, use run_production_sweeps.sh with higher trials
#
# Run time: ~5-10 minutes (vs 30-45 minutes for full version)

set -e  # Exit on error

echo "========================================="
echo "REACHABILITY ANALYSIS - FAST DEMO"
echo "========================================="
echo ""
echo "This FAST version generates:"
echo "  - 6 density plots (3 τ × 2 versions)"
echo "  - 2 K-sweep plots (d=30, 2 versions)"
echo "  - 2 CSV files with complete data"
echo ""
echo "Using REDUCED parameters for speed:"
echo "  - trials=25 (vs 150 for full)"
echo "  - rho_max=0.05 (vs 0.15 for full)"
echo ""
echo "Estimated time: 5-10 minutes"
echo "========================================="
echo ""

# Create output directory
mkdir -p fig/comparison

# Fast demonstration parameters
TRIALS_DENSITY=25   # Reduced from 150
RHO_MAX=0.05        # Reduced from 0.15
TRIALS_K=50         # Reduced from 300
FLUSH_EVERY=5       # Flush CSV every 5 points (more frequent for fast demo)

echo "[1/4] Running density sweep (unreachable) - FAST..."
echo "      d ∈ {20, 30, 40, 50}, ρ = 0-${RHO_MAX} step 0.01, τ ∈ {0.90, 0.95, 0.99}"
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE \
  --dims 20,30,40,50 \
  --rho-max $RHO_MAX \
  --rho-step 0.01 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_DENSITY \
  --y unreachable \
  --csv fig/comparison/density_gue_fast.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue_fast.csv \
  --type density \
  --ensemble GUE \
  --y unreachable

echo ""
echo "[2/4] Running density sweep (reachable) - FAST..."
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE \
  --dims 20,30,40,50 \
  --rho-max $RHO_MAX \
  --rho-step 0.01 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_DENSITY \
  --y reachable \
  --csv fig/comparison/density_gue_fast.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue_fast.csv \
  --type density \
  --ensemble GUE \
  --y reachable

echo ""
echo "[3/4] Running K-sweep for d=30 (unreachable) - FAST..."
echo "      d=30, K=2-14, τ ∈ {0.90, 0.95, 0.99}"
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE \
  -d 30 \
  --k-max 14 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_K \
  --y unreachable \
  --csv fig/comparison/k30_gue_fast.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/k30_gue_fast.csv \
  --type k-multi-tau \
  --ensemble GUE \
  --y unreachable

echo ""
echo "[4/4] Running K-sweep for d=30 (reachable) - FAST..."
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE \
  -d 30 \
  --k-max 14 \
  --taus 0.90,0.95,0.99 \
  --trials $TRIALS_K \
  --y reachable \
  --csv fig/comparison/k30_gue_fast.csv \
  --flush-every $FLUSH_EVERY

echo ""
echo "   [Refreshing plots from CSV...]"
python -m reach.cli plot-from-csv \
  --csv fig/comparison/k30_gue_fast.csv \
  --type k-multi-tau \
  --ensemble GUE \
  --y reachable

echo ""
echo "========================================="
echo "FAST DEMO COMPLETE!"
echo "========================================="
echo ""
echo "Generated files (fig/comparison/):"
echo ""
echo "Density plots (6 files):"
ls -1 fig/comparison/three_criteria_vs_density_GUE_tau*_{unreachable,reachable}.png 2>/dev/null | tail -6
echo ""
echo "K-sweep plots (2 files):"
ls -1 fig/comparison/K_sweep_multi_tau_GUE_d30*_{unreachable,reachable}.png 2>/dev/null | tail -2
echo ""
echo "CSV data (2 files):"
echo "  - density_gue_fast.csv"
echo "  - k30_gue_fast.csv"
echo ""
echo "⚠️  NOTE: This is a FAST DEMONSTRATION VERSION"
echo "For full publication quality, use:"
echo "  ./run_production_sweeps.sh"
echo "  (with trials=150 for density, trials=300 for K-sweep)"
echo "========================================="
