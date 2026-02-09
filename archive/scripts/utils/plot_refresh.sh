#!/bin/bash
# Refresh plots from existing CSV files (without recomputing)
#
# Usage:
#   ./scripts/plot_refresh.sh                    # Refresh all
#   ./scripts/plot_refresh.sh density            # Refresh density plots only
#   ./scripts/plot_refresh.sh k-multi-tau        # Refresh K-sweep plots only
#
# This is useful while a long run is in progress - you can peek at
# partial results by re-rendering plots from the growing CSV.

set -e

PLOT_TYPE=${1:-all}

echo "========================================="
echo "PLOT REFRESH FROM CSV"
echo "========================================="
echo ""

# Check for CSV files
DENSITY_CSV="fig/comparison/density_gue.csv"
K_CSV="fig/comparison/k30_gue.csv"

if [ "$PLOT_TYPE" = "all" ] || [ "$PLOT_TYPE" = "density" ]; then
    if [ -f "$DENSITY_CSV" ]; then
        echo "Refreshing density plots from $DENSITY_CSV..."
        echo ""

        # Unreachable plots
        echo "  [1/2] Generating unreachable density plots..."
        python -m reach.cli plot-from-csv \
            --csv $DENSITY_CSV \
            --type density \
            --ensemble GUE \
            --y unreachable

        echo ""

        # Reachable plots
        echo "  [2/2] Generating reachable density plots..."
        python -m reach.cli plot-from-csv \
            --csv $DENSITY_CSV \
            --type density \
            --ensemble GUE \
            --y reachable

        echo ""
    else
        echo "⚠️  Density CSV not found: $DENSITY_CSV"
        echo "   Run a density sweep first, or specify the correct CSV path."
        echo ""
    fi
fi

if [ "$PLOT_TYPE" = "all" ] || [ "$PLOT_TYPE" = "k-multi-tau" ]; then
    if [ -f "$K_CSV" ]; then
        echo "Refreshing K-sweep plots from $K_CSV..."
        echo ""

        # Unreachable plots
        echo "  [1/2] Generating unreachable K-sweep plot..."
        python -m reach.cli plot-from-csv \
            --csv $K_CSV \
            --type k-multi-tau \
            --ensemble GUE \
            --y unreachable

        echo ""

        # Reachable plots
        echo "  [2/2] Generating reachable K-sweep plot..."
        python -m reach.cli plot-from-csv \
            --csv $K_CSV \
            --type k-multi-tau \
            --ensemble GUE \
            --y reachable

        echo ""
    else
        echo "⚠️  K-sweep CSV not found: $K_CSV"
        echo "   Run a K-sweep first, or specify the correct CSV path."
        echo ""
    fi
fi

echo "========================================="
echo "PLOT REFRESH COMPLETE"
echo "========================================="
echo ""
echo "Generated plots are in fig/comparison/"
echo ""
echo "💡 TIP: Run this script anytime while a long sweep is running"
echo "   to see partial results from the growing CSV file."
echo ""
