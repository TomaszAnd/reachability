#!/bin/bash
# Monitor production sweep progress
# Usage: ./monitor_production.sh [interval_seconds]

INTERVAL=${1:-300}  # Default: check every 5 minutes (300 seconds)
PID_FILE="production.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Error: production.pid not found"
    echo "Is the production sweep running?"
    exit 1
fi

PID=$(cat "$PID_FILE")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 PRODUCTION SWEEP MONITOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PID: $PID"
echo "Check interval: $INTERVAL seconds"
echo "Press Ctrl+C to stop monitoring"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

while true; do
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Status Check — $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if process is still running
    if ps -p $PID > /dev/null 2>&1; then
        ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')
        echo "✓ Process running (PID=$PID, elapsed: $ELAPSED)"
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ PROCESS COMPLETED!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Show final results
        echo ""
        echo "Generated files:"
        echo ""
        echo "PNG files:"
        ls -lh fig_summary/three_criteria_vs_density_GUE*.png 2>/dev/null | wc -l | xargs echo "  Density plots:"
        ls -lh fig_summary/K_sweep_multi_tau_GUE*.png 2>/dev/null | wc -l | xargs echo "  K-sweep plots:"
        echo ""
        echo "CSV files:"
        [ -f fig_summary/density_gue.csv ] && echo "  density_gue.csv: $(wc -l < fig_summary/density_gue.csv) lines" || echo "  density_gue.csv: NOT FOUND"
        [ -f fig_summary/k30_gue.csv ] && echo "  k30_gue.csv: $(wc -l < fig_summary/k30_gue.csv) lines" || echo "  k30_gue.csv: NOT FOUND"
        echo ""
        echo "Run verification:"
        echo "  python validate_implementation.py"
        break
    fi

    # Recent log output
    echo ""
    echo "📄 Recent log (last 15 lines):"
    tail -n 15 production.log | sed 's/^/  /'

    # File progress
    echo ""
    echo "📁 File progress:"

    # CSV files
    if [ -f fig_summary/density_gue.csv ]; then
        LINES=$(wc -l < fig_summary/density_gue.csv)
        echo "  ✓ density_gue.csv: $LINES lines (~193 expected)"
    else
        echo "  ⏳ density_gue.csv: pending..."
    fi

    if [ -f fig_summary/k30_gue.csv ]; then
        LINES=$(wc -l < fig_summary/k30_gue.csv)
        echo "  ✓ k30_gue.csv: $LINES lines (~66 expected)"
    else
        echo "  ⏳ k30_gue.csv: pending..."
    fi

    # PNG files
    DENSITY_COUNT=$(ls -1 fig_summary/three_criteria_vs_density_GUE_tau*.png 2>/dev/null | wc -l | tr -d ' ')
    KSWEEP_COUNT=$(ls -1 fig_summary/K_sweep_multi_tau_GUE*.png 2>/dev/null | wc -l | tr -d ' ')
    TOTAL_PNG=$((DENSITY_COUNT + KSWEEP_COUNT))

    echo "  PNG files: $TOTAL_PNG / 8 (density: $DENSITY_COUNT/6, k-sweep: $KSWEEP_COUNT/2)"

    if [ $TOTAL_PNG -gt 0 ]; then
        echo ""
        echo "  Latest files:"
        ls -lt fig_summary/*.png 2>/dev/null | head -3 | awk '{print "    " $9 " (" $5 ")"}'
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⏳ Next check in $INTERVAL seconds..."
    echo "   (Press Ctrl+C to stop monitoring)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    sleep $INTERVAL
done
