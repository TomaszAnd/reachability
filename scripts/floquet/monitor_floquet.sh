#!/bin/bash
# Monitor GEO2 Floquet experiment progress

echo "============================================================"
echo "GEO2 Floquet Experiment Monitor"
echo "============================================================"
echo ""

# Check if experiment is running
if ps aux | grep -v grep | grep "run_geo2_floquet.py" > /dev/null; then
    echo "✓ Experiment is RUNNING"
    echo ""

    # Show latest log entries
    echo "Latest progress (last 15 lines):"
    echo "------------------------------------------------------------"
    tail -15 logs/geo2_floquet_quick.log
    echo "------------------------------------------------------------"
    echo ""

    # Show process info
    echo "Process info:"
    ps aux | grep -v grep | grep "run_geo2_floquet.py" | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "TIME:", $10}'
    echo ""

    # Estimate progress
    if [ -f logs/geo2_floquet_quick.log ]; then
        n_completed=$(grep -c "Results:" logs/geo2_floquet_quick.log)
        total_points=5  # ρ ∈ [0.02, 0.10] with step 0.02 = 5 points
        echo "Progress: $n_completed / $total_points density points completed"

        if [ $n_completed -gt 0 ]; then
            percent=$((100 * n_completed / total_points))
            echo "  ~$percent% complete"
        fi
    fi

else
    echo "⚠️  Experiment is NOT running"
    echo ""

    # Check if it completed
    if grep -q "Experiment complete" logs/geo2_floquet_quick.log 2>/dev/null; then
        echo "✓ Experiment COMPLETED!"
        echo ""

        # Show final results
        echo "Final results:"
        echo "------------------------------------------------------------"
        grep "Results:" logs/geo2_floquet_quick.log | tail -5
        echo "------------------------------------------------------------"
        echo ""

        # Show output files
        echo "Output files:"
        ls -lh data/raw_logs/geo2_floquet_*.pkl 2>/dev/null || echo "  No output files found"

    else
        echo "Last log entries:"
        echo "------------------------------------------------------------"
        tail -10 logs/geo2_floquet_quick.log 2>/dev/null || echo "  No log file found"
        echo "------------------------------------------------------------"
    fi
fi

echo ""
echo "Commands:"
echo "  Watch live: tail -f logs/geo2_floquet_quick.log"
echo "  Check this: ./monitor_floquet.sh"
echo "  Plot when done: python3 scripts/plot_geo2_floquet.py data/raw_logs/geo2_floquet_*.pkl"
echo ""
