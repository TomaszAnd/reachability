#!/bin/bash
# Progress monitoring for GEO2 comprehensive experiments
# Usage: ./scripts/check_geo2_progress.sh

echo "========================================================================"
echo "GEO2 EXPERIMENT PROGRESS CHECK - $(date)"
echo "========================================================================"
echo ""

# Check if PIDs exist
if [ -f .geo2_fixed_pid ]; then
    FIXED_PID=$(cat .geo2_fixed_pid)
    echo "Fixed Weights Experiment:"
    echo "  PID: $FIXED_PID"
    if ps -p $FIXED_PID > /dev/null 2>&1; then
        echo "  Status: RUNNING"
    else
        echo "  Status: COMPLETED or FAILED"
    fi
else
    echo "Fixed Weights Experiment: NOT STARTED (no .geo2_fixed_pid)"
fi
echo ""

if [ -f .geo2_optimized_pid ]; then
    OPT_PID=$(cat .geo2_optimized_pid)
    echo "Optimized Weights Experiment:"
    echo "  PID: $OPT_PID"
    if ps -p $OPT_PID > /dev/null 2>&1; then
        echo "  Status: RUNNING"
    else
        echo "  Status: COMPLETED or FAILED"
    fi
else
    echo "Optimized Weights Experiment: NOT STARTED (no .geo2_optimized_pid)"
fi
echo ""

echo "------------------------------------------------------------------------"
echo "LOG FILE TAILS (last 15 lines each)"
echo "------------------------------------------------------------------------"
echo ""

# Fixed weights log
FIXED_LOG=$(ls -t logs/geo2_fixed_weights_*.log 2>/dev/null | head -1)
if [ -f "$FIXED_LOG" ]; then
    echo "=== Fixed Weights Log: $FIXED_LOG ==="
    tail -15 "$FIXED_LOG"
    echo ""
else
    echo "=== No fixed weights log found ==="
    echo ""
fi

# Optimized weights log
OPT_LOG=$(ls -t logs/geo2_optimized_weights_*.log 2>/dev/null | head -1)
if [ -f "$OPT_LOG" ]; then
    echo "=== Optimized Weights Log: $OPT_LOG ==="
    tail -15 "$OPT_LOG"
    echo ""
else
    echo "=== No optimized weights log found ==="
    echo ""
fi

echo "------------------------------------------------------------------------"
echo "OUTPUT FILES"
echo "------------------------------------------------------------------------"
echo ""

# Check pickle files
FIXED_PKL=$(ls -t data/raw_logs/geo2_fixed_weights_*.pkl 2>/dev/null | head -1)
OPT_PKL=$(ls -t data/raw_logs/geo2_optimized_weights_*.pkl 2>/dev/null | head -1)

if [ -f "$FIXED_PKL" ]; then
    echo "Fixed Weights Data:"
    ls -lh "$FIXED_PKL"
    echo ""
else
    echo "Fixed Weights Data: NOT FOUND"
    echo ""
fi

if [ -f "$OPT_PKL" ]; then
    echo "Optimized Weights Data:"
    ls -lh "$OPT_PKL"
    echo ""
else
    echo "Optimized Weights Data: NOT FOUND"
    echo ""
fi

echo "------------------------------------------------------------------------"
echo "DISK USAGE"
echo "------------------------------------------------------------------------"
echo ""

du -sh logs/ 2>/dev/null || echo "No logs directory"
du -sh data/raw_logs/ 2>/dev/null || echo "No data/raw_logs directory"

echo ""
echo "========================================================================"
echo "Progress check complete at $(date)"
echo "========================================================================"
