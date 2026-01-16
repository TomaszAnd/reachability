#!/bin/bash
# Monitor production figure generation

echo "=========================================="
echo "Production Run Monitor"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "generate_three_criteria_comparison" > /dev/null; then
    echo "✓ Production run is ACTIVE"
    ps aux | grep -v grep | grep "generate_three_criteria_comparison" | awk '{print "  PID:", $2, "  CPU:", $3"%", "  MEM:", $4"%", "  TIME:", $10}'
else
    echo "✗ No production run detected"
fi

echo ""
echo "Generated figures:"
ls -lht fig/comparison/*.png 2>/dev/null | head -10 || echo "  No figures found yet"

echo ""
echo "Expected output:"
echo "  - unreachable_vs_k_over_d2_GUE_tau0.95.png"
echo "  - reachable_vs_k_over_d2_GUE_tau0.95.png"

echo ""
echo "To watch in real-time:"
echo "  watch -n 30 './monitor_production.sh'"
