#!/bin/bash
echo "Detailed Progress Check"
echo "======================"
echo ""
echo "Process Status:"
ps aux | grep -v grep | grep "generate_three_criteria" | awk '{print "  PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%, Runtime: " $10}'
echo ""
echo "System Load:"
uptime
echo ""
echo "Python Memory Usage:"
ps aux | grep python | grep -v grep | head -3 | awk '{print "  " $3 "% CPU, " $4 "% MEM, " $11}'
echo ""
echo "Latest 5 figures:"
ls -lt fig/comparison/*.png 2>/dev/null | head -5
echo ""
echo "Waiting for:"
echo "  - unreachable_vs_k_over_d2_GUE_tau0.95.png"
echo "  - reachable_vs_k_over_d2_GUE_tau0.95.png"
