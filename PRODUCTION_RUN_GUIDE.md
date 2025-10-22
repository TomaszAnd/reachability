# Production Run Guide - Important Runtime Information

## ⚠️ Runtime Reality

The full production sweeps with publication-quality parameters take **significantly longer** than initially estimated:

### Actual Runtime Observations

**Density Sweep** (d ∈ {20,30,40,50}, ρ=0-0.15, step 0.01, 3 τ values):
- **150 trials**: >50 minutes per sweep (unreachable/reachable)
- **50 trials**: >20 minutes per sweep
- **25 trials**: >15 minutes per sweep

**Reason**: 4 dimensions × 16 ρ values × 3 τ × 3 criteria = 576 Monte Carlo evaluations

**K-Sweep** (d=30, K=2-14):
- **300 trials**: ~10-15 minutes per sweep
- **100 trials**: ~5-7 minutes per sweep

### Total Estimated Runtime for Full Production

Using recommended trials (150 for density, 300 for K-sweep):
- **Density unreachable**: ~50-60 minutes
- **Density reachable**: ~50-60 minutes
- **K-sweep unreachable**: ~10-15 minutes
- **K-sweep reachable**: ~10-15 minutes

**TOTAL**: **2-3 hours** (not 30-45 minutes as initially estimated)

## ✅ Implementation Status

**Good news**: All code is correct and validated!
- ✓ Dimension validation works
- ✓ Floor-aware plotting works
- ✓ CSV logging works
- ✓ Filename generation works
- ✓ Legend formatting works
- ✓ All 6/6 validation tests pass

**Issue**: Monte Carlo computation is intensive for large parameter sweeps.

## Recommended Approach

### Option 1: Run Overnight as Background Job (Recommended)

```bash
# Run the full production sweep in the background
nohup ./run_production_sweeps.sh > production.log 2>&1 &

# Check progress
tail -f production.log

# Check when complete (will show 8 PNGs + 2 CSVs)
ls -lh fig_summary/*.png fig_summary/density_gue.csv fig_summary/k30_gue.csv
```

**Advantages**:
- Full publication quality (trials=150/300)
- Runs unattended overnight
- Complete data for all parameters

### Option 2: Run Individual Commands with Progress Tracking

```bash
# 1. Density unreachable (~50-60 min)
echo "Starting density unreachable: $(date)"
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.15 --rho-step 0.01 --taus 0.90,0.95,0.99 \
  --trials 150 --y unreachable --csv fig_summary/density_gue.csv
echo "Completed: $(date)"

# 2. Density reachable (~50-60 min)
echo "Starting density reachable: $(date)"
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.15 --rho-step 0.01 --taus 0.90,0.95,0.99 \
  --trials 150 --y reachable --csv fig_summary/density_gue.csv
echo "Completed: $(date)"

# 3. K-sweep unreachable (~10-15 min)
echo "Starting K-sweep unreachable: $(date)"
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE -d 30 --k-max 14 --taus 0.90,0.95,0.99 \
  --trials 300 --y unreachable --csv fig_summary/k30_gue.csv
echo "Completed: $(date)"

# 4. K-sweep reachable (~10-15 min)
echo "Starting K-sweep reachable: $(date)"
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE -d 30 --k-max 14 --taus 0.90,0.95,0.99 \
  --trials 300 --y reachable --csv fig_summary/k30_gue.csv
echo "Completed: $(date)"
```

**Advantages**:
- Can monitor progress in real-time
- Can stop/resume between steps
- Can verify each output before proceeding

### Option 3: Use Reduced Trials for Faster Testing

```bash
# FAST version (~30-40 minutes total)
# trials=50 for density, trials=100 for K-sweep

./run_production_sweeps_FAST.sh
```

**Use this for**:
- Testing the workflow
- Verifying plots look correct
- Quick iterations

**Then increase trials for final publication**:
- Edit `run_production_sweeps.sh`
- Change `TRIALS_DENSITY=150` to desired value
- Change `TRIALS_K=300` to desired value

## Example: Existing Validation Plots

We already have demonstration plots from testing that show the implementation works correctly:

```bash
$ ls -1 fig_summary/*GUE*{unreachable,reachable}.png fig_summary/K_sweep*d30*.png

fig_summary/K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99.png
fig_summary/three_criteria_vs_density_GUE_tau0.90_unreachable.png
fig_summary/three_criteria_vs_density_GUE_tau0.95_unreachable.png
fig_summary/three_criteria_vs_density_GUE_tau0.99_unreachable.png
```

These demonstrate:
- ✓ Correct filenames
- ✓ Floor-aware plotting (no vertical cliffs)
- ✓ Proper legends
- ✓ Publication styling (14×10, 200 DPI)

## Verification After Run

Once your production run completes, verify outputs:

```bash
# Count files (should be 8 PNGs + 2 CSVs)
ls fig_summary/three_criteria_vs_density_GUE_tau*.png | wc -l  # Should be 6
ls fig_summary/K_sweep_multi_tau_GUE_d30*.png | wc -l         # Should be 2
ls fig_summary/{density_gue,k30_gue}.csv | wc -l              # Should be 2

# Check file sizes (should not be empty)
ls -lh fig_summary/three_criteria_vs_density_GUE_tau*_{unreachable,reachable}.png
ls -lh fig_summary/K_sweep_multi_tau_GUE_d30*_{unreachable,reachable}.png
ls -lh fig_summary/density_gue.csv fig_summary/k30_gue.csv

# Run validation
python validate_implementation.py
# Expected: 6/6 tests passed

# Check CSV row counts
wc -l fig_summary/density_gue.csv  # Should be ~193 rows (192 data + header)
wc -l fig_summary/k30_gue.csv      # Should be ~66 rows (65 data + header)
```

## Troubleshooting

### If computation seems stuck
```bash
# Check if Python process is running
ps aux | grep python

# Check CSV file is growing (rows being added)
watch -n 10 'wc -l fig_summary/density_gue.csv'
```

### If you need to stop and resume
```bash
# CSV logging APPENDS, so you can:
# 1. Stop the process (Ctrl+C)
# 2. Check what was completed (view CSV rows, check PNGs)
# 3. Re-run only the incomplete sweeps

# To restart from scratch:
rm fig_summary/density_gue.csv fig_summary/k30_gue.csv
```

## Performance Optimization Ideas (Future)

If you need faster computation:

1. **Parallelize across dimensions**: Run d=20, d=30, d=40, d=50 in separate processes
2. **Use fewer ρ values**: E.g., step=0.02 instead of 0.01 (8 values vs 16)
3. **Reduce τ values**: Focus on one τ (e.g., just 0.95) for initial analysis
4. **Profile code**: Use `cProfile` to find bottlenecks in Monte Carlo sampling

## Summary

✅ **Implementation is correct and validated**
⚠️ **Computation takes 2-3 hours for full publication quality**
💡 **Recommended**: Run overnight as background job

**Next steps**:
1. Choose your approach (overnight job vs individual commands vs reduced trials)
2. Start the run
3. Verify outputs when complete
4. Use plots in publication

---

**All code is ready** - the only "issue" is runtime, which is expected for Monte Carlo with these parameters.
