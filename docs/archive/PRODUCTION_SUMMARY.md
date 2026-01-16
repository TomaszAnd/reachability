# Production Run Summary

**Date**: 2025-11-25
**Status**: In Progress - Canonical Basis Running

---

## Overview

This document summarizes the production runs for quantum reachability analysis across multiple ensembles.

---

## ✅ Completed Tasks

### 1. GEO2 2×2 Lattice (d=16) - **COMPLETE**
- **Parameters**: trials=400, τ=0.95
- **Runtime**: 25 minutes (21:04 - 21:28, Nov 24)
- **Data points**: 6 density points (K=2,4,6,8,10,12)
- **Output**: `fig/comparison/geo2_2x2_d16_publication.png` (250 KB)
- **Status**: ✅ Success

### 2. Haar Uniformity Verification - **COMPLETE**
- **Purpose**: Validate qutip.rand_ket() samples Haar-uniformly
- **Tests**: Overlap distribution, frame potential
- **Dimensions tested**: d=4, d=8
- **Output Files**:
  - `fig/haar_uniformity_test_d4.png` (285 KB)
  - `fig/haar_uniformity_test_d8.png` (265 KB)
  - `fig/haar_uniformity_test_summary.png` (192 KB)
- **Status**: ✅ All tests passed (KS test p>0.01)

### 3. Krylov Convergence Analysis - **COMPLETE**
- **Parameters**: dims=[10,12,14], k_values=[6,8,10], max_iterations=[5,10,20,50,100], trials=150, τ=0.95
- **Runtime**: ~30 minutes (Nov 24)
- **Output**: `fig/krylov/krylov_convergence_canonical.png` (627 KB)
- **Status**: ✅ Success - Shows clear convergence patterns
- **Key findings**:
  - Flat lines indicate rapid convergence
  - Higher k values (6-10) needed to approach τ=0.95

### 4. Enhanced Data Logging - **TESTED**
- **Feature**: Save raw scores to pickle files for post-hoc analysis
- **Test run**: GEO2 d=16, 30 trials
- **Output**: `data/raw_logs/raw_data_GEO2_20251124_200032.pkl` (9.9 KB)
- **Status**: ✅ Working correctly

---

## ⏳ In Progress Tasks

### 5. Canonical Basis Density Sweep - **RUNNING**
- **Started**: 08:59 CET, Nov 25
- **Parameters**: dims=[10,12,14], rho_max=0.15, rho_step=0.02, trials=300, τ=0.95
- **Current progress**: d=10 (6/7 density points complete, 86%)
- **Estimated completion**: ~2-3 hours total (11:00-12:00 CET)
- **Runtime per K (d=10)**:
  - K=2: 3s
  - K=4: 7s
  - K=6: 38s
  - K=8: 187s (~3min)
  - K=10: 182s (~3min)
  - K=12: 284s (~5min)
  - K=14: In progress
- **Expected output**: `fig/comparison/three_criteria_vs_density_canonical_tau0.95_unreachable.png`
- **Process ID**: 20500
- **Log file**: `logs/canonical_production_20251125_085815_v2.log`

---

## ❌ Aborted/Failed Tasks

### 6. GEO2 2×3 Lattice (d=64) - **ABORTED**
- **Started**: 21:28 CET, Nov 24
- **Aborted**: 08:50 CET, Nov 25 (11.5 hours elapsed)
- **Reason**: **Exponential runtime growth - impractical at trials=300**
- **Progress at abort**: 3 of 20 density points (15%)
- **Runtime per K (d=64)**:
  - K=8: 10.5 min
  - K=16: 54 min (5.1× increase)
  - K=25: **6.5 hours** (7.2× increase)
  - K=33: **3.8+ hours** (still running when killed)
- **Projected completion**: Several days
- **Recommendation**:
  - Skip d=64 for now
  - OR re-run with much reduced parameters (trials=50-100, rho_step=0.005)
  - Focus on d=16 (completed successfully)

---

## Output Files Summary

### Completed Outputs (Ready for Publication):
```
fig/comparison/geo2_2x2_d16_publication.png          250 KB  ✅
fig/krylov/krylov_convergence_canonical.png          627 KB  ✅
fig/haar_uniformity_test_d4.png                      285 KB  ✅
fig/haar_uniformity_test_d8.png                      265 KB  ✅
fig/haar_uniformity_test_summary.png                 192 KB  ✅
data/raw_logs/raw_data_GEO2_20251124_200032.pkl      10 KB   ✅
```

### In Progress:
```
fig/comparison/three_criteria_vs_density_canonical_tau0.95_unreachable.png  ⏳
```

### Not Generated (Aborted):
```
fig/comparison/geo2_2x3_d64_publication.png          ❌ (too slow)
```

---

## Computational Lessons Learned

### 1. **GEO2 d=64 is Impractical at High Trials**
The runtime scales exponentially with K for large dimensions:
- d=16: Manageable (~25 min for trials=400)
- d=64: Impractical (days for trials=300)

**Recommendation**: For GEO2 d=64, use:
- trials=50-100 (not 300)
- rho_step=0.005 (coarser, fewer points)
- Or skip entirely and focus on d=16

### 2. **Canonical Basis is Much Faster**
Despite having basis size = d²:
- d=10: ~20-30 minutes for 7 density points
- d=12: ~40-60 minutes estimated
- d=14: ~60-90 minutes estimated
- **Total**: 2-3 hours for all three dimensions

This is because:
- Smaller dimensions (10,12,14 vs 64)
- Sparse operators (2 non-zero elements)
- Lower K values relative to d

### 3. **K>=2 Constraint Requires Careful rho_step Selection**
For any dimension d:
- Minimum ρ = 2/d²
- Ensure rho_step ≥ 2/d² to avoid K=1 errors

Examples:
- d=10: rho_step ≥ 0.02
- d=16: rho_step ≥ 0.008 (auto-adjusted by scripts)
- d=64: rho_step ≥ 0.0005 (fine enough)

---

## Scripts Fixed/Created

### 1. CLI Dimension Validation (reach/cli.py)
**Fixed**: Added ensemble-aware validation
- GOE/GUE: Restricted to [20,30,40,50]
- GEO2: Flexible power-of-2 dimensions
- **canonical**: Flexible (typical 10,12,14)

### 2. Canonical Ensemble Support (reach/cli.py)
**Added**: "canonical" to ensemble choices in 3 commands:
- `three-criteria-vs-K`
- `three-criteria-vs-density`
- `three-criteria-vs-K-multi-tau`

### 3. GEO2 Publication Script (scripts/generate_geo2_publication.py)
**Fixed**:
- Removed inset functionality (cleaner plots)
- Added K>=2 auto-adjustment for both single and comparison modes
- Updated docstrings

### 4. Overnight Production Script (scripts/run_overnight_production.sh)
**Created**: Automated production runs
- 4 tasks: GEO2 2×2, GEO2 2×3, Canonical, Krylov
- Comprehensive logging
- Error handling

---

## Monitoring Commands

### Check Canonical Production Progress:
```bash
# Monitor log
tail -f logs/canonical_production_20251125_085815_v2.log

# Check process
ps aux | grep generate_production_plots

# Count completed density points
grep "Collected.*trials" logs/canonical_production_20251125_085815_v2.log | wc -l

# Kill if needed
kill $(cat logs/canonical_production.pid)
```

### Check Output Files:
```bash
# All plots
ls -lh fig/comparison/*.png fig/krylov/*.png fig/*.png

# Canonical plot (when complete)
ls -lh fig/comparison/*canonical*.png
```

---

## Next Steps (After Canonical Completes)

### 1. Verify Output Quality
- Check canonical plot looks correct
- All three criteria visible
- Error bars reasonable
- Proper labels and legend

### 2. Optional: Run Higher-Resolution Krylov
If more detail needed:
```bash
python scripts/krylov_convergence_canonical.py \
    --dims 10,12,14 \
    --k-values 6,8,10 \
    --max-iterations 5,10,20,50,100 \
    --trials 300 \
    --tau 0.95 \
    --output fig/krylov/krylov_convergence_canonical_production.png
```

### 3. Optional: GEO2 d=64 with Reduced Parameters
If GEO2 d=64 data is needed:
```bash
python scripts/generate_geo2_publication.py \
    --config 2x3 \
    --trials 100 \  # Reduced from 300
    --tau 0.95
```
**Note**: Will still take 4-6 hours but more manageable.

### 4. Create Comparison Plots
Generate side-by-side comparisons:
- GUE vs GOE vs canonical vs GEO2
- Different τ values
- Ensemble-specific behaviors

---

## Runtime Summary

| Task | Status | Runtime | Trials |
|------|--------|---------|--------|
| GEO2 2×2 (d=16) | ✅ Done | 25 min | 400 |
| Haar Tests | ✅ Done | ~5 min | - |
| Krylov Convergence | ✅ Done | ~30 min | 150 |
| GEO2 2×3 (d=64) | ❌ Aborted | 11.5 hrs (15% complete) | 300 |
| Canonical (d=10,12,14) | ⏳ Running | 2-3 hrs estimated | 300 |

**Total successful runtime**: ~1 hour
**Total wasted runtime**: ~11.5 hours (GEO2 d=64)
**Expected remaining**: ~2-3 hours (canonical)

---

## Files Created This Session

### Code:
- `scripts/run_overnight_production.sh` (production automation)
- `scripts/generate_geo2_publication.py` (fixed and enhanced)
- `reach/cli.py` (canonical support added)

### Documentation:
- `/tmp/production_run_commands.md` (comprehensive reference)
- `/tmp/overnight_production_summary.txt` (session summary)
- `/tmp/cleanup_completion_summary.txt` (task completion)
- `PRODUCTION_SUMMARY.md` (this file)

### Data:
- `logs/canonical_production_20251125_085815_v2.log` (ongoing)
- `logs/overnight_nohup.log` (from overnight run)
- `data/raw_logs/raw_data_GEO2_*.pkl` (test data)

---

**Last updated**: 2025-11-25 09:16 CET
**Current status**: Canonical basis running (ETA: 11:00-12:00 CET)
