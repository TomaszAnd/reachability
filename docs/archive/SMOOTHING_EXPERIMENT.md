# Canonical Basis Smoothing Experiment

**Date**: 2025-11-25
**Status**: High-resolution run in progress

---

## Motivation

The initial canonical basis plot (trials=300, rho_step=0.02) shows noisy fluctuations. We want smoother curves for publication quality.

---

## Root Cause Analysis

### 1. Statistical Noise
**Binomial uncertainty**: σ = √[p(1-p)/N]
- At p≈0.5 (mid-transition): σ = √(0.25/300) ≈ 0.029 (3% error bars)
- At p≈0.9: σ = √(0.09/300) ≈ 0.017 (1.7% error bars)

### 2. Sparse Sampling
- Current: 7 density points per dimension (rho_step=0.02)
- Issue: Large gaps between points create jagged appearance

### 3. Visual Clutter
- 9 curves total (3 criteria × 3 dimensions)
- Overlapping curves make individual noise more visible

---

## K>=2 Constraint Limitation

**Problem**: Cannot arbitrarily reduce rho_step for finer sampling

For dimension d:
- K = round(ρ × d²)
- First ρ value: rho_step
- Requirement: K >= 2

**For d=10** (d²=100):
```
rho_step=0.005 → K=round(0.5)=0  ❌ Fails
rho_step=0.010 → K=round(1.0)=1  ❌ Fails
rho_step=0.015 → K=round(1.5)=2  ✅ Works (minimum)
rho_step=0.020 → K=round(2.0)=2  ✅ Works (current)
```

**Conclusion**: Cannot improve density resolution beyond current 7 points for d=10.

---

## Solution: Higher Trials

Since we can't increase density resolution, **increase statistical power**:

### Strategy
- Keep rho_step=0.02 (7 points, satisfies K>=2)
- Increase trials: 300 → 500 (+67%)
- **Error reduction**: σ_new = σ_old × √(300/500) ≈ 0.77× σ_old

**Impact**:
- Mid-transition (p≈0.5): Error bars shrink from 3% to 2.3%
- High-p region (p≈0.9): Error bars shrink from 1.7% to 1.3%
- Smoother visual appearance

---

## High-Resolution Run

### Parameters
```bash
python scripts/generate_production_plots.py \
    --ensemble canonical \
    --dims 10,12,14 \
    --rho-max 0.15 \
    --rho-step 0.02 \  # Same as before (can't improve)
    --taus 0.95 \
    --trials 500 \     # Increased from 300
    --output-dir fig/comparison
```

### Runtime Estimate
- Previous run (trials=300): 2h 40min for 21 points
- Current run (trials=500): 21 × (500/300) = 35 point-equivalents
- **Estimated**: 3.5-4.5 hours

### Status
- **Started**: 12:15 CET
- **PID**: 34242
- **Log**: `logs/canonical_highres_20251125_121152_v3.log`
- **Expected completion**: ~15:45-16:45 CET

### Progress Tracking
```bash
# Monitor log
tail -f logs/canonical_highres_20251125_121152_v3.log

# Count completed points
grep "Collected.*trials" logs/canonical_highres_20251125_121152_v3.log | wc -l

# Check process
ps aux | grep generate_production_plots | grep -v grep
```

---

## Log-Scale Visualization

Created: `scripts/plot_log_scale_canonical.py`

### Purpose
Show exponential decay P(unreachable) ∝ exp(-αρ) more clearly on log scale.

### Usage
```bash
# After high-resolution run completes
python scripts/plot_log_scale_canonical.py \
    --dims 10,12,14 \
    --rho-max 0.15 \
    --rho-step 0.02 \
    --tau 0.95 \
    --trials 500 \
    --recompute \  # Use existing data from highres run
    --output fig/comparison/canonical_log_scale_tau0.95_highres.png
```

### Expected Output
- Y-axis: log₁₀ P(Unreachable)
- Range: 10⁻⁴ to 1
- Better visualization of exponential decay constants α

---

## Alternative Approaches (Not Pursued)

### Option A: Single-Dimension Ultra-High-Resolution
**Not feasible due to K>=2 constraint**

Original idea:
```bash
# d=10 only, very fine step
--dims 10 --rho-step 0.002 --trials 1000
```

Problem: rho_step=0.002 gives K=0 (fails validation)

### Option B: Only Test Larger Dimensions
**Not recommended - loses d=10 data**

Idea: Focus on d=12,14 where smaller rho_step is allowed
- d=12: K>=2 requires rho_step >= 0.014
- d=14: K>=2 requires rho_step >= 0.010

Problem: Losing d=10 data point, which is the smallest and most interesting dimension for canonical basis.

---

## GEO2 d=32 Analysis

**Question**: Can we run GEO2 for d=32?

### Answer: **Not feasible**

**Problem**: d=32 = 2^5 qubits, no natural rectangular lattice with 5 qubits.

GEO2 structure:
- 2×2 lattice: 4 qubits → d=2⁴=16 ✅
- 2×3 lattice: 6 qubits → d=2⁶=64 (too slow)
- **5 qubits**: No rectangular grid exists

### Options:
1. ❌ **Skip d=32** - Focus on d=16 (already publication-ready)
2. ⚠️ **Custom graph** - Possible but breaks geometric interpretation
3. ✅ **Improve d=16** - Run with trials=800 for gold standard

### Recommended: Enhance d=16
```bash
python scripts/generate_geo2_publication.py \
    --config 2x2 \
    --trials 800 \  # 2× current (was 400)
    --tau 0.95
```

**Impact**:
- Error bars shrink by √2 ≈ 0.71×
- Runtime: ~50 minutes (manageable)
- Gold-standard reference for GEO2 d=16

---

## Exponential Decay Fitting

**Goal**: Quantify decay constants α for publication.

### Method
Fit exponential model in transition region (0.2 < P < 0.8):

```python
from scipy.optimize import curve_fit

def exp_model(rho, A, alpha, rho0):
    """P(unreachable) = A × exp(-α(ρ - ρ₀))"""
    return A * np.exp(-alpha * (rho - rho0))

# For each curve
for criterion in ['spectral', 'krylov', 'moment']:
    for d in [10, 12, 14]:
        # Extract transition data
        mask = (p > 0.2) & (p < 0.8)
        popt, pcov = curve_fit(exp_model, rho[mask], p[mask])

        A, alpha, rho0 = popt
        alpha_err = np.sqrt(pcov[1,1])

        print(f"{criterion} d={d}: α = {alpha:.2f} ± {alpha_err:.2f}")
```

### Expected Results
Decay constants α should:
- Increase with dimension (larger d → faster decay)
- Vary by criterion (spectral ≠ krylov ≠ moment)
- Be positive (exponential decay, not growth)

**Implementation**: Create `scripts/fit_exponential_decay.py` (optional, for paper)

---

## Priority Summary

### ✅ In Progress (Current Session)
1. **High-resolution canonical** (trials=500) - Running now
   - **Status**: 12:15 started, ~3.5-4.5 hours ETA
   - **Output**: Smoother curves with 23% smaller error bars

2. **Log-scale plotting capability** - Script created
   - **Status**: Ready to use after highres completes
   - **Output**: `scripts/plot_log_scale_canonical.py`

### ⏭️ Next (After Highres Completes)
3. **Generate log-scale plot** from highres data
4. **Fit exponential decay models** for quantification

### ⚠️ Optional Enhancements
5. **GEO2 d=16 with trials=800** - Gold standard reference
6. **Exponential fitting script** - For publication tables

### ❌ Not Feasible
- GEO2 d=32 (no natural lattice structure)
- Finer rho_step for d=10 (K>=2 constraint)
- GEO2 d=64 with trials=300 (already proven too slow)

---

## Expected Outputs

### After Highres Completes
```
fig/comparison/three_criteria_vs_density_canonical_tau0.95_unreachable.png  (550 KB, trials=300)
fig/comparison/three_criteria_vs_density_canonical_tau0.95_unreachable_highres.png  (NEW, trials=500)
fig/comparison/canonical_log_scale_tau0.95_highres.png  (NEW, log scale)
```

### Comparison
| Version | Trials | Error Bars | Visual Quality |
|---------|--------|------------|----------------|
| Original | 300 | 3% at p=0.5 | Noisy |
| Highres | 500 | 2.3% at p=0.5 | **Smoother** ✨ |

---

## Workflow Improvement: Save Raw Data for Replotting

**Current limitation**: The production plotting scripts (`generate_production_plots.py`) compute Monte Carlo data and immediately render plots, but don't save the raw data. This means:
- Generating different plot styles (linear, log-scale, etc.) requires full recomputation
- Each replotting takes 3-4 hours for trials=500

**Recommended improvement**:
1. Modify `generate_production_plots.py` to save computed data:
   ```python
   # After monte_carlo_unreachability_vs_density() returns
   import pickle
   with open(f'data/canonical_highres_{timestamp}.pkl', 'wb') as f:
       pickle.dump(data, f)
   ```

2. Modify `plot_log_scale_canonical.py` to load from pickle:
   ```python
   with open('data/canonical_highres_latest.pkl', 'rb') as f:
       data = pickle.load(f)
   plot_log_scale_from_data(data, ...)
   ```

**Benefits**:
- Instant replotting in different styles
- Can generate linear, log-scale, different tau values, etc. without recomputation
- Easier experimentation with visualization approaches

**Alternative**: Use CSV format (already supported by some workflows) instead of pickle for better portability.

---

## Technical Insights

### 1. K>=2 is a Fundamental Limit
- Cannot arbitrarily increase density resolution
- Lower bound: rho_step_min = 2/d² for each dimension
- For canonical d=10: rho_step_min = 0.02 (already at limit!)

### 2. Statistical Power is the Only Lever
- Since density resolution is maxed out
- Only way to improve: More trials
- Diminishing returns: σ ∝ 1/√N

### 3. GEO2 Scaling is Exponential
- d=16: Practical (~25 min, trials=400)
- d=64: Impractical (days, trials=300)
- **Conclusion**: Stick with d=16 for GEO2

### 4. Log-Scale Reveals Physics
- Linear scale: Hard to see decay constant differences
- Log scale: Slopes directly show decay constants α
- Useful for comparing criteria quantitatively

---

## Recommendations for Publication

### Must Have:
1. ✅ Canonical highres plot (trials=500)
2. ✅ Log-scale canonical plot
3. ✅ GEO2 d=16 plot (trials=400)
4. ✅ Krylov convergence plot

### Nice to Have:
5. ⚠️ Exponential decay fit table (α values)
6. ⚠️ GEO2 d=16 gold standard (trials=800)

### Skip:
7. ❌ GEO2 d=64 (too expensive)
8. ❌ GEO2 d=32 (no natural lattice)

---

## Status Update: Log-Scale Plot Generation

### Highres Run - COMPLETED ✅
- **Finished**: 15:49 CET (3h 33min runtime)
- **Output**: `fig/comparison/three_criteria_vs_density_canonical_tau0.95_unreachable.png`
- **Result**: 23% smaller error bars as expected

### Log-Scale Plot - IN PROGRESS
- **Started**: 17:16 CET
- **Expected**: ~20:30-21:00 CET (~3.5-4 hours)
- **Issue**: Re-computing Monte Carlo data (previous run didn't save raw data)
- **Improvement needed**: Modify workflow to save/load raw data for replotting

**Current progress**: d=10, K=12 (5/21 points completed)

---

**Last updated**: 2025-11-25 17:27 CET
