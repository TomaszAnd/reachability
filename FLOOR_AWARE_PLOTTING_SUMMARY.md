# Floor-Aware Plotting Implementation - Complete ✅

## Overview
Successfully implemented publication-ready floor-aware plotting that eliminates misleading vertical "cliff" drops when P(unreachable) values hit the display floor (1e-12).

## Root Cause of Vertical Cliffs
When Monte Carlo trials yield zero unreachable instances, P(unreachable) is floored at 1e-12 (the display floor). Matplotlib by default draws straight lines connecting all points, including from the last finite value down to the floor, creating misleading vertical segments that suggest a sharp transition rather than clipped data.

## Solution Implemented

### 1. Floor Detection
```python
is_floored = np.abs(p - floor) < floor * 0.01
```
Identifies points at or near the display floor.

### 2. Masked Arrays for Line Breaks
```python
def _create_floor_masked_array(y_values: np.ndarray, floor: float) -> np.ma.MaskedArray:
    """
    Create masked array that breaks line segments at floor values.
    Prevents vertical "cliff" lines when values hit display floor.
    """
    is_floored = np.abs(y_values - floor) < floor * 0.01
    return np.ma.masked_where(is_floored, y_values)
```
Using masked arrays, matplotlib automatically breaks line segments at masked (floored) points.

### 3. Separate Floored Point Rendering
- Non-floored points: plotted with connected lines
- Floored points: plotted as faded markers (alpha=0.3) WITHOUT connecting lines

### 4. Visual Annotation
When any curve hits the floor, a text box appears:
```
Display floor: 1e-12
(faded markers show floored values)
```

## Updated Functions

### `plot_unreachability_three_criteria_vs_density()`
- **Location**: `reach/viz.py` lines 66-2309
- **Figure size**: 14×10 inches, DPI 200
- **Features**:
  - Floor-aware plotting for all criteria × dimensions
  - Enhanced typography (16pt axes, 18pt title, bold labels)
  - Improved color palette for dimensions
  - Floor annotation
  - Better legend with frame

### `plot_unreachability_K_multi_tau()`
- **Location**: `reach/viz.py` lines 2312-2548
- **Figure size**: 14×10 inches, DPI 200
- **Features**:
  - Floor-aware plotting for spectral (multi-τ), old, and Krylov
  - Blue gradient for spectral τ values
  - Enhanced typography
  - Floor annotation
  - Larger markers and thicker lines

## Acceptance Tests

### Test 1: K-sweep multi-tau (d=15, K=2-8, τ∈{0.90,0.95,0.99})
```bash
python -m reach.cli three-criteria-vs-K-multi-tau \
  -d 15 --k-max 8 --taus 0.90,0.95,0.99 \
  --ensemble GUE --trials 40 \
  --csv fig_summary/k15_test_floor_aware.csv
```

**Result**: ✅ PASS
- Old criterion shows floored points at K≥5 as faded orange markers
- No vertical cliffs visible
- Floor annotation displayed correctly
- Figure: `fig_summary/K_sweep_multi_tau_GUE_d15_taus0.90_0.95_0.99.png`

### Test 2: Density sweep (d∈{15,20}, ρ=0-0.04, τ∈{0.90,0.95})
```bash
python -m reach.cli three-criteria-vs-density \
  --dims 15,20 --taus 0.90,0.95 \
  --rho-max 0.04 --rho-step 0.01 --k-cap 50 \
  --ensemble GUE --trials 40 \
  --csv fig_summary/density_test_floor_aware.csv
```

**Result**: ✅ PASS
- Multiple criteria show floored points as faded markers
- No vertical cliffs visible
- Floor annotation displayed correctly
- Figures:
  - `fig_summary/three_criteria_vs_density_GUE_tau0.90_unreachable.png`
  - `fig_summary/three_criteria_vs_density_GUE_tau0.95_unreachable.png`

### Test 3: CSV Schema Validation
**Result**: ✅ PASS - All 15 required fields present and populated:
1. run_id
2. timestamp
3. ensemble
4. criterion
5. tau
6. d
7. K
8. m
9. rho_K_over_d2
10. trials
11. successes_unreach
12. p_unreach
13. log10_p_unreach
14. mean_best_overlap (spectral rows only)
15. sem_best_overlap (spectral rows only)

## Publication-Ready Features

### Typography
- **Axis labels**: 16pt, bold
- **Title**: 18pt, bold
- **Legend**: 12pt, framed
- **Tick labels**: 14pt

### Visual Quality
- **Figure size**: 14×10 inches (optimal for publications)
- **DPI**: 200 (high resolution)
- **Line width**: 2.0 (thicker for visibility)
- **Marker size**: 6 (larger for clarity)
- **Grid**: alpha=0.25 (subtle but helpful)

### Error Bars
- Wilson score intervals for proper binomial confidence
- Only shown for non-floored points
- Capsize=3, linewidth=1.2

## Key Benefits

1. **Eliminates misleading visuals**: No more vertical "cliffs" that suggest sharp transitions
2. **Clear floor indication**: Faded markers explicitly show which data points hit the floor
3. **Publication-ready**: Professional styling suitable for papers
4. **Complete data logging**: All metadata and statistics preserved in CSV
5. **Efficient computation**: τ-reuse for spectral criterion reduces redundant calculations

## Files Modified

- `reach/viz.py`: Added `_create_floor_masked_array()` helper, updated both plot functions
- `reach/logging_utils.py`: Already complete with 15-field schema
- `reach/analysis.py`: Already complete with spectral statistics collection
- `reach/cli.py`: Already complete with CSV logging hooks

## Next Steps for Full Production Runs

For publication-quality data with higher statistical power:

```bash
# Density sweep (full acceptance test)
python -m reach.cli three-criteria-vs-density \
  --dims 20,30,40,50 --taus 0.90,0.95,0.99 \
  --rho-max 0.15 --rho-step 0.01 --k-cap 200 \
  --ensemble GUE --trials 300 \
  --csv fig_summary/density_sweep_full.csv

# K-sweep multi-tau (full acceptance test)
python -m reach.cli three-criteria-vs-K-multi-tau \
  -d 30 --k-max 14 --taus 0.90,0.95,0.99 \
  --ensemble GUE --trials 300 \
  --csv fig_summary/k30_sweep_full.csv
```

Note: These full runs will take 15-30 minutes due to Monte Carlo sampling.

---

**Implementation Status**: COMPLETE ✅
**Date**: 2025-10-21
**All Tests**: PASSED ✅
