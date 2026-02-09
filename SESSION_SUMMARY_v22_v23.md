# Session Summary: v22 → v23

**Date**: 2026-02-04 to 2026-02-05
**Scope**: Speed optimizations, Krylov corrections, publication plots, test suite overhaul

---

## Executive Summary

Sessions v22 and v23 completed a major infrastructure upgrade:

1. **v22**: Added `hams_np` speed optimization (1-1.79× speedup), launched corrected Krylov experiments with m=min(K,d)
2. **v23**: Regenerated all publication plots with corrected Krylov data, removed GUE from tables, rewrote test suite

**Key outcome**: GEO2 Krylov criterion now shows genuine phase transitions (previously P=0 everywhere due to m=d bug).

---

## v22: Speed Optimizations

### Changes Made

**1. hams_np Parameter in mathematics.py**

Added optional `hams_np` parameter to avoid repeated QObj→numpy conversion:

```python
def spectral_overlap(hams, phi, psi, lambdas, hams_np=None):
    """
    hams_np: Optional pre-extracted numpy arrays [h.full() for h in hams].
             If provided, skips QObj→numpy conversion (1.3-1.8× speedup).
    """
    if hams_np is None:
        hams_np = [h.full() for h in hams]
    # ... rest of computation
```

Applied to:
- `spectral_overlap()`
- `spectral_overlap_with_grad()`
- `krylov_score()`
- `krylov_score_with_grad()`

**2. Optimizer Integration in optimize.py**

Modified `maximize_spectral_overlap()` and `maximize_krylov_score()` to:
- Pre-extract numpy arrays once at start
- Pass `hams_np` to all criterion calls during optimization
- Avoid redundant conversion in objective + gradient functions

### Validation

| Test | Result |
|------|--------|
| Accuracy change | Zero (bit-identical results) |
| Speedup (d=8) | 1.0× (minimal benefit) |
| Speedup (d=16) | 1.3× |
| Speedup (d=32) | 1.5× |
| Speedup (d=64) | 1.79× |

### Corrected Krylov Experiments Launched

The original overnight experiments used `m=d` (full Hilbert space), causing GEO2 Krylov to trivially return P=0 everywhere. v22 launched corrected experiments:

| Ensemble | Runtime | Output |
|----------|---------|--------|
| Canonical | 604 min | `data/krylov_corrected/canonical_krylov_corrected_20260204_222726.csv` |
| GEO2 | 323 min | `data/krylov_corrected/geo2_krylov_corrected_20260204_222747.csv` |

Parameters: m=min(K,d), d∈{8,16,32,64}, K∈{5,10,20}, τ∈{0.9,0.95,0.99}, 5 trials each.

---

## v23: Publication Plots & Test Suite

### Part 1: Publication Plots Regenerated

**Data Merging Strategy**:
- Overnight data (spectral + moment): Valid, kept as-is
- Overnight Krylov: Invalid (m=d bug), dropped
- Corrected Krylov: Merged in by (d, K) join

**Script Changes** (`scripts/production/plot_overnight_v7style.py`):

```python
def load_merged_data(ensemble):
    """Load overnight spectral/moment + corrected Krylov data."""
    # Drop stale Krylov columns from overnight
    krylov_cols_to_drop = [c for c in df_overnight.columns if c.startswith('krylov_')]
    df_base = df_overnight.drop(columns=krylov_cols_to_drop)
    # Left join with corrected Krylov
    df_merged = df_base.merge(krylov_merge, on=['d', 'K'], how='left')
    return df_merged
```

**Fermi-Dirac Fit Fix** (for sharp GEO2 Krylov transitions):
```python
def fit_fermi(rho, P):
    # Better initial guess for sharp transitions
    idx = np.argmin(np.abs(P - 0.5))
    rho_c_init = rho[idx]
    popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                        p0=[rho_c_init, 0.005],
                        bounds=([0, 0.0001], [1.0, 0.5]))  # Allow very small delta
```

**Generated Plots** (7 files in `fig/publication/`):

| File | Description |
|------|-------------|
| `Kc_vs_d.png` | K_c vs d scaling (now includes GEO2 Krylov: K_c = 0.45d + 2.6) |
| `canonical_3panel.png` | Canonical 3-panel: spectral, Krylov, moment |
| `geo2_3panel.png` | GEO2 3-panel with corrected Krylov fits |
| `geo2_3panel_vs_K.png` | GEO2 vs K (alternative view) |
| `combined_criteria_canonical_d64_tau099.png` | All criteria at d=64, τ=0.99 |
| `combined_criteria_geo2_d32_tau099.png` | GEO2 combined (Krylov now shows transition) |
| `geo2_combined_criteria_vs_K_d32_tau099.png` | GEO2 combined vs K |

### Parts 2-6: Benchmark Overhaul

**GUE Removed from All Tables**:
- `tex/optimizer_comparison_table.tex`: Changed from GUE to canonical
- `tex/criteria_accuracy_table.tex`: Removed GUE rows, kept canonical + GEO2
- `tex/criteria_comparison_table.tex`: Removed GUE rows

**Rationale**: GUE was not used in final publication; canonical and GEO2 are the primary ensembles.

### Part 7: TEX Documentation

Created `tex/TEX_TABLES_DOCUMENTATION.md` documenting:
- What each table contains
- Column definitions
- How data was generated (scripts, parameters, seeds)
- Key findings
- Ground truth methodology (for accuracy tables)
- Data provenance

### Part 9: Test Suite Rewrite

Rewrote `tex/appendix_tests.tex` with new structure:

| Category | Tests | Description |
|----------|-------|-------------|
| A | 3 | Mathematical correctness (identity cases) |
| B | 2 | Gradient correctness (analytical vs FD) |
| C | 3 | Optimizer convergence |
| D | 2 | Criterion monotonicity |
| E | 2 | Speed optimization (hams_np) validation |
| F | 3 | Krylov m-parameter behavior |
| G | 3 | Model/ensemble generation |
| H | 3 | **NEW**: Data pipeline audit |

**Test H (Data Pipeline)** added to `scripts/tests/run_all_tests.py`:
- H1: Overnight CSV integrity
- H2: Corrected Krylov CSV integrity (m=min(K,d))
- H3: Merged data consistency

All 21 tests pass.

---

## File Changes Summary

### Modified Files

| File | Changes |
|------|---------|
| `reach/mathematics.py` | Added `hams_np` parameter to 4 functions |
| `reach/optimize.py` | Pre-extract hams_np, pass to criterion calls |
| `scripts/production/plot_overnight_v7style.py` | Data merging, fit fixes, corrected Krylov integration |
| `scripts/tests/run_all_tests.py` | Added Test H (data pipeline audit) |
| `tex/appendix_tests.tex` | Complete rewrite with tests A-H |
| `tex/optimizer_comparison_table.tex` | GUE → canonical |
| `tex/criteria_accuracy_table.tex` | Removed GUE rows |
| `tex/criteria_comparison_table.tex` | Removed GUE rows |

### New Files

| File | Description |
|------|-------------|
| `tex/TEX_TABLES_DOCUMENTATION.md` | Table documentation |
| `SESSION_SUMMARY_v22_v23.md` | This file |
| `data/krylov_corrected/canonical_krylov_corrected_20260204_222726.csv` | Corrected Krylov data |
| `data/krylov_corrected/geo2_krylov_corrected_20260204_222747.csv` | Corrected Krylov data |

### Regenerated Plots

All 7 publication plots in `fig/publication/` regenerated with corrected Krylov data.

---

## Key Technical Insights

### 1. Krylov m-Parameter Bug

**Original bug**: Using m=d caused Krylov subspace to span entire Hilbert space, making R*=1 trivially achievable for all targets.

**Fix**: Use m=min(K,d) to constrain Krylov dimension to control degrees of freedom.

**Impact**: GEO2 Krylov now shows genuine phase transitions:
- Before: P(unreachable) = 0 for all K (uninformative)
- After: Clear transition at K_c ≈ 0.45d + 2.6

### 2. Sharp Transition Fitting

GEO2 Krylov transitions are extremely sharp (δ ≈ 0.0003 vs δ ≈ 0.05 for spectral). Required:
- Better initial p0 guess (find P=0.5 crossing point)
- Smaller delta bounds in curve_fit (0.0001 instead of 0.001)
- Direct fitting from corrected Krylov data (not merged data)

### 3. Data Aggregation for Merging

Overnight data has multiple τ values (0.9, 0.95, 0.99) per (d, K). Merging required:
1. Aggregate overnight to one row per (d, K) with mean values
2. Deduplicate corrected Krylov before merge
3. Left join on (d, K)

---

## Commands for Reproduction

```bash
# Regenerate plots from corrected data
cd scripts/production
python plot_overnight_v7style.py

# Run full test suite
python scripts/tests/run_all_tests.py

# Verify corrected Krylov data
python -c "
import pandas as pd
df = pd.read_csv('data/krylov_corrected/geo2_krylov_corrected_20260204_222747.csv')
print(f'Rows: {len(df)}')
print(f'm=min(K,d) check: {all(df[\"krylov_m\"] == df[[\"K\", \"d\"]].min(axis=1))}')
"
```

---

## Remaining Work

The following items from v23 spec were documented but not fully implemented:

1. **Confusion matrix analysis**: Full unreachable vs reachable classification benchmarks
2. **Iterations vs accuracy tradeoff**: Detailed optimizer iteration studies
3. **Time_vs_K.png**: Noted for removal but may need explicit deletion

These are lower priority given the core objectives (corrected Krylov plots, GUE removal, test suite) are complete.

---

**Session completed**: 2026-02-05
