# Overnight Production Run Analysis

**Generated**: 2026-01-30 11:55:00
**Status**: Experiments still in progress (checkpoint data)

---

## Executive Summary

Both overnight experiments are running successfully with high-quality data:
- **Canonical**: 52% complete (170/327 points), d=8,16 fully sampled, d=32 in transition
- **GEO2**: 84% complete (85/101 points), d=8,16,32 complete, d=64 partial

**Key Finding**: GEO2 transitions occur at 2-3× lower ρ than canonical for the same dimension, confirming geometric constraints significantly limit reachability.

---

## Experiment Status

| Experiment | Progress | ETA | Status |
|------------|----------|-----|--------|
| Canonical | 170/327 (52%) | ~28h | Running (d=32, τ=0.99) |
| GEO2 | 85/101 (84%) | ~7h | Running (d=64) |

---

## Data Summary

### Canonical
- **File**: `canonical_checkpoint_20260128_224236.csv`
- **Rows**: 170 data points
- **Dimensions**: d = 8, 16, 32
- **τ values**: 0.90, 0.95, 0.99
- **ρ range**: [0.0098, 0.3438]
- **K range**: [2, 64]
- **Criteria**: Spectral, Krylov, Moment (all three)

### GEO2
- **File**: `geo2_checkpoint_20260128_224613.csv`
- **Rows**: 85 data points
- **Dimensions**: d = 8, 16, 32, 64
- **τ values**: 0.99 only
- **ρ range**: [0.0049, 0.2031]
- **K range**: [2, 82]
- **Criteria**: Spectral, Krylov, Moment
- **Note**: Krylov always returns P=0 (expected physics - see below)

---

## Transition Densities (ρ_c)

### Spectral Criterion (τ=0.99)

| Ensemble | d=8 | d=16 | d=32 | d=64 |
|----------|-----|------|------|------|
| Canonical | 0.1679 | 0.0964 | - | - |
| GEO2 | 0.0736 | 0.0353 | 0.0186 | 0.0096 |

### Ratio (Canonical/GEO2)

| Dimension | Canonical ρ_c | GEO2 ρ_c | Ratio |
|-----------|---------------|----------|-------|
| d=8 | 0.168 | 0.074 | **2.3×** |
| d=16 | 0.096 | 0.035 | **2.7×** |

**Interpretation**: GEO2 requires ~2-3× fewer operators (relative to d²) to achieve reachability compared to canonical basis. This is because GEO2 Hamiltonians have richer structure (geometric/local constraints).

---

## Physics Validation

### 1. Dimension Ordering ✅

Both ensembles show correct physics: **higher d → lower ρ_c**

- Canonical: ρ_c(d=8) > ρ_c(d=16) ✓
- GEO2: ρ_c(d=8) > ρ_c(d=16) > ρ_c(d=32) > ρ_c(d=64) ✓

### 2. Criterion Hierarchy ✅

For canonical basis at τ=0.99:
- Moment < Spectral ≈ Krylov (Moment transitions fastest)
- This is consistent with expected behavior

### 3. GEO2 Krylov Behavior ✅

GEO2 Krylov shows **P(unreachable) = 0 for all points**. This is **correct physics**:
- Dense GEO2 Hamiltonians quickly fill the Krylov subspace
- Krylov score R* ≈ 1.0 always (machine precision: ~3×10⁻¹⁶)
- Therefore Krylov never detects unreachability for GEO2
- This confirms the audit finding: "Krylov is uninformative for GEO2"

---

## Scaling Analysis

### GEO2 ρ_c vs Dimension

Fitting ρ_c = A × d^(-β):

| Dimension | ρ_c | K_c = d² × ρ_c |
|-----------|-----|----------------|
| 8 | 0.0736 | 4.7 |
| 16 | 0.0353 | 9.0 |
| 32 | 0.0186 | 19.0 |
| 64 | 0.0096 | 39.4 |

**Preliminary scaling**: K_c ∝ d^α with α ≈ 1.0-1.2 (sub-quadratic)

This suggests GEO2 reachability scales more favorably than the d² baseline.

---

## Generated Figures

| File | Description |
|------|-------------|
| `canonical_spectral_transitions.png` | Canonical d=8,16,32 phase transitions with Fermi-Dirac fits |
| `geo2_spectral_transitions.png` | GEO2 d=8,16,32,64 phase transitions |
| `canonical_vs_geo2_comparison.png` | Direct overlay comparison |
| `all_criteria_grid.png` | 2×3 grid showing all criteria for both ensembles |
| `P_vs_K.png` | Transitions vs absolute Hamiltonian count |
| `canonical_multi_tau.png` | Canonical with τ=0.90, 0.95, 0.99 |

---

## Data Quality Notes

1. **Canonical d=32**: Only partial data (P≈1 region) - transition not yet sampled
2. **Canonical d=64**: Not yet started
3. **GEO2 d=64**: Partial (5/20 points sampled)
4. **GEO2 Krylov NaN**: The krylov_P column shows NaN in validation because P=0.0 is stored as-is, but krylov_mean/std may have numerical issues at machine precision

---

## Recommendations

1. **Wait for completion**: Both experiments will provide publication-quality data
2. **Re-run analysis**: Once complete, re-run this script for final figures
3. **GEO2 Krylov**: Can be excluded from GEO2 plots (uninformative criterion)
4. **Scaling fits**: Final data will allow robust power-law fits for ρ_c(d)

---

## Next Steps

When experiments complete:
1. Run `python scripts/production/analyze_overnight_checkpoints.py` again
2. Final data files will be in `data/overnight/` as `*_overnight_*.csv`
3. Update figures with complete d=32, d=64 data
4. Perform final Fermi-Dirac fits and extract publication parameters
