# Publication Plot Documentation

**Generated:** 2026-02-08 20:46


## Data Provenance


### IMPORTANT: Merged Data Sources

**Spectral and Moment data** come from overnight runs (valid).
**Krylov data** comes from CORRECTED runs with `m=min(K,d)` (valid).

The overnight Krylov columns used `m=d` (full Hilbert space), which caused:
- GEO2 Krylov to be trivially P=0 everywhere (Krylov subspace = full R^d)
- Canonical Krylov to have different (less meaningful) transitions

The corrected Krylov experiments use `m=min(K,d)`, creating genuine phase transitions.

### Canonical Ensemble
- **Spectral/Moment**: `data/overnight/canonical_overnight_20260128_224236.csv`
  - 327 data points, 142.9 hours runtime
- **Krylov (corrected)**: `data/krylov_corrected/canonical_krylov_corrected_20260204_222726.csv`
  - 108 data points, ~10 hours runtime, m=min(K,d)
- **Dimensions**: d in {8, 16, 32, 64}
- **Threshold**: tau = 0.99 (plots)
- **Trials per point**: 200

### GEO2 Ensemble
- **Spectral/Moment**: `data/overnight/geo2_overnight_20260128_224613.csv`
  - 101 data points, 47.2 hours runtime
- **Krylov (corrected)**: `data/krylov_corrected/geo2_krylov_corrected_20260204_222747.csv`
  - 96 data points, ~5.4 hours runtime, m=min(K,d)
- **Dimensions**: d in {8, 16, 32, 64}
- **Threshold**: tau = 0.99
- **Trials per point**: 200 (100 for d=64)

## Plot Generation

- **Script**: `scripts/production/plot_overnight_v7style.py`
- **Based on**:
  - `scripts/canonical/generate_publication_dual_versions.py` (canonical v7 style)
  - `scripts/geo2/plot_geo2_v3.py` (GEO2 v3 style)
- **Date generated**: 2026-02-08
- **Style reference**: Unified style documented in CLAUDE.md

---

## Plot Descriptions

### Plot 1: `combined_criteria_canonical.png`
All three criteria (Spectral, Krylov, Moment) on a single plot for d=64, tau=0.99.
Style: `combined_criteria_d26_v7_exp.png`. X-axis: rho = K/d^2. Y-axis: P(unreachable).

### Plot 2: `combined_criteria_geo2.png`
Same as Plot 1 for GEO2 ensemble, d=64, tau=0.99.
Style: `geo2_d64_summary_v3.png`. Note: Krylov shows P~0 everywhere (correct physics).

### Plot 3: `canonical_3panel.png`
3-panel figure (Moment | Spectral | Krylov). Each panel shows all dimensions (d=8,16,32,64).
Style: `final_summary_3panel_v7_exp.png`. tau=0.99.

### Plot 4: `geo2_3panel.png`
Same 3-panel layout for GEO2 ensemble, tau=0.99.
Style: `geo2_main_v3.png`.

### Plot 5: `Kc_vs_d.png`
Critical K_c vs dimension d for both ensembles, all fitted criteria.
Style: `Kc_vs_d_v7_exp.png`. Linear scale.

### Plot 6: `geo2_combined_criteria_vs_K.png`
Same as Plot 2 but with K (absolute count) on x-axis instead of rho.

### Plot 7: `geo2_3panel_vs_K.png`
Same as Plot 4 but with K on x-axis.

### Plot 8: `canonical_combined_criteria_vs_K_d{d}_tau099.png`
All three criteria on single plot with K on x-axis, for each canonical dimension.

### Plot 9: `canonical_3panel_vs_K.png`
Same as Plot 3 but with K (absolute count) on x-axis instead of rho.
X-axis range extends to K_max from data.

---

## Fit Function Specifications

### Fermi-Dirac (Spectral and Krylov Criteria)

$$P(\rho) = \frac{1}{1 + \exp\!\left(\frac{\rho - \rho_c}{\Delta}\right)}$$

- $\rho_c$: transition density (P = 0.5 at $\rho = \rho_c$)
- $\Delta$: transition width
- $K_c = \rho_c \times d^2$

### Simple Exponential (Moment Criterion)

$$P(\rho) = \exp\!\left(-\frac{\rho}{\lambda}\right)$$

- $\lambda$: decay scale
- $\rho_{1/2} = \lambda \ln 2$
- $K_c = d^2 \lambda \ln 2$

### Linear K_c Scaling

$$K_c = a \cdot d + b$$

---

## Canonical Ensemble Fit Parameters (tau = 0.99)

### Spectral Criterion

| d | rho_c | Delta | K_c | R^2 |
|---|-------|-------|-----|-----|
| 8 | 0.16790 | 0.02711 | 10.7 | 0.996 |
| 16 | 0.09638 | 0.01574 | 24.7 | 0.982 |
| 32 | 0.05518 | 0.00914 | 56.5 | 0.978 |
| 64 | 0.02905 | 0.00434 | 119.0 | 0.965 |

### Krylov Criterion

| d | rho_c | Delta | K_c | R^2 |
|---|-------|-------|-----|-----|
| 8 | 0.15545 | 0.02993 | 9.9 | 0.964 |
| 16 | 0.08410 | 0.02424 | 21.5 | 0.962 |
| 32 | 0.04332 | 0.02020 | 44.4 | 0.839 |
| 64 | 0.02765 | 0.00876 | 113.2 | 0.887 |

### Moment Criterion

| d | Fit Type | Parameters | K_c | R^2 |
|---|----------|------------|-----|-----|
| 8 | Exponential | lambda=0.06706, rho_1/2=0.04648 | 3.0 | 0.833 |
| 16 | Exponential | lambda=0.03087, rho_1/2=0.02140 | 5.5 | 0.967 |
| 32 | Exponential | lambda=0.01583, rho_1/2=0.01097 | 11.2 | 0.982 |
| 64 | Exponential | lambda=0.00786, rho_1/2=0.00545 | 22.3 | 0.992 |

---

## GEO2 Ensemble Fit Parameters (tau = 0.99)

### Spectral Criterion

| d | rho_c | Delta | K_c | R^2 |
|---|-------|-------|-----|-----|
| 8 | 0.07362 | 0.00896 | 4.7 | 0.999 |
| 16 | 0.03527 | 0.00303 | 9.0 | 0.998 |
| 32 | 0.01860 | 0.00142 | 19.0 | 0.991 |
| 64 | 0.00976 | 0.00060 | 40.0 | 0.999 |

### Krylov Criterion

| d | rho_c | Delta | K_c | R^2 |
|---|-------|-------|-----|-----|
| 8 | 0.07776 | 0.00300 | 5.0 | 1.000 |
| 16 | 0.03730 | 0.00097 | 9.5 | 1.000 |
| 32 | 0.01833 | 0.00037 | 18.8 | 1.000 |
| 64 | 0.00903 | 0.00010 | 37.0 | 1.000 |

### Moment Criterion

| d | Fit Type | Parameters | K_c | R^2 |
|---|----------|------------|-----|-----|
| 8 | Exponential | lambda=0.04240, rho_1/2=0.02939 | 1.9 | 0.613 |
| 16 | Exponential | lambda=0.00789, rho_1/2=0.00547 | 1.4 | 0.568 |
| 32 | Exponential | lambda=0.00106, rho_1/2=0.00073 | 0.8 | 0.999 |

---

## K_c Linear Scaling

$$K_c = a \cdot d + b$$

| Ensemble | Criterion | a (slope) | b (intercept) | R^2 |
|----------|-----------|-----------|---------------|-----|
| Canonical | Krylov | 1.86 | -8.5 | 0.990 |
| Canonical | Moment | 0.35 | 0.1 | 1.000 |
| Canonical | Spectral | 1.94 | -5.6 | 1.000 |
| GEO2 | Krylov | 0.57 | 0.4 | 1.000 |
| GEO2 | Moment | -0.05 | 2.2 | 0.989 |
| GEO2 | Spectral | 0.63 | -0.8 | 0.999 |

---

## Notes on Krylov m-Parameter Correction

**Background**: The original overnight experiments used `m=d` (full Hilbert space 
dimension) for the Krylov subspace. This caused:

1. **GEO2 Krylov was trivially P=0**: Dense GEO2 Hamiltonians never cause early 
   Arnoldi breakdown, so the Krylov basis spans all of R^d regardless of K.
2. **Canonical Krylov was also affected**: Though Arnoldi breakdown occurs earlier 
   for sparse canonical operators, using m=d still inflates reachability scores.

**Correction**: The corrected experiments use `m=min(K,d)`, so the Krylov rank 
grows with K (number of Hamiltonians), creating genuine phase transitions.

**Result**: GEO2 Krylov now shows real transitions (P drops from 1.0 to 0.0 as 
K increases), and is included in K_c vs d scaling.

---

## Formatting Changes from v7/v3

1. **Removed figure-level suptitles** (`fig.suptitle()` calls removed)
2. **Removed fit equation text boxes** from plot legends (equations documented here)
3. **Fit curves at 50% opacity** (`alpha=0.5` on all fit lines)
4. **Font sizes increased ~2pt** from v7/v3 originals for single-column LaTeX readability
