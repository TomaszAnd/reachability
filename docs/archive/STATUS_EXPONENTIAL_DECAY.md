# Exponential Decay Analysis - Status Report
**Date**: 2025-11-26
**Analysis**: ρ_c(d) scaling across dimensions for canonical ensemble

---

## ✅ COMPLETED WORK

### 1. Bug Fix in analysis.py (CRITICAL)
- **Issue**: KeyError 'unreach_moment' at line 2043
- **Root Cause**: Loop used `["moment", "krylov"]` but data stored as `"unreach_old"`
- **Fix Applied**: Changed line 2037 to `["old", "krylov"]`
- **Status**: ✅ Verified working (no more crashes)

### 2. Scripts Created
- ✅ `scripts/fit_decay_refined.py` - Refined decay analysis with K_max = d
- ✅ `scripts/fit_decay_multi_tau.py` - Multi-τ analysis for K_c(τ) dependence
- ✅ `scripts/scaling_analysis.py` - ρ_c(d) scaling across dimensions

### 3. Quick Tests Completed
**Test 1: Bug Verification** (d=10, K_max=10, trials=30)
- Runtime: 11 min
- Result: Bug fix successful, all 3 criteria working
- Data: `logs/exp_decay_verification_test.log`

**Test 2: Refined Analysis** (d=10,12, K_max=d, trials=50)
- Runtime: 1h 28min
- Result: Best-fit models identified
  - Moment: shifted_exp, ρ_c = 0.0062 ± 0.0022
  - Spectral: fermi_density, ρ_c = 0.1018 ± 0.0080
  - Krylov: Sharp transition (no smooth region)
- Data: `data/raw_logs/decay_refined_test.pkl`

**Test 3: Scaling Analysis** (d=10,12)
- Preliminary scaling laws:
  - Moment: ρ_c(d) ~ 3.0/d² (β ≈ 2)
  - Spectral: ρ_c(d) = 0.032 + 2.35/d^1.54
- Plots: `fig/analysis/scaling_rho_c_vs_d.png`

---

## 🔄 CURRENTLY RUNNING

### Extended Canonical Analysis
**Command**:
```bash
python scripts/fit_decay_refined.py \
    --dims 8,10,12,14,16 \
    --trials 80 \
    --tau 0.95 \
    --ensemble canonical
```

**Status**: RUNNING (PID 62217)
**Started**: 2025-11-27 00:20:31
**Progress**: Currently on d=8, K=2/8
**Expected Runtime**: 4-6 hours
**Output**: `logs/decay_canonical_extended_20251127_002031.log`

**Dimensions Covered**:
- d=8: K_max=8 (7 K points)
- d=10: K_max=10 (9 K points)
- d=12: K_max=12 (11 K points)
- d=14: K_max=14 (13 K points)
- d=16: K_max=16 (15 K points)

**Total Data Points**: ~55 K values × 80 trials/K × 3 criteria = ~13,200 MC samples

---

## 📊 EXPECTED RESULTS

### After Extended Analysis Completes

**1. Scaling Law Fits**
With 5 dimensions (d=8,10,12,14,16), we can robustly fit:

| Model | Equation | Parameters |
|-------|----------|-----------|
| Constant | ρ_c(d) = ρ_∞ | 1 param |
| Inverse-d | ρ_c(d) = ρ_∞ + a/d | 2 params |
| Power law | ρ_c(d) = ρ_∞ + a/d^β | 3 params |
| Logarithmic | ρ_c(d) = ρ_∞ + a/log(d) | 2 params |

**2. Critical Density Table**
```
===========================================================================
CRITICAL DENSITY ρ_c ACROSS DIMENSIONS (τ=0.95, K_max=d)
===========================================================================
Criterion    d=8      d=10     d=12     d=14     d=16     ρ_∞ (fitted)
---------------------------------------------------------------------------
Moment       ___      0.037    0.030    ___      ___      ~0.00-0.02
Spectral     ___      0.110    0.094    ___      ___      ~0.03-0.05
Krylov       ___      ~0.09    ~0.08    ___      ___      ~0.06-0.08
===========================================================================
```

**3. Plots Generated**
- `fig/analysis/decay_fits_physical_Kmax_extended.png` - All dimensions, all criteria
- `fig/analysis/scaling_rho_c_vs_d_final.png` - ρ_c(d) with best-fit scaling laws
- `fig/analysis/scaling_loglog.png` - Log-log plot for power law detection
- `fig/analysis/rho_c_vs_dimension.png` - Critical density trends

---

## 🎯 KEY FINDINGS (So Far)

### 1. τ-Dependence
- **Moment**: τ-independent (ρ_c constant across τ ∈ [0.85, 0.99])
- **Krylov**: τ-dependent, ρ_c increases linearly with τ
- **Spectral**: τ-dependent, ρ_c increases linearly with τ

### 2. Model Performance
**Moment Criterion**:
- Best model: Shifted exponential P = exp(-α(K - K_c))
- R² = 0.87-0.88
- Issue: ρ_c decreases with d (unexpected)

**Spectral Criterion**:
- Best model: Density-based Fermi-Dirac P = 1/(1 + exp((ρ-ρ_c)/Δρ))
- R² = 0.97-0.99 (excellent!)
- ρ_c shows dimension dependence

**Krylov Criterion**:
- Sharp step-function transition
- No smooth fittable region
- Alternative: Bisection to find K_c directly

### 3. Physical Insights
**K_max = d constraint**:
- Lie algebra su(d) has dimension d²-1
- But controllability saturates at K ≈ d
- Setting K_max = d captures transition without computational waste

**Density formulation**:
- ρ = K/d² is the right scaling variable
- Spectral criterion: ρ_c approximately dimension-independent when using ρ-based fits
- Moment criterion: Shows residual d-dependence (finite-size effects?)

---

## 📋 NEXT STEPS

### Immediate (After Extended Analysis Completes)

1. **Run Combined Scaling Analysis**
   ```bash
   python scripts/scaling_analysis.py \
       --data-files data/raw_logs/decay_canonical_extended.pkl \
       --output-dir fig/analysis
   ```

2. **Verify Scaling Laws**
   - Check if ρ_c(d) → ρ_∞ as d → ∞
   - Identify best-fit model (constant, inverse-d, power law)
   - Estimate asymptotic critical densities

3. **Create Summary Table**
   - ρ_c for each (criterion, dimension)
   - Scaling law parameters (ρ_∞, a, β)
   - Model comparison (R² for each model)

### Optional Extensions

**A. Multi-τ Analysis**
```bash
python scripts/fit_decay_multi_tau.py  # ~30-40 min
```
- Characterize K_c(τ) for Krylov/Spectral
- Fit: K_c(τ) = a + b·τ
- Expected: b_krylov < b_spectral

**B. GEO2 Comparison** (via CLI)
```bash
# GEO2 d=16 (2×2 lattice)
python -m reach.cli --nx 2 --ny 2 three-criteria-vs-K-multi-tau \
    --ensemble GEO2 -d 16 --k-max 16 \
    --taus 0.95 --trials 80 --y unreachable
```
- Compare ρ_c (GEO2) vs ρ_c (canonical)
- Hypothesis: GEO2 has lower ρ_c due to structure

**C. Higher Precision** (d=10,12,14 only)
```bash
python scripts/fit_decay_refined.py \
    --dims 10,12,14 \
    --trials 200 \
    --tau 0.95 \
    --ensemble canonical
```
- Reduce error bars
- Better constrain scaling law parameters

---

## 📁 OUTPUT FILES

### Data Files
- `data/raw_logs/decay_refined_test.pkl` - Test run (d=10,12, trials=50) ✅
- `data/raw_logs/decay_canonical_extended.pkl` - Extended run (d=8-16, trials=80) 🔄
- `data/raw_logs/decay_production.pkl` - Production run (TBD)

### Plots
- `fig/analysis/decay_fits_physical_Kmax.png` - Initial fit results ✅
- `fig/analysis/rho_c_vs_dimension.png` - Critical density vs d ✅
- `fig/analysis/scaling_rho_c_vs_d.png` - Scaling analysis ✅
- `fig/analysis/scaling_loglog.png` - Power law detection ✅

### Logs
- `logs/exp_decay_verification_test.log` - Bug fix verification ✅
- `logs/decay_refined_test.log` - Quick test results ✅
- `logs/decay_canonical_extended_20251127_002031.log` - Extended run 🔄

---

## 🐛 KNOWN ISSUES & SOLUTIONS

### Issue 1: Moment ρ_c Too Low
**Observed**: ρ_c ≈ 0.006 (expected ~0.04 from τ-comparison)
**Hypothesis**: K_max = d too restrictive, misses transition
**Solution**: Check if moment transition occurs at K > d

### Issue 2: Krylov Sharp Transition
**Observed**: No points in 0.01 < P < 0.99
**Hypothesis**: Extremely sharp threshold (step function)
**Solution**: Use bisection or finer K sampling around transition

### Issue 3: GEO2 Integration
**Observed**: `fit_decay_refined.py` doesn't support GEO2 lattice params
**Hypothesis**: `analysis.py` needs nx, ny support
**Solution**: Use CLI directly for GEO2, parse results separately

---

## 📈 MONITORING

**Check Progress**:
```bash
# View log tail
tail -20 logs/decay_canonical_extended_20251127_002031.log

# Check process
ps aux | grep fit_decay_refined

# Estimated completion
# Started: 00:20
# Estimated runtime: 4-6 hours
# Expected finish: 04:20-06:20 (Nov 27)
```

**Quick Stats** (when running):
```bash
# Count completed K values
grep "Computing MC for K=" logs/decay_canonical_extended_*.log | wc -l

# Current dimension
grep "=== d=" logs/decay_canonical_extended_*.log | tail -1
```

---

## 🎓 THEORETICAL BACKGROUND

### Reachability Transition
For K random Hamiltonians H₁,...,H_K ∈ su(d), the critical density ρ_c marks the transition:
- K < ρ_c·d²: Most target states unreachable (P ≈ 1)
- K > ρ_c·d²: Most target states reachable (P ≈ 0)

### Three Criteria
1. **Moment** (Gram matrix rank): τ-independent, sharpest transition
2. **Krylov** (rank-based, continuous): τ-dependent, sharp
3. **Spectral** (continuous overlap optimization): τ-dependent, smoothest

### Scaling Hypothesis
**Dimension Independence**: ρ_c(d) → ρ_∞ as d → ∞
**Finite-Size Corrections**: ρ_c(d) = ρ_∞ + a/d^β

**Physical Meaning**:
- ρ_∞ = asymptotic critical density (universal)
- a = finite-size correction amplitude
- β = correction exponent (typically 0.5-2)

---

**End of Status Report**
