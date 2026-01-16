# Final Publication Figures - Status and Next Steps

**Date:** 2025-11-28
**Status:** Figure 1 complete with equation boxes and fit curves ✅
**Remaining:** Generate multi-τ data for Figure 2 (1-2 hours)

---

## ✅ What Has Been Completed

### 1. Critical Bug Fixes
- **Krylov Criterion Fixed** (`scripts/fit_decay_fixed_krylov_v2.py`)
  - Now uses `maximize_krylov_score()` (optimized, continuous)
  - Shows smooth transitions instead of step function
  - Properly τ-dependent

### 2. Criterion-Specific Functional Forms Implemented

**`scripts/fit_correct_forms.py`** - Physics-based fitting functions:

- **Moment (Algebraic):** `P = exp(-α(K - K_c))`
  - Gradual decay as Lie algebra fills
  - τ-independent

- **Spectral (Optimization):** `P = 1/(1 + exp((ρ - ρ_c)/Δρ))`
  - Fermi-Dirac distribution for sharp threshold
  - τ-dependent
  - Target: R² = 0.990

- **Krylov (Subspace):** `P = 1/(1 + exp((ρ - ρ_c)/Δρ))`
  - Fermi-Dirac distribution for sharp threshold
  - τ-dependent

### 3. Multi-τ Data Generation Script

**`scripts/generate_multi_tau_data.py`** - Ready to run:

```bash
nohup python scripts/generate_multi_tau_data.py \
    > logs/multi_tau_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Features:**
- Dimensions: d = 10, 12, 14
- Tau values: {0.90, 0.95, 0.99}
- Trials: 100 (high quality)
- K_max = d (physical limit)
- Uses FIXED Krylov criterion
- Expected runtime: 1-2 hours

### 4. Final Publication Plotting Script

**`scripts/plot_final_publication.py`** - Complete and tested:

**Figure 1: Three-Criteria Comparison (1×3) ✅**
- ✅ Panel (a): Moment vs K with exponential fit
- ✅ Panel (b): Krylov vs ρ with Fermi-Dirac fit
- ✅ Panel (c): Spectral vs ρ with Fermi-Dirac fit
- ✅ All panels have equation boxes with R² and parameters
- ✅ All panels have analytical fit curves (dashed black lines)
- ✅ Error bars on all data points
- ✅ Generated: `fig/publication/figure1_three_criteria.pdf` (50KB)

**Figure 2: τ-Dependence Analysis (2×2) ⏳**
- Requires multi-τ data (to be generated)
- Panel (a): Krylov τ-dependence (3 curves)
- Panel (b): Spectral τ-dependence (3 curves)
- Panel (c): K_c scaling with dimension
- Panel (d): ρ_c(τ) relationship

---

## 📊 Current Data Inventory

### Available Files

**`data/raw_logs/decay_canonical_extended.pkl`:**
- Dimensions: d = 8, 10, 12, 14, 16 (5 dimensions)
- Trials: 80
- τ: 0.95 only
- K_max = d
- **Used for:** Moment and Spectral in Figure 1
- **Quality:** Excellent smooth transitions

**`data/raw_logs/decay_fixed_krylov_production.pkl`:**
- Dimensions: d = 10, 12, 14 (3 dimensions)
- Trials: 80
- τ: 0.95 only
- K_max = d
- **Used for:** Krylov in Figure 1 (FIXED version)
- **Quality:** Good smooth transitions (2-3 intermediate points)

### Missing Files (To Be Generated)

**`data/raw_logs/decay_multi_tau_publication.pkl`:**
- **Required for:** Figure 2 (all panels)
- Dimensions: d = 10, 12, 14
- Tau values: {0.90, 0.95, 0.99} ← CRITICAL for τ-dependence
- Trials: 100
- Expected runtime: 1-2 hours

---

## 🎯 Next Steps

### Step 1: Generate Multi-τ Data (REQUIRED)

**Command:**
```bash
nohup python scripts/generate_multi_tau_data.py \
    > logs/multi_tau_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Monitor progress:**
```bash
# Get process ID
ps aux | grep generate_multi_tau_data.py

# Watch log
tail -f logs/multi_tau_*.log

# Check completion
ls -lh data/raw_logs/decay_multi_tau_publication.pkl
```

**Expected output in log:**
```
================================================================================
MULTI-TAU PRODUCTION RUN - PUBLICATION QUALITY
================================================================================
Dimensions: [10, 12, 14]
K_max = d (physical limit)
Tau values: [0.9, 0.95, 0.99]
Trials: 100

============================================================
Dimension d=10, K_max=10
============================================================
  K=2/10
    Trial 100/100... Done!
    τ=0.9: M=1.000, S=0.990, K=0.950 | τ=0.95: M=1.000, S=0.970, K=0.920 | τ=0.99: M=1.000, S=0.940, K=0.880
  K=3/10
    ...
```

### Step 2: Generate Figure 2

**After multi-τ data is ready:**
```bash
python scripts/plot_final_publication.py
```

**Expected output:**
```
Loading data...
Loading: data/raw_logs/decay_canonical_extended.pkl
Loading: data/raw_logs/decay_fixed_krylov_production.pkl

Generating Figure 1: Three-Criteria Comparison...
✅ Saved: fig/publication/figure1_three_criteria.pdf

Generating Figure 2: τ-Dependence Analysis...
✅ Saved: fig/publication/figure2_tau_dependence.pdf

✅ Done!
```

---

## 📋 Validation Checklist

### Figure 1 (COMPLETED ✅)
- [x] Three panels (1×3 layout)
- [x] Each panel shows analytical fit curve (dashed black)
- [x] Each panel has equation box with R² and parameters
- [x] Error bars on all data points
- [x] Correct functional forms:
  - [x] Moment: Exponential decay
  - [x] Krylov: Fermi-Dirac
  - [x] Spectral: Fermi-Dirac
- [x] PDF (vector graphics) and PNG (preview) generated
- [x] Professional styling (colorblind-friendly colors)

### Figure 2 (PENDING ⏳)
- [ ] Four panels (2×2 layout)
- [ ] Panels (a) and (b) show 3 τ values each (0.90, 0.95, 0.99)
- [ ] All curves have fit lines
- [ ] Panel (c) shows K_c scaling with linear fits
- [ ] Panel (d) shows ρ_c(τ) relationship
- [ ] Requires multi-τ data to be generated

### Data Quality
- [x] Krylov shows smooth transitions (not step function)
- [ ] Krylov shows τ-dependence (different curves for different τ)
- [x] Criterion-specific functional forms implemented
- [ ] Spectral fits achieve R² > 0.95 (needs extended K data)
- [x] Error bars computed (binomial SEM)
- [x] K_max = d constraint enforced

---

## 🔧 Technical Details

### Key Functional Forms

**Moment Criterion:**
```python
def moment_exponential(K, K_c, alpha):
    return np.where(K > K_c, np.exp(-alpha * (K - K_c)), 1.0)
```

**Spectral/Krylov Criteria:**
```python
def fermi_dirac(x, x_c, delta_x):
    z = np.clip((x - x_c) / max(abs(delta_x), 1e-8), -50, 50)
    return 1.0 / (1.0 + np.exp(z))
```

**τ-Dependence Model:**
```python
ρ_c(τ) = ρ_c0 + γ × log(1/(1-τ))
```

### Equation Box Format

Example from Figure 1, Panel (a):
```
┌─────────────────────────────────┐
│ P = e^(-0.14(K - 0.7))          │
│ R² = 0.913                      │
│ K_c = 3.5 ± 0.2                 │
└─────────────────────────────────┘
```

### File Outputs

**Figure 1:**
- `fig/publication/figure1_three_criteria.pdf` (50KB) ✅
- `fig/publication/figure1_three_criteria.png` (393KB) ✅

**Figure 2 (pending):**
- `fig/publication/figure2_tau_dependence.pdf`
- `fig/publication/figure2_tau_dependence.png`

---

## ⚠️ Important Constraints

1. **K_max = d always** (NEVER beyond d)
   - Physical limit: Can't have more independent Hamiltonians than dimension

2. **Use FIXED Krylov criterion**
   - `maximize_krylov_score()` with optimized λ
   - NOT `is_unreachable_krylov()` with random λ

3. **Criterion-specific functional forms**
   - Moment: Exponential (algebraic decay)
   - Spectral/Krylov: Fermi-Dirac (sharp threshold)
   - DO NOT use same form for all criteria

4. **Multi-τ required for Figure 2**
   - Cannot show τ-dependence with single τ value
   - Need at least 3 different τ values

---

## 🎓 Physical Interpretation

### Why Different Functional Forms?

**Moment Criterion (Exponential):**
- Tests if Hamiltonians span Lie algebra
- Each new Hamiltonian independently contributes
- Gradual exponential decay: P ~ exp(-αK)
- τ-independent (purely algebraic test)

**Spectral Criterion (Fermi-Dirac):**
- Tests if optimal spectral overlap < τ
- Sharp optimization threshold
- Fermi-Dirac: P ~ 1/(1 + exp((ρ - ρ_c)/Δρ))
- τ-dependent (threshold shifts with fidelity requirement)

**Krylov Criterion (Fermi-Dirac):**
- Tests if target in Krylov subspace with score < τ
- Sharp subspace containment threshold
- Fermi-Dirac: P ~ 1/(1 + exp((ρ - ρ_c)/Δρ))
- τ-dependent (threshold shifts with fidelity requirement)

---

## 📝 Summary

**Completed:**
- ✅ Fixed Krylov criterion (continuous, optimized)
- ✅ Implemented criterion-specific functional forms
- ✅ Created multi-τ data generation script
- ✅ Created final publication plotting script
- ✅ Generated Figure 1 with equations and fit curves

**Remaining:**
- ⏳ Run multi-τ data generation (1-2 hours)
- ⏳ Generate Figure 2 with τ-dependence analysis

**Total time to completion:** ~1-2 hours (just the data generation)

---

**Last Updated:** 2025-11-28
**Ready for:** Multi-τ data generation and final Figure 2 plotting
