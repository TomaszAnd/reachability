# Spectral Extension Experiment - Status & Next Steps

**Created:** 2025-12-16
**Status:** ✅ All three requested tasks completed
**Experiment:** 🔄 Spectral extension running (ETA: ~20-30 minutes remaining)

---

## Task Completion Summary

### ✅ TASK 1: Documentation Updated
**Files Created:**
- `PROJECT_CONTEXT.md` - Comprehensive AI assistant context (6.8 KB)
- `README_REACHABILITY.md` - Project overview and user documentation (5.1 KB)

**Content includes:**
- Complete project state as of Dec 16, 2025
- Critical bug fixes (ground state initialization)
- Data pipeline and merging strategies
- Publication figures (v3 is latest)
- Key results and scaling laws
- Coding patterns and common pitfalls

### ✅ TASK 2: Spectral Extension Launched
**Script:** `scripts/run_spectral_extension.py`
**PID:** 54498
**Log:** `logs/spectral_extension_20251216.log`

**Experiment Details:**
- **Purpose:** Extend Spectral criterion data to ρ=0.2 for dimensions d=14,18,22,26
- **Note:** d=10 already has full coverage to ρ=0.2
- **K values:** 34 total across 4 dimensions
- **Trials per K:** 50 (consistent with original experiments)
- **Threshold:** τ=0.99
- **Ensemble:** Canonical basis
- **Runtime:** ~50-60 minutes total (started 13:44:57)

**Progress (as of last check):**
```
✅ d=14: 6/6 K values complete (K=28-38)
🔄 d=18: 2/8 K values complete (K=36,40 done)
⏳ d=22: 0/9 K values (pending)
⏳ d=26: 0/11 K values (pending)

Overall: 8/34 K values (24%)
```

**Check Status:**
```bash
scripts/check_spectral_status.sh
```

### ✅ TASK 3: Error Bars Added to Linearized Fits
**Script:** `scripts/add_error_bars_to_linearized.py`
**Output:** `fig/publication/linearized_fits_physical_tau099_v4.png` (374 KB)

**Improvements:**
- Proper error propagation for logit transformation: `logit_sem = sem / (P*(1-P))`
- Error bars displayed with `ax.errorbar()` (capsize=2, elinewidth=1)
- Filters to transition region (0.01 < P < 0.99) to avoid log singularities
- Loads merged data: Moment (comprehensive + extension), Spectral (FIXED only), Krylov (FIXED + DENSE)

**Fixed Bug:**
- Corrected moment extension data access pattern (no nested 'moment' key)

---

## Current Experiment Status

**Running:** PID 54498
**CPU Time:** ~7-8 minutes
**Real Time:** ~15-20 minutes elapsed
**ETA:** ~20-30 minutes remaining

**Intermediate Output:**
- `data/raw_logs/spectral_extension_20251216_134457.pkl` (3.5 KB)
- Contains d=14 complete results
- Updates after each dimension completes

**Sampling Strategy:**
```python
EXTENSION_RANGES = {
    14: list(range(28, 40, 2)),    # 6 values: 28,30,32,34,36,38
    18: list(range(36, 66, 4)),    # 8 values: 36,40,44,48,52,56,60,64
    22: list(range(44, 98, 6)),    # 9 values: 44,50,56,62,68,74,80,86,92
    26: list(range(52, 136, 8)),   # 11 values: 52,60,68,76,84,92,100,108,116,124,132
}
```

**Monitor Progress:**
```bash
# Quick status check
scripts/check_spectral_status.sh

# Watch log in real-time
tail -f logs/spectral_extension_20251216.log

# Check process
ps -p 54498
```

---

## When Experiment Completes

### Step 1: Verify Completion
```bash
scripts/check_spectral_status.sh
```

Look for "✅ EXPERIMENT COMPLETE!" message.

### Step 2: Merge Spectral Data
**Script:** `scripts/merge_spectral_extension.py`

This script will:
1. Load original FIXED Spectral data (krylov_spectral_canonical_20251215_154634.pkl)
2. Load new Spectral extension data (spectral_extension_*.pkl)
3. Merge intelligently (avoid duplicates, sort by K)
4. Save complete merged dataset
5. Report coverage statistics

**Run:**
```bash
python scripts/merge_spectral_extension.py
```

**Expected Output:**
- `data/raw_logs/spectral_complete_merged_TIMESTAMP.pkl`
- Coverage summary showing all dimensions now extend to ρ≈0.2

### Step 3: Update Linearized Fits with Complete Data
**Update `scripts/add_error_bars_to_linearized.py`:**

Change the Spectral data loading from:
```python
# OLD - FIXED data only
with open('data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl', 'rb') as f:
    fixed_spectral = pickle.load(f)
```

To:
```python
# NEW - Complete merged data
with open('data/raw_logs/spectral_complete_merged_TIMESTAMP.pkl', 'rb') as f:
    merged_spectral = pickle.load(f)
```

Then regenerate the plot:
```bash
python scripts/add_error_bars_to_linearized.py
```

This will create a v5 plot with complete Spectral coverage to ρ=0.2.

### Step 4: Update Publication Figures (Optional)
If you want to regenerate all publication figures with the complete Spectral data:

**Update `scripts/generate_publication_figures_final.py`:**
- Change Spectral data source to merged file
- Regenerate all figures

**Run:**
```bash
python scripts/generate_publication_figures_final.py
```

---

## Data Coverage Summary

### Current Coverage (Pre-Extension)

**MOMENT:**
- d=10: 5 points, ρ ∈ [0.1600, 0.2000] ✓
- d=14: 18 points, ρ ∈ [0.1122, 0.1990] ✓
- d=18: 37 points, ρ ∈ [0.0864, 0.1975] ✓
- d=22: 63 points, ρ ∈ [0.0702, 0.1983] ✓
- d=26: 96 points, ρ ∈ [0.0592, 0.1997] ✓

**SPECTRAL (BEFORE extension):**
- d=10: 12 points, ρ ∈ [0.0200, 0.2000] ✓
- d=14: 16 points, ρ ∈ [0.0102, 0.1327] ← needs extension
- d=18: 20 points, ρ ∈ [0.0062, 0.1049] ← needs extension
- d=22: 25 points, ρ ∈ [0.0041, 0.0826] ← needs extension
- d=26: 29 points, ρ ∈ [0.0030, 0.0710] ← needs extension

**KRYLOV:**
- d=10: 13 points, ρ ∈ [0.0200, 0.2000] ✓
- d=14: 17 points, ρ ∈ [0.0102, 0.1327]
- d=18: 21 points, ρ ∈ [0.0062, 0.1049]
- d=22: 25 points, ρ ∈ [0.0041, 0.0826]
- d=26: 30 points, ρ ∈ [0.0030, 0.0710]

### Expected Coverage (Post-Extension)

**SPECTRAL (AFTER merge):**
- d=10: 12 points, ρ ∈ [0.0200, 0.2000] ✓ (no change)
- d=14: 22 points, ρ ∈ [0.0102, 0.1939] ✓ (+6 points)
- d=18: 28 points, ρ ∈ [0.0062, 0.1975] ✓ (+8 points)
- d=22: 34 points, ρ ∈ [0.0041, 0.1901] ✓ (+9 points)
- d=26: 40 points, ρ ∈ [0.0030, 0.1953] ✓ (+11 points)

---

## Key Results & Scaling Laws (τ=0.99)

From the comprehensive analysis:

**Critical K values (ρ_c = K_c/d²):**
- **Moment:** K_c ≈ 0.40d + 0.3 → ρ_c ≈ 0.40/d
- **Spectral:** K_c ≈ 1.95d - 5.9 → ρ_c ≈ 1.95
- **Krylov:** K_c ≈ 0.97d - 0.2 → ρ_c ≈ 0.97

**Physical Interpretation:**
- Moment criterion is most restrictive (lowest ρ_c)
- Spectral criterion is least restrictive (highest ρ_c)
- All three show power-law scaling with dimension

---

## Scripts Created/Modified

**New Scripts:**
1. `scripts/run_spectral_extension.py` - Extension experiment (running)
2. `scripts/merge_spectral_extension.py` - Data merging script (ready to run)
3. `scripts/add_error_bars_to_linearized.py` - Linearized fits with error bars (v4 complete)
4. `scripts/check_spectral_status.sh` - Quick status checker

**New Documentation:**
1. `PROJECT_CONTEXT.md` - Comprehensive AI context
2. `README_REACHABILITY.md` - Project overview
3. `SPECTRAL_EXTENSION_STATUS.md` - This file

---

## Quick Reference Commands

```bash
# Check experiment status
scripts/check_spectral_status.sh

# Monitor progress
tail -f logs/spectral_extension_20251216.log

# When complete: Merge data
python scripts/merge_spectral_extension.py

# Regenerate linearized fits (v5)
python scripts/add_error_bars_to_linearized.py

# Check process status
ps -p 54498

# View intermediate data
python -c "import pickle; d=pickle.load(open('data/raw_logs/spectral_extension_20251216_134457.pkl','rb')); print(d.keys())"
```

---

## Notes

- **Ground state initialization:** All Spectral/Krylov experiments use `fock_state(d, 0)` (NOT random states)
- **Ensemble:** Canonical basis (structured, sparse operators)
- **Seed offset:** Extension uses seed offset +99999 to avoid overlap with original experiments
- **Intermediate saving:** Results saved after each dimension completes
- **Error handling:** Optimization failures default to S=0.0 with warning

---

**Last Updated:** 2025-12-16 (during experiment run)
**Estimated Completion:** 2025-12-16 ~14:15-14:30
