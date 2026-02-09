# Data Provenance & Access Patterns

**Last Updated:** 2025-12-16
**Status:** ✅ All data collected, merged, and validated

This document provides complete provenance information for all raw data files used in the reachability analysis, including dates, source scripts, data structures, and proper access patterns.

---

## Executive Summary

**Current Data Status (2025-12-16):**
- ✅ **Moment**: Complete coverage (ρ ∈ [0.002, 0.200]) via OLD comprehensive + NEW extension
- ✅ **Spectral**: Complete coverage (ρ ∈ [0.010, 0.200]) via FIXED + EXTENSION merge
- ✅ **Krylov**: Complete coverage (ρ ∈ [0.020, 0.200] for d=10; partial for d>10)
- ✅ **Publication Figures**: v3 figures regenerated with merged data (2025-12-16)

**Key Files for Analysis:**
- Moment: `comprehensive_reachability_20251209_153938.pkl` + `moment_extension_all_dims_20251215_160333.pkl`
- Spectral: `spectral_complete_merged_20251216_153002.pkl` ⭐ **USE THIS**
- Krylov: `krylov_spectral_canonical_20251215_154634.pkl` + `krylov_dense_20251216_112335.pkl`

---

## Table of Contents

1. [Data Files Inventory](#data-files-inventory)
2. [Data Structures & Access Patterns](#data-structures--access-patterns)
3. [Merging Strategies](#merging-strategies)
4. [Critical Bugs & Fixes](#critical-bugs--fixes)
5. [How to Load Data Correctly](#how-to-load-data-correctly)
6. [Coverage Maps](#coverage-maps)

---

## Data Files Inventory

### Moment Criterion

| File | Date | Points/Dim | ρ Range | Source Script | Notes |
|------|------|-----------|---------|---------------|-------|
| `comprehensive_reachability_20251209_153938.pkl` | 2025-12-09 | 36-38 | 0.002-0.150 | (original comprehensive experiment) | **OLD** - Contains transition region |
| `moment_extension_all_dims_20251215_160333.pkl` | 2025-12-15 | 5-96 | 0.060-0.200 | `run_moment_extension_all_dims.py` | **NEW** - High ρ extension, mostly saturated |

**Data Structure:**
- **OLD comprehensive**: `results['moment'][d]` → `{'K': [], 'P': [], 'sem': []}`
- **NEW extension**: `results[d]` → `{'K': [], 'P': [], 'sem': []}` (NO nested 'moment' key!)

**⚠️ Critical Bug:** Many scripts incorrectly accessed OLD data as `results[d]['moment']` instead of `results['moment'][d]`, causing them to skip the transition region entirely. **Fixed in:** `add_error_bars_to_linearized.py:43`, `generate_publication_figures_final.py:55`

### Spectral Criterion

| File | Date | Points/Dim | ρ Range | Source Script | Notes |
|------|------|-----------|---------|---------------|-------|
| `krylov_spectral_canonical_20251215_154634.pkl` | 2025-12-15 | 12-29 | 0.010-0.133 (d>10) | `run_krylov_spectral_canonical.py` | **FIXED** - Ground state bug corrected |
| `spectral_extension_20251216_134457.pkl` | 2025-12-16 | 6-11 | 0.077-0.195 | `run_spectral_extension.py` | **EXTENSION** - Extends to ρ=0.2 |
| `spectral_complete_merged_20251216_153002.pkl` | 2025-12-16 | 12-40 | 0.010-0.200 | `merge_spectral_extension.py` | ⭐ **USE THIS** - Complete merged data |

**Data Structure:**
- **FIXED/EXTENSION**: `spectral[d]` → `{'K': [], 'P': [], 'sem': [], 'S_values': [], 'rho': []}`
- **MERGED**: Same structure as above, combines FIXED + EXTENSION intelligently

**Ground State Bug:** Original experiments used random initial states instead of `fock_state(d, 0)`. Fixed in Dec 2025. All "FIXED" and subsequent files use correct ground state.

### Krylov Criterion

| File | Date | Points/Dim | ρ Range | Source Script | Notes |
|------|------|-----------|---------|---------------|-------|
| `krylov_spectral_canonical_20251215_154634.pkl` | 2025-12-15 | 12-29 | 0.010-0.133 (d>10) | `run_krylov_spectral_canonical.py` | **FIXED** - Contains both Spectral and Krylov |
| `krylov_dense_20251216_112335.pkl` | 2025-12-16 | 5-20 | Transition region | `run_krylov_dense_experiment.py` | **DENSE** - High sampling near K_c |

**Data Structure:**
- **FIXED**: `results[d]['krylov']` → `{'K': [], 'P': [], 'sem': [], 'R_values': []}`
- **DENSE**: `krylov[d]` → `{'K': [], 'P': [], 'sem': []}`

**Note:** The FIXED file contains BOTH Spectral and Krylov results in a single file.

---

## Data Structures & Access Patterns

### Correct Access Patterns

```python
import pickle
import numpy as np

# ======================================================================
# MOMENT DATA
# ======================================================================

# OLD comprehensive (transition region)
with open('data/raw_logs/comprehensive_reachability_20251209_153938.pkl', 'rb') as f:
    old_data = pickle.load(f)

for d in [10, 14, 18, 22, 26]:
    if d in old_data['results']['moment']:  # ✅ CORRECT
        mom = old_data['results']['moment'][d]
        K = np.array(mom['K'])
        P = np.array(mom['P'])
        sem = np.array(mom['sem'])

# ❌ WRONG: old_data['results'][d]['moment']  # This skips the data!

# NEW extension (high ρ)
with open('data/raw_logs/moment_extension_all_dims_20251215_160333.pkl', 'rb') as f:
    ext_data = pickle.load(f)

for d in [10, 14, 18, 22, 26]:
    if d in ext_data['results']:  # ✅ CORRECT
        mom_ext = ext_data['results'][d]  # No nested 'moment' key!
        K = np.array(mom_ext['K'])
        P = np.array(mom_ext['P'])
        sem = np.array(mom_ext['sem'])

# ======================================================================
# SPECTRAL DATA (USE MERGED FILE!)
# ======================================================================

# ⭐ RECOMMENDED: Use the complete merged file
with open('data/raw_logs/spectral_complete_merged_20251216_153002.pkl', 'rb') as f:
    spectral_data = pickle.load(f)

for d in [10, 14, 18, 22, 26]:
    if d in spectral_data['spectral']:  # ✅ CORRECT
        spec = spectral_data['spectral'][d]
        K = np.array(spec['K'])
        P = np.array(spec['P'])
        sem = np.array(spec['sem'])
        S_values = np.array(spec['S_values'])  # Optional
        rho = K / d**2

# ======================================================================
# KRYLOV DATA
# ======================================================================

# FIXED Krylov (full range)
with open('data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl', 'rb') as f:
    fixed_data = pickle.load(f)

for d in [10, 14, 18, 22, 26]:
    if d in fixed_data['results']:  # ✅ CORRECT
        K = np.array(fixed_data['results'][d]['K'])
        kryl = fixed_data['results'][d]['krylov']
        P = np.array(kryl['P'])
        sem = np.array(kryl['sem'])

# DENSE Krylov (transition detail)
with open('data/raw_logs/krylov_dense_20251216_112335.pkl', 'rb') as f:
    dense_data = pickle.load(f)

for d in [10, 14, 18, 22, 26]:
    if d in dense_data['krylov']:  # ✅ CORRECT
        K = np.array(dense_data['krylov'][d]['K'])
        P = np.array(dense_data['krylov'][d]['P'])
        sem = np.array(dense_data['krylov'][d]['sem'])
```

---

## Merging Strategies

### Moment: OLD Comprehensive + NEW Extension

**Goal:** Combine transition region (OLD) with high-ρ saturation region (NEW)

**Strategy:**
1. Load both OLD comprehensive and NEW extension
2. For each dimension d:
   - Extract K, P, sem from OLD comprehensive (`results['moment'][d]`)
   - Extract K, P, sem from NEW extension (`results[d]`)
   - Remove duplicates (keep OLD values if K appears in both)
   - Concatenate arrays
   - Sort by K
3. Result: Complete ρ coverage from ~0.002 to 0.200

**Reference Implementation:** `scripts/add_error_bars_to_linearized.py` lines 38-75

### Spectral: FIXED + EXTENSION

**Goal:** Extend coverage from ρ~0.13 to ρ~0.20 for dimensions d>10

**Strategy:**
1. Load FIXED data (ground state corrected, ρ up to ~0.13 for d>10)
2. Load EXTENSION data (new points from ρ~0.08 to ρ~0.20)
3. For each dimension d:
   - Extract K values from both sources
   - Identify new K values in EXTENSION not present in FIXED
   - Merge data arrays, avoiding duplicates
   - Sort by K
4. Save merged result to `spectral_complete_merged_TIMESTAMP.pkl`

**Reference Implementation:** `scripts/merge_spectral_extension.py`

**Result:** `spectral_complete_merged_20251216_153002.pkl` - **USE THIS FILE**

### Krylov: FIXED + DENSE

**Goal:** Combine broad coverage (FIXED) with high-resolution transition sampling (DENSE)

**Strategy:**
1. Load FIXED Krylov (full ρ range, coarse sampling)
2. Load DENSE Krylov (transition region only, fine sampling)
3. For each dimension d:
   - Start with all FIXED K values
   - Add DENSE K values that don't exist in FIXED
   - Concatenate arrays, avoiding duplicates
   - Sort by K
4. Result: Best of both worlds - full coverage + transition detail

**Reference Implementation:** `scripts/generate_publication_figures_final.py` lines 94-129

---

## Critical Bugs & Fixes

### Bug 1: Moment Data Access Pattern (FIXED 2025-12-16)

**Problem:** Incorrect data structure access caused scripts to skip the OLD comprehensive Moment data (transition region) and only load the NEW extension (mostly saturated, P≈0).

**Symptoms:**
- Linearized Moment plots showing ρ range 0.06-0.17 instead of 0.02-0.15
- Only 5-96 points per dimension instead of 41-134
- Missing transition region entirely

**Root Cause:**
```python
# ❌ WRONG (was used in many scripts)
if d in comprehensive['results']:
    mom = comprehensive['results'][d]['moment']  # Fails silently!

# ✅ CORRECT
if d in comprehensive['results']['moment']:
    mom = comprehensive['results']['moment'][d]
```

**Files Fixed:**
- `scripts/add_error_bars_to_linearized.py:43`
- `scripts/generate_publication_figures_final.py:55` (already correct)

### Bug 2: Ground State Initialization (FIXED 2025-12-15)

**Problem:** Original Spectral and Krylov experiments used random initial states instead of the canonical ground state `|0⟩ = |00...0⟩`.

**Impact:** Biased S* and R* values, incorrect phase transition measurements.

**Fix:** All experiments after 2025-12-15 use `fock_state(d, 0)` as initial state.

**Files Affected:**
- All files dated before 2025-12-15: ❌ **DO NOT USE**
- All files with "FIXED" or dated >= 2025-12-15: ✅ **CORRECT**

---

## How to Load Data Correctly

### Recommended Loading Function

```python
def load_complete_data():
    """
    Load all merged reachability data (recommended approach).

    Returns:
        dict: {'moment': {d: {...}}, 'spectral': {d: {...}}, 'krylov': {d: {...}}}
    """
    import pickle
    import numpy as np

    # 1. Load OLD Moment comprehensive (transition region)
    with open('data/raw_logs/comprehensive_reachability_20251209_153938.pkl', 'rb') as f:
        old_data = pickle.load(f)

    # 2. Load NEW Moment extension (high ρ)
    with open('data/raw_logs/moment_extension_all_dims_20251215_160333.pkl', 'rb') as f:
        moment_ext = pickle.load(f)

    # 3. Load MERGED Spectral data (FIXED + EXTENSION)
    with open('data/raw_logs/spectral_complete_merged_20251216_153002.pkl', 'rb') as f:
        merged_spectral = pickle.load(f)

    # 4. Load FIXED Krylov data (for merging)
    with open('data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl', 'rb') as f:
        fixed_krylov = pickle.load(f)

    # 5. Load DENSE Krylov data (transition detail)
    with open('data/raw_logs/krylov_dense_20251216_112335.pkl', 'rb') as f:
        dense_krylov = pickle.load(f)

    merged = {'moment': {}, 'spectral': {}, 'krylov': {}}

    # === MOMENT: Merge OLD + NEW ===
    for d in [10, 14, 18, 22, 26]:
        K_list, P_list, sem_list = [], [], []

        # OLD comprehensive (transition region)
        if d in old_data['results']['moment']:  # ✅ CORRECT ACCESS
            mom = old_data['results']['moment'][d]
            K_list.extend(mom['K'])
            P_list.extend(mom['P'])
            sem_list.extend(mom['sem'])

        # NEW extension (high ρ)
        if d in moment_ext['results']:  # ✅ CORRECT ACCESS
            mom_ext = moment_ext['results'][d]  # No nested 'moment' key!
            for i, k in enumerate(mom_ext['K']):
                if k not in K_list:  # Avoid duplicates
                    K_list.append(k)
                    P_list.append(mom_ext['P'][i])
                    sem_list.append(mom_ext['sem'][i])

        if K_list:
            K_arr = np.array(K_list)
            P_arr = np.array(P_list)
            sem_arr = np.array(sem_list)
            sort_idx = np.argsort(K_arr)
            merged['moment'][d] = {
                'K': K_arr[sort_idx],
                'P': P_arr[sort_idx],
                'sem': sem_arr[sort_idx],
                'rho': K_arr[sort_idx] / d**2
            }

    # === SPECTRAL: Use MERGED data (already merged!) ===
    for d in [10, 14, 18, 22, 26]:
        if d in merged_spectral['spectral']:
            merged['spectral'][d] = merged_spectral['spectral'][d]

    # === KRYLOV: Merge FIXED + DENSE ===
    for d in [10, 14, 18, 22, 26]:
        K_list, P_list, sem_list = [], [], []

        # FIXED (full range)
        if d in fixed_krylov['results']:
            K_fixed = np.array(fixed_krylov['results'][d]['K'])
            kryl = fixed_krylov['results'][d]['krylov']
            K_list.extend(K_fixed.tolist())
            P_list.extend(kryl['P'])
            sem_list.extend(kryl['sem'])

        # DENSE (transition detail)
        if d in dense_krylov['krylov']:
            for i, k in enumerate(dense_krylov['krylov'][d]['K']):
                if k not in K_list:  # Avoid duplicates
                    K_list.append(k)
                    P_list.append(dense_krylov['krylov'][d]['P'][i])
                    sem_list.append(dense_krylov['krylov'][d]['sem'][i])

        if K_list:
            K_arr = np.array(K_list)
            P_arr = np.array(P_list)
            sem_arr = np.array(sem_list)
            sort_idx = np.argsort(K_arr)
            merged['krylov'][d] = {
                'K': K_arr[sort_idx],
                'P': P_arr[sort_idx],
                'sem': sem_arr[sort_idx],
                'rho': K_arr[sort_idx] / d**2
            }

    return merged
```

**Usage:**
```python
data = load_complete_data()

# Access Moment data for d=14
moment_14 = data['moment'][14]
print(f"Moment d=14: {len(moment_14['K'])} points, ρ ∈ [{moment_14['rho'].min():.4f}, {moment_14['rho'].max():.4f}]")

# Access Spectral data for d=22
spectral_22 = data['spectral'][22]
print(f"Spectral d=22: {len(spectral_22['K'])} points, ρ ∈ [{spectral_22['rho'].min():.4f}, {spectral_22['rho'].max():.4f}]")
```

---

## Coverage Maps

### Final Data Coverage (Post-Merge, 2025-12-16)

| Criterion | d=10 | d=14 | d=18 | d=22 | d=26 |
|-----------|------|------|------|------|------|
| **MOMENT** | 41 pts<br>ρ: 0.020-0.200 | 54 pts<br>ρ: 0.010-0.199 | 73 pts<br>ρ: 0.006-0.198 | 100 pts<br>ρ: 0.004-0.198 | 134 pts<br>ρ: 0.003-0.200 |
| **SPECTRAL** | 12 pts<br>ρ: 0.020-0.200 | 22 pts<br>ρ: 0.010-0.194 | 28 pts<br>ρ: 0.006-0.198 | 34 pts<br>ρ: 0.004-0.190 | 40 pts<br>ρ: 0.003-0.195 |
| **KRYLOV** | 13 pts<br>ρ: 0.020-0.200 | 17 pts<br>ρ: 0.010-0.133 | 21 pts<br>ρ: 0.006-0.105 | 25 pts<br>ρ: 0.004-0.083 | 30 pts<br>ρ: 0.003-0.071 |

**Key Observations:**
- ✅ Moment: Full coverage to ρ=0.2 for all dimensions
- ✅ Spectral: Full coverage to ρ~0.19-0.20 for all dimensions (achieved via FIXED + EXTENSION merge)
- ⚠️ Krylov: Full coverage only for d=10; limited to ρ~0.07-0.13 for d>10

---

## Publication Figures Status

**Current Version:** v7 dual-version (regenerated 2024-12-24)

**Data Sources Used in v7:**
- Moment: OLD comprehensive + NEW extension (merged)
- Spectral: **MERGED FIXED + EXTENSION** (complete coverage to ρ=0.2)
- Krylov: FIXED + DENSE (merged)

**Generated Figures:**
1. `final_summary_3panel_v7_exp.png` (749 KB) - 3-panel decay (exp version)
2. `final_summary_3panel_v7_pow2.png` (750 KB) - 3-panel decay (pow2 version)
3. `combined_criteria_d26_v7_exp.png` (280 KB) - d=26 comparison (exp)
4. `combined_criteria_d26_v7_pow2.png` (283 KB) - d=26 comparison (pow2)
5. `Kc_vs_d_v7_exp.png` (252 KB) - K_c scaling (exp)
6. `Kc_vs_d_v7_pow2.png` (244 KB) - K_c scaling (pow2)
7. `linearized_fits_v7_exp.png` (578 KB) - Linearized fits ln(P)
8. `linearized_fits_v7_pow2.png` (581 KB) - Linearized fits log₂(P)

**Generation Script:** `scripts/generate_publication_dual_versions.py`

**Fit Functions:**

**Moment (exp version):**
- **Formula**: P(ρ) = exp(-α d² (ρ - ρ_c))
- **Critical K**: K_c = d² ρ_c + ln(2)/α
- **Linearization**: ln(P) = -α d² (ρ - ρ_c)

**Moment (pow2 version):**
- **Formula**: P(ρ) = 2^(-ρ/ρ_c)
- **Critical K**: K_c = d² ρ_c
- **Linearization**: log₂(P) = -ρ/ρ_c

**Spectral/Krylov (both versions):**
- **Formula**: P(ρ) = 1/(1 + exp((ρ - ρ_c)/Δ))
- **Critical K**: K_c = d² ρ_c

---

## Quick Reference Card

**Need to load data for analysis? Use this:**

```python
# ⭐ SPECTRAL: Always use the merged file
with open('data/raw_logs/spectral_complete_merged_20251216_153002.pkl', 'rb') as f:
    spectral = pickle.load(f)['spectral']

# ⭐ MOMENT: Merge OLD + NEW
with open('data/raw_logs/comprehensive_reachability_20251209_153938.pkl', 'rb') as f:
    moment_old = pickle.load(f)['results']['moment']  # ✅ CORRECT!

with open('data/raw_logs/moment_extension_all_dims_20251215_160333.pkl', 'rb') as f:
    moment_new = pickle.load(f)['results']  # No nested 'moment' key!

# Then merge as shown in load_complete_data()

# ⭐ KRYLOV: Merge FIXED + DENSE
with open('data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl', 'rb') as f:
    krylov_fixed = pickle.load(f)['results']

with open('data/raw_logs/krylov_dense_20251216_112335.pkl', 'rb') as f:
    krylov_dense = pickle.load(f)['krylov']

# Then merge as shown in load_complete_data()
```

**Common Pitfalls:**
1. ❌ Accessing Moment OLD as `results[d]['moment']` → **Use `results['moment'][d]`**
2. ❌ Using separate FIXED/EXTENSION Spectral files → **Use merged file**
3. ❌ Using pre-2025-12-15 files → **Use only FIXED or later files**

---

**For Questions:** See `scripts/add_error_bars_to_linearized.py` or `scripts/generate_publication_figures_final.py` for reference implementations.

**Last Experiment:** Spectral extension completed 2025-12-16 15:24:09
**Last Figure Generation:** 2025-12-16 (v3 with merged Spectral data)
