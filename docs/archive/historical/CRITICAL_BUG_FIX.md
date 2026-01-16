# CRITICAL BUG FIX: Algorithm Inconsistency in optimize_lambda Modes

**Date**: 2025-12-29
**Severity**: CRITICAL
**Status**: FIXED

---

## Bug Description

The `optimize_lambda=True` and `optimize_lambda=False` modes in `monte_carlo_unreachability_vs_density()` were computing **completely different physical quantities** for the spectral criterion, making all comparisons meaningless.

### What Was Happening

**optimize_lambda=False** (lines 1771-1783):
```python
# BEFORE (WRONG):
lambda_fixed = rng.randn(K) / np.sqrt(K)
H_combined = sum(lam * H for lam, H in zip(lambda_fixed, hams))
U = (-1j * H_combined).expm()  # Matrix exponential
overlap_qobj = psi.dag() * U * phi
spectral_overlap_fixed = float(np.abs(overlap_val)**2)
```

This computed: **|⟨ψ|e^(-iH)|φ⟩|²** (time-evolved overlap at t=1)

**optimize_lambda=True** (lines 1759-1770):
```python
result = optimize.maximize_spectral_overlap(psi, phi, hams, ...)
# This calls mathematics.spectral_overlap() which uses:
eigenvalues, eigenvectors = eigendecompose(H_lambda)  # Eigendecomposition
return Σₙ |⟨n|φ⟩* ⟨n|ψ⟩|  # Spectral overlap
```

This computed: **Σₙ |⟨φₙ|ψₙ⟩|** (spectral overlap via eigendecomposition)

### Verification Results

Test case (d=16, K=5):
- **S_expm** (time evolution) = 0.10406826
- **S_eigh** (spectral decomposition) = 0.85425697
- **Difference**: 87.82% ❌

**These are COMPLETELY DIFFERENT quantities!**

---

## The Fix

Changed `optimize_lambda=False` to use the SAME formula as `optimize_lambda=True`:

**AFTER (CORRECT)** (lines 1771-1778):
```python
# Use fixed random λ ~ N(0, 1/√K) without optimization
# CRITICAL FIX: Use the SAME spectral_overlap formula as optimized mode
lambda_fixed = rng.randn(K) / np.sqrt(K)
spectral_overlap_fixed = mathematics.spectral_overlap(
    lambda_fixed, psi, phi, hams
)
spectral_best_values.append(spectral_overlap_fixed)
```

Now both modes compute: **Σₙ |⟨φₙ|ψₙ⟩|** via `mathematics.spectral_overlap()`

---

## Impact

### Before Fix
- ❌ Comparison between modes was **meaningless**
- ❌ Fixed mode computed time evolution overlap (wrong criterion)
- ❌ All GEO2 fixed λ results **INVALID**

### After Fix
- ✅ Both modes now use the SAME spectral overlap formula
- ✅ Fixed mode uses random λ ~ N(0, 1/√K)
- ✅ Optimized mode maximizes over λ ∈ [-1,1]^K
- ✅ Comparisons are now **meaningful**

---

## Files Modified

1. **reach/analysis.py** (lines 1771-1778)
   - Replaced `expm()` computation with `mathematics.spectral_overlap()`

2. **scripts/verify_algorithm_consistency.py** (new)
   - Verification script that exposed the bug

3. **scripts/test_fixed_algorithm.py** (new)
   - Integration test to verify the fix

---

## Lessons Learned

1. **Different algorithms for same criterion = bug**
   - Even if both are "spectral overlaps", they must use the SAME formula

2. **Time evolution ≠ Spectral decomposition**
   - |⟨ψ|e^(-iH)|φ⟩|² measures overlap after unitary evolution
   - Σₙ |⟨n|φ⟩* ⟨n|ψ⟩| measures overlap in eigenbasis
   - These are fundamentally different physical quantities!

3. **Test both code paths**
   - Conditional branches (`if optimize_lambda:`) can hide bugs
   - Need integration tests that exercise BOTH branches

---

## Verification

```bash
# Run verification (will show inconsistency):
python3 scripts/verify_algorithm_consistency.py

# Run integration test (confirms fix):
python3 scripts/test_fixed_algorithm.py
```

Expected: Both modes now produce consistent, comparable results.

---

**End of Bug Report**
