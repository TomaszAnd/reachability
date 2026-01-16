# GEO2LOCAL Optimization Investigation Report

**Date**: 2024-12-24
**Issue**: "Optimized" vs "Gaussian" GEO2 results are nearly identical
**Status**: ROOT CAUSE IDENTIFIED

## Summary

The "optimized weights" and "fixed weights" GEO2 experiments produce identical results because **both use the exact same code path**. The `geo2_optimize_weights` flag passed by `run_geo2_comprehensive_optimized.py` is **silently ignored** by the analysis code.

## Critical Findings

### 1. Missing Parameter
File: `scripts/run_geo2_comprehensive_optimized.py:88`
```python
geo2_optimize_weights=True  # KEY DIFFERENCE: Optimized weights!
```

**Problem**: This parameter doesn't exist in `analysis.monte_carlo_unreachability_vs_density()`. Python silently accepts it via `**ensemble_params` and then ignores it.

### 2. Identical Code Paths
File: `reach/analysis.py:1698-1756`

Both approaches execute:
```python
# Generate random Hamiltonian ensemble (IDENTICAL for both)
hams = models.random_hamiltonian_ensemble(
    d, K, ensemble, seed=rng.randint(0, 2**31 - 1), **ensemble_params
)

# BOTH approaches optimize over λ
result = optimize.maximize_spectral_overlap(psi, phi, hams, ...)  # Line 1748
krylov_result = optimize.maximize_krylov_score(psi, phi, hams, ...)  # Line 1735
```

### 3. What GEO2 Actually Does

**Current behavior** (for BOTH "optimized" and "fixed"):
1. Sample K operators from P_2(G) basis (size L) **without replacement**
2. Weight each operator: g_a ~ N(0,1)
3. Form Hamiltonian set: H = {H_1, ..., H_K}
4. **Optimize λ** to maximize S* = max_λ |⟨ψ|U(λ)|φ⟩|²
5. **Optimize λ** to maximize R* = max_λ ‖P_Km(H(λ))|φ⟩‖²

**What "optimized" SHOULD mean** (not implemented):
- Find optimal weights g_a to maximize reachability
- OR: Choose optimal K operators from P_2(G) basis
- OR: Something fundamentally different from random Gaussian weights

## Why Results Look Identical

Both experiments:
- Use same Hamiltonian generation (`models.random_hamiltonian_ensemble`)
- Use same optimization routines (`maximize_spectral_overlap`, `maximize_krylov_score`)
- Differ only in filename labels

Observed ρ_c values:
- **Optimized** d=16: ρ_c=0.0402, K_c=10.3
- **Fixed** d=16: ρ_c ~0.04, K_c ~10  (nearly identical!)

## Conceptual Confusion

The term "optimization" has **two meanings** in this codebase:

1. **Parameter optimization** (what we do): Find λ ∈ ℝ^K to maximize S* or R*
   - This is ALWAYS done for Spectral and Krylov criteria
   - GUE, GOE, and GEO2 all use this

2. **Ensemble optimization** (what we thought "optimized" meant): Choose better Hamiltonians
   - NOT implemented for GEO2
   - Would require optimizing basis selection or Gaussian weights g_a

## Comparison with Canonical Ensemble

**Canonical (GUE/GOE)**:
- Basis: ALL d×d Hermitian matrices
- Weights: g_a ~ N(0,1) for GOE, g_a ~ N(0,1) + iN(0,1) for GUE
- Total ensemble size: infinite
- Sample K operators uniformly

**GEO2**:
- Basis: P_2(G) = {1-local + 2-local Pauli on lattice}
- Weights: g_a ~ N(0,1)
- Total ensemble size: L = 3n + 9|E| (finite!)
- Sample K operators from fixed basis

**Key insight**: For Canonical, the ensemble is so large that "random sample" ≈ "optimized sample". For GEO2, the basis is MUCH smaller (L=48 for 2×2), so choosing good operators SHOULD matter more!

## Recommendations

### Option A: Fix the "Fixed Weights" Approach

Make "fixed weights" truly use **fixed random λ**, not optimized λ:

```python
# In analysis.py, add parameter:
def monte_carlo_unreachability_vs_density(..., optimize_lambda=True, ...):

    if optimize_lambda:
        # Current behavior
        result = optimize.maximize_spectral_overlap(psi, phi, hams, ...)
    else:
        # NEW: Use fixed random λ
        lambda_random = rng.randn(K) / np.sqrt(K)
        H_combined = sum(lam * H for lam, H in zip(lambda_random, hams))
        spectral_overlap = |⟨ψ|U(H_combined)|φ⟩|²
```

This would show:
- **Optimized** (current): max_λ S*  (should perform BETTER)
- **Fixed**: S* at random λ  (should perform WORSE, higher K_c needed)

### Option B: Implement True Ensemble Optimization

Optimize the BASIS selection or weights g_a:

```python
def optimize_geo2_ensemble(nx, ny, K, target, method='L-BFGS-B'):
    """
    Find optimal K operators from P_2(G) and optimal weights g_a
    to maximize reachability for a given target state.
    """
    geo2 = GeometricTwoLocal(nx, ny)
    L = geo2.L  # Total basis size

    # Optimization variables:
    # - Which K operators to use (combinatorial)
    # - Weights g_a for each operator

    # This is a hard discrete+continuous optimization problem!
    ...
```

This is much more complex and computationally expensive.

### Option C: Accept Current Results as Intrinsic GEO2 Properties

Relabel figures:
- "Optimized" → "GEO2 (optimized λ)"
- "Fixed" → "GEO2 (optimized λ, different seed)"

And acknowledge that:
- GEO2 intrinsically performs similarly regardless of basis sampling
- The sparse geometric structure dominates the results
- Optimization over λ is already being done (as intended)

## Recommended Next Steps

1. **Immediate**: Relabel existing figures to avoid confusion
   - `geo2_optimized_summary.png` → `geo2_lambda_opt_seed1_summary.png`
   - `geo2_fixed_summary.png` → `geo2_lambda_opt_seed2_summary.png`

2. **Short-term**: Implement Option A to create true "fixed λ" comparison
   - Add `optimize_lambda` parameter to analysis functions
   - Run experiments with `optimize_lambda=False`
   - Compare optimized vs non-optimized λ (should show clear difference)

3. **Long-term**: Consider if Option B is scientifically interesting
   - Is ensemble optimization meaningful for GEO2?
   - Or is the geometric structure already optimal?

## Files to Update

If proceeding with Option A:

1. `reach/analysis.py`:
   - Add `optimize_lambda=True` parameter to `monte_carlo_unreachability_vs_density()`
   - Add conditional logic to skip optimization when `optimize_lambda=False`

2. `scripts/run_geo2_comprehensive_*.py`:
   - Update to pass `optimize_lambda` parameter
   - `optimized.py` → keep `optimize_lambda=True` (current behavior)
   - `fixed.py` → add `optimize_lambda=False` (NEW behavior)

3. Documentation:
   - Clarify what "optimization" means in different contexts
   - Explain the difference between λ-optimization and ensemble-optimization

## Conclusion

The "bug" is actually a **missing feature**: there is no true "fixed weights" implementation that skips λ-optimization. Both experiments optimize λ, producing identical results.

Recommendation: **Implement Option A** to create meaningful comparison between:
- GEO2 with optimized λ (current)
- GEO2 with random fixed λ (new)

This will show whether optimization over the geometric basis parameters λ actually helps, or if the sparse structure alone determines reachability.
