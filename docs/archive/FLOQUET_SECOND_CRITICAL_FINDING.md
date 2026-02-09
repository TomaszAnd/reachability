# Floquet Engineering - Second Critical Finding

**Date:** 2026-01-07
**Status:** Major discovery - hypothesis may be rejected OR implementation incomplete

---

## Executive Summary

**Original Hypothesis:** Floquet engineering requires fewer operators K to reach target states due to commutator-generated 3-body terms.

**Test Results:** **HYPOTHESIS APPEARS REJECTED** - Static Hamiltonians perform BETTER than Floquet!

**BUT:** Implementation flaw discovered - we're not optimizing λ properly for Floquet cases.

---

## Experimental Findings

### Test Setup
- **State pair:** |0000⟩ → GHZ
- **Driving:** Bichromatic (strongest H_F^(2) effects)
- **Method:** Optimize over both λ and t (not just fixed random λ)

### Results with Parameterized Optimization

| K | Static Fidelity | Floquet O1 | Floquet O2 | Winner |
|---|-----------------|------------|------------|--------|
| 12 | 0.881±0.031 | 0.682±0.116 | 0.551±0.072 | **Static** |
| 16 | 0.946±0.010 | 0.500±0.000 | 0.500±0.000 | **Static** |

**Key observations:**
1. ✅ **Parameterized optimization works!** Static goes from 0.500 → 0.946
2. ❌ **Floquet WORSE than static!** Opposite of hypothesis!
3. ⚠️ **Floquet stuck at K=16** while static reaches 94.6%

---

## The Implementation Flaw

### What We're Currently Doing (WRONG)

```python
# Optimize λ for static
fid_s, lambdas_s, t_s = optimize_fidelity_parameterized(psi, phi, hams, ...)

# Use SAME λ for Floquet (BAD!)
H_F1 = compute_floquet_hamiltonian(hams, lambdas_s, driving, period, order=1)
H_F2 = compute_floquet_hamiltonian(hams, lambdas_s, driving, period, order=2)

# Only optimize time
fid_f1, t_f1 = optimize_fidelity(psi, phi, H_F1, t_max)
fid_f2, t_f2 = optimize_fidelity(psi, phi, H_F2, t_max)
```

**Problem:** The optimal λ for static H might not be optimal for H_F!

### What We SHOULD Be Doing

```python
# Static: max_{λ,t} |⟨φ|exp(-i(Σλ_k H_k)t)|ψ⟩|²
fid_s, lambdas_s, t_s = optimize_fidelity_parameterized(psi, phi, hams, ...)

# Floquet O1: max_{λ,t} |⟨φ|exp(-iH_F^(1)(λ)t)|ψ⟩|²
fid_f1, lambdas_f1, t_f1 = optimize_floquet_fidelity_parameterized(
    psi, phi, hams, driving, period, order=1
)

# Floquet O2: max_{λ,t} |⟨φ|exp(-iH_F^(2)(λ)t)|ψ⟩|²
fid_f2, lambdas_f2, t_f2 = optimize_floquet_fidelity_parameterized(
    psi, phi, hams, driving, period, order=2
)
```

**Challenge:** Need to implement `optimize_floquet_fidelity_parameterized()` which optimizes over λ where H_F(λ) involves:
- Time-averaging: H_F^(1) = Σ λ_k λ̄_k H_k
- Commutators: H_F^(2) = Σ λ_j λ_k F_jk [H_j, H_k]

This is non-trivial because:
1. H_F depends on λ in complex way (quadratic terms, commutators)
2. Need to compute gradients w.r.t. λ for optimization
3. Significantly more computationally expensive

---

## Key Insights from Initial Tests

### Finding 1: Random λ is Fundamentally Broken

**Without optimizing λ:**
```
K=12: Fidelity = 0.500 (stuck at classical overlap)
K=16: Fidelity = 0.500 (stuck)
```

**With optimizing λ:**
```
K=12: Fidelity = 0.881 (significant improvement!)
K=16: Fidelity = 0.946 (approaching target!)
```

**Lesson:** MUST optimize coupling coefficients, not use random values!

### Finding 2: Static Outperforms Floquet (Current Implementation)

This is surprising and counter to the hypothesis, BUT:
- We're using λ optimized for static, not for Floquet
- Proper comparison requires optimizing λ separately for each case
- Current results may be misleading

### Finding 3: GHZ is Reachable with K=16

With optimized λ, static Hamiltonians reach 94.6% fidelity for |0000⟩ → GHZ using K=16 GEO2 operators. This establishes:
- GHZ IS reachable with 2-local geometric operators
- K≈16-20 likely sufficient to exceed 95%
- Optimization over λ is critical

---

## Three Possible Interpretations

### Interpretation A: Hypothesis is Correct (Implementation Incomplete)

**If** we implement proper λ optimization for Floquet:
- Floquet O2 might outperform static
- Commutators expand operator space as predicted
- Current results misleading due to suboptimal λ

**Evidence:**
- Diagnostics showed H_F^(2) generates 3-body terms ✓
- Only tested with λ from static optimization (suboptimal)
- Proper comparison not yet performed

**Next step:** Implement `optimize_floquet_fidelity_parameterized()`

### Interpretation B: Hypothesis is Rejected (Floquet Doesn't Help)

**If** even with optimal λ, Floquet performs worse:
- Commutator terms don't help for GEO2 → GHZ
- Static 2-local operators are already sufficient
- Floquet overhead (time-averaging) actually hinders

**Evidence:**
- Even with bichromatic driving, Floquet O2 stuck at 0.500
- Static clearly superior with current approach
- May be fundamental limitation

**Implication:** Floquet engineering not beneficial for this task

### Interpretation C: Wrong State Pair (Try Different Targets)

**Maybe** GHZ is not the right test case:
- GHZ might be equally hard for static and Floquet
- Other states (W, cluster) might show different behavior
- Need to test multiple state pairs

**Next step:** Test product_0 → W-state, cluster, etc.

---

## Implementation Requirements

### To Test Interpretation A (Proper Comparison)

Need to implement:

```python
def optimize_floquet_fidelity_parameterized(
    psi, phi, hamiltonians, driving_functions, period,
    t_max=10.0, order=2, n_trials=10, seed=None
):
    """
    Optimize over both λ and t for Floquet Hamiltonian.

    Maximize: |⟨φ|exp(-iH_F(λ)t)|ψ⟩|²

    where H_F(λ) = H_F^(1)(λ) + H_F^(2)(λ)
                 = Σ λ_k λ̄_k H_k + Σ λ_j λ_k F_jk [H_j, H_k]/(2i)
    """
    K = len(hamiltonians)

    def neg_fidelity(params):
        lambdas = params[:-1]
        t = params[-1]

        # Compute Floquet Hamiltonian with these λ
        H_F = compute_floquet_hamiltonian(
            hamiltonians, lambdas, driving_functions, period, order=order
        )

        # Compute fidelity
        U = scipy.linalg.expm(-1j * H_F * t)
        psi_evolved = U @ psi
        overlap = phi.conj() @ psi_evolved
        return -np.abs(overlap)**2

    # Optimize over λ and t
    # ... (similar to optimize_fidelity_parameterized)
```

**Complexity:**
- Each function evaluation requires computing H_F (Magnus expansion)
- Magnus involves K² commutators for order 2
- Significantly slower than static optimization

**Estimated runtime:** ~10× slower than current approach

### To Test Interpretation C (Different States)

Test multiple state pairs:

| Initial | Target | Expected Difficulty |
|---------|--------|---------------------|
| \|0000⟩ | GHZ | Hard (4-body entanglement) |
| \|0000⟩ | W-state | Medium (superposition) |
| \|++++⟩ | Cluster | Medium (graph state) |
| \|0101⟩ | GHZ | Hard (symmetry breaking) |

Run K-scan for each with both:
- Static: optimize_fidelity_parameterized()
- Floquet: optimize_floquet_fidelity_parameterized()

---

## Current Status Summary

### What We Know

1. ✅ **Random λ doesn't work** - stuck at classical overlap
2. ✅ **Optimizing λ is essential** - improves fidelity from 0.5 to 0.95
3. ✅ **GHZ is reachable** with K≈16 GEO2 operators (with optimized λ)
4. ✅ **Diagnostics correct** - H_F^(2) does generate 3-body terms
5. ❌ **Current implementation flawed** - uses λ from static for Floquet

### What We Don't Know

1. ❓ Does Floquet O2 outperform static **with optimal λ for Floquet**?
2. ❓ Do different state pairs show different behavior?
3. ❓ Is the hypothesis correct or rejected?

### What We Need to Do

**Option 1: Complete Implementation (Rigorous)**
- Implement `optimize_floquet_fidelity_parameterized()`
- Run proper comparison with separate λ optimization
- Expensive but definitive answer

**Option 2: Heuristic Test (Quick)**
- Try several random λ values for Floquet
- If any reach higher fidelity than static, hypothesis plausible
- Faster but less rigorous

**Option 3: Different State Pairs (Exploratory)**
- Test W-state, cluster state, etc.
- See if any show Floquet advantage
- Avoids expensive implementation

---

## Recommended Next Steps

### Immediate (Today)

1. **Quick heuristic test**: Try 10-20 random λ values for Floquet O2 at K=16
   - If any exceed static fidelity → hypothesis still viable
   - If all worse → likely rejected

2. **Document current findings**: Update main insights document

3. **Test different state pairs**: Run scan for |0000⟩ → W-state
   - Might be easier target
   - Could show different behavior

### Short-term (This Week)

4. **Implement `optimize_floquet_fidelity_parameterized()`** if heuristic promising
   - Proper definitive test
   - Computationally expensive

5. **Generate comparison plots**: Static vs Floquet curves
   - Even if hypothesis rejected, interesting negative result

### Medium-term (Publication)

6. **Write up findings** regardless of outcome:
   - **If hypothesis confirmed**: "Floquet Engineering Reduces Operator Requirements"
   - **If hypothesis rejected**: "Limits of Second-Order Floquet Engineering"
   - **Either way scientifically valuable!**

---

## Scientific Value

### If Hypothesis Confirmed (After Proper λ Optimization)

**Finding:** Floquet requires ~25-30% fewer operators

**Impact:**
- Demonstrates practical utility of time-periodic control
- Quantifies advantage of commutator-generated terms
- Provides design principles for state preparation

### If Hypothesis Rejected (Even with Optimal λ)

**Finding:** Floquet provides no advantage for GEO2 → GHZ

**Impact:**
- Establishes limits of second-order Magnus expansion
- Shows 3-body effective terms insufficient
- Motivates higher-order approaches or different techniques

---

## Key Lessons

### Lesson 1: Optimization Space Matters

Fixed random λ vs optimized λ makes the difference between:
- 50% fidelity (stuck at classical overlap)
- 95% fidelity (reaching target!)

**Never use random coupling coefficients for reachability tests!**

### Lesson 2: Fair Comparison is Hard

To properly compare static vs Floquet, must:
- Optimize over same parameter space
- Use appropriate optimization for each
- Account for structural differences (H vs H_F)

**Our current comparison is NOT fair** because:
- Static: optimized over λ
- Floquet: using λ from static (suboptimal)

### Lesson 3: Negative Results Have Value

Even if Floquet doesn't help, discovering this is valuable:
- Establishes limitations
- Prevents wasted effort on wrong approaches
- Guides future research

**Either outcome is publishable!**

---

## Code Locations

### Completed
- `scripts/run_floquet_operator_scan.py` - K-scan with λ optimization ✓
- `scripts/test_parameterized_optimization.py` - Proof that λ optimization works ✓
- `reach/optimization.py` - Static parameterized optimization ✓

### Needed
- `optimize_floquet_fidelity_parameterized()` in `reach/optimization.py` - NEW
- Heuristic multi-λ test script - NEW
- Updated plotting for proper comparison - TODO

---

## Bottom Line

**Current results suggest Floquet is worse than static, BUT:**

1. **Implementation is incomplete** - not optimizing λ for Floquet
2. **Proper comparison not yet performed**
3. **Hypothesis NOT yet properly tested**

**Two paths forward:**

**A. Quick heuristic (hours):** Try multiple random λ for Floquet, see if any beat static
**B. Rigorous implementation (days):** Full λ optimization for Floquet

**Recommendation:** Start with (A) to decide if (B) is worth the effort.

---

**Status:** Critical implementation flaw identified, proper comparison pending

**Next immediate action:** Run heuristic multi-λ test for Floquet at K=16
