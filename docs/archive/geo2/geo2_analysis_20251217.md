# GEO2 Comprehensive Analysis & Experimental Redesign

**Date:** December 17, 2025
**Analysis of:** Completed GEO2 experiments (Fixed & Optimized Weights)

---

## Executive Summary

The comprehensive GEO2 experiments revealed a **fundamental limitation** of the Optimized Weights approach (Approach 1): it is constrained by the finite basis size L, which prevents comparison with Fixed Weights (Approach 2a) at dimensions d ≥ 32.

**Key Finding:** Only d=16 (2×2 lattice) allows fair comparison between approaches, as the basis size L=48 is sufficient to cover the entire phase transition region (max_ρ = 0.188 >> ρ_c ≈ 0.04).

---

## Part 1: GEO2 Basis Size Constraints

### GEO2 Basis Formula
```
L = 3n + 9|E|
```
where:
- n = number of qubits (n = nx × ny)
- |E| = number of edges (for open boundary conditions)
- d = 2^n (Hilbert space dimension)

### Constraint Table

| Lattice | n  | \|E\| | d    | L   | max_ρ   | Can cover ρ=0.05? |
|---------|-----|-------|------|-----|---------|-------------------|
| 2×2     | 4   | 4     | 16   | 48  | 0.1875  | ✓ YES             |
| 1×5     | 5   | 4     | 32   | 51  | 0.0498  | ✗ BARELY          |
| 2×3     | 6   | 7     | 64   | 81  | 0.0198  | ✗ NO              |
| 1×7     | 7   | 6     | 128  | 75  | 0.0046  | ✗ NO              |
| 2×4     | 8   | 10    | 256  | 114 | 0.0017  | ✗ NO              |
| 3×3     | 9   | 12    | 512  | 135 | 0.0005  | ✗ NO              |

**Critical insight:** max_ρ = L/d² → 0 as n → ∞ because L grows linearly (~12n) while d² grows exponentially (4^n).

---

## Part 2: Completed Experiment Status

### Approach 1: Optimized Weights
- **Status:** COMPLETED (30.7 minutes)
- **Configurations completed:** 1/3 (only d=16)
- **Reason for incompleteness:** Basis size limits
  - 1×5 lattice (d=32): Hit limit at K=53 > L=51
  - 2×3 lattice (d=64): Hit limit at K=86 > L=81

### Approach 2a: Fixed Weights
- **Status:** RUNNING (Config 3/3, d=64, ~25% complete)
- **d=16 (2×2):** ✓ COMPLETE, 30.4 min
- **d=32 (1×5):** ✓ COMPLETE, 537.7 min (8.96 hours)
- **d=64 (2×3):** ⏳ IN PROGRESS, ETA 10-15 hours

---

## Part 3: Transition Analysis from Data

### d=16 (2×2 lattice)
- **Spectral:** ρ_c ≈ 0.0352, K_c ≈ 9
- **Krylov:**   ρ_c ≈ 0.0391, K_c ≈ 10

### d=32 (1×5 lattice)
- **Moment:**   ρ_c ≈ 0.0039, K_c ≈ 4
- **Spectral:** ρ_c ≈ 0.0176, K_c ≈ 18

**Observation:** Transitions occur at ρ ~ 0.02-0.04 for these systems.

---

## Part 4: The Fundamental Limitation

For Approach 1 (Optimized Weights), we sample K operators from the GEO2 basis, requiring **K ≤ L**.

### Can We Cover the Transitions?

| Lattice | d   | L  | max_ρ  | Est. ρ_c | Can cover? |
|---------|-----|----|--------|----------|------------|
| 2×2     | 16  | 48 | 0.1875 | ~0.04    | ✓ YES      |
| 1×5     | 32  | 51 | 0.0498 | ~0.02    | ✓ BARELY   |
| 2×3     | 64  | 81 | 0.0198 | ~0.01    | ✗ MARGINAL |

**Conclusion:** Only d=16 provides reliable coverage for fair comparison.

---

## Part 5: Experimental Design Recommendations

### OPTION A: Focus on d=16 Only (RECOMMENDED)

d=16 (2×2 lattice) is the **ONLY** dimension where:
1. Both approaches can cover the full transition (max_ρ = 0.188 >> ρ_c ≈ 0.04)
2. Computational cost is reasonable (~30 min per approach)
3. Fair comparison is scientifically meaningful

**Proposed experiment:**
- **Lattice:** 2×2 (d=16, L=48)
- **K range:** 2, 4, 6, ..., 45 (step 2)
- **Trials:** 200
- **Threshold:** τ = 0.99
- **Question:** "Does optimizing weights vs. using random weights change reachability?"

### OPTION B: Accept Different Operating Regimes

Recognize that the two approaches answer **fundamentally different questions**:

#### Approach 2a (Fixed Weights):
> "Given K random GEO2 Hamiltonians with typical weights λ ~ N(0, 1/√L), what fraction of targets are reachable?"

- Tests ensemble-average properties
- No limit on K
- Relevant for: "What happens with many random Hamiltonians?"

#### Approach 1 (Optimized Weights):
> "Given K GEO2 basis operators, can we find optimal weights to reach targets?"

- Tests controllability structure
- Limited by basis size K ≤ L
- Relevant for: "What's achievable with optimal control?"

**These are DIFFERENT scientific questions and shouldn't be directly compared.**

### OPTION C: Wait for d=64 Completion

Current status: d=64 is ~25% complete, ETA 10-15 hours

This will provide:
- Complete K_c(d) scaling for Fixed Weights across d=16, 32, 64
- Ability to extrapolate to larger systems
- Understanding of how phase transition scales with dimension

---

## Summary & Next Steps

### Key Scientific Findings

1. **Optimized Weights (Approach 1) is fundamentally limited by basis size L**
2. **For d ≥ 32, max_ρ = L/d² becomes too small to cover phase transitions**
3. **Only d=16 allows fair comparison between approaches**
4. **The two approaches test different physical scenarios**

### Immediate Recommendations

1. ✓ **PLOTS GENERATED:** Publication-quality plots created in `fig/geo2/`
2. ⏳ **WAIT:** Let Fixed Weights d=64 complete (10-15 hours remaining)
3. 🔬 **OPTIONAL:** Run high-resolution d=16 comparison (both approaches)
4. 📊 **ANALYZE:** Compare transition locations and widths at d=16
5. 📝 **DOCUMENT:** Clearly explain the fundamental difference in paper

### Files Created

| File | Description |
|------|-------------|
| `fig/geo2/geo2_fixed_3panel_tau0.99.png` | Fixed weights: d=16, d=32 |
| `fig/geo2/geo2_optimized_3panel_tau0.99.png` | Optimized weights: d=16 |
| `fig/geo2/geo2_comparison_d16.png` | Direct comparison at d=16 |
| `fig/geo2/geo2_fixed_scaling.png` | K_c vs d for fixed weights |
| `fig/geo2/geo2_optimized_scaling.png` | K_c vs d for optimized weights |
| `scripts/plot_geo2_results.py` | Plotting script |

---

## Fit Results (Fermi-Dirac)

### Fixed Weights (Approach 2a)
- **d=32, Spectral:** ρ_c = 0.0178, K_c = 18.2, Δ = 0.0014

### Optimized Weights (Approach 1)
- **d=16, Spectral:** ρ_c = 0.0402, K_c = 10.3, Δ = 0.0044

**Note:** The d=16 optimized weights transition occurs at higher ρ compared to d=32 fixed weights, but direct comparison across dimensions is not meaningful due to different lattice geometries and basis structures.

---

## Conclusion

The GEO2 experiments successfully generated publication-quality data and revealed an important methodological constraint: **optimized weights and fixed weights test different aspects of quantum reachability** and can only be fairly compared at small dimensions (d=16).

For the paper, we recommend:
1. Present both approaches in their respective regimes
2. Clearly explain the L ≤ K constraint for optimized weights
3. Focus detailed comparison on d=16 where both approaches are viable
4. Discuss the different scientific questions each approach addresses

This provides a more honest and scientifically rigorous treatment than attempting to force a comparison across incompatible parameter regimes.
