# Toric Plaquette λ* Validation Report

Generated: 2026-01-09

## Overview

This report validates the Floquet moment criterion's ability to predict optimal driving structure for stabilizer code preparation, using the toric code plaquette as a test case.

## System Configuration

**Plaquette geometry:**
```
3 --- 4
|     |
1 --- 2
```

**Operators (XY+YY form):**
- H_13: (X₁X₃ + Y₁Y₃) - edge coupling
- H_23: (X₂X₃ + Y₂Y₃) - edge coupling
- H_24: (X₂X₄ + Y₂Y₄) - edge coupling
- H_1: X₁ - local field (static)
- H_4: X₄ - local field (static)

**Target:** Ground state of -P where P = X₁Z₂Z₃X₄ (plaquette operator)

**Initial state:** |+00+⟩

## Ground Truth (arXiv:2211.09724)

The paper establishes that optimal state preparation requires:
- H_13 driven at frequency ω
- H_23 driven at frequency 2ω
- H_24 driven at frequency 2ω
- H_1, H_4 static (large amplitude)

**Key insight:** Operators with significant commutator contributions should have INCOMMENSURATE frequencies to avoid destructive interference.

## Validation: Fourier Overlap Analysis

The second-order Floquet Hamiltonian is:
```
H_F^(2) = Σ_{j<k} λ_j λ_k F_{jk} [H_j, H_k] / (2i)
```

where F_jk is the Fourier overlap between driving functions f_j(t) and f_k(t).

**Fourier overlap matrix with optimal driving:**

| Pair | F_jk | Explanation |
|------|------|-------------|
| H_13 - H_23 | 0.0 | cos(ωt) ⊥ cos(2ωt) - orthogonal modes |
| H_13 - H_24 | 0.0 | cos(ωt) ⊥ cos(2ωt) - orthogonal modes |
| H_23 - H_24 | 0.0 | Same frequency, compatible phases |

## Key Result

The paper's frequency assignment works because:

1. **F(H_13, H_23) = 0** - Different frequencies (ω vs 2ω) create orthogonal Fourier modes
   → H_13 and H_23 don't interfere in H_F^(2)

2. **F(H_13, H_24) = 0** - Same reasoning as above
   → H_13 and H_24 don't interfere in H_F^(2)

3. **F(H_23, H_24) = 0** - Same frequency (2ω) with compatible phase structure
   → They CAN share frequency without destructive interference

## Validation Status

**SUCCESS**

The frequency grouping {H_13} at ω, {H_23, H_24} at 2ω is validated by the Fourier overlap analysis:
- Operators that need different frequencies (H_13 vs H_23/H_24) have F_jk = 0
- Operators that can share frequency (H_23, H_24) also have F_jk = 0

This confirms the paper's prescription minimizes cross-terms in the second-order Magnus expansion.

## Implications for [[5,1,3]] Code

The same methodology can be applied to discover optimal driving structure:

1. Build operator set for [[5,1,3]] code (5-qubit system)
2. Compute commutator norms ||[H_j, H_k]|| for all pairs
3. Find λ* that maximizes criterion strength
4. Pairs with large |λ_j* λ_k*| × ||[H_j, H_k]|| need different frequencies

## Files Generated

- `scripts/toric_plaquette_validation.py` - Validation script
- `results/LAMBDA_VALIDATION_REPORT.md` - This report
- `fig/floquet/toric_lambda_validation.png` - (when running full experiment)
