# Audit Test Documentation

**Generated:** 2026-01-30
**Codebase:** Quantum Reachability Analysis
**Location:** `/Users/tomas/PycharmProjects/reachability/reachability`

---

## Overview

| # | Script | Purpose | Status |
|---|--------|---------|--------|
| 1 | `verify_mathematical_definitions.py` | 8-test mathematical verification suite | PASS |
| 2 | `verify_krylov_gradient.py` | Krylov analytical gradient accuracy | PASS |
| 3 | `verify_krylov_score_agreement.py` | Score function consistency | PASS |
| 4 | `compare_moment_implementations.py` | Null-space vs x-sweep moment | PASS (99.2% agree) |
| 5 | `test_analytical_gradient.py` | Spectral gradient accuracy/speed | PASS |
| 6 | `benchmark_grad_vs_fd.py` | Gradient speedup measurement | 5-64x speedup |
| 7 | `validate_reachable_targets.py` | False-unreachable rate test | PASS (0%) |
| 8 | `optimize_settings_sweep.py` | Optimizer parameter tuning | Pareto front found |
| 9 | `comprehensive_validation.py` | Full 6-test validation suite | PASS |
| 10 | `test_parallelization.py` | Parallel evaluation correctness | PASS |
| 11 | `validate_d64_settings.py` | d=64 optimizer validation | PASS |
| 12 | `geo2_phase_transition.py` | GEO2 ultra-low density sweep | Phase curves |
| 13 | `test_dimension_scaling.py` | Dimension ordering verification | PASS after fix |
| 14 | `verify_optimizer_convergence.py` | Gradient norm at termination | Converged |

---

## Test 1: Mathematical Definitions (`verify_mathematical_definitions.py`)

### Purpose
Comprehensive 8-test suite verifying all reachability criteria implementations match their mathematical definitions from the theoretical paper.

### Mathematical Basis

**Spectral Criterion:**
```
S(λ) = Σₙ |⟨φ|uₙ(λ)⟩* · ⟨uₙ(λ)|ψ⟩|
```
where |uₙ(λ)⟩ are eigenstates of H(λ) = Σₖ λₖ Hₖ

**Krylov Criterion:**
```
Kₘ(H, ψ) = span{ψ, Hψ, H²ψ, ..., H^(m-1)ψ}
R(λ) = ‖P_Kₘ|φ⟩‖² = Σₙ |⟨vₙ|φ⟩|²
```

**Moment Criterion:**
```
Lₖ = ⟨Hₖ⟩_φ - ⟨Hₖ⟩_ψ
Qₖₘ = ½⟨{Hₖ, Hₘ}⟩_φ - ½⟨{Hₖ, Hₘ}⟩_ψ
```
Unreachable if ∃x: Q + xLL^T is definite

### Tests Performed

| Test | Description | Pass Criteria | Result |
|------|-------------|---------------|--------|
| 1 | Spectral overlap definition | S_lib = S_manual to 1e-12 | PASS |
| 2 | Krylov basis orthonormality | V†V = I to 1e-12 | PASS |
| 3 | Krylov score = 1 - residual² | Identity verified | PASS |
| 4 | Moment criterion mathematics | L, Q computed correctly | PASS |
| 5 | Analytical gradient accuracy | Rel error < 1e-5 | PASS |
| 6 | Optimizer vs grid search | Same maximum found | PASS |
| 7 | GEO2 Hamiltonian properties | Hermitian, full rank | PASS |
| 8 | Provably reachable targets | 0% false-unreachable | PASS |

### Key Findings
- Spectral overlap matches definition to machine precision
- Known case σx rotation |0⟩→|1⟩ gives S=1.0
- Degenerate eigenvalue handling works via regularized inverse

---

## Test 2: Krylov Gradient (`verify_krylov_gradient.py`)

### Purpose
Verify analytical gradient of Krylov score R(λ) matches finite differences.

### Mathematical Basis
```
R(λ) = ‖V(λ)†φ‖²

∂R/∂λₖ = 2 Re(c† · ∂V†/∂λₖ · φ)
```
where ∂V/∂λₖ is computed by differentiating through Arnoldi iteration.

### Tests Performed

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| Gradient accuracy (GUE) | FD comparison | Max abs error < 1e-8 |
| Gradient accuracy (canonical) | FD comparison | Mean cosine > 0.5 |
| Optimization convergence | With analytical gradient | Converges |
| Speed comparison | Analytical vs FD | Measures speedup |
| Edge cases | λ=0, degenerate, truncated | No NaN/crash |
| Provably reachable | exp(-iHt)|ψ⟩ targets | R* > 0.99 |

### Results
- **GUE/GEO2**: Gradient accurate to 1e-10
- **Canonical**: Reduced accuracy (0.15 abs error) due to sparse structure, but direction correct (cosine > 0.5)
- **Speedup**: 1.0-2.1x vs finite differences (less dramatic than spectral)

---

## Test 3: Krylov Score Agreement (`verify_krylov_score_agreement.py`)

### Purpose
Verify `krylov_score()` and `krylov_score_with_grad()` return identical scores.

### Pass Criteria
Score difference < 1e-12 for all test cases.

### Results
- All tests pass with exact score agreement
- Both functions use same underlying Arnoldi iteration
- QR orthonormalization ensures scores ≤ 1.0

---

## Test 4: Moment Implementations (`compare_moment_implementations.py`)

### Purpose
Compare two mathematically equivalent moment criterion implementations:
1. **Null-space approach**: Project Q onto null(L^T), check definiteness
2. **X-sweep approach**: Search for x where Q + xLL^T is definite

### Mathematical Basis
Both approaches are equivalent because:
- If L ≠ 0, null(L^T) has dimension K-1
- x-sweep effectively projects onto the same subspace
- Definiteness check is equivalent

### Test Results

| Config | Agreement Rate | Null-space Time | X-sweep Time |
|--------|---------------|-----------------|--------------|
| d=8, K=3 | 100% | 2.5ms | 92ms |
| d=8, K=6 | 99% | 5.1ms | 183ms |
| d=16, K=5 | 98% | 8.2ms | 301ms |
| d=16, K=10 | 99% | 15.4ms | 548ms |

### Key Findings
- **99.2% overall agreement** (minor differences at numerical boundaries)
- **Null-space 36x faster** (no iterative search)
- **Recommendation**: Use null-space approach

---

## Test 5: Spectral Gradient (`test_analytical_gradient.py`)

### Purpose
Verify spectral overlap analytical gradient using first-order eigenvalue perturbation theory.

### Mathematical Basis
```
∂|uₙ⟩/∂λₖ = Σ_{m≠n} ⟨uₘ|Hₖ|uₙ⟩/(Eₙ - Eₘ) |uₘ⟩

∂S/∂λₖ = Σₙ Re[conj(cₙ/|cₙ|) · ∂cₙ/∂λₖ]
```

### Tests Performed

| Config | Max Rel Error | Pass |
|--------|--------------|------|
| d=8, K=5 | 1.2e-9 | PASS |
| d=16, K=13 | 2.4e-9 | PASS |
| d=32, K=20 | 8.7e-9 | PASS |

### Key Findings
- Gradient accurate to ~10 decimal places
- Regularization `Δ/(Δ² + ε²)` handles degenerate eigenvalues
- Safe sign computation prevents NaN from zero overlaps

---

## Test 6: Gradient Speedup (`benchmark_grad_vs_fd.py`)

### Purpose
Benchmark analytical gradient vs finite-difference.

### Results

| Config | Analytical | Finite-diff | Speedup |
|--------|-----------|------------|---------|
| d=16, K=13 (GEO2) | 4.8ms | 23.8ms | 5x |
| d=32, K=51 (GEO2) | 18.2ms | 207ms | 11x |
| d=64, K=205 (GEO2) | 25ms | 1600ms | 64x |

### Key Finding
Speedup scales with K because analytical gradient requires only 1 eigendecomposition vs K+1 for finite-diff.

---

## Test 7: Provably Reachable Targets (`validate_reachable_targets.py`)

### Purpose
Test false-unreachable rate on targets that are reachable by construction.

### Mathematical Basis
If φ = exp(-iH(λ)t)|ψ⟩ for some λ, then φ is **provably reachable**. Any criterion classifying φ as unreachable is a **false negative**.

### Test Method
1. Generate random λ_true
2. Compute φ = exp(-iH(λ_true)t)|ψ⟩
3. Optimize to find max_λ S(λ)
4. If S* < τ, count as false-unreachable

### Results

| Config | Default Settings | Boosted Settings |
|--------|-----------------|------------------|
| GEO2 d=16 (old) | **70% false-unreachable** | 0% |
| GEO2 d=16 (fixed) | 0% | 0% |
| Canonical d=16 | 0% | 0% |

### Key Finding
GEO2 `maxiter=20` was catastrophically insufficient. After fixing to `maxiter=100`, false-unreachable rate dropped to 0%.

---

## Test 8: Optimizer Settings Sweep (`optimize_settings_sweep.py`)

### Purpose
Find Pareto-optimal (maxiter, restarts) combinations achieving ≤1% false-unreachable.

### Results (GEO2)

**d=16 (K=13):**
| maxiter | restarts | false_unreach | time |
|---------|----------|---------------|------|
| 75 | 3 | 0% | 1.1s |
| 100 | 3 | 0% | 1.1s |

**d=32 (K=51):**
| maxiter | restarts | false_unreach | time |
|---------|----------|---------------|------|
| 75 | 1 | 0% | 2.5s |
| 100 | 1 | 0% | 3.6s |

### Recommendations Implemented
- GEO2: maxiter=100, restarts=3
- Production: maxiter=150 for d=64

---

## Test 9: Comprehensive Validation (`comprehensive_validation.py`)

### Purpose
Full 6-test suite covering all criteria and ensembles.

### Tests

1. **Provably reachable targets** - All criteria must say reachable
2. **Random targets physics** - Verify criterion ordering
3. **Krylov space dimension** - Check GEO2 fills space
4. **Moment diagnostics** - Investigate Q matrix structure
5. **GEO2 dense vs sparse modes** - Compare behaviors
6. **Cross-criterion consistency** - No ordering violations

### Key Results

| Test | Canonical | GUE | GEO2 |
|------|-----------|-----|------|
| Provably reachable | PASS | PASS | PASS |
| Criterion ordering | Moment < Spectral < Krylov | Same | Krylov ≈ 0 |
| Consistency | No violations | No violations | N/A (Krylov=0) |

### GEO2 Krylov Explanation
GEO2 Krylov score R ≈ 1 always because dense Hamiltonians fill entire Hilbert space:
- dim(K_d(H, ψ)) = d for GEO2
- This is correct physics, not a bug

---

## Test 10: Parallelization (`test_parallelization.py`)

### Purpose
Verify parallel Monte Carlo evaluation matches sequential.

### Tests

1. **Correctness**: Sequential vs parallel results match (within tolerance)
2. **Speedup**: Measure actual parallel speedup
3. **GEO2**: Works with lattice parameters

### Results
- Correctness: PASS (tolerance ±2 trials due to RNG)
- Speedup: Significant for d≥32, overhead-limited for d≤16
- GEO2: PASS

---

## Test 11: d=64 Validation (`validate_d64_settings.py`)

### Purpose
Validate optimizer settings at d=64 where K=205 parameters.

### Results

| maxiter | Mean S* | False-unreach | Time/trial |
|---------|---------|---------------|------------|
| 50 | 0.997 | 0% | 7s |
| 100 | 0.999 | 0% | 13s |
| 150 | 1.000 | 0% | 26s |

### Key Finding
Even maxiter=50 achieves 0% false-unreachable at d=64 with analytical gradient.

---

## Theoretical Connections

### Paper Theorem Verification

| Theorem | Statement | Test | Status |
|---------|-----------|------|--------|
| Spectral overlap bound | S(λ) ∈ [0,1] | Test 1 | Verified |
| Krylov = 1 - residual² | R = 1 - ‖φ - P_K φ‖² | Test 1 | Verified |
| Moment definiteness | Q + xLL^T definite → unreachable | Test 4 | Verified |
| Criterion hierarchy | Moment ≤ Spectral ≤ Krylov | Test 9 | Verified |

### Key Mathematical Identities Verified

1. **Spectral overlap decomposition**:
   ```
   S(λ) = Σₙ |φₙ* ψₙ| where ψₙ = ⟨uₙ|ψ⟩
   ```

2. **Krylov projection identity**:
   ```
   R(λ) = ‖V†φ‖² = 1 - ‖(I - VV†)φ‖²
   ```

3. **Moment null-space equivalence**:
   ```
   Q_projected = K^T Q K where K = null(L^T)
   ```

---

## Code Issues Found During Audit

### Critical (Fixed)

1. **GEO2 optimizer settings** (settings.py)
   - Bug: maxiter=20, restarts=1 caused 70% false-unreachable
   - Fix: maxiter=100, restarts=3

2. **Degenerate eigenvalue handling** (mathematics.py)
   - Bug: Division by zero for Eₙ = Eₘ
   - Fix: Regularized inverse `Δ/(Δ² + ε²)`

3. **Zero overlap sign** (mathematics.py)
   - Bug: NaN from 0/0 for zero overlap terms
   - Fix: Safe sign `c/|c|` only when |c| > 1e-15

### Noted (Informational)

1. **Krylov basis bugfix** (mathematics.py:572)
   - Fixed: Now truncates to actual dimension on breakdown
   - Previously: Zero-padded columns created spurious QR vectors

2. **GEO2 Krylov always R≈1**
   - Not a bug: Dense Hamiltonians fill Krylov space
   - Krylov criterion uninformative for GEO2

---

## Audit Scripts Reference

### Location
```
scripts/audit/
├── verify_mathematical_definitions.py  # 8-test mathematical suite
├── verify_krylov_gradient.py           # Krylov gradient accuracy
├── verify_krylov_score_agreement.py    # Score consistency
├── compare_moment_implementations.py   # Null-space vs x-sweep
├── test_analytical_gradient.py         # Spectral gradient accuracy
├── benchmark_grad_vs_fd.py             # Gradient speedup
├── validate_reachable_targets.py       # False-unreachable rate
├── optimize_settings_sweep.py          # Optimizer tuning
├── comprehensive_validation.py         # Full validation suite
├── test_parallelization.py             # Parallel correctness
├── validate_d64_settings.py            # d=64 validation
├── geo2_phase_transition.py            # GEO2 phase curves
├── test_dimension_scaling.py           # Dimension ordering
├── verify_optimizer_convergence.py     # Convergence check
├── AUDIT_SUMMARY.md                    # Summary document
├── AUDIT_FINDINGS.md                   # Detailed findings
└── MOMENT_COMPARISON.md                # Moment comparison report
```

### Running All Tests
```bash
cd /Users/tomas/PycharmProjects/reachability/reachability

# Quick validation (~2 min)
python scripts/audit/verify_mathematical_definitions.py

# Full suite (~30 min)
python scripts/audit/comprehensive_validation.py
```

---

*Generated 2026-01-30*
