# Quantum Reachability Code Audit Findings

**Date:** 2026-01-27
**Auditor:** Claude Code (Opus 4.5)
**Scope:** GEO2LOCAL dimension-ordering anomaly investigation

---

## Executive Summary

The dimension-ordering anomaly in GEO2LOCAL plots is caused by **severely under-powered optimization settings for the GEO2 ensemble**. The GEO2 settings use `maxiter=20` (vs 200 default) and `restarts=1` (vs 2 default), which is catastrophically insufficient for high-dimensional parameter spaces where K = O(d²).

**With proper optimizer settings (maxiter≥50, restarts≥2), the anomaly disappears entirely.**

---

## 1. Optimizer Configuration Analysis

### Current Settings (from `settings.py`)

| Parameter | Default (Canonical/GOE/GUE) | GEO2 | Impact |
|-----------|---------------------------|------|--------|
| `restarts` | 2 | **1** | 2× fewer starting points |
| `maxiter` | 200 | **20** | **10× fewer iterations** |
| `ftol` | 1e-8 | 1e-4 | 10,000× coarser convergence |
| `jac` | None (finite-diff) | None (finite-diff) | Same |
| `method` | L-BFGS-B | L-BFGS-B | Same |
| `bounds` | [-1,1]^K | [-1,1]^K | Same |

### Parameter Space Scaling

For density ρ, the optimizer must search K = round(ρ × d²) parameters:

| d | K at ρ=0.05 | K at ρ=0.10 | GEO2 maxiter | Default maxiter |
|---|-------------|-------------|--------------|-----------------|
| 16 | 13 | 26 | 20 | 200 |
| 32 | 51 | 102 | 20 | 200 |
| 64 | 205 | 410 | 20 | 200 |

At d=64, L-BFGS-B with finite-difference gradients needs K+1=206 function evaluations per gradient step. With `maxiter=20`, the optimizer can take at most 20 gradient steps — barely moving from its random starting point.

### Gradient Computation

No explicit Jacobian is provided (`jac=None`). Scipy uses 2-point finite differences with absolute step size `epsilon ≈ 1.5e-8`. Each gradient evaluation requires K+1 objective evaluations, where each evaluation involves:
1. Building H(λ) = Σ λ_k H_k (K matrix additions)
2. Full eigendecomposition of d×d matrix via `scipy.linalg.eigh`
3. State projections and overlap computation

---

## 2. Validation Results

### Test 2: Provably Reachable Targets (d=16, K=13, tau=0.99)

Targets generated as φ = exp(-iH(λ)t)|ψ⟩ are reachable **by construction**.

| Criterion | GEO2 Default | GEO2 Boosted | Canonical Default |
|-----------|-------------|-------------|-------------------|
| Spectral false-unreachable | **70.0%** | 0.0% | 0.0% |
| Krylov false-unreachable | 0.0% | 0.0% | 0.0% |
| Mean spectral score | 0.976 | 1.000 | 1.000 |
| Mean Krylov score | 1.000 | 1.000 | 1.000 |

**Critical finding:** With GEO2 default settings (`maxiter=20`), the spectral criterion incorrectly classifies **70% of provably reachable targets as unreachable**. The Krylov criterion is unaffected because it appears to converge faster.

### Test 4: Optimization vs Random Lambda

All ensembles showed meaningful optimization improvement at d=16 and d=32 with default settings, confirming the optimizer is running (not just random sampling). However, the improvement is insufficient to reach the true maximum.

---

## 3. Dimension Ordering Analysis (KEY RESULT)

### Test 5: GEO2 at fixed ρ=0.05, τ=0.99

| Optimizer Config | d=16 P(unreach) | d=32 P(unreach) | Ordering |
|-----------------|-----------------|-----------------|----------|
| Default (maxiter=20, r=1) | 0.72 | 0.76 | **d=16 drops first (ANOMALOUS)** |
| Medium (maxiter=50, r=2) | 0.00 | 0.00 | TIED |
| Boosted (maxiter=200, r=3) | 0.00 | 0.00 | TIED |

**The anomaly is entirely an optimizer artifact.** With `maxiter≥50` and `restarts≥2`, both dimensions show P(unreachable)=0 at ρ=0.05, completely eliminating the anomalous ordering.

The reason d=16 drops first with default settings: K=13 parameters are easier to optimize with 20 iterations than K=51 parameters.

### Why Canonical Doesn't Show the Anomaly

Canonical uses `maxiter=200` and `restarts=2`, providing adequate optimization even at high K. The canonical ordering (higher d drops first) reflects genuine physics.

---

## 4. Maxiter and Restart Effect (Test 3)

### Effect of maxiter on mean S* (GEO2)

| maxiter | d=16 (K=13) | d=32 (K=51) |
|---------|------------|------------|
| 10 | 0.948 | 0.945 |
| **20** (current) | **0.972** | **0.969** |
| 50 | 0.989 | 0.993 |
| 100 | 0.995 | 0.999 |
| 200 | 0.996 | 1.000 |

At `maxiter=20`, both d=16 and d=32 are significantly below the true optimum (≈1.0 at ρ=0.05). The saturation point is around `maxiter=50-100`.

### Effect of restarts on mean S* (GEO2, maxiter=20)

| Restarts | d=16 (K=13) | d=32 (K=51) |
|----------|------------|------------|
| 1 (current) | 0.978 | 0.984 |
| 2 | 0.983 | 0.988 |
| 5 | 0.994 | 0.989 |
| 10 | 0.996 | 0.992 |

Restarts help, but cannot compensate for insufficient maxiter. The primary bottleneck is maxiter, not restarts.

### Method Comparison (d=16, K=13)

| Method | Score | nfev | Time |
|--------|-------|------|------|
| L-BFGS-B | 0.997 | 2646 | 0.95s |
| Nelder-Mead | 0.984 | 687 | 0.42s |
| Powell | 0.989 | 1141 | 0.45s |

L-BFGS-B is the best performer but requires the most function evaluations. The derivative-free methods are competitive but not superior.

### Canonical Behavior

For canonical basis (d=16, K=13), maxiter has almost no effect:
- maxiter=10: S*=0.352
- maxiter=200: S*=0.360

This is because canonical operators are extremely sparse (2 non-zero elements each), and the optimization landscape has fewer local features. The optimizer converges quickly regardless of maxiter.

---

## 5. Code Correctness Verification

### Mathematical Implementation ✓

| Criterion | LaTeX Definition | Code Implementation | Match? |
|-----------|-----------------|-------------------|--------|
| Spectral | S(λ) = Σ√(pᵢqᵢ) | `Σ|φₙ*ψₙ|` | ✓ |
| Krylov | ‖P_Kₘ(H(λ))\|φ⟩‖² | `‖V†φ‖²` via Arnoldi | ✓ |
| Moment | Gram matrix definiteness | Anticommutator + null space | ✓ |

### Bugs Found

1. **`models.py` corrupted content**: Lines 144-233 contained an accidentally pasted CV formatting prompt (markdown text inside Python code). **Fixed** — removed the extraneous content.

2. **`models.py` typo**: `CanonicalBasis.__init__` parameter was `cdinclude_identity` instead of `include_identity`, causing a `NameError` when constructing canonical basis with keyword arguments. **Fixed**.

3. **Inconsistent Krylov optimization**: In `monte_carlo_unreachability_vs_K_multi_tau` (analysis.py:2001), the Krylov criterion uses a single random λ with `is_unreachable_krylov` (binary rank test), while `monte_carlo_unreachability_vs_density` (analysis.py:1746) properly optimizes Krylov score over λ when `optimize_lambda=True`. This inconsistency means different analysis functions give different Krylov results.

---

## 6. Root Cause Summary

The dimension-ordering anomaly in GEO2LOCAL plots has a **single root cause**:

> **GEO2 optimization settings (`maxiter=20`, `restarts=1`) are catastrophically insufficient for the K = O(d²) parameter space.**

The settings were chosen for speed (documented as "AGGRESSIVE for performance" in settings.py), achieving a 2.3-4.4× speedup. However, this came at the cost of optimization quality, producing **70% false-unreachable rate on provably reachable targets** and creating the observed dimension-ordering anomaly.

---

## 7. Recommendations & Actions Taken

### Fix Applied: Updated `settings.py`

Based on the settings sweep (`optimize_settings_sweep.py`), the following settings achieve ≤1% false-unreachable rate across d=16 and d=32:

```python
GEO2_RESTARTS = 3       # Was: 1 (3 needed for d=16 K=13; d=32 works with 1)
GEO2_MAXITER = 75       # Was: 20 (minimum for 0% false-unreachable at both dims)
GEO2_FTOL = 1e-6        # Was: 1e-4
```

#### Settings Sweep Results (Pareto-Optimal, ≤1% false-unreachable)

**d=16 (K=13):**
| maxiter | restarts | false_unreach | time/trial |
|---------|----------|---------------|------------|
| 150 | 3 | 0.0% | 823ms |
| 100 | 3 | 0.0% | 1134ms |
| 75 | 3 | 0.0% | 1138ms |

**d=32 (K=51):**
| maxiter | restarts | false_unreach | time/trial |
|---------|----------|---------------|------------|
| 75 | 1 | 0.0% | 2520ms |
| 100 | 1 | 0.0% | 3594ms |
| 50 | 3 | 0.0% | 6018ms |

### Code Optimization: Analytical Gradient (IMPLEMENTED)

Added `spectral_overlap_with_grad()` to `mathematics.py` using first-order eigenvalue perturbation theory. Updated `optimize.py` to automatically use it with L-BFGS-B.

**Gradient accuracy**: Matches finite-difference to ~10 decimal places (relative error < 1e-9).

**Speedup benchmarks** (L-BFGS-B analytical grad vs L-BFGS-B finite-diff, same maxiter=100, restarts=2):

| Config | Analytical | Finite-diff | Speedup | Score |
|--------|-----------|------------|---------|-------|
| d=16, K=13 | 244ms | 1192ms | **4.9×** | 0.9975 vs 0.9981 |
| d=32, K=51 | 917ms | 10043ms | **11.0×** | 0.9995 vs 0.9993 |

The gradient speedup effectively compensates for the increased maxiter/restarts, making the corrected settings comparable in wall-time to the old aggressive settings.

### Data Regeneration

The existing GEO2 production data (`geo2_production_complete_*.pkl`) was generated with the aggressive settings and should be **regenerated** with corrected settings. The current plots are misleading.

---

## 8. Comprehensive Validation Results (Phase 2)

### 8.1 Provably Reachable Targets (All Criteria)

| Config | Spectral | Krylov | Moment (qt) | Moment (np) |
|--------|----------|--------|-------------|-------------|
| Canonical d=16, K=13 | 0% ✓ | 0% ✓ | 0% ✓ | 0% ✓ |
| GUE d=16, K=13 | 0% ✓ | 0% ✓ | 0% ✓ | 0% ✓ |
| GEO2 d=16, K=13 | 0% ✓ | 0% ✓ | 0% ✓ | 0% ✓ |
| GEO2 d=32, K=20 | **10%** ⚠ | 0% ✓ | 0% ✓ | 0% ✓ |

The spectral criterion at d=32, K=20 still has a 10% false-unreachable rate with maxiter=100, restarts=3, indicating the optimizer doesn't always converge. Settings updated to maxiter=100.

### 8.2 GEO2 Phase Transition (Corrected Moment)

Previous moment results showed P(moment)=1 everywhere due to a **bug in the test script** (wrong function interface: passed QuTiP Qobj to a numpy function, and accessed a tuple as a dict). After fixing:

| Criterion | d=16 ρ_c | d=16 K_c | d=32 ρ_c | d=32 K_c |
|-----------|---------|---------|---------|---------|
| Krylov | <0.008 | <2 | <0.003 | <3 |
| Moment | ≈0.012 | ≈3 | <0.005 | <5 |
| Spectral | ≈0.039 | ≈10 | ≈0.025 | ≈26 |

**Ordering**: Krylov ≈ Moment << Spectral for GEO2.

### 8.3 Why Krylov = 0 Everywhere for GEO2 (Physics, Not Bug)

The Krylov space K_m(H(λ), |ψ⟩) = span{|ψ⟩, H|ψ⟩, H²|ψ⟩, ...} fills the entire d-dimensional Hilbert space in exactly d steps because:

- **GEO2 Hamiltonians are full-rank (rank = d)**: Each H_i = (1/√L) Σ_a g_a P_a with random Gaussian weights across all L Pauli basis operators
- **Single H(λ) generates the full space**: rank(H) = d means the Krylov space reaches dimension d
- **Result**: R(λ) = ‖P_K|φ⟩‖² = 1 for any target |φ⟩

| Ensemble | K=2 Krylov dim | K=13 Krylov dim | H_i rank |
|----------|---------------|----------------|----------|
| GEO2 | 16/16 | 16/16 | 16 (full) |
| Canonical | 2/16 | 7/16 | 2 (sparse) |

This is physically correct — GEO2's dense Hamiltonians provide complete quantum control with even K=2 operators. The Krylov criterion is uninformative for GEO2.

### 8.4 Cross-Criterion Consistency: PASS

Tested 50 random targets each for Canonical and GUE at K=5. Zero violations of the expected ordering (spectral-reachable implies Krylov-reachable).

### 8.5 Gradient Numerical Fix

The analytical gradient had a **degenerate eigenvalue bug**: when Eₙ ≈ Eₘ, the perturbation theory expression 1/(Eₙ-Eₘ) diverges. Fixed with regularized inverse:

```python
# Before (NaN for degenerate eigenvalues):
inv_delta_E = 1.0 / delta_E

# After (smooth regularization):
inv_delta_E = delta_E / (delta_E**2 + 1e-12)
```

Also fixed sign computation to avoid NaN from zero overlap terms:
```python
safe_abs_c = np.where(abs_c > 1e-15, abs_c, 1.0)
sign_c = np.where(abs_c > 1e-15, c / safe_abs_c, 0.0)
```

---

## 9. d=64 Validation (Phase 3)

Validated optimizer settings at d=64 (2x3 lattice, K=205 at ρ=0.05).

### Gradient Performance at d=64

| Metric | Value |
|--------|-------|
| Analytical gradient time | 0.025s |
| Finite-diff extrapolated | 1.6s |
| **Speedup** | **64×** |
| Max relative error | 5.3e-8 |

### Maxiter Sweep at d=64, K=205

| maxiter | Mean S* | Min S* | False-unreach | Time (5 trials) |
|---------|---------|--------|---------------|-----------------|
| 50 | 0.9965 | 0.9945 | 0/5 | 35s |
| 75 | 0.9986 | 0.9976 | 0/5 | 50s |
| 100 | 0.9993 | 0.9988 | 0/5 | 66s |
| **150** | **0.9998** | **0.9996** | **0/5** | **129s** |
| 200 | 0.9999 | 0.9999 | 0/5 | 130s |

### Provably Reachable Targets (maxiter=150, restarts=3)

- **False-unreachable rate: 0/10 = 0%**
- Mean S*: 0.9999 (min=0.9996, max=1.0000)
- Mean time per trial: 20.2s

### Conclusion

Even `maxiter=50` achieves 0% false-unreachable at d=64 with the analytical gradient.
The recommended production setting of `maxiter=150` provides ample safety margin.

---

## Appendix: Test Scripts

All test scripts are in `scripts/audit/`:

| Script | Purpose |
|--------|---------|
| `test_optimizer_quick.py` | Quick optimizer improvement check (d=16) |
| `test_optimizer_behavior.py` | Full optimizer comparison with boosted settings |
| `compare_optimizers.py` | Method/maxiter/restart comparison |
| `verify_optimization_runs.py` | Verify optimizer beats random lambda |
| `validate_reachable_targets.py` | False-unreachable rate on known-reachable targets |
| `test_dimension_scaling.py` | Dimension ordering with different optimizer configs |
| `optimize_settings_sweep.py` | (maxiter, restarts) sweep for optimal settings |
| `test_analytical_gradient.py` | Gradient accuracy and speed verification |
| `benchmark_grad_vs_fd.py` | L-BFGS-B analytical vs finite-diff benchmark |
| `geo2_phase_transition.py` | Ultra-low density sweep for actual GEO2 ρ_c |
| `comprehensive_validation.py` | Full validation suite (6 tests, all criteria) |
| `verify_optimizer_convergence.py` | Gradient norm check at termination |
| `validate_d64_settings.py` | d=64 validation (K=205, gradient, maxiter sweep) |
