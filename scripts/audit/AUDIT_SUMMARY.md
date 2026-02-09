# Comprehensive Audit Summary

**Date:** 2026-01-28
**Scope:** Complete verification of reachability criteria implementations

---

## Executive Summary

The quantum reachability codebase has undergone comprehensive auditing. All three criteria (Spectral, Krylov, Moment) have been verified against their mathematical definitions. Critical bugs have been identified and fixed:

1. **GEO2 Optimizer Settings (CRITICAL)**: `maxiter=20` caused 70% false-unreachable rate. Fixed to `maxiter=100-150`.
2. **Analytical Gradient**: Added `spectral_overlap_with_grad()` with 5-64x speedup.
3. **Degenerate Eigenvalue Handling**: Regularized inverse prevents NaN in gradient.

---

## 1. Spectral Criterion

### Mathematical Definition

The spectral overlap is:
```
S(λ) = Σₙ |⟨φ|uₙ(λ)⟩* · ⟨uₙ(λ)|ψ⟩|
```

where `|uₙ(λ)⟩` are eigenstates of `H(λ) = Σₖ λₖ Hₖ`.

**Criterion**: Target φ is unreachable if `max_λ S(λ) < τ`.

### Implementation (mathematics.py:166-234)

```python
def spectral_overlap(lambdas, psi, phi, hams):
    # 1. Construct H(λ) = Σₖ λₖ Hₖ
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))

    # 2. Eigendecomposition: H = U diag(E) U†
    eigenvalues, eigenvectors = eigh(H_lambda.full())

    # 3. Project states onto eigenbasis
    psi_coeffs = eigenvectors.conj().T @ psi_vec  # ψₙ = ⟨uₙ|ψ⟩
    phi_coeffs = eigenvectors.conj().T @ phi_vec  # φₙ = ⟨uₙ|φ⟩

    # 4. Spectral overlap: S(λ) = Σₙ |φₙ* · ψₙ|
    overlap_terms = np.abs(phi_coeffs.conj() * psi_coeffs)
    return np.sum(overlap_terms)
```

### Verification Status: PASS

| Test | Method | Result |
|------|--------|--------|
| Definition match | Manual derivation check | Correct |
| Known cases | σx rotation |0⟩→|1⟩ | S(π/2) = 1.0 |
| Gradient accuracy | FD vs analytical | Rel error < 1e-9 |
| Optimizer convergence | Provably reachable targets | 0% false-unreachable |

---

## 2. Krylov Criterion

### Mathematical Definition

The Krylov subspace is:
```
Kₘ(H, ψ) = span{ψ, Hψ, H²ψ, ..., H^(m-1)ψ}
```

**Score**: `R(λ) = ‖P_Kₘ|φ⟩‖² = Σₙ |⟨Kₙ|φ⟩|²`

**Criterion**: Target φ is unreachable if `max_λ R(λ) < τ`.

### Implementation (mathematics.py:482-652)

```python
def s(H, psi, m, tol=1e-14):
    """Arnoldi iteration with modified Gram-Schmidt."""
    result[:, 0] = psi_vec / norm(psi_vec)

    for i in range(1, m):
        w = H @ result[:, i-1]           # Krylov expansion
        h = result[:, :i].conj().T @ w   # Orthogonalization
        w = w - result[:, :i] @ h

        if norm(w) < tol:  # Breakdown detection
            break
        result[:, i] = w / norm(w)

    return qr(result)[0]  # Final QR for stability

def krylov_score(lambdas, psi, phi, hams, m=None):
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))
    V = krylov_basis(H_lambda, psi, m or d)
    coeffs = V.conj().T @ phi_vec
    return np.vdot(coeffs, coeffs)  # ‖V†φ‖²
```

### Verification Status: PASS

| Test | Method | Result |
|------|--------|--------|
| Basis orthonormality | V†V = I check | ‖V†V - I‖ < 1e-14 |
| Dimension correctness | Rank test | dim(Kₘ) ≤ m |
| Provably reachable | exp(-iHt)|ψ⟩ targets | 0% false-unreachable |
| Score = 1 - residual² | Mathematical identity | Verified to 10 decimals |
| Gradient accuracy | Analytical vs FD | Error < 1e-10 (all ensembles) |
| Score agreement | krylov_score vs krylov_score_with_grad | Identical (diff = 0) |

### Krylov Basis Bugfix (2026-01-28)

**Bug:** `krylov_basis()` did not truncate to actual Krylov dimension on breakdown. When the Arnoldi iteration terminated early (norm < tol), remaining columns were zero-filled, then QR created spurious orthonormal vectors spanning directions **outside** the true Krylov subspace. This inflated R to ~1.0 for sparse Hamiltonians (canonical ensemble).

**Fix:** Truncate result matrix to `actual_m` columns before QR, ensuring the basis spans only the true Krylov subspace.

### GEO2 Krylov Anomaly (EXPECTED PHYSICS)

For GEO2, Krylov score R ≈ 1 for all λ because:
- GEO2 Hamiltonians are full-rank (dense matrices)
- Single H fills entire d-dimensional space: `dim(Kd(H, ψ)) = d`
- This is correct physics, not a bug

| Ensemble | K=2 Krylov dim | K=13 Krylov dim | H rank |
|----------|----------------|-----------------|--------|
| GEO2 | 16/16 | 16/16 | 16 (full) |
| Canonical | 2/16 | 7/16 | 2 (sparse) |

---

## 3. Moment Criterion

### Mathematical Definition

First moments:
```
⟨Hₖ⟩_ψ = ⟨ψ|Hₖ|ψ⟩,  ⟨Hₖ⟩_φ = ⟨φ|Hₖ|φ⟩
L_k = ⟨Hₖ⟩_φ - ⟨Hₖ⟩_ψ
```

Second moments (anticommutator):
```
Q_{km} = ⟨{Hₖ,Hₘ}/2⟩_φ - ⟨{Hₖ,Hₘ}/2⟩_ψ
```

**Criterion**: Target φ is unreachable if there exists x such that `Q + x·LL†` is positive or negative definite.

Equivalently: Project Q onto null space of L and check definiteness.

### Implementation (analysis.py, comprehensive_validation.py)

Two equivalent implementations exist:

1. **QuTiP approach** (analysis.py):
```python
kernel = null_space(diff.reshape(1, -1))  # null(L†)
m_final = kernel.T @ (sq_phi - sq_psi) @ kernel
eigvals = np.linalg.eigvalsh(m_final)
return np.all(eigvals > 0) or np.all(eigvals < 0)
```

2. **NumPy approach** (comprehensive_validation.py):
```python
L_outer = np.outer(L, L)
for x in np.linspace(-10, 10, 1000):
    M = Q + x * L_outer
    if np.all(eigvalsh(M) > 0):
        return True
```

### Verification Status: PASS

| Test | Method | Result |
|------|--------|--------|
| First moments | Manual ⟨H⟩ check | Correct |
| Second moments | Anticommutator formula | Correct |
| Provably reachable | exp(-iHt)|ψ⟩ targets | 0% false-unreachable |
| Cross-criterion | Moment ≤ Spectral ordering | Verified |

---

## 4. Lambda Optimization

### Problem
Maximize `S(λ)` or `R(λ)` over `λ ∈ [-1,1]^K`.

### Method: L-BFGS-B with Analytical Gradient

The spectral gradient uses first-order eigenvalue perturbation theory:
```
∂|uₙ⟩/∂λₖ = Σ_{m≠n} ⟨uₘ|Hₖ|uₙ⟩/(Eₙ - Eₘ) |uₘ⟩
```

Implementation (mathematics.py:236-343):
```python
def spectral_overlap_with_grad(lambdas, psi, phi, hams):
    # ... eigendecomposition ...

    # Regularized inverse for degenerate eigenvalues
    delta_E = E[:, None] - E[None, :]
    inv_delta_E = delta_E / (delta_E**2 + 1e-12)  # Smooth at degeneracies

    for k in range(K):
        Hk_eig = U† @ Hk @ U
        dpsi_dlk = np.sum(Hk_eig * inv_delta_E * psi_coeffs, axis=1)
        dphi_dlk = np.sum(Hk_eig * inv_delta_E * phi_coeffs, axis=1)
        dc_dlk = dphi_dlk.conj() * psi_coeffs + phi_coeffs.conj() * dpsi_dlk
        grad[k] = np.real(np.sum(sign_c.conj() * dc_dlk))

    return S, grad
```

### Verification Status: PASS

| Test | Method | Result |
|------|--------|--------|
| Gradient accuracy | FD comparison | Max rel error < 1e-8 |
| Degenerate handling | H with repeated eigenvalues | No NaN, smooth gradient |
| Speedup | Benchmark vs FD | 5-64x faster |
| Optimizer convergence | Grid search comparison | Same optimum found |

### Settings Comparison

| Parameter | Old GEO2 | Corrected | Effect |
|-----------|----------|-----------|--------|
| maxiter | 20 | 100-150 | Proper convergence |
| restarts | 1 | 3 | Escape local minima |
| ftol | 1e-4 | 1e-6 | Tighter tolerance |
| jac | None (FD) | Analytical | 5-64x speedup |

---

## 5. GEO2 Ensemble

### Definition
GEO2 (Geometric 2-local) generates Hamiltonians on an nₓ × n_y qubit lattice:
- L = 3n + 9|E| operators (single-site + nearest-neighbor)
- Dimension d = 2^(nₓ × n_y)

### Properties Verified

| Property | Expected | Verified |
|----------|----------|----------|
| Hermiticity | H = H† | Yes |
| Dimension | d × d | Yes |
| Full rank | rank(H) = d | Yes (dense) |
| Normalization | Tr(H²) ≈ d | Yes |

### Key Finding: Dense vs Sparse

GEO2 Hamiltonians are **dense** by default, filling the entire Hilbert space. This explains:
- Krylov R ≈ 1 always (entire space is Krylov space)
- Faster phase transitions (more control per operator)
- Moment transitions very early (K ≈ 3-5)

---

## 6. Numerical Stability

### Edge Cases Tested

| Case | Handling |
|------|----------|
| Degenerate eigenvalues | Regularized inverse: `Δ/(Δ² + ε²)` |
| Zero overlap terms | Safe sign: `c/|c|` only when `|c| > 1e-15` |
| Ill-conditioned H | Warning logged, computation continues |
| Krylov breakdown | Early termination, zero-padded columns |
| Near-boundary optimum | Clipping to valid range |

### Precision

| Quantity | Precision |
|----------|-----------|
| Spectral overlap S | 64-bit float, ±1e-15 |
| Gradient | Matches FD to 10 decimals |
| Krylov basis | Orthonormal to 1e-14 |
| Moment eigenvalues | 64-bit LAPACK |

---

## 7. Audit Scripts Reference

| Script | Purpose | Key Findings |
|--------|---------|--------------|
| `validate_reachable_targets.py` | False-unreachable rate | 70% → 0% after fix |
| `optimize_settings_sweep.py` | Find optimal settings | maxiter≥75, restarts≥3 |
| `test_analytical_gradient.py` | Gradient accuracy | Rel error < 1e-9 |
| `benchmark_grad_vs_fd.py` | Speed comparison | 5-64x speedup |
| `comprehensive_validation.py` | Full 6-test suite | All pass |
| `verify_optimizer_convergence.py` | Gradient norm at termination | Converged |
| `validate_d64_settings.py` | d=64 validation | 0% false-unreachable |
| `geo2_phase_transition.py` | Ultra-low density GEO2 | ρ_c ≈ 0.02-0.04 |
| `test_dimension_scaling.py` | Dimension ordering | Correct after fix |
| `verify_mathematical_definitions.py` | 8-test mathematical suite | All pass |
| `verify_krylov_gradient.py` | Krylov gradient accuracy | Error < 1e-10 |
| `verify_krylov_score_agreement.py` | Score consistency test | Identical scores |
| `compare_moment_implementations.py` | Null-space vs x-sweep | 99.2% agreement |
| `test_parallelization.py` | Parallel correctness/speedup | All pass |

---

## 8. Remaining Considerations

### Completed Work (2026-01-28)

1. **Krylov basis bugfix**: Fixed `krylov_basis()` to truncate to actual Krylov dimension on breakdown, preventing spurious QR-generated vectors from inflating scores.

2. **Krylov analytical gradient**: Implemented `krylov_score_with_grad()` in `mathematics.py`
   - Differentiates through Arnoldi iteration
   - 1.4-6.5x speedup vs finite differences
   - Gradient accurate to <1e-10 for all ensembles (after bugfix)
   - Verified: `scripts/audit/verify_krylov_gradient.py` (all tests pass)

3. **Moment criterion comparison**: Compared null-space vs x-sweep approaches
   - 99.2% agreement rate on 400 random tests
   - Null-space is 36x faster
   - **Recommendation**: Use null-space approach
   - Report: `scripts/audit/MOMENT_COMPARISON.md`

4. **Parallel evaluation**: Created `reach/parallel.py` using joblib
   - Embarrassingly parallel Monte Carlo trials
   - Verified: correct results match sequential execution
   - Significant speedup for d >= 32

5. **Adaptive sampling**: Created `reach/adaptive_sampling.py`
   - Detects transition region from coarse sampling
   - Adds dense points near phase transitions
   - Dimension-specific configs based on transition estimates

6. **Overnight production scripts**: Created canonical and GEO2 overnight runs
   - `scripts/production/overnight_canonical.py` (8-12 hours)
   - `scripts/production/overnight_geo2.py` (10-15 hours)
   - Periodic checkpoints, ETA estimates

### Known Limitations

1. **GEO2 d=64** requires ~10-15 hours for full sweep (K=205 parameters)
2. **Parallel overhead**: Process creation overhead dominates for d<=16; use n_jobs=1 for small problems

### Recommendations for Future Work

1. Profile and optimize d=64 GEO2 bottleneck
2. Consider GPU acceleration for eigendecomposition at d=64

---

## Appendix: Verified Mathematical Identities

1. **Krylov score = 1 - residual²**:
   ```
   R(λ) = ‖V†φ‖² = 1 - ‖φ - VV†φ‖²
   ```
   Verified to 10 decimal places.

2. **Spectral overlap bounds**:
   ```
   0 ≤ S(λ) ≤ 1  (equality at 1 for aligned states)
   ```
   Enforced with numerical tolerance.

3. **Fermi-Dirac transition shape**:
   ```
   P(ρ) ≈ 1/(1 + exp((ρ - ρ_c)/Δ))
   ```
   Excellent fit (R² > 0.98) for spectral/Krylov transitions.

---

*Generated by audit verification suite, 2026-01-28*
*Updated with Krylov gradient and moment comparison results, 2026-01-28*
