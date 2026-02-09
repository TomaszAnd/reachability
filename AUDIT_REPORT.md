# Audit Report

**Date**: 2026-02-09
**Branch**: audit
**Auditor**: Claude Code (Opus 4.6)

## Summary

- Total Python files reviewed: 15 (reach/) + 77 (scripts/) + 26 (tests/) = 118
- Total data files verified: 21 CSV + multiple PKL
- Issues found: 7
- Issues fixed: 3
- Issues requiring author action: 4
- Tests passing: 34/34

## Mathematical Verification

### Spectral Criterion (mathematics.py:166-238)
- [x] Formula: S(lambda) = Sum_n |<phi|u_n><u_n|psi>| = Sum_n sqrt(p_n * q_n) -- CORRECT
- [x] Eigenvector extraction via scipy.linalg.eigh (Hermitian-specialized) -- CORRECT
- [x] Probability computation: |phi_coeffs.conj() * psi_coeffs| -- CORRECT
- [x] Bounds validation and clipping -- CORRECT

### Spectral Gradient (mathematics.py:241-352)
- [x] First-order perturbation theory: d|u_n>/dlambda_k = Sum_{m!=n} <u_m|H_k|u_n>/(E_n-E_m) |u_m> -- CORRECT
- [x] Regularized degenerate eigenvalue handling: delta/(delta^2 + eps^2) with eps=1e-12 -- CORRECT
- [x] Chain rule for |c| derivative: d|c|/dlambda = Re(sign(c)* . dc/dlambda) -- CORRECT
- [x] Gradient shape matches parameter count K -- VERIFIED
- [x] Finite-difference agreement: max_error = 0.0 (Test E) -- PASS

### Krylov Criterion (mathematics.py:491-670)
- [x] Arnoldi iteration with modified Gram-Schmidt orthogonalization -- CORRECT
- [x] QR reorthogonalization for numerical stability -- CORRECT
- [x] Score formula: R = ||V^dagger phi||^2 = Sum |c_n|^2 -- CORRECT
- [x] R = 1 - epsilon_res^2 identity verified to 10 decimal places -- CORRECT
- [x] m=min(K,d) parameter handling -- CORRECT (verified in corrected CSVs)
- [x] Krylov breakdown detection (norm < 1e-14) -- CORRECT

### Krylov Gradient (mathematics.py:673-828)
- [x] Derivative propagation through Arnoldi: dw/dlambda_k = H_k @ v_{i-1} + H @ dv_{i-1}/dlambda_k -- CORRECT
- [x] Normalization derivative: d(v/||v||)/dlambda = (dw - v Re<v,dw>)/||w|| -- CORRECT
- [x] QR-orthonormalized basis for score, Arnoldi basis for gradient -- CORRECT
- [x] Finite-difference agreement: max_error = 0.0 (Test E) -- PASS

### Moment Criterion (moment_criteria.py:18-90)
- [x] Linear constraint: L[k] = <H_k>_phi - <H_k>_psi -- CORRECT
- [x] Quadratic form: Q[k,m] = <{H_k, H_m}/2>_phi - <{H_k, H_m}/2>_psi -- CORRECT
- [x] Positive definiteness search: Q + x*L*L^T > 0 implies Q|_{ker(L)} > 0 -- CORRECT
- [x] Eigenvalue sign check logic -- CORRECT (violation of necessary condition for reachability)

### Optimization (optimize.py)
- [x] L-BFGS-B with analytical gradient for spectral and Krylov -- CORRECT
- [x] Multi-restart strategy with proper seeding -- CORRECT
- [x] hams_np precomputation for speed -- CORRECT
- [x] Bounds handling (native for L-BFGS-B, clipping for unconstrained) -- CORRECT

### Models (models.py)
- [x] GOE: H = (A + A^T)/sqrt(2), A ~ N(0,1) -- CORRECT
- [x] GUE: H = (A + A^dagger)/sqrt(2), A ~ N(0,1) + iN(0,1) -- CORRECT
- [x] Canonical basis: {X_jk, Y_jk, Z_j, I}, total d^2 operators -- CORRECT
- [x] GEO2: Pauli basis P_2(G), L = 3n + 9|E| operators -- CORRECT (validated by assertion)
- [x] Random state generation with explicit seeding -- CORRECT

## Code Quality

### Docstrings
- mathematics.py: 13/13 functions documented
- optimize.py: 8/8 functions documented
- models.py: 20/20 functions documented (including class methods)
- moment_criteria.py: 4/4 functions documented

### Code Health
- All 118 Python files compile without errors
- 1 SyntaxWarning in scripts/floquet/five_qubit_code_discovery.py:73 (invalid escape in docstring)
- No unused imports in core library (reach/)
- No duplicate function definitions
- Consistent snake_case naming throughout

## Data Integrity

### Overnight CSV Data
- Canonical: 327 rows, dims=[8,16,32,64], 0 NaN in key columns
- GEO2: 101 rows, dims=[8,16,32,64], 20 NaN in stale krylov columns (expected, replaced during merge)

### Corrected Krylov Data
- Canonical corrected: 108 rows, m=min(K,d) violations: 0
- GEO2 corrected: 96 rows, m=min(K,d) violations: 0
- Canonical dense: 156 rows, m=min(K,d) violations: 0
- GEO2 dense: 134 rows, m=min(K,d) violations: 0

## Test Results

### Comprehensive Test Suite (scripts/tests/run_all_tests.py)
```
Test A (Krylov m Sensitivity):    9 passed, 0 failed -> PASS
Test B (Analytical Limits):       3 passed, 0 failed -> PASS
Test C (Monotonicity in K):       2 passed, 0 failed -> PASS
Test D (Pipeline Consistency):    4 passed, 0 failed -> PASS
Test E (Gradient Edge Cases):     4 passed, 0 failed -> PASS
Test F (Reproducibility):         3 passed, 0 failed -> PASS
Test G (Lie Algebra Structure):   4 passed, 0 failed -> PASS
Test H (Data Pipeline Audit):     5 passed, 0 failed -> PASS

Total: 34 passed, 0 failed (4.1s)
```

### Benchmark Results (100 samples per category)
- Krylov: 100% accuracy all configurations
- Spectral: 96-100% (FN at d=32 due to optimizer local minima)
- 1 FP at GEO2 d=8 spectral (borderline moment target)

## Issues Fixed

1. **README.md:182** - Fixed broken link: `DATA_PROVENANCE.md` -> `docs/DATA_PROVENANCE.md`
2. **main-cl.tex:212** - Fixed typo: "gis not" -> "is not"
3. **main-cl.tex:297** - Fixed typo: "it's criticla that is that" -> "it is critical that"

## Issues Requiring Author Action

1. **main-cl.tex:89** - Missing `fig/fig1.png` (schematic diagram for introduction)
2. **main-cl.tex** - 5 unresolved `\todoko` markers (lines 100, 213, 286, 297, 298)
3. **docs/QUICK_START.md** - References non-existent test scripts (lines 48, 57, 254)
4. **scripts/floquet/five_qubit_code_discovery.py:73** - Invalid escape sequence in docstring

## Scripts Audit

### plot_overnight_v7style.py - PRODUCTION READY
- CSV loading/merging: correct
- Color scheme: DIM_COLORS and CRIT_COLORS match spec
- Fit functions: Fermi-Dirac (spectral/krylov), exponential (moment)
- No dead code or debug prints

### confusion_matrix_benchmark.py - PRODUCTION READY
- Ground truth: reachable via evolution, unreachable via moment criterion
- Confusion matrix: correct TP/FP/TN/FN computation
- Sample sizes: 100 per category, SEED_BASE=123456
- No dead code or debug prints

## File Inventory

| Category | Count |
|----------|-------|
| Core library (reach/*.py) | 15 |
| Scripts (scripts/**/*.py) | 77 |
| Test files | 26 |
| Data files (CSV) | 21 |
| Publication plots | 30 |
| Benchmark plots | 10 |
| LaTeX files | 13 |

## Recommendations

1. Create the missing schematic figure (fig/fig1.png) for the LaTeX manuscript
2. Resolve \todoko markers in manuscript before submission
3. Update QUICK_START.md to remove references to non-existent test scripts
4. Fix invalid escape sequence in five_qubit_code_discovery.py
5. Consider adding pytest as a dependency for CI/CD smoke tests
