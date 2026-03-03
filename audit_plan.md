# Codebase Audit Plan (v0.4.2)

## src/criteria.py

### CRITICAL BUG: `nwhatsp.linalg.qr` typo (line 550)
- **Issue**: `nwhatsp.linalg.qr(V, mode="reduced")` should be `np.linalg.qr(...)`. This crashes ALL forward-only Krylov evaluations (i.e., `evaluate(x, return_gradient=False)`).
- **Fix**: Replace `nwhatsp` with `np`. Add explanatory comment block about QR re-orthogonalization. Refactor `_lanczos_basis` to accept `apply_qr: bool = True` parameter.

### SpectralCriterion.evaluate() sparse conversion (lines 407-409)
- **Issue**: Two-step conversion: `_construct_H(lambdas)` returns CSR, then `.toarray()` converts to dense. The intermediate `issparse` check is redundant since `_construct_H` always returns CSR when `_use_sparse` is True.
- **Fix**: Call `self._construct_H(lambdas).toarray()` directly. Add comment explaining why eigendecompose needs dense input.

### Missing comment on eps_reg (line 444)
- **Issue**: `eps_reg = 1e-12` regularization for degenerate eigenvalues lacks explanation.
- **Fix**: Add comment noting the Spectral gradient is approximate at degenerate eigenvalues.

### _single_restart exception handler (line 222)
- **Issue**: `logger.debug` silently swallows optimization failures. Users won't see these unless they explicitly enable debug logging.
- **Fix**: Change to `logger.warning`.

## src/models.py

### QubitGridModel._build_basis_sparse calls _build_lattice_edges twice (lines 372, 378)
- **Issue**: `_build_lattice_edges()` is called once in the 2-local loop (line 372) and again for validation (line 378). The method recomputes edges from scratch each time.
- **Fix**: Cache the result of `_build_lattice_edges()` and reuse it.

### Missing d=256 memory warning on CanonicalQuditModel
- **Issue**: CanonicalQuditModel at d=256 creates d²=65536 dense (d,d) complex128 matrices = ~68GB. No warning or documentation in the code.
- **Fix**: Add a comment on the class documenting this limitation.

## src/sampling.py

### AdaptiveSweepConfig.min_hamiltonians default too high (line 385)
- **Issue**: Default `min_hamiltonians=20` wastes compute in flat regions. Value should be 10.
- **Fix**: Change default to 10.

### Missing bootstrap_rho_c function
- **Issue**: No function to compute confidence intervals on rho_c.
- **Fix**: Add `bootstrap_rho_c(df, criterion, n_boot=1000)` using binomial resampling.

## src/plotting.py

### CRIT_COLORS don't match canonical scheme (lines 36-40)
- **Issue**: Current: moment=blue, spectral=red, krylov=green. Should be: moment=green, spectral=blue, krylov=orange (per CLAUDE.md "Recent Audit Findings").
- **Fix**: Update to moment='#2ca02c', spectral='#1f77b4', krylov='#ff7f0e'.

### CRIT_MARKERS exists but CRIT_LABELS missing
- **Issue**: No CRIT_LABELS dict for consistent label text.
- **Fix**: Add `CRIT_LABELS = {'moment': 'Moment', 'spectral': 'Spectral', 'krylov': 'Krylov'}`.

## src/criteria_jax.py

### Missing documentation comments
- **Issue**: Several design decisions lack inline documentation:
  1. `_krylov_score_jax` doesn't use QR (intentional for autodiff compatibility) — not documented.
  2. `1e-28` breakdown threshold is `KRYLOV_BREAKDOWN_TOL**2` — not explained.
  3. `static_argnums=(4,)` causes JIT recompilation for different `m` values — not documented.
- **Fix**: Add clarifying comments at each location.

## scripts/overnight_sweep_v95.py

### Hardcoded colors using matplotlib C-notation (line 69)
- **Issue**: `CRIT_COLORS = {'moment': 'C2', 'spectral': 'C0', 'krylov': 'C1'}` doesn't match the canonical hex color scheme.
- **Fix**: Import CRIT_COLORS from src.plotting or update to match canonical hex values.

## scripts/tests/run_all_tests.py

### Test registration pattern
- **Issue**: Tests A-H are registered in `ALL_TESTS` dict. New tests (I-R) need to follow the same pattern: define a `test_X_name()` function returning `{'passed': N, 'failed': M, 'details': [...]}` and register in `ALL_TESTS`.

## Phase summary

| Phase | Files | Changes |
|-------|-------|---------|
| 2 | criteria.py | Fix nwhatsp typo, add QR parameter |
| 3 | criteria.py | Cleanup sparse conversion, improve logging |
| 4 | models.py | Cache lattice edges, document d=256 |
| 5 | sampling.py, plotting.py, overnight_sweep_v95.py | Reduce min_hamiltonians, unify colors, add bootstrap |
| 6 | criteria_jax.py | Add documentation comments |
| 7 | run_all_tests.py | Add 10 new tests (I-R) |
| 8 | scripts/profile_d256.py | New timing profile script |
| 9 | scripts/production/overnight_sweep_v96.py | New two-pass sweep script |
| 10 | CLAUDE.md | Version bump to v0.4.2 |
