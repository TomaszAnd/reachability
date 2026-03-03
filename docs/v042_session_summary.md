# v0.4.2 Session Summary

**Branch:** `fast`
**Version:** v0.4.1 -> v0.4.2
**Commits:** 9 (`c7ccf42` through `a90c83d`)

---

## Commits

| # | Hash      | Summary                                                        |
|---|-----------|----------------------------------------------------------------|
| 1 | `c7ccf42` | criteria: fix nwhatsp typo in _lanczos_basis, add QR parameter |
| 2 | `b10baef` | criteria: cleanup sparse conversion, improve logging/comments  |
| 3 | `546c2cc` | models: cache lattice edges, document d=256 limitation         |
| 4 | `388b485` | sampling/plotting: reduce min_hamiltonians, unify colors, add bootstrap |
| 5 | `c3e80fa` | criteria_jax: add documentation comments                       |
| 6 | `3f642e8` | tests: add 10 tests covering audit findings                    |
| 7 | `cc5489c` | scripts: add d=256 timing profile                              |
| 8 | `117d616` | scripts: add v96 sweep with two-pass sampling and publication plots |
| 9 | `a90c83d` | docs: update CLAUDE.md to v0.4.2, add audit plan               |

---

## Critical Bug Fix: nwhatsp Typo

The most important change in this session was fixing a typo in `src/criteria.py` where `nwhatsp.linalg.qr` appeared instead of `np.linalg.qr`. This bug crashed ALL Krylov forward-only evaluations (i.e., any evaluation that did not go through the gradient/optimization path). The fix is verified by new test I1.

---

## Files Modified

### `src/criteria.py`
- Fixed `nwhatsp.linalg.qr` to `np.linalg.qr` (critical bug)
- Added `apply_qr` parameter to `_lanczos_basis` method, allowing callers to control QR re-orthogonalization
- Simplified sparse conversion logic in `SpectralCriterion`
- Changed `logger.debug` to `logger.warning` for important diagnostics
- Added comment documenting `eps_reg` regularization parameter

### `src/models.py`
- Cached `_build_lattice_edges()` result in `QubitGridModel` to avoid redundant recomputation
- Added d=256 memory warning to `CanonicalQuditModel` docstring

### `src/sampling.py`
- Changed `AdaptiveSweepConfig.min_hamiltonians` default from 20 to 10
- Added `estimate_rho_c(df, criterion)` function: linear interpolation at P=0.5
- Added `bootstrap_rho_c(df, criterion, n_boot, seed)` function: bootstrap 95% CI for rho_c

### `src/plotting.py`
- Updated `CRIT_COLORS` dictionary: moment=green, spectral=blue, krylov=orange
- Added `CRIT_LABELS` dictionary for consistent legend labels

### `src/criteria_jax.py`
- Added documentation comments explaining QR omission rationale
- Added comments on breakdown tolerance behavior
- Added notes about JIT recompilation triggers

### `src/__init__.py`
- Version bumped to `"0.4.2"`

### `CLAUDE.md`
- Updated to v0.4.2 with all changes documented

---

## New Files

### `scripts/tests/run_all_tests.py` (extended)
Extended with Test I group containing 10 new tests, bringing the total from 34 to 44.

### `scripts/profile_d256.py`
Timing profile script for d=256. Profiles Canonical d=128 and QubitGrid d=256, then extrapolates expected runtimes for larger configurations.

### `scripts/production/overnight_sweep_v96.py`
Production sweep script featuring:
- Two-pass K selection (coarse log-spaced scan, then dense linear around transitions)
- `AdaptiveSweep` with early termination
- Publication-quality plots with bootstrap confidence intervals
- `--quick` mode for fast validation runs

### `audit_plan.md`
Detailed audit plan document covering code quality, correctness, and performance targets.

---

## New Functions

### `estimate_rho_c(df, criterion)` — `src/sampling.py`
Estimates the critical density rho_c by linear interpolation at the P=0.5 threshold in a sweep DataFrame.

### `bootstrap_rho_c(df, criterion, n_boot, seed)` — `src/sampling.py`
Computes a bootstrap 95% confidence interval for rho_c using resampled sweep data.

### `_lanczos_basis(self, H, apply_qr=True)` — `src/criteria.py`
Refactored with new `apply_qr` parameter. When `True` (default), applies QR re-orthogonalization to the Lanczos vectors. When `False`, skips QR (used intentionally in the gradient path to avoid discontinuous gradients).

---

## New Tests (Group I: Audit Coverage)

| Test | Description                                          |
|------|------------------------------------------------------|
| I1   | Krylov forward no crash (nwhatsp fix verification)   |
| I2   | Krylov random search returns ReachabilityResult      |
| I3   | QR vs no-QR both return valid scores                 |
| I4   | Spectral gradient vs finite difference               |
| I5   | Krylov gradient vs finite difference                 |
| I6   | Moment non-monotonicity in K                         |
| I7   | Sparse vs dense score agreement (QubitGrid)          |
| I8   | estimate_rho_c on synthetic data                     |
| I9   | bootstrap_rho_c on synthetic data                    |
| I10  | AdaptiveSweep SEM check                              |

**Test status:** All 44 tests passing (34 original + 10 new).

---

## Key Findings

### QR Re-orthogonalization
- Without QR, the Lanczos basis drifts significantly: ||V^dagger V - I|| measured up to 6.36
- Score errors without QR reach up to 0.42
- The forward evaluation path now applies QR by default, producing correct scores
- The gradient path intentionally omits QR because QR introduces discontinuities that break gradient-based optimization

### Random Search Performance
- Random-50 search gives 100% verdict agreement with L-BFGS-B for Krylov at d >= 32
- Speed improvement: 9-45x faster than L-BFGS-B
- Spectral criterion MUST continue using L-BFGS-B: random search achieves only 0-34% agreement due to narrow peaks in the Spectral optimization landscape

### Lattice Edge Caching
- `QubitGridModel` now caches the result of `_build_lattice_edges()`, eliminating redundant graph construction on repeated calls

---

## v96 Sweep Script vs v95

| Feature                  | v95              | v96                              |
|--------------------------|------------------|----------------------------------|
| K selection              | Fixed grid       | Two-pass (coarse + dense around transitions) |
| Sweep engine             | Basic DensitySweep | AdaptiveSweep with early termination |
| Confidence intervals     | None             | Bootstrap CIs for rho_c          |
| Quick validation mode    | No               | `--quick` flag supported          |
