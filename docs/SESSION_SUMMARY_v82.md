# Session Summary v82: Spectral Crossing Analysis, Optimizer Fix, Moment Deep Dive

## Objective
Investigate Spectral d=12 vs d=16 curve crossing, fix optimizer convergence at
small K, deep analysis of Moment non-monotonicity, and refine quickstart notebook.

## Task 1: Spectral Crossing — NO CROSSING (Resolved)

### Investigation
Ran rho-matched comparison at d=8, 12, 16 with 100 trials, restarts=5.

### Result
At every matched rho, P strictly decreases with d:

| rho | d=8 | d=12 | d=16 |
|-----|-----|------|------|
| ~0.05 | 1.00 | 1.00 | 1.00 |
| ~0.10 | 0.99 | 0.78 | 0.48 |
| ~0.15 | 0.66 | 0.28 | 0.05 |
| ~0.20 | 0.37 | 0.03 | 0.00 |
| ~0.25 | 0.12 | 0.02 | 0.00 |

The apparent "crossing" at fixed K=25 was because d=12 (rho=0.174) and d=16
(rho=0.098) are at **different rho values**. When rho is matched, no crossing occurs.

Phase transition shifts left with increasing d, as expected.

## Task 2: Moment Tolerance/Gamma Analysis — Insensitive (Confirmed Real)

### Investigation
Tested Moment with 5 configurations:
- Tolerance: 1e-8, 1e-10, 1e-12
- Gamma: ±1e3, ±1e4, ±1e6

### Result
**All configurations give identical P at every K.** Non-monotonicity is completely
insensitive to tolerance and gamma. Confirmed real mathematical behavior.

### Old Code Comparison
Moment implementation on `clean` branch is identical to `fast` — same Q formula,
same L formula, same gamma=±1000, same tol=1e-10. Non-monotonicity was always
present but not previously investigated.

## Task 3: Adaptive Restarts for Small K

### Change
Added adaptive restart logic in `OptimizableCriterion._maximize()`:
```python
effective_restarts = max(restarts, 10 - K) if K < 10 else restarts
```

At K=2 this gives 8 restarts (vs default 2), at K=5 gives 5, at K>=10 no change.

### Impact
Modest improvement in individual evaluation quality. Does not eliminate all
paired violations (L-BFGS-B is inherently local), but provides more coverage
of the landscape at small K where few parameters mean fewer local maxima.

## Task 4: Quickstart Notebook Refinement

### Changes
1. Switched to d=8, 16, 32 (powers of 2, cleaner progression)
2. Rho-based K selection: K = int(rho * d^2) for rho in [0.02-0.40]
3. Increased n_hamiltonians from 20 to 30 (300 trials per K)
4. Used DIM_COLORS for consistent coloring
5. Updated runtime estimate (~15-20 min)

### Results
- d=8, 16, 32 show clear phase transitions shifting left with d
- No curve crossings when plotted against rho
- Moment shows expected non-monotonic behavior
- QubitGrid: Moment transitions very fast, Spectral/Krylov show smooth curves
- Early stopping kicks in properly at high K

## Files Modified

| File | Changes |
|------|---------|
| `src/criteria.py` | Adaptive restarts for small K in `_maximize()` |
| `notebooks/quickstart.ipynb` | d=8,16,32, rho-based K, DIM_COLORS, 30 hamiltonians |
| `README.md` | (from v81, already committed) |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/tests/test_spectral_crossing.py` | Rho-matched Spectral comparison |
| `scripts/tests/test_moment_tolerance.py` | Tolerance/gamma sensitivity analysis |
| `docs/MOMENT_ANALYSIS.md` | Detailed Moment non-monotonicity analysis |
| `docs/SESSION_SUMMARY_v82.md` | This file |

## Tests
All 34 tests pass.

## Key Conclusions

| Question | Answer |
|----------|--------|
| Do Spectral curves cross? | **No** — only appears at fixed K (different rho) |
| Is Moment non-monotonicity numerical? | **No** — insensitive to tol and gamma |
| Does Moment code differ from old? | **No** — identical implementation |
| Do adaptive restarts help? | **Modestly** — more coverage at small K |
| Best K sampling strategy? | **Rho-based** — ensures comparability across d |
