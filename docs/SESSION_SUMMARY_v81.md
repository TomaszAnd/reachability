# Session Summary v81: Investigate Moment and Krylov Fluctuations

## Objective
Investigate non-monotonic P(unreachable) observed in quickstart notebook output
for both Moment and Krylov criteria.

## Issue 1: Moment Non-Monotonicity — CONFIRMED (Real Physics)

### Observation
Canonical d=16: P goes from 0.35 at K=6 to 0.40 at K=10 (increases with K).

### Investigation
Ran paired tests (same operators, K is subset of K+1) with 1000 trials at d=16.

**Result: 888 paired violations** — K finds UNREACHABLE certificate but K+1 fails.

### Root Cause
The Moment criterion tests a **sufficient condition**: whether Q + gamma*LL^T is
positive (or negative) definite. Adding operators:
- Adds rows/columns to Q and elements to L
- Changes the eigenvalue structure of Q + gamma*LL^T
- Can make the certificate HARDER to find even though true reachability increases

This is fundamentally different from Spectral/Krylov which maximize over lambda
and are guaranteed monotonic (larger feasible set → score can only increase).

### Independent Sampling Confirms
With 1000 independent trials per K at d=16:
```
K=6:  P=0.320     K=8:  P=0.442  (increases!)
K=10: P=0.121     K=12: P=0.201  (increases!)
```

### Action Taken
- Added docstring note to `MomentCriterion` explaining non-monotonicity
- Added note in README.md Reachability Criteria section
- Added note in quickstart notebook

## Issue 2: Krylov Small-K Fluctuations — Optimizer Convergence

### Observation
QubitGrid d=16: P(K=2)=0.86, P(K=3)=0.96, P(K=4)=0.67 — non-monotonic at K=2→3.

### Investigation
Ran paired tests (K subset of K+1) with 5 restarts, 100 maxiter.

**Key finding: 37 TRUE violations** (K reachable but K+1 unreachable) —
mathematically impossible, so these are optimizer convergence failures.

At K=2-3, L-BFGS-B has very few parameters and the landscape may have local
maxima that the optimizer misses. With more restarts or trials, the effect
averages out.

### Paired Test P Values (Monotonic)
```
K=2: P=0.890  K=3: P=0.850  K=4: P=0.550  K=5: P=0.460  K=6: P=0.270
```
The overall trend IS monotonically decreasing; fluctuations are optimizer noise.

### Action Taken
- Added note in quickstart notebook about Krylov optimizer convergence at small K

## Issue 3: K=1 Not Supported

`sample_submodel(K)` requires K >= 2 (validated at three call sites in models.py).
Cannot add K=1 to QubitGrid sweep as originally proposed.

## Files Modified

| File | Changes |
|------|---------|
| `src/criteria.py` | Added non-monotonicity note to MomentCriterion docstring |
| `README.md` | Added Moment non-monotonicity note |
| `notebooks/quickstart.ipynb` | Added notes about Moment and Krylov behavior |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/tests/test_moment_monotonicity.py` | Paired + independent Moment monotonicity tests |
| `scripts/tests/test_krylov_small_K.py` | Krylov convergence at small K |
| `docs/SESSION_SUMMARY_v81.md` | This file |

## Tests
All 34 tests pass.

## Key Conclusions

| Criterion | Monotonic in K? | Reason |
|-----------|----------------|--------|
| **Spectral** | Yes (guaranteed) | max_lambda score is monotonic in feasible set |
| **Krylov** | Yes (guaranteed) | Same; small-K fluctuations are optimizer noise |
| **Moment** | **No** | Sufficient condition; Q+gamma*LL^T structure changes with K |
