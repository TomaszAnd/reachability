# Session Summary v76: Eigendecomposition Optimization

## Objective
Optimize eigendecomposition to reduce Spectral d=256 time from ~200s (projected)
to practical levels.

## Key Discovery: scipy eigh is 2x FASTER than numpy at d≥256 on M1

The v76 spec assumed numpy.eigh (heevd) would be 2-4x faster than scipy (heevr).
**The opposite is true on Apple M1 Accelerate**:

| d | numpy.eigh (heevd) | scipy.eigh default (heevr) | Speedup |
|---|-------------------|---------------------------|---------|
| 64 | 0.76ms | 1.05ms | numpy 1.4x |
| 128 | 5.20ms | 6.63ms | numpy 1.3x |
| 192 | 14.49ms | 10.13ms | **scipy 1.4x** |
| 256 | 39.15ms | 21.02ms | **scipy 1.9x** |
| 384 | 95.18ms | 70.88ms | **scipy 1.3x** |
| 512 | 246.80ms | 143.55ms | **scipy 1.7x** |

**Root cause**: Apple Accelerate's MRRR implementation (heevr) is highly optimized
for Apple Silicon at large matrix sizes. The divide-and-conquer algorithm (heevd)
has higher memory overhead that hurts on M1's unified memory architecture.

## Implementation

Updated `eigendecompose()` in `math_utils.py`:
- d < 256: numpy.linalg.eigh (heevd) — faster for small matrices
- d ≥ 256: scipy.linalg.eigh (heevr default, overwrite_a=True) — 1.9x faster

## Spectral d=256 Benchmark

| K | Time/trial (restarts=3) | Previous estimate |
|---|------------------------|------------------|
| 30 | 33s | ~200s |
| 60 | 49s | ~200s |
| 90 | 61s | ~200s |
| 114 | 94s | ~200s |

The v73 estimate of ~200s was based on extrapolation. Actual measurement with
scipy eigh: **33-94s depending on K** (avg ~55s).

## Parallel Restarts Investigation

| Approach | Speedup | Why |
|----------|---------|-----|
| ThreadPoolExecutor (existing) | 0.97x | BLAS thread contention |
| Multiprocessing (3 workers) | 1.10x | Memory bandwidth limited on M1 |
| Multiprocessing (4 workers) | 0.92x | Worse — overhead > parallelism |

**Conclusion**: M1's unified memory architecture limits parallel eigh benefit.
Not worth implementing.

## Projected Sweep Times

### Spectral d=256 (with scipy eigh, avg 50s/trial)

| Config | Trials | Time |
|--------|--------|------|
| 100h × 5t × 10K | 5,000 | **69 hours (3 days)** |
| 50h × 5t × 15K | 3,750 | **52 hours (2 days)** |
| 50h × 10t × 10K | 5,000 | 69 hours |
| 200h × 10t × 10K | 20,000 | 278 hours |

### All criteria d=256

| Criterion | Avg time/trial | 5,000 trials |
|-----------|---------------|-------------|
| Moment | ~1ms | ~5 sec |
| Krylov | ~5s | ~7 hours |
| Spectral | ~50s | ~69 hours |
| **Total** | | **~76 hours (3 days)** |

## Additional Findings

### BLAS Configuration
- NumPy uses Apple Accelerate (confirmed via `np.__config__.show()`)
- Accelerate's eigh is already single-threaded (no multi-core benefit per call)
- This explains why ThreadPoolExecutor doesn't help for optimizer parallelism

### scipy.linalg.eigh Flags
- `overwrite_a=True`: Avoids internal copy (minor speedup)
- `check_finite=False`: Skips validation (minor speedup)
- `driver='evd'`: 0.35x at d=256 — SLOWER (opposite of what scipy #9212 suggests for Intel)

## Files Modified

- `src/math_utils.py` — Dimension-adaptive eigendecompose (scipy for d≥256)

## Commits

- `f4e21f9` v74: Lazy dense basis + production sweep + GPU audit
- `899ff58` v75: Spectral speedup investigation
- (this session) v76: scipy eigh optimization

## Recommendations

1. **Spectral d=256 is now practical as a multi-day run** (~3 days for 5K trials)
2. Run Krylov+Moment first (~7 hours), then Spectral separately
3. For Spectral: use 50h × 5t × 10K (reduced trials, explicit error bars)
4. Could also use AdaptiveSweep to reduce trials in flat regions
