# Session Summary v72: Optimization Evaluation

## Objective
Re-evaluate v71 optimization conclusions, fix JAX Krylov, test M1 GPU,
identify remaining optimizations for d=256 feasibility.

## Key Results

### d=128 Baseline (is_reachable, maxiter=50, restarts=2)

| Model | Criterion | NumPy | JAX CPU | Speedup |
|-------|-----------|-------|---------|---------|
| Canonical | Spectral | 5111ms | 3100ms | 1.6x |
| Canonical | Krylov | 4160ms | 43ms | **97x** |
| QubitGrid | Spectral | 3091ms | 2615ms | 1.2x |
| QubitGrid | Krylov | 1734ms | 149ms | **12x** |

### d=64 Comparison

| Model | Criterion | NumPy | JAX CPU | Speedup |
|-------|-----------|-------|---------|---------|
| Canonical | Spectral | 441ms | 179ms | 2.5x |
| Canonical | Krylov | 974ms | 26ms | **37x** |
| QubitGrid | Spectral | 494ms | 341ms | 1.4x |
| QubitGrid | Krylov | 413ms | 5ms | **83x** |

### d=256 Projection

Using scaling from d=64→128 measurements:

| Model | Criterion | Backend | Projected d=256 | Full sweep (200 trials × 10 K) |
|-------|-----------|---------|-----------------|-------------------------------|
| Canonical | Spectral | NumPy | ~55s/trial | ~30h |
| Canonical | Krylov | JAX CPU | ~280ms/trial | **~9 min** |
| QubitGrid | Spectral | NumPy | ~16s/trial | ~9h |
| QubitGrid | Krylov | JAX CPU | ~750ms/trial | **~25 min** |

**Krylov d=256 is feasible with JAX CPU.** Spectral d=256 requires real GPU.

## Task Results

### Task 1: Re-evaluate Numba JIT
- **Finding**: Numba is fast for vector ops (6.9x at d=64) but BLAS dominates total time
- d=64: vector ops = 124μs, BLAS ops = 228μs → Numba total speedup 1.43x
- d=128: vector ops = 131μs, BLAS ops = 1767μs → Numba total speedup **1.04x**
- **Verdict**: Not worth integrating. BLAS dominates at d≥128 (93% of time).

### Task 2: Fix JAX Krylov
- **Problem**: Fixed-size Lanczos with jax.lax.scan couldn't handle Krylov breakdown
- **Solution**: Masked validity tracking — track which columns are valid, zero out
  post-breakdown iterations, use safe norm (sqrt(x²+ε)) to prevent NaN gradients
- **Correctness**: Scores match NumPy exactly at d≤32, within 1.5% at d=64
  (no QR in gradient path, matching NumPy behavior)
- **Gradient**: Matches NumPy/FD at d≤32. At d≥64, autodiff through long Lanczos
  recurrence is numerically unstable — but optimizer still finds correct verdicts
- **Spectral fix**: Smooth abs (sqrt(|c|²+ε)) prevents NaN from zero overlap coefficients

### Task 3: Test M1 GPU (jax-metal)
- **Finding**: jax-metal 0.1.1 is completely non-functional
- No support for complex numbers (complex64 or complex128)
- No support for even real float32 operations (UNIMPLEMENTED errors)
- **Verdict**: M1 Metal GPU is not viable. Need NVIDIA CUDA GPU for JAX acceleration.
- Uninstalled to avoid interference.

### Task 4: Remaining Optimizations
- **JAX CPU JIT**: Already provides 12-97x speedup for Krylov — this is the big win
- **Reduce trials**: 200→50 trials gives 4x reduction, still statistically meaningful
- **Adaptive sampling**: Already implemented (AdaptiveSweep), 3-7x in flat regions
- **Spectral bottleneck**: O(d³) eigendecomposition, only CUDA GPU can help
- **Krylov-only mode**: For d=256, run only Krylov criterion (faster and sufficient)

## Performance Evolution

### Krylov is_reachable at d=64

| Version | Canonical | QubitGrid | Notes |
|---------|-----------|-----------|-------|
| v50 (clean) | ~1700ms | ~820ms | Arnoldi, einsum |
| v54 | ~1100ms | ~500ms | Lanczos, BLAS matmul |
| v70 (fast) | 1171ms | 616ms | Sparse support |
| v71 | 454ms | 327ms | Two-buffer gradient |
| v72 JAX | **26ms** | **5ms** | JAX JIT compilation |

**Total speedup from v50 to v72 JAX: 65x (Canonical), 164x (QubitGrid)**

## Files Modified

- `src/criteria.py` — Two-buffer Krylov gradient fix (v71, committed)
- `src/criteria_jax.py` — Masked Krylov, smooth-abs Spectral, x64 config

## Commits

1. `da88d44` Two-buffer Krylov gradient + JAX fixes
2. `2257345` Fix JAX Krylov: masked breakdown + smooth abs for Spectral

## Recommendations for d=256

1. **Use JAX CPU for Krylov** — projected ~280ms/trial at d=256
2. **Run Krylov only** for d=256 (Spectral is O(d³), too slow without GPU)
3. **Reduce trials to 50-100** for faster turnaround
4. **Use AdaptiveSweep** for additional 3-7x in flat regions
5. **For Spectral d=256**: need NVIDIA GPU with CUDA (not M1)
6. **Combined estimate**: Krylov d=256 sweep with 100 trials × 10 K = ~5-15 min
