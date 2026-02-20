# Session Summary v70: Fast Branch Optimization

## Objective
Create optimized `fast` branch targeting d=256 in under 6 hours.

## Phases Completed

### Phase 1: Sparse Matrix Support
- **Files**: `src/models.py`, `src/criteria.py`
- QubitGrid basis stored natively as `scipy.sparse.csr_matrix`
- Sparse matvec in Lanczos: `H @ v` is O(nnz) not O(d²)
- Sparse H_k matvecs in Spectral and Krylov gradients
- Sparse Moment criterion avoids building dense `_hams_array`
- QubitGrid operators are 1.56% nonzero at d=64
- Added LATTICE_CONFIGS for d=128 (1x7) and d=256 (2x4)

### Phase 2: Vectorize Spectral Gradient
- Replaced per-k gradient loop with batched operations
- **Critical fix**: `einsum('di,kdj->kij')` → broadcast `Uc @ HkU_all`
  - einsum: 57ms, broadcast matmul: 0.6ms (100x faster at d=64)
  - Root cause: einsum wasn't dispatching to BLAS
- Spectral `is_reachable` d=64: **9076ms → 344ms (26x speedup)**

### Phase 3: Adaptive Sampling
- **Files**: `src/sampling.py`
- New `AdaptiveSweep` and `AdaptiveSweepConfig` classes
- Batch processing with early termination based on SEM confidence
- Zero-detection stops when all criteria show P < threshold
- Reduces trials from 150 to ~10-20 in flat regions

### Phase 4: Optional JAX Backend
- **Files**: `src/backend.py`, `src/criteria_jax.py`
- `REACHABILITY_BACKEND=jax` environment variable
- JIT-compiled Spectral and Krylov scores with `jax.grad` autodiff
- Optional dependency — not imported unless requested

### Phase 5: Parallel Optimization Restarts
- **Files**: `src/criteria.py`
- `ThreadPoolExecutor` for parallel multi-restart optimization
- `parallel=True` parameter in `is_reachable()`
- NumPy/BLAS releases GIL during matrix operations

## Benchmark Results (d=64, K=32, maxiter=50, restarts=2)

| Model | Criterion | Before (clean) | After (fast) | Speedup |
|-------|-----------|----------------|--------------|---------|
| Canonical | Spectral | ~9100ms | 344ms | **26x** |
| Canonical | Krylov | ~1700ms | 1171ms | 1.5x |
| QubitGrid | Spectral | ~9800ms | 487ms | **20x** |
| QubitGrid | Krylov | ~820ms | 616ms | 1.3x |

Note: Krylov speedup is modest because the Lanczos iteration is already O(d²m)
and sparse helps mainly in the forward pass (gradient still needs dense H for
batch dV computation). The Spectral speedup is dramatic because the critical
einsum bottleneck was not using BLAS.

## Combined Speedup with Adaptive Sampling

For a full sweep, the adaptive sampling provides an additional 3-7x reduction
by avoiding trials in P≈0 and P≈1 regions. Combined:
- Spectral sweep: ~26x criterion + ~5x sampling = **~130x total**
- Krylov sweep: ~1.5x criterion + ~5x sampling = **~7.5x total**

## API Changes

### New exports
- `src.sampling.AdaptiveSweep`, `src.sampling.AdaptiveSweepConfig`
- `src.backend` module (numpy/jax abstraction)
- `src.criteria_jax.SpectralCriterionJAX`, `src.criteria_jax.KrylovCriterionJAX`

### New parameters
- `is_reachable(parallel=True)` — parallel multi-restart optimization
- `QuantumModel.basis_sparse` — sparse basis property
- `QuantumModel.has_sparse` — whether sparse basis is available

### Version
- `src.__version__` updated to `"0.4.0"`

## Known Issues / Limitations
1. **JAX not tested**: No JAX installation available on this machine
2. **Parallel restarts**: Limited benefit at small d (BLAS already efficient)
3. **Krylov gradient sparse**: The gradient still needs dense H for batch dV;
   fully sparse gradient would require restructuring the Lanczos differentiation
4. **d=256 timing not measured**: Needs actual production run to verify 6h target

## Code Structure (after optimization)
```
src/
├── __init__.py          # v0.4.0
├── backend.py           # NEW: numpy/jax abstraction
├── models.py            # MODIFIED: sparse basis support, d=128/256 configs
├── criteria.py          # MODIFIED: sparse Lanczos, vectorized grad, parallel restarts
├── criteria_jax.py      # NEW: JAX implementations (optional)
├── sampling.py          # MODIFIED: AdaptiveSweep
├── math_utils.py        # Unchanged
└── plotting.py          # Unchanged
```

## Next Steps
1. Run production benchmarks at d=128 and d=256
2. Test JAX backend on GPU hardware
3. Consider Numba JIT for Lanczos inner loop (Phase 4 from original plan)
4. Profile and optimize Krylov gradient for large d
