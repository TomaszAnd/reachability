# Session Summary v74: Comprehensive Optimization Audit

## Objective
Audit ALL remaining optimizations for d=256 feasibility, test M1 GPU backends,
implement memory optimizations.

## Key Results

### Task 1: M1 GPU Backend Testing

| Backend | eigh support | Complex128 | Speedup | Status |
|---------|-------------|------------|---------|--------|
| MLX 0.30.6 | CPU only (not GPU) | No (complex64 only) | 1.5x (f32 CPU) | Unusable for GPU |
| PyTorch MPS | Not implemented | No (no float64 on MPS) | N/A | Dead end |
| jax-metal | Broken | No | N/A | Broken (v72) |

**MLX details:**
- GPU eigh: "not yet supported on GPU" — must use CPU stream
- CPU eigh (complex64): 1.5x faster than NumPy at d=256 (31ms vs 49ms)
- GPU matmul (complex64): 5.7x faster than NumPy (1.2ms vs 6.8ms)
- Float32 score accuracy: 1.6e-8 error — fine for verdicts
- **But**: even with 1.5x eigh, Spectral d=256 is still ~360 hours. Not viable.

**Conclusion: No M1 GPU solution exists for eigendecomposition. CUDA GPU required.**

### Task 2: Reduced Krylov m

| m | Score (avg) | Verdict | Time |
|---|------------|---------|------|
| 32 | 0.486 | All UNREACHABLE | 4.2s |
| 64 | 0.611 | All UNREACHABLE | 10.4s |
| 128 | 0.842 | All UNREACHABLE | 21.8s |
| 192 | 0.999 | All REACHABLE | 16.7s |
| 256 | 1.000 | All REACHABLE | 6.1s |

- m<d gives **wrong verdicts** (false UNREACHABLE)
- m<d is also **slower** because optimizer works harder on lower scores
- **Verdict: m=d is required. No speedup possible via reduced m.**

### Task 3: Lazy Dense Basis (Memory Optimization)

**Implemented**: QubitGrid submodels no longer build dense basis arrays.

Before:
- `sample_submodel()` converted all sparse ops to dense: 60MB at d=256
- `ReachabilityCriterion.__init__` stored dense `self.hams`: another 60MB
- Total: ~120MB per criterion

After:
- `sample_submodel()` keeps sparse only: `sub._basis = None`
- `ReachabilityCriterion` stores `self.hams = None` when sparse available
- Dense array only built if explicitly needed (via lazy `_hams_array` property)
- All evaluate paths use sparse matvecs for QubitGrid

**Memory savings: ~120MB per criterion at d=256 (K=60)**
- Before: ~160MB peak
- After: ~43MB total, 157MB peak (dominated by optimizer temporaries)

### Task 4: Buffer Reuse Assessment

Measured allocation overhead vs compute at d=256:
- `np.zeros()` allocation: 0.04ms per call
- `array.fill(0)` reuse: 0.07ms per call (SLOWER)
- One BLAS matmul: 0.37ms

**Verdict: Not worth implementing. numpy's allocator is already efficient.**

## Files Modified

- `src/criteria.py` — Lazy dense basis: `self.hams = None` when sparse available,
  `_hams_array` property builds from sparse on demand
- `src/models.py` — `QubitGridModel.sample_submodel()` no longer converts to dense;
  `_SubModel.sample_submodel()` no longer forces dense conversion

## Summary Table

| Optimization | Impact | Verdict |
|-------------|--------|---------|
| MLX GPU eigh | Not supported on GPU | Not viable |
| PyTorch MPS eigh | Not implemented | Not viable |
| MLX CPU eigh (f32) | 1.5x faster | Still ~360h for Spectral d=256 |
| Reduced Krylov m | Wrong verdicts | Not viable |
| Lazy dense basis | 120MB saved | **Implemented** |
| Buffer reuse | -0.1% (slower) | Not worth it |

## Final d=256 Feasibility

| Model | Criterion | Backend | Time/trial | Sweep (1000 trials, 10 K) |
|-------|-----------|---------|------------|--------------------------|
| QubitGrid | Krylov | NumPy | 4.3s | **72 min** |
| QubitGrid | Moment | NumPy | ~1ms | **~10 sec** |
| QubitGrid | Spectral | NumPy | ~200s | ~550 hours |
| Canonical | ANY | ANY | N/A | OOM (K=65536) |

**QubitGrid Krylov+Moment d=256 is production-ready at ~72 min.**
**Spectral d=256 requires NVIDIA CUDA GPU — no M1 solution exists.**
