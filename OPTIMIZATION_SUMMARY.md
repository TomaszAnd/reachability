# Optimization Summary

This document summarizes all performance optimizations implemented in the reachability analysis codebase.

## Overview

| Version | Optimization | Component | Speedup | Status |
|---------|--------------|-----------|---------|--------|
| v47 | np.tensordot for H(lambda) | All criteria | ~1.5x | Done |
| v47 | Early termination (score >= tau) | Spectral/Krylov | ~1.5x | Done |
| v47 | Vectorized Krylov gradient | Krylov | ~3x | Done |
| v47 | Vectorized Moment Q matrix | Moment | ~2x | Done |
| v51 | numpy.linalg.eigh (replace scipy) | Spectral | 1.2-1.9x | Done |
| v51 | Relax ftol 1e-8 -> 1e-6 | Optimizer | ~21% | Done |
| v52 | einsum -> BLAS matmul | Krylov gradient | 1.6-2.2x | Done |
| v52 | run_parallel() with joblib | DensitySweep | ~Nx* | Done |
| v54 | Lanczos iteration (replace Arnoldi) | Krylov | 1.5-2.0x | Done |
| v56 | Lazy _hams_array initialization | Memory | N/A | Done |

*Thread-based parallelism shows no benefit due to numpy BLAS internal threading.
Process-based parallelism would help for larger workloads.

## Cumulative Speedup

### Measured (v53 benchmark, d=8,12,16)
- Before (v50): 346.4s total
- After (v53 sequential): 217.4s total
- **Sequential speedup: 1.6x**

### Per-Component Improvements

#### Krylov Criterion (v54 Lanczos)
| Dimension | Before (Arnoldi) | After (Lanczos) | Speedup |
|-----------|------------------|-----------------|---------|
| d=8 | 1.2ms | 0.8ms | 1.5x |
| d=16 | 3.8ms | 2.4ms | 1.6x |
| d=32 | 58.6ms | 29.9ms | 2.0x |

The speedup improves with dimension because Lanczos gradient is O(K*d*m) vs Arnoldi's O(K*d*m^2).

#### Eigendecomposition (v51)
| Dimension | scipy.linalg.eigh | numpy.linalg.eigh | Speedup |
|-----------|-------------------|-------------------|---------|
| d=8 | 0.035ms | 0.019ms | 1.9x |
| d=16 | 0.067ms | 0.046ms | 1.5x |
| d=32 | 0.29ms | 0.24ms | 1.2x |

## Technical Details

### v47: Vectorization
- Replaced Python loops with numpy operations
- Used `np.tensordot` for Hamiltonian construction: `H = np.tensordot(lambdas, hams_array, axes=(0, 0))`
- Vectorized gradient computations across all K parameters simultaneously

### v51: Eigendecomposition & Optimizer
- Benchmarked scipy vs numpy eigh; numpy faster for d <= 128
- Relaxed optimizer tolerance (ftol 1e-8 -> 1e-6) with no verdict changes

### v52: BLAS Matmul
- Replaced `np.einsum('kdi,d->ki', ...)` with `.conj().transpose(0,2,1) @ w`
- BLAS level-3 operations are heavily optimized on modern CPUs

### v54: Lanczos Iteration
- Replaced Arnoldi (full Gram-Schmidt) with Lanczos (3-term recurrence)
- For Hermitian H, Lanczos is mathematically equivalent but faster
- Each step uses only 2 previous vectors instead of all previous vectors
- Gradient differentiation: O(K*d) per step vs Arnoldi's O(K*d*j)

### v56: Memory Optimization
- Lazy `_hams_array` initialization: stacked array only built when first needed
- Avoids duplicate storage: `self.hams` list no longer eagerly copied to array
- `clear_cache()` method for explicit memory release after computation

## Scaling Estimates

| d | K | Spectral/trial | Krylov/trial | Est. 200 trials |
|---|---|----------------|--------------|-----------------|
| 8 | 16 | 12ms | 0.8ms | 3s |
| 16 | 64 | 32ms | 2.4ms | 7s |
| 32 | 256 | 235ms | 30ms | 53s |
| 64 | 1024 | ~880ms* | ~150ms* | ~3.4min |
| 128 | 4096 | ~3900ms* | ~1000ms* | ~16min |

*Extrapolated. Spectral scales as d^2.15, Krylov as d^2.78.

## Memory Usage

The dominant memory consumer is `_hams_array` (K x d x d x 16 bytes for complex128):

| d | K | _hams_array size |
|---|---|-----------------|
| 8 | 64 | 64 KB |
| 16 | 256 | 1 MB |
| 32 | 1024 | 16 MB |
| 64 | 4096 | 256 MB |

With lazy initialization (v56), this array is only allocated when gradient
computation is needed, and can be explicitly freed via `clear_cache()`.
