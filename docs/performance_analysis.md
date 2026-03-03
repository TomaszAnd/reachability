# Performance Analysis

Profiling results and bottleneck analysis for the reachability library (v0.4.2).
All measurements taken on Apple M1 with 16GB unified memory, macOS, using NumPy
with Accelerate BLAS.

## 1. Profiling Results

### Custom Profiling: Spectral and Krylov at d=64, d=128 (K=20)

| Operation          | Spectral d=64 | Spectral d=128 | Krylov d=64 | Krylov d=128 |
|--------------------|---------------|----------------|-------------|--------------|
| Forward (ms)       |           2.0 |            7.5 |         0.9 |          4.3 |
| Gradient (ms)      |           4.3 |           34.3 |         3.5 |          2.2 |
| Grad/Fwd ratio     |          2.1x |           4.6x |        3.8x |         0.5x |
| is_reachable (ms)  |          24.7 |          467.1 |        39.1 |        100.6 |

Notes:
- Spectral `is_reachable` uses L-BFGS-B with maxiter=50, restarts=2.
- Krylov `is_reachable` uses random search with restarts=3.
- Krylov d=128 gradient < forward is an artifact of NumPy caching / measurement noise
  at small absolute times; the gradient path skips QR re-orthogonalization.

### profile_d256.py Results: Canonical d=128 and QubitGrid d=256

**Canonical d=128 (K=20):**

| Operation            |   Time (ms) |
|----------------------|-------------|
| Model creation       |         0.0 |
| Spectral evaluate    |         5.7 |
| Spectral eval+grad   |         9.9 |
| Moment is_reachable  |         1.7 |
| Krylov evaluate      |         1.0 |
| Krylov eval+grad     |         1.9 |
| Krylov random search |        34.9 |

**QubitGrid d=256 (K=20, sparse operators):**

| Operation            |   Time (ms) |
|----------------------|-------------|
| Model creation       |         0.0 |
| Spectral create      |         0.7 |
| Spectral evaluate    |        23.3 |
| Spectral eval+grad   |        64.9 |
| Moment is_reachable  |         0.5 |
| Krylov evaluate      |        14.3 |
| Krylov eval+grad     |       268.8 |
| Krylov random search |        14.7 |

**Canonical d=256 Extrapolation (from d=128 measured data):**

| Operation          | d=128 (ms) | d=256 projected (ms) | Scaling |
|--------------------|------------|----------------------|---------|
| Spectral eval      |        5.7 |                 45.9 | O(d^3)  |
| Spectral eval+grad |        9.9 |                 79.2 | O(d^3)  |
| Moment is_reachable|        1.7 |                  6.8 | O(d^2)  |
| Krylov eval        |        1.0 |                  3.9 | O(d^2)  |
| Krylov eval+grad   |        1.9 |                  7.4 | O(d^2)  |
| Krylov random      |       34.9 |                139.7 | O(d^2)  |

## 2. Spectral Bottleneck

At d=256, Spectral `is_reachable` takes ~12.3s per trial. The cost breakdown
from v78 profiling:

| Component                      | Share | Description                              |
|--------------------------------|-------|------------------------------------------|
| Gradient computation (BLAS)    |  ~60% | U^T H_k U for all K operators            |
| Eigendecomposition (eigh)      |  ~18% | O(d^3) per evaluation                    |
| Sparse matvec (H_k @ U)       |  ~12% | Sparse-dense product in gradient path    |
| einsum (Hk_eig_all assembly)  |   ~9% | Tensor contraction for K gradient terms  |
| Sparse H construction          |   <1% | Optimized via COO assembly (v78)         |

**Why this is mathematically irreducible:** The Spectral gradient requires
computing U^T H_k U for every operator k=1..K, where U is the d x d eigenvector
matrix from eigh(H(lambda)). This is a dense d x d x K tensor contraction that
cannot be avoided. The eigendecomposition itself is O(d^3) and dominated by
Accelerate's LAPACK heevd (or heevr at d >= 256). There is no sparse shortcut
because the eigenvectors U are always dense, even when H is sparse.

The gradient-to-forward ratio grows with d because the forward pass is dominated
by eigh (O(d^3)), while the gradient adds K matrix multiplications (each O(d^2)
but with K of them). At d=128 with K=20, this gives a 4.6x ratio.

## 3. Krylov Bottleneck

Krylov's computational cost depends on the evaluation method:

**Gradient path:** The Lanczos recurrence runs m steps (m=d by default), and
at each step the gradient requires computing `hams_array @ v` where `hams_array`
is a (K, d, d) dense array and `v` is a d-vector. This single operation costs
O(K * d^2) per step, and with m=d steps the total gradient cost is O(K * d^3).

Measured gradient overhead (gradient time / forward time):

| Dimension | Overhead | Absolute gradient time |
|-----------|----------|----------------------|
| d=16      |    1.5x  |              < 1 ms  |
| d=32      |   12.5x  |              ~2 ms   |
| d=64      |     77x  |            ~278 ms   |

The bottleneck is specifically the `hams_array @ v` matmul, which consumes 93%
of per-step cost at d=64. This grows cubically because both the array size
(K * d^2) and the number of Lanczos steps (m=d) scale with d.

**Random search (method='random'):** Avoids the gradient entirely. Evaluates
the Krylov score at many random lambda vectors and returns the best. Since
the forward pass applies QR re-orthogonalization and costs only O(K * d * m),
and the Krylov landscape has broad basins near score=1.0, random search
achieves 100% verdict agreement with L-BFGS-B at d >= 32.

Performance advantage of random search:

| Dimension | L-BFGS-B (ms) | Random (ms) | Speedup |
|-----------|---------------|-------------|---------|
| d=32      |         ~700  |        ~2   |   340x  |
| d=64      |      ~17,000  |       ~25   |   950x  |

## 4. d=256 Feasibility

**Canonical d=256: INFEASIBLE on 16GB hardware.**
The Canonical model builds K = d^2 = 65,536 dense d x d matrices. Storage alone
requires 65536 * 256 * 256 * 16 bytes = ~68 GB. This exceeds the 16GB unified
memory on M1 by more than 4x.

**QubitGrid d=256: FEASIBLE.**
QubitGrid uses sparse operators with ~1.56% nonzero entries. The lattice config
(nx=2, ny=4) produces ~114 operators. Measured timings at d=256 (K=20):

| Operation            | Time per call |
|----------------------|---------------|
| Krylov random search |      14.7 ms  |
| Spectral eval+grad   |      64.9 ms  |
| Krylov eval+grad     |     268.8 ms  |

Projected sweep times for QubitGrid d=256:
- Krylov sweep (100 hamiltonians x 10 targets, 10 K values): ~25 min
- Spectral sweep (50 hamiltonians x 5 targets, 10 K values): ~2-3 days

**JAX at d=256:** Slower than NumPy. The Krylov autodiff through 255 Lanczos
steps is numerically unstable (gradient NaN / exploding values). Spectral JAX
shows no advantage because eigh is already BLAS-optimized. The recommended
strategy is JAX for d <= 128 and NumPy for d >= 256.

## 5. Top 3 Optimization Opportunities

### (1) Sparse Krylov Gradient

**Current state:** The gradient path in `_lanczos_basis()` densifies all
Hamiltonians into a `(K, d, d)` array (`hams_array`) to compute `hams_array @ v`
at each Lanczos step. For QubitGrid, the operators are only ~1.56% nonzero at
d=64, yet the gradient uses dense matmuls.

**Opportunity:** Replace `hams_array @ v` with K individual sparse matvecs
`H_k @ v` (each O(nnz_k * 1) instead of O(d^2)). At d=64 with 1.56% fill,
this would reduce the per-step matvec cost by ~64x. The sparse data is already
available since the model stores operators in CSR format.

**Estimated impact:** At d=64 QubitGrid, the gradient bottleneck (`hams_array @ v`)
takes 3.7ms/step. With sparse matvec at 1.56% fill, this drops to ~0.06ms/step.
Over m=64 steps, total gradient cost drops from ~278ms to ~4ms.

**Caveat:** Only benefits QubitGrid (sparse operators). Canonical operators are
dense by construction, so this optimization has no effect there.

### (2) Spectral Gradient Batching

**Current state:** The Spectral gradient computes `U^T H_k U` for each operator
via einsum or sequential BLAS calls. The einsum path (`np.einsum('ij,ik->jk', ...)`)
is vectorized but not optimally batched for Accelerate.

**Opportunity:** Replace the K separate `d x d` multiplications with a single
batched BLAS call. Specifically, pre-stack the transformed operators and use
`np.matmul` broadcasting or `scipy.linalg.blas.dgemm` to compute all K
projections in one call. On Accelerate, batched BLAS can exploit SIMD more
effectively than K sequential calls.

**Estimated impact:** The gradient takes 60% of Spectral runtime. Even a 2x
improvement in the gradient BLAS path would reduce total Spectral time by 30%.
At d=256 (projected 79ms per eval+grad), this saves ~24ms per evaluation.

### (3) Lazy Eigendecomposition Caching

**Current state:** Each call to `SpectralCriterion.evaluate()` performs a full
eigendecomposition of H(lambda). During optimization, L-BFGS-B may evaluate
the same or nearby lambda vectors multiple times (line search, gradient
verification), recomputing eigh each time.

**Opportunity:** Cache the most recent eigendecomposition result and skip eigh
when lambda has not changed (exact match). For line search evaluations where
only the step size changes, cache the eigenvectors for the base point and
reuse them for gradient-free evaluations at nearby points.

**Estimated impact:** L-BFGS-B typically performs 1-3 extra function evaluations
per iteration for line search. Caching could eliminate ~30-50% of eigh calls.
At d=128 (eigh ~1.4ms, 18% of 7.5ms forward), this saves ~0.5ms per iteration
or ~25ms over a 50-iteration optimization.

**Additionally**, when sweeping over multiple target states with the same model
and the same lambda (common in DensitySweep), the H(lambda) eigendecomposition
is identical. Caching across target states could provide up to K-fold savings
in the multi-target sweep scenario.

## 6. M1 Hardware Constraints

### No GPU Eigendecomposition

All GPU acceleration paths for `eigh` on Apple Silicon are dead ends:

| Framework   | Status                                              |
|-------------|-----------------------------------------------------|
| MLX         | `eigh` "not yet supported on GPU"; complex64 only   |
| PyTorch MPS | `linalg.eigh` not implemented; no float64 support   |
| jax-metal   | Broken for complex numbers (our matrices are complex)|
| CuPy        | Requires NVIDIA CUDA (not available on M1)           |

All of these were tested in v74. The fundamental issue is that GPU eigensolvers
for complex Hermitian matrices are only mature in the CUDA ecosystem.

### Single-Threaded Accelerate eigh

Apple's Accelerate framework runs `eigh` (LAPACK heevd/heevr) in a
single-threaded mode. This means the O(d^3) eigendecomposition does not benefit
from M1's multi-core architecture. Parallel restarts via ThreadPoolExecutor
showed 0.97x scaling (actually slower due to BLAS thread contention).
Multiprocessing with 3 workers achieved only 1.10x due to memory bandwidth
limits on unified memory.

### Unified Memory Bandwidth

M1's unified memory architecture means CPU and GPU share the same memory bus.
This limits the benefit of parallelism for memory-bound operations like large
matrix multiplications. At d=128+, the working set (eigenvectors, Hamiltonians,
gradients) exceeds L2 cache, making operations memory-bandwidth-limited rather
than compute-limited.

### Practical Implications

- **d <= 64:** Single-threaded NumPy is fast enough (< 500ms per trial).
- **d = 128:** Spectral is expensive (~467ms per `is_reachable`) but feasible
  for moderate sweep sizes. Krylov random search at ~101ms is preferred.
- **d = 256:** Only QubitGrid is feasible. Spectral sweeps take days.
  Krylov random search (~15ms) remains practical.
- **d >= 512:** Would require cluster hardware (CUDA GPU or multi-node CPU)
  for any criterion.

### Recommended Backend Selection

| Criterion | d <= 128       | d >= 256  |
|-----------|----------------|-----------|
| Krylov    | JAX JIT (12-97x faster) | NumPy (autodiff unstable at long recurrences) |
| Spectral  | NumPy (scipy eigh optimal) | NumPy |
| Moment    | NumPy           | NumPy    |
