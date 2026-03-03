# CLAUDE.md - Project Context

## Overview

Quantum reachability analysis: testing whether target quantum states are reachable
under parameterized Hamiltonians using spectral, Krylov, and moment criteria.

## Current State (v0.4.3)

- Branch: `fast` (pushed to GitHub)
- All 58 tests passing (34 original + 10 audit + 8 extended + 6 rho_c)
- Class-based API, pure numpy (no qutip dependency)
- Optional JAX backend for GPU acceleration

### Completed Features
- Class-based API (`QuantumModel`, `*Criterion`, `DensitySweep`)
- Lanczos-based Krylov iteration (O(Kd²) vs O(Kd²m) Arnoldi)
- **Sparse matrix support** for QubitGrid (1.56% nonzero at d=64)
- **Vectorized Spectral gradient** (26x speedup via BLAS matmul)
- **Adaptive sampling** (`AdaptiveSweep`) with early termination
- **Optional JAX backend** for GPU acceleration
- **Parallel optimization restarts** via ThreadPoolExecutor
- LATTICE_CONFIGS for d=128 (1x7) and d=256 (2x4)
- **Lazy dense basis** — QubitGrid avoids building dense arrays (saves ~120MB at d=256)
- Lazy Hamiltonian array construction (`_hams_array_cache`)
- numpy.eigh eigendecomposition (faster than scipy for d≤128)
- Publication-quality plotting with Fermi-Dirac fits
- Two Hamiltonian families: CanonicalQudit and QubitGrid
- Comprehensive Krylov analysis notebook explaining QubitGrid P=0
- **Separate `spectral_restarts`** field in SweepConfig (5-10, vs Krylov's 3)
- **Eigendecomposition caching** in SpectralCriterion (avoids redundant eigh during line search)
- **Sparse Krylov gradient** — keeps H sparse throughout gradient loop
- **Chunked Spectral gradient** — processes in batches of 64 to limit memory
- **QubitGrid d=256** in v96 sweep (Moment+Krylov only, Spectral infeasible)
- **rho_c summary export** — CSV/JSON from v96 sweep, `load_rho_c_summary()` helper

### Key Findings
1. **Spectral/Krylov**: Show Fermi-Dirac phase transitions P(ρ) = 1/(1+exp((ρ-ρc)/Δ))
2. **Moment**: Shows exponential decay P(ρ) = exp(-ρ/λ)
3. **QubitGrid Krylov**: Always P=0 (generic Paulis span full Hilbert space)
4. **Scaling**: ρc decreases with d as expected

## Key Files

| File | Purpose |
|------|---------|
| `src/models.py` | `QuantumModel` ABC, `CanonicalQuditModel`, `QubitGridModel` |
| `src/criteria.py` | `ReachabilityCriterion` ABC, `OptimizableCriterion`, `SpectralCriterion`, `KrylovCriterion`, `MomentCriterion` |
| `src/sampling.py` | `DensitySweep`, `SweepConfig`, `AdaptiveSweep`, `AdaptiveSweepConfig` |
| `src/math_utils.py` | `eigendecompose`, `compute_binomial_sem`, `clip_to_bounds` |
| `src/plotting.py` | Color schemes (DIM_COLORS, CRIT_COLORS) and fit functions |
| `src/backend.py` | Backend abstraction (numpy/jax) |
| `src/criteria_jax.py` | JAX-accelerated criteria (optional) |
| `scripts/benchmark.py` | Performance benchmark suite |
| `notebooks/quickstart.ipynb` | Interactive demo (~10-15 min) |
| `notebooks/moment_nonmonotonicity_analysis.ipynb` | Moment non-monotonicity analysis |
| `notebooks/krylov_qubitgrid_analysis.ipynb` | Why QubitGrid Krylov is always P=0 |

## Naming Conventions

- `CanonicalQuditModel` (not CanonicalModel)
- `QubitGridModel` (not GeometricTwoLocalModel)
- `OptimizableCriterion` — intermediate ABC for Spectral/Krylov (has `_maximize`)
- K is the primary parameter (not rho = K/d^2)

## Data Flow

- CSV data: `data/canonical/`, `data/qubitgrid/`
- Plots: `fig/canonical/`, `fig/qubitgrid/`

### Production Data
- v58 run: Canonical d=16,32,64 and QubitGrid d=16,32,64 with 200 trials
- v63 run: Canonical d=16,32 with 500 trials, improved K sampling (d=64 incomplete)

## Running Tests

```bash
python scripts/tests/run_all_tests.py         # Full (58 tests)
python scripts/tests/run_all_tests.py --quick  # Quick (~3s)
```

## Common Tasks

```bash
# Generate publication plots from overnight CSV data
python scripts/production/plot_overnight_v7style.py

# Run confusion matrix benchmark
python scripts/tests/confusion_matrix_benchmark.py

# Run production sweeps (auto JAX/NumPy backend)
python scripts/production/sweep_production.py --model qubitgrid --dims 16 32 64 128 --krylov-only
python scripts/production/sweep_production.py --model canonical --dims 16 32 64

# Run overnight production experiments (legacy)
python scripts/production/overnight_canonical.py
python scripts/production/overnight_qubitgrid.py
```

## Code Structure

```
src/
├── models.py         # QuantumModel, CanonicalQuditModel, QubitGridModel (sparse support)
├── criteria.py       # SpectralCriterion, KrylovCriterion, MomentCriterion (vectorized, sparse)
├── criteria_jax.py   # JAX-accelerated criteria (optional)
├── sampling.py       # DensitySweep, SweepConfig, AdaptiveSweep
├── backend.py        # numpy/jax backend abstraction
├── math_utils.py     # eigendecompose, Krylov utilities
├── plotting.py       # Fit functions, color schemes
└── __init__.py       # Package exports (v0.4.0)

scripts/
├── production/
│   ├── overnight_canonical.py   # Canonical production runs
│   ├── overnight_qubitgrid.py   # QubitGrid production runs
│   ├── sweep_production.py      # Production sweep (auto JAX/NumPy selection)
│   └── plot_overnight_v7style.py # Publication figure generation
└── tests/
    ├── run_all_tests.py            # 58 tests in 11 groups (A-K)
    └── confusion_matrix_benchmark.py

notebooks/
├── quickstart.ipynb                  # Getting started (~5-8 min)
└── krylov_qubitgrid_analysis.ipynb   # Krylov analysis

data/
├── canonical/    # Canonical model CSV results
└── qubitgrid/    # QubitGrid model CSV results
```

## Style Guide

- Use `Edit` for targeted changes, `Read` before editing
- Prefer numpy arrays (no qutip.Qobj in interfaces)
- Use `np.random.SeedSequence.spawn()` for parallel-safe RNG
- NEVER use `.entropy` on spawned SeedSequence — use `.generate_state(1)[0]`
- Publication plots: DIM_COLORS (8=blue, 16=orange, 32=green, 64=red)
- CRIT_COLORS (moment=green, spectral=blue, krylov=orange)

## Performance

### is_reachable() at d=64, K=32, maxiter=50, restarts=2

| Model | Spectral (NumPy) | Krylov (NumPy) | Krylov (JAX CPU) |
|-------|-----------------|----------------|------------------|
| Canonical | 441ms | 454ms | **26ms** |
| QubitGrid | 494ms | 327ms | **5ms** |

### is_reachable() at d=128, K=64, maxiter=50, restarts=2

| Model | Spectral (NumPy) | Krylov (NumPy) | Krylov (JAX CPU) |
|-------|-----------------|----------------|------------------|
| Canonical | 5111ms | 4160ms | **43ms** |
| QubitGrid | 3091ms | 1734ms | **149ms** |

### JAX Backend

JAX JIT provides 12-97x speedup for Krylov by compiling the entire Lanczos
loop into a single XLA computation. Spectral sees modest improvement (1.2-2.5x)
since eigendecomposition is already BLAS-optimized.

**Known JAX limitations:**
- Krylov gradient numerically unstable at d≥64 (autodiff through long recurrence)
- Spectral gradient NaN at some d values (eigenvalue degeneracy)
- M1 Metal GPU not supported (jax-metal broken for complex numbers)
- Verdicts match NumPy at all tested dimensions

### d=256 Feasibility

- **Canonical d=256**: K=d²=65536 operators, requires >64GB RAM — infeasible on 16GB
- **QubitGrid d=256 Krylov random**: ~11ms/trial — very fast, included in v96 sweep
- **QubitGrid d=256 Spectral**: ~41s/trial (is_reachable, r=10) — infeasible for overnight
- **QubitGrid d=256 Moment**: ~0.7ms/trial — very fast
- **v96 sweep strategy**: d≥256 uses Moment+Krylov only (Spectral skipped)
- **JAX at d=256**: SLOWER than NumPy (autodiff through 255 Lanczos steps unstable)
- **Strategy**: JAX for d≤128, NumPy for d≥256

### Eigendecomposition: scipy vs numpy on M1

On Apple M1 Accelerate, scipy default (heevr/MRRR) is **2x faster** than numpy (heevd)
at d≥256. `eigendecompose()` uses dimension-adaptive backend selection:
- d < 256: numpy.linalg.eigh (heevd, divide & conquer)
- d ≥ 256: scipy.linalg.eigh (heevr, MRRR, overwrite_a=True)

### Parallel Restarts on M1

- Accelerate's eigh is single-threaded per call
- ThreadPoolExecutor: 0.97x (BLAS thread contention)
- Multiprocessing (3 workers): 1.10x (memory bandwidth limited)
- Multiprocessing (4 workers): 0.92x (overhead > parallelism)
- Not viable on M1's unified memory architecture

### M1 GPU Dead Ends (v74)

No GPU acceleration for eigh on Apple Silicon:
- MLX: eigh "not yet supported on GPU", complex64 only (no complex128)
- PyTorch MPS: linalg.eigh not implemented, no float64
- jax-metal: broken for complex numbers
- All paths require NVIDIA CUDA for GPU eigendecomposition

### Spectral Bottleneck (v78 profiling)

At d=256 Spectral (12.3s total):
- 60% gradient computation (vectorized BLAS, mathematically irreducible)
- 18% eigh eigendecomposition
- 12% sparse matvec in gradient (H_k @ U)
- 9% einsum in gradient
- <1% sparse H construction (optimized via COO assembly in v78)

### Sparse H Construction

Uses COO assembly (5-12x faster than CSR loop):
- Pre-extracts COO data at `ReachabilityCriterion.__init__` time
- Vectorized scaling via `np.repeat(lambdas, nnz_per_op) * coo_data`
- Single COO→CSR conversion per call

### JAX Backend Selection

| Criterion | d <= 128 | d >= 256 |
|-----------|----------|---------|
| Krylov | JAX JIT (12-97x faster) | NumPy (autodiff unstable) |
| Spectral | NumPy (scipy eigh optimal) | NumPy |
| Moment | NumPy | NumPy |

### Benchmarks

```bash
python scripts/benchmark.py --quick              # Quick benchmark
python scripts/benchmark.py --dims 16 32 64 128  # Full benchmark
python scripts/benchmark_scaling.py              # Scaling analysis
python scripts/benchmark_platform.py             # Platform-specific eigh
python scripts/estimate_sweep_time.py            # Sweep time projection
```
## Recent Changes (v0.4.3)

1. **`spectral_restarts` field**: Added to `SweepConfig` and `AdaptiveSweepConfig`. Spectral L-BFGS-B needs 5-10 restarts (vs Krylov random search's 3) to avoid false UNREACHABLE verdicts. Dimension-adaptive: 5 for d≤32, 10 for d≥64.
2. **rho_c summary export**: v96 sweep saves `rho_c_summary.csv` and `rho_c_summary.json`. New `load_rho_c_summary()` helper in sampling.py.
3. **Eigendecomposition caching**: `SpectralCriterion._cached_eigendecompose()` keyed on `lambdas.tobytes()`. Avoids redundant eigh during L-BFGS-B line search (same lambdas → same H → same eigh).
4. **Sparse Krylov gradient**: Keeps H sparse throughout gradient loop instead of densifying via `.toarray()`. Sparse matvec for both forward (`H @ v_j`) and gradient (`H @ dV_curr.T`).
5. **Chunked Spectral gradient**: Processes gradient in batches of 64 operators. Limits peak memory from O(K·d²) to O(64·d²) for large K.
6. **QubitGrid d=256 in v96 sweep**: Moment+Krylov only (Spectral infeasible at 41s/trial). Krylov random: 11ms, Moment: 0.7ms. `--no-256` flag to skip.
7. **14 new tests** (Tests J+K): Extended coverage (8 tests) + rho_c estimation (6 tests). Total: 58 tests in 11 groups (A-K).

### Previous Changes (v0.4.2)

1. **CRITICAL BUG FIX**: Fixed `nwhatsp.linalg.qr` typo in `_lanczos_basis()`. All Krylov forward-only evaluations were crashing.
2. **_lanczos_basis refactored**: Now accepts `apply_qr: bool = True` parameter.
3. **10 new tests** (Test I): Audit coverage tests.
4. **v96 sweep script**: Two-pass K selection, publication-quality plots, bootstrap CIs, power-law fits.
5. **New functions**: `estimate_rho_c()` and `bootstrap_rho_c()` in sampling.py.

## Audit Findings (v0.4.2)

### CRITICAL BUG: criteria.py line 550
`nwhatsp.linalg.qr` is a typo for `np.linalg.qr`. Crashes all Krylov forward-only evaluations. Fix immediately.

### QR re-orthogonalization is essential
- Without QR, Lanczos orthogonality loss is catastrophic at d≥16 (‖V†V − I‖ up to 6.36)
- Score errors up to 0.42 — verdict-changing at τ=0.99
- Forward path (return_gradient=False) applies QR → correct scores
- Gradient path (return_gradient=True) does NOT apply QR → approximate scores (intentional: QR has discontinuous gradients)

### Krylov optimizer: Random-50 is optimal
Benchmarked at d=16,32 transition regions: Random-50 gives 100% verdict agreement with all other methods while being 9–45× faster. Do NOT use Nelder-Mead or Powell for Krylov. Spectral MUST use L-BFGS-B.

### Color scheme (canonical, must match between plotting.py and sweep scripts)
- Moment: green (#2ca02c), circle (o)
- Spectral: blue (#1f77b4), square (s)
- Krylov: orange (#ff7f0e), triangle (^)

### v96 sweep config
- AdaptiveSweepConfig.min_hamiltonians = 10 (reduced from 20)
- Two-pass K selection: coarse scan → dense around ρ_c
- Bootstrap ρ_c confidence intervals
- `spectral_restarts`: coarse pass=3, main pass=5 (d≤32) or 10 (d≥64)
- QubitGrid d=256: Moment+Krylov only (`--no-256` to skip)
- Outputs: per-dimension CSVs + `rho_c_summary.csv` + `rho_c_summary.json`
