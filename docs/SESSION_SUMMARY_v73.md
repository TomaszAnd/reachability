# Session Summary v73: Spectral Optimization and Production Script

## Objective
Investigate Spectral speedup options, test d=256 feasibility, create production sweep script.

## Key Results

### Task 1: Spectral Bottleneck Analysis
- eigh eigendecomposition dominates: 73.8% (d=32), 97% (d=64), 98.1% (d=128), 99.1% (d=256)
- Float32 gives ZERO speedup on M1 (Accelerate uses same LAPACK for both)
- **Verdict**: No viable CPU Spectral optimization. Need CUDA GPU.

### Task 2: JAX Krylov Verdict Correctness
- Canonical d=16: 100%, d=32: 96%, d=64: 94%
- QubitGrid d=16-64: 100%
- Meets >95% threshold for production use (QubitGrid) and near-threshold for Canonical

### Task 3: d=256 Feasibility
- **Canonical d=256**: K=d²=65536 operators → >64GB RAM, infeasible on 16GB machine
- **QubitGrid d=256**: K=114, NumPy Krylov evaluate=238ms, is_reachable=4305ms
- **JAX at d=256**: 8619ms — SLOWER than NumPy (autodiff through 255 Lanczos steps unstable)
- **Strategy**: JAX for d≤128, NumPy for d≥256
- **Projected sweep**: QubitGrid d=256 Krylov-only, 100×10 trials, 10 K values ≈ 72 min

### Task 4: Production Sweep Script
- Created `scripts/production/sweep_production.py`
- Auto-selects JAX for d≤128, NumPy for d≥256
- Supports both CanonicalQudit and QubitGrid models
- CLI arguments for all parameters (model, dims, n-hams, n-targets, etc.)
- Krylov-only mode for large dimensions
- Adaptive sampling support
- Early stopping after consecutive zero-P K values
- Checkpoint saving per (d, tau) combination
- Tested: both QubitGrid+JAX and Canonical+all-criteria work correctly

### Comprehensive Benchmark (from v72)

| d | Model | Criterion | NumPy | JAX CPU | Speedup |
|---|-------|-----------|-------|---------|---------|
| 64 | Canonical | Spectral | 441ms | 179ms | 2.5x |
| 64 | Canonical | Krylov | 974ms | 26ms | **37x** |
| 64 | QubitGrid | Spectral | 494ms | 341ms | 1.4x |
| 64 | QubitGrid | Krylov | 413ms | 5ms | **83x** |
| 128 | Canonical | Spectral | 5111ms | 3100ms | 1.6x |
| 128 | Canonical | Krylov | 4160ms | 43ms | **97x** |
| 128 | QubitGrid | Spectral | 3091ms | 2615ms | 1.2x |
| 128 | QubitGrid | Krylov | 1734ms | 149ms | **12x** |

## Files Modified/Created

- `scripts/production/sweep_production.py` — NEW: Production sweep with auto backend selection
- `CLAUDE.md` — Updated with sweep script, d=256 findings, Spectral bottleneck notes
- `docs/SESSION_SUMMARY_v73.md` — This file

## Recommendations

1. **d≤128**: Use `sweep_production.py` with JAX Krylov (12-97x speedup)
2. **d=256 QubitGrid**: Use `sweep_production.py --krylov-only` with NumPy (~72 min for 1000 trials)
3. **d=256 Canonical**: Infeasible on 16GB — needs machine with >64GB RAM
4. **Spectral at d≥128**: Only viable with NVIDIA CUDA GPU
5. **Publication runs**: Use `--adaptive` flag for 3-7x reduction in flat regions
