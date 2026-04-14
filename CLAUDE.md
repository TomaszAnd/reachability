# CLAUDE.md - Project Context

## Overview

Quantum reachability analysis: testing whether target quantum states are reachable
under parameterized Hamiltonians using spectral, Krylov, and moment criteria.

## Current State (v0.4.5 / v100 sweep)

- Branch: `fast` (pushed to GitHub)
- All 65 tests passing (34 original + 10 audit + 8 extended + 6 rho_c + 7 v0.4.4)
- Class-based API, pure numpy (no qutip dependency)
- Optional JAX backend for GPU acceleration

### Active Task: v100 Production Sweep

IMPORTANT: The v96 sweep process should be killed before launching v100.
Check `logs/v96_resume_can.pid` and kill it.

v100 sweep improvements over v96:
1. Per-criterion early stopping (don't sample Krylov at ρ >> ρ_c(Krylov))
2. ρ_max caps per (model, d) — no empty whitespace in plots
3. Consistent optimizer: L-BFGS-B when K_c < 100, random_polish when K_c ≥ 100
4. Fermi-Dirac curve fitting for Spectral/Krylov, exponential for Moment
5. Linear fit on ρ_c vs 1/d with confidence bands
6. Phased execution: d≤64 first, then d=128, then d=256

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
- **Eigendecomposition caching** in SpectralCriterion
- **Sparse Krylov gradient** — keeps H sparse throughout gradient loop
- **Chunked Spectral gradient** — processes in batches of 64 to limit memory
- **rho_c summary export** — CSV/JSON from sweep, `load_rho_c_summary()` helper
- **`random_polish` optimizer** — random search + short L-BFGS-B polish
- **Block sparse gradient** — `(K*d, d)` CSR block matrix replaces K-loops
- **`K_max` for CanonicalQuditModel** — truncates basis for d>=256

### Key Findings
1. **Criterion-specific fits**: Exponential for Moment (P(0)=1), Log-logistic for Spectral/Krylov
2. **Canonical**: K_c ~ αd for Moment (α=0.34) and Krylov (α=0.68), giving ρ_c ~ 1/d
3. **Canonical Spectral**: K_c ~ 0.590·d·ln(d) (R²=0.993, coupon-collector argument)
4. **QubitGrid Moment/Krylov**: K_c ≈ 2-3 (constant), ρ_c ~ 1/d²
5. **QubitGrid Spectral**: K_c ~ 0.601d (R²=0.997, includes d=128). Transition at d=128 confirmed (v105b, 20 restarts).
6. **QubitGrid K_max limitation**: K_max = 3n+9|E| grows polynomially, K_c grows exponentially. d=128 borderline, d≥256 infeasible.

### CRITICAL: Known Issues
- **Moment non-monotonicity**: Expected physics — sufficient condition, not necessary. Fit the descending envelope only.

### QubitGrid Spectral at d≥128

K_max for QG = 3·n_qubits + 9·|E(lattice)|. Grows polynomially in n_qubits,
while K_c ~ 0.601·d grows exponentially.

| d   | Best lattice | K_max | K_c(fit) | Status |
|-----|-------------|-------|----------|--------|
| 128 | 1×7         | 75    | 78.3     | Transition onset visible (P=0.813 at K=75) |
| 256 | 2×4         | 114   | ~154     | Infeasible (gap=-40) |

v105b probe (20 L-BFGS-B restarts, 150 trials): transition found at d=128!
P drops from 1.0 (K≤66) to 0.813 (K=75). K_c=78.3 just beyond K_max=75.
Previous probes (v104, 5 restarts) missed it — needed 20 restarts to resolve.
K_c=0.601d scaling (R²=0.997) now includes d=128 data point.
d=256: K_max=114 << K_c≈154, no lattice geometry helps.

### v104/v105 Sweep Infrastructure

- d=128 model: ~400 MB (Can) / ~120 MB (QG sparse)
- d=256 model: ~1.5 GB (K_max=1500 dense 256×256 complex matrices)
- **NEVER run d=128 and d=256 sweeps simultaneously** — OOM risk
- Incremental CSV saving: crash-safe, auto-resume on restart
- v105b adaptive tiers: light effort (25 trials) for plateau/tail, heavy (150 trials,
  20 restarts) for critical zone. Cuts runtime 3-4x vs uniform settings.
- Restart cap fix in criteria.py: `max(restarts, 12)` allows caller to request >12

### Sweep Scripts
- `sweep_v104_high_stats.py`: phased d=128/256, `--n-hams`/`--n-targets` CLI
- `sweep_v105b_adaptive.py`: L-BFGS-B only, tiered effort, `--run d128/2/all`

### Krylov L-BFGS-B Timing
- d=128: ~6s/restart (29s total with 5 restarts)
- d=256: ~25s/restart (125s total with 5 restarts) — very expensive

## Key Files

| File | Purpose |
|------|---------|
| `src/models.py` | `QuantumModel` ABC, `CanonicalQuditModel`, `QubitGridModel` |
| `src/criteria.py` | `ReachabilityCriterion` ABC, `OptimizableCriterion`, `SpectralCriterion`, `KrylovCriterion`, `MomentCriterion` |
| `src/sampling.py` | `DensitySweep`, `SweepConfig`, `AdaptiveSweep`, `AdaptiveSweepConfig` |
| `src/math_utils.py` | `eigendecompose`, `compute_binomial_sem`, `clip_to_bounds` |
| `src/plotting.py` | Color schemes (DIM_COLORS, CRIT_COLORS) and fit functions |
| `src/backend.py` | Backend abstraction (numpy/jax) |
| `scripts/production/overnight_sweep_v101.py` | v101 sweep (6 dims, per-K timeout, two-pass K selection) |
| `scripts/production/overnight_sweep_v102.py` | **CURRENT** v102 sweep (25 even K, L-BFGS-B universal, 5min timeout) |
| `scripts/production/sweep_v104_high_stats.py` | v104 high-stats sweep (d=128/256, incremental save, resume, OOM-safe) |
| `scripts/production/sweep_v105b_adaptive.py` | v105b L-BFGS-B sweep with adaptive tiered effort |
| `scripts/plot_v100.py` | Plotting with curve fitting (used by v101/v102) |

## Naming Conventions

- `CanonicalQuditModel` (not CanonicalModel)
- `QubitGridModel` (not GeometricTwoLocalModel)
- `OptimizableCriterion` — intermediate ABC for Spectral/Krylov (has `_maximize`)
- K is the primary parameter (not rho = K/d^2)

## Style Guide

- Use `Edit` for targeted changes, `Read` before editing
- Prefer numpy arrays (no qutip.Qobj in interfaces)
- Use `np.random.SeedSequence.spawn()` for parallel-safe RNG
- NEVER use `.entropy` on spawned SeedSequence — use `.generate_state(1)[0]`
- CRIT_COLORS: moment=green (C2), spectral=blue (C0), krylov=orange (C1)
- CRIT_MARKERS: moment='o', spectral='s', krylov='^'
- Plots: NO TITLES. Error bars with capsize=3. Grid alpha=0.3.

## Curve Fitting Specifications

### Moment: Simple Exponential
```python
P(ρ) = exp(-ρ / λ)    # enforces P(0) = 1
ρ_c = λ · ln(2)        # P(ρ_c) = 0.5
```
- Physically motivated: random matrix decay
- Fermi-Dirac gives P(0) < 1 for Moment (e.g. 0.79 at Can d=32) — do NOT use
- QG Moment: step function (vertical dashed line) when <3 transition points

### Spectral/Krylov: Log-Logistic
```python
P(ρ) = (ρ_c/ρ)^β / (1 + (ρ_c/ρ)^β)
```
- P(0) = 1 exactly, P(ρ_c) = 0.5, naturally asymmetric (flat plateau → sharp drop)
- Fit parameters: ρ_c, β (steepness)
- Plot as semi-transparent line (alpha=0.5) over data points
- Fit on transition region mask (0.02 < P < 0.98) via scipy curve_fit
- If fit fails (<3 transition points or R²<0.5): vertical dashed line at ρ_c

### Power-Law Scaling (ρ_c vs d)
```python
ρ_c = a * d^(-α)
```
- Log-log regression, weighted by 1/CI_width
- Canonical Moment/Krylov: α ≈ 1 (ρ_c ~ 1/d)
- Canonical Spectral: α ≈ 0.65 (K_c super-linear ~d^1.35)
- QubitGrid Moment/Krylov: α ≈ 2 (K_c constant, ρ_c ~ 1/d²)
- QubitGrid Spectral: α ≈ 0.9 (ρ_c ~ 1/d)

### K_c vs d
- K_c = α·d (forced through origin) for linear series
- Horizontal line at K_c = const for QG Moment/Krylov
- Both on log-log axes

## Optimizer Selection Logic

```
spectral_method selection:
  K_c < 100  →  'L-BFGS-B'       (small K: need precise gradient optimization)
  K_c ≥ 100  →  'random_polish'   (large K: random search + polish)
```

CRITICAL: K_max random sampling (v0.4.6) — CanonicalQuditModel with K_max now randomly
samples from the full d² operator types (X, Y, Z). The old sequential truncation produced
only X-type operators concentrated in rows 0-5, making Spectral scores cap at ~0.6.
With balanced sampling, random_polish achieves score ≥ 0.99 at d=256 for K ≥ 800.

IMPORTANT: QubitGrid d=128 has K_max=75, K_c≈85. Since K_c < 100, use L-BFGS-B.
The spectral_restarts should be 10 for all QubitGrid dims.

## ρ_max Caps (v100)

```python
RHO_MAX = {
    'canonical': {16: 0.20, 32: 0.20, 64: 0.10, 128: 0.06, 256: 0.03},
    'qubitgrid': {16: 0.08, 32: 0.04, 64: 0.02, 128: 0.008},
}
```

## Running Tests

```bash
python scripts/tests/run_all_tests.py         # Full (65 tests)
python scripts/tests/run_all_tests.py --quick  # Quick (~3s)
```

## Common Commands

```bash
# Launch v102 sweep (phased)
nohup python -u scripts/production/overnight_sweep_v102.py > logs/v102_sweep.log 2>&1 &
echo $! > logs/v102_sweep.pid

# Monitor
tail -f logs/v102_sweep.log

# Quick test (d=16 only, ~30 min)
python scripts/production/overnight_sweep_v102.py --quick

# Generate plots from existing data
python scripts/plot_v100.py
```

## Performance

### Spectral Method Timing at Various d

| d | K_c | Method | Time/trial | Notes |
|---|-----|--------|-----------|-------|
| 16 | 25 | L-BFGS-B | ~50ms | Fast, precise |
| 32 | 55 | L-BFGS-B | ~200ms | Still fast |
| 64 | 122 | L-BFGS-B | ~500ms | Acceptable |
| 128 | 273 | random_polish | ~2s | K_c≥100 threshold |
| 256 | ~700 | random_polish | ~90s | Balanced basis fix required |

### JAX Backend Selection

| Criterion | d <= 128 | d >= 256 |
|-----------|----------|---------|
| Krylov | JAX JIT (12-97x faster) | NumPy (autodiff unstable) |
| Spectral | NumPy (scipy eigh optimal) | NumPy |
| Moment | NumPy | NumPy |
