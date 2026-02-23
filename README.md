# Quantum Reachability Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Test whether a target quantum state |ψ⟩ is reachable from initial state |φ⟩
under parameterized Hamiltonian H(λ) = Σ_k λ_k H_k,
without explicit time evolution.

## Installation

```bash
pip install numpy scipy pandas matplotlib
```

## Quick Start

```python
from src.models import CanonicalQuditModel
from src.criteria import SpectralCriterion

# Create model and sample K=10 operators
model = CanonicalQuditModel(dim=8, seed=42)
submodel = model.sample_submodel(10)

# Test reachability
phi = submodel.init_state()
psi = submodel.random_state()
result = SpectralCriterion(submodel, phi, psi, tau=0.99).is_reachable()
print(f"{result.verdict.value}: score={result.score:.3f}")
```

See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for a complete tutorial.

## Reachability Criteria

| Criterion | Method | Verdict |
|-----------|--------|---------|
| **Spectral** | Maximize spectral overlap S(λ) via L-BFGS-B | REACHABLE if S ≥ τ, else UNREACHABLE |
| **Krylov** | Maximize Krylov projection R(λ) via Lanczos + L-BFGS-B | REACHABLE if R ≥ τ, else UNREACHABLE |
| **Moment** | Check Q + γLL^T ≻ 0 (no optimization needed) | UNREACHABLE if definite, else INCONCLUSIVE |

**Note on Moment:** Unlike Spectral/Krylov, the Moment criterion tests a *sufficient* condition
for unreachability. Its measured P(unreachable) is **not guaranteed monotonically decreasing** in K,
because adding operators changes the Q and L matrices in ways that can make the positive
definiteness certificate harder to find.

## Monte Carlo Sweeps

Sweep over operator count K to measure P(unreachable) as a function of density ρ = K/d²:

```python
from src.models import CanonicalQuditModel
from src.sampling import DensitySweep, SweepConfig

model = CanonicalQuditModel(dim=16, seed=42)
config = SweepConfig(n_hamiltonians=20, n_targets=10, tau=0.99)
sweep = DensitySweep(model, config)
results = sweep.run(K_values=[5, 10, 15, 20, 25],
                    criteria=['moment', 'spectral', 'krylov'])
print(results[['K', 'rho', 'moment_P', 'spectral_P', 'krylov_P']])
```

## Hamiltonian Families

| Model | Basis | Dimension |
|-------|-------|-----------|
| `CanonicalQuditModel` | Generalized Pauli basis {X_jk, Y_jk, Z_j} | Any d |
| `QubitGridModel` | 1- and 2-local Paulis on nx×ny lattice | d = 2^(nx·ny) |

## Notebooks

- [`quickstart.ipynb`](notebooks/quickstart.ipynb) — Getting started with both models (~10-15 min)
- [`moment_nonmonotonicity_analysis.ipynb`](notebooks/moment_nonmonotonicity_analysis.ipynb) — Why Moment P(K) is non-monotonic
- [`krylov_qubitgrid_analysis.ipynb`](notebooks/krylov_qubitgrid_analysis.ipynb) — Why Krylov P=0 for QubitGrid

## Project Structure

```
src/                    Core library
├── models.py           QuantumModel, CanonicalQuditModel, QubitGridModel
├── criteria.py         SpectralCriterion, KrylovCriterion, MomentCriterion
├── sampling.py         DensitySweep, SweepConfig
├── math_utils.py       Eigendecomposition, Hamiltonian construction
└── plotting.py         Color schemes (DIM_COLORS, CRIT_COLORS)

notebooks/              Interactive tutorials
scripts/production/     Overnight experiments
scripts/tests/          Test suite (34 tests)
```

## Running Tests

```bash
python scripts/tests/run_all_tests.py         # Full suite (34 tests)
python scripts/tests/run_all_tests.py --quick  # Quick mode (~1s)
```

## Algorithm Selection

The code automatically selects optimal algorithms based on problem size.

### Eigendecomposition

| Dimension | Backend | LAPACK Driver | Rationale |
|-----------|---------|---------------|-----------|
| d < 256 | `numpy.linalg.eigh` | heevd (divide & conquer) | Lower overhead for small matrices |
| d >= 256 | `scipy.linalg.eigh` | heevr (MRRR) | 1.3-2x faster on Apple M1 Accelerate |

Threshold: `SCIPY_EIGH_THRESHOLD = 256` in `src/math_utils.py`.
On Intel/AMD CPUs the optimal driver may differ — run `scripts/benchmark_platform.py`
to determine the best backend for your hardware.

### Hamiltonian Construction

| Sparsity | Method | Speedup |
|----------|--------|---------|
| Sparse (QubitGrid) | COO assembly with vectorized scaling | 5-12x vs CSR loop |
| Dense (Canonical) | `np.tensordot` | N/A |

Dispatched automatically by `_construct_H()` in `src/criteria.py`.

### Optimizer

All optimization-based criteria (Spectral, Krylov) use L-BFGS-B:
- Bounded optimization (lambda in [-1, 1]^K)
- Combined objective+gradient via `jac=True` (avoids duplicate eigendecomposition)
- Multi-restart with early termination when score >= tau
- Default: `maxiter=200`, `restarts=2`, `ftol=1e-6`

**Adaptive restarts** ensure reliable convergence:

| Condition | Effective restarts | Rationale |
|-----------|-------------------|-----------|
| K < 10 | max(restarts, 12-K) | Narrow basins at small K |
| 0.05 < ρ < 0.30 | max(restarts, 8) | Hard landscape in transition |
| d ≥ 32 | restarts + 2·⌊(d-32)/32⌋ | Harder at large dimension |
| Always | ≥ 5 | Minimum for reliability |

### Reachability Criteria Behavior

| Criterion | Monotonic in K? | Speed | Notes |
|-----------|-----------------|-------|-------|
| Moment | **No** | Fastest | Sufficient condition; P can increase with K |
| Spectral | Yes (guaranteed) | Medium | max over λ expands feasible set |
| Krylov | Yes (guaranteed) | Medium | max over λ expands feasible set |

### Optional JAX Backend

Install `jax` for Krylov speedup at small dimensions via JIT compilation:

```bash
pip install jax jaxlib
```

| Criterion | d <= 128 | d >= 256 |
|-----------|----------|---------|
| Krylov | **JAX JIT: 12-97x faster** | NumPy (autodiff unstable) |
| Spectral | NumPy (scipy eigh optimal) | NumPy |
| Moment | NumPy | NumPy |

JAX is auto-detected by `scripts/production/sweep_production.py`.

## Performance

### Criterion Timing at d=256 (QubitGrid, Apple M1, maxiter=50, restarts=1)

| Criterion | Time/trial | Publication sweep (2,500 trials) |
|-----------|-----------|----------------------------------|
| Moment | ~1.5ms | ~4 sec |
| Krylov | ~8.5s | ~5.9 hours |
| Spectral | ~7.6s | ~5.3 hours |
| **Total** | | **~11.2 hours** |

### Recommended Sweep Configurations (d=256)

| Purpose | Trials | Approx. time |
|---------|--------|-------------|
| Quick test (20h x 3t x 5K) | 300 | ~1.3 hours |
| Publication (50h x 5t x 10K) | 2,500 | ~11.2 hours |
| Full (100h x 10t x 10K) | 10,000 | ~44.8 hours |

Use `scripts/estimate_sweep_time.py` to project times for your hardware.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{reachability2026,
  author = {TomaszAnd},
  title = {Quantum Reachability Analysis},
  year = {2026},
  url = {https://github.com/TomaszAnd/reachability}
}
```

See also [CITATION.cff](CITATION.cff).
