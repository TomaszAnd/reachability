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

- [`quickstart.ipynb`](notebooks/quickstart.ipynb) — Getting started with both models (~5-8 min)
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

## Performance

### Eigendecomposition Backend

The Spectral criterion's bottleneck is eigendecomposition (97-99% of runtime at d>=64).
The `eigendecompose()` function auto-selects the fastest LAPACK driver by dimension:

- **d < 256**: `numpy.linalg.eigh` (heevd, divide & conquer)
- **d >= 256**: `scipy.linalg.eigh` (heevr, MRRR algorithm)

On Apple M1/M2/M3 with Accelerate, heevr is **2x faster** than heevd at d>=256.
On Intel/AMD CPUs the optimal driver may differ — run `scripts/benchmark_platform.py`
to determine the best backend for your hardware.

### Criterion Performance at d=256 (QubitGrid, M1)

| Criterion | Time/trial | 5,000 trials |
|-----------|-----------|-------------|
| Moment    | ~1ms      | ~5 sec      |
| Krylov    | ~5s       | ~7 hours    |
| Spectral  | ~50s      | ~69 hours   |

### Optional JAX Backend

Install `jax` for Krylov speedup via JIT compilation:

```bash
pip install jax jaxlib
```

| Criterion | d <= 128 | d >= 256 |
|-----------|----------|---------|
| Krylov | JAX JIT: 12-97x faster | NumPy (autodiff unstable) |
| Spectral | NumPy (scipy eigh optimal) | NumPy |
| Moment | NumPy | NumPy |

JAX is auto-detected by `scripts/production/sweep_production.py`.

### Recommended Sweep Configurations (d=256)

| Purpose | n_hamiltonians x n_targets x K_values | Approx. time |
|---------|---------------------------------------|-------------|
| Quick test | 20 x 3 x 5K | ~5 hours |
| Publication | 50 x 5 x 10K | ~3 days |

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
