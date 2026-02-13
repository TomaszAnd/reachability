# Quantum Reachability Analysis

Test whether a target quantum state |phi> is reachable from initial state |psi>
under parameterized Hamiltonian H(lambda) = sum_k lambda_k H_k,
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
psi = submodel.init_state()
phi = submodel.random_state()
result = SpectralCriterion(submodel, psi, phi, tau=0.99).is_reachable()
print(f"{result.verdict.value}: score={result.score:.3f}")
```

See [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) for a complete tutorial.

## API Overview

### Models

| Class | Description |
|-------|-------------|
| `CanonicalQuditModel` | Generalized Pauli basis {X_jk, Y_jk, Z_j, I} for dimension d |
| `QubitGridModel` | Two-local Paulis on nx x ny qubit lattice (d = 2^(nx*ny)) |

### Criteria

| Criterion | Formula | Verdict |
|-----------|---------|---------|
| Spectral | S(lambda) = sum \|<u_n\|phi>\*<u_n\|psi>\| | REACHABLE / UNREACHABLE |
| Krylov | R(lambda) = \|\|P_Km \|phi>\|\|^2 | REACHABLE / UNREACHABLE |
| Moment | Q + gamma\*LL^T > 0 | UNREACHABLE / INCONCLUSIVE |

### Sweeps

```python
from src.sampling import DensitySweep, SweepConfig

sweep = DensitySweep(model, SweepConfig.fast())
results = sweep.run(K_values=[5, 10, 15, 20])
```

## Project Structure

```
src/                    Core library
├── models.py           QuantumModel, CanonicalQuditModel, QubitGridModel
├── criteria.py         SpectralCriterion, KrylovCriterion, MomentCriterion
├── sampling.py         DensitySweep, SweepConfig
├── math_utils.py       Eigendecomposition, Hamiltonian construction
└── plotting.py         Color schemes (DIM_COLORS, CRIT_COLORS)

notebooks/quickstart.ipynb   Interactive tutorial
scripts/production/          Overnight experiments
scripts/tests/               Test suite (34 tests)
```

## Running Tests

```bash
python scripts/tests/run_all_tests.py         # Full suite
python scripts/tests/run_all_tests.py --quick  # Quick mode (~1s)
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in academic work, please cite using [CITATION.cff](CITATION.cff).
