# Quantum Reachability Analysis

Analysis of reachability in quantum control systems using spectral, Krylov, and moment criteria.

Given a parameterized Hamiltonian family `H(lambda) = sum_k lambda_k H_k`, these criteria determine whether a target state `|phi>` is reachable from initial state `|psi>` without explicit time evolution.

## Installation

```bash
pip install numpy scipy pandas matplotlib
pip install qutip  # Only needed for legacy compatibility
```

## Quick Start

```python
from reach.models import CanonicalModel
from reach.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion
from reach.sampling import DensitySweep, SweepConfig

# Create Hamiltonian family (d=8, basis of d^2=64 operators)
model = CanonicalModel(dim=8, seed=42)

# Sample submodel with K operators
submodel = model.sample_submodel(K=10)

# Test single (psi, phi) pair
psi = submodel.init_state()
phi = submodel.random_state()

spectral = SpectralCriterion(submodel, psi, phi, tau=0.99)
result = spectral.is_reachable(maxiter=100, restarts=3)
print(f"Verdict: {result.verdict.value}, Score: {result.score:.4f}")

# Run Monte Carlo sweep
sweep = DensitySweep(model, SweepConfig.fast())
results = sweep.run(K_values=[5, 10, 15, 20])
sweep.save("results.csv")
```

## Project Structure

```
reach/
  models.py      QuantumModel, CanonicalModel, GeometricTwoLocalModel
  criteria.py    SpectralCriterion, KrylovCriterion, MomentCriterion
  sampling.py    DensitySweep, SweepConfig
  math_utils.py  Shared numerical utilities
  plotting.py    Publication color schemes and fit functions
  legacy.py      qutip-based backward compatibility functions
  mathematics.py Legacy: spectral_overlap, krylov_score (qutip interface)
  optimize.py    Legacy: maximize_spectral_overlap (qutip interface)

notebooks/
  quickstart.py  Interactive example demonstrating the full API

scripts/
  production/    Long-running experiments and publication plot generation
  benchmarks/    Benchmark scripts (confusion matrix, timing)
  audit/         Mathematical verification scripts
```

## Criteria

| Criterion | Formula | Optimization | Verdict |
|-----------|---------|--------------|---------|
| Spectral | S(lambda) = sum \|<u_n\|phi>* <u_n\|psi>\| | L-BFGS-B | REACHABLE / UNREACHABLE |
| Krylov | R(lambda) = \|\|P_K \|phi>\|\|^2 | L-BFGS-B | REACHABLE / UNREACHABLE |
| Moment | Q + xLL^T > 0 | Grid search | UNREACHABLE / INCONCLUSIVE |

## Hamiltonian Families

**CanonicalModel**: Generalized Pauli basis {X_jk, Y_jk, Z_j, I} for arbitrary dimension d. Submodels select K operators without replacement.

**GeometricTwoLocalModel**: Pauli chains on rectangular qubit lattice (nx x ny). Dimension d = 2^(nx*ny). Submodels are random weighted sums of all L basis operators, matching the GEO2 definition from [arXiv:2510.06321](https://arxiv.org/abs/2510.06321).

Known lattice configurations: d=8 (1x3), d=16 (2x2), d=32 (1x5), d=64 (2x3).

## Reproducing Publication Plots

```bash
# Generate publication figures from overnight CSV data
python scripts/production/plot_overnight_v7style.py

# Run confusion matrix benchmark
python scripts/benchmarks/confusion_matrix_benchmark.py
```

## Design Principles

- **K is primary**, not rho = K/d^2
- **sample_submodel(K)** returns a QuantumModel, not a list
- **numpy arrays** throughout (no qutip in new interfaces)
- **SeedSequence.spawn()** for parallel-safe RNG
- **Three-valued Verdict**: REACHABLE, UNREACHABLE, INCONCLUSIVE
