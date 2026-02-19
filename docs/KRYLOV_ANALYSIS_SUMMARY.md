# Krylov Analysis Summary: Why QubitGrid Krylov is Always P=0

## Key Insight

Generic Pauli Hamiltonians (as used in QubitGrid) are "highly mixing" -- the Krylov subspace K_m(H, phi) quickly spans the entire Hilbert space. When dim(K_m) = d, any target state is trivially reachable (score = 1.0), so P(unreachable) = 0 for all rho. This is correct physics, not a bug.

## Demonstration

At d=4 (2 qubits), combining just 2 Pauli operators (X1xY2 + Z1xX2) produces a Krylov basis of dimension 4/4 (full space). In contrast, a single canonical diagonal operator Z0 gives dimension 1/4.

At d=8, QubitGrid reaches full Krylov dimension by K=9 operators, while Canonical at K=19 only reaches dim/d ~ 0.99.

## Why It Matters

| Criterion | QubitGrid result | Canonical result |
|-----------|------------------|------------------|
| Krylov | P=0 (all reachable) | Phase transition |
| Spectral | Phase transition | Phase transition |
| Moment | Phase transition | Phase transition |

Spectral and Moment detect unreachability through different physics (eigenvalue constraints and moment conservation) and remain informative. For publication, we omit QubitGrid Krylov and focus on Spectral and Moment.

## Notebook

See `notebooks/krylov_qubitgrid_analysis.ipynb` for the full analysis (16 cells, ~10 sec runtime) with:
- Explicit 2-qubit demonstration
- Canonical vs QubitGrid contrast
- Scaling plot of Krylov dimension vs K
- Comprehensive summary with comparison tables
