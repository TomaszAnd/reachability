# Benchmark Plots

Generated: 2026-02-08
Script: scripts/benchmarks/confusion_matrix_benchmark.py

## Plot Inventory

| File | Description | Data Source |
|------|-------------|-------------|
| confusion_matrix_canonical.png | TP/FP/TN/FN for spectral/krylov on canonical | Computed on-the-fly |
| confusion_matrix_geo2.png | Same for GEO2 | Computed on-the-fly |
| accuracy_vs_time.png | Accuracy vs wall time, maxiter=50 | Computed on-the-fly |
| iterations_vs_accuracy.png | Score vs maxiter for d=8,16,32 | Computed on-the-fly |
| optimizer_heatmap.png | Spectral S* by optimizer method | criteria_time_accuracy.py |
| optimizer_heatmap_krylov.png | Krylov R* by optimizer method | criteria_time_accuracy.py |

## Ground Truth Methodology

### Reachable targets
Constructed via `|phi> = exp(-iH(lam_0)t)|psi>` with:
- Random weights `lam_0 ~ Uniform[-1, 1]^K`
- Random evolution time `t ~ Uniform[0.5, 5.0]`
- K = max(10, d) generators to ensure rich control algebra
- By construction, S* = 1 and R* = 1 at the true parameters

### Unreachable targets
Certified by second-order moment criterion (provable certificate):
1. Use K_small = 3 generators (minimal control)
2. Draw random target |phi> from Haar measure
3. Compute linear constraint L_n = <H_n>_phi - <H_n>_psi and its kernel N
4. Compute quadratic form Q_nm = <{H_n, H_m}>_phi - <{H_n, H_m}>_psi
5. Project: Q_proj = N^T Q N
6. If all eigenvalues of Q_proj have same sign (all > 0 or all < 0),
   the target is provably unreachable (no lambda can satisfy L=0 and Q=0)
7. Candidates failing certification are discarded (~30-60% pass rate)

## Threshold

All classifications use tau = 0.99: a target is predicted reachable if score >= tau.

## Configuration

- Dimensions: d in {8, 16, 32}
- Reachable targets per dim: 100 (K = max(10, d))
- Unreachable targets per dim: 100 (K = 3, moment-certified)
- Optimizer: L-BFGS-B, maxiter=50, restarts=3
- Ensembles: canonical, GEO2

## Data Generation

### Script
`scripts/benchmarks/confusion_matrix_benchmark.py`

### Process
1. For each ensemble (canonical, GEO2) and dimension d in {8, 16, 32}:
2. Generate 100 reachable targets (see Ground Truth Methodology)
3. Generate 100 unreachable targets (moment-certified)
4. Evaluate spectral and Krylov criteria on all 200 targets
5. Compute confusion matrix (TP/FP/TN/FN)
6. Plot confusion matrices per ensemble
7. Generate LaTeX table (`tex/confusion_matrix_table.tex`)
8. Plot accuracy_vs_time.png (scatter by ensemble/criterion/dimension)
9. Run iterations sweep (maxiter in {5, 10, 20, 50, 100, 200})
10. Plot iterations_vs_accuracy.png

### Runtime
Approximately 20-30 minutes for full benchmark (100 samples x 2 ensembles x 3 dimensions x 2 criteria).

### Reproducibility
Set SEED_BASE in the script for reproducible results. Default: 123456.

## Key Findings (100 samples per category)

1. **L-BFGS-B** is best optimizer for both Spectral and Krylov criteria
2. **Krylov achieves 100% accuracy** across all configurations (zero FP, zero FN)
3. **Spectral** achieves >= 97% accuracy; FN at d=32 are optimizer local minima
4. **Analytical gradients** provide 5-64x speedup over finite differences
5. One false positive (GEO2, d=8 spectral): borderline moment-certified target
6. FN rate increases with dimension for spectral (0% at d=8, 4-6% at d=32)
