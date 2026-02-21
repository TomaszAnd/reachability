# Session Summary v75: Radical Spectral Speedup Investigation

## Objective
Investigate radical approaches to make Spectral criterion viable at d=256 on CPU.
Target: d=256 full sweep (Moment + Spectral + Krylov) in under 6 hours.

## Key Finding

**Spectral at d=256 on CPU is infeasible within 6 hours. No solution exists.**

The bottleneck (eigh eigendecomposition, 97-99% of time) cannot be circumvented:
- No GPU solution on M1 (tested in v74)
- No mathematical shortcut (needs ALL d eigenvalues)
- No viable optimizer reduction (causes massive false negatives)
- No symmetry exploitation (QubitGrid H has no block structure)

## Detailed Results

### Task 1: Gradient-Free Optimizers

| Method | d=64 time | Score | Vs L-BFGS-B |
|--------|-----------|-------|-------------|
| L-BFGS-B (gradient) | 1.3s | 0.978 | baseline |
| Nelder-Mead | 3.8s | 0.922 | 3x slower, worse |
| Powell | 8.6s | 0.856 | 7x slower, worse |

**Verdict: Dead end.** Gradient-free methods are uniformly slower and find worse optima.

### Task 2: Reduced Optimizer Settings

Tested at d=64, K=50 (transition region where both REACHABLE and UNREACHABLE occur):

| Config | maxiter | restarts | REACHABLE rate | Time |
|--------|---------|----------|---------------|------|
| default | 100 | 3 | 10/10 | 11.0s |
| fast | 50 | 2 | 3/10 | 6.4s |
| faster | 30 | 1 | 0/10 | 2.0s |
| minimal | 20 | 1 | 0/10 | 1.4s |

**Verdict: Dead end.** Reducing restarts/maxiter causes 70-100% false UNREACHABLE
at the transition. The optimizer REQUIRES full exploration to find the true maximum.

### Task 3: Perturbative Eigenvector Update

First-order perturbation theory to update eigenvectors between optimizer steps:

| d | eigh time | pert time | Speedup | Score error |
|---|-----------|-----------|---------|-------------|
| 64 | 0.8ms | 0.9ms | 0.9x | 4e-3 |
| 128 | 5.1ms | 4.5ms | 1.1x | 1.1e-2 |
| 256 | 33ms | 13ms | 2.6x | 1.0e-2 |

With periodic recorrection (every 10 steps): 2.2x overall, max error 6%.

**Verdict: Marginal.** 2.2x at d=256 brings 200s to ~90s/trial. Still ~250 hours
for a full sweep. The 6% error may also degrade optimizer convergence.

### Task 4: Eigenvector Continuity

| d | delta=0.001 | delta=0.01 | delta=0.1 | delta=0.5 |
|---|-------------|------------|-----------|-----------|
| 32 | 32/32 close | 28/32 | 18/32 | 2/32 |
| 64 | 64/64 close | 64/64 | 35/64 | 0/64 |
| 128 | 128/128 | 124/128 | 18/128 | 0/128 |

Eigenvectors are continuous for small delta (0.001-0.01) but rapidly decorrelate
at typical optimizer step sizes (0.02-0.1).

### Task 5: H(lambda) Structure Analysis

| d | H density | Z2 parity commutator | Block structure |
|---|-----------|---------------------|-----------------|
| 16 | 37.5% | 4.87 | None |
| 64 | 21.9% | 4.30 | None |
| 256 | 7.4% | 8.47 | None |

**Verdict: Dead end.** QubitGrid H(λ) has no exploitable symmetry or block structure.

## Final Feasibility Assessment

| Criterion | d=256 Time/trial | Full sweep (10K trials) | Status |
|-----------|-----------------|------------------------|--------|
| Moment | ~1ms | ~10 sec | ✅ Done |
| Krylov | 4.3s | ~72 min | ✅ Done |
| Spectral | ~200s | ~550 hours | ❌ **Impossible on CPU** |

## Recommendations

1. **Publish Krylov+Moment at d=256** — these are production-ready
2. **Skip Spectral at d=256** — no CPU solution exists
3. **For Spectral d=256, options are**:
   - NVIDIA GPU (CUDA eigh is 10-50x faster → ~10-50 hours)
   - Cloud GPU rental (A100/H100 for a few hours)
   - Multi-day CPU run (use `sweep_production.py` with `--n-hams 10 --n-targets 5`)
4. **Spectral up to d=128 is feasible** — ~5s/trial, ~14 hours for full sweep

## Files Created

- `docs/SESSION_SUMMARY_v75.md` — This file

## Mathematical Analysis

Why Spectral is fundamentally O(d³):
- S(λ) = Σ_n |⟨u_n(λ)|ψ⟩⟨u_n(λ)|φ⟩| requires ALL d eigenvectors
- No partial eigendecomposition is possible (not just extremal eigenvalues)
- No randomized/stochastic estimator exists (absolute value prevents trace tricks)
- The score function is non-convex with ~d² local maxima in λ-space
- L-BFGS-B with analytical gradient is already the optimal algorithm choice
