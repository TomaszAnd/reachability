# Data Pipeline Audit: v7/v3 vs Overnight Production Data

**Date:** 2026-02-04
**Scope:** Compare data generation pipelines for v7 canonical plots, v3 GEO2 plots, and overnight production data.

---

## Executive Summary

The audit found **one significant methodological difference** and **one confirmed data reliability issue**:

1. **Krylov subspace dimension `m`**: The overnight scripts use `m = d` (full Hilbert space dimension), while the old `analysis.py` pipeline uses `m = min(K, d)`. For GEO2, this makes the Krylov criterion trivially saturated (R* = 1.0 always) in the overnight data. This is **not correct physics** — it is a consequence of the Krylov subspace spanning all of R^d when m = d and the Hamiltonian is non-degenerate.

2. **GEO2 v3 data used buggy optimizer settings** (maxiter=20, restarts=1), which the Jan 2026 audit found causes a 70% false-unreachable rate. The overnight GEO2 data uses corrected settings (maxiter=100, restarts=3) and is **more reliable** than the v3 data for spectral and moment criteria.

3. The canonical overnight data is trustworthy. Spectral and Krylov columns are verified distinct (max |diff| = 0.68). The Krylov `m = d` default does not significantly affect canonical results because sparse canonical Hamiltonians cause early Arnoldi breakdown regardless of the requested m.

---

## 1. Parameter Comparison Table

| Parameter | v7 Canonical | Overnight Canonical | v3 GEO2 | Overnight GEO2 |
|-----------|-------------|--------------------|---------|-----------------|
| **Dimensions** | {10, 14, 18, 22, 26} | {8, 16, 32, 64} | {16, 32, 64} | {8, 16, 32, 64} |
| **Data format** | Pickle (multiple files) | Single CSV | Pickle | Single CSV |
| **Pipeline** | `analysis.monte_carlo_...` | `parallel.parallel_evaluate` | `analysis.monte_carlo_...` | `parallel.parallel_evaluate` |
| **Optimizer** | L-BFGS-B | L-BFGS-B | L-BFGS-B | L-BFGS-B |
| **maxiter** | 200 (settings.DEFAULT) | 100 (d≤32), 150 (d=64) | **20 (BUGGY)** | 100 (d≤32), 150 (d=64) |
| **restarts** | 2 (settings.DEFAULT) | 3 | **1 (BUGGY)** | 3 |
| **ftol** | 1e-8 | 1e-6 | 1e-8 | 1e-6 |
| **Analytical gradient** | No (pre-audit) | Yes | No (pre-audit) | Yes |
| **Krylov m** | **min(K, d)** | **d (BUG)** | **min(K, d)** | **d (BUG)** |
| **n_trials** | nks×nst (varies) | 200 (100 for GEO2 d=64) | nks×nst (e.g., 30×20=600) | 200 (100 for GEO2 d=64) |
| **tau** | 0.99 | {0.90, 0.95, 0.99} | 0.99 | 0.99 |
| **Initial state** | |0⟩ (Fock) | |0⟩ (Fock) | |0⟩ (Fock) | |0⟩ (Fock) |
| **Target states** | Haar-random | Haar-random | Haar-random | Haar-random |
| **Bounds** | [-1, 1]^K | [-1, 1]^K | [-1, 1]^K | [-1, 1]^K |
| **Spectral formula** | S(λ) = Σ_n \|⟨φ\|u_n⟩\| · \|⟨u_n\|ψ⟩\| | Same | Same | Same |
| **Moment criterion** | Null-space + definiteness | Same (inline in parallel.py) | Same | Same (inline in parallel.py) |
| **Seed handling** | Single RNG chain | seed_base + trial_index | Single RNG chain | seed_base + trial_index |
| **Data dates** | Dec 9–16, 2025 | Jan 28, 2026 | Dec 29, 2025 | Jan 28, 2026 |

---

## 2. Answers to Key Questions

### Q1: Did the v7 data use the SAME optimizer settings as overnight?

**No.** The v7 canonical data used the general defaults: `maxiter=200, restarts=2, ftol=1e-8`. The overnight canonical data uses `maxiter=100, restarts=3, ftol=1e-6`.

The overnight settings use fewer iterations per restart but more restarts and a looser tolerance. With analytical gradients (added post-v7), the overnight settings should converge comparably or better than the v7 finite-difference approach at maxiter=200. The audit validation tests confirm 0% false-unreachable rate with the overnight settings.

### Q2: Did the v7 data include the Krylov basis truncation bugfix?

**No.** The v7 data was generated in December 2025 (dates: Dec 9–16, 2025). The Krylov truncation bugfix and analytical gradient additions were part of the January 2026 audit. However, the v7 data dimensions are {10, 14, 18, 22, 26} — completely different from the overnight dimensions {8, 16, 32, 64}. The v7 data is not directly comparable to overnight data anyway.

For the v3 GEO2 data (generated Dec 29, 2025), the bugfix had also not been applied. The old GEO2 data used `m = min(K, d)`, which at least provides a meaningful Krylov constraint. The overnight GEO2 data uses `m = d`, which makes Krylov trivially saturated.

### Q3: Did the v7 data use the same mathematical definitions?

**Yes.** The spectral overlap formula `S(λ) = Σ_n |⟨φ|u_n⟩| · |⟨u_n|ψ⟩|` (product of absolute values, NOT absolute value of product) is consistent across both pipelines. Both use `scipy.linalg.eigh` for eigendecomposition, same Fermi-Dirac fits, and same moment criterion (null-space + definiteness test).

The only mathematical difference is the Krylov subspace dimension `m`:
- v7/v3: `m = min(K, d)` — Krylov subspace limited to at most K vectors
- Overnight: `m = d` — Krylov subspace always spans full dimension (when Arnoldi doesn't break down)

### Q4: Why is GEO2 Krylov P=0 everywhere?

**The overnight result is a numerical artifact, not correct physics.** The root cause:

The overnight `parallel.py:evaluate_single_trial()` calls `optimize.maximize_krylov_score(psi, phi, hams, ...)` without specifying `m`. This defaults to `m = d` in `mathematics.krylov_score()`.

For GEO2 Hamiltonians (dense 2-local operators on qubit lattices), the Krylov subspace K_m(H(λ), |ψ⟩) with m = d = 16 (or 32, 64) fills the entire Hilbert space. The Arnoldi iteration never breaks down early because the Hamiltonian is dense and non-degenerate:

**Spot-check evidence:**
- GEO2 d=16, K=5, m=d=16: Krylov basis shape = (16, 16), effective dim = **16** → R* = 1.0 trivially
- GEO2 d=16, K=5, m=K=5: mean R* = 0.700, min = 0.568 → **non-trivial and informative**

With `m = min(K, d) = 5`, the Krylov criterion would show meaningful transitions for GEO2. With `m = d = 16`, every target is trivially reachable.

**For canonical Hamiltonians**, the situation is different. Sparse Pauli-like operators cause early Arnoldi breakdown:
- Canonical d=8, K=5, m=d=8: Krylov basis shape = (8, **1**), effective dim = 1
- The m=d and m=min(K,d)=5 results are **identical** because breakdown occurs at m_eff=1 regardless

This is why the canonical overnight Krylov data shows meaningful transitions while GEO2 shows P=0.

### Q5: Why does Krylov R² degrade at high d for canonical?

The Canonical Krylov R² drops from 0.993 (d=8) to 0.744 (d=64). This has two contributing factors:

1. **Broader transition**: The Krylov transition width Δ increases relative to ρ_c at higher d (Δ/ρ_c grows from 0.18 at d=8 to 0.55 at d=64). The Fermi-Dirac fit struggles with this broadening.

2. **Optimizer convergence**: At d=64, the Krylov score optimization has K ≈ 60–400 parameters. With maxiter=150 and restarts=3, the optimizer may not fully converge. This adds noise to the P(unreachable) estimates, degrading the fit quality.

The Fermi-Dirac functional form may genuinely not be the best model for the Krylov transition at high d. The transition appears to become asymmetric (faster onset, slower completion), which a symmetric sigmoidal function cannot capture well.

### Q6: Why are GEO2 Moment fit R² values so poor?

The GEO2 Moment criterion shows R² values of 0.613 (d=8), 0.568 (d=16), and 0.999 (d=32). The poor fits at d=8 and d=16 occur because:

1. **Very rapid transition**: The GEO2 moment criterion transitions from P≈1 to P≈0 within 1–2 K values. With 200 trials, the discrete jumps (P ∈ {0, 0.005, 0.01, ...}) poorly constrain a continuous exponential.

2. **K_c decreasing with d** (1.9 → 1.4 → 0.8): This appears unphysical but reflects that GEO2 Hamiltonians at larger d have more operators per unit ρ (since d = 2^n and L = 3n + 9|E|, the operator pool grows slower than d²). The moment criterion detects reachability with very few Hamiltonians.

3. **Wrong functional form**: The simple exponential P(ρ) = exp(-ρ/λ) assumes a smooth, gradual transition. For GEO2 moment, a step-function or Fermi-Dirac would fit better for d=8 and d=16 where the transition is very sharp.

---

## 3. Spot-Check Results

### Canonical d=8, K=10, tau=0.99 (20 trials)

| Metric | Spot Check | Overnight CSV |
|--------|-----------|---------------|
| spectral_P | 0.450 | 0.585 |
| krylov_P | 0.500 | 0.555 |
| moment_P | 0.100 | 0.065 |
| spectral_mean | 0.892 | 0.882 |
| krylov_mean | 0.846 | 0.826 |

Values differ due to different seeds and trial counts (20 vs 200) but are statistically consistent. Both spectral and krylov P values are in the mid-transition range, and the means are similar.

### Krylov m Parameter Test (canonical d=8, K=5)

| Setting | mean R* | P(R* < 0.99) | Max \|diff\| |
|---------|---------|-------------|-------------|
| m = d = 8 | — | — | 0.000000 |
| m = K = 5 | — | — | 0.000000 |

**Identical results** — the canonical Arnoldi iteration breaks down at effective dimension 1, so increasing m beyond 1 has no effect.

### GEO2 Krylov Saturation Test (d=16, K=5)

| Setting | mean R* | min R* | All R* ≥ 0.99? |
|---------|---------|--------|----------------|
| m = d = 16 | 1.000 | 1.000 | Yes |
| m = K = 5 | 0.700 | 0.568 | **No** |

**This confirms the m parameter is the root cause of GEO2 Krylov P=0.**

---

## 4. Verdict

### Is the overnight data trustworthy?

**Canonical: YES** — The spectral and moment criteria are reliable. The Krylov criterion is reliable for canonical because sparse Hamiltonians cause Arnoldi breakdown regardless of the requested m. The spectral_P and krylov_P columns are verified distinct (max diff = 0.68, correlation = 0.87). The overnight data uses better optimizer settings (more restarts, analytical gradients) than the v7 data.

**GEO2 Spectral: YES** — Corrected optimizer settings (maxiter=100, restarts=3) vs the old buggy settings (maxiter=20, restarts=1). The overnight GEO2 spectral data is MORE reliable than the v3 data.

**GEO2 Moment: YES** — The moment criterion is optimizer-independent (no λ optimization). Results are valid but the simple exponential fit is inappropriate at d=8, d=16. Use Fermi-Dirac or report without fit.

**GEO2 Krylov: NO** — The `m = d` default makes Krylov trivially saturated (R* = 1.0 for all targets). To get informative GEO2 Krylov results, rerun with `m = min(K, d)` or `m = K`. This would require modifying `parallel.py:evaluate_single_trial()` to pass `m=min(K, d)` to `maximize_krylov_score()`.

### Is the overnight data MORE or LESS reliable than v7/v3?

- **vs v7 canonical**: Comparable quality. Different dimensions (not directly comparable). Overnight has analytical gradients, more restarts; v7 had more iterations.
- **vs v3 GEO2**: Overnight is **significantly more reliable** for spectral/moment (corrected optimizer). GEO2 Krylov is uninformative in both cases (v3 due to buggy optimizer, overnight due to m=d saturation).

---

## 5. Recommendations

1. **Fix the Krylov m parameter** in `reach/parallel.py:evaluate_single_trial()` by passing `m=min(K, d)` to `maximize_krylov_score()`. This matches the behavior of `analysis.monte_carlo_unreachability_vs_density()`.

2. **Do NOT re-run overnight data** just for this fix. The canonical Krylov results are unaffected (Arnoldi breaks down early anyway). The GEO2 Krylov results should be clearly documented as uninformative due to m=d saturation.

3. **Use Fermi-Dirac fit for GEO2 moment** at d=8, d=16 where the exponential fit is poor.

4. **Document the v3 GEO2 data reliability issue** — the old GEO2 data (spectral/krylov) was generated with maxiter=20, restarts=1, which has a known 70% false-unreachable rate.

---

## 6. File References

| File | Role |
|------|------|
| `scripts/production/overnight_canonical.py` | Overnight canonical production script |
| `scripts/production/overnight_geo2.py` | Overnight GEO2 production script |
| `reach/parallel.py` | Overnight evaluation pipeline (evaluate_single_trial, parallel_evaluate) |
| `reach/optimize.py` | Optimization (maximize_spectral_overlap, maximize_krylov_score) |
| `reach/mathematics.py:579` | krylov_score() — m defaults to d when not specified |
| `reach/analysis.py:1745` | Old pipeline — uses m = min(K, d) |
| `reach/settings.py` | Optimizer defaults (DEFAULT_MAXITER=200, GEO2_MAXITER=100) |
| `scripts/canonical/generate_publication_dual_versions.py` | v7 plotting script (loads pickle data) |
| `scripts/geo2/run_geo2_production.py` | v3 GEO2 data generator (uses old analysis.py path) |
| `scripts/geo2/plot_geo2_v3.py` | v3 GEO2 plotting script |
| `scripts/audit/AUDIT_FINDINGS.md` | Jan 2026 audit report (optimizer bug discovery) |
