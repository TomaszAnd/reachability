### Krylov Criterion (3-Curve Overlays)

Compare Krylov rank criterion alongside spectral overlap and moment criterion.

**Rank sweep (fixed d=20, K=4; varying m):**
```bash
# GUE ensemble
python -m reach.cli --seed 1 --summary three-criteria-vs-m --ensemble GUE -d 20 -K 4 \
  --m-values 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --tau 0.99 --trials 150

# GOE ensemble
python -m reach.cli --seed 2 --summary three-criteria-vs-m --ensemble GOE -d 20 -K 4 \
  --m-values 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --tau 0.99 --trials 150
```

**K sweep (fixed d=20; varying K; Krylov m = K):**
```bash
# GUE ensemble
python -m reach.cli --seed 1 --summary three-criteria-vs-K --ensemble GUE -d 20 \
  --k-values 2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --tau 0.99 --trials 150 --krylov-m K

# GOE ensemble
python -m reach.cli --seed 2 --summary three-criteria-vs-K --ensemble GOE -d 20 \
  --k-values 2,3,4,5,6,7,8,9,10,11,12,13,14 \
  --tau 0.99 --trials 150 --krylov-m K
```

**Outputs:**
- `fig_summary/unreachability_vs_rank_three_{ENS}_d{d}_K{K}_tau{tau}.png`
- `fig_summary/unreachability_vs_k_three_{ENS}_d{d}_tau{tau}.png`

**Notes:**
- **Krylov criterion**: Checks if target state φ lies in the Krylov subspace K_m(H, ψ) = span{ψ, Hψ, ..., H^(m-1)ψ} via projection-residual test (see below). Threshold-free (τ-independent).
- **Spectral overlap**: Threshold-based test. Maximizes spectral overlap S(λ) over parameter space and checks if max S(λ) < τ.
- **Moment criterion**: Moment-based definiteness check. Uses second-moment matrix analysis. Threshold-free (τ-independent).

---

### Krylov Membership Test (Projection Residual)

**Mathematical Criterion:**

The Krylov criterion tests whether the target state φ belongs to the m-dimensional Krylov subspace K_m(H, ψ). Let V be an orthonormal basis of K_m (computed via Arnoldi iteration). Then:

```
φ ∈ K_m(H, ψ)  ⟺  ‖φ - V(V†φ)‖₂ ≤ ε_proj
```

where:
- **V(V†φ)** is the orthogonal projection of φ onto K_m
- **‖·‖₂** is the Euclidean norm
- **ε_proj = 1e-10** (default projection tolerance)

**Why Projection Over Rank?**

Previous implementations used matrix rank comparison: `rank([V | φ]) > rank(V)`. However, this is **numerically brittle** because:
1. Matrix rank depends on a tolerance threshold that must be tuned
2. Rank is a global property, not specific to the vector φ
3. Near-membership cases (φ close to K_m) produce unstable results

The projection-residual test is **geometrically direct**: it measures the distance from φ to the subspace. A small residual (≪ 1) means φ ∈ K_m; a residual ≈ 1 means φ is far from K_m.

**Implementation:**
- Primary test: Projection residual (stable, direct)
- Fallback: Rank comparison with `rank_tol = 1e-8` (safety net only)

**Numerical Tolerance:**

For typical dimensions (d ≤ 30), `ε_proj = 1e-10` provides a large safety margin above machine precision (~1e-15) while accounting for accumulated QR errors in Arnoldi iteration. For larger dimensions (d > 50), adaptive scaling `ε_proj = 1e-10 * sqrt(d)` may be needed.

---

### Error Bars on Log-Scale Plots

**Wilson Score Interval (Asymmetric, Floor-Aware):**

When plotting probabilities on a log scale, symmetric binomial standard errors (SEM = √(p(1-p)/n)) produce nonsensical results:
- Bars extend below 0 when p is small
- Gigantic vertical spans near the display floor (1e-12)
- No clipping to valid [0,1] range

**Solution:** Use **Wilson score intervals** (≈1σ confidence) for binomial proportions:

```
Given: k successes in n trials (p̂ = k/n)
Center: p_c = (p̂ + z²/(2n)) / (1 + z²/n)
Margin: δ = (z / (1 + z²/n)) × √(p̂(1-p̂)/n + z²/(4n²))
Interval: [max(0, p_c - δ), min(1, p_c + δ)]
```

where z ≈ 1.0 for 68% confidence (≈1σ).

**Floor-Aware Clipping:**

On log plots with `DISPLAY_FLOOR = 1e-12`:
- Lower bound clipped: `lo = max(lo, DISPLAY_FLOOR)`
- Upper bound unchanged
- Result: **Asymmetric error bars** that respect the display floor

**Hidden Error Bars:**

Error bars are hidden when they are meaningless:
1. **p ≤ DISPLAY_FLOOR**: Point at floor, no bar
2. **k = 0 or k = n**: No variance (all success/failure), no bar
3. **n < 5**: Too few samples for Wilson approximation

---

**Implementation Note:**
The `krylov_basis()` function in `reach/mathematics.py` is copied from `jupyter_Project_Reachability/reach_bib.py::compute_Krylov_basis()` with minimal adaptation (parameter renaming, type handling, QR compression). The Arnoldi iteration algorithm remains unchanged.
