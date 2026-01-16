# GEO2 Ensemble Analysis Plan

**Date:** 2025-12-16
**Status:** Planning Phase
**Purpose:** Comprehensive analysis of GEO2 ensemble reachability and comparison with Canonical ensemble

---

## Executive Summary

The GEO2 (Geometric Two-Local) ensemble is fully implemented in `reach/models.py` and follows the arXiv definition from [2510.06321](https://arxiv.org/abs/2510.06321). This document outlines the current status, identifies gaps, and proposes experiments to answer key questions about locality and geometric structure effects on quantum reachability.

---

## 1. GEO2 Ensemble Definition

### Mathematical Formulation

**Basis:** P₂(G) contains all 1-local and 2-local Pauli terms on a rectangular lattice:
- 1-local: {X, Y, Z}_i for each site i
- 2-local: {X, Y, Z}_i ⊗ {X, Y, Z}_j for each edge (i,j)

**Hamiltonian:** H = Σ_a λ_a H_a where:
- H_a are the Pauli operators from P₂(G)
- λ_a ~ N(0, 1/L) are Gaussian weights
- L = |P₂(G)| = 3n + 9|E| (n = number of sites, |E| = number of edges)

**Dimension:** d = 2^n where n = nx × ny (qubit lattice)

### Key Features

1. **Sparse Structure**: Each Hamiltonian has L = 3n + 9|E| non-zero Pauli terms
   - Example: 2×2 lattice → n=4, |E|=4 (open BC) → L = 12 + 36 = 48
   - Compare with GUE/GOE: d² = 16² = 256 independent elements

2. **Fixed Weights**: Unlike Canonical ensemble where we optimize λ, GEO2 samples λ ~ N(0, 1/L) once per Hamiltonian

3. **Geometric Constraints**: Only nearest-neighbor interactions, respecting lattice topology

---

## 2. Current Implementation Status

### Code Location

**File:** `reach/models.py` (lines 206-376)
**Class:** `GeometricTwoLocal`

**Key Methods:**
```python
class GeometricTwoLocal:
    def __init__(self, nx: int, ny: int, periodic: bool = False, backend: str = "sparse")
        # Build Pauli basis P₂(G) with L = 3n + 9|E| operators

    def sample_lambda(self, rng: np.random.RandomState) -> np.ndarray:
        # Sample λ ~ N(0, 1/L)

    def sample_hamiltonian(self, rng: np.random.RandomState) -> qutip.Qobj:
        # Generate H = Σ_a λ_a H_a
```

### CLI Support

**Commands:**
```bash
# Density sweep
python -m reach.cli --nx 2 --ny 2 three-criteria-vs-density \
  --ensemble GEO2 -d 16 --rho-max 0.05 --rho-step 0.01 \
  --taus 0.95 --trials 100

# K-sweep
python -m reach.cli --nx 2 --ny 3 three-criteria-vs-K-multi-tau \
  --ensemble GEO2 -d 64 --k-max 10 --taus 0.95 --trials 100
```

**Required flags:** `--nx` and `--ny` (lattice dimensions)
**Auto-validation:** `d` must equal `2^(nx×ny)`

### Existing Infrastructure

**Scripts:**
- `scripts/generate_geo2_publication.py` - Publication-quality density plots
- `scripts/test_geo2_1x5_feasibility.py` - Feasibility testing

**Data:**
- `data/raw_logs/raw_data_GEO2_20251124_200032.pkl`
  - Lattice: 2×2 (d=16, n=4)
  - Density: ρ ∈ [0.01, 0.03], step=0.01 (K=3, 6, 9)
  - Trials: 30 (5×6 sampling)
  - Tau: 0.95
  - **Coverage:** Very limited - only 3 data points!

---

## 3. Critical Insight: Weight Optimization vs. Fixed Sampling

### The Key Difference

**Canonical Ensemble:**
- Generate K Hamiltonians with independent structure
- Optimize λ ∈ ℝ^K to maximize reachability criteria
- Question: "Can we find weights λ such that target is reachable?"

**GEO2 Ensemble (current implementation):**
- Generate K Hamiltonians with fixed random weights λ_k ~ N(0, 1/L_k)
- No optimization - use the sampled weights as given
- Question: "Is target reachable with these specific random weights?"

### Interpretation

This is **Approach 2a** from the user's plan:
- Generate random weights from GEO2 distribution
- Check reachability with GIVEN weights (no optimization)
- Reachability becomes probabilistic over:
  1. Random target states |φ⟩
  2. Random Hamiltonian instances (different Pauli terms)
  3. Random weight realizations λ

### Implications

- **Not directly comparable to Canonical** - different optimization assumptions
- **Still valuable** - tests intrinsic GEO2 ensemble properties
- **Need to decide**: Keep current interpretation or modify?

---

## 4. Existing Data Analysis

### Current Coverage

**Single Configuration:**
| Lattice | dim | K values | ρ range | Trials | Tau |
|---------|-----|----------|---------|--------|-----|
| 2×2 | 16 | 3, 6, 9 | 0.01-0.03 | 30 | 0.95 |

**Data Points:** Only 3 × 30 = 90 total trials

**Critical Limitations:**
1. **Single dimension** - cannot study scaling
2. **Very low ρ** - doesn't reach phase transition
3. **Few K values** - insufficient resolution
4. **Low trials** - high statistical uncertainty

### What's Missing

To match Canonical analysis, need:

**Multi-dimensional sweeps:**
| Lattice (nx×ny) | Hilbert dim d | Sites n | Operators L |
|----------------|---------------|---------|-------------|
| 2×2 | 16 | 4 | 48 |
| 1×5 | 32 | 5 | 75 |
| 2×3 | 64 | 6 | 90 |
| 1×7 | 128 | 7 | 105 |
| 2×4 | 256 | 8 | 120 |

**Dense ρ sampling:**
- ρ ∈ [0.002, 0.15] with step 0.002-0.01 (dimension-dependent)
- Must capture phase transition (expected at ρ ~ 0.01-0.05 for GEO2)

**High statistics:**
- 150-300 trials per (d, K, τ) point
- Match Canonical ensemble trial counts

---

## 5. Research Questions

### Primary Questions

1. **Does GEO2 have a phase transition like Canonical?**
   - Expect sharper transition due to sparse structure
   - Similar to Canonical basis (also sparse, also shows discontinuous behavior)

2. **How does K_c scale with dimension?**
   - Canonical: K_c ≈ 0.016d² (Moment), 0.066d² (Spectral), 0.037d² (Krylov)
   - GEO2: K_c ≈ ?d² or K_c ≈ ?n or K_c ≈ ?L?

3. **Is ρ_c = K_c/d² universal across ensembles?**
   - If yes: GEO2 should have similar ρ_c values
   - If no: geometric constraints shift the transition

4. **How does locality affect the transition?**
   - GEO2 (2-local) vs Canonical (sparse) vs GUE (dense)
   - Does limited connectivity make states harder to reach?

### Secondary Questions

5. **Effect of lattice geometry?**
   - 1D chains (1×n) vs 2D grids (2×3, 3×2)
   - Periodic vs open boundary conditions

6. **Criterion ordering?**
   - Canonical: Moment < Krylov < Spectral (in terms of K_c)
   - GEO2: Same ordering or different?

7. **Transition sharpness?**
   - GEO2 expected to have sharper Δ (width) due to sparsity
   - Quantify: measure Δ_GEO2 vs Δ_Canonical

---

## 6. Proposed Experiments

### Phase 1: Dimension Scaling (τ=0.95, ρ-sweeps)

**Goal:** Establish K_c(d) scaling for each criterion

| Lattice | dim d | ρ_max | ρ_step | K_max | Trials | Est. Time |
|---------|-------|-------|--------|-------|--------|-----------|
| 2×2 | 16 | 0.10 | 0.002 | 26 | 150 | ~1 hr |
| 1×5 | 32 | 0.08 | 0.003 | 82 | 150 | ~3 hr |
| 2×3 | 64 | 0.06 | 0.004 | 164 | 150 | ~8 hr |
| 1×7 | 128 | 0.05 | 0.005 | 410 | 100 | ~20 hr |

**Rationale:**
- Chose ρ_step to ensure ~20-30 data points per dimension
- Higher ρ_max for smaller d (transition occurs earlier)
- Reduced trials for d=128 (computational cost)

**Output:**
- 3-panel plots (Moment, Spectral, Krylov) for each dimension
- Fit ρ_c and Δ for each criterion
- K_c vs d scaling plot

### Phase 2: Direct Comparison with Canonical

**Goal:** Compare GEO2 vs Canonical for same dimension

**Fixed dimension:** d=16 (2×2 lattice for GEO2)

| Ensemble | Structure | Operators | ρ_max | Trials |
|----------|-----------|-----------|-------|--------|
| GEO2 | 2×2 lattice | L=48 sparse | 0.10 | 200 |
| Canonical | Generic sparse | 60 sparse | 0.10 | 200 |
| GUE | Dense | d²=256 dense | 0.10 | 200 |

**Output:**
- Overlay plot: all three ensembles, same axes
- ρ_c comparison table
- Δ (transition width) comparison

### Phase 3: Lattice Geometry Study (optional)

**Goal:** Test effect of lattice topology

**Fixed sites:** n=6 qubits (d=64)

| Lattice | Edges |E| | Operators L | Boundary |
|---------|-------|-----------|-------------|----------|
| 1×6 chain | 5 | 63 | open |
| 2×3 grid | 7 | 81 | open |
| 2×3 grid | 11 | 117 | periodic |

**Expected:** More edges → higher L → easier reachability → lower K_c

---

## 7. Implementation Plan

### Step 1: High-Resolution 2×2 Sweep (Proof of Concept)

**Purpose:** Verify setup and estimate compute time

```bash
python3 scripts/generate_geo2_publication.py --config 2x2 --trials 150
```

**Expected output:**
- `fig/comparison/geo2_2x2_d16_publication.png`
- Clean phase transition curves
- Runtime: ~1 hour

**Check:**
- Do all three criteria show transitions?
- Are transitions sharp (as expected for sparse structure)?
- Where is ρ_c? (expect ρ_c < 0.05 for all criteria)

### Step 2: Create Comprehensive GEO2 Sweep Script

**New script:** `scripts/run_geo2_comprehensive.py`

**Features:**
- Multi-dimensional sweep (d=16, 32, 64, 128)
- Adaptive ρ_step (finer for smaller d)
- Data saving to `.pkl` files
- Progress logging with time estimates
- Resume capability (load partial results)

**Structure:**
```python
def run_geo2_sweep(
    lattices: List[Tuple[int, int]],  # [(nx, ny), ...]
    rho_ranges: List[Tuple[float, float, float]],  # [(rho_max, rho_step), ...]
    tau: float = 0.95,
    trials: int = 150,
    output_dir: str = "data/raw_logs",
    resume_from: Optional[str] = None,
):
    # Loop over lattices
    for (nx, ny), (rho_max, rho_step) in zip(lattices, rho_ranges):
        # Run density sweep
        data = analysis.monte_carlo_unreachability_vs_density(...)

        # Save partial results
        save_checkpoint(data, output_dir, nx, ny)
```

### Step 3: Generate Comparison Plots

**New script:** `scripts/plot_geo2_vs_canonical.py`

**Features:**
- Load GEO2 data from Phase 1
- Load Canonical data from existing `.pkl` files
- Generate overlay plots for d=16
- Generate scaling plots (K_c vs d for both ensembles)

**Output plots:**
1. `geo2_vs_canonical_d16_tau095.png` - Direct comparison
2. `geo2_scaling_Kc_vs_d.png` - K_c(d) for GEO2
3. `ensemble_comparison_scaling.png` - GEO2 vs Canonical vs GUE

### Step 4: Documentation Update

**Files to update:**
1. `DATA_PROVENANCE.md` - Add GEO2 data sources
2. `README.md` - Add GEO2 ensemble description
3. `CLAUDE.md` - Add GEO2 usage examples
4. `GEO2_RESULTS.md` (new) - Summary of findings

---

## 8. Expected Results

### Hypothesis: GEO2 Shows Sharp Transitions at Lower K_c

**Reasoning:**
1. **Sparsity** - Only L = 3n + 9|E| ≈ 12n operators (vs d² for GUE)
2. **Locality** - Nearest-neighbor only (vs all-to-all for GUE)
3. **Similar to Canonical** - Both have sparse, structured operators

**Predictions:**

| Ensemble | Moment ρ_c | Spectral ρ_c | Krylov ρ_c | Transition Δ |
|----------|-----------|-------------|-----------|-------------|
| GUE | 0.016 | 0.066 | 0.037 | 0.010-0.011 |
| Canonical | ~0.03? | ~0.08? | ~0.05? | <0.005 (sharp!) |
| GEO2 | ~0.02? | ~0.07? | ~0.04? | <0.005 (sharp!) |

**Key test:** If GEO2 ρ_c values are similar to Canonical but lower than GUE, this supports the hypothesis that **sparse geometric structure** makes reachability harder (requires fewer operators).

---

## 9. Open Questions & Caveats

### Interpretation of GEO2 Results

**Question:** Should we optimize weights or use fixed random weights?

**Current implementation:** Fixed random weights (Approach 2a)
- Pro: Matches arXiv definition exactly
- Con: Not directly comparable to Canonical (which optimizes)

**Alternative:** Optimize weights (Approach 1)
- Pro: Direct comparison with Canonical
- Con: Deviates from GEO2 literature definition

**Recommendation:**
- Run Phase 1 with **fixed weights** (current implementation)
- If results are interesting, consider adding **optimized weights** variant for comparison

### Computational Cost

**Estimated total time for Phase 1:**
- 2×2 (d=16): ~1 hour
- 1×5 (d=32): ~3 hours
- 2×3 (d=64): ~8 hours
- 1×7 (d=128): ~20 hours
- **Total: ~32 hours** (~1.3 days)

**Recommendation:** Run in parallel if multiple cores available, or sequentially overnight.

### Lattice Size Limitations

**d=128 is already pushing limits:**
- Hilbert space dimension: 2^7 = 128
- GEO2 operators: L = 3(7) + 9(6) = 75 (manageable)
- Matrix operations: 128×128 dense (still feasible)

**Beyond d=128:**
- 2×5 lattice: d = 2^10 = 1024 (very large!)
- Need sparse matrix optimizations or approximations

---

## 10. Success Criteria

### Phase 1 Success

1. **Clean phase transitions** - All three criteria show sigmoid curves
2. **Fit quality** - R² > 0.90 for Fermi-Dirac fits
3. **Scaling law** - K_c vs d fits linear or power law
4. **Reproducibility** - Error bars < 0.05 for all points

### Phase 2 Success

1. **Clear ensemble ordering** - GEO2 vs Canonical vs GUE shows systematic differences
2. **Universal ρ_c?** - Determine if ρ_c = K_c/d² is ensemble-independent
3. **Sharpness quantified** - Measure Δ_GEO2 vs Δ_Canonical vs Δ_GUE

### Publication-Ready

1. **3-panel GEO2 plots** - Publication-quality figures for all dimensions
2. **Comparison plots** - GEO2 vs Canonical overlay
3. **Scaling analysis** - K_c(d), ρ_c(d), Δ(d) for all ensembles
4. **Comprehensive documentation** - Methods, data provenance, interpretation

---

## 11. Timeline & Next Steps

### Immediate (Dec 16-17, 2025)

- [x] Complete GEO2 codebase analysis
- [x] Document current status and gaps
- [ ] Run 2×2 high-resolution sweep (proof of concept)
- [ ] Verify plots and check transition locations

### Short-term (Dec 18-20, 2025)

- [ ] Implement `run_geo2_comprehensive.py`
- [ ] Run Phase 1 experiments (d=16, 32, 64, 128)
- [ ] Generate GEO2 scaling plots

### Medium-term (Dec 21-23, 2025)

- [ ] Implement `plot_geo2_vs_canonical.py`
- [ ] Generate comparison plots (Phase 2)
- [ ] Document findings in `GEO2_RESULTS.md`

### Long-term (Dec 24+, 2025)

- [ ] (Optional) Phase 3: Lattice geometry study
- [ ] (Optional) Implement optimized-weights variant
- [ ] Prepare publication figures

---

## 12. References

**arXiv papers:**
- [2510.06321](https://arxiv.org/abs/2510.06321) - GEO2 ensemble definition

**Code references:**
- `reach/models.py:206-376` - GeometricTwoLocal class
- `scripts/generate_geo2_publication.py` - Existing plotting script
- `data/raw_logs/raw_data_GEO2_20251124_200032.pkl` - Existing data (limited)

**Related analyses:**
- `DATA_PROVENANCE.md` - Canonical ensemble data documentation
- `TASKS_COMPLETE_20251216.md` - Recent Canonical analysis completion

---

**Last updated:** 2025-12-16
**Status:** Ready for proof-of-concept Phase 1 experiments

