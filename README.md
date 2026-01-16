# reach: Time-Free Quantum Reachability Analysis


## Overview

**reach** is a Python package for analyzing quantum state reachability using time-free criteria. Instead of explicit time evolution, we determine whether a target state lies within the reachable subspace of a parameterized Hamiltonian family `H(λ) = Σᵢ λᵢ Hᵢ`.

### What It Does

- **Compares 3 reachability criteria**: Spectral overlap (τ-based), moment criterion (moment-based), and Krylov subspace
- **Runs density sweeps**: Analyze P(unreachability) vs control density ρ = K/d² across dimensions {20, 30, 40, 50}
- **Runs K-sweeps**: Analyze P(unreachability) vs number of Hamiltonians K at fixed dimension

### Key Features

- **Time-free analysis**: No explicit time evolution required
- **Random matrix ensembles**: GOE, GUE, and [GEO2](https://arxiv.org/html/2510.06321v1) (geometric two-local on qubit lattices, d=2^n) support
- **Monte Carlo estimation**: Robust probability estimates with error bars
- **Streaming mode**: CSV written incrementally, enables resumable runs and partial plotting
- **Floor-aware plotting**: No vertical "cliffs" at display floor (1e-12)
- **Dual y-axis**: Both P(unreachable) and P(reachable) versions
- **Full reproducibility**: Deterministic seeding and centralized configuration

---

## Quick Start (5 minutes)

### Installation

#### Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows (with WSL recommended)
- **Memory**: 4GB RAM minimum, 8GB+ recommended for d=50 calculations
- **Disk Space**: ~500MB for package + generated plots

#### Install from Source

```bash
# Clone repository
git clone https://github.com/TomaszAnd/reachability.git
cd reachability

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Verify installation
python -m reach.cli --help
```

#### Dependencies

Core dependencies (automatically installed):
- **NumPy** (≥1.21): Array operations, linear algebra
- **SciPy** (≥1.7): Optimization (L-BFGS-B), eigendecomposition
- **Matplotlib** (≥3.5): Visualization and plotting
- **Pandas** (≥1.3): CSV handling and data analysis
- **QuTiP** (≥4.7): Quantum operations (Arnoldi, Krylov subspaces)

Optional dependencies:
- **pytest** (≥7.0): Run test suite
- **tqdm**: Progress bars for long runs

#### Verification

Run quick smoke test to verify installation:

```bash
# Test CLI functionality
python -m reach.cli --help

# Run test suite (if pytest installed)
pytest tests/ -v

# Quick functional test (~30 seconds)
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE --dims 20 \
  --rho-max 0.02 --rho-step 0.01 \
  --taus 0.95 --trials 10 \
  --y unreachable
```

#### Troubleshooting

**ImportError: No module named 'reach'**
```bash
# Ensure you're in the reachability directory
cd /path/to/reachability
pip install -e .
```

**QuTiP installation issues**
```bash
# QuTiP requires Cython - install separately first
pip install cython
pip install qutip
```

**Memory errors for large dimensions**
```bash
# Reduce dimension or trials for testing
python -m reach.cli three-criteria-vs-density \
  --dims 20,30 \      # Instead of 20,30,40,50
  --trials 50 \       # Instead of 150
  --y unreachable
```

**Slow eigendecomposition**
```bash
# Check NumPy is using optimized BLAS/LAPACK
python -c "import numpy; numpy.show_config()"

# Consider installing OpenBLAS or MKL
pip install numpy[mkl]  # Intel MKL (fastest)
```

### Generate a Single Figure (Fast)

```bash
# Quick density plot (takes ~2 minutes)
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.05 --rho-step 0.01 \
  --taus 0.95 --trials 25 \
  --y unreachable

# Output: fig/comparison/three_criteria_vs_density_GUE_tau0.95_unreachable.png
```

#### GEO2 Example (Geometric Lattice Ensemble)

```bash
# GEO2 on 2×2 lattice (4 qubits, d=16)
python -m reach.cli --nx 2 --ny 2 three-criteria-vs-K-multi-tau \
  --ensemble GEO2 -d 16 --k-max 10 \
  --taus 0.99 --trials 100 --y unreachable

# GEO2 requires power-of-2 dimensions: d = 2^(nx×ny)
# Use --nx, --ny to specify lattice shape
# Optional: --periodic for periodic boundary conditions
# GEO2 filenames reflect true dimensions (e.g., d512 for 3×3 lattice)
# For denser K/d² sampling, use 3×3 or larger lattices
```

### Plot from Existing CSV (No Recomputation)

If you have CSV data from a previous run:

```bash
# Generate plots from CSV (instant)
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable
```

---

## Reproducing Publication Figures

Generate all v7 dual-version publication figures from existing data:

```bash
python scripts/generate_publication_dual_versions.py 2>&1 | tee logs/dual_version_generation.log
```

**Output**: 8 publication-quality figures in `fig/publication/`:
- `final_summary_3panel_v7_{exp,pow2}.png` - 3-panel decay curves (Moment, Spectral, Krylov)
- `combined_criteria_d26_v7_{exp,pow2}.png` - All three criteria at d=26
- `Kc_vs_d_v7_{exp,pow2}.png` - Critical K scaling with dimension
- `linearized_fits_v7_{exp,pow2}.png` - Linearized fits (ln vs log₂)

**Dual-version support**:
- **EXP version**: P(ρ) = exp(-α d² (ρ - ρ_c)), linearized as ln(P)
- **POW2 version**: P(ρ) = 2^(-ρ/ρ_c), linearized as log₂(P)

**Data sources**: See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for complete data lineage, merging strategies, and access patterns.

**Reproduction details**: See [CLAUDE.md](CLAUDE.md#reproducing-publication-figures) for step-by-step workflow.

---

## Production Runs

### Fast Demo (~5-10 minutes)

```bash
./run_production_sweeps_FAST.sh
```

Generates 8 plots + 2 CSVs with reduced parameters.

### Full Production (~2-3 hours)

```bash
# Background run with logging
nohup ./run_production_sweeps.sh > production.log 2>&1 &
echo $! > production.pid

# Monitor progress (in another terminal)
tail -f production.log

# Refresh plots from growing CSV
./scripts/plot_refresh.sh
```

Generates publication-ready plots with `trials=150` (density) and `trials=300` (K-sweep).

---

## Plotting Pipeline (TL;DR)

### Streaming CSV Mode

Use `--flush-every N` to write CSV incrementally:

```bash
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.15 --rho-step 0.01 \
  --taus 0.90,0.95,0.99 --trials 150 \
  --csv fig/comparison/density_gue.csv \
  --flush-every 10  \  # <-- Flush every 10 data points
  --y unreachable
```

**Benefits**:
- CSV grows during computation (survives interrupts)
- Resume from partial data
- Plot mid-run progress with `plot-from-csv`

### Plot from CSV

Read existing CSV and generate plots (useful for partial results):

```bash
# Density plots
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable

# K-sweep plots
python -m reach.cli plot-from-csv \
  --csv fig/comparison/k30_gue.csv \
  --type k-multi-tau \
  --ensemble GUE \
  --y unreachable
```

**Handles gracefully**:
- Partial CSVs (missing data points)
- Mixed dimensions (filters/warns)
- Empty files (clear errors)

### CSV Schema (15 fields)

```
run_id, timestamp, ensemble, criterion, tau, d, K, m,
rho_K_over_d2, trials, successes_unreach, p_unreach,
log10_p_unreach, mean_best_overlap, sem_best_overlap
```

### Display Floor & Styling

- **Floor**: `1e-12` for log₁₀ plots (prevents -∞)
- **Floor-aware rendering**: Broken lines at floor, faded markers
- **Wilson intervals**: Asymmetric error bars for binomial data
- **Styling**: 14×10 inches, 200 DPI, bold 16pt axis labels, 18pt title

---

## Mathematical Foundation

### Problem Statement

**Given**:
- Initial quantum state |ψ⟩ ∈ ℋ (Hilbert space dimension d)
- Target quantum state |φ⟩ ∈ ℋ
- Parameterized Hamiltonian family: H(λ) = Σᵢ₌₁ᴷ λᵢ Hᵢ where λ ∈ [-1,1]ᴷ

**Question**: Can we reach |φ⟩ from |ψ⟩ via time evolution under some choice of control parameters λ?

**Time-free approach**: Rather than explicitly computing |ψ(t)⟩ = exp(-iH(λ)t)|ψ⟩, we analyze spectral properties of H(λ) to determine reachability without time evolution.

---

### Spectral Overlap Criterion (τ-based)

**Definition**: Measures alignment of initial/target states with the eigenbasis of H(λ):
```
S(λ) = Σₙ |⟨n(λ)|φ⟩|·|⟨n(λ)|ψ⟩|
```

where {|n(λ)⟩} are eigenstates of H(λ).

**Reachability Test**:
```
max_{λ∈[-1,1]ᴷ} S(λ) ≥ τ  →  reachable (high confidence)
max_{λ∈[-1,1]ᴷ} S(λ) < τ  →  unreachable (high confidence)
```

**Properties**:
- Range: S(λ) ∈ [0, 1] (bounded by Cauchy-Schwarz inequality)
- Perfect alignment: S(λ) = 1 ⟺ guaranteed reachability
- Threshold: τ ∈ [0.90, 0.99] controls sensitivity (higher τ → stricter test)
- **Requires optimization**: Must maximize S(λ) over K-dimensional parameter space

**Optimization Strategy**:
- **Method**: L-BFGS-B (limited-memory BFGS with box constraints)
- **Bounds**: λᵢ ∈ [-1, 1] for all i = 1, ..., K
- **Convergence**: Function tolerance ftol = 1e-8
- **Multiple restarts**: 2-5 random initializations to avoid local maxima
- **Typical iterations**: 50-200 per optimization depending on (K, d)
- **Cost per evaluation**: O(d³) for eigendecomposition

---

### Moment Criterion (τ-free)

**Definition**: Analyzes the geometry of the reachable set via second-moment analysis. No parameter optimization required.

**Mathematical Formulation**:

Given Hamiltonian generators {H₁, H₂, ..., Hₖ}, construct:

**Moment matrix** (K × K):
```
M_{ij} = ⟨ψ|Hᵢ Hⱼ|ψ⟩ - ⟨ψ|Hᵢ|ψ⟩⟨ψ|Hⱼ|ψ⟩
```

**Moment vector** (K × 1):
```
v_i = ⟨φ|Hᵢ|ψ⟩ - ⟨φ|ψ⟩⟨ψ|Hᵢ|ψ⟩
```

**Reachability Test**:
```
If M is positive definite AND v^T M^{-1} v < threshold
  →  unreachable
Otherwise
  →  possibly reachable
```

**Properties**:
- **No threshold parameter** (τ-independent)
- **No optimization** (analytical, deterministic)
- **Fast**: O(K³) for matrix inversion + O(d³) for Hamiltonian expectations
- Based on convex geometry: tests if target lies outside moment cone
- Conservative estimate: may classify some unreachable states as "possibly reachable"

**Advantages**:
- Extremely fast compared to spectral criterion (~100× faster for K=5, d=20)
- Deterministic results (no optimizer variance)
- Good baseline for quick analysis

**Limitations**:
- Currently only supports GOE/GUE ensembles (GEO2 support in development)
- More conservative than spectral criterion

---

### Krylov Criterion (τ-free, m-dependent)

**Definition**: Tests if target state lies in the Krylov subspace spanned by repeated Hamiltonian applications.

**Krylov Subspace**:
```
K_m(H, ψ) = span{|ψ⟩, H|ψ⟩, H²|ψ⟩, ..., H^(m-1)|ψ⟩}
```

**Reachability Test** (projection-residual method):

1. **Build orthonormal basis** V of K_m(H, ψ) via Arnoldi iteration
2. **Project target**: |φ_proj⟩ = V(V^† |φ⟩)
3. **Compute residual**: r = ||φ⟩ - |φ_proj⟩||₂

**Criterion**:
```
r < ε_proj (typically 1e-10)  →  φ ∈ K_m(H, ψ)  →  reachable
r ≥ ε_proj                     →  φ ∉ K_m(H, ψ)  →  unreachable
```

**Properties**:
- **No threshold parameter** (ε_proj is numerical tolerance, not tunable)
- **Rank-dependent**: Larger m → larger subspace → more states reachable
- Standard choice: **m = min(K, d)** for maximum coverage
- **Numerically stable**: Uses QR-based Arnoldi iteration
- **Computational cost**: O(m·d²) for Arnoldi + O(d) for projection test

**Advantages**:
- Works for all ensembles (GOE, GUE, GEO2)
- Clear geometric interpretation (subspace membership)
- Moderate computational cost (between moment and spectral)

---

### Comparison of Three Criteria

| Property | Spectral | Moment | Krylov |
|----------|----------|--------|--------|
| **Threshold** | Yes (τ ∈ [0.90, 0.99]) | No | No (ε_proj fixed) |
| **Optimization** | Yes (expensive) | No (analytical) | No (iterative basis) |
| **Complexity** | O(restarts × iters × d³) | O(K³ + d³) | O(m·d²) |
| **Ensembles** | All (GOE/GUE/GEO2) | GOE/GUE only | All (GOE/GUE/GEO2) |
| **Deterministic** | No (optimizer-dependent) | Yes | Yes |
| **Sensitivity** | High (tunable via τ) | Conservative | Rank-dependent (via m) |
| **Typical Runtime** | Seconds to minutes | Milliseconds | Milliseconds |

**Recommendation**:
- **Quick analysis**: Start with moment (GOE/GUE) or Krylov (all ensembles)
- **Publication-quality**: Use spectral with multiple τ values
- **Comprehensive study**: Compare all three to understand agreement/disagreement

**Typical Agreement**:
- Spectral vs Krylov: ~80-90% agreement (τ=0.95, m=min(K,d))
- Moment vs Spectral: Moment more conservative by ~10-20%
- Disagreement concentrated near critical density ρ_c

---

### Monte Carlo Estimation

We estimate probabilities via Monte Carlo sampling over random ensembles:

**Procedure**:
1. Sample Nₕ random Hamiltonians from chosen ensemble (GOE/GUE/GEO2)
2. Sample Nₛ random state pairs {(|ψᵢ⟩, |φᵢ⟩)} uniformly from unit sphere
3. Test reachability for each combination
4. Estimate: P(unreachable) = (# unreachable) / (Nₕ × Nₛ)

**Error Estimation**:
- **Method**: Wilson score intervals (asymmetric binomial confidence intervals)
- **Floor-aware**: Probabilities < 1e-12 handled specially in log plots
- **Typical sampling**: Nₕ ∈ [20, 60], Nₛ ∈ [3, 15]

---

### Random Hamiltonian Ensembles

**GOE (Gaussian Orthogonal Ensemble)**:
- Real symmetric matrices (H = H^T)
- Time-reversal symmetric systems
- Distribution: Hᵢⱼ ~ N(0, 1/√d)

**GUE (Gaussian Unitary Ensemble)**:
- Complex Hermitian matrices (H = H^†)
- No time-reversal symmetry (magnetic field, spin-orbit)
- Distribution: Re(Hᵢⱼ), Im(Hᵢⱼ) ~ N(0, 1/(2√d))

**GEO2 (Geometric Two-Local)** [[arXiv:2510.06321](https://arxiv.org/abs/2510.06321)]:
- Physically motivated: nearest-neighbor interactions on qubit lattice
- Structure: H = Σₐ λₐ Hₐ where {Hₐ} are L = 3n + 9|E| geometric operators
- For nx × ny lattice: n = nx·ny sites, d = 2^n dimension
- Distribution: λₐ ~ N(0, 1) independently
- **Relevance**: Near-term quantum devices, coupled qubit architectures

---

### Control Density ρ = K/d²

**Definition**: Normalized control parameter density
```
ρ = K/d²
```

**Physical Interpretation**:
- **Low density** (ρ < 0.05): Sparse control, most states unreachable
- **Transition regime** (ρ ≈ 0.05-0.10): Sharp phase transition
- **High density** (ρ > 0.10): Dense control, most states reachable

**Why K/d² scaling?**
- Hilbert space size ~d, full controllability requires ~d² independent operators
- ρ provides dimension-independent measure for fair comparison

**Typical Behavior**:
- **Critical density**: ρ_c ≈ 0.07-0.10 (ensemble-dependent)
- **Sharp transition**: P(unreachable) drops from ~0.9 to ~0.1 over Δρ ≈ 0.03

---

## Data Quality and Methodology

### Physical Data Treatment (No Artificial Clipping)

This project uses statistically rigorous treatment of probability data:

1. **Wilson Score Intervals**: Boundary values (P=0, P=1) are treated as genuine
   physical outcomes with proper binomial uncertainty, not clipped to arbitrary ε.

2. **Transition-Region Fitting**: Linearized fits (exponential, Fermi-Dirac) use only
   informative transition points (0 < k < N), excluding boundary points that provide
   no slope information.

3. **Quality Metrics**: Every fit reports:
   - N_transition: Number of informative points used
   - Frac_transition: Fraction of data in transition region
   - Quality flag: "good" (≥10 pts, ≥20%), "marginal" (≥5 pts, ≥10%), or "insufficient"

### Linearized Fits for Functional Form Validation

The `scripts/plot_linearized_physical.py` script validates theoretical predictions:

**Panel (a): Moment Criterion - Exponential Decay**
```
P(K) = exp(-α(K - K_c))  →  log(P) = -α·K + α·K_c
```
Linear fit of log(P) vs K extracts critical control density K_c and decay rate α.

**Panel (b): Spectral Criterion - Fermi-Dirac**
```
P(K) = 1/(1 + exp((K - K_c)/Δ))  →  logit(P) = -K/Δ + K_c/Δ
```
Linear fit of logit(P) vs K extracts transition midpoint K_c and width Δ.

**Panel (c): Krylov Criterion - Step-like Transition**
```
P(K) = 1/(1 + exp((K - K_c)/Δ))  with Δ → 0
```
Near-discontinuous transition requires dense sampling (ΔK ~ Δ/3 ≈ 0.2-0.3).

**Typical Results (τ=0.99)**:
- Moment: 90-95% transition points, R² > 0.95 (excellent quality)
- Spectral: 50-60% transition points, R² > 0.85 (good quality)
- Krylov: 7-13% transition points, R² variable (insufficient data, needs denser K sampling)

### Quality-Aware Experimental Design

**Problem**: Krylov has sharp transitions (Δ ≈ 0.5-1.5), requiring dense sampling near K_c
to capture the transition region.

**Solution**: Use adaptive K grids with spacing ~ Δ/3:
```python
# Coarse grid for initial K_c estimate
K_coarse = [2, 4, 6, 8, 10, 12, 14]

# Dense grid near estimated K_c (e.g., K_c ≈ 10 for d=10)
K_dense = np.arange(8, 12, 0.3)  # Δ/3 ≈ 0.5/3 ≈ 0.17

# Combine for comprehensive coverage
K_adaptive = sorted(set(K_coarse) | set(K_dense))
```

See `docs/clipping_methodology.md` for full mathematical derivations and alternative approaches.

---

## Practical Usage Examples

### Example 1: Quick Single-Criterion Test

Test reachability with spectral criterion only, minimal parameters:

```bash
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE \
  --dims 20 \
  --rho-max 0.05 \
  --rho-step 0.01 \
  --taus 0.95 \
  --trials 25 \
  --y unreachable

# Output: fig/comparison/three_criteria_vs_density_GUE_tau0.95_unreachable.png
# Runtime: ~2 minutes
```

**Use case**: Quick exploration to understand basic behavior before committing to full parameter sweeps.

### Example 2: Criterion Comparison Study

Compare all three criteria (spectral, moment, Krylov) across multiple dimensions:

```bash
python -m reach.cli three-criteria-vs-density \
  --ensemble GOE \
  --dims 20,30,40,50 \
  --rho-max 0.15 \
  --rho-step 0.01 \
  --taus 0.95 \
  --trials 150 \
  --csv fig/comparison/density_goe.csv \
  --flush-every 10 \
  --y unreachable

# Output:
#   - three_criteria_vs_density_GOE_tau0.95_unreachable.png
#   - density_goe.csv (incremental, resumable)
# Runtime: ~45 minutes
```

**Use case**: Publication-quality comparison showing agreement/disagreement between criteria, dimension scaling, and critical density ρ_c.

### Example 3: Threshold Sensitivity Analysis

Study how spectral threshold τ affects classification:

```bash
python -m reach.cli three-criteria-vs-K-multi-tau \
  --ensemble GUE \
  -d 30 \
  --k-max 14 \
  --taus 0.90,0.95,0.99 \
  --trials 300 \
  --csv fig/comparison/k30_gue.csv \
  --flush-every 10 \
  --y unreachable

# Output:
#   - K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_unreachable.png
#   - k30_gue.csv
# Runtime: ~30 minutes
```

**Use case**: Understand how threshold choice affects sensitivity; identifies robust parameter regimes.

### Example 4: Streaming Mode with Mid-Run Plotting

For long runs, monitor progress by plotting partial results:

**Terminal 1** (run sweep):
```bash
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE \
  --dims 20,30,40,50 \
  --rho-max 0.15 \
  --rho-step 0.01 \
  --taus 0.90,0.95,0.99 \
  --trials 150 \
  --csv fig/comparison/density_gue.csv \
  --flush-every 10 \
  --y unreachable
```

**Terminal 2** (monitor and plot):
```bash
# Watch CSV grow
watch -n 30 'wc -l fig/comparison/density_gue.csv'

# Refresh plots every 5 minutes
watch -n 300 './scripts/plot_refresh.sh density'

# Or manually:
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable
```

**Use case**: Long-running parameter sweeps (1-3 hours); enables early detection of issues and progress monitoring.

### Example 5: GEO2 Geometric Ensemble (Qubit Lattices)

Analyze reachability on physically motivated qubit lattice Hamiltonians:

```bash
# 2×2 lattice (4 qubits, d=16)
python -m reach.cli --nx 2 --ny 2 three-criteria-vs-K-multi-tau \
  --ensemble GEO2 \
  -d 16 \
  --k-max 10 \
  --taus 0.99 \
  --trials 100 \
  --y unreachable \
  --csv fig/comparison/k16_geo2.csv

# Optional: Add periodic boundary conditions
python -m reach.cli --nx 2 --ny 2 --periodic three-criteria-vs-K-multi-tau \
  --ensemble GEO2 \
  -d 16 \
  --k-max 10 \
  --taus 0.99 \
  --trials 100 \
  --y unreachable

# For density sweeps, use larger lattices (3×3 → d=512):
python -m reach.cli --nx 3 --ny 3 three-criteria-vs-density \
  --ensemble GEO2 \
  -d 512 \
  --rho-max 0.05 \
  --rho-step 0.005 \
  --taus 0.99 \
  --trials 50 \
  --y unreachable
```

**Use case**: Near-term quantum devices with geometric connectivity constraints; validates reachability theory on physically realizable Hamiltonian families.

**Note**: GEO2 requires power-of-2 dimensions (d = 2^n where n = nx×ny). See [arXiv:2510.06321](https://arxiv.org/abs/2510.06321) for ensemble details.

---

## File Map

### Core Package (`reach/`)

| File | Purpose |
|------|---------|
| `cli.py` | Command-line interface, subcommands, argument parsing |
| `analysis.py` | Monte Carlo loops, density/K sweeps (pure computation) |
| `viz.py` | Plot functions for computed data (pure rendering) |
| `viz_csv.py` | Plot functions for CSV data (streaming/partial results) |
| `logging_utils.py` | CSV logging, `StreamingCSVWriter` class |
| `models.py` | GOE/GUE ensemble generation, random states |
| `mathematics.py` | Eigendecomposition, spectral overlap, Krylov tests |
| `optimize.py` | Maximize S(λ) using scipy.optimize |
| `settings.py` | All config constants, defaults, hyperparameters |

### Scripts & Helpers

| File | Purpose |
|------|---------|
| `run_production_sweeps.sh` | Full production run (2-3 hrs, trials=150/300) |
| `run_production_sweeps_FAST.sh` | Fast demo (5-10 min, reduced params) |
| `scripts/plot_refresh.sh` | Re-render plots from existing CSVs |
| `scripts/generate_summary_figs.py` | Generate all publication-quality plots |

### Output (`fig/comparison/`)

**Density plots** (6 files):
```
three_criteria_vs_density_GUE_tau{0.90,0.95,0.99}_{unreachable,reachable}.png
```

**K-sweep plots** (2 files):
```
K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_{unreachable,reachable}.png
```

**CSV data** (2 files):
```
density_gue.csv      # Density sweep data (all dims, all τ)
k30_gue.csv          # K-sweep data (d=30, multiple τ)
```

---

## Module Interaction Flow

```
User → cli.py (parse args)
       ↓
models.py (generate Hamiltonians & states)
       ↓
analysis.py (Monte Carlo loops)
       ↓
optimize.py (maximize S(λ))
       ↓
mathematics.py (compute S(λ))
       ↓
analysis.py (collect results)
       ↓
viz.py / viz_csv.py (render figures)
       ↓
fig/comparison/*.png + *.csv
```

---

## CLI Examples

### Density Sweep

```bash
# Full parameter space
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE \
  --dims 20,30,40,50 \
  --rho-max 0.15 \
  --rho-step 0.01 \
  --taus 0.90,0.95,0.99 \
  --trials 150 \
  --csv fig/comparison/density_gue.csv \
  --flush-every 10 \
  --y unreachable
```

**Output**: 3 PNG files (one per τ) + appends to CSV

### K-Sweep (Multi-Tau)

```bash
# Fixed dimension, sweep K
python -m reach.cli three-criteria-vs-K-multi-tau \
  --ensemble GUE \
  -d 30 \
  --k-max 14 \
  --taus 0.90,0.95,0.99 \
  --trials 300 \
  --csv fig/comparison/k30_gue.csv \
  --flush-every 10 \
  --y unreachable
```

**Output**: 1 PNG file (all τ overlaid) + appends to CSV

### Plot from CSV

```bash
# Density plots from CSV
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable

# K-sweep plots from CSV
python -m reach.cli plot-from-csv \
  --csv fig/comparison/k30_gue.csv \
  --type k-multi-tau \
  --ensemble GUE \
  --y reachable

# Filter to specific tau values
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable \
  --taus 0.95,0.99
```

**Output**: PNG files (no computation, reads from CSV)

---

## Reproducibility

All runs are deterministic when using the same seed:

```python
# In settings.py
SEED = 42  # Default random seed
```

Key config in `settings.py`:
- `FULL_SAMPLING = (30, 15)` → nks=30, nst=15 (450 trials)
- `FAST_SAMPLING = (5, 3)` → nks=5, nst=3 (15 trials)
- `DISPLAY_FLOOR = 1e-12` → Floor for log plots
- `DEFAULT_TAU = 0.95` → Default spectral threshold

---

## Testing

```bash
# Smoke tests (fast)
pytest tests/ -v

# Implementation validation
python validate_implementation.py

# Streaming mode validation
python validate_streaming_mode.py
```

---


## Quick Reference Card

```bash
# Fast demo (5-10 min)
./run_production_sweeps_FAST.sh

# Full production (2-3 hrs, background)
nohup ./run_production_sweeps.sh > production.log 2>&1 &

# Monitor progress
tail -f production.log

# Refresh plots from CSV (while running)
./scripts/plot_refresh.sh

# Plot from CSV (manual)
python -m reach.cli plot-from-csv --csv fig/comparison/density_gue.csv --type density --ensemble GUE --y unreachable

# Help
python -m reach.cli --help
python -m reach.cli three-criteria-vs-density --help
```

---

---

## Three-Criteria Comparison with Continuous Krylov

### Overview

The continuous Krylov implementation enables fair comparison between Spectral and Krylov criteria by optimizing both over the parameter space λ ∈ [-1,1]ᴷ.

### Mathematical Details

**Krylov score:** R(λ) = ‖P_Kₘ(H(λ))|φ⟩‖²

where P_Kₘ is the orthogonal projection onto the Krylov subspace K_m(H(λ), |ψ⟩).

**Key identity:** R(λ) = 1 - ε²_res where ε_res is the residual norm (verified to 10 decimal places).

**Optimization:** R* = max_{λ} R(λ) via L-BFGS-B with multi-restart.

### Usage Example

```python
from reach import analysis, viz

# Run three-criteria comparison
data = analysis.monte_carlo_unreachability_vs_density(
    dims=[14, 16, 18],
    rho_max=0.12,
    rho_step=0.01,
    taus=[0.95],
    ensemble="GUE",
    nks=25,
    nst=15
)

# Generate figures
viz.plot_unreachability_three_criteria_vs_density(
    data=data,
    ensemble="GUE",
    outdir="fig/comparison",
    trials=375,
    y_axis="unreachable"
)
```

### Ensemble-Specific Behavior

| Ensemble | Operator Structure | Expected Behavior |
|----------|-------------------|-------------------|
| **GUE** | Dense (d² non-zeros) | Smooth transitions |
| **GOE** | Dense (d² non-zeros) | Similar to GUE |
| **Canonical** | Sparse (2 non-zeros each) | Sharp/discontinuous transitions |
| **GEO2** | Sparse Pauli chains | Sharper than GUE |

**Note on Canonical Basis:** Discontinuous behavior is physical, not a bug. With k sparse operators (each having only 2 non-zero elements), the parameterized Hamiltonian H(λ) can't span enough of Hilbert space at low density, leading to binary-like reachability.

### Continuous vs Binary Krylov

| Feature | Binary (old) | Continuous (new) |
|---------|--------------|------------------|
| Parameters | Fixed/Random λ | Optimized λ* |
| Output | Boolean | Score R* ∈ [0,1] |
| Threshold | Hard-coded | Adjustable τ |
| Comparison | Unfair | Fair (both optimize) |
| Statistics | None | Mean/SEM available |

---

**Version**: 0.3.0
**Last updated**: 2026-01-15
**Python**: 3.10+
**Status**: Production-ready with criterion ordering analysis and integrability study



## Criterion Ordering Analysis

### Overview

The **criterion ordering analysis** investigates the relative performance of Spectral vs Krylov criteria across different Hamiltonian ensembles and integrability levels.

### Key Finding: Krylov < Spectral is Universal

The Krylov criterion consistently detects reachability at **lower operator density** than Spectral:

| Ensemble | d | Ratio ρ_c(S)/ρ_c(K) | Interpretation |
|----------|---|---------------------|----------------|
| **GEO2** | 16 | 1.7 | Krylov needs 41% fewer operators |
| **GEO2** | 32 | 6.0 | Krylov needs 83% fewer operators |
| **GEO2** | 64 | 13.4 | Krylov needs 93% fewer operators |
| **Canonical** | 10-26 | ~1.6 | Krylov needs ~38% fewer operators |

### Scaling Analysis

Power-law fits K_c ∝ d^α reveal different scaling:

| Ensemble | Criterion | Exponent α | Interpretation |
|----------|-----------|------------|----------------|
| GEO2 | Krylov | 1.24 | Near-linear (efficient) |
| GEO2 | Spectral | 2.47 | Superquadratic (inefficient) |
| Canonical | Both | ~1.0 | Linear |

### λ-Independence of Krylov

The Krylov criterion is **nearly λ-independent** due to scaling invariance:
```
H(cλ)^k|ψ⟩ = c^k H(λ)^k|ψ⟩  →  Same span!
```

Only the **direction** of λ matters, not its magnitude. This explains why Fixed λ ≈ Optimized λ for Krylov.

### Scripts

```bash
# Generate publication figures (ratio, K_c scaling, λ explanation)
python scripts/analysis/create_publication_figures.py

# Run dimension dependence analysis
python scripts/analysis/dimension_dependence.py

# Analyze Fixed vs Optimized gap
python scripts/analysis/fixed_vs_optimized.py
```

### Output Files

- `fig/publication/main_ratio_vs_dimension.png` - Main result figure
- `fig/publication/kc_vs_dimension.png` - K_c scaling analysis
- `fig/publication/lambda_explanation.png` - Why Krylov is λ-independent
- `docs/CRITERION_ORDERING_ANALYSIS.md` - Detailed analysis documentation

---

## Integrability Study

### Overview

The **integrability study** investigates how quantum chaos (measured by level spacing statistics) affects criterion performance.

### Hamiltonian Models

1. **Integrable Ising**: H = Σ Jᵢ σᶻᵢσᶻᵢ₊₁ + Σ hᵢ σᶻᵢ (Poisson, ⟨r⟩ ≈ 0.39)
2. **Near-Integrable**: H = J Σ σᶻσᶻ + h Σ σᶻ + g Σ σˣ (tunable via g)
3. **Chaotic Heisenberg**: Random XX+YY+ZZ couplings (GOE, ⟨r⟩ ≈ 0.53)

### Key Results

| Model | ⟨r⟩ | Spectral | Krylov |
|-------|-----|----------|--------|
| Integrable | 0.39 | **FAILS** (P=1 always) | Works |
| Near-integrable | 0.16 | **FAILS** | Works |
| Chaotic | 0.54 | Works (ρ_c ≈ 0.04) | Works (ρ_c ≈ 0.04) |

**Key insight**: Spectral criterion **fails** for integrable systems because eigenstates are λ-independent. Krylov works universally because it probes dynamics, not eigenstructure.

### Running Experiments

```bash
# Basic three-model comparison (d=8, 16)
python scripts/integrability/three_models_comparison.py

# Extended study (d=8, 16, 32 + g-sweep)
python scripts/integrability/extended_integrability_study.py --mode both --trials 50
```

### Output Files

- `fig/integrability/three_models_comparison.png` - Main comparison figure
- `fig/integrability/extended_integrability_comparison.png` - d=32 results
- `fig/integrability/g_sweep_analysis.png` - Transverse field sweep
- `docs/INTEGRABILITY_METHODOLOGY.md` - Complete methodology documentation

---

## GEO2LOCAL Experiments

### Overview
GEO2LOCAL studies quantum reachability for 2D geometric lattice Hamiltonians with local (2-body) interactions.

### Default Plotting
```bash
python3 scripts/plot_geo2_v3.py
```

### Key Results
- **Linear scaling:** ρ_c = 0.0455 + 0.00220×d (R² = 0.929)
- **Spectral criterion:** Requires λ-optimization (Fixed flat at P=1)
- **Krylov criterion:** Approximately λ-independent (Fixed ≈ Optimized)
- **Moment criterion:** Very weak (always satisfied at low ρ)

### Data Location
- Production data: `data/raw_logs/geo2_production_complete_*.pkl`
- Figures: `fig/geo2/geo2_*_v3.png`

### Detailed Analysis
See `docs/GEO2_ANALYSIS_SUMMARY.md` for comprehensive results.
