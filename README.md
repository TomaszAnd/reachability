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
git clone https://github.com/yourusername/reachability.git
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

**Version**: 0.1.0
**Last updated**: 2025-10-22
**Python**: 3.10+
**Status**: Production-ready with streaming CSV and incremental plotting
