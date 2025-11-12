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

```bash
# Clone and install
git clone https://github.com/yourusername/reachability.git
cd reachability
pip install -e .

# Verify installation
python -m reach.cli --help
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

### Core Concept

Given initial state |ψ⟩, target state |φ⟩, and Hamiltonian family H(λ) = Σᵢ λᵢ Hᵢ:

**Spectral overlap**:
```
S(λ) = Σₙ |⟨n(λ)|φ⟩* ⟨n(λ)|ψ⟩| ∈ [0,1]
```

where |n(λ)⟩ are eigenstates of H(λ).

**Unreachability criterion** (spectral, τ-based):
```
max_{λ∈[-1,1]ᴷ} S(λ) < τ  →  unreachable
```

We estimate `P(unreachability)` via Monte Carlo over random ensembles.

### Three Criteria Compared

1. **Spectral** (NEW): τ-based threshold on maximum overlap
2. **Moment** (τ-free): Moment-based definiteness check (classical)
3. **Krylov**: Projection-residual test on Krylov subspace

See detailed mathematical descriptions below.

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
