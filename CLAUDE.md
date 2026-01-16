# Working with Claude Code Chat on `reach/`

This guide helps Claude Code Chat (and other AI coding assistants) work effectively on this quantum reachability analysis repository.

---

## Goals

- **Small, atomic PRs**: Each change should be reviewable in one sitting
- **Incremental edits**: Prefer targeted `Edit` operations over full file rewrites
- **Clear commit messages**: Use conventional commit format (`feat:`, `fix:`, `docs:`, `test:`)
- **Documentation-driven**: Update docs alongside code changes

---

## Ground Rules

### ✅ Do

- Stage and commit frequently (small, logical units)
- Show diffs before committing (`git diff --staged`)
- Use `Edit` tool for targeted changes (preserves context)
- Run validation after changes (`validate_implementation.py`, `validate_streaming_mode.py`)
- Check for side effects (linting, imports, dependencies)
- Reference line numbers when discussing code (e.g., `cli.py:425`)

### ❌ Don't

- Rewrite git history (no `git commit --amend` on pushed commits, no force pushes)
- Run external services without explicit permission
- Make unrelated changes in the same commit
- Skip validation/tests when available
- Use bash for file operations when specialized tools exist (use `Read`, `Edit`, `Write`, `Grep`, `Glob`)

---

## Standard Commands & Workflows

### Searching & Exploring

```bash
# Find files by pattern
ls -la reach/*.py
ls -la scripts/*.sh

# Search code for patterns
grep -r "DISPLAY_FLOOR" reach/
grep -n "class.*Writer" reach/logging_utils.py

# Check what's changed
git status
git diff
git log --oneline -n 10
```

### Running Tests & Validation

```bash
# Smoke tests (fast)
pytest tests/ -v

# Implementation validation (checks dimension constraints, CSV schema, etc.)
python validate_implementation.py

# Streaming mode validation (checks plot-from-csv functionality)
python validate_streaming_mode.py
```

### Running Sweeps

**Fast demo** (~5-10 minutes, reduced parameters):
```bash
./run_production_sweeps_FAST.sh
```

**Full production** (~2-3 hours, publication-quality):
```bash
nohup ./run_production_sweeps.sh > production.log 2>&1 &
echo $! > production.pid
```

**Single commands** (for testing):
```bash
# Density sweep
python -m reach.cli three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.15 --rho-step 0.01 \
  --taus 0.90,0.95,0.99 --trials 150 \
  --csv fig/comparison/density_gue.csv \
  --flush-every 10 --y unreachable

# K-sweep
python -m reach.cli three-criteria-vs-K-multi-tau \
  --ensemble GUE -d 30 --k-max 14 \
  --taus 0.90,0.95,0.99 --trials 300 \
  --csv fig/comparison/k30_gue.csv \
  --flush-every 10 --y unreachable
```

**GEO2 ensemble** (geometric two-local Hamiltonians on lattices):
```bash
# GEO2 requires lattice parameters: --nx and --ny
# Dimension = 2^(nx×ny), so 2×2 lattice → 4 qubits → d=16

# K-sweep for 2×2 lattice (d=16)
python -m reach.cli --nx 2 --ny 2 three-criteria-vs-K-multi-tau \
  --ensemble GEO2 -d 16 --k-max 10 \
  --taus 0.99 --trials 100 --y unreachable \
  --csv fig/comparison/k16_geo2.csv

# Optional: --periodic for periodic boundary conditions
# python -m reach.cli --nx 2 --ny 2 --periodic three-criteria-vs-K-multi-tau ...

# Verification: Check operator count L = 3n + 9|E|
# For 2×2 open: n=4 sites, |E|=4 edges → L = 12 + 36 = 48 ✓
# For 2×2 periodic: n=4 sites, |E|=8 edges → L = 12 + 72 = 84 ✓

# GEO2 ensemble definition: https://arxiv.org/html/2510.06321v1
# Note: GEO2 uses qubit lattices with d=2^n (power-of-2 dimensions only)
#       Filenames reflect true dimensions (e.g., d512 for 3×3 lattice)
#       For density sweeps, use 3×3 or larger to get meaningful K/d² sampling
```

### Refreshing Plots from CSV (Streaming Mode)

**While a long run is in progress**, you can peek at partial results:

```bash
# Refresh all plots from existing CSVs
./scripts/plot_refresh.sh

# Refresh density plots only
./scripts/plot_refresh.sh density

# Refresh K-sweep plots only
./scripts/plot_refresh.sh k-multi-tau

# Manual plot-from-csv command
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable
```

### Monitoring Long Runs

**Two-terminal workflow**:

Terminal 1 (run sweep):
```bash
./run_production_sweeps.sh
```

Terminal 2 (monitor progress):
```bash
# Check process status
ps aux | grep reach.cli

# Watch log
tail -f production.log

# Check CSV growth
watch -n 30 'wc -l fig/comparison/density_gue.csv'

# Refresh plots periodically
watch -n 300 './scripts/plot_refresh.sh density'
```

**Check for completion**:
```bash
# Verify output files
ls -lh fig/comparison/three_criteria_vs_density_GUE*.png
ls -lh fig/comparison/K_sweep_multi_tau_GUE*.png

# Count CSV rows
wc -l fig/comparison/density_gue.csv  # Should be ~193
wc -l fig/comparison/k30_gue.csv      # Should be ~66
```

### Committing Changes

```bash
# Stage specific files
git add reach/cli.py reach/logging_utils.py

# Show what's staged
git diff --staged

# Commit with clear message
git commit -m "feat: add --flush-every flag for streaming CSV writes"

# Atomic commits (one logical change per commit)
git log --oneline -n 5
```

---

## When the Run is Long

### Streaming CSV Mode

The `--flush-every N` flag enables incremental CSV writing:

- **What it does**: Flushes CSV buffer to disk every N data points
- **Why**: Long runs can be interrupted and resumed; you can plot partial results mid-run
- **Default**: `flush_every=10` (good balance between I/O and data safety)
- **Context manager**: `StreamingCSVWriter` in `reach/logging_utils.py` automatically flushes remaining buffer on exit/interrupt

### Plot-from-CSV Workflow

Instead of recomputing, read existing CSV and generate plots:

```bash
# Generate plots from partial or complete CSV
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable \
  --outdir fig_summary/

# Optional: filter to specific tau values
python -m reach.cli plot-from-csv \
  --csv fig/comparison/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable \
  --taus 0.95,0.99
```

**Handles gracefully**:
- Partial CSVs (missing dimensions or data points)
- Mixed run IDs (appends from multiple runs)
- Empty files (clear error messages)

### Helper Scripts

**`scripts/plot_refresh.sh`** - Re-render all plots from existing CSVs:
```bash
./scripts/plot_refresh.sh           # All plots
./scripts/plot_refresh.sh density   # Density plots only
./scripts/plot_refresh.sh k-multi-tau  # K-sweep plots only
```

---

## Checklists

### PR Checklist

- [ ] Code changes are minimal and focused
- [ ] Tests/validation scripts pass
- [ ] Documentation updated (README, docstrings)
- [ ] Clear commit messages (conventional format)
- [ ] No unrelated changes bundled together
- [ ] Git history is clean (no merge commits unless intentional)

### Plotting Checklist

- [ ] Floor-aware rendering (no vertical cliffs at display floor)
- [ ] Correct filenames (matches spec in docs)
- [ ] Dimensions validated for density plots: exactly {20, 30, 40, 50}
- [ ] Both y-axis versions generated (unreachable + reachable)
- [ ] CSV has all 15 fields with correct schema
- [ ] Legends formatted correctly (e.g., "Spectral (τ=0.95) • d=30")
- [ ] Publication styling (14×10 inches, 200 DPI, bold labels)

### Visualization Standards (CRITICAL ⚠️)

**DO NOT use artificial clipping for linearized fits**:

❌ **WRONG** (Artificial clipping):
```python
EPSILON = 1e-6
P_clipped = np.clip(P, EPSILON, 1 - EPSILON)
log_P = np.log(P_clipped)  # Creates artificial plateaus at boundaries
```

✅ **CORRECT** (Physical treatment):
```python
# Classify points by their physical state
k = np.round(P * N).astype(int)  # Recover trial counts
transition_mask = (k > 0) & (k < N)  # Exclude k=0 and k=N

# Fit only informative transition region
P_trans = P[transition_mask]
log_P = np.log(P_trans)  # No clipping needed - all 0 < P < 1

# Report quality metrics
n_trans = np.sum(transition_mask)
frac_trans = n_trans / len(P)
quality = 'good' if (n_trans >= 10 and frac_trans >= 0.20) else 'marginal'
```

**Why this matters**:
- P=0 and P=1 are genuine binomial outcomes (k=0/N or k=N/N), not numerical errors
- Artificial clipping creates fake plateaus and biases slope estimates
- Wilson score intervals provide proper uncertainty bounds without arbitrary ε
- Transition-region-only fitting gives honest R² values and quality assessment

**Reference**: See `docs/clipping_methodology.md` and `scripts/plot_linearized_physical.py` for complete implementation.

### Filename Conventions

**Density plots**:
```
three_criteria_vs_density_{ensemble}_tau{tau:.2f}_{y}.png

Examples:
  three_criteria_vs_density_GUE_tau0.90_unreachable.png
  three_criteria_vs_density_GUE_tau0.95_reachable.png
```

**K-sweep plots**:
```
K_sweep_multi_tau_{ensemble}_d{d}_taus{tau_str}_{y}.png

Examples:
  K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_unreachable.png
  K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_reachable.png
```

**CSV files**:
```
density_gue.csv      # Density sweep data (all dimensions, all tau)
k30_gue.csv          # K-sweep data (d=30, multiple tau)
```

---

## Directory Structure

```
reachability/
├── reach/              # Core Python package (13 modules)
├── scripts/
│   ├── analysis/       # Criterion ordering analysis scripts
│   ├── integrability/  # Integrability study scripts
│   ├── canonical/      # Canonical basis scripts (publication figures)
│   ├── geo2/           # GEO2 lattice experiment scripts
│   ├── floquet/        # Floquet engineering scripts
│   ├── utils/          # Utility/monitoring scripts
│   └── archive/        # Deprecated scripts
├── data/
│   ├── raw_logs/       # Pickle files with experimental data
│   └── analysis/       # Processed analysis results
├── fig/
│   ├── publication/    # Publication-ready criterion ordering figures
│   ├── integrability/  # Integrability study figures
│   ├── canonical/      # Canonical basis publication figures
│   ├── geo2/           # GEO2 lattice experiments
│   ├── floquet/        # Floquet engineering experiments
│   ├── spectral/       # Spectral criterion figures
│   ├── analysis/       # Analysis comparison figures
│   ├── krylov/         # Krylov criterion figures
│   └── comparison/     # Multi-criteria comparisons
├── docs/               # Documentation (~12 current files)
│   ├── archive/        # Historical documentation (55+ files)
│   │   ├── floquet/    # Archived Floquet docs
│   │   ├── geo2/       # Archived GEO2 docs
│   │   └── historical/ # Other archived docs
│   └── *.md            # Current documentation
├── logs/               # Execution logs
│   └── archive/        # Old log files
├── results/            # Experimental results
└── tests/              # Unit tests
```

---

## File Map

| Path | Purpose |
|------|---------|
| `reach/cli.py` | Command-line interface, argument parsing, command dispatch |
| `reach/analysis.py` | Pure computation: Monte Carlo loops, density/K sweeps |
| `reach/viz.py` | Pure rendering: plot functions (matplotlib), includes CSV-based plotting |
| `reach/logging_utils.py` | CSV logging, `StreamingCSVWriter` class |
| `reach/models.py` | GOE/GUE ensemble generation, random states |
| `reach/mathematics.py` | Eigendecomposition, spectral overlap, Krylov tests |
| `reach/optimize.py` | Maximization of S(λ) using scipy.optimize |
| `reach/settings.py` | All configuration constants, defaults, hyperparameters |
| `scripts/utils/plot_refresh.sh` | Helper to re-render plots from CSV |
| `scripts/canonical/generate_summary_figs.py` | Generate spectral/comparison figures |
| `scripts/canonical/generate_publication_dual_versions.py` | Generate v7 publication figures |
| `scripts/geo2/plot_geo2_v3.py` | Generate GEO2 figures |
| `scripts/floquet/generate_floquet_plots.py` | Generate Floquet figures |
| `scripts/analysis/create_publication_figures.py` | Criterion ordering publication figures |
| `scripts/analysis/dimension_dependence.py` | Dimension scaling analysis |
| `scripts/analysis/fixed_vs_optimized.py` | λ-optimization gap analysis |
| `scripts/integrability/three_models_comparison.py` | Integrability study (3 models) |
| `scripts/integrability/extended_integrability_study.py` | Extended study (d=32, g-sweep) |
| `run_production_sweeps.sh` | Full production run (2-3 hours, trials=150/300) |
| `run_production_sweeps_FAST.sh` | Fast demo (5-10 min, reduced parameters) |

---

## Reproducing Publication Figures

### Quick Start

Generate all v7 dual-version publication figures:

```bash
python scripts/canonical/generate_publication_dual_versions.py 2>&1 | tee logs/dual_version_generation.log
```

This creates 8 publication figures (4 plots × 2 versions: exp and pow2) in `fig/canonical/`:
- `final_summary_3panel_v7_{exp,pow2}.png` - 3-panel decay curves
- `combined_criteria_d26_v7_{exp,pow2}.png` - All criteria at d=26
- `Kc_vs_d_v7_{exp,pow2}.png` - Critical K scaling with dimension
- `linearized_fits_v7_{exp,pow2}.png` - Linearized fits (ln vs log₂)

### Data Provenance

Complete data lineage documentation in **[DATA_PROVENANCE.md](DATA_PROVENANCE.md)**:
- Raw data file inventory with timestamps and sources
- Merging strategies for Moment, Spectral, and Krylov datasets
- Data structure access patterns
- Critical bugs and fixes
- Coverage maps

### Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/canonical/generate_publication_dual_versions.py` | Generate all v7 dual-version figures |
| `scripts/load_moment_data.py` | Merge OLD comprehensive + NEW extension |
| `scripts/load_spectral_data.py` | Load merged Spectral data |
| `scripts/load_krylov_data.py` | Merge FIXED + DENSE Krylov data |

### Dual-Version Implementation

The v7 publication figures support **two Moment fit formulas**:

**EXP version** (physical parameterization):
```
P(ρ) = exp(-α d² (ρ - ρ_c))
K_c = d² ρ_c + ln(2)/α
Linearization: ln(P) = -α d² (ρ - ρ_c)
```

**POW2 version** (half-life parameterization):
```
P(ρ) = 2^(-ρ/ρ_c)
K_c = d² ρ_c
Linearization: log₂(P) = -ρ/ρ_c
```

Both versions use the same Spectral/Krylov Fermi-Dirac fits:
```
P(ρ) = 1/(1 + exp((ρ - ρ_c)/Δ))
K_c = d² ρ_c
```

### Reproduction Workflow

1. **Check data files exist**:
```bash
ls -lh data/raw_logs/comprehensive_reachability_20251209_153938.pkl
ls -lh data/raw_logs/moment_extension_all_dims_20251215_160333.pkl
ls -lh data/raw_logs/spectral_complete_merged_20251216_153002.pkl
ls -lh data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl
ls -lh data/raw_logs/krylov_dense_20251216_112335.pkl
```

2. **Generate figures**:
```bash
python scripts/canonical/generate_publication_dual_versions.py 2>&1 | tee logs/dual_version_generation.log
```

3. **Verify outputs**:
```bash
ls -lh fig/canonical/*_v7_*.png
# Should show 8 files (~250-750 KB each)
```

4. **Check logs for fit quality**:
```bash
cat logs/dual_version_generation.log | grep "R²"
# All R² > 0.95 for good fits
```

---

## Production Documentation

All production workflows, mathematical foundations, and usage examples are now consolidated in **[README.md](README.md)**.

---

## Common Patterns

### Adding a new CLI subcommand

1. Add parser in `create_parser()` in `cli.py`
2. Create handler function `cmd_<name>(args)` in `cli.py`
3. Add to `command_map` in `main()`
4. Update `README.md` and this file with usage example
5. Add validation test if applicable

### Adding a new plot type

1. Implement plotting function in `viz.py` (supports both computed data and CSV input)
2. Add CLI subcommand or extend existing one
3. Update filename conventions in this file
4. Add to plotting checklist
5. Test with real and mock data

### Modifying CSV schema

1. Update `REACHABILITY_CSV_FIELDS` in `logging_utils.py`
2. Update all CSV writers to include new fields
3. Update `validate_implementation.py` to check schema
4. Update documentation (this file, README, production docs)
5. Handle backward compatibility if CSVs exist

---

## Tips for AI Assistants

- **Context is king**: Always `Read` files before editing
- **Show your work**: Display diffs before committing
- **Incremental wins**: Small PRs are easier to review and safer to merge
- **Trust but verify**: Run validation after code changes
- **Documentation debt**: Update docs when changing behavior
- **User intent**: If unclear, ask before making assumptions

---

## Emergency Procedures

**If a long run needs to be stopped**:
```bash
# Find process ID
ps aux | grep reach.cli

# Graceful stop (allows CSV flush)
kill <PID>

# Force stop (if hung)
kill -9 <PID>

# Check what was saved
ls -lh fig/comparison/*.csv
wc -l fig/comparison/*.csv
```

**If plots are missing/wrong**:
```bash
# Re-generate from existing CSV (no recomputation)
./scripts/plot_refresh.sh

# Validate CSV schema
python -c "import pandas as pd; df = pd.read_csv('fig/comparison/density_gue.csv'); print(df.columns.tolist())"

# Check for dimension violations
python validate_implementation.py
```

**If commits need cleanup** (before pushing):
```bash
# Amend last commit (add forgotten file)
git add <file>
git commit --amend --no-edit

# Interactive rebase (squash commits)
git rebase -i HEAD~3

# BUT: Never rewrite pushed history without team consensus
```

---

---

## Ensemble Support

The reachability analysis supports four Hamiltonian ensembles:

### 1. GUE (Gaussian Unitary Ensemble)
- Complex Hermitian matrices with Gaussian distribution
- Generic quantum systems without time-reversal symmetry
- **Operator structure**: Dense (all d² elements non-zero)
- **Status**: ✅ Fully validated, production-ready

### 2. GOE (Gaussian Orthogonal Ensemble)
- Real symmetric matrices
- Time-reversal symmetric systems
- **Operator structure**: Dense (all d² elements non-zero)
- **Status**: ✅ Available, similar behavior to GUE

### 3. Canonical Basis
- Structured basis: {X_jk, Y_jk, Z_j, I}
- Pauli-like operators generalized to arbitrary dimension d
- **Operator structure**: Extremely sparse (only 2 non-zero elements per operator)
- **Status**: ✅ Validated (discontinuous behavior is physical)
- **Note**: Sharp transitions expected due to sparse structure - at low k/d², operators can't span enough of Hilbert space

**Canonical Basis Validation Results** (d=8):
| k | Non-zeros | Spectral S* | GUE S* |
|---|-----------|-------------|--------|
| 3 | 6/64 (9%) | 0.62 | 0.96 |
| 8 | 16/64 (25%) | 0.98 | 1.00 |
| 16 | 32/64 (50%) | 1.00 | 1.00 |

### 4. GEO2 (2D Geometric Lattice)
- Nearest-neighbor couplings on 2D qubit grid
- Parameters: nx, ny (grid size), periodic (boundary conditions)
- Dimension: d = 2^(nx×ny), so d=16 for 2×2, d=512 for 3×3
- **Operator structure**: Sparse Pauli chains (L = 3n + 9|E| operators)
- **Status**: ⚠️ Limited testing (need higher resolution sweeps)
- **Note**: Sharper transitions expected due to geometric constraints

---

## Continuous Krylov Implementation

### Mathematical Formulation

**Krylov score:** R(λ) = ‖P_Kₘ(H(λ))|φ⟩‖²

where P_Kₘ is projection onto Krylov subspace K_m(H(λ), |ψ⟩).

**Optimization:** R* = max_{λ ∈ [-1,1]ᴷ} R(λ) via L-BFGS-B

**Threshold criterion:** Unreachable if R* < τ

### Implementation Details

- `krylov_score()` in mathematics.py
- `maximize_krylov_score()` in optimize.py (line ~299)
- Integrated into `monte_carlo_unreachability_vs_density()` (analysis.py)
- Both Spectral (S*) and Krylov (R*) now τ-dependent

### Key Insight: R(λ) = 1 - ε²_res

The Krylov score equals 1 minus the squared residual norm, verified to 10 decimal places.

### Validation Status

- ✅ Unit tests pass (test_krylov_continuous.py)
- ✅ Integration tests pass
- ✅ Verified: R(λ) = 1 - ε²_res mathematically
- ✅ GUE ensemble: Smooth curves, correct physical trends
- ✅ Canonical basis: Sharp transitions confirmed as physical (sparse operators)
- ⚠️ GEO2: Needs higher resolution testing (only d=16 tested)

---

**Last updated**: 2025-11-21
**For questions**: See README.md, production docs, or ask the user



## GEO2LOCAL Workflow

### Data Generation
```bash
python3 scripts/geo2/run_geo2_production.py
```
- Dimensions: d=16, 32, 64
- Runtime: ~40 hours

### Plotting (Canonical v3 Style)
```bash
python3 scripts/geo2/plot_geo2_v3.py
```
- Output: `fig/geo2/geo2_*_v3.png`
- Style: Wheat-colored equation text boxes

### Key Files
| Purpose | Location |
|---------|----------|
| Production data | `data/raw_logs/geo2_production_complete_*.pkl` |
| Plotting script | `scripts/geo2/plot_geo2_v3.py` |
| Analysis summary | `docs/GEO2_ANALYSIS_SUMMARY.md` |

### Scientific Findings
- Linear scaling: ρ_c = 0.0455 + 0.00220×d
- Krylov is λ-independent (Fixed ≈ Optimized)
- Spectral requires λ-optimization

---

## Criterion Ordering Analysis Workflow

### Overview
Analyzes relative performance of Spectral vs Krylov criteria across ensembles.

### Key Finding
**Krylov < Spectral is universal**: Krylov detects reachability at lower operator density.

| Ensemble | d | Ratio ρ_c(S)/ρ_c(K) |
|----------|---|---------------------|
| GEO2 | 16→64 | 1.7 → 13.4 (grows) |
| Canonical | 10→26 | ~1.6 (stable) |

### Scaling Exponents
- GEO2 Krylov: K_c ∝ d^1.24 (efficient)
- GEO2 Spectral: K_c ∝ d^2.47 (inefficient)
- Canonical: Both ∝ d^1.0 (linear)

### Generate Figures
```bash
python scripts/analysis/create_publication_figures.py
```

### Output Files
| File | Description |
|------|-------------|
| `fig/publication/main_ratio_vs_dimension.png` | Main result |
| `fig/publication/kc_vs_dimension.png` | K_c scaling |
| `fig/publication/lambda_explanation.png` | λ-independence |
| `docs/CRITERION_ORDERING_ANALYSIS.md` | Full documentation |

---

## Integrability Study Workflow

### Overview
Studies how quantum chaos affects criterion performance using level spacing statistics.

### Models
1. **Integrable Ising**: Diagonal, ⟨r⟩ ≈ 0.39 (Poisson)
2. **Near-Integrable**: Tunable via transverse field g
3. **Chaotic Heisenberg**: Random couplings, ⟨r⟩ ≈ 0.53 (GOE)

### Key Finding
**Spectral FAILS for integrable systems** (eigenstates λ-independent)
**Krylov works universally** (probes dynamics, not eigenstructure)

### Run Experiments
```bash
# Basic study (d=8, 16)
python scripts/integrability/three_models_comparison.py

# Extended study (d=32, g-sweep, 50 trials)
python scripts/integrability/extended_integrability_study.py --mode both --trials 50
```

### Output Files
| File | Description |
|------|-------------|
| `fig/integrability/three_models_comparison.png` | 3-model comparison |
| `fig/integrability/extended_integrability_comparison.png` | d=32 results |
| `fig/integrability/g_sweep_analysis.png` | g-sweep analysis |
| `docs/INTEGRABILITY_METHODOLOGY.md` | Complete methodology |

### r-Ratio Reference Values
| Ensemble | ⟨r⟩ | Physical Character |
|----------|-----|-------------------|
| Poisson | 0.386 | Integrable |
| GOE | 0.531 | Chaotic (time-reversal) |
| GUE | 0.600 | Chaotic (no TR) |

---

**Last updated**: 2026-01-15
