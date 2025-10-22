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
  --csv fig_summary/density_gue.csv \
  --flush-every 10 --y unreachable

# K-sweep
python -m reach.cli three-criteria-vs-K-multi-tau \
  --ensemble GUE -d 30 --k-max 14 \
  --taus 0.90,0.95,0.99 --trials 300 \
  --csv fig_summary/k30_gue.csv \
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
  --csv fig_summary/k16_geo2.csv

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
  --csv fig_summary/density_gue.csv \
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
watch -n 30 'wc -l fig_summary/density_gue.csv'

# Refresh plots periodically
watch -n 300 './scripts/plot_refresh.sh density'
```

**Check for completion**:
```bash
# Verify output files
ls -lh fig_summary/three_criteria_vs_density_GUE*.png
ls -lh fig_summary/K_sweep_multi_tau_GUE*.png

# Count CSV rows
wc -l fig_summary/density_gue.csv  # Should be ~193
wc -l fig_summary/k30_gue.csv      # Should be ~66
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
  --csv fig_summary/density_gue.csv \
  --type density \
  --ensemble GUE \
  --y unreachable \
  --outdir fig_summary/

# Optional: filter to specific tau values
python -m reach.cli plot-from-csv \
  --csv fig_summary/density_gue.csv \
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

## File Map

| Path | Purpose |
|------|---------|
| `reach/cli.py` | Command-line interface, argument parsing, command dispatch |
| `reach/analysis.py` | Pure computation: Monte Carlo loops, density/K sweeps |
| `reach/viz.py` | Pure rendering: plot functions (uses matplotlib) |
| `reach/viz_csv.py` | CSV-based plotting for partial/incremental results |
| `reach/logging_utils.py` | CSV logging, `StreamingCSVWriter` class |
| `reach/models.py` | GOE/GUE ensemble generation, random states |
| `reach/mathematics.py` | Eigendecomposition, spectral overlap, Krylov tests |
| `reach/optimize.py` | Maximization of S(λ) using scipy.optimize |
| `reach/settings.py` | All configuration constants, defaults, hyperparameters |
| `scripts/plot_refresh.sh` | Helper to re-render plots from CSV |
| `run_production_sweeps.sh` | Full production run (2-3 hours, trials=150/300) |
| `run_production_sweeps_FAST.sh` | Fast demo (5-10 min, reduced parameters) |
| `validate_implementation.py` | Validation tests for production requirements |
| `validate_streaming_mode.py` | Validation tests for plot-from-csv functionality |

---

## Production Documentation Links

For detailed production workflows, see:

- **[README_PRODUCTION.md](README_PRODUCTION.md)** - Quick start for production runs
- **[PRODUCTION_RUN_GUIDE.md](PRODUCTION_RUN_GUIDE.md)** - Detailed runtime info, parameters, monitoring
- **[PRODUCTION_RUN_SUMMARY.md](PRODUCTION_RUN_SUMMARY.md)** - Comprehensive status, expected outputs
- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** - Implementation details, validation results

---

## Common Patterns

### Adding a new CLI subcommand

1. Add parser in `create_parser()` in `cli.py`
2. Create handler function `cmd_<name>(args)` in `cli.py`
3. Add to `command_map` in `main()`
4. Update `README.md` and this file with usage example
5. Add validation test if applicable

### Adding a new plot type

1. Implement plotting function in `viz.py` or `viz_csv.py`
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
ls -lh fig_summary/*.csv
wc -l fig_summary/*.csv
```

**If plots are missing/wrong**:
```bash
# Re-generate from existing CSV (no recomputation)
./scripts/plot_refresh.sh

# Validate CSV schema
python -c "import pandas as pd; df = pd.read_csv('fig_summary/density_gue.csv'); print(df.columns.tolist())"

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

**Last updated**: 2025-10-22
**For questions**: See README.md, production docs, or ask the user
