# Audit Checklist for Reachability Project

## Code Quality

### Core Library (reach/)
- [ ] mathematics.py: All functions have docstrings
- [ ] mathematics.py: Type hints on public functions
- [ ] mathematics.py: No dead code
- [ ] optimize.py: All functions have docstrings
- [ ] optimize.py: Type hints on public functions
- [ ] optimize.py: No dead code
- [ ] models.py: All functions have docstrings
- [ ] models.py: Type hints on public functions

### Scripts
- [ ] All scripts have module-level docstrings
- [ ] All scripts are executable (shebang line)
- [ ] No hardcoded paths (use relative or configurable)
- [ ] Scripts handle missing files gracefully

## Data Integrity

### Overnight Data
- [ ] canonical_overnight_20260128_224236.csv exists and is valid
- [ ] geo2_overnight_20260128_224613.csv exists and is valid
- [ ] Column names match expected schema
- [ ] No NaN values in critical columns

### Corrected Krylov Data
- [ ] canonical_krylov_corrected_20260204_222726.csv exists
- [ ] geo2_krylov_corrected_20260204_222747.csv exists
- [ ] m=min(K,d) was used (verify in script logs or column values)

## Plot Verification

### Publication Plots
- [ ] All plots use unified color scheme (DIM_COLORS, CRIT_COLORS)
- [ ] All plots have consistent font sizes
- [ ] Fit curves visible at alpha=0.5
- [ ] No figure suptitles
- [ ] Axes labeled correctly

### Benchmark Plots
- [ ] Confusion matrices show 100 samples
- [ ] accuracy_vs_time.png has correct markers
- [ ] iterations_vs_accuracy.png shows all dimensions

## Documentation

### Markdown Files
- [ ] CLAUDE.md is up-to-date with current conventions
- [ ] README.md has accurate directory structure
- [ ] PUBLICATION_PLOTS.md lists all generated plots
- [ ] BENCHMARKS_README.md describes methodology completely

### LaTeX
- [ ] main-cl.tex compiles without errors (with revtex4-2)
- [ ] All figures referenced exist
- [ ] Table values match benchmark results
- [ ] Equations fit in single column

## Test Coverage

- [ ] Gradient tests (spectral and Krylov)
- [ ] Optimizer convergence tests
- [ ] Data integrity tests

## Cleanup Tasks

- [ ] Remove any temporary/debug files
- [ ] Remove any .pyc or __pycache__ directories
- [ ] Remove any unused scripts in scripts/
- [ ] Consolidate duplicate functionality
- [ ] Remove commented-out code blocks
