# Publication-Ready Plots - Quick Start Guide

## ✅ Implementation Status: COMPLETE

All requirements have been implemented and validated. The code is ready to generate publication-quality plots.

## Validation Results

```bash
$ python validate_implementation.py

✓ test_dimension_validation: PASSED
✓ test_filenames: PASSED
✓ test_csv_schema: PASSED
✓ test_floor_aware_helper: PASSED
✓ test_legend_format: PASSED
✓ test_styling_parameters: PASSED

Total: 6/6 tests passed
```

## Quick Start

### Generate All Plots (Recommended)

```bash
./run_production_sweeps.sh
```

**Time**: 30-45 minutes
**Output**: 8 PNG files + 2 CSV files in `fig_summary/`

### What Gets Generated

#### Density Plots (6 files)
- `three_criteria_vs_density_GUE_tau0.90_unreachable.png`
- `three_criteria_vs_density_GUE_tau0.90_reachable.png`
- `three_criteria_vs_density_GUE_tau0.95_unreachable.png`
- `three_criteria_vs_density_GUE_tau0.95_reachable.png`
- `three_criteria_vs_density_GUE_tau0.99_unreachable.png`
- `three_criteria_vs_density_GUE_tau0.99_reachable.png`

Each shows:
- X-axis: K/d² (density from 0 to 0.15)
- Y-axis: log₁₀ P(unreachable) or log₁₀ P(reachable)
- All 3 criteria: Spectral (at that τ), Old, Krylov
- All 4 dimensions: d ∈ {20, 30, 40, 50}

#### K-Sweep Plots (2 files)
- `K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_unreachable.png`
- `K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_reachable.png`

Each shows:
- X-axis: K (number of Hamiltonians, 2-14)
- Y-axis: log₁₀ P(unreachable) or log₁₀ P(reachable)
- Spectral at 3 τ values (blue gradient: light→dark)
- Old criterion (dashed)
- Krylov (dotted, m=min(K,d))

#### CSV Files (2 files)
- `density_gue.csv`: All density sweep data
- `k30_gue.csv`: All K-sweep data

## Key Features Implemented

### ✅ Hard Requirements Met

1. **Dimension Validation**: Enforces exactly {20, 30, 40, 50}
   ```
   ERROR if wrong dimensions:
   "Density sweep requires EXACTLY dims=[20, 30, 40, 50], got dims=[20, 30]"
   ```

2. **Floor-Aware Plotting**:
   - No vertical "cliff" segments
   - Floored points shown as faded markers (alpha=0.3)
   - Line segments broken at floor using masked arrays
   - Annotation: "Display floor: 1e-12"

3. **Publication Styling**:
   - 14×10 inches, 200 DPI
   - Bold 16pt axis labels, 18pt title, 12pt legend
   - linewidth=2.0, markersize=6
   - Grid with minor ticks

4. **Correct Legends**:
   - Density: "Spectral (τ=0.95) • d=30", "Old • d=40", "Krylov (m=min(K,d)) • d=50"
   - K-sweep: "Spectral (τ=0.90)", "Old criterion", "Krylov (m=min(K,d))"

5. **CSV Logging**: All 15 fields, appends to existing files

## Manual Commands

If you prefer to run individually:

```bash
# Density sweep (unreachable)
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.15 --rho-step 0.01 --taus 0.90,0.95,0.99 \
  --trials 150 --y unreachable --csv fig_summary/density_gue.csv

# Density sweep (reachable)
python -m reach.cli --summary three-criteria-vs-density \
  --ensemble GUE --dims 20,30,40,50 \
  --rho-max 0.15 --rho-step 0.01 --taus 0.90,0.95,0.99 \
  --trials 150 --y reachable --csv fig_summary/density_gue.csv

# K-sweep (unreachable)
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE -d 30 --k-max 14 --taus 0.90,0.95,0.99 \
  --trials 300 --y unreachable --csv fig_summary/k30_gue.csv

# K-sweep (reachable)
python -m reach.cli --summary three-criteria-vs-K-multi-tau \
  --ensemble GUE -d 30 --k-max 14 --taus 0.90,0.95,0.99 \
  --trials 300 --y reachable --csv fig_summary/k30_gue.csv
```

## Documentation

- **`IMPLEMENTATION_COMPLETE.md`**: Full implementation details and validation
- **`PUBLICATION_READY_IMPLEMENTATION.md`**: Complete technical specification
- **`FLOOR_AWARE_PLOTTING_SUMMARY.md`**: Floor-aware plotting details
- **`run_production_sweeps.sh`**: Automated production script

## Validation

Run validation tests:
```bash
python validate_implementation.py
```

Expected: **6/6 tests passed**

---

**Ready to generate production plots**: Run `./run_production_sweeps.sh`
