# Quantum Reachability: Multi-Dimensional Scaling Analysis Guide

**Created**: 2025-11-25
**Status**: Analysis tools ready, awaiting execution

---

## Overview

This guide describes the comprehensive analysis workflow for understanding how quantum reachability scales across:
- **Dimensions** (d = 10, 12, 14, 16, ...)
- **Thresholds** (τ = 0.85, 0.90, 0.95, 0.99)
- **Ensembles** (canonical, GEO2)
- **Criteria** (spectral, Krylov, moment)

---

## Scientific Questions

### Scaling Laws
1. How does critical density ρ_c scale with dimension?
   - Is there a power law: **ρ_c ∝ d^(-β)**?
   - What are the exponents β for each criterion?

2. Do transitions become sharper with increasing dimension?
   - Measure transition width: Δρ = ρ_90 - ρ_10
   - Does Δρ → 0 as d → ∞?

### Universality
3. Are canonical and GEO2 ensembles universal?
   - Do they follow the same scaling exponents?
   - Or does sparse structure (canonical) vs geometric structure (GEO2) matter?

### Threshold Sensitivity
4. How does τ affect transitions?
   - Does τ just shift curves uniformly (ρ_c → ρ_c + const)?
   - Or does it change slopes/shapes?

5. Are some criteria more sensitive to τ than others?
   - Does moment criterion show larger shifts?

---

## Analysis Tools

### 1. Separate-by-Criterion Plots

**Script**: `scripts/plot_by_criterion_separate.py`

**Purpose**: Reduce visual clutter by creating 3 clean plots instead of 1 plot with 9 curves.

**Output**:
- `canonical_spectral_vs_density_tau0.95.png` (3 curves: d=10,12,14)
- `canonical_krylov_vs_density_tau0.95.png` (3 curves: d=10,12,14)
- `canonical_moment_vs_density_tau0.95.png` (3 curves: d=10,12,14)

**Usage**:
```bash
# From computed data (fast if data exists)
python scripts/plot_by_criterion_separate.py \
    --dims 10,12,14 \
    --rho-max 0.15 \
    --rho-step 0.02 \
    --tau 0.95 \
    --trials 500 \
    --output-dir fig/comparison
```

**Runtime**: 3-4 hours (full Monte Carlo computation)

**Value**: Immediate visual clarity improvement

---

### 2. τ-Comparison Sweep

**Script**: `scripts/tau_comparison_sweep.py`

**Purpose**: Quantify threshold sensitivity by running with multiple τ values.

**Output**:
- `tau_comparison_spectral_canonical.png` (all d × τ combinations)
- `tau_comparison_krylov_canonical.png` (all d × τ combinations)
- `tau_comparison_moment_canonical.png` (all d × τ combinations)
- `critical_density_vs_tau_canonical.png` (ρ_c vs τ curves)

**Usage**:
```bash
# Standard run (4 tau values, trials=200 for speed)
python scripts/tau_comparison_sweep.py \
    --dims 10,12,14 \
    --taus 0.85,0.90,0.95,0.99 \
    --trials 200 \
    --output-dir fig/comparison

# High-quality run (more trials)
python scripts/tau_comparison_sweep.py \
    --dims 10,12,14 \
    --taus 0.85,0.90,0.95,0.99 \
    --trials 500 \
    --output-dir fig/comparison
```

**Runtime**:
- trials=200: ~2-3 hours (4 tau values × 21 points = 84 density points)
- trials=500: ~6-8 hours (high quality)

**Value**: Reveals how threshold choice affects all criteria

---

### 3. Scaling Analysis Plot

**Script**: `scripts/plot_scaling_analysis.py`

**Purpose**: Create unified 2×2 figure showing all scaling behaviors.

**Panels**:
1. **Critical Density vs Dimension**: Shows ρ_c ∝ d^(-β) with fitted exponents
2. **Transition Width vs Dimension**: Shows if transitions sharpen with d
3. **Critical Density vs Tau**: Shows threshold sensitivity
4. **Ensemble Comparison**: Shows canonical vs GEO2 universality

**Usage**:
```bash
# From saved pickle data (instant!)
python scripts/plot_scaling_analysis.py \
    --canonical-data data/canonical_highres_20251125.pkl \
    --geo2-data data/geo2_highres_20251125.pkl \
    --output fig/comparison/scaling_analysis.png
```

**Runtime**: <10 seconds (if data files exist)

**Value**: Complete physics picture in one figure

**Note**: Requires saved data files. See "Data Pipeline" section below.

---

### 4. Log-Scale Plot

**Script**: `scripts/plot_log_scale_canonical.py`

**Purpose**: Visualize exponential decay P(unreachable) ∝ exp(-αρ) on log scale.

**Output**: `canonical_log_scale_tau0.95_highres.png`

**Usage**:
```bash
python scripts/plot_log_scale_canonical.py \
    --dims 10,12,14 \
    --rho-max 0.15 \
    --rho-step 0.02 \
    --tau 0.95 \
    --trials 500 \
    --recompute \
    --output fig/comparison/canonical_log_scale_tau0.95_highres.png
```

**Runtime**: 3-4 hours (full Monte Carlo)

**Value**: Shows decay constants α directly (slopes on log scale)

**Status**: Currently running (started 17:16 CET, ETA ~20:30 CET)

---

## Data Pipeline for Efficient Replotting

### Problem
Current workflow: Compute → Plot → (want different plot style) → Recompute (3-4 hours!)

### Solution
Save raw data once, replot instantly in any style.

### Implementation

**Step 1**: Modify production scripts to save data

```python
# In scripts/generate_production_plots.py or custom script
import pickle
from datetime import datetime

# After monte_carlo_unreachability_vs_density() call:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_file = f"data/production_runs/canonical_highres_{timestamp}.pkl"

Path(data_file).parent.mkdir(parents=True, exist_ok=True)

with open(data_file, 'wb') as f:
    pickle.dump({
        'data': data,  # The computed Monte Carlo results
        'metadata': {
            'ensemble': 'canonical',
            'dims': dims,
            'rho_max': rho_max,
            'rho_step': rho_step,
            'taus': taus,
            'trials': trials,
            'timestamp': timestamp,
        }
    }, f)

print(f"Saved raw data to {data_file}")
```

**Step 2**: Create symlink for "latest" data

```bash
cd data/production_runs
ln -sf canonical_highres_20251125_174500.pkl canonical_highres_latest.pkl
```

**Step 3**: Use saved data in all plotting scripts

```python
# All plotting scripts check for --data-file argument
if args.data_file:
    with open(args.data_file, 'rb') as f:
        saved = pickle.load(f)
    data = saved['data']
    # Plot instantly!
else:
    # Fallback: recompute
    data = analysis.monte_carlo_unreachability_vs_density(...)
```

**Benefits**:
- Generate 10 different plot styles in 10 seconds (not 40 hours!)
- Experiment with visualizations freely
- Reproducibility: exact same data for all plots

---

## GEO2 d=32 Analysis

### Question
Can we run GEO2 for d=32 to bridge the gap between d=16 and d=64?

### Answer: **NOT SUPPORTED**

**Reason**: GEO2 implementation requires rectangular lattice (nx × ny)

```python
# In reach/models.py, class GeometricTwoLocal
# Dimension: d = 2^(nx × ny)

# 2×2 lattice: 4 qubits → d = 2^4 = 16 ✅
# 2×3 lattice: 6 qubits → d = 2^6 = 64 ✅
# 5 qubits: No rectangular lattice exists ❌
```

**d=32 requires 5 qubits** (2^5 = 32), but there's no rectangular grid with 5 sites.

### Alternatives

#### Option A: Enhance d=16 (RECOMMENDED)
Run GEO2 with higher quality parameters:

```bash
python scripts/generate_geo2_publication.py \
    --config 2x2 \
    --trials 800 \    # 2× current (400 → 800)
    --rho-step 0.004 \  # 2× finer (0.008 → 0.004)
    --tau 0.95 \
    --output-dir fig/comparison
```

**Runtime**: ~50 minutes
**Benefit**: Gold-standard reference for GEO2 d=16

#### Option B: Custom Graph Implementation (NOT RECOMMENDED)
Modify `GeometricTwoLocal` class to accept arbitrary NetworkX graphs:

```python
# Would require significant code changes:
# - Replace _build_lattice_edges() with graph.edges()
# - Update dimension validation
# - Test operator count formula

# Effort: ~2-4 hours of development + testing
# Value: Limited (d=32 doesn't add much scientific insight)
```

**Recommendation**: Skip d=32, focus on high-quality d=16

---

## Execution Strategy

### Priority Ranking

#### HIGH PRIORITY (Do these first)

**1. τ-Comparison Sweep** (trials=200)
```bash
nohup python scripts/tau_comparison_sweep.py \
    --dims 10,12,14 \
    --taus 0.85,0.90,0.95,0.99 \
    --trials 200 \
    --output-dir fig/comparison \
    > logs/tau_comparison.log 2>&1 &
```
- **Runtime**: 2-3 hours
- **Value**: HIGH - reveals threshold sensitivity
- **Cost/benefit**: Excellent (new physics insight per compute hour)

**2. Implement Data Pipeline** (1 hour coding)
- Modify `generate_production_plots.py` to save pickle files
- Add `--data-file` argument to all plotting scripts
- **Value**: HIGH - enables all future instant replotting

**3. Generate Scaling Analysis Plot** (from existing data)
```bash
python scripts/plot_scaling_analysis.py \
    --canonical-data data/canonical_highres_latest.pkl \
    --output fig/comparison/scaling_analysis.png
```
- **Runtime**: <10 seconds (if data exists)
- **Value**: HIGH - comprehensive physics summary
- **Blocker**: Requires data pipeline (task #2)

#### MEDIUM PRIORITY

**4. Enhanced GEO2 d=16** (trials=800)
```bash
python scripts/generate_geo2_publication.py \
    --config 2x2 \
    --trials 800 \
    --rho-step 0.004 \
    --tau 0.95 \
    --output-dir fig/comparison
```
- **Runtime**: ~50 minutes
- **Value**: MEDIUM - improves existing plot quality

**5. Separate-by-Criterion Plots** (from saved data)
```bash
python scripts/plot_by_criterion_separate.py \
    --data-file data/canonical_highres_latest.pkl \
    --output-dir fig/comparison
```
- **Runtime**: <10 seconds (with data pipeline)
- **Value**: MEDIUM - visual clarity improvement

#### LOW PRIORITY / SKIP

**6. GEO2 d=32 Custom Graph** ❌ SKIP
- **Effort**: 2-4 hours development
- **Value**: LOW - doesn't add much scientific insight
- **Recommendation**: Focus on high-quality d=16 instead

**7. Additional Tau Values** ⚠️ SKIP (already have 4)
- Current: τ ∈ [0.85, 0.90, 0.95, 0.99]
- More values unlikely to reveal new physics

---

## Recommended Execution Plan

### Tonight (2025-11-25 Evening)

1. ✅ **DONE**: High-resolution canonical (trials=500) - Completed 15:49 CET
2. 🔄 **RUNNING**: Log-scale plot - Started 17:16 CET, ETA ~20:30 CET
3. **START**: τ-comparison sweep (trials=200) - Start after log-scale completes

**Command**:
```bash
# After log-scale completes (~20:30)
nohup python scripts/tau_comparison_sweep.py \
    --dims 10,12,14 \
    --taus 0.85,0.90,0.95,0.99 \
    --trials 200 \
    --output-dir fig/comparison \
    > logs/tau_comparison_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs/tau_comparison.pid
```

**Expected completion**: ~23:00-00:00 CET

### Tomorrow Morning (2025-11-26)

1. **Implement data pipeline** (1 hour)
   - Modify production scripts to save pickle files
   - Test loading and replotting

2. **Generate scaling analysis plot** (instant with data)
   - Use saved canonical highres data
   - Optionally include GEO2 d=16 data

3. **Generate separate-by-criterion plots** (instant with data)
   - Create 3 clean plots for canonical

4. **Optional: Enhanced GEO2 d=16** (50 min)
   - If time permits and gold standard desired

---

## Expected Outputs

### Plot Files

After full execution, you'll have:

```
fig/comparison/
├── three_criteria_vs_density_canonical_tau0.95_unreachable.png  # ✅ Exists (highres)
├── canonical_log_scale_tau0.95_highres.png                      # 🔄 Running
├── tau_comparison_spectral_canonical.png                        # ⏳ Pending
├── tau_comparison_krylov_canonical.png                          # ⏳ Pending
├── tau_comparison_moment_canonical.png                          # ⏳ Pending
├── critical_density_vs_tau_canonical.png                        # ⏳ Pending
├── canonical_spectral_vs_density_tau0.95.png                    # ⏳ Pending
├── canonical_krylov_vs_density_tau0.95.png                      # ⏳ Pending
├── canonical_moment_vs_density_tau0.95.png                      # ⏳ Pending
├── scaling_analysis.png                                         # ⏳ Pending
└── geo2_2x2_d16_publication_highres.png                         # ⏳ Optional
```

### Data Files

```
data/production_runs/
├── canonical_highres_20251125_174500.pkl  # To be created
├── canonical_highres_latest.pkl           # Symlink to latest
├── tau_comparison_20251125_203000.pkl     # To be created
└── geo2_highres_20251125_*.pkl            # Optional
```

---

## Scientific Insights Expected

### From τ-Comparison
- **Threshold shift**: Δρ_c = ρ_c(τ=0.99) - ρ_c(τ=0.85) for each criterion
- **Sensitivity ranking**: Which criterion is most sensitive to τ?
- **Slope changes**: Does τ only shift curves or also change shapes?

### From Scaling Analysis
- **Power law exponents**: β values for ρ_c ∝ d^(-β)
- **Universality test**: Do canonical and GEO2 have same β?
- **Transition sharpening**: Does Δρ → 0 as d → ∞?

### From Log-Scale Plot
- **Decay constants**: α values from slopes on log scale
- **Exponential regime**: Where does exp(-αρ) fit break down?

---

## Troubleshooting

### If computation is too slow
- Reduce trials: 500 → 200 (error bars 1.6× larger, but faster)
- Reduce dimensions: [10,12,14] → [10,14] (fewer points)
- Coarsen rho_step: 0.02 → 0.03 (fewer density points)

### If plots look cluttered
- Use separate-by-criterion plots (3 clean plots instead of 1 busy plot)
- Adjust tau_alphas to make curves more distinct
- Use fewer tau values: [0.85,0.90,0.95,0.99] → [0.90,0.95,0.99]

### If data files missing
- Check `data/production_runs/` directory exists
- Verify pickle.dump() succeeded (check file size > 0)
- Fallback: Re-run computation without --data-file flag

---

## References

**Smoothing experiment**: See `SMOOTHING_EXPERIMENT.md` for details on:
- K>=2 constraint (limits rho_step)
- Statistical noise analysis
- GEO2 d=64 computational limits

**Production runs**: See `PRODUCTION_SUMMARY.md` for:
- Completed runs and timings
- Output files and sizes
- Lessons learned

**Codebase guide**: See `CLAUDE.md` for:
- File structure
- Command usage
- Development workflow

---

**Last updated**: 2025-11-25 17:45 CET
**Next update**: After τ-comparison completes (~23:00 CET)
