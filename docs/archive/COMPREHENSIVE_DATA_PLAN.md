# Comprehensive Data Generation Plan

## Status: Ready to Execute

### Created Scripts
- ✅ `scripts/plot_config.py` - Consistent colors and styling
- ⏳ `scripts/generate_comprehensive_data_TEST.py` - 10-minute test version
- ⏳ `scripts/generate_comprehensive_data.py` - Full production (12-24 hours)
- ⏳ `scripts/analyze_moment_alpha_scaling.py` - α(d) analysis
- ⏳ `scripts/analyze_krylov_b_scaling.py` - b(d) analysis
- ⏳ `scripts/plot_three_criteria_overlay.py` - Overlay plot
- ⏳ `scripts/plot_krylov_spectral_correlation.py` - Correlation plot

### Execution Options

#### Option A: Quick Test (10 minutes)
```bash
python scripts/generate_comprehensive_data_TEST.py
# Dimensions: [10, 14]
# Tau: [0.95, 0.99]
# Trials: 30/20/20 (moment/spectral/krylov)
```

#### Option B: Full Production (12-24 hours)
```bash
nohup python scripts/generate_comprehensive_data.py > logs/generation_stdout.log 2>&1 &
# Dimensions: [10, 14, 18, 22, 26]
# Tau: [0.78, 0.82, 0.86, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
# Trials: 200/100/100 (moment/spectral/krylov)
```

### Expected Outcomes

**Data Quality**:
- Moment K_c ≈ 0.5-1.5 (exponential onset, verified)
- Spectral/Krylov smooth Fermi-Dirac transitions
- Enhanced logging: max_overlap, max_krylov_score, convergence

**Analysis**:
- α(d) scaling determined (α·d = const?)
- b(d) scaling determined (linear vs power-law)
- Correlation R² > 0.7 for Krylov vs Spectral

**Publication Figures**:
- Consistent colors across all plots
- All error bars visible
- 200+ DPI for print quality
