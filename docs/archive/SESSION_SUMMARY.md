# Session Summary: Physical Methodology & Experimental Design
**Date**: 2025-12-15
**Status**: Documentation Complete, Experiments Ready to Launch

---

## ✅ Completed Tasks

### 1. Documentation Updates

**README.md** (lines 464-527): Added "Data Quality and Methodology" section
- Physical data treatment (Wilson intervals, no artificial clipping)
- Linearized fit methodology for all criteria
- Quality metrics: good (≥10 pts, ≥20%), marginal (≥5, ≥10%), insufficient
- Experimental design guidelines for Krylov dense sampling

**CLAUDE.md** (lines 269-302): Added "Visualization Standards (CRITICAL ⚠️)" section
- Clear WRONG vs CORRECT examples
- Explicit warning against `np.clip(P, epsilon, 1-epsilon)`
- Physical rationale: P=0 and P=1 are genuine binomial outcomes
- References to `docs/clipping_methodology.md` and `scripts/plot_linearized_physical.py`

### 2. Experimental Design Infrastructure

**scripts/design_krylov_dense_sampling.py** (197 lines):
- Estimates K_c and Δ from empirical scaling: K_c ≈ 0.9d + 1, Δ ≈ 0.06d
- Generates adaptive K grids with dense sampling (spacing ~ Δ/3) near K_c
- Provides cost estimates: ~0.1 hours total for all dimensions
- Predicts quality improvement: 3-4 → 12-19 transition points per dimension

**experimental_designs/krylov_dense_sampling.json** (304 lines):
- Complete K grids for d ∈ {10, 14, 18, 22, 26}
- 28-43 K values per dimension (vs 30 currently)
- Dense sampling in transition windows around estimated K_c

### 3. Centralized Plot Styling

**scripts/plot_config.py**: Enhanced from 54 to 215 lines
- **Anti-clipping documentation** in module header
- **Color schemes**: Dimensions, criteria, quality statuses
- **Typography**: Font sizes (title:18, axis:14, legend:10), weights
- **Figure sizes**: Single (10×7), two-panel (16×7), three-panel (22×7), four-panel (22×14)
- **Point styling**: Separate configs for boundary (faded) vs transition (prominent)
- **Line styling**: Fit lines, reference lines, grids
- **Export settings**: Standard (150 DPI) vs publication (300 DPI)
- **Quality thresholds** and helper functions: `classify_data_quality()`, `apply_axis_styling()`

### 4. Experiment Infrastructure

**scripts/monitor_experiments.py** (78 lines):
- Monitors running experiments via PID files
- Shows recent log output
- Lists generated data files with timestamps and sizes

---

## 📊 Key Results from Physical Analysis

From `logs/physical_linearized_analysis.log`:

| Criterion | N_transition | Frac % | R² | Quality |
|-----------|-------------|--------|----|---------|
| **Moment** | 33-36 | 89-97% | >0.95 | Excellent ✅ |
| **Spectral** | 14-18 | 47-60% | >0.85 | Good ✅ |
| **Krylov** | 2-4 | 7-13% | variable | Insufficient ⚠️ |

**Diagnosis**: Krylov has Δ ≈ 0.5-1.5 (very sharp transition), needs dense sampling with spacing ΔK ~ Δ/3 ≈ 0.2-0.3.

---

## ⚠️ Critical Issue: Ensemble Consistency

**ALL existing data uses CanonicalBasis**, not GUE:
- `comprehensive_reachability_20251209_153938.pkl` → CanonicalBasis
- All publication figures → CanonicalBasis
- Physical linearized analysis → CanonicalBasis

**Action Required**: Experiments MUST use CanonicalBasis for consistency.

---

## 🔧 Verified reach API

```python
# Ensemble
from reach.models import CanonicalBasis, random_states
import numpy as np

ensemble = CanonicalBasis(dim=10, include_identity=True)
rng = np.random.RandomState(seed)
hams = ensemble.sample_k_operators(k=5, rng=rng)  # List[qutip.Qobj]

# States
states = random_states(n=10, dim=10, seed=42)  # List[qutip.Qobj]
psi = states[0]  # Initial state
phi = states[1]  # Target state

# Krylov criterion
from reach.optimize import maximize_krylov_score

result = maximize_krylov_score(
    psi=psi,
    phi=phi,
    hams=hams,
    m=None,  # Auto: m = min(K, dim)
    bounds=None,  # Default: [(-1,1)] * K
    restarts=2,
    maxiter=200
)
# result = {'best_value': 0.95, 'best_x': [...], 'success': True, 'nfev': 100, 'runtime_s': 0.5, 'method': 'L-BFGS-B'}
# Unreachable if result['best_value'] < tau

# Spectral criterion
from reach.optimize import maximize_spectral_overlap

result = maximize_spectral_overlap(
    psi=psi,
    phi=phi,
    hams=hams,
    bounds=None,
    method='L-BFGS-B',
    restarts=2,
    maxiter=200
)
# result = {'best_value': S*, 'best_x': [...], 'success': True, ...}
# Unreachable if result['best_value'] < tau

# Moment criterion
# Uses Gram matrix definiteness check (see reach/analysis.py:1009-1125)
# No single-query function - use batch function moment_criterion_probabilities()
```

---

## 📝 Next Steps (User Action Required)

### Option 1: Use Existing CLI (Recommended - Fast)

The existing `reach.cli` already handles CanonicalBasis correctly. Use it with custom K grids:

```bash
# Load K grid from JSON
K_GRID=$(python3 -c "
import json
with open('experimental_designs/krylov_dense_sampling.json') as f:
    design = json.load(f)
K_values = ','.join(map(str, design['d10']['K_grid']))
print(K_values)
")

# Run dense Krylov sampling for d=10
python3 -m reach.cli three-criteria-vs-K-multi-tau \
  --ensemble CanonicalBasis \
  -d 10 \
  --k-values "$K_GRID" \
  --taus 0.99 \
  --trials 300 \
  --csv fig/comparison/krylov_dense_d10_canonical.csv \
  --flush-every 10 \
  --y unreachable
```

Repeat for d ∈ {14, 18, 22, 26}.

**Estimated time**: ~30 minutes per dimension (300 trials × 28-43 K values).

### Option 2: Create Custom Script

If you need more control, use the verified API above to create:
- `scripts/run_krylov_dense_canonical.py`
- `scripts/run_moment_extension_canonical.py`

See the API verified above for exact signatures.

### Option 3: Alternative Function Fitting (Immediate)

While experiments run, analyze existing Spectral data with alternative functional forms (Richards, Gompertz, etc.) to identify best fit for the "shoulder" region (P > 0.9).

This uses existing data - no new experiments needed.

---

## 📁 Files Created/Modified Today

1. ✅ `README.md` - Data Quality section added (lines 464-527)
2. ✅ `CLAUDE.md` - Visualization Standards added (lines 269-302)
3. ✅ `scripts/design_krylov_dense_sampling.py` - New (197 lines)
4. ✅ `experimental_designs/krylov_dense_sampling.json` - New (304 lines)
5. ✅ `scripts/plot_config.py` - Enhanced (54 → 215 lines)
6. ✅ `scripts/monitor_experiments.py` - New (78 lines)
7. ⚠️ `scripts/run_krylov_dense_experiment.py` - Created but needs CanonicalBasis fix
8. ⏸️ `scripts/run_moment_extension.py` - Not yet created
9. ⏸️ `scripts/fit_alternative_functions_physical.py` - Not yet created

---

## 🎯 Recommended Workflow

1. **Use existing CLI** (Option 1 above) for Krylov dense sampling - fastest path to results
2. **Monitor progress** with `python3 scripts/monitor_experiments.py`
3. **While waiting**, create and run alternative function fitting on existing Spectral data
4. **After completion**, regenerate `scripts/plot_linearized_physical.py` with new data
5. **Compare** old vs new Krylov quality metrics

---

## 🔍 Quality Assessment Framework

The new framework provides explicit quality metrics for all linearized fits:

```python
from scripts.plot_config import classify_data_quality

n_trans = 15  # Number of transition points
frac_trans = 0.45  # Fraction of data in transition

quality = classify_data_quality(n_trans, frac_trans)
# Returns: 'good', 'marginal', or 'insufficient'
```

**Thresholds**:
- Good: n_trans ≥ 10 AND frac_trans ≥ 0.20
- Marginal: n_trans ≥ 5 AND frac_trans ≥ 0.10
- Insufficient: Below marginal

---

## 📚 Key Philosophy

**"No artificial clipping. No arbitrary parameters. Just physics and statistics."**

- P=0 and P=1 are **genuine binomial outcomes** (k=0/N or k=N/N), not numerical errors
- Use **Wilson score intervals** for proper uncertainty bounds
- Fit **only transition regions** (0 < k < N) - boundaries provide no slope information
- Report **explicit quality metrics** (N_trans, frac_trans, quality flag)

See `docs/clipping_methodology.md` for full mathematical treatment.

---

## ✨ Session Complete!

All documentation and infrastructure is in place. The path forward is clear:
- Execute Option 1 (CLI) or Option 2 (custom scripts)
- Monitor with `scripts/monitor_experiments.py`
- Enjoy physically rigorous, publication-ready results! 🎉
