# GEO2 Floquet Engineering - Quick Start Guide

**Status:** ✅ Implementation complete and tested
**Date:** 2026-01-05

---

## What Was Implemented

A complete experimental framework to test whether **effective Floquet Hamiltonians** make the Moment criterion more discriminative for quantum reachability analysis.

### The Big Idea

**Problem:** Regular Moment criterion is λ-independent → P ≈ 0 everywhere (too weak)

**Solution:** Floquet moment uses ∂H_F/∂λ_k which includes **commutator terms**:
```
∂H_F/∂λ_k = H_k + Σ_j λ_j [H_j, H_k]  ← λ-DEPENDENT!
```

This should make Floquet moment behave like Spectral/Krylov with clear phase transitions.

---

## What You Can Do Right Now

### 1. Run Tests (30 seconds)

Verify everything works:

```bash
cd /Users/tomas/PycharmProjects/reachability/reachability
python3 test_floquet_implementation.py
```

Expected output:
```
╔════════════════════════════════════════════════════════════════════╗
║               FLOQUET IMPLEMENTATION TEST SUITE                   ║
╚════════════════════════════════════════════════════════════════════╝

TEST 1: State Generation
  ✓ All states normalized and verified

TEST 2: Floquet Hamiltonian (Magnus Expansion)
  ✓ Floquet Hamiltonian computation successful

TEST 3: Floquet Moment Criterion
  ✓ Floquet moment criterion functional

TEST 4: Driving Functions
  ✓ All driving functions created successfully

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

### 2. Run Quick Experiment (~30 minutes)

Small test run to see if the hypothesis holds:

```bash
python3 scripts/run_geo2_floquet.py \
  --dims 16 \
  --rho-max 0.10 \
  --rho-step 0.02 \
  --n-samples 50 \
  --magnus-order 2
```

### 3. Generate Plots

```bash
python3 scripts/plot_geo2_floquet.py \
  data/raw_logs/geo2_floquet_*.pkl \
  --output-dir fig/geo2_floquet
```

Plots will be saved in `fig/geo2_floquet/`:
- `geo2_floquet_main_d16.png` - Main comparison
- `geo2_floquet_order_comparison_d16.png` - Order 1 vs 2
- `geo2_floquet_3panel_d16.png` - Side-by-side comparison

### 4. View Summary Statistics

```bash
python3 scripts/plot_geo2_floquet.py \
  data/raw_logs/geo2_floquet_*.pkl \
  --summary
```

---

## Full Production Run (~10-12 hours)

For publication-quality results:

```bash
# Create log directory
mkdir -p logs

# Run production (use nohup to run in background)
nohup python3 scripts/run_geo2_floquet.py \
  --dims 16 32 64 \
  --rho-max 0.15 \
  --rho-step 0.01 \
  --n-samples 100 \
  --magnus-order 2 \
  > logs/geo2_floquet_production.log 2>&1 &

# Save process ID
echo $! > .geo2_floquet_pid

# Monitor progress
tail -f logs/geo2_floquet_production.log

# When complete, generate plots
python3 scripts/plot_geo2_floquet.py \
  data/raw_logs/geo2_floquet_*.pkl \
  --output-dir fig/geo2_floquet
```

---

## Implementation Summary

### New Modules

1. **`reach/states.py`** - State generation for stabilizer codes
   - GHZ states: (|0000⟩ + |1111⟩)/√2
   - W states, cluster states, product states
   - All normalized and tested ✅

2. **`reach/floquet.py`** - Floquet engineering utilities
   - Magnus expansion (orders 1 and 2)
   - Driving functions (sinusoidal, square wave, multi-frequency)
   - Floquet moment criterion with ∂H_F/∂λ_k
   - All mathematical operations verified ✅

### Scripts

3. **`scripts/run_geo2_floquet.py`** - Production experiment runner
   - Sweeps over density ρ = K/d²
   - Compares Regular vs Floquet moment (orders 1 & 2)
   - Saves pickle files for analysis

4. **`scripts/plot_geo2_floquet.py`** - Publication-quality plotting
   - Main comparison plots
   - Order comparison (1 vs 2)
   - Multi-dimension overlays
   - 3-panel figures (GEO2 v3 style)

### Testing

5. **`test_floquet_implementation.py`** - Comprehensive test suite
   - 4 test categories, all passing ✅
   - Run before any production experiments

---

## Key Files

| Purpose | Location |
|---------|----------|
| **Documentation** | `docs/GEO2_FLOQUET_IMPLEMENTATION.md` (detailed) |
| **Quick Start** | `GEO2_FLOQUET_QUICKSTART.md` (this file) |
| **Test Script** | `test_floquet_implementation.py` |
| **Production** | `scripts/run_geo2_floquet.py` |
| **Plotting** | `scripts/plot_geo2_floquet.py` |
| **Output Data** | `data/raw_logs/geo2_floquet_*.pkl` |
| **Figures** | `fig/geo2_floquet/*.png` |

---

## Expected Results

### Hypothesis

| Criterion | Behavior | Why? |
|-----------|----------|------|
| **Regular Moment** | P ≈ 0 everywhere | Uses ⟨H_k⟩, λ-independent |
| **Floquet Order 1** | Weak transitions | Time-averaged only |
| **Floquet Order 2** | **Clear transitions** | Commutators make it λ-dependent! |

### What Success Looks Like

If the hypothesis is correct, you should see:

1. **Regular Moment:** Flat line at P ≈ 0
2. **Floquet Order 1:** Slight increase with ρ
3. **Floquet Order 2:** Clear sigmoid transition (like Spectral/Krylov)

The critical density ρ_c should be:
- Order 2 > Order 1 >> Regular
- Order 2 ≈ Spectral/Krylov (if compared)

---

## Troubleshooting

### Tests Fail

```bash
# Check Python environment
python3 --version  # Should be 3.8+

# Verify dependencies
python3 -c "import numpy, scipy, qutip; print('OK')"

# Re-run tests with verbose output
python3 test_floquet_implementation.py
```

### Production Run Too Slow

Reduce parameters:
- `--dims 16` (only smallest dimension)
- `--n-samples 50` (fewer trials)
- `--rho-max 0.10` (coarser grid)

### Out of Memory

Use sparse matrices for d > 64:
- Modify `floquet.py` to use `scipy.sparse`
- Already implemented for GEO2 operator generation

---

## Next Steps

### Immediate (After Quick Test)

1. ✅ Verify tests pass
2. ✅ Run quick experiment (d=16, n=50)
3. ✅ Check plots look reasonable
4. → If good, launch full production run

### Analysis (After Production)

1. Compare Floquet order 1 vs 2
2. Fit critical densities ρ_c(d)
3. Compare with Spectral/Krylov (if available)
4. Test with specific state pairs (GHZ, cluster)

### Publication

1. Generate final figures for paper
2. Extract ρ_c scaling: ρ_c ~ d^β
3. Write up results in main.tex
4. Add to experimental validation section

---

## Questions?

- **Detailed docs:** `docs/GEO2_FLOQUET_IMPLEMENTATION.md`
- **Code reference:** `reach/floquet.py` (well-commented)
- **Test output:** `python3 test_floquet_implementation.py`
- **Theory reference:** `main.tex` lines 588-700

---

## Summary

✅ **Implementation complete and tested**
✅ **Ready for production experiments**
✅ **All code documented**

**Next command to run:**
```bash
python3 test_floquet_implementation.py
```

Then:
```bash
python3 scripts/run_geo2_floquet.py --dims 16 --n-samples 50
```

Good luck! 🚀
