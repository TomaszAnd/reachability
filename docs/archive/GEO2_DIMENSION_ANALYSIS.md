# GEO2 Higher Dimensions: Feasibility Analysis

**Date**: 2025-11-25
**Status**: 1×5 lattice (d=32) is SUPPORTED

---

## Executive Summary

**Question**: Can we run GEO2 for d=32 to fill the gap between d=16 and d=64?

**Answer**: ✅ **YES - GEO2 1×5 lattice (d=32) is supported!**

### Key Findings

1. **1×5 Lattice Structure**
   - Linear chain of 5 qubits
   - d = 2^5 = 32 ✓
   - 51 operators (L = 3n + 9|E| = 15 + 36)
   - 4 edges: (0,1), (1,2), (2,3), (3,4)

2. **Runtime Estimate**
   - Based on d=16 benchmark: 25 minutes for trials=400
   - Estimated for d=32: **2-4 hours** for trials=300
   - Scaling: d² factor in optimization (32²/16² = 4×)

3. **Recommendation**: **PROCEED with reduced parameters**
   - Priority: Lower than τ-comparison (more physics insight)
   - Suggested approach: Option B (Enhanced d=16) or Option C (1×5 with reduced trials)

---

## Technical Verification

### Code Inspection

From `reach/models.py`, the `GeometricTwoLocal` class:

```python
def __init__(self, nx: int, ny: int, periodic: bool = False, backend: str = "sparse"):
    if nx < 1 or ny < 1:
        raise ValueError(f"Lattice dimensions must be ≥ 1, got nx={nx}, ny={ny}")

    self.n_sites = nx * ny
    self.dim = 2 ** self.n_sites  # d = 2^(nx×ny)
```

**Validation**: ✅ nx=1, ny=5 passes all checks

###  Direct Test

```python
from reach import models

# Test 1×5 lattice creation
geo2 = models.GeometricTwoLocal(nx=1, ny=5, periodic=False, backend='sparse')

# Results:
# ✓ Lattice created successfully
# ✓ Sites: 5
# ✓ Dimension: 32
# ✓ Operators: 51
# ✓ Hamiltonian generation works
```

**Result**: ✅ Full support confirmed

---

## Geometry Comparison

| Lattice | nx×ny | Sites | d | Edges | Operators L | Geometry |
|---------|-------|-------|---|-------|-------------|----------|
| **2×2** | 2×2 | 4 | 16 | 4 (open) / 8 (periodic) | 48 / 84 | 2D grid |
| **1×5** | 1×5 | 5 | 32 | 4 | 51 | 1D chain |
| **2×3** | 2×3 | 6 | 64 | 7 (open) | 81 | 2D grid |

**Scientific question**: Does 1D chain geometry differ from 2D grid at the same dimension?
- 1×4 (d=16) vs 2×2 (d=16) - Geometric effect test

---

## Runtime Analysis

### Benchmark Data

From GEO2 2×2 (d=16) completed run:
- **trials=400**: 25 minutes total
- **Per K value**: ~2-3 minutes (trials=400)

### Scaling Estimate for 1×5 (d=32)

**Optimization complexity**: O(d²) per Hamiltonian
- d=16 → d=32: (32/16)² = 4× slower per trial
- trials=300 vs trials=400: 0.75× faster

**Net scaling**: 4× × 0.75 = 3×

**Estimated runtime**: 25 min × 3 = **75 minutes** (1.25 hours)

### For Density Sweep

Assuming ρ ∈ [0, 0.05], step 0.004:
- ρ values: [0.004, 0.008, 0.012, ..., 0.05]
- Number of points: 13 points
- Unique K values: ~10-12 (some rho map to same K)

**Total runtime estimate**: 10 K values × (75 min / 10) = **2-3 hours**

**Verdict**: ✅ Feasible within reasonable timeframe

---

## Execution Options

### Option A: 1×5 Production Run (RECOMMENDED IF TIME PERMITS)

```bash
# Create custom script or use CLI directly
python -m reach.cli --nx 1 --ny 5 --ensemble GEO2 \
    three-criteria-vs-K-multi-tau \
    --ensemble GEO2 -d 32 \
    --k-max 15 \
    --taus 0.95 \
    --trials 300 \
    --y unreachable
```

**Parameters**:
- trials=300 (balanced quality/speed)
- K range: 2-15 (covers ρ ≈ 0-0.015)
- Single tau=0.95 for speed

**Runtime**: ~2-3 hours
**Value**: Fills d=16-64 gap, tests 1D vs 2D geometry

### Option B: Enhanced GEO2 d=16 (GOLD STANDARD)

```bash
python scripts/generate_geo2_publication.py \
    --config 2x2 \
    --trials 800 \      # 2× current quality
    --rho-step 0.002 \  # 2× finer resolution
    --tau 0.95
```

**Runtime**: ~50 minutes
**Value**: Definitive reference with minimal error bars

### Option C: Both 1×5 and Enhanced d=16 (COMPREHENSIVE)

Run sequentially:
1. Enhanced d=16 (~50 min)
2. 1×5 production (~2-3 hours)

**Total**: ~3-4 hours
**Value**: Complete GEO2 dimension coverage with high quality

---

## Alternative Geometries to Explore

### 1×4 Lattice (d=16)

**Purpose**: Compare 1D chain vs 2D grid at same dimension

```bash
python -m reach.cli --nx 1 --ny 4 --ensemble GEO2 \
    three-criteria-vs-K-multi-tau \
    --ensemble GEO2 -d 16 \
    --k-max 10 \
    --taus 0.95 \
    --trials 300 \
    --y unreachable
```

**Runtime**: ~25 minutes
**Scientific value**: Tests if geometry matters (1D chain vs 2D grid)

### 2×3 Lattice with Reduced Parameters (d=64)

**Purpose**: Make d=64 feasible

```bash
python scripts/generate_geo2_publication.py \
    --config 2x3 \
    --trials 100 \      # Reduced from 300
    --rho-step 0.004 \  # Coarser
    --rho-max 0.03 \    # Truncate early
    --tau 0.95
```

**Runtime**: ~4-6 hours (vs days at trials=300)
**Trade-off**: Larger error bars but still publication-worthy

---

## Priority Ranking

### HIGH PRIORITY (Do First)
1. **τ-comparison sweep** (canonical, trials=200) - 2-3 hours
   - **Scientific value**: HIGHEST - reveals threshold sensitivity
   - **Computational cost**: Moderate
   - **Impact**: New physics insight per compute hour

### MEDIUM PRIORITY (If Time Permits)
2. **Enhanced GEO2 d=16** (trials=800) - 50 minutes
   - **Scientific value**: HIGH - gold standard reference
   - **Computational cost**: Low
   - **Impact**: Minimal error bars, publication-ready

3. **GEO2 1×5 (d=32)** - 2-3 hours
   - **Scientific value**: MEDIUM - fills dimensional gap
   - **Computational cost**: Moderate
   - **Impact**: Completes scaling analysis

### LOW PRIORITY (Optional)
4. **1×4 lattice** (geometry comparison) - 25 minutes
   - **Scientific value**: LOW-MEDIUM - tests geometric universality

5. **2×3 reduced** (d=64) - 4-6 hours
   - **Scientific value**: LOW - already know d=64 is expensive
   - **Better strategy**: Focus on lower dimensions with high quality

---

## Recommended Execution Plan

### Tonight (2025-11-25 Evening)

1. ✅ **DONE**: Highres canonical (trials=500) - Completed 15:49 CET
2. 🔄 **RUNNING**: Log-scale plot - Started 17:16, ETA ~19:30 CET (62% complete)
3. **START AFTER LOG-SCALE**: τ-comparison sweep (canonical, trials=200)
   - Priority: HIGHEST scientific value
   - Runtime: 2-3 hours
   - Expected completion: ~23:00-00:00 CET

```bash
# After log-scale completes (~19:30)
nohup python scripts/tau_comparison_sweep.py \
    --dims 10,12,14 \
    --taus 0.85,0.90,0.95,0.99 \
    --trials 200 \
    --output-dir fig/comparison \
    > logs/tau_comparison_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > logs/tau_comparison.pid
```

### Tomorrow (2025-11-26 Morning)

1. **Enhanced GEO2 d=16** (~50 min)
   ```bash
   python scripts/generate_geo2_publication.py \
       --config 2x2 --trials 800 --rho-step 0.002 --tau 0.95
   ```

2. **Optional: GEO2 1×5 (d=32)** (~2-3 hours)
   - Only if time permits and interest in geometric effects

3. **Implement data pipeline** (~1 hour coding)
   - Enable instant replotting from saved data

4. **Generate scaling analysis plot** (instant with pipeline)

---

## Expected Scientific Insights

### From 1×5 Lattice (d=32)

1. **Dimensional scaling**: ρ_c(d=32) bridges d=16 and d=64
2. **Power law confirmation**: Verify ρ_c ∝ d^(-β) holds for intermediate d
3. **Geometric universality**: Compare 1D chain (1×5) with 2D grids

### From Enhanced d=16

1. **Gold standard reference**: Minimal error bars (√2 smaller)
2. **Publication-quality**: Smoother curves, definitive results
3. **Benchmark**: Reference for all future GEO2 comparisons

### From 1×4 vs 2×2 Comparison

1. **Geometry effect**: Does 1D vs 2D matter at fixed d=16?
2. **Connectivity**: Linear chain (3 edges) vs grid (4 edges)
3. **Universality test**: Are reachability transitions geometry-independent?

---

## Why Not d=32 First?

**τ-comparison has higher priority** because:

1. **New physics**: Threshold sensitivity not yet explored
2. **Canonical basis**: More interesting (sparse vs dense operators)
3. **Multiple dimensions**: Tests scaling across d=10,12,14
4. **Publication impact**: Reveals fundamental threshold dependence
5. **Cost/benefit**: Same runtime (2-3 hours) but more scientific value

**GEO2 1×5 (d=32) is valuable** but:
- Incremental improvement (fills dimensional gap)
- Doesn't test fundamentally new physics
- Can be done later if needed

---

## Implementation Notes

### CLI Usage for Custom Lattices

The reach CLI supports arbitrary lattices via `--nx` and `--ny`:

```bash
# Global flags BEFORE subcommand
python -m reach.cli --nx 1 --ny 5 --ensemble GEO2 \
    <subcommand> \
    --ensemble GEO2 \  # Repeat ensemble flag for subcommand
    -d 32 \
    <other-flags>
```

**Known issue**: Some CLI argument parsing quirks with global vs subcommand flags.

**Workaround**: Use scripts/generate_geo2_publication.py and add '1x5' config option.

### Script Modification

To add 1×5 support to `generate_geo2_publication.py`:

```python
# In main():
CONFIGS = {
    '2x2': (2, 2, False),
    '2x3': (2, 3, False),
    '3x2': (3, 2, False),
    '1x5': (1, 5, False),  # ADD THIS
}
```

**Effort**: 5 minutes to add config option

---

## Conclusion

✅ **GEO2 1×5 lattice (d=32) IS FULLY SUPPORTED**

**Recommendation**:
1. **Tonight**: Focus on τ-comparison (highest scientific value)
2. **Tomorrow**: Enhanced GEO2 d=16 (50 min, gold standard)
3. **If time permits**: GEO2 1×5 (d=32) for dimensional bridge

**Estimated total compute time**:
- τ-comparison: 2-3 hours
- Enhanced d=16: 50 minutes
- Optional 1×5: 2-3 hours
- **Total**: 3.5-6.5 hours depending on choices

**Expected outputs**:
- 4 τ-comparison plots
- 1 gold-standard GEO2 d=16 plot
- Optional: 1 GEO2 d=32 plot
- Scaling analysis plot (from combined data)

---

**Last updated**: 2025-11-25 18:15 CET
**Next action**: Wait for log-scale plot completion (~19:30), then start τ-comparison
