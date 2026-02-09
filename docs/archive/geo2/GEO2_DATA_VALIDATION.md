# GEO2 Data Validation Report

**Generated:** 2026-01-13
**Status:** VALID

---

## Executive Summary

All GEO2 data has been validated. The data generation pipeline correctly implements:
- Dimension handling: d = 2^(nx×ny) for qubit lattices
- Operator count: L = 3n + 9|E| for open boundary conditions
- Data completeness: All 18 data series present (3 dimensions × 3 criteria × 2 approaches)

**Recommendation:** VALID - No issues found

---

## Data File Information

| Property | Value |
|----------|-------|
| File | `data/raw_logs/geo2_production_complete_20251229_160541.pkl` |
| Size | 14,795 bytes (14.4 KB) |
| MD5 | `db29938a0fe422a76867c09a273cc5eb` |
| Timestamp | 2025-12-29 16:05:41 |
| Tau | 0.99 |

---

## Lattice Configurations

| Lattice | Sites (n) | Edges (\|E\|) | Dimension (d) | Operators (L) | ρ_max | K_max |
|---------|-----------|---------------|---------------|---------------|-------|-------|
| 2×2 | 4 | 4 | 16 | 48 | 0.15 | 38 |
| 1×5 | 5 | 4 | 32 | 51 | 0.12 | 122 |
| 2×3 | 6 | 7 | 64 | 81 | 0.10 | 409 |

---

## Dimension Verification

All dimensions correctly computed as d = 2^(nx×ny):

| Lattice | nx | ny | n = nx×ny | d = 2^n | Expected | Status |
|---------|----|----|-----------|---------|----------|--------|
| 2×2 | 2 | 2 | 4 | 16 | 16 | ✓ |
| 1×5 | 1 | 5 | 5 | 32 | 32 | ✓ |
| 2×3 | 2 | 3 | 6 | 64 | 64 | ✓ |

---

## Operator Count Verification

Formula: L = 3n + 9|E| where n = sites, |E| = edges (open boundary)

### 2×2 Lattice (d=16)
```
  0---1
  |   |
  2---3

Sites: n = 4
Edges: |E| = 4 (horizontal: 2, vertical: 2)
L = 3×4 + 9×4 = 12 + 36 = 48 operators ✓
```

### 1×5 Lattice (d=32)
```
0---1---2---3---4

Sites: n = 5
Edges: |E| = 4 (all horizontal)
L = 3×5 + 9×4 = 15 + 36 = 51 operators ✓
```

### 2×3 Lattice (d=64)
```
  0---1
  |   |
  2---3
  |   |
  4---5

Sites: n = 6
Edges: |E| = 7 (horizontal: 3, vertical: 4)
L = 3×6 + 9×7 = 18 + 63 = 81 operators ✓
```

---

## Data Completeness Check

### Fixed λ Approach

| Dimension | Moment | Spectral | Krylov |
|-----------|--------|----------|--------|
| d=16 | 15 points ✓ | 15 points ✓ | 15 points ✓ |
| d=32 | 12 points ✓ | 12 points ✓ | 12 points ✓ |
| d=64 | 10 points ✓ | 10 points ✓ | 10 points ✓ |

### Optimized λ Approach

| Dimension | Moment | Spectral | Krylov |
|-----------|--------|----------|--------|
| d=16 | 15 points ✓ | 15 points ✓ | 15 points ✓ |
| d=32 | 12 points ✓ | 12 points ✓ | 12 points ✓ |
| d=64 | 10 points ✓ | 10 points ✓ | 10 points ✓ |

**Total:** 18/18 data series complete

---

## Runtime Analysis

| Dimension | Fixed λ | Optimized λ | Optimization Overhead |
|-----------|---------|-------------|-----------------------|
| d=16 | 1.3 min | 53.9 min | 41× |
| d=32 | 5.1 min | 260.1 min | 51× |
| d=64 | 511.3 min | 1614.4 min | 3.2× |

**Note:** d=64 fixed took 8.5 hours due to eigendecomposition costs at this dimension.

**Total runtime:** ~41 hours

---

## Code Implementation Verification

### Dimension Handling (reach/models.py:232-233)
```python
self.n_sites = nx * ny
self.dim = 2 ** self.n_sites
```
**Status:** Correct - dimension is 2^(number of qubits)

### Operator Count Assertion (reach/models.py:239-245)
```python
edges = self._build_lattice_edges()
expected_L = 3 * self.n_sites + 9 * len(edges)
assert self.L == expected_L, ...
```
**Status:** Correct - built-in validation ensures L = 3n + 9|E|

### Boundary Condition Handling (reach/models.py:247-270)
The `_build_lattice_edges()` method correctly implements:
- Open boundary: edges only between adjacent sites within grid
- Periodic boundary: wraps edges at grid boundaries
**Status:** Correct implementation

---

## Data Structure Verification

Sample data structure for `(d=16, tau=0.99, 'spectral')`:
```python
{
    'K': array([...]),      # Number of Hamiltonians
    'rho': array([...]),    # K/d² density values
    'p': array([...]),      # P(unreachable) probability
    'err': array([...]),    # Standard error
    'mean_overlap': array([...]),  # Mean optimized overlap S*
    'sem_overlap': array([...])    # SEM of overlap
}
```

### ρ Value Verification

For d=16 with K sampling:
- K=3 → ρ = 3/256 ≈ 0.0117 ✓
- K=5 → ρ = 5/256 ≈ 0.0195 ✓
- K=8 → ρ = 8/256 = 0.0312 ✓

All ρ values correctly computed as K/d².

---

## Generating Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/run_geo2_production.py` | Data generation | ✓ Verified |
| `scripts/plot_geo2_v3.py` | Figure generation | ✓ Verified |

### Data Loading in plot_geo2_v3.py (line 73-78)
```python
def load_data():
    files = sorted(Path("data/raw_logs").glob("geo2_production_complete_*.pkl"))
    if not files:
        raise FileNotFoundError("No GEO2 data found")
    with open(files[-1], 'rb') as f:
        return pickle.load(f)
```
**Status:** Correctly loads the most recent production file

---

## Figures Generated

All 6 geo2 figures in `fig/geo2/` traced back to this data file:

| Figure | Generated From | Status |
|--------|---------------|--------|
| `geo2_main_v3.png` | Fixed + Optimized data | ✓ |
| `geo2_scaling_v3.png` | Optimized spectral fits | ✓ |
| `geo2_d16_summary_v3.png` | d=16 all criteria | ✓ |
| `geo2_d32_summary_v3.png` | d=32 all criteria | ✓ |
| `geo2_d64_summary_v3.png` | d=64 all criteria | ✓ |
| `geo2_linearized_v3.png` | All linearized fits | ✓ |

---

## Anomalies or Issues

**None found.** All data passes validation checks:

1. ✓ Dimensions match lattice sizes
2. ✓ Operator counts match formula L = 3n + 9|E|
3. ✓ All 18 data series complete
4. ✓ Data structure correct
5. ✓ ρ values correctly computed
6. ✓ Generating scripts verified

---

## Recommendations

| Category | Recommendation |
|----------|----------------|
| Data | **VALID** - No rerun needed |
| Scripts | Keep `scripts/run_geo2_production.py` for reproducibility |
| Figures | All figures correctly reflect underlying data |

---

## Regeneration Command

If data needs to be regenerated:
```bash
cd /Users/tomas/PycharmProjects/reachability/reachability
python scripts/run_geo2_production.py
```

**Expected runtime:** ~41 hours for full production quality
