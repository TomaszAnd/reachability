# PROJECT_CONTEXT.md - AI Assistant Context for Reachability Project

## Project Context
This project analyzes phase transitions in quantum state reachability.

**Main question**: Given K random Hamiltonians, can we reach a random target state |φ⟩ from an initial state |ψ⟩?

Three reachability criteria are analyzed:
- **Moment**: Based on moment constraints (always reachable)
- **Spectral**: S* ≥ τ (spectral overlap threshold)
- **Krylov**: R* ≥ τ (Krylov subspace reachability threshold)

## Current State (Dec 16, 2025)

### Completed Experiments
1. ✅ Comprehensive Moment reachability (all d, full transition)
2. ✅ Moment extension to ρ=0.2 (all d)
3. ✅ FIXED Krylov/Spectral (ground state bug corrected)
4. ✅ DENSE Krylov sampling around K_c (100 trials/K)
5. ✅ Multi-τ sweep for tau analysis (τ=0.8 to 0.99)
6. ✅ Publication figures v3 with merged data
7. 🔄 Spectral extension to ρ=0.2 (IN PROGRESS for d>10)

### Critical Bug Fixes Applied

#### 1. Ground State Initialization Bug
**Problem**: Used `random_states` for initial state in Krylov/Spectral
**Fix**: Changed to `fock_state(d, 0)` (ground state)
**Impact**: Fixed data in `krylov_spectral_canonical_20251215_154634.pkl`

#### 2. Data Merging Issue
**Problem**: v2 figures only used DENSE Krylov data (6-9 points, transition only)
**Fix**: Merge FIXED (full range) + DENSE (transition detail) with deduplication
**Impact**: v3 figures have 13-30 points covering P=0 → transition → P=1

### Publication Figures Status

**v3 (CURRENT - Dec 16, 2025)**:
- All 8 figures generated in `fig/publication/*_v3.png`
- Complete data coverage: ρ ∈ [0.02, 0.20]
- Merged Krylov data: FIXED + DENSE
- NEW: tau_d_comprehensive_analysis plot

**Known issues**:
- Spectral data for d>10 doesn't extend to ρ=0.2 (experiment in progress)
- Linearized fits missing error bars (can be added in v4)

## Data Pipeline

```
RAW DATA SOURCES:
├── comprehensive_reachability_20251209_153938.pkl
│   └── OLD Moment data (transition region)
├── moment_extension_all_dims_20251215_160333.pkl
│   └── NEW Moment extension (high ρ)
├── krylov_spectral_canonical_20251215_154634.pkl
│   └── FIXED Krylov/Spectral (ground state corrected, full K range)
├── krylov_dense_20251216_112335.pkl
│   └── DENSE Krylov (100 trials/K around K_c)
└── decay_tau_sweep_comprehensive.pkl
    └── Multi-tau data (10 tau values, d=10,12,14)

MERGING STRATEGY:
├── Moment: OLD comprehensive + NEW extension
│   └── Result: 41-134 points per dimension
├── Spectral: FIXED data only
│   └── Result: 12-29 points per dimension
└── Krylov: FIXED + DENSE (deduplicated, sorted by K)
    └── Result: 13-30 points per dimension

OUTPUT:
└── fig/publication/*_v3.png (8 figures total)
```

## Important Files

### Core Library
- `reach/models.py` - State and Hamiltonian generation
- `reach/optimize.py` - Krylov/Spectral optimization functions
- `reach/mathematics.py` - Core mathematical operations

### Data Generation Scripts
- `scripts/run_moment_extension_all_dims.py` - Moment extension
- `scripts/run_krylov_spectral_canonical.py` - Krylov/Spectral fixed experiment
- `scripts/run_krylov_dense_sampling.py` - Dense Krylov around K_c
- `scripts/run_spectral_extension.py` - Spectral extension (to be launched)

### Analysis Scripts
- `scripts/generate_publication_figures_final.py` - Main figure generator (v1-v3)
- `scripts/generate_tau_analysis.py` - Tau dependence plots

## Coding Patterns

### Correct Imports
```python
from reach.models import fock_state, random_states, random_hamiltonian_ensemble
from reach.optimize import maximize_krylov_score, maximize_spectral_overlap
```

### Ground State Initialization (CRITICAL!)
```python
# CORRECT - Ground state
psi = fock_state(d, 0)

# WRONG - Random state (old bug)
# psi = random_states(n=1, dim=d, seed=seed)[0]  # DON'T USE THIS!
```

### Data Structure Access

**For krylov_spectral_canonical (FIXED data)**:
```python
with open('data/raw_logs/krylov_spectral_canonical_20251215_154634.pkl', 'rb') as f:
    data = pickle.load(f)

K = np.array(data['results'][d]['K'])
P_spectral = np.array(data['results'][d]['spectral']['P'])
sem_spectral = np.array(data['results'][d]['spectral']['sem'])
P_krylov = np.array(data['results'][d]['krylov']['P'])
sem_krylov = np.array(data['results'][d]['krylov']['sem'])
```

**For krylov_dense (DENSE data)**:
```python
with open('data/raw_logs/krylov_dense_20251216_112335.pkl', 'rb') as f:
    data = pickle.load(f)

K = data['krylov'][d]['K']
P = data['krylov'][d]['P']
sem = data['krylov'][d]['sem']
```

**For comprehensive_reachability (OLD Moment data)**:
```python
with open('data/raw_logs/comprehensive_reachability_20251209_153938.pkl', 'rb') as f:
    data = pickle.load(f)

# Access via results[d]['moment']
K = data['results'][d]['moment']['K']
P = data['results'][d]['moment']['P']
```

### Data Merging Pattern
```python
# Merge FIXED + DENSE Krylov data
K_list, P_list, sem_list = [], [], []

# 1. Add FIXED data (full range)
for i, k in enumerate(K_fixed):
    K_list.append(k)
    P_list.append(P_fixed[i])
    sem_list.append(sem_fixed[i])

# 2. Add DENSE data (avoid duplicates)
for i, k in enumerate(K_dense):
    if k not in K_list:
        K_list.append(k)
        P_list.append(P_dense[i])
        sem_list.append(sem_dense[i])

# 3. Sort by K
K_arr = np.array(K_list)
P_arr = np.array(P_list)
sem_arr = np.array(sem_list)
sort_idx = np.argsort(K_arr)

merged_data = {
    'K': K_arr[sort_idx],
    'P': P_arr[sort_idx],
    'sem': sem_arr[sort_idx],
    'rho': K_arr[sort_idx] / d**2
}
```

## Next Steps (Priority Order)

### HIGH PRIORITY
1. ✅ Complete v3 publication figures with merged data
2. 🔄 Run Spectral extension experiment for d=14,18,22,26
3. ⏳ Add error bars to linearized fits (v4 update)

### MEDIUM PRIORITY
4. ⏳ Final documentation cleanup
5. ⏳ Verify all data consistency
6. ⏳ Run validation tests on merged data

### LOW PRIORITY
7. ⏳ Generate supplementary figures
8. ⏳ Write analysis scripts for K_c extrapolation

## Common Pitfalls to Avoid

1. **Don't use random initial states for Krylov/Spectral** - Always use `fock_state(d, 0)`
2. **Don't forget to merge FIXED + DENSE Krylov data** - v2 only had DENSE (incomplete)
3. **Check for duplicate K values when merging** - Can cause fitting issues
4. **Always sort merged data by K** - Required for plotting and fitting
5. **Verify SEM is non-zero** - Replace zeros with small value (1e-3) for curve fitting
6. **Don't mix data versions** - Use consistent data files (FIXED, not OLD buggy ones)

## Key Results Summary

### Scaling Laws (τ=0.99)
- Moment: K_c = 0.40d + 0.3 (ρ_c decreases with d)
- Spectral: K_c = 1.95d - 5.9 (ρ_c ≈ constant)
- Krylov: K_c = 0.97d - 0.2 (ρ_c ≈ constant)

### Fit Quality
- Fermi-Dirac (2 params): Mean R² = 0.958
- Richards (3 params): Mean R² = 0.973
- Krylov with merged data: R² > 0.99

### Tau Dependence (d=10)
| τ | Krylov K_c | Krylov Δ | Spectral K_c | Spectral Δ |
|---|------------|----------|--------------|------------|
| 0.80 | 7.18 | 0.46 | 8.87 | 0.83 |
| 0.90 | 8.16 | 0.30 | 9.96 | 0.78 |
| 0.99 | 9.05 | 0.09 | 10.00 | 0.26 |

**Trend**: Higher τ → Higher K_c, Sharper transitions (smaller Δ)

## Monitoring Long-Running Experiments

```bash
# Check Spectral extension progress
tail -f logs/spectral_extension_*.log

# Check if process is running
ps -p $(cat .spectral_extension_pid)

# Monitor output file size
watch -n 30 'ls -lh data/raw_logs/spectral_extension_*.pkl'
```

## Quick Reference: File Locations

| Type | Location |
|------|----------|
| Raw data | `data/raw_logs/*.pkl` |
| Figures | `fig/publication/*.png` |
| Logs | `logs/*.log` |
| Scripts | `scripts/*.py` |
| Source code | `reach/*.py` |

Last updated: 2025-12-16
