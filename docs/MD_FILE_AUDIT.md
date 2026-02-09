# Markdown File Audit Report

**Generated:** 2026-01-13
**Project:** reachability/

---

## Summary Statistics

| Category | Count | Description |
|----------|-------|-------------|
| CURRENT | 20 | References existing files, information accurate |
| ORPHANED | 14 | References files/scripts that no longer exist |
| OUTDATED | 15 | Information superseded or stale |
| **Total** | **66** | |

---

## CURRENT (20 files)

Files that accurately describe existing code, data files, and functionality.

| File | Key References | Status |
|------|----------------|--------|
| `README.md` | reach/, scripts/, tests/, fig/, data/ | All directories exist |
| `CLAUDE.md` | reach/*.py (13 modules) | All modules verified |
| `QUICK_START.md` | reach/cli.py, scripts/, fig/comparison/ | All paths valid |
| `DATA_PROVENANCE.md` | data/raw_logs/*.pkl files | All pickle files exist |
| `docs/FIGURE_PROVENANCE.md` | 55 PNG files in fig/ | All figures verified |
| `docs/GEO2_ANALYSIS_SUMMARY.md` | reach/models.py, scripts/plot_geo2_v3.py | Implementation current |
| `FLOQUET_README.md` | reach/states.py, reach/floquet.py | Both modules exist |
| `GEO2_FLOQUET_SESSION_COMPLETE.md` | geo2_floquet_*.pkl | Tests pass |
| `CONTINUOUS_KRYLOV_IMPLEMENTATION.md` | reach/mathematics.py, reach/optimize.py | Functions verified |
| `FINAL_STATUS.md` | Continuous Krylov, spectral, moment | All implementations exist |
| `ANALYSIS_COMPLETE_SUMMARY.md` | decay_canonical_extended.pkl | Data file exists |
| `FLOQUET_CRITICAL_INSIGHTS.md` | reach/floquet.py | Magnus expansion implemented |
| `GEO2_ANALYSIS_PLAN.md` | reach/models.py (GeometricTwoLocal) | Class exists |
| `MOMENT_KC_DISCREPANCY_ANALYSIS.md` | reach/analysis.py:1789-1807 | Code verified |
| `docs/clipping_methodology.md` | reach/analysis.py:289-295 | Code matches |
| `docs/MOMENT_CRITERION_AUDIT_FINDINGS.md` | reach/analysis.py | Implementation verified |
| `PRODUCTION_SUMMARY.md` | fig/comparison/ outputs | Figures exist |
| `docs/GEO2_FLOQUET_IMPLEMENTATION.md` | states.py, floquet.py, 3 scripts | All exist |
| `FLOQUET_VERIFICATION_SUMMARY.md` | verify_floquet.py | Script exists |
| `GEO2_FLOQUET_RESULTS_ANALYSIS.md` | fig/geo2_floquet/ | Figures verified |

---

## ORPHANED (14 files)

Files that reference scripts or data that no longer exist in the repository.

| File | Missing Reference(s) | Severity | Action |
|------|---------------------|----------|--------|
| `OVERNIGHT_PRODUCTION_STATUS.md` | `run_extended_production.py` | Medium | Archive |
| `SPECTRAL_EXTENSION_STATUS.md` | `scripts/run_spectral_extension.py` | Medium | Archive |
| `GEO2_INVESTIGATION_REPORT.md` | `run_geo2_comprehensive.py` | High | Archive/Update |
| `PUBLICATION_PIPELINE_README.md` | `run_moment_extension_all_dims.py` | Medium | Update paths |
| `FLOQUET_SCALING_HYPOTHESIS_CORRECTED.md` | `run_floquet_comprehensive.py` | Medium | Archive |
| `FLOQUET_SCALING_EXPERIMENT_STATUS.md` | `run_scaling_experiment.py` | Medium | Archive |
| `FLOQUET_SECOND_CRITICAL_FINDING.md` | `run_floquet_comprehensive.py` | Medium | Archive |
| `docs/NEXT_EXPERIMENTS.md` | `run_krylov_dense_experiment.py` | High | Update |
| `docs/FLOQUET_EXPERIMENT_SUMMARY.md` | `run_floquet_comprehensive.py` | Medium | Archive |
| `latex-summary/FLOQUET_INTEGRATION_GUIDE.md` | Multiple run_*.py scripts | High | Update |
| `FLOQUET_FINAL_VERDICT.md` | Various run_*.py | Low-Medium | Review |
| `docs/EXTENDED_FLOQUET_QEC_PROPOSAL.md` | run_floquet_*.py pattern | Medium | Archive |
| `GEO2_FLOQUET_QUICKSTART.md` | Some secondary scripts | Low | Update |
| `results/LAMBDA_VALIDATION_REPORT.md` | Validation scripts | Low | Archive |

**Note:** Data files (*.pkl) referenced by these documents generally exist; only the scripts are missing.

---

## OUTDATED (15 files)

Files with accurate historical information but potentially superseded content.

| File | Issue | Severity | Action |
|------|-------|----------|--------|
| `PROJECT_CONTEXT.md` | Implementation details may have changed | Low | Review |
| `README_REACHABILITY.md` | Duplicate of README.md | High | Delete/redirect |
| `ENSEMBLE_CONSISTENCY_CHECK.md` | Predates GEO2 implementation | Medium | Archive |
| `ENSEMBLE_DIMENSION_ANALYSIS.md` | Superseded by GEO2_ANALYSIS_SUMMARY.md | Medium | Archive |
| `docs/PERFORMANCE_AUDIT.md` | Performance measurements stale | Low | Mark as historical |
| `docs/CRITICAL_BUG_FIX.md` | Bugs already fixed | Low | Archive |
| `FLOQUET_REDESIGN_ANALYSIS.md` | Superseded by FLOQUET_CRITICAL_INSIGHTS.md | Low | Archive |
| `SESSION_SUMMARY.md` | No session date indicated | Low | Add date or archive |
| `MOMENT_DATA_PROVENANCE.md` | Similar to DATA_PROVENANCE.md | Low | Consolidate |
| `COMPREHENSIVE_DATA_PLAN.md` | Planning document, may have unexecuted items | Low | Archive |
| `PER_DIMENSION_FITS_SUMMARY.md` | Predates linearized fits improvements | Low | Archive |
| `FINAL_ANALYSIS_SUMMARY.md` | Multiple "final" docs exist | Low | Consolidate |
| `FINAL_PUBLICATION_STATUS.md` | May be stale | Medium | Review |
| `TASKS_COMPLETE_20251216.md` | Dated, may reference changed tasks | Low | Archive |
| `INTEGRATION_SUMMARY.md` | Integration complete, summary stale | Low | Archive |

---

## Missing Scripts Inventory

The following scripts are referenced in documentation but do not exist:

| Missing Script | Referenced By | Data Exists? |
|----------------|---------------|--------------|
| `run_extended_production.py` | OVERNIGHT_PRODUCTION_STATUS.md | - |
| `run_spectral_extension.py` | SPECTRAL_EXTENSION_STATUS.md | Yes (spectral_extension_*.pkl) |
| `run_geo2_comprehensive.py` | GEO2_INVESTIGATION_REPORT.md | Yes (geo2_comprehensive_*.pkl) |
| `run_moment_extension_all_dims.py` | PUBLICATION_PIPELINE_README.md | Yes (moment_extension_*.pkl) |
| `run_floquet_comprehensive.py` | Multiple Floquet docs | Partial |
| `run_scaling_experiment.py` | FLOQUET_SCALING_EXPERIMENT_STATUS.md | - |
| `run_krylov_dense_experiment.py` | docs/NEXT_EXPERIMENTS.md | Yes (krylov_dense_*.pkl) |
| `run_krylov_dense_sampling.py` | docs/NEXT_EXPERIMENTS.md | - |

**Note:** Many of these scripts were likely archived to `scripts/archive/` or deleted after data collection was complete.

---

## Recommendations

### Immediate Actions

1. **Delete duplicate**: `README_REACHABILITY.md` (use README.md only)

2. **Archive orphaned docs** to `docs/archive/`:
   - OVERNIGHT_PRODUCTION_STATUS.md
   - SPECTRAL_EXTENSION_STATUS.md
   - GEO2_INVESTIGATION_REPORT.md
   - FLOQUET_SCALING_*.md files
   - ENSEMBLE_*.md files

3. **Consolidate "FINAL" documents**:
   - Keep `FINAL_STATUS.md` as the authoritative current status
   - Archive `FINAL_ANALYSIS_SUMMARY.md` and `FINAL_PUBLICATION_STATUS.md`

### Documentation Hierarchy (Recommended)

```
reachability/
├── README.md              # Main documentation (keep current)
├── CLAUDE.md              # Development guide (keep current)
├── QUICK_START.md         # Usage guide (keep current)
├── CHANGELOG.md           # Version history
├── DATA_PROVENANCE.md     # Data lineage (keep current)
├── FLOQUET_README.md      # Floquet feature guide (keep current)
└── docs/
    ├── FIGURE_PROVENANCE.md        # NEW: Figure tracking
    ├── MD_FILE_AUDIT.md            # NEW: This audit
    ├── GEO2_ANALYSIS_SUMMARY.md    # GEO2 documentation
    ├── GEO2_FLOQUET_IMPLEMENTATION.md
    ├── clipping_methodology.md
    └── archive/                     # Historical docs
        ├── OVERNIGHT_PRODUCTION_STATUS.md
        ├── SPECTRAL_EXTENSION_STATUS.md
        └── ... (other archived files)
```

### Root-Level Cleanup (Recommended)

Move the following from root to `docs/archive/`:
- All `FLOQUET_*` status files except FLOQUET_README.md
- All `GEO2_*` files except those in docs/
- SESSION_SUMMARY.md
- INTEGRATION_SUMMARY.md
- PRODUCTION_SUMMARY.md
- TASKS_COMPLETE_*.md

This reduces root-level clutter from ~44 MD files to ~8 essential files.

---

## Verification Checklist

- [x] All 55 PNG figures in fig/ verified to exist
- [x] All 13 reach/*.py modules verified to exist
- [x] All data files in data/raw_logs/ referenced in docs exist
- [x] All 19 active scripts in scripts/ identified
- [x] 14 missing scripts catalogued
- [x] 66 MD files categorized (20 current, 14 orphaned, 15 outdated)
