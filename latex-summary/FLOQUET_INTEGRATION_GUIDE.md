# Floquet Results Integration Guide

**Date:** 2026-01-08
**Purpose:** Quick reference for integrating Floquet experimental results into main.tex

---

## Files Created

### LaTeX Subsections
1. **`floquet_results.tex`** (17 KB, ~350 lines)
   - Main experimental results subsection
   - Contains 9 subsubsections with complete analysis
   - Publication-ready with table, figure, equations

2. **`floquet_qec_experiment.tex`** (9.3 KB, ~200 lines)
   - Future work: 5-qubit perfect code experiment
   - Optional appendix or future work section
   - Fully designed and ready to run

### Figures (in `../fig/floquet/`)
- `floquet_static_comparison.png` (260 KB) - **Main figure**
- `floquet_static_logscale.png` (252 KB)
- `floquet_improvement_ratio.png` (136 KB)
- `floquet_static_bars.png` (140 KB)

---

## Integration Instructions

### Step 1: Include in main.tex

Add after the Krylov section (or wherever appropriate):

```latex
% Floquet-Enhanced Moment Criterion
\input{floquet_results.tex}

% Optional: Future work on QEC codes
\input{floquet_qec_experiment.tex}
```

### Step 2: Verify References

The subsections use these references (ensure they're defined in main.tex):

**Internal references:**
- `\ref{main}` → Theorem 2 (static moment criterion)
- `\ref{def:floqmag}` → Floquet-Magnus expansion definition

**Citations:**
- `\cite{Goldman_2014}` → Floquet theory
- `\cite{Laflamme_1996}` → 5-qubit perfect code
- `\cite{Kolesnikow_2024}` → GKP code (for QEC section)

### Step 3: Check Figure Paths

Figures referenced with relative path from `latex-summary/`:
```latex
\includegraphics[width=0.85\textwidth]{fig/floquet/floquet_static_comparison.png}
```

If compiling from a different directory, adjust path accordingly.

### Step 4: Compile and Check

```bash
cd latex-summary
pdflatex main.tex
bibtex main  # If using bibliography
pdflatex main.tex
pdflatex main.tex

# Check:
# - Figure appears correctly
# - Table formatting is good
# - Cross-references resolve
# - Equations are numbered correctly
```

---

## Key Content Highlights

### Section Structure

```
3.X Floquet-Enhanced Moment Criterion (floquet_results.tex)
  3.X.1 Motivation
  3.X.2 Mathematical Formulation
    - Floquet-Magnus expansion
    - Derivative ∂H_F/∂λ_k
    - Matrix formulation Q_F + x L_F L_F^T ≻ 0
    - Critical distinction: λ-dependence
  3.X.3 Driving Function Selection
    - Bichromatic scheme
    - Fourier overlap F_jk
  3.X.4 Experimental Protocol
  3.X.5 Results
    - Table 1: Comparison data
    - Figure 1: Main comparison plot
    - Fitted parameters
  3.X.6 Physical Interpretation
    - Why Floquet is stronger
    - Intermediate regime analysis
    - Connection to hierarchy
  3.X.7 Formulation Clarifications
    - Scalar vs vector explained
  3.X.8 Implications for Quantum Control
  3.X.9 Next Steps: QEC Code Targets

3.Y Future Work: QEC Code Preparation (floquet_qec_experiment.tex)
  3.Y.1 Motivation
  3.Y.2 5-Qubit Perfect Code
  3.Y.3 Experimental Design
  3.Y.4 Predicted Outcomes
  3.Y.5 Physical Interpretation
  3.Y.6 Computational Cost
  3.Y.7 Extensions
  3.Y.8 Expected Impact
```

### Critical Issues Resolved

✅ **Scalar vs Vector Formulation**
- Section 3.X.2: Defines both forms
- Section 3.X.7: Explicitly clarifies the relationship
- L_F(λ) = λ^T **L**_F (scalar function vs coefficient vector)

✅ **Driving Function Selection**
- Section 3.X.3: Dedicated subsection
- Bichromatic scheme fully explained
- Fourier overlap F_jk derived
- Measured ||H_F^(2)||/||H_F^(1)|| ≈ 6%

✅ **λ-Dependence**
- Emphasized in Section 3.X.2 (Critical Distinction)
- Equation for ∂H_F/∂λ_k shows explicit λ-dependence
- Implementation note explains 100 λ searches per trial

✅ **Hierarchy Connection**
- Section 3.X.6.4: "Connection to Criterion Hierarchy"
- Explicit values: λ_static < λ_floquet < λ_spectral < λ_krylov
- 0.00276 < 0.00296 < 0.069 < 0.041

---

## Key Results Summary

### Quantitative Findings

| Metric | Static | Floquet | Ratio |
|--------|--------|---------|-------|
| λ (decay scale) | 0.00276 | 0.00296 | 1.07 |
| α (decay rate) | 1.417 | 1.320 | 0.93 |
| R² (fit quality) | 0.977 | 0.932 | — |
| Peak improvement | — | +217% at K=3 | — |
| Statistical sig. | — | p=0.008 at K=3 | — |

### Physical Interpretation

**Why Floquet wins:**
1. Commutators expand operator space (2-body → 3-body)
2. λ-optimization finds best drive configuration
3. Peak effect in intermediate regime (K=3-4)

**Why improvement is modest:**
- Global: 7% (λ ratio = 1.07)
- Point-wise: Up to +217% at K=3
- Concentrated in specific regime, not uniform

---

## Labels Created

For cross-referencing:

**Sections:**
- `\label{sec:floquet_results}` - Main subsection
- `\label{sec:future_floquet}` - QEC experiment

**Equations:**
- `\label{eq:floquet_second_order}` - H_F^(2) expression
- `\label{eq:floquet_derivative}` - ∂H_F/∂λ_k formula
- `\label{eq:floquet_criterion_matrix}` - Matrix test

**Tables & Figures:**
- `\label{tab:floquet_static_comparison}` - Results table
- `\label{fig:floquet_static_comparison}` - Main figure

---

## Common Issues & Solutions

### Issue: Figure doesn't appear
**Solution:** Check path is correct relative to main.tex location
```latex
% If main.tex is in latex-summary/
\includegraphics{fig/floquet/floquet_static_comparison.png}  % Correct

% If main.tex is in root/
\includegraphics{latex-summary/fig/floquet/floquet_static_comparison.png}  % Adjust
```

### Issue: References don't resolve
**Solution:** Ensure Theorem 2 has label `\label{main}`
```latex
\begin{theorem}
\label{main}
...
\end{theorem}
```

### Issue: Citations undefined
**Solution:** Add to bibliography:
```bibtex
@article{Goldman_2014,
  title={Periodically driven quantum systems: effective Hamiltonians and engineered gauge fields},
  author={Goldman, N. and Dalibard, J.},
  journal={Physical Review X},
  volume={4},
  pages={031027},
  year={2014}
}
```

### Issue: Equation numbering conflicts
**Solution:** LaTeX handles this automatically. If issues persist, check for duplicate `\label{}` commands.

---

## Testing Checklist

Before submitting:

- [ ] Main figure appears correctly
- [ ] Table is properly formatted
- [ ] All cross-references resolve (no "??" in PDF)
- [ ] Citations resolve (no "[?]" in PDF)
- [ ] Equations are numbered correctly
- [ ] No LaTeX errors or warnings
- [ ] Figure caption is complete
- [ ] Section numbers are correct

---

## Next Actions

### Immediate
1. Review `floquet_results.tex` for any needed adjustments
2. Include in main.tex
3. Compile and check

### Optional
1. Include `floquet_qec_experiment.tex` as future work
2. Run 5-qubit code experiment:
   ```bash
   python3 scripts/run_5qubit_code_experiment.py
   ```
3. Add QEC results if experiment confirms advantage

### Future Extensions
- System size scaling (d=16, 32, 64)
- Driving function optimization
- Third-order Magnus expansion
- Different QEC codes

---

## Contact & Support

**Documentation:**
- Main summary: `docs/FLOQUET_EXPERIMENT_SUMMARY.md`
- Next experiments: `docs/NEXT_EXPERIMENTS.md`
- LaTeX results: `latex-summary/floquet_results.tex`
- QEC experiment: `latex-summary/floquet_qec_experiment.tex`

**Scripts:**
- Plot generation: `scripts/generate_floquet_plots.py`
- 5-qubit experiment: `scripts/run_5qubit_code_experiment.py`
- Criterion implementation: `reach/moment_criteria.py`

**Data:**
- Static results: `results/scaling_static_20260107_160738.pkl`
- Floquet results: `results/scaling_floquet_o2_20260108_040340.pkl`

---

**Last updated:** 2026-01-08
**Status:** ✅ Ready for integration
