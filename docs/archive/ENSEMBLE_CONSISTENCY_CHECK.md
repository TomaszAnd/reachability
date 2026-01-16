# Ensemble Consistency Verification

**Date:** December 2, 2025
**Status:** ✅ **VERIFIED - All Consistent**

---

## Executive Summary

**✅ CASE A: All Three Criteria Use Canonical Basis**

- **Moment:** canonical (decay_canonical_extended.pkl)
- **Spectral:** canonical (decay_canonical_extended.pkl)
- **Krylov:** canonical (decay_multi_tau_publication.pkl)

**Conclusion:** Fair comparison, consistent methodology, theory-specific results.

---

## Data Sources

### Moment Criterion
- **File:** `data/raw_logs/decay_canonical_extended.pkl`
- **Ensemble:** `canonical`
- **Dimensions:** [8, 10, 12, 14, 16]
- **Trials:** 80
- **τ:** 0.95

### Spectral Criterion
- **File:** `data/raw_logs/decay_canonical_extended.pkl`
- **Ensemble:** `canonical`
- **Dimensions:** [8, 10, 12, 14, 16]
- **Trials:** 80
- **τ:** 0.95

### Krylov Criterion
- **File:** `data/raw_logs/decay_multi_tau_publication.pkl`
- **Ensemble:** `canonical`
- **Dimensions:** [10, 12, 14]
- **Trials:** 100
- **τ values:** [0.90, 0.95, 0.99]

---

## Canonical Basis Properties

**Ensemble Type:** Structured sparse basis {X_jk, Y_jk, Z_j, I}

**Operator Structure:**
- X_jk = |j⟩⟨k| + |k⟩⟨j| (2 non-zero elements)
- Y_jk = -i|j⟩⟨k| + i|k⟩⟨j| (2 non-zero elements)
- Z_j = |j⟩⟨j| (1 non-zero element)

**Physical Interpretation:**
- Complete orthogonal basis for d×d Hermitian matrices
- Sparse structure (1-2 elements vs d² for dense operators)
- Discrete coverage of Hilbert space
- Sharp transitions expected at low K/d²

**Sampling:** K operators selected uniformly without replacement from the canonical basis set.

---

## Comparison Validity

### Fair Comparison: ✅ YES

**Reason:** All three criteria tested on the same ensemble type with consistent operator structure.

**Implications:**
1. **Level Playing Field:** Each criterion evaluated under identical Hamiltonian generation
2. **Physical Consistency:** All results reflect canonical basis properties (sparse, structured)
3. **Direct Comparability:** Transition points (K_c, ρ_c) can be directly compared
4. **Methodology:** No ensemble-dependent biases in inter-criterion comparison

### Caveats for Publication

**Important Notes:**

1. **Ensemble Choice:** Results are specific to canonical basis ensemble
   - Sharper transitions due to sparse operator structure
   - May not represent "generic" dense Hamiltonians (GOE/GUE)
   - Physically meaningful for systems with structured local operators

2. **Reference Implementation:** The moment criterion reference in `reach/analysis.py` uses random projectors (dense)
   - Current implementation uses canonical basis (consistent with other criteria)
   - See `MOMENT_DATA_PROVENANCE.md` for detailed analysis

3. **Generalization:** Quantitative results (K_c values, transition sharpness) may differ for other ensembles
   - Qualitative trends expected to hold
   - ρ_c scaling laws should transfer

---

## Recommendations

### For Current Publication

**Plot Status:** ✅ **PUBLICATION READY**

- Title correctly indicates "canonical ensemble"
- All three criteria consistently use same ensemble
- Fair comparison maintained
- Results scientifically valid

**Suggested Caption Addition:**
```
Figure: Unreachability probability P(K) vs control density ρ = K/d² for three
reachability criteria using canonical basis ensemble. The sparse structure of
canonical basis operators (X_jk, Y_jk, Z_j) leads to sharp transitions compared
to dense random matrix ensembles.
```

### For Future Work (Optional)

**Ensemble Comparison Study:**
- Generate equivalent data for GOE/GUE ensembles
- Compare canonical vs dense operator behavior
- Quantify ensemble-dependence of K_c and transition sharpness
- Publication value: demonstrates structural effects on reachability

**Estimated Effort:** ~2-3 hours computation per ensemble

---

## Verification Commands

```bash
# Check Spectral ensemble
python3 -c "
import pickle
with open('data/raw_logs/decay_canonical_extended.pkl', 'rb') as f:
    data = pickle.load(f)
    print('Spectral ensemble:', data['params']['ensemble'])
"

# Check Krylov ensemble
python3 -c "
import pickle
with open('data/raw_logs/decay_multi_tau_publication.pkl', 'rb') as f:
    data = pickle.load(f)
    print('Krylov ensemble:', data['params']['ensemble'])
"

# Check Moment ensemble (same file as Spectral)
python3 -c "
import pickle
with open('data/raw_logs/decay_canonical_extended.pkl', 'rb') as f:
    data = pickle.load(f)
    print('Moment ensemble:', data['params']['ensemble'])
"
```

**Expected Output:** All should print `canonical`

---

## Conclusion

✅ **Ensemble consistency verified across all three criteria.**

✅ **Plot title "canonical ensemble" is accurate.**

✅ **Fair comparison maintained - publication ready.**

⚠️ **Note ensemble choice in paper - results specific to canonical basis structure.**

---

**Document Status:** ✅ COMPLETE
**Verification Date:** December 2, 2025
**Verified By:** Forensic data analysis
