# Extended Floquet-QEC Experiment: From Detection to Design

**Date:** 2026-01-08
**Status:** Research Proposal
**Priority:** High - Transforms criterion from diagnostic to design tool

---

## Executive Summary

**Current Result:** K_c^floquet = 4 for [[5,1,3]] code
**Critique:** Nearly tautological—weight-4 stabilizers obviously need 4-body terms

**Proposed Extension:** Use optimal λ* from Floquet criterion to **design** actual driving protocols for QEC code preparation

**Scientific Impact:** Transforms moment criterion from "yes/no" diagnostic into constructive design tool connecting:
1. Reachability analysis (moment criterion)
2. Floquet engineering (Wei-Norman formalism)
3. QEC code preparation (practical quantum computing)

---

## Phase 1: Optimal λ* Extraction and Analysis ✅ IMPLEMENTED

### Objective
Extract and analyze optimal coupling vectors λ* that maximize discriminative power.

### Implementation
**File:** `scripts/extract_optimal_lambda.py` (completed)

**Key Functions:**
```python
def optimize_lambda_for_criterion(psi, phi, hams, driving_functions, period, order=2):
    """Find λ* maximizing min eigenvalue of Q_F + x L_F L_F^T."""

def analyze_lambda_star(lambda_star, hams, driving_functions, period, order=2):
    """Analyze which commutators [H_j, H_k] dominate in λ*."""
```

**Usage:**
```bash
# For 5-qubit code at K=4:
python3 scripts/extract_optimal_lambda.py \
  --target 5qubit \
  --K 4 \
  --order 2 \
  --method differential_evolution \
  --seed 42

# For random Haar state:
python3 scripts/extract_optimal_lambda.py \
  --target haar \
  --K 3 \
  --order 2 \
  --method differential_evolution \
  --seed 42
```

**Expected Outputs:**
- `results/lambda_star_5qubit_K4_o2_YYYYMMDD_HHMMSS.pkl`
- Analysis of:
  - Top operator weights (which H_k dominate?)
  - Top commutator contributions (which [H_j, H_k] matter most?)
  - λ* distribution statistics

**Scientific Questions:**
1. Is there structure in λ* relating to stabilizer generators?
2. Do certain commutator pairs consistently dominate?
3. How does λ* change with K and Magnus order?

---

## Phase 2: Magnus Order Classification of QEC Codes

### Objective
Classify codes by minimum Magnus order required for unreachability detection.

### Implementation Status
🔨 **PARTIAL** - Framework exists, needs extension for multiple codes

**Required:**
```python
def construct_steane_logical_zero():
    """Construct [[7,1,3]] Steane code |0_L⟩."""
    # 7 qubits, d=128, weight-3 stabilizers
    pass

def construct_shor_logical_zero():
    """Construct [[9,1,3]] Shor code |0_L⟩."""
    # 9 qubits, d=512, weight-2 stabilizers
    pass

def test_magnus_order_classification(codes_dict, magnus_orders=[1, 2, 3]):
    """
    For each code in {[[5,1,3]], [[7,1,3]], [[9,1,3]]}:
    For each Magnus order O in {1, 2, 3}:
        Find K_c using O-th order Floquet expansion

    Returns:
        classification: Dict mapping (code, order) → K_c
    """
    pass
```

**Expected Classification:**
| Code | Magnus-1 K_c | Magnus-2 K_c | Magnus-3 K_c | Interpretation |
|------|--------------|--------------|--------------|----------------|
| [[5,1,3]] | ∅ | 4 | ? | Requires 2nd-order (single commutators) |
| [[7,1,3]] | ∅ | ? | ? | TBD |
| [[9,1,3]] | ∅ | ? | ? | TBD |

**Scientific Insight:** This classification reveals which codes benefit most from Floquet engineering and guides experimental priorities.

---

## Phase 3: From λ* to Driving Protocol Design (Wei-Norman Formalism)

### Objective
Show that optimal λ* can inform actual Floquet driving protocol design.

### Theoretical Foundation

**From arXiv:2103.15923 (Floquet Engineering of Lie Algebraic Quantum Systems):**

The Wei-Norman ansatz for Floquet engineering:
```
f_k(t) = M_k1(t) · ṁ(t) + M_k2(t) · h_eff_k
```

where driving functions f_k(t) are determined by:
- Desired effective Hamiltonian H_F^eff
- Micro-motion gauge choice m(t)

**Key Connection:** Optimal λ* specifies the desired operator weighting in H_F^eff!

### Implementation Status
⚠️ **REQUIRES RESEARCH** - Wei-Norman inversion is mathematically sophisticated

**Challenges:**
1. For SU(2^n) systems, Wei-Norman requires solving differential equations for micro-motion parameters m_α(t)
2. Not all H_F^eff may be realizable with given Hamiltonian basis {H_k}
3. Numerical stability of differential equation solvers for high-dimensional systems

**Proposed Approach:**
```python
def design_driving_protocol_from_lambda_star(lambda_star, hams, target_state):
    """
    Use optimal λ* to construct driving protocol.

    Steps:
    1. Construct target H_F from λ*:
       H_F_target = Σ_k λ*_k H_k + Σ_{j<k} λ*_j λ*_k F_jk [H_j, H_k]

    2. Apply Wei-Norman formalism to find f_k(t):
       - Solve differential equations for micro-motion m_α(t)
       - Extract driving functions via M_k1(t) · ṁ(t)

    3. Verify via time evolution:
       - Simulate U(T) under H(t) = Σ_k λ*_k f_k(t) H_k
       - Compute fidelity F = |⟨0_L|ψ(nT)⟩|²
    """

    # Step 1: Construct target H_F from λ*
    H_F_target = construct_floquet_hamiltonian(lambda_star, hams, order=2)

    # Step 2: Wei-Norman inversion (REQUIRES IMPLEMENTATION)
    # See arXiv:2103.15923 Eqs. (10)-(15)
    # This is the mathematically challenging part
    driving_functions = wei_norman_inversion(H_F_target, hams)  # TODO

    # Step 3: Verification via time evolution
    fidelity = simulate_floquet_evolution(
        psi_initial=ground_state,
        psi_target=target_state,
        driving_functions=driving_functions,
        n_periods=100
    )

    return driving_functions, fidelity
```

**Required Literature Review:**
- arXiv:2103.15923 (Wei-Norman for Floquet)
- arXiv:2410.10467 (Perturbative Floquet Hamiltonian)
- Original Wei-Norman papers (1963)
- Numerical methods for Lie algebra differential equations

**Timeline Estimate:**
- Literature review: 1-2 weeks
- Implementation: 2-4 weeks
- Verification and debugging: 1-2 weeks
- **Total:** ~1.5-2 months

---

## Phase 4: Reachability Boundary Mapping

### Objective
Map the boundary between reachable and provably unreachable regimes as a function of K and driving parameters.

### Implementation Status
🔨 **STRAIGHTFORWARD** - Extension of existing code

```python
def map_reachability_boundary(psi, phi, hams, K_range, n_lambda_samples=1000):
    """
    For each K, estimate the fraction of λ vectors that prove unreachability.

    Returns:
        K_values: Operator counts tested
        P_unreachable: Fraction proving unreachable
        boundary_K: Estimated K where P transitions from 1 to 0
    """
    results = {}
    for K in K_range:
        successes = 0
        for _ in range(n_lambda_samples):
            lambda_vec = np.random.randn(K)
            if floquet_criterion_succeeds(psi, phi, hams[:K], lambda_vec, order=2):
                successes += 1
        results[K] = successes / n_lambda_samples

    # Find boundary via interpolation
    boundary_K = find_transition_K(results)
    return results, boundary_K
```

**Key Questions:**
1. Is the transition sharp or gradual?
2. How does the boundary depend on code structure (stabilizer weights)?
3. Does λ-optimization shift the boundary location?

---

## Phase 5: Comparison with Direct Fidelity Computation

### Objective
Validate Floquet criterion predictions against actual achievable fidelities.

### Implementation Status
🔨 **REQUIRES VARIATIONAL OPTIMIZATION** - New functionality needed

```python
def compare_criterion_with_fidelity(target_state, hams, K_values, n_periods=100):
    """
    For each K:
    1. Check if Floquet criterion proves unreachability
    2. Compute best achievable fidelity via variational optimization

    If criterion says unreachable but fidelity is high → false positive (bad)
    If criterion says inconclusive but fidelity is low → false negative (OK)
    """
    for K in K_values:
        # Criterion check
        unreachable_proven = floquet_criterion(psi, target_state, hams[:K])

        # Variational fidelity optimization
        # Find best driving protocol via gradient descent on fidelity
        best_fidelity, best_protocol = variational_floquet_preparation(
            psi_initial=ground_state,
            psi_target=target_state,
            hams=hams[:K],
            n_periods=n_periods
        )

        print(f"K={K}: Criterion={'UNREACHABLE' if unreachable_proven else 'INCONCLUSIVE'}, "
              f"Best fidelity={best_fidelity:.4f}")
```

**Expected Insight:** The criterion should correctly predict unreachability when best achievable fidelity is low, calibrating the criterion's "tightness."

---

## Implementation Priority

| Phase | Status | Priority | Difficulty | Timeline |
|-------|--------|----------|------------|----------|
| **1. λ* Extraction** | ✅ Complete | High | Medium | Done |
| **2. Magnus Classification** | 🔨 Partial | High | Low | 1-2 weeks |
| **4. Boundary Mapping** | 🔨 Design | Medium | Low | 1-2 weeks |
| **5. Fidelity Comparison** | 🔨 Design | Medium | Medium | 2-3 weeks |
| **3. Wei-Norman Design** | ⚠️ Research | Very High | **Very High** | 1.5-2 months |

**Recommended Order:**
1. **Phase 1** ✅ (done) → Run on 5-qubit code, analyze λ*
2. **Phase 2** → Implement Steane/Shor codes, classify by Magnus order
3. **Phase 4** → Map reachability boundaries for all codes
4. **Phase 5** → Validate with variational optimization
5. **Phase 3** → Wei-Norman inversion (parallel research track)

---

## Expected Publications

### Paper 1: λ* Analysis and Magnus Classification (Near-term)
**Target:** Physical Review A or Quantum
**Content:**
- Optimal λ* extraction algorithm
- Commutator dominance analysis
- Magnus order classification of [[5,1,3]], [[7,1,3]], [[9,1,3]]
- Reachability boundary mapping

**Timeline:** 2-3 months (Phases 1, 2, 4, 5)

### Paper 2: Wei-Norman Driving Protocol Design (Long-term)
**Target:** Physical Review X or Nature Physics
**Content:**
- Wei-Norman inversion algorithm for SU(2^n)
- λ*-derived driving protocols for QEC preparation
- Experimental verification (fidelity, robustness)
- Comparison with naive (random λ) protocols

**Timeline:** 6-9 months (Phase 3 completion)

---

## Key Insight: Why This Matters

**Current experiment answers:** "Does Floquet criterion detect unreachability for QEC codes?"
**Answer:** Yes, but trivially (K_c=4 for weight-4 stabilizers)

**Extended experiment answers:** "Can the Floquet moment criterion **guide the design** of optimal Floquet driving protocols for QEC code preparation?"
**Potential answer:** Yes, transforming the criterion from diagnostic to design tool

This would demonstrate that:
- Optimal λ* encodes the essential physics for code preparation
- Wei-Norman inversion converts λ* into explicit driving protocols
- These protocols achieve high-fidelity code state preparation
- The moment criterion provides constructive guidance, not just "yes/no"

---

## Next Steps (Immediate)

1. **Run λ* extraction on 5-qubit code:**
   ```bash
   python3 scripts/extract_optimal_lambda.py --target 5qubit --K 4 --order 2
   ```

2. **Analyze results:**
   - Which commutators dominate?
   - Any correlation with stabilizer generators?

3. **Implement Steane/Shor codes:**
   - Add to `scripts/run_5qubit_code_experiment.py`
   - Run Magnus order classification

4. **Literature review for Wei-Norman:**
   - arXiv:2103.15923 detailed study
   - Identify numerical methods for Lie algebra ODEs
   - Assess feasibility for SU(2^5) system

---

## References

1. **arXiv:2103.15923** - Floquet Engineering of Lie Algebraic Quantum Systems
2. **arXiv:2410.10467** - Perturbative framework for arbitrary Floquet Hamiltonian
3. **Laflamme et al. (1996)** - 5-qubit perfect code
4. **Steane (1996)** - [[7,1,3]] code
5. **Shor (1995)** - [[9,1,3]] code

---

**Last updated:** 2026-01-08
**Status:** Phase 1 complete, Phases 2-5 designed, Phase 3 requires research
