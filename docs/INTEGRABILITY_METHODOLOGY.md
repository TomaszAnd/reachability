# Integrability Study Methodology

**Date:** 2026-01-14
**Status:** Complete Documentation

This document provides detailed methodology for the integrability experiments that relate quantum chaos to criterion ordering.

---

## 1. Overview

### Research Question

**How does the integrability/chaos of Hamiltonian ensembles affect the relative performance of Spectral vs Krylov criteria?**

### Key Finding

- **Integrable systems**: Spectral criterion FAILS (cannot detect reachability)
- **Chaotic systems**: Both criteria work, with similar critical densities
- **Krylov is universal**: Works regardless of integrability level

---

## 2. State Choice Justification

### 2.1 Why Haar-Random States?

Our experiments use **Haar-uniform random states** for both initial $|\psi\rangle$ and target $|\phi\rangle$. This choice is well-motivated:

**1. Universality Principle**

Haar-random states represent "typical" quantum states in Hilbert space. For a $d$-dimensional system:
- The Haar measure is the unique unitarily-invariant probability measure on pure states
- Almost all states are "generic" (form a set of measure 1)
- Special states (e.g., product states, eigenstates) form a set of measure 0

**Mathematical definition:** $|\psi\rangle = U|0\rangle$ where $U \sim \text{Haar}(\mathcal{U}(d))$

**2. Unbiased Sampling**

Haar-random states avoid introducing bias toward any particular basis:
- **Computational basis states** would favor diagonal Hamiltonians
- **Eigenstates of test Hamiltonians** would trivially satisfy reachability
- **Product states** would introduce entanglement-dependent biases

By using Haar-random states, our results characterize the **average-case** behavior, not corner cases.

**3. Connection to Quantum Ergodic Hypothesis**

In chaotic quantum systems, eigenstates behave like random vectors. The Berry conjecture states that high-energy eigenstates of chaotic systems are statistically indistinguishable from Haar-random states.

This means our Haar-random initial/target states probe the same physics as:
- Thermal states at infinite temperature: $\rho \propto I$
- Time-averaged states under ergodic dynamics

**4. Scaling Properties**

Haar-random states have well-characterized statistics:
$$\mathbb{E}[|\langle n|\psi\rangle|^2] = \frac{1}{d}$$
$$\text{Var}[|\langle n|\psi\rangle|^2] = \frac{1}{d^2(d+1)}$$

The **Porter-Thomas distribution** governs component magnitudes:
$$P(|\langle n|\psi\rangle|^2 = x) = d \cdot e^{-dx}$$

This ensures reproducible, well-understood initial conditions.

### 2.2 Alternative State Choices (Not Used)

| State Type | Why Not Used |
|------------|--------------|
| **Computational basis** | Biased toward diagonal H; gives ⟨n\|ψ⟩ = δ_{n,k} |
| **Product states** | Only span tiny fraction of Hilbert space |
| **Coherent states** | Require oscillator structure not present here |
| **Eigenstates of H** | Trivially reachable/unreachable; not interesting |

### 2.3 Implementation

```python
from qutip import rand_ket

# Generate Haar-uniform random states
psi = rand_ket(d, seed=rng)  # Initial state
phi = rand_ket(d, seed=rng)  # Target state

# QuTiP's rand_ket uses: U|0⟩ where U is Haar-random
```

---

## 3. Hamiltonian Models

### 3.1 Integrable Ising Model

$$H = \sum_{i=1}^{n-1} J_i \sigma^z_i \sigma^z_{i+1} + \sum_{i=1}^{n} h_i \sigma^z_i$$

**Properties:**
- **Diagonal in computational basis**: All terms commute with each other
- **Eigenstates**: $|n\rangle = |b_1 b_2 \ldots b_n\rangle$ (computational basis states)
- **Level statistics**: Poisson ($\langle r \rangle \approx 0.39$)

**Implementation:**
```python
# Coefficients sampled randomly
J_i ~ N(0, 1)  # Nearest-neighbor coupling
h_i ~ N(0, 1)  # On-site field
```

**Why Spectral Fails:**
The eigenstates are FIXED regardless of λ values:
$$|n(\lambda)\rangle = |n\rangle \quad \text{(λ-independent)}$$

Therefore the spectral overlap is constant:
$$S(\lambda) = \sum_n |\langle n|\phi\rangle| |\langle n|\psi\rangle| = \text{constant}$$

λ-optimization cannot help → P(unreachable) = 1.0 always.

---

### 3.2 Near-Integrable Ising Model

$$H = J \sum_{i} \sigma^z_i \sigma^z_{i+1} + h \sum_{i} \sigma^z_i + g \sum_{i} \sigma^x_i$$

**Properties:**
- **Transverse field g breaks integrability**: Mixes computational basis states
- **Small g**: Near-integrable, intermediate level statistics
- **Large g**: Approaches another integrable limit (X-basis diagonal)
- **Maximum chaos**: Around $g \approx J$

**Implementation:**
```python
J = 1.0       # Fixed coupling strength
h ~ N(0, 0.5) # Random longitudinal field
g = 0.3       # Transverse field (integrability-breaking parameter)
```

**Observed behavior:**
- Level statistics: $\langle r \rangle \approx 0.15-0.20$ (sub-Poisson due to symmetry)
- Spectral: Still fails for g = 0.3 (too close to integrable)
- Krylov: Works, transitions at ρ_c ≈ 0.05-0.06

---

### 3.3 Chaotic Heisenberg Model

$$H = \sum_{i=1}^{n-1} \left( J^x_i \sigma^x_i \sigma^x_{i+1} + J^y_i \sigma^y_i \sigma^y_{i+1} + J^z_i \sigma^z_i \sigma^z_{i+1} \right) + \text{on-site terms}$$

**Properties:**
- **Random couplings in all directions**: No preferred basis
- **Fully chaotic**: Level statistics follow Random Matrix Theory (GOE)
- **Delocalized eigenstates**: Spread across all basis states

**Implementation:**
```python
# All couplings sampled randomly
J^x_i, J^y_i, J^z_i ~ N(0, 1)  # Two-site couplings
h^x_i, h^y_i, h^z_i ~ N(0, 0.5) # On-site fields
```

**Observed behavior:**
- Level statistics: $\langle r \rangle \approx 0.53$ (GOE)
- Spectral: Works, transitions at ρ_c ≈ 0.04
- Krylov: Works, transitions at ρ_c ≈ 0.04 (similar to Spectral)

---

## 4. Level Spacing Statistics (r-Ratio Theory)

### 4.1 Motivation: Why Level Statistics?

The distribution of energy level spacings is a **universal diagnostic** for quantum chaos:

- **Integrable systems**: Energy levels are **uncorrelated** (like random numbers)
- **Chaotic systems**: Energy levels exhibit **repulsion** (levels "avoid" each other)

This distinction arises from the structure of eigenstates, which directly impacts our reachability criteria.

### 4.2 Definition of the r-Ratio

The r-ratio (Oganesyan-Huse ratio) measures level repulsion without unfolding:

$$r_n = \frac{\min(s_n, s_{n+1})}{\max(s_n, s_{n+1})} \in [0, 1]$$

where $s_n = E_{n+1} - E_n$ are adjacent level spacings (ordered eigenvalues).

**Advantages over traditional spacing ratio:**
- No spectral unfolding required (avoids systematic errors)
- Bounded between 0 and 1 (easy interpretation)
- Robust to density-of-states variations

### 4.3 Theoretical Reference Values

The mean r-ratio $\langle r \rangle$ has exact analytical values for different universality classes:

| Ensemble | $\langle r \rangle$ | Formula | Physical System |
|----------|---------------------|---------|-----------------|
| **Poisson** | 0.386 | $\frac{4\ln 2 - 1}{2} \approx 0.386$ | Integrable |
| **GOE** | 0.531 | $\frac{4 - 2\sqrt{3}}{\pi} \cdot \frac{\Gamma(5/6)}{\Gamma(1/3)} \approx 0.531$ | Chaotic (time-reversal) |
| **GUE** | 0.600 | $\frac{2\sqrt{3}}{\pi} - \frac{1}{2} \approx 0.600$ | Chaotic (no time-reversal) |
| **GSE** | 0.676 | (numerical) | Chaotic (spin-orbit) |

### 4.4 Derivation of Poisson Result

For uncorrelated levels (Poisson), consecutive spacings $s_n, s_{n+1}$ are i.i.d. exponential:
$$P(s) = \bar{s}^{-1} e^{-s/\bar{s}}$$

The ratio $r = \min(s_1, s_2)/\max(s_1, s_2)$ has PDF:
$$P(r) = \frac{2}{(1+r)^2}$$

Therefore:
$$\langle r \rangle_{\text{Poisson}} = \int_0^1 r \cdot \frac{2}{(1+r)^2} dr = 2\ln 2 - 1 \approx 0.386$$

### 4.5 Level Repulsion in Chaotic Systems

For chaotic systems (GOE), the spacing distribution follows Wigner-Dyson statistics:
$$P(s) \propto s^\beta \exp(-c_\beta s^2)$$

where $\beta = 1$ (GOE), $\beta = 2$ (GUE), $\beta = 4$ (GSE).

The key physics is **level repulsion**: $P(s) \to 0$ as $s \to 0$.
- Levels cannot get arbitrarily close
- This increases the mean r-ratio (smaller spacings are suppressed)

### 4.6 Connection to Spectral Criterion Failure

**Why does Spectral fail for integrable systems?**

For integrable Hamiltonians (diagonal in some basis):
1. Eigenstates $|n\rangle$ are **fixed** (basis-independent of λ)
2. The spectral overlap $S(\lambda) = \sum_n |\langle n|\phi\rangle||\langle n|\psi\rangle|$ is **constant**
3. λ-optimization has no effect → $S^* = S(0)$

For Haar-random states:
$$\mathbb{E}[|\langle n|\psi\rangle||\langle n|\phi\rangle|] \approx \frac{\pi}{4d}$$

So $S \approx d \cdot \frac{\pi}{4d} = \frac{\pi}{4} \approx 0.785 < \tau = 0.99$

**Result:** Random states have low overlap with ANY fixed eigenbasis → Spectral always says "unreachable"

### 4.7 Why Krylov Works Regardless

Krylov criterion depends on **dynamical spreading**, not eigenstructure:

$$\mathcal{K}_m = \text{span}\{|\psi\rangle, H|\psi\rangle, H^2|\psi\rangle, \ldots\}$$

Even for diagonal $H$:
- $H|b_1 b_2 \ldots b_n\rangle = E_{b_1 b_2 \ldots}|b_1 b_2 \ldots b_n\rangle$
- Powers $H^k$ generate different eigenvalue weightings
- The span eventually covers the relevant eigenspace

**Key insight:** Krylov probes "can the dynamics push toward target?" which is a weaker requirement than "do eigenstates align with both states?"

### 4.8 Computation Protocol

```python
def compute_r_ratio(eigenvalues):
    """Compute mean r-ratio from ordered eigenvalues."""
    E = np.sort(eigenvalues)
    spacings = np.diff(E)

    r_values = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i+1]
        r = min(s1, s2) / max(s1, s2)
        r_values.append(r)

    return np.mean(r_values)

# For each trial:
# 1. Generate K Hamiltonians
# 2. Sample random λ ~ N(0,1), normalize to unit sphere
# 3. Construct H(λ) = Σᵢ λᵢ Hᵢ
# 4. Compute eigenvalues {Eₙ}
# 5. Calculate r-ratio
# Average over 10+ trials per (model, K) combination
```

---

## 5. Criteria Definitions

### 5.1 Moment Criterion (Weakest)

**Definition:**
Check if energy moment constraints can be satisfied:
$$\langle H^l \rangle_\psi = \langle H^l \rangle_\phi \quad \text{for } l = 1, 2$$

**Implementation:**
Uses `mathematics.is_unreachable_moment(psi, phi, hams)` which checks if the quadratic form $Q + xL^2$ is positive definite for some $x$.

**Returns:** `True` if state is classified as UNREACHABLE.

**Why it always shows P=0:**
- For random Haar-distributed states, moment conditions are almost always satisfiable
- Only detects "obviously unreachable" cases with very constrained states
- Necessary condition but not sufficient

---

### 5.2 Spectral Criterion (Medium)

**Definition:**
$$S^*(\lambda) = \max_\lambda \sum_n |\langle n(\lambda)|\phi\rangle| |\langle n(\lambda)|\psi\rangle|$$

where $|n(\lambda)\rangle$ are eigenstates of $H(\lambda) = \sum_i \lambda_i H_i$.

**Implementation:**
```python
result = optimize.maximize_spectral_overlap(psi, phi, hams,
    restarts=3, maxiter=50, seed=...)
unreachable = result["best_value"] < tau  # tau = 0.99
```

**Optimization:** L-BFGS-B with multiple random restarts on λ ∈ [-1,1]^K.

**Returns:** `True` if $S^* < \tau$ (UNREACHABLE).

**Why it fails for integrable systems:**
- Eigenstates $|n(\lambda)\rangle$ are fixed (don't depend on λ)
- No λ-optimization possible → $S(\lambda) = $ constant
- Random states have low overlap with fixed eigenbasis

---

### 5.3 Krylov Criterion (Strongest)

**Definition:**
$$R^*(\lambda) = \max_\lambda \|P_{\mathcal{K}_m}|\phi\rangle\|^2$$

where $\mathcal{K}_m = \text{span}\{|\psi\rangle, H|\psi\rangle, \ldots, H^{m-1}|\psi\rangle\}$.

**Implementation:**
```python
m = min(K, d)  # Krylov dimension
result = optimize.maximize_krylov_score(psi, phi, hams, m=m,
    restarts=3, maxiter=50, seed=...)
unreachable = result["best_value"] < tau
```

**Returns:** `True` if $R^* < \tau$ (UNREACHABLE).

**Why it works for all systems:**
- Krylov subspace depends on H's action on states
- Even diagonal H generates spanning subspace (through repeated application)
- Nearly λ-independent: $H(c\lambda)^k|\psi\rangle = c^k H(\lambda)^k|\psi\rangle$ → same span

---

## 6. Experimental Protocol

### 6.1 For Each (Model, Dimension, K) Combination:

```python
for trial in range(N_trials):
    # 1. Generate Hamiltonians
    hams = generate_ensemble(model, n_qubits, K, rng)

    # 2. Sample random states
    psi = rand_ket(d)  # Haar-uniform initial state
    phi = rand_ket(d)  # Haar-uniform target state

    # 3. Apply each criterion
    moment_unreachable = eval_moment(psi, phi, hams)
    spectral_unreachable = eval_spectral(psi, phi, hams, tau)
    krylov_unreachable = eval_krylov(psi, phi, hams, tau)

    # 4. Count unreachable outcomes
    ...

# 5. Compute statistics
P_moment = count_moment / N_trials
P_spectral = count_spectral / N_trials
P_krylov = count_krylov / N_trials
```

### 6.2 Parameters Used

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N_trials | 25 | Balance between statistics and runtime |
| tau | 0.99 | Standard threshold for criteria |
| restarts | 3 | Sufficient for small K |
| maxiter | 50 | Fast convergence for these problems |
| K_max | 16 | Up to full Hilbert space for d=16 |

---

## 7. Key Results

### 7.1 Summary Table

| Model | d | ⟨r⟩ | Moment | Spectral | Krylov |
|-------|---|-----|--------|----------|--------|
| Integrable | 8 | 0.41 | P=0 | P=1 always | Transitions |
| Near-integrable | 8 | 0.19 | P=0 | P=1 always | Transitions |
| Chaotic | 8 | 0.54 | P=0 | Transitions | Transitions |
| Integrable | 16 | 0.35 | P=0 | P=1 always | Transitions |
| Near-integrable | 16 | 0.16 | P=0 | P=1 always | Transitions |
| Chaotic | 16 | 0.52 | P=0 | ρ_c ≈ 0.04 | ρ_c ≈ 0.04 |

### 7.2 Physical Interpretation

**Integrable → Spectral fails:**
- Fixed eigenbasis cannot be optimized
- λ-optimization has no effect
- P(unreachable) = 1.0 always

**Chaotic → Both work:**
- Flexible eigenbasis allows optimization
- Both criteria show similar ρ_c
- Krylov ≈ Spectral when eigenbasis is "random"

**Krylov is universal:**
- Works for all integrability levels
- Nearly λ-independent
- Based on dynamical spreading, not eigenstructure

---

## 8. Connection to Main Finding

### Why Krylov < Spectral is Universal

1. **Krylov probes dynamics**: "Can H repeatedly push |ψ⟩ toward |φ⟩?"
2. **Spectral probes statics**: "Is there an H whose eigenstates align with both?"

The dynamical question is **weaker** than the static question:
- If eigenstates align → dynamics can reach target (sufficient)
- But dynamics can reach target even without alignment (Krylov works for integrable!)

### Why GEO2 Ratio Grows

GEO2's geometric locality creates **pseudo-integrable** behavior:
- Structured eigenspaces resist optimization
- As d grows, structure becomes more rigid
- Spectral struggles, Krylov maintains efficiency

This mirrors the integrable/chaotic distinction:
- **Local structure** → Spectral disadvantaged
- **Random structure** → Both criteria equal

---

## 9. λ-Dependence Analysis

This section provides detailed mathematical analysis of why Spectral and Krylov criteria behave differently under λ-optimization.

### 9.1 Setup: Linear Combination of Hamiltonians

Given K Hamiltonians $\{H_1, \ldots, H_K\}$, we consider:
$$H(\lambda) = \sum_{i=1}^K \lambda_i H_i$$

where $\lambda = (\lambda_1, \ldots, \lambda_K) \in \mathbb{R}^K$.

### 9.2 Spectral Criterion: λ-Dependent

**Definition:**
$$S(\lambda) = \sum_n |\langle n(\lambda)|\phi\rangle| \cdot |\langle n(\lambda)|\psi\rangle|$$

where $|n(\lambda)\rangle$ are eigenstates of $H(\lambda)$.

**Why λ-dependent:**

Consider scaling $\lambda \to c\lambda$ for constant $c > 0$:
$$H(c\lambda) = c \cdot H(\lambda)$$

The eigenvalues scale: $E_n(c\lambda) = c \cdot E_n(\lambda)$

But the eigenstates are **unchanged**: $|n(c\lambda)\rangle = |n(\lambda)\rangle$

So uniform scaling doesn't help. However, **changing the ratios** $\lambda_i/\lambda_j$ **does** rotate the eigenbasis:

$$|n(\lambda)\rangle \neq |n(\lambda')\rangle \quad \text{for } \lambda_i/\lambda_j \neq \lambda'_i/\lambda'_j$$

**Physical interpretation:** Different λ-directions select different "preferred bases" for the combined Hamiltonian. The spectral overlap depends on how well these bases align with both $|\psi\rangle$ and $|\phi\rangle$.

**For integrable systems:** All $H_i$ commute, so they share a common eigenbasis:
$$[H_i, H_j] = 0 \implies |n(\lambda)\rangle = |n\rangle \quad \forall \lambda$$

This makes $S(\lambda) = S(0)$ constant, and λ-optimization is useless.

### 9.3 Krylov Criterion: λ-Independent

**Definition:**
$$R(\lambda) = \|P_{\mathcal{K}_m(H(\lambda), |\psi\rangle)} |\phi\rangle\|^2$$

where the Krylov subspace is:
$$\mathcal{K}_m(H, |\psi\rangle) = \text{span}\{|\psi\rangle, H|\psi\rangle, H^2|\psi\rangle, \ldots, H^{m-1}|\psi\rangle\}$$

**Key Theorem: Scaling Invariance**

For any $c \neq 0$:
$$\mathcal{K}_m(cH, |\psi\rangle) = \mathcal{K}_m(H, |\psi\rangle)$$

**Proof:**
$$(cH)^k |\psi\rangle = c^k H^k |\psi\rangle$$

Since $c^k$ is just a scalar, $\text{span}\{(cH)^k|\psi\rangle\} = \text{span}\{H^k|\psi\rangle\}$.

Therefore:
$$\mathcal{K}_m(H(c\lambda), |\psi\rangle) = \mathcal{K}_m(H(\lambda), |\psi\rangle)$$

**Implication:** Only the **direction** of $\lambda$ matters, not its magnitude. The effective parameter space is reduced from $\mathbb{R}^K$ to the $(K-1)$-sphere $S^{K-1}$.

### 9.4 The "Fixed ≈ Optimized" Observation

Empirically, we observe that for Krylov:
$$R^*_{\text{opt}} = \max_\lambda R(\lambda) \approx R(\lambda_0) = R_{\text{fixed}}$$

for essentially any fixed $\lambda_0$ with generic ratios.

**Why there IS a small gap (even though theory predicts near-equality):**

The scaling invariance theorem (Section 9.3) only guarantees that $\mathcal{K}_m(c\lambda) = \mathcal{K}_m(\lambda)$ for **scalar** scaling. However, when we change the **direction** of $\lambda$ on the unit sphere $S^{K-1}$, the Krylov subspace **does** change:

$$\lambda \neq \mu \text{ (non-collinear)} \implies \mathcal{K}_m(H(\lambda), |\psi\rangle) \neq \mathcal{K}_m(H(\mu), |\psi\rangle)$$

**Sources of the small gap:**

1. **Direction sensitivity**: The Krylov subspace depends on λ-direction through:
   $$H(\lambda)^k|\psi\rangle = \left(\sum_i \lambda_i H_i\right)^k |\psi\rangle$$
   This is a polynomial in the $\lambda_i$, so different directions give different subspaces.

2. **Multinomial expansion**: For $k \geq 2$:
   $$(H_1 + H_2)^k \neq H_1^k + H_2^k$$
   Cross-terms like $H_1 H_2 + H_2 H_1$ introduce non-trivial λ-dependence.

3. **Transition region sensitivity**: Near the critical density $\rho_c$, small changes in subspace geometry can flip the criterion outcome.

**Why the gap is SMALL:**

1. **Generic subspace argument**: For random Hamiltonians, the Krylov subspace for any λ-direction spans a "typical" $m$-dimensional subspace. All typical subspaces have similar projection properties.

2. **Measure concentration**: In high dimensions, random subspaces concentrate around their mean properties (Johnson-Lindenstrauss-type phenomenon).

3. **Continuity**: The map $\lambda \mapsto P_{\mathcal{K}_m(\lambda)}$ is continuous, so nearby λ give nearby projections.

**Mathematical formalization:**

Let $G_{m,d}$ be the Grassmannian of $m$-dimensional subspaces in $\mathbb{C}^d$. The map:
$$\Phi: S^{K-1} \to G_{m,d}, \quad \lambda \mapsto \mathcal{K}_m(H(\lambda), |\psi\rangle)$$

has the following properties:
- $\Phi$ is continuous (from continuity of eigenvalue perturbation theory)
- For generic $\{H_i\}$ and $|\psi\rangle$, the image $\Phi(S^{K-1})$ is a **small** subset of $G_{m,d}$
- The projection score $R(\lambda) = \|P_{\Phi(\lambda)}|\phi\rangle\|^2$ varies **slowly** over $S^{K-1}$

**Quantitative estimates:**

For random $|\phi\rangle$ and typical $\mathcal{K}_m$:
$$\mathbb{E}[R] \approx \frac{m}{d}, \quad \text{Var}[R] \approx \frac{m(d-m)}{d^2(d+1)}$$

The variance is $O(1/d)$, explaining why the gap decreases with dimension.

This contrasts sharply with Spectral, where eigenbasis alignment is highly sensitive to exact λ-ratios.

### 9.5 Numerical Evidence

From our experiments:

| Criterion | Gap = P(Fixed) - P(Optimized) | Trend with d |
|-----------|-------------------------------|--------------|
| Spectral | Large (0.2-0.4 at transition) | Decreases |
| Krylov | Small (0.01-0.05) | Decreases |

The Spectral gap being larger confirms that eigenbasis alignment benefits strongly from optimization. The Krylov gap being small confirms the scaling invariance theory.

### 9.6 Connection to Integrability

**Why Spectral fails for integrable systems:**

When $[H_i, H_j] = 0$ for all $i, j$:
- All Hamiltonians share the same eigenbasis
- $|n(\lambda)\rangle = |n\rangle$ is **independent of λ**
- $S(\lambda) = S(0)$ is constant
- Optimization cannot improve overlap

**Why Krylov still works:**

Even for diagonal (integrable) $H$:
$$H|b_1 \ldots b_n\rangle = E_{b_1\ldots b_n} |b_1 \ldots b_n\rangle$$

Applying $H$ repeatedly to a generic state $|\psi\rangle = \sum_{\mathbf{b}} c_{\mathbf{b}} |\mathbf{b}\rangle$:
$$H^k |\psi\rangle = \sum_{\mathbf{b}} c_{\mathbf{b}} E_{\mathbf{b}}^k |\mathbf{b}\rangle$$

These vectors are **linearly independent** (generically), so:
$$\dim(\mathcal{K}_m) = \min(m, \text{support}(|\psi\rangle))$$

For Haar-random $|\psi\rangle$, support is full ($d$ dimensions), so Krylov achieves full rank for $m \geq d$.

---

## 10. Limitations

1. **Small system sizes**: Only tested d ≤ 16 (computational constraints)
2. **Specific models**: May not generalize to all integrable/chaotic systems
3. **Fixed g value**: Near-integrable tested at g=0.3 only
4. **Limited trials**: 25 trials per point (larger samples would reduce noise)

---

## 11. References

1. `scripts/integrability/three_models_comparison.py` - Implementation
2. `fig/integrability/three_models_comparison.png` - Results figure
3. `fig/integrability/rho_c_vs_r_ratio.png` - Correlation plot
4. `reach/mathematics.py` - Criterion implementations
5. `reach/optimize.py` - λ-optimization routines

---

**End of Methodology Document**
