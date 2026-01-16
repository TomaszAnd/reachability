# Ensemble Dimension Analysis

## Executive Summary

**Canonical basis and GEO2Local ensembles are NOT directly comparable due to fundamentally different dimensional scaling.** Canonical basis operates on a **single qudit** with Hilbert space dimension `d ∈ {8,10,12,14,16}`, while GEO2Local operates on `n_sites = nx × ny` **qubits** with exponential Hilbert space dimension `2^n_sites`. The control density ρ = K/d² uses `d` to mean different things: in canonical, `d` is the qudit dimension; in GEO2Local, `d = 2^n_sites` is the total Hilbert space dimension. These ensembles explore different physical regimes: **single large qudit vs many small qubits**.

---

## Canonical Basis Ensemble

### Dimension Parameter `d`

**Definition**: `d` = Hilbert space dimension of a **single d-level qudit**

**Python Code Evidence** (reach/models.py):
```python
# Lines 99-123
class CanonicalBasis:
    """
    Canonical basis for d×d Hermitian matrices.

    Args:
        dim: Hilbert space dimension       # ← Single qudit dimension
        include_identity: If True, include identity matrix
    """

    def __init__(self, dim: int, include_identity: bool = True):
        if dim < 2:
            raise ValueError(f"Dimension must be ≥ 2, got {dim}")

        self.dim = dim                    # ← d is Hilbert space dimension
        self.include_identity = include_identity

        # Build canonical basis operators
        self.operators = self._build_canonical_basis()
        self.L = len(self.operators)

        # Validate operator count
        expected_L = dim * dim if include_identity else dim * dim - 1  # ← d² operators
```

**Interpretation**:
- When we vary `d ∈ {8, 10, 12, 14, 16}` in plots, we are varying the **qudit Hilbert space dimension**.
- This is a **single quantum system** (not a composite of multiple qubits).
- Operators are `d × d` Hermitian matrices.

### Operator Space Size

**Formula**: L = d² total operators (d²-1 if identity excluded)

**Breakdown**:
- X-type operators: d(d-1)/2
- Y-type operators: d(d-1)/2
- Z-type operators: d-1
- Identity: 1
- **Total**: d(d-1)/2 + d(d-1)/2 + (d-1) + 1 = d²

### Hilbert Space Dimension

- **Single qudit**: Hilbert space dimension = `d`
- **Composite system**: Not used in canonical basis ensemble (single qudit only)
- **Conclusion**: `d` represents **qudit Hilbert space dimension** (d = 8 means 8-level qudit)

---

## GEO2Local Ensemble

### Dimension Parameters

**Python Code Evidence** (reach/models.py):
```python
# Lines 206-245
class GeometricTwoLocal:
    """
    Gaussian Geo-Local (GEO2) ensemble on a rectangular lattice.

    Basis: P_2(G) contains all 1-local {X,Y,Z}_i and 2-local {X,Y,Z}_i⊗{X,Y,Z}_j
    Pauli terms on lattice sites and nearest-neighbor edges.

    Formula: L = 3n + 9|E(G)| where n = nx * ny sites, |E(G)| = number of edges.

    Args:
        nx: Lattice width (number of sites in x direction)
        ny: Lattice height (number of sites in y direction)
        periodic: Use periodic boundary conditions
        backend: "sparse" (default) or "dense" operator construction
    """

    def __init__(self, nx: int, ny: int, periodic: bool = False, backend: str = "sparse"):
        if nx < 1 or ny < 1:
            raise ValueError(f"Lattice dimensions must be ≥ 1, got nx={nx}, ny={ny}")

        self.nx = nx
        self.ny = ny
        self.periodic = periodic
        self.backend = backend
        self.n_sites = nx * ny                # ← Number of qubits
        self.dim = 2 ** self.n_sites         # ← Total Hilbert space = 2^n (qubit-specific!)

        # Build sparse Pauli basis
        self.Hs = self._build_pauli_basis()
        self.L = len(self.Hs)

        # Validate operator count: L = 3n + 9|E|
        edges = self._build_lattice_edges()
        expected_L = 3 * self.n_sites + 9 * len(edges)
        assert self.L == expected_L, (
            f"Operator count mismatch: got {self.L}, expected {expected_L} "
            f"(3n={3*self.n_sites} + 9|E|={9*len(edges)})"
        )
```

### Key Parameters

- **`n_sites` (nx × ny)**: Number of lattice sites (qubits)
- **`self.dim = 2**n_sites`**: Total Hilbert space dimension for n qubits
- **Per-site dimension**: d = 2 (qubit) **[HARDCODED - no 'd' parameter in __init__]**

### Qubit vs Qudit

**Current Implementation**: **Qubit-specific (d=2 hardcoded)**
- Line 233: `self.dim = 2 ** self.n_sites` explicitly uses base 2
- No parameter to generalize to d>2 qudits
- Pauli basis {X, Y, Z} is qubit-specific (2×2 matrices)

**Paper (arxiv:2510.06321v1)**:
- Defines GEO2Local exclusively for qubits
- Uses Pauli operators {X, Y, Z} which are 2×2 matrices
- Complexity arguments assume d=2 per site

**Generalization Possibility**:
- **Mathematically**: Yes, generalized Pauli matrices (Weyl operators) exist for d>2 qudits
- **In current codebase**: No, would require significant refactoring to support d>2

---

## Comparison: Canonical vs GEO2Local

### Dimensional Correspondence

| Ensemble | Parameter Varied | Per-Site Dim | Total Hilbert Dim | Operator Space Dim | Control Density Denominator |
|----------|------------------|--------------|-------------------|-------------------|----------------------------|
| **Canonical** | d (qudit dim) | d (single system) | d | d² | K/d² |
| **GEO2Local** | n_sites = nx×ny | 2 (qubit) | 2^n | 3n + 9\|E\| | K/(2^n)² |

### Scaling Comparison

#### Canonical Basis
- Vary d: d = 8 → 10 → 12 → 14 → 16
- Hilbert space: 8 → 10 → 12 → 14 → 16 (linear in d)
- Operator space: 64 → 100 → 144 → 196 → 256 (quadratic in d)

#### GEO2Local
- Vary n_sites: n = 4 → 5 → 6 → 7 → 8 (for example)
- Hilbert space: 16 → 32 → 64 → 128 → 256 (exponential: 2^n)
- Operator space: 3n + 9|E| ≈ 12 → 15 → 18 → 21 → 24 (linear in n for small lattices)

### Direct Comparability

**Question**: Can we plot both on same axes?

**Answer**: **NO - ensembles are NOT directly comparable** because:

1. **Different per-site dimensions**: Canonical uses d-level qudits; GEO2Local uses d=2 qubits
2. **Different scaling regimes**:
   - Canonical: Single qudit with polynomial Hilbert space (d)
   - GEO2Local: Many qubits with exponential Hilbert space (2^n)
3. **Different physical systems**:
   - Canonical: Single d-dimensional quantum system
   - GEO2Local: Lattice of n two-level systems

**What "d" means in control density ρ = K/d²:**
- **Canonical**: d is qudit dimension (8, 10, 12, 14, 16)
- **GEO2Local**: d = 2^n_sites is total Hilbert dimension (16, 32, 64, 128, 256...)

**Practical consequence**:
- A canonical d=16 system is a **single 16-level qudit**
- A GEO2Local with d=16 system is **4 qubits** (2×2 lattice, since 2^4 = 16)
- These are fundamentally different physical systems!

### Recommendation

**For LaTeX documentation**:
- **DO NOT claim direct comparability** between ensembles
- **DO clarify dimensional meanings explicitly**:
  - Canonical: "We vary qudit dimension d ∈ {8,10,12,14,16} for a single d-level quantum system"
  - GEO2Local: "We vary lattice size n = nx × ny qubits with total Hilbert dimension 2^n"
- **DO explain different regimes**:
  - "Canonical basis explores single high-dimensional qudits"
  - "GEO2Local explores many low-dimensional qubits with geometric constraints"

---

## Scaling Analysis

### Canonical Basis

**What we vary**: Qudit dimension d (single system)
- d = 8: Hilbert dim = 8, Operators = 64
- d = 10: Hilbert dim = 10, Operators = 100
- d = 12: Hilbert dim = 12, Operators = 144

**Control Density**: ρ = K/d²
- Natural choice because operator space is d²-dimensional
- ρ measures fraction of available operator space used

**Physical Interpretation**: Larger qudit → more energy levels → more operators needed for control

**Evidence** (reach/cli.py):
```python
# Line 1167
"rho_K_over_d2": K / (args.dim**2),  # ← Uses d² denominator
```

### GEO2Local

**What we vary**: Lattice size n_sites = nx × ny (number of qubits)
- n=4 (2×2): Hilbert dim = 16, Operators ≈ 48
- n=9 (3×3): Hilbert dim = 512, Operators ≈ 108

**Control Density Options**:
- **Option A**: K/(2^n)² → extremely small due to exponential Hilbert space
- **Option B**: K/n → scales with lattice size
- **Option C**: K/(3n + 9|E|) → scales with operator space size

**Correct Choice**: **Option A: K/(2^n)² = K/d²** where d = 2^n

**Evidence** (reach/cli.py):
```python
# Lines 1059, 1167, 1414, 1447
"rho_K_over_d2": K / (args.dim**2),  # ← Uses dim² where dim = 2^n_sites
```

**Interpretation**:
- For GEO2Local, the code uses `dim = 2^n_sites` as the effective dimension
- Control density ρ = K/(2^n)² becomes exponentially small as lattice grows
- This reflects the exponential growth of Hilbert space

---

## Qudit Generalization of GEO2Local

### Is It Possible?

**Mathematical**: Yes
- Generalized Pauli matrices (Weyl operators) exist for arbitrary d
- d-dimensional Pauli group: {X^a Z^b : a,b ∈ Z_d} with d² elements
- Geometric 2-local constraints still meaningful for qudits

**From Paper (arxiv:2510.06321v1)**:
- Paper focuses exclusively on qubits (d=2)
- All complexity arguments assume qubit systems
- No mention of qudit generalization

**Our Implementation** (reach/models.py):
- **Qubit-specific**: Line 233 hardcodes `self.dim = 2 ** self.n_sites`
- Would require refactoring `_build_pauli_basis()` to support d>2
- Currently uses qutip Pauli matrices which are 2×2

**Recommendation**:
**"GEO2Local is qubit-specific in both the paper and our implementation; qudit generalization would require extending the Pauli basis to generalized Weyl operators and is not supported in the current codebase."**

---

## Conclusions for LaTeX

### Main Points to Address

1. **Canonical Basis**:
   - ✅ "We vary qudit dimension d ∈ {8,10,12,14,16}"
   - ✅ "Single d-level quantum system with Hilbert space dimension d"
   - ✅ "Operator space dimension d²"
   - ✅ "Control density ρ = K/d² measures fraction of d²-dimensional operator space"

2. **GEO2Local**:
   - ✅ "Qubit-specific ensemble (d=2 per site)"
   - ✅ "We vary lattice size n_sites = Lx × Ly"
   - ✅ "Total Hilbert space dimension = 2^n_sites (exponentially large)"
   - ✅ "Operator space dimension L = 3n + 9|E| (linearly large)"
   - ✅ "NOT directly comparable to canonical basis at same 'd'"
   - ✅ "Control density ρ = K/(2^n)² reflects exponential Hilbert space growth"

3. **Comparison Strategy**:
   - ❌ **DO NOT say**: "Both ensembles show similar ρ_c ~ 1/d² scaling"
   - ✅ **DO say**: "Ensembles explore different physical regimes:"
     - Canonical: Single high-dimensional qudit (polynomial Hilbert space)
     - GEO2Local: Many low-dimensional qubits (exponential Hilbert space)
   - ✅ **DO clarify**: "While both use control density ρ = K/d², the meaning of 'd' differs fundamentally"

### Recommended LaTeX Changes

**Section: Canonical Basis** (main.tex line ~111)
- ADD: "For the canonical basis ensemble, d represents the Hilbert space dimension of a **single d-level qudit**. When we vary d ∈ {8,10,12,14,16}, we are exploring different qudit dimensions, not composite systems of multiple qubits."

**Section: GEO2Local** (main.tex line ~123)
- **REPLACE** existing paragraph with:
  - General Hamiltonian equation from arxiv:2510.06321v1
  - k=2 specialization with explicit operator count
  - Dimensional clarification: "GEO2Local is inherently **qubit-based** (d=2 per site). The total Hilbert space dimension 2^n grows exponentially with lattice size n = Lx × Ly."
  - **Comparison note**: "This contrasts with the canonical basis ensemble (Section~\ref{sec:canonical}), which explores a single qudit with Hilbert space dimension d ∈ {8,10,12,14,16}. Despite both using control density ρ = K/d², the ensembles probe different physical regimes: GEO2Local examines many weakly-coupled qubits under geometric constraints, while canonical basis examines a single strongly-interacting qudit."

**Section: Results** (if comparing ensembles)
- **ADD disclaimer**: "Direct quantitative comparison between canonical basis and GEO2Local scaling is precluded by their fundamentally different dimensional structures. Canonical basis operates on a single d-dimensional qudit, while GEO2Local operates on n = log₂(d) qubits arranged on a lattice. For example, d=16 represents a 16-level qudit in canonical basis, but 4 qubits (2×2 lattice) in GEO2Local. These ensembles illuminate complementary aspects of reachability: canonical basis reveals how qudit dimension affects control, while GEO2Local reveals how geometric locality constrains control in spatially extended systems."

---

## Verification Log

### Canonical Basis Evidence

**File**: `reach/models.py`
**Lines**: 99-176
**Key Code**:
```python
class CanonicalBasis:
    def __init__(self, dim: int, include_identity: bool = True):
        self.dim = dim  # ← d is Hilbert space dimension of single qudit
        # ...
        d = self.dim  # Line 143
        # Operators are d×d matrices (lines 150, 159, 167)
```

**Conclusion**: `d` represents **single qudit Hilbert space dimension**

### GEO2Local Evidence

**File**: `reach/models.py`
**Lines**: 206-245
**Key Code**:
```python
class GeometricTwoLocal:
    def __init__(self, nx: int, ny: int, periodic: bool = False, backend: str = "sparse"):
        self.n_sites = nx * ny               # Number of qubits
        self.dim = 2 ** self.n_sites        # ← d=2 hardcoded, exponential Hilbert space
```

**Conclusion**: d=2 hardcoded (qubit-specific), `dim = 2^n_sites` is total Hilbert space dimension

### Control Density Evidence

**File**: `reach/cli.py`
**Lines**: 1059, 1167, 1414, 1447
**Key Code**:
```python
"rho_K_over_d2": K / (args.dim**2),  # ← All ensembles use K/d²
```

**Conclusion**: Control density denominator is **d²** where d = Hilbert space dimension (different meanings for different ensembles!)

---

## Critical Questions - ANSWERED

1. ✅ **What is d in canonical plots?**
   - **Answer**: d is the **qudit Hilbert space dimension** (single d-level quantum system)

2. ✅ **What is dim in GEO2Local?**
   - **Answer**: `dim = 2^n_sites` is the **total Hilbert space dimension** for n qubits (exponential in lattice size)

3. ✅ **Are ensembles comparable?**
   - **Answer**: **NO** - fundamentally different scaling regimes (single high-d qudit vs many d=2 qubits)

4. ✅ **Is GEO2Local qubit-only?**
   - **Answer**: **YES** - d=2 hardcoded in implementation (line 233 of models.py), paper also assumes qubits

---

## End of Analysis
