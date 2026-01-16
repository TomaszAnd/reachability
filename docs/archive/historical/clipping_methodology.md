# Clipping Methodology for Linearized Fits Analysis

## Executive Summary

This document analyzes the data clipping methodology used in linearized fits analysis for quantum reachability experiments. It identifies where clipping occurs in the pipeline, explains why it's necessary, documents current issues, and proposes improved approaches for publication-quality analysis.

**Key Findings:**
- **Two-stage clipping**: Data floored at `1e-12` during generation, then clipped to `[1e-6, 1-1e-6]` at visualization
- **Root cause of boundary values**: True binomial outcomes (0/N or N/N), not numerical artifacts
- **Main issue**: Hard clipping creates artificial plateaus that bias linear regression
- **Recommendation**: Transition-region-only fitting with explicit data quality metrics

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data Pipeline Analysis](#2-data-pipeline-analysis)
3. [Current Clipping Methodology](#3-current-clipping-methodology)
4. [Problems with Hard Clipping](#4-problems-with-hard-clipping)
5. [Alternative Approaches](#5-alternative-approaches)
6. [Recommended Methodology](#6-recommended-methodology)
7. [Implementation Guide](#7-implementation-guide)
8. [Quality Metrics](#8-quality-metrics)

---

## 1. Problem Statement

### Why Clipping is Needed

When applying log-scale transforms to probability data for linearization:

| Transform | Domain | Issue | Mathematical Expression |
|-----------|--------|-------|------------------------|
| **log(P)** | P ∈ (0, 1) | log(0) = -∞ | Exponential linearization |
| **logit(P)** | P ∈ (0, 1) | logit(0) = -∞, logit(1) = +∞ | Fermi-Dirac linearization |
| **log(1-P)** | P ∈ [0, 1) | log(0) = -∞ | Complementary analysis |

**Consequences of undefined values:**
- **Plotting**: Infinite axis limits, rendering failures
- **Fitting**: Undefined loss function, optimizer divergence
- **Statistics**: Undefined mean/variance, NaN propagation

### When P = 0 or P = 1 Occurs

In quantum reachability Monte Carlo experiments:

```python
# From reach/analysis.py line 289-295
unreachable_count = 0
total_count = 0

for trial in trials:
    if S_star < tau:  # Spectral overlap below threshold
        unreachable_count += 1
    total_count += 1

probability = unreachable_count / total_count  # Can be 0/N or N/N
```

**Physical interpretation of boundary values:**

| P value | Meaning | Frequency | Example |
|---------|---------|-----------|---------|
| **P = 0** | All states reachable (0/N unreachable) | Sparse (low K, large d) | K=2, d=26: all operators span full space |
| **P = 1** | All states unreachable (N/N unreachable) | Dense (large K, small d) | K=25, d=10: over-determined system |
| **0 < P < 1** | Partial reachability | Most common | Transition region |

These are **genuine physical outcomes**, not numerical artifacts.

---

## 2. Data Pipeline Analysis

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION STAGE                             │
│                    (reach/analysis.py)                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
    Monte Carlo Loop: N_hamiltonians × N_states trials
                                ↓
    For each trial:
      1. Generate H₁, ..., Hₖ (random matrices or canonical basis)
      2. Generate initial state |ψ⟩ = |0⟩
      3. Generate target state |φ⟩ (Haar random)
      4. Optimize: S* = max_λ S(λ) or R* = max_λ R(λ)
      5. Classify: unreachable if S* < τ or R* < τ
                                ↓
    Raw probability: P_raw = unreachable_count / total_count
    ↓ ↓ ↓
    **FIRST CLIPPING** (analysis.py:295)
    P_data = max(DISPLAY_FLOOR, P_raw)
    where DISPLAY_FLOOR = 1e-12
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA STORAGE                                      │
│                    (pickle files)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION STAGE                               │
│                    (scripts/plot_*.py)                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
    Load P_data from pickle
                                ↓
    **SECOND CLIPPING** (plot scripts)
    P_clipped = clip(P_data, ε, 1-ε)
    where ε = 1e-6
                                ↓
    Apply transforms:
      - log(P_clipped) for exponential
      - logit(P_clipped) = log(P/(1-P)) for Fermi-Dirac
                                ↓
    Linear regression: y = m*x + b
                                ↓
    Extract physical parameters from (m, b)
```

### Source Code Locations

| Stage | File | Line | Operation | Value |
|-------|------|------|-----------|-------|
| Generation | `reach/analysis.py` | 295 | `max(DISPLAY_FLOOR, probability)` | `1e-12` |
| Plotting | `scripts/plot_linearized_fits.py` | 24-26 | `clip_probabilities(P, epsilon)` | `1e-6` |
| Plotting | `scripts/plot_log_scale_analysis.py` | 23-25 | `clip_probabilities(P, epsilon)` | `1e-6` |

### No Smoothing or Regularization

**Important**: The reach package does NOT apply:
- Laplace smoothing: (count + α) / (total + 2α)
- Beta-binomial conjugate prior
- Bayesian credible intervals
- Continuity corrections

The data is **raw binomial proportions** with only boundary flooring for visualization purposes.

---

## 3. Current Clipping Methodology

### Implementation

```python
# From plot_linearized_fits.py lines 23-26
EPSILON = 1e-6  # Clipping bounds

def clip_probabilities(P, epsilon=EPSILON):
    """Clip probabilities to [epsilon, 1-epsilon] to avoid log(0) and division by zero."""
    return np.clip(P, epsilon, 1 - epsilon)
```

### Effect on Transforms

For `ε = 1e-6`:

| Original P | Clipped P | log(P) | logit(P) | Comment |
|------------|-----------|--------|----------|---------|
| `0.0` | `1e-6` | `-13.8` | `-13.8` | Artificial floor |
| `1e-12` | `1e-6` | `-13.8` | `-13.8` | Indistinguishable |
| `1e-8` | `1e-6` | `-13.8` | `-13.8` | Lost information |
| `1e-4` | `1e-4` | `-9.2` | `-9.2` | Preserved |
| `0.5` | `0.5` | `-0.69` | `0.0` | Midpoint |
| `0.999` | `0.999` | `-0.001` | `+6.9` | Preserved |
| `0.9999` | `0.9999` | `-0.0001` | `+9.2` | Preserved |
| `1.0` | `1-1e-6` | `-1e-6` | `+13.8` | Artificial ceiling |

**Key observation**: Multiple distinct probability values collapse to the same clipped value, creating artificial data concentration at boundaries.

### Resulting Issues in Linearized Plots

Looking at the output from `plot_linearized_fits.py`:

```
Panel (c): KRYLOV (τ=0.99) - Step-like Fermi-Dirac Linearization
d      K_c        Δ          R²         Notes
10     9.562      0.4759     0.7812     Step-like
14     13.621     0.6687     0.7997     Gradual
18     16.815     0.8890     0.8120     Gradual
22     20.811     1.0828     0.7946     Gradual
26     24.881     1.2700     0.7848     Gradual
```

**R² ~ 0.78-0.81**: Moderate fit quality, likely degraded by boundary plateaus

---

## 4. Problems with Hard Clipping

### Issue 1: Plateau Artifacts

**Problem**: Many data points cluster at clip boundaries

```
logit(P) vs K plot:
    +15 ┤                                  ████  ← Ceiling plateau
    +10 ┤                              ●●●
     +5 ┤                         ●●●
      0 ┤                    ●●●
     -5 ┤              ●●●
    -10 ┤         ●●●
    -15 ┤  ████                                  ← Floor plateau
        └────────────────────────────────────────→ K
```

**Consequence**: Linear regression includes non-informative plateau points, biasing slope and intercept estimates.

### Issue 2: Information Loss

**Example**: For Krylov at τ=0.99:
- d=10: Many K values yield P ≈ 1 (all unreachable)
- After clipping: All become P = 1-1e-6 → logit(P) = +13.8
- Cannot distinguish K where P = 0.999 from K where P = 1.000

**Lost information**:
- Rate of approach to saturation
- Curvature near boundary
- Confidence in extreme values

### Issue 3: Biased Parameter Estimates

Hard clipping biases Fermi-Dirac parameters:

```python
# True model: P = 1/(1 + exp((K - K_c)/Δ))
# Linearized: logit(P) = -K/Δ + K_c/Δ

# With hard clipping:
# - Plateaus reduce slope magnitude → overestimate Δ
# - Artificial floor/ceiling shift intercept → bias K_c
```

**Observed effect**: Δ estimates likely **too large** due to plateau-flattened slopes.

### Issue 4: Non-Differentiability

```python
# Clip function derivative:
dP_clip/dP = {
    0   if P < ε or P > 1-ε,  # ← Derivative vanishes at boundaries
    1   if ε ≤ P ≤ 1-ε
}
```

**Problem**: Gradient-based uncertainty propagation (e.g., bootstrap, jackknife) breaks down at boundaries.

### Issue 5: False Confidence

R² values computed on clipped data give misleading quality metrics:
- R² = 0.95 looks good
- But may include 40% plateau points with zero variance
- True transition-region fit could be R² = 0.85 with fewer points

---

## 5. Alternative Approaches

### Method 1: Soft Clipping (Smoothed Boundaries)

**Concept**: Replace hard clip with smooth sigmoid transition

```python
def soft_clip(P, epsilon=1e-6, steepness=100):
    """
    Soft clipping with differentiable boundaries.

    P_soft = ε + (1-2ε) * sigmoid(steepness * (P - 0.5))

    Properties:
    - P=0 → ε (smooth approach)
    - P=1 → 1-ε (smooth approach)
    - P=0.5 → 0.5 (midpoint preserved)
    - Differentiable everywhere
    - Monotonic (preserves ordering)
    """
    from scipy.special import expit  # sigmoid
    return epsilon + (1 - 2*epsilon) * expit(steepness * (P - 0.5))
```

**Pros**:
- Differentiable for uncertainty propagation
- No artificial plateaus
- Smooth transition from data to boundary

**Cons**:
- Introduces arbitrary `steepness` parameter
- Still distorts data near boundaries
- Doesn't address root cause (boundary data itself)

**When to use**: Bootstrap/jackknife resampling, gradient-based optimization

---

### Method 2: Weighted Fitting

**Concept**: Downweight points near boundaries instead of clipping

```python
def fit_logit_weighted(K, P, power=0.5):
    """
    Weighted least squares with boundary downweighting.

    Weight: w(P) = (P * (1-P))^power
    - P near 0 or 1: w → 0 (low weight)
    - P = 0.5: w = 0.25^power (maximum weight)

    Transition region naturally dominates fit.
    """
    # Clip for transform (still needed)
    P_clipped = np.clip(P, 1e-6, 1-1e-6)
    logit_P = np.log(P_clipped / (1 - P_clipped))

    # Compute weights
    weights = (P * (1 - P)) ** power
    weights = np.maximum(weights, 1e-3)  # Floor to avoid zero weight

    # Weighted linear regression
    from scipy.optimize import curve_fit
    def linear(x, slope, intercept):
        return slope * x + intercept

    popt, pcov = curve_fit(linear, K, logit_P, sigma=1/np.sqrt(weights))
    slope, intercept = popt

    # Extract Fermi-Dirac parameters
    Delta = -1.0 / slope
    K_c = -intercept / slope

    # Compute weighted R²
    predictions = linear(K, slope, intercept)
    ss_res = np.sum(weights * (logit_P - predictions)**2)
    ss_tot = np.sum(weights * (logit_P - np.average(logit_P, weights=weights))**2)
    R2_weighted = 1 - ss_res / ss_tot

    return {'K_c': K_c, 'Delta': Delta, 'R2': R2_weighted}
```

**Pros**:
- Uses all data points
- Automatic downweighting of unreliable boundaries
- Statistically principled (variance weighting)
- No arbitrary thresholds

**Cons**:
- Requires choosing `power` parameter
- Weighted R² not directly comparable to standard R²
- Boundary points still included (albeit with low weight)

**When to use**: Full data analysis, sensitivity studies

---

### Method 3: Transition-Region-Only Fitting (RECOMMENDED)

**Concept**: Explicitly exclude boundary plateaus, fit only transition region

```python
def fit_logit_transition_only(K, P, P_min=0.01, P_max=0.99):
    """
    Fit only data points in transition region P ∈ [P_min, P_max].

    Returns:
    - fit_params: {K_c, Delta, R², n_transition, K_range}
    - quality_flag: "good", "marginal", or "insufficient"
    """
    # Filter to transition region
    mask = (P > P_min) & (P < P_max)
    n_transition = np.sum(mask)

    if n_transition < 3:
        return None, "insufficient"

    K_trans = K[mask]
    P_trans = P[mask]

    # No clipping needed - all P ∈ (P_min, P_max)
    logit_P = np.log(P_trans / (1 - P_trans))

    # Standard linear regression
    from scipy.stats import linregress
    result = linregress(K_trans, logit_P)

    slope = result.slope
    intercept = result.intercept
    R2 = result.rvalue**2

    # Extract Fermi-Dirac parameters
    Delta = -1.0 / slope
    K_c = -intercept / slope

    # Quality assessment
    if n_transition >= 10:
        quality = "good"
    elif n_transition >= 5:
        quality = "marginal"
    else:
        quality = "insufficient"

    return {
        'K_c': K_c,
        'Delta': Delta,
        'slope': slope,
        'intercept': intercept,
        'R2': R2,
        'n_transition': n_transition,
        'K_range': (K_trans.min(), K_trans.max()),
        'P_bounds': (P_min, P_max)
    }, quality
```

**Pros**:
- **Clean fits**: Only transition data, no plateaus
- **Honest R²**: Reflects true fit quality in relevant region
- **Explicit quality metrics**: N_transition reported
- **No arbitrary parameters**: Boundaries chosen physically (e.g., P ∈ [0.01, 0.99])
- **Interpretable**: "Fit based on 12 transition points from K=8 to K=18"

**Cons**:
- **Discards data**: Boundary points excluded
- **May have few points**: N_transition can be small if transition is sharp
- **Requires sufficient sampling**: Need enough K values in transition region

**When to use**: **PRIMARY METHOD for publication-quality analysis**

---

### Method 4: Complementary Log-Log Transform

**Concept**: Alternative linearization for asymmetric transitions

```python
def cloglog_transform(P):
    """
    Complementary log-log: cloglog(P) = log(-log(1-P))

    Suitable for Gompertz-like models: P = exp(-exp(-α(K - K_c)))
    """
    P_clipped = np.clip(P, 1e-6, 1-1e-12)  # Different clip for 1-P
    return np.log(-np.log(1 - P_clipped))
```

**When to use**: If moment criterion shows asymmetric decay

---

### Method 5: Probit Transform (Bounded Tails)

**Concept**: Use inverse normal CDF instead of logit

```python
from scipy.stats import norm

def probit_transform(P):
    """
    Probit: probit(P) = Φ^(-1)(P) where Φ is normal CDF.

    Properties:
    - P=0 → -∞ (but smoother tail than logit)
    - P=1 → +∞ (but smoother tail than logit)
    - Standard in dose-response analysis
    """
    P_clipped = np.clip(P, 1e-10, 1-1e-10)
    return norm.ppf(P_clipped)
```

**When to use**: If Gaussian cumulative hypothesis is more appropriate than logistic

---

## 6. Recommended Methodology

### For Publication-Quality Analysis

#### Step 1: Identify Transition Region

```python
# Criterion-specific boundaries
TRANSITION_BOUNDS = {
    'moment': (0.01, 0.95),     # Plateau at high P, exclude it
    'spectral': (0.01, 0.99),   # Symmetric transition
    'krylov': (0.01, 0.99),     # Step-like but use same bounds
}
```

#### Step 2: Fit Transition Region Only

```python
for d in dimensions:
    K = data['K']
    P = data['P']

    P_min, P_max = TRANSITION_BOUNDS[criterion]

    fit_result, quality = fit_logit_transition_only(K, P, P_min, P_max)

    if quality == "insufficient":
        print(f"WARNING: d={d} has only {fit_result['n_transition']} transition points")
        print(f"         Recommend denser K sampling near K_c ~ {estimate_K_c(K, P)}")
```

#### Step 3: Report Quality Metrics

Include in figure annotations:

```
d=26: K_c = 24.9 ± 0.8, Δ = 1.27 ± 0.15
      R² = 0.94 (N_trans = 12, K ∈ [18, 30])
```

#### Step 4: Visualize Excluded Regions

```python
# Plot all data with transparency
ax.plot(K, logit(P), 'o', alpha=0.2, color='gray', label='Excluded')

# Highlight transition region
mask = (P > P_min) & (P < P_max)
ax.plot(K[mask], logit(P[mask]), 'o', alpha=0.8, color=color, label=f'd={d}')

# Shade excluded regions
ax.axhline(logit(P_min), color='red', ls=':', alpha=0.3, label=f'P={P_min}')
ax.axhline(logit(P_max), color='red', ls=':', alpha=0.3, label=f'P={P_max}')
ax.fill_between([K.min(), K.max()], logit(P_max), 15, alpha=0.1, color='red')
ax.fill_between([K.min(), K.max()], -15, logit(P_min), alpha=0.1, color='red')
```

### Quality Thresholds

| N_transition | Quality | Recommendation |
|--------------|---------|----------------|
| ≥ 10 | **Good** | Proceed with fit |
| 5-9 | **Marginal** | Report with caveat |
| < 5 | **Insufficient** | Collect more data or flag as unreliable |

---

## 7. Implementation Guide

### For Moment Criterion (Exponential)

```python
# Transform: log(P) = -α·K + α·K_c
# Only fit P < 0.95 to avoid plateau

mask = P < 0.95
K_fit = K[mask]
P_fit = P[mask]

log_P = np.log(P_fit)  # No clipping needed if P < 0.95

slope, intercept, r, p, stderr = linregress(K_fit, log_P)

alpha = -slope
K_c = -intercept / slope
```

### For Spectral/Krylov (Fermi-Dirac)

```python
# Transform: logit(P) = -K/Δ + K_c/Δ
# Fit P ∈ [0.01, 0.99]

P_min, P_max = 0.01, 0.99
mask = (P > P_min) & (P < P_max)

K_fit = K[mask]
P_fit = P[mask]

logit_P = np.log(P_fit / (1 - P_fit))  # No clipping needed

slope, intercept, r, p, stderr = linregress(K_fit, logit_P)

Delta = -1.0 / slope
K_c = -intercept / slope
```

### Uncertainty Propagation

```python
# Use scipy.linregress standard errors
slope_err = stderr_slope
intercept_err = stderr_intercept

# Propagate to Delta and K_c (first-order)
Delta_err = slope_err / slope**2
K_c_err = np.sqrt((intercept_err/slope)**2 + (intercept*slope_err/slope**2)**2)
```

---

## 8. Quality Metrics

### Data Quality Assessment

For each (d, τ) combination, report:

1. **N_total**: Total number of K values measured
2. **N_transition**: Number of K values in transition region
3. **K_range**: Range of K values used in fit
4. **P_range**: Actual P values in transition region
5. **R²**: Coefficient of determination
6. **Residual analysis**: Plot residuals vs K

### Example Quality Report

```
Krylov Criterion (τ=0.99) - Data Quality Summary
================================================================
d    N_total  N_trans  K_range      P_range         R²     Flag
----------------------------------------------------------------
10   28       5        [8, 12]      [0.02, 0.98]   0.89   marginal
14   28       8        [11, 18]     [0.01, 0.99]   0.92   good
18   28       12       [14, 25]     [0.01, 0.98]   0.94   good
22   28       10       [18, 28]     [0.02, 0.97]   0.93   good
26   28       11       [22, 32]     [0.01, 0.99]   0.95   good
================================================================

Recommendations:
- d=10: Marginal fit (N_trans=5). Consider denser K sampling near K=10.
- All other dimensions: Good quality fits (N_trans ≥ 8).
```

### Residual Diagnostics

```python
def plot_residuals(K_fit, logit_P_fit, logit_P_pred):
    """
    Plot residuals to check for systematic deviations.

    Good fit: residuals scattered randomly around zero
    Bad fit: systematic curvature, trends, or heteroscedasticity
    """
    residuals = logit_P_fit - logit_P_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs K
    ax1.plot(K_fit, residuals, 'o')
    ax1.axhline(0, color='red', ls='--')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs K (check for trends)')

    # Histogram of residuals
    ax2.hist(residuals, bins=10, edgecolor='black')
    ax2.axvline(0, color='red', ls='--')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residual distribution (check for normality)')

    plt.tight_layout()
    return residuals
```

---

## Summary of Recommendations

### For Current Analysis (Quick Fix)

1. ✅ **Already done**: Created `plot_linearized_fits.py` with hard clipping
2. ⚠️ **Acknowledge limitations**: Note that R² includes boundary plateaus
3. 📊 **Report N_transition**: Add to figure annotations

### For Publication (Rigorous Analysis)

1. **Implement transition-region-only fitting**
2. **Generate quality report** with N_transition for all (d, τ)
3. **Flag insufficient data** and recommend additional experiments
4. **Plot residuals** to validate linearity assumption
5. **Report uncertainties** on K_c and Δ from regression standard errors

### For Future Data Collection

1. **Adaptive K sampling**: Place more K values near estimated K_c
2. **Target transition region**: Ensure N_transition ≥ 10 for all dimensions
3. **Use K spacing ~ Δ/2**: Capture transition with 4-5 points per width

---

## Appendix: Mathematical Details

### Logit Transform Properties

For Fermi-Dirac model:
```
P(K) = 1 / (1 + exp((K - K_c)/Δ))

Rearranging:
1/P - 1 = exp((K - K_c)/Δ)
log(1/P - 1) = (K - K_c)/Δ
log((1-P)/P) = (K - K_c)/Δ
-log(P/(1-P)) = (K - K_c)/Δ

Therefore:
logit(P) = log(P/(1-P)) = -(K - K_c)/Δ = -K/Δ + K_c/Δ

Linear form: y = mx + b
where:
  y = logit(P)
  x = K
  m = -1/Δ  (slope)
  b = K_c/Δ  (intercept)

Solving for parameters:
  Δ = -1/m
  K_c = -b/m = b*Δ
```

### Binomial Confidence Intervals

For P = k/n, the Wilson score interval (recommended over Wald for boundary cases):

```python
from scipy.stats import norm

def wilson_interval(k, n, alpha=0.05):
    """
    Wilson score confidence interval for binomial proportion.

    Better than normal approximation for small n or extreme p.
    """
    z = norm.ppf(1 - alpha/2)
    p_hat = k / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator

    lower = max(0, center - margin)
    upper = min(1, center + margin)

    return lower, upper
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-15
**Author**: Analysis of `reach` package pipeline
**Status**: Recommended methodology for linearized fits analysis
