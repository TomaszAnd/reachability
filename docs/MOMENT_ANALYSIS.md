# Moment Criterion Non-Monotonicity Analysis

## Mathematical Definition

The Moment criterion tests whether the matrix

    M(gamma) = Q + gamma * L * L^T

is positive (or negative) definite for gamma = +/-1000, where:

- **L[k]** = <H_k>_psi - <H_k>_phi  (first moment difference)
- **Q[k,m]** = <{H_k, H_m}>_psi - <{H_k, H_m}>_phi  (second moment anticommutator difference)

If M(gamma) is definite for some gamma, the target state psi is certified UNREACHABLE.
Otherwise, the criterion returns INCONCLUSIVE.

## Why Non-Monotonicity Occurs

### Spectral/Krylov: Guaranteed Monotonic

For Spectral and Krylov criteria, increasing K (adding operators) expands the
feasible set of lambda vectors. The maximum score can only increase or stay the same:

    max_{lambda in R^{K+1}} S(lambda) >= max_{lambda in R^K} S(lambda)

This guarantees P(unreachable) is monotonically decreasing in K.

### Moment: NOT Guaranteed Monotonic

Adding operator H_{K+1} changes both Q and L:

- Q grows from K x K to (K+1) x (K+1) with new cross-terms
- L grows from K to K+1 with a new element

The positive definiteness of M(gamma) = Q + gamma*LL^T depends on the
**eigenvalue structure** of the combined matrix, which can change unpredictably:

1. New cross-terms in Q may introduce negative eigenvalues
2. The new L element changes the rank-1 perturbation gamma*LL^T
3. The projection onto ker(L) changes, affecting the effective Q

## Empirical Evidence

### Paired Test (d=16, 1000 trials, K is subset of K+1)

**888 paired violations**: K finds UNREACHABLE certificate but K+1 fails.

This is not noise — the same operators, same target states, just more operators
in the K+1 case cause the certificate to fail.

### Independent Test (d=16, 1000 trials per K)

Non-monotonic behavior clearly visible:
```
K=6:  P=0.320    K=8:  P=0.442  (P increases with K!)
K=10: P=0.121    K=12: P=0.201  (P increases with K!)
```

### Tolerance/Gamma Insensitivity

All tolerance/gamma configurations give **identical** results:
- tol = 1e-8, 1e-10, 1e-12: same P at every K
- gamma = 1e3, 1e4, 1e6: same P at every K

This rules out numerical artifacts.

### Old vs Current Code

The Moment implementation on `clean` and `fast` branches is **identical**:
same Q formula, same L formula, same gamma=+/-1000, same tol=1e-10.
Non-monotonicity was always present but not previously investigated.

## Comparison of Criteria

| Property | Spectral/Krylov | Moment |
|----------|-----------------|--------|
| What it tests | max_lambda score(lambda) >= tau | Exists gamma: M(gamma) definite |
| Type of condition | Necessary (score < tau -> unreachable) | Sufficient (definite -> unreachable) |
| Monotonicity in K | Guaranteed | NOT guaranteed |
| Why | Larger feasible set can only help | Adding operators changes Q,L structure |

## Implications for Analysis

1. **Do not fit Moment data with monotonic functions** (exponential decay, Fermi-Dirac).
   The underlying behavior is not monotonic.

2. **Use Spectral/Krylov for quantitative phase transition analysis**.
   Only these criteria have the mathematical guarantee needed for curve fitting.

3. **Moment is still useful as a fast screening tool**.
   At each K independently, it correctly identifies a subset of unreachable states.
   The non-monotonicity affects cross-K comparisons, not individual K evaluations.

4. **Error bars on Moment P are not the primary source of fluctuation**.
   Even with 1000 trials and zero SEM, the non-monotonicity persists.
