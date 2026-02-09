# [[5,1,3]] Perfect Code: Floquet Driving Discovery Report

Generated: 2026-01-09T20:20:14.045947


## Overview

This report documents the discovery of optimal Floquet driving frequencies
for preparing the logical |0_L> state of the [[5,1,3]] perfect code.


## System Configuration

**Ring geometry (5 qubits):**
```
    1 --- 2
   /       \
  5         3
   \       /
    4 ----
```


**XY Coupling Operators:**
- H_12
- H_23
- H_34
- H_45
- H_51

## Commutator Analysis

**Large commutator pairs (||[H_j, H_k]||_F > 10% of max):**

| Pair | ||[H_j, H_k]||_F |
|------|-----------------|
| H_12 - H_23 | 16.00 |
| H_12 - H_51 | 16.00 |
| H_23 - H_34 | 16.00 |
| H_34 - H_45 | 16.00 |
| H_45 - H_51 | 16.00 |

## Discovered Frequency Assignment

Using graph coloring on the conflict graph (edges = large commutator pairs):

**Frequency 1ω:** H_12, H_34
**Frequency 2ω:** H_23, H_45
**Frequency 3ω:** H_51

## Validation

Fourier overlap |F_jk| for large commutator pairs:

| Pair | ||[H_j,H_k]|| | Freq j | Freq k | |F_jk| | Status |
|------|-------------|--------|--------|-------|--------|
| H_12-H_23 | 16.00 | 1ω | 2ω | 0.0000 | OK |
| H_12-H_51 | 16.00 | 1ω | 3ω | 0.0000 | OK |
| H_23-H_34 | 16.00 | 2ω | 1ω | 0.0000 | OK |
| H_34-H_45 | 16.00 | 1ω | 2ω | 0.0000 | OK |
| H_45-H_51 | 16.00 | 2ω | 3ω | 0.0000 | OK |

## Comparison with Toric Plaquette

| Property | Toric (4 qubits) | [[5,1,3]] (5 qubits) |
|----------|------------------|----------------------|
| XY Operators | 3 | 5 |
| Large commutator pairs | 2 | 5 |
| Frequency groups needed | 2 | 3 |

## Conclusion

The discovered frequency assignment successfully minimizes Fourier overlap
for all large commutator pairs, validating the graph coloring approach.
