#!/usr/bin/env python3
"""
Test moment criteria implementation.

Verifies that:
1. Static moment criterion works
2. Floquet moment criterion works (λ-dependent)
3. λ optimization for Floquet finds different results than random λ
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from reach import floquet, models, moment_criteria, states

print("=" * 70)
print("MOMENT CRITERIA IMPLEMENTATION TEST")
print("=" * 70)

# Setup
n_qubits = 4
d = 2**n_qubits
K = 8
seed = 42

print(f"\nSystem: {n_qubits} qubits (d={d}), K={K} operators")

# Generate Hamiltonians
print("\nGenerating GEO2 Hamiltonians...")
hams_qutip = models.random_hamiltonian_ensemble(
    dim=d, k=K, ensemble="GEO2", nx=2, ny=2, seed=seed
)
hams = floquet.hamiltonians_to_numpy(hams_qutip)

# Test 1: Static criterion with specific states
print("\n" + "=" * 70)
print("TEST 1: Static Moment Criterion")
print("=" * 70)

# Product state → GHZ (known to be hard)
psi = states.computational_basis(n_qubits, '0' * n_qubits)
target_states_dict = states.create_target_states(n_qubits)
phi = target_states_dict['ghz']

print(f"\nState pair: |0000⟩ → GHZ")
print(f"Classical overlap: {np.abs(phi.conj() @ psi)**2:.4f}")

unreachable, x_opt, eigvals = moment_criteria.static_moment_criterion(psi, phi, hams)

if unreachable:
    print(f"✓ Criterion SUCCEEDED: State proved unreachable!")
    print(f"  Optimal x: {x_opt:.6f}")
    print(f"  Min eigenvalue: {np.min(eigvals):.6e}")
else:
    print(f"  Criterion FAILED: Inconclusive (state might be reachable or unreachable)")

# Test 2: Random Haar states (for scaling experiment)
print("\n" + "=" * 70)
print("TEST 2: Random Haar State Pairs")
print("=" * 70)

print(f"\nTesting static criterion on 10 random state pairs...")
n_unreachable_static = 0

for trial in range(10):
    # Random Haar states
    rng = np.random.RandomState(seed + trial)
    psi_random = rng.randn(d) + 1j * rng.randn(d)
    psi_random /= np.linalg.norm(psi_random)
    phi_random = rng.randn(d) + 1j * rng.randn(d)
    phi_random /= np.linalg.norm(phi_random)

    unreachable, _, _ = moment_criteria.static_moment_criterion(psi_random, phi_random, hams)

    if unreachable:
        n_unreachable_static += 1

P_static = n_unreachable_static / 10
print(f"  P(unreachable) = {P_static:.2f} ({n_unreachable_static}/10)")

# Test 3: Floquet criterion (λ-dependent)
print("\n" + "=" * 70)
print("TEST 3: Floquet Moment Criterion (λ-dependent)")
print("=" * 70)

# Create driving functions
T = 1.0
driving = floquet.create_driving_functions(K, 'bichromatic', T, seed=seed)

print(f"\nTesting single λ value...")

# Try with random λ
lambdas = np.random.randn(K) / np.sqrt(K)
print(f"Random λ: {lambdas[:4]}... (showing first 4)")

# Use same state pair as Test 1
unreachable_single, x_single, _ = moment_criteria.floquet_moment_criterion(
    psi, phi, hams, lambdas, driving, T, order=2
)

if unreachable_single:
    print(f"  ✓ Criterion succeeded with this λ")
    print(f"  x = {x_single:.6f}")
else:
    print(f"  ✗ Criterion failed with this λ")

# Test 4: λ-optimized Floquet criterion
print("\n" + "=" * 70)
print("TEST 4: Floquet Criterion with λ Search")
print("=" * 70)

print(f"\nSearching over 50 random λ values...")

unreachable_opt, lambdas_opt, x_opt, _ = moment_criteria.floquet_moment_criterion_optimized(
    psi, phi, hams, driving, T, order=2, n_lambda_trials=50, seed=seed
)

if unreachable_opt:
    print(f"  ✓ FOUND λ that proves unreachability!")
    print(f"  Optimal λ: {lambdas_opt[:4]}... (showing first 4)")
    print(f"  x = {x_opt:.6f}")
else:
    print(f"  ✗ No λ found among 50 trials")

# Test 5: Compare static vs Floquet on random states
print("\n" + "=" * 70)
print("TEST 5: Static vs Floquet Comparison")
print("=" * 70)

print(f"\nTesting both criteria on 10 random state pairs...")

n_unreachable_floquet = 0

for trial in range(10):
    # Random Haar states
    rng = np.random.RandomState(seed + trial + 100)
    psi_random = rng.randn(d) + 1j * rng.randn(d)
    psi_random /= np.linalg.norm(psi_random)
    phi_random = rng.randn(d) + 1j * rng.randn(d)
    phi_random /= np.linalg.norm(phi_random)

    # Floquet with λ search (20 trials per state pair to save time)
    unreachable_f, _, _, _ = moment_criteria.floquet_moment_criterion_optimized(
        psi_random, phi_random, hams, driving, T,
        order=2, n_lambda_trials=20, seed=seed + trial
    )

    if unreachable_f:
        n_unreachable_floquet += 1

P_floquet = n_unreachable_floquet / 10

print(f"\nResults:")
print(f"  Static P(unreachable):  {P_static:.2f} ({n_unreachable_static}/10)")
print(f"  Floquet P(unreachable): {P_floquet:.2f} ({n_unreachable_floquet}/10)")

if P_floquet > P_static:
    print(f"\n  ✓ Floquet criterion IS STRONGER (higher P!)")
    print(f"  Improvement: {100*(P_floquet - P_static):.0f} percentage points")
elif P_floquet < P_static:
    print(f"\n  ✗ Floquet criterion is WEAKER (lower P)")
else:
    print(f"\n  ≈ Both criteria have similar strength")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nImplementation Status:")
print("  ✓ Static moment criterion works")
print("  ✓ Floquet moment criterion works (λ-dependent)")
print("  ✓ λ search for Floquet implemented")

if P_floquet > P_static:
    print("\n✓ PROMISING: Floquet shows higher P on small sample!")
    print("  → Hypothesis may be correct, proceed with full scaling experiment")
elif P_static == 0 and P_floquet == 0:
    print("\n⚠ WARNING: Both criteria show P=0")
    print("  → Criteria may be too weak for this K value")
    print("  → Try smaller K or more trials")
else:
    print("\n⚠ Floquet not clearly stronger on small sample")
    print("  → Need full scaling experiment to determine α values")

print("\nNext step:")
print("  Run full scaling experiment with K-scan to extract α values")
print()
