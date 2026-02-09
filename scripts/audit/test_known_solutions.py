#!/usr/bin/env python3
"""
Test Suite: Analytically Solvable Cases

Tests reachability criteria against problems with known analytical solutions.
These tests verify correctness without relying on numerical optimization accuracy.

Test Cases:
A. Single qubit rotation (σx): |0⟩ → |1⟩ is reachable
B. Two-qubit Bell state preparation
C. Unreachable state (orthogonal subspace)
D. Exact evolution targets
E. Identity Hamiltonian (trivial case)
F. Diagonal Hamiltonian (no mixing)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import qutip
from scipy.linalg import expm
from reach import mathematics, optimize, models


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# =============================================================================
# TEST A: Single Qubit Rotation
# =============================================================================

def test_single_qubit_rotation():
    """
    Verify |0⟩ → |1⟩ is reachable via σx.

    Mathematical basis:
    - H = σx has eigenstates |+⟩ = (|0⟩+|1⟩)/√2, |-⟩ = (|0⟩-|1⟩)/√2
    - |0⟩ = (|+⟩ + |-⟩)/√2, |1⟩ = (|+⟩ - |-⟩)/√2
    - Spectral overlap: S = |⟨+|1⟩*⟨+|0⟩| + |⟨-|1⟩*⟨-|0⟩|
                         = |1/2| + |-1/2| = 1.0
    - Evolution: exp(-iσx·t)|0⟩ = cos(t)|0⟩ - i·sin(t)|1⟩
    - At t=π/2: reaches -i|1⟩ (|1⟩ with global phase)
    """
    banner("TEST A: Single Qubit Rotation (σx)")

    psi = qutip.basis(2, 0)  # |0⟩
    phi = qutip.basis(2, 1)  # |1⟩
    H = qutip.sigmax()
    hams = [H]

    # Test 1: Spectral overlap at λ=1 should be exactly 1.0
    S = mathematics.spectral_overlap(np.array([1.0]), psi, phi, hams)
    print(f"\n  Test 1: S(|0⟩→|1⟩, σx, λ=1) = {S:.10f}")
    print(f"  Expected: 1.0")
    test1_pass = abs(S - 1.0) < 1e-10
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: Spectral overlap at λ=0.5 should also be 1.0 (eigenstates don't change)
    S_half = mathematics.spectral_overlap(np.array([0.5]), psi, phi, hams)
    print(f"\n  Test 2: S(λ=0.5) = {S_half:.10f}")
    print(f"  Expected: 1.0 (eigenstates independent of λ scale)")
    test2_pass = abs(S_half - 1.0) < 1e-10
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: Krylov score should be 1.0 (|1⟩ ∈ span{|0⟩, σx|0⟩} = span{|0⟩, |1⟩})
    R = mathematics.krylov_score(np.array([1.0]), psi, phi, hams)
    print(f"\n  Test 3: R(|0⟩→|1⟩, σx) = {R:.10f}")
    print(f"  Expected: 1.0 (Krylov space = full space)")
    test3_pass = abs(R - 1.0) < 1e-10
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    # Test 4: With K=2, optimizer should find S* = 1.0
    # Note: Optimizer requires K >= 2, so we add σz
    sy = qutip.sigmay()
    hams2 = [H, sy]
    result = optimize.maximize_spectral_overlap(psi, phi, hams2, maxiter=50, restarts=1)
    print(f"\n  Test 4: max_λ S(λ) with [σx, σy] = {result['best_value']:.10f}")
    print(f"  Expected: 1.0")
    test4_pass = result['best_value'] > 0.999
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    return test1_pass and test2_pass and test3_pass and test4_pass


# =============================================================================
# TEST B: Two-Qubit Bell State
# =============================================================================

def test_two_qubit_bell_state():
    """
    Verify Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 is reachable from |00⟩.

    Mathematical basis:
    - Bell state is entangled: cannot be reached by local operations alone
    - Krylov criterion is more sensitive than spectral for some Hamiltonians
    - This test verifies: Krylov detects reachability when spectral is conservative
    """
    banner("TEST B: Two-Qubit Bell State")

    # States
    psi = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))  # |00⟩
    bell = (qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0)) +
            qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))).unit()  # (|00⟩+|11⟩)/√2

    # Hamiltonians
    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    I2 = qutip.qeye(2)

    H1 = qutip.tensor(sx, I2)   # σx ⊗ I
    H2 = qutip.tensor(I2, sx)   # I ⊗ σx
    H3 = qutip.tensor(sz, sz)   # σz ⊗ σz (interaction)
    hams = [H1, H2, H3]

    # Test 1: Spectral overlap (may be conservative)
    result_s = optimize.maximize_spectral_overlap(
        psi, bell, hams, maxiter=100, restarts=3, seed=42
    )
    print(f"\n  Test 1: max_λ S(λ) = {result_s['best_value']:.6f}")
    print(f"  Note: S* can be < 1 even for reachable states (conservative criterion)")
    test1_pass = True  # We just observe the value, not a strict pass/fail
    print(f"  Result: OBSERVED")

    # Test 2: Krylov score should be ~1.0 (Bell IS reachable)
    result_k = optimize.maximize_krylov_score(
        psi, bell, hams, maxiter=100, restarts=3, seed=42
    )
    print(f"\n  Test 2: max_λ R(λ) = {result_k['best_value']:.6f}")
    print(f"  Expected: ~1.0 (Bell state in Krylov space)")
    test2_pass = result_k['best_value'] > 0.99
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: Criterion ordering: Krylov should detect reachability even if spectral doesn't
    print(f"\n  Test 3: Criterion ordering")
    print(f"  S* = {result_s['best_value']:.4f}, R* = {result_k['best_value']:.4f}")
    test3_pass = result_k['best_value'] >= result_s['best_value'] - 0.01
    print(f"  Expected: R* >= S* (Krylov is stronger criterion)")
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    # Test 4: Without interaction term, local ops don't span full space
    hams_local = [H1, H2]  # Only local operations
    result_local_k = optimize.maximize_krylov_score(
        psi, bell, hams_local, maxiter=100, restarts=3, seed=42
    )
    print(f"\n  Test 4: R* with local ops only = {result_local_k['best_value']:.6f}")
    print(f"  Expected: < 1.0 (Bell state requires entangling operation)")
    test4_pass = result_local_k['best_value'] < 1.0
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    return test1_pass and test2_pass and test3_pass and test4_pass


# =============================================================================
# TEST C: Unreachable State (Orthogonal Subspace)
# =============================================================================

def test_unreachable_orthogonal():
    """
    Verify unreachable state detection when target is in orthogonal subspace.

    Mathematical basis:
    - H = diag(1, 2, 0, 0) acts only on first two basis states
    - |ψ⟩ = |0⟩ is in support of H
    - |φ⟩ = |2⟩ is outside support of H
    - Krylov space K(H, |0⟩) = span{|0⟩, |1⟩} (2D, not full)
    - |2⟩ ⊥ K(H, |0⟩), so R* = 0
    """
    banner("TEST C: Unreachable State (Orthogonal Subspace)")

    d = 4
    psi = qutip.basis(d, 0)  # |0⟩
    phi = qutip.basis(d, 2)  # |2⟩ (orthogonal to Krylov space)

    # H only mixes |0⟩ and |1⟩
    H_mat = np.diag([1.0, 2.0, 0.0, 0.0])
    H_mat[0, 1] = H_mat[1, 0] = 0.5  # Off-diagonal coupling
    H = qutip.Qobj(H_mat)
    hams = [H, H]  # Duplicate for K>=2 requirement

    # Test 1: Krylov score should be ~0 (|2⟩ not in Krylov space)
    R = mathematics.krylov_score(np.array([1.0, 0.0]), psi, phi, hams)
    print(f"\n  Test 1: R(λ=1) = {R:.10f}")
    print(f"  Expected: 0.0 (|2⟩ orthogonal to Krylov space)")
    test1_pass = R < 0.01
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: Krylov dimension should be 2 (not full d=4)
    V = mathematics.krylov_basis(H, psi, m=d)
    krylov_dim = V.shape[1]
    print(f"\n  Test 2: Krylov dimension = {krylov_dim}")
    print(f"  Expected: 2 (H mixes only |0⟩ and |1⟩)")
    test2_pass = krylov_dim == 2
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: Spectral overlap should also be low
    S = mathematics.spectral_overlap(np.array([1.0, 0.0]), psi, phi, hams)
    print(f"\n  Test 3: S(λ=1) = {S:.6f}")
    print(f"  Expected: < 1.0 (limited mixing)")
    test3_pass = S < 0.5
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    # Test 4: is_unreachable_krylov should return True
    unreachable = mathematics.is_unreachable_krylov(H, psi, phi, m=d)
    print(f"\n  Test 4: is_unreachable_krylov = {unreachable}")
    print(f"  Expected: True")
    test4_pass = unreachable == True
    print(f"  Result: {'PASS' if test4_pass else 'FAIL'}")

    return test1_pass and test2_pass and test3_pass and test4_pass


# =============================================================================
# TEST D: Exact Evolution Target
# =============================================================================

def test_exact_evolution():
    """
    Verify φ = exp(-iHt)|ψ⟩ is always classified as reachable.

    Mathematical basis:
    - By construction, φ is exactly reachable from ψ
    - S* = R* = 1.0 (up to numerical precision)
    - This tests that optimization finds the true λ
    """
    banner("TEST D: Exact Evolution Target")

    configs = [
        ("d=8, K=4, GUE", 8, 4, "GUE", {}),
        ("d=16, K=8, canonical", 16, 8, "canonical", {}),
        ("d=16, K=8, GEO2", 16, 8, "GEO2", {'nx': 2, 'ny': 2}),
    ]

    all_pass = True

    for label, d, K, ensemble, params in configs:
        print(f"\n  {label}:")

        n_trials = 10
        tau = 0.99
        false_unreach_s = 0
        false_unreach_k = 0

        for trial in range(n_trials):
            seed = 5000 + trial
            rng = np.random.RandomState(seed)

            psi = models.fock_state(d, 0)
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)

            # Generate provably reachable target
            lambdas_true = rng.uniform(-1, 1, K)
            t = 1.0
            H_lambda = sum(l * H.full() for l, H in zip(lambdas_true, hams))
            U = expm(-1j * H_lambda * t)
            phi_arr = U @ psi.full().flatten()
            phi = qutip.Qobj(phi_arr.reshape(-1, 1)).unit()

            # Optimize
            res_s = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=100, restarts=3, seed=seed
            )
            res_k = optimize.maximize_krylov_score(
                psi, phi, hams, maxiter=100, restarts=3, seed=seed
            )

            if res_s['best_value'] < tau:
                false_unreach_s += 1
            if res_k['best_value'] < tau:
                false_unreach_k += 1

        print(f"    Spectral false-unreachable: {false_unreach_s}/{n_trials}")
        print(f"    Krylov false-unreachable: {false_unreach_k}/{n_trials}")

        passed = false_unreach_s == 0 and false_unreach_k == 0
        print(f"    Result: {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_pass = False

    return all_pass


# =============================================================================
# TEST E: Identity Hamiltonian
# =============================================================================

def test_identity_hamiltonian():
    """
    Test with H = I (identity).

    Mathematical basis:
    - exp(-iλI·t) = exp(-iλt)·I (global phase only)
    - Cannot reach any state different from ψ
    - S(λ) = |⟨φ|ψ⟩| (independent of λ)
    """
    banner("TEST E: Identity Hamiltonian")

    d = 4
    psi = qutip.basis(d, 0)
    phi = qutip.basis(d, 1)  # Orthogonal to psi

    H = qutip.qeye(d)  # Identity
    hams = [H, H]  # Need K>=2 for optimizer, but spectral_overlap works with K=1

    # Test 1: Spectral overlap = |⟨φ|ψ⟩| = 0 for orthogonal states
    S = mathematics.spectral_overlap(np.array([1.0, 0.0]), psi, phi, hams)
    print(f"\n  Test 1: S(|0⟩→|1⟩, I) = {S:.10f}")
    print(f"  Expected: 0.0 (identity has trivial eigenbasis)")

    # Note: Identity eigenspace is degenerate - any basis works
    # With standard basis, S = |⟨1|0⟩| = 0
    test1_pass = S < 0.01
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: Same state should give S = 1
    phi_same = psi
    S_same = mathematics.spectral_overlap(np.array([1.0, 0.0]), psi, phi_same, hams)
    print(f"\n  Test 2: S(|0⟩→|0⟩, I) = {S_same:.10f}")
    print(f"  Expected: 1.0")
    test2_pass = abs(S_same - 1.0) < 1e-10
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    return test1_pass and test2_pass


# =============================================================================
# TEST F: Diagonal Hamiltonian
# =============================================================================

def test_diagonal_hamiltonian():
    """
    Test with diagonal H = diag(E_0, E_1, ..., E_{d-1}).

    Mathematical basis:
    - Eigenstates are computational basis states |n⟩
    - S(λ) = Σₙ |⟨φ|n⟩*⟨n|ψ⟩|
    - No mixing between basis states
    """
    banner("TEST F: Diagonal Hamiltonian")

    d = 4
    psi = qutip.basis(d, 0)

    # Diagonal Hamiltonian with distinct eigenvalues
    H_mat = np.diag([1.0, 2.0, 3.0, 4.0])
    H = qutip.Qobj(H_mat)
    hams = [H, H]  # Duplicate for consistent K

    # Test 1: |0⟩ → |0⟩ should give S = 1
    phi = qutip.basis(d, 0)
    S = mathematics.spectral_overlap(np.array([1.0, 0.0]), psi, phi, hams)
    print(f"\n  Test 1: S(|0⟩→|0⟩) = {S:.10f}")
    print(f"  Expected: 1.0")
    test1_pass = abs(S - 1.0) < 1e-10
    print(f"  Result: {'PASS' if test1_pass else 'FAIL'}")

    # Test 2: |0⟩ → |1⟩ should give S = 0 (orthogonal eigenstates)
    phi = qutip.basis(d, 1)
    S = mathematics.spectral_overlap(np.array([1.0, 0.0]), psi, phi, hams)
    print(f"\n  Test 2: S(|0⟩→|1⟩) = {S:.10f}")
    print(f"  Expected: 0.0 (no mixing)")
    test2_pass = S < 1e-10
    print(f"  Result: {'PASS' if test2_pass else 'FAIL'}")

    # Test 3: |0⟩ → superposition (|0⟩+|1⟩)/√2
    phi = (qutip.basis(d, 0) + qutip.basis(d, 1)).unit()
    S = mathematics.spectral_overlap(np.array([1.0, 0.0]), psi, phi, hams)
    print(f"\n  Test 3: S(|0⟩→(|0⟩+|1⟩)/√2) = {S:.10f}")
    print(f"  Expected: 1/√2 ≈ 0.707")
    test3_pass = abs(S - 1/np.sqrt(2)) < 1e-6
    print(f"  Result: {'PASS' if test3_pass else 'FAIL'}")

    return test1_pass and test2_pass and test3_pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("ANALYTICALLY SOLVABLE TEST SUITE")
    print("Testing reachability criteria against known solutions")

    results = {}

    results['single_qubit'] = test_single_qubit_rotation()
    results['bell_state'] = test_two_qubit_bell_state()
    results['unreachable'] = test_unreachable_orthogonal()
    results['exact_evolution'] = test_exact_evolution()
    results['identity'] = test_identity_hamiltonian()
    results['diagonal'] = test_diagonal_hamiltonian()

    # Summary
    banner("TEST SUMMARY")

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_pass = False

    print(f"\n{'='*70}")
    print(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
