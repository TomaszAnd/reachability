#!/usr/bin/env python3
"""
Simple test script for continuous Krylov score implementation.

Tests:
1. Basic krylov_score() computation
2. Score bounds (R ∈ [0,1])
3. Relationship with residual: R = 1 - ε²_res
4. Optimization via maximize_krylov_score()
5. Comparison with binary criterion
"""

import numpy as np
import qutip

from reach import mathematics, models, optimize


def test_krylov_score_basic():
    """Test basic krylov_score computation."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic krylov_score computation")
    print("=" * 60)

    # Setup simple problem: d=4, k=2
    d, k = 4, 2
    hams = models.random_hamiltonian_ensemble(d, k, "GUE", seed=42)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=43)[0]

    # Test at random parameters
    lambdas = np.array([0.5, -0.3])
    score = mathematics.krylov_score(lambdas, psi, phi, hams, m=d)

    print(f"Dimension d={d}, K={k}")
    print(f"Parameters λ={lambdas}")
    print(f"Krylov score R(λ) = {score:.6f}")

    # Check bounds (with small tolerance for numerical precision)
    tol = 1e-10
    assert -tol <= score <= 1.0 + tol, f"Score {score} outside [0,1] (with tolerance)"
    print("✓ Score is in valid range [0,1]")

    return score


def test_score_residual_relationship():
    """Test that R(λ) = 1 - ε²_res."""
    print("\n" + "=" * 60)
    print("TEST 2: Score vs residual relationship")
    print("=" * 60)

    # Setup
    d, k = 6, 3
    hams = models.random_hamiltonian_ensemble(d, k, "GUE", seed=44)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=45)[0]

    lambdas = np.array([0.2, 0.7, -0.5])

    # Compute Krylov score
    score = mathematics.krylov_score(lambdas, psi, phi, hams, m=d)

    # Compute residual manually using krylov_basis
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))
    V = mathematics.krylov_basis(H_lambda, psi, d)
    phi_vec = phi.full().flatten()
    proj = V @ (V.conj().T @ phi_vec)
    resid = phi_vec - proj
    resid_norm = np.linalg.norm(resid)

    # Check relationship: R = 1 - ε²_res
    score_from_residual = 1.0 - resid_norm**2

    print(f"Krylov score R(λ) = {score:.10f}")
    print(f"From residual: 1 - ε² = {score_from_residual:.10f}")
    print(f"Residual norm ε = {resid_norm:.10f}")

    # Check agreement (within numerical tolerance)
    assert np.abs(score - score_from_residual) < 1e-9, "Score-residual mismatch"
    print("✓ Relationship R(λ) = 1 - ε²_res verified")

    return score, resid_norm


def test_optimize_krylov_score():
    """Test maximize_krylov_score optimization."""
    print("\n" + "=" * 60)
    print("TEST 3: Krylov score optimization")
    print("=" * 60)

    # Setup
    d, k = 6, 3
    hams = models.random_hamiltonian_ensemble(d, k, "GUE", seed=46)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=47)[0]

    # Optimize Krylov score
    result = optimize.maximize_krylov_score(
        psi, phi, hams, m=d, restarts=3, maxiter=100, seed=48
    )

    print(f"Optimization result:")
    print(f"  R* = {result['best_value']:.6f}")
    print(f"  λ* = {result['best_x']}")
    print(f"  Function evals: {result['nfev']}")
    print(f"  Success: {result['success']}")
    print(f"  Runtime: {result['runtime_s']:.3f}s")

    # Check that optimized score is better than random
    random_lambda = np.random.uniform(-1, 1, k)
    random_score = mathematics.krylov_score(random_lambda, psi, phi, hams, m=d)

    print(f"\nComparison:")
    print(f"  Optimized R* = {result['best_value']:.6f}")
    print(f"  Random R = {random_score:.6f}")

    assert result["best_value"] >= random_score - 1e-6, "Optimization didn't improve score"
    print("✓ Optimization found better or equal score")

    return result


def test_comparison_with_binary():
    """Compare continuous score with binary criterion."""
    print("\n" + "=" * 60)
    print("TEST 4: Continuous vs binary criterion")
    print("=" * 60)

    # Setup
    d, k = 8, 4
    hams = models.random_hamiltonian_ensemble(d, k, "GUE", seed=49)
    psi = models.fock_state(d, 0)

    # Test on multiple target states
    n_tests = 5
    targets = models.random_states(n_tests, d, seed=50)

    print(f"Testing on {n_tests} random target states\n")

    for i, phi in enumerate(targets):
        # Optimize Krylov score
        result = optimize.maximize_krylov_score(
            psi, phi, hams, m=d, restarts=2, maxiter=50, seed=51 + i
        )

        # Check binary criterion at optimal parameters
        H_opt = sum(lam * H for lam, H in zip(result["best_x"], hams))
        is_unreachable = mathematics.is_unreachable_krylov(H_opt, psi, phi, d)

        # Also check spectral overlap for comparison
        s_opt = mathematics.spectral_overlap(result["best_x"], psi, phi, hams)

        print(f"Target {i+1}:")
        print(f"  R* = {result['best_value']:.4f}")
        print(f"  Binary unreachable: {is_unreachable}")
        print(f"  Spectral S* = {s_opt:.4f}")

        # Consistency check: high R* should mean reachable
        if result["best_value"] > 0.99:
            assert not is_unreachable, "High R* but binary says unreachable"
            print("  ✓ Consistent: high R* → reachable")
        elif result["best_value"] < 0.01:
            assert is_unreachable, "Low R* but binary says reachable"
            print("  ✓ Consistent: low R* → unreachable")

    print("\n✓ Binary and continuous criteria are consistent")


def test_edge_cases():
    """Test edge cases and numerical stability."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge cases")
    print("=" * 60)

    d, k = 4, 2
    hams = models.random_hamiltonian_ensemble(d, k, "GUE", seed=52)
    psi = models.fock_state(d, 0)

    # Test 1: phi = psi (should be perfectly reachable)
    score_same = mathematics.krylov_score(np.array([1.0, 0.0]), psi, psi, hams, m=d)
    print(f"φ = ψ: R(λ) = {score_same:.10f}")
    assert score_same > 0.99, "φ = ψ should give R ≈ 1"
    print("✓ Same state gives R ≈ 1")

    # Test 2: Orthogonal state (should have lower score)
    phi_orth = models.fock_state(d, d - 1)  # Last basis state
    score_orth = mathematics.krylov_score(np.array([0.5, 0.5]), psi, phi_orth, hams, m=d)
    print(f"φ ⊥ ψ (in std basis): R(λ) = {score_orth:.6f}")
    print("✓ Orthogonal state computed")

    # Test 3: Different Krylov ranks
    phi_rand = models.random_states(1, d, seed=53)[0]
    for m in [2, 4, d]:
        score_m = mathematics.krylov_score(np.array([0.3, -0.7]), psi, phi_rand, hams, m=m)
        print(f"m={m}: R(λ) = {score_m:.6f}")

    print("✓ Different Krylov ranks computed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CONTINUOUS KRYLOV SCORE IMPLEMENTATION TESTS")
    print("=" * 60)

    try:
        test_krylov_score_basic()
        test_score_residual_relationship()
        test_optimize_krylov_score()
        test_comparison_with_binary()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
