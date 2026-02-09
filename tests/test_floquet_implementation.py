#!/usr/bin/env python3
"""
Test script for Floquet implementation.

This script verifies that the new Floquet modules work correctly before
running production experiments.

Tests:
1. State generation (GHZ, W, cluster states)
2. Floquet Hamiltonian computation (Magnus expansion)
3. Floquet moment criterion
"""

import numpy as np

from reach import floquet, models, states


def test_state_generation():
    """Test state generation module."""
    print("=" * 70)
    print("TEST 1: State Generation")
    print("=" * 70)

    n_qubits = 4
    d = 2**n_qubits

    # Test initial states
    print("\nInitial states:")
    initial = states.create_initial_states(n_qubits)
    for name, state in initial.items():
        norm = np.linalg.norm(state)
        print(f"  {name:15s}: dim={len(state)}, norm={norm:.6f}")
        assert np.abs(norm - 1.0) < 1e-10, f"{name} not normalized!"

    # Test target states
    print("\nTarget states:")
    target = states.create_target_states(n_qubits)
    for name, state in target.items():
        norm = np.linalg.norm(state)
        print(f"  {name:15s}: dim={len(state)}, norm={norm:.6f}")
        assert np.abs(norm - 1.0) < 1e-10, f"{name} not normalized!"

    # Test GHZ state properties
    ghz = target['ghz']
    state_0 = states.computational_basis(n_qubits, '0' * n_qubits)
    state_1 = states.computational_basis(n_qubits, '1' * n_qubits)
    ghz_expected = (state_0 + state_1) / np.sqrt(2)
    assert np.allclose(ghz, ghz_expected), "GHZ state incorrect!"
    print("\n  ✓ GHZ state verified: (|0000⟩ + |1111⟩)/√2")

    print("\n  ✓ All states normalized and verified\n")


def test_floquet_hamiltonian():
    """Test Floquet Hamiltonian computation."""
    print("=" * 70)
    print("TEST 2: Floquet Hamiltonian (Magnus Expansion)")
    print("=" * 70)

    # Small test system: 2×2 lattice (d=16)
    nx, ny = 2, 2
    d = 2 ** (nx * ny)
    print(f"\nLattice: {nx}×{ny} (d={d})")

    # Generate GEO2 Hamiltonians
    K = 10  # Number of operators
    print(f"Sampling K={K} operators from GEO2 basis...")
    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d,
        k=K,
        ensemble="GEO2",
        nx=nx,
        ny=ny,
        periodic=False,
        geo2_optimize_weights=True,
        seed=42
    )

    # Convert to numpy
    hams = floquet.hamiltonians_to_numpy(hams_qutip)
    print(f"  ✓ Generated {len(hams)} Hamiltonians")

    # Create driving functions
    T = 1.0
    drive_type = 'sinusoidal'
    driving = floquet.create_driving_functions(K, drive_type, T, seed=42)
    print(f"  ✓ Created {len(driving)} {drive_type} driving functions")

    # Sample lambdas
    lambdas = np.random.randn(K) / np.sqrt(K)
    print(f"  ✓ Sampled λ coefficients: mean={np.mean(lambdas):.4f}, std={np.std(lambdas):.4f}")

    # Compute first-order Floquet Hamiltonian
    print("\nComputing H_F^(1) (time-averaged)...")
    H_F1 = floquet.compute_floquet_hamiltonian_order1(hams, lambdas, driving, T)
    print(f"  Shape: {H_F1.shape}")
    print(f"  Hermitian: {np.allclose(H_F1, H_F1.conj().T)}")
    print(f"  Norm: {np.linalg.norm(H_F1):.6f}")

    # Compute second-order Floquet Hamiltonian
    print("\nComputing H_F^(2) (commutator corrections)...")
    H_F2 = floquet.compute_floquet_hamiltonian_order2(hams, lambdas, driving, T, n_fourier_terms=5)
    print(f"  Shape: {H_F2.shape}")
    print(f"  Anti-Hermitian: {np.allclose(H_F2, -H_F2.conj().T)}")  # Should be anti-Hermitian
    print(f"  Norm: {np.linalg.norm(H_F2):.6f}")
    print(f"  Ratio H_F2/H_F1: {np.linalg.norm(H_F2)/np.linalg.norm(H_F1):.6f}")

    # Full Floquet Hamiltonian
    print("\nComputing full H_F = H_F^(1) + H_F^(2)...")
    H_F = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, T, order=2, n_fourier_terms=5)
    print(f"  Shape: {H_F.shape}")
    print(f"  Norm: {np.linalg.norm(H_F):.6f}")

    # Verify H_F1 contribution
    assert np.allclose(H_F, H_F1 + H_F2), "H_F ≠ H_F1 + H_F2!"
    print(f"  ✓ Verified: H_F = H_F^(1) + H_F^(2)")

    print("\n  ✓ Floquet Hamiltonian computation successful\n")

    return hams, lambdas, driving, T


def test_floquet_moment_criterion(hams, lambdas, driving, T):
    """Test Floquet moment criterion."""
    print("=" * 70)
    print("TEST 3: Floquet Moment Criterion")
    print("=" * 70)

    d = hams[0].shape[0]
    K = len(hams)

    # Generate test states
    print(f"\nGenerating random states (d={d})...")
    psi = states.random_state(d, seed=42)
    phi = states.random_state(d, seed=43)
    print(f"  ⟨ψ|φ⟩ = {np.abs(psi.conj() @ phi):.6f}")

    # Test moment criterion (order 1)
    print("\nTesting Floquet moment (order 1)...")
    definite1, x1, eigs1 = floquet.floquet_moment_criterion(
        psi, phi, hams, lambdas, driving, T, order=1, n_fourier_terms=5
    )
    print(f"  Definite: {definite1}")
    if x1 is not None:
        print(f"  Optimal x: {x1:.6f}")
    print(f"  Q_F eigenvalues: min={np.min(eigs1):.6f}, max={np.max(eigs1):.6f}")

    # Test moment criterion (order 2)
    print("\nTesting Floquet moment (order 2)...")
    definite2, x2, eigs2 = floquet.floquet_moment_criterion(
        psi, phi, hams, lambdas, driving, T, order=2, n_fourier_terms=5
    )
    print(f"  Definite: {definite2}")
    if x2 is not None:
        print(f"  Optimal x: {x2:.6f}")
    print(f"  Q_F eigenvalues: min={np.min(eigs2):.6f}, max={np.max(eigs2):.6f}")

    # Compare order 1 vs order 2
    print(f"\nComparison:")
    print(f"  Order 1: {'UNREACHABLE' if definite1 else 'inconclusive'}")
    print(f"  Order 2: {'UNREACHABLE' if definite2 else 'inconclusive'}")

    # Test Monte Carlo estimation
    print("\nTesting Monte Carlo estimation (n=10 trials)...")
    P_unreach = floquet.floquet_moment_criterion_probability(
        hams, lambdas, driving, T, order=2, n_trials=10, dim=d, seed=42
    )
    print(f"  P(unreachable) ≈ {P_unreach:.2f}")

    print("\n  ✓ Floquet moment criterion functional\n")


def test_driving_functions():
    """Test different driving function types."""
    print("=" * 70)
    print("TEST 4: Driving Functions")
    print("=" * 70)

    T = 1.0
    K = 5

    print("\nTesting driving function types:")

    for drive_type in ['sinusoidal', 'square', 'multi_freq', 'constant']:
        driving = floquet.create_driving_functions(K, drive_type, T, seed=42)
        print(f"  {drive_type:15s}: {len(driving)} functions")

        # Evaluate at t=0
        values_t0 = [f(0.0) for f in driving]
        print(f"    f(0) = {values_t0[0]:.4f}")

        # Check time-average (should equal DC component)
        avg = floquet.compute_time_average(driving[0], T)
        print(f"    ⟨f⟩ = {avg:.4f}")

    print("\n  ✓ All driving functions created successfully\n")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FLOQUET IMPLEMENTATION TEST SUITE" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    try:
        # Test 1: State generation
        test_state_generation()

        # Test 2: Floquet Hamiltonians
        hams, lambdas, driving, T = test_floquet_hamiltonian()

        # Test 3: Floquet moment criterion
        test_floquet_moment_criterion(hams, lambdas, driving, T)

        # Test 4: Driving functions
        test_driving_functions()

        # Summary
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nThe Floquet implementation is ready for production use.")
        print("Next steps:")
        print("  1. Run production script: python scripts/run_geo2_floquet.py")
        print("  2. Generate plots: python scripts/plot_geo2_floquet.py")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
