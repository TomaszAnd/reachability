#!/usr/bin/env python3
"""
Floquet Engineering Diagnostics

This script runs comprehensive diagnostics to verify that the Floquet
implementation will show meaningful results in experiments.

Critical checks:
1. Driving functions have non-zero DC component
2. Fourier overlaps F_jk are non-zero
3. H_F^(1) and H_F^(2) are both significant
4. Commutators generate new terms
5. Floquet can reach higher fidelity than static

Run this BEFORE launching production experiments!
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from reach import floquet, models, optimization, states


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_driving_functions():
    """Check that driving functions have appropriate properties."""
    print_header("TEST 1: Driving Function Properties")

    T = 1.0
    K = 6

    for drive_type in ['sinusoidal', 'offset_sinusoidal', 'bichromatic', 'constant']:
        print(f"\n  Testing {drive_type}:")

        # Create driving functions
        driving = floquet.create_driving_functions(K, drive_type, T, seed=42)

        # Test first function
        f = driving[0]

        # Time-average
        avg = floquet.compute_time_average(f, T, n_points=1000)
        print(f"    Time average ⟨f⟩ = {avg:.6f}")

        # Check at t=0
        val_0 = f(0.0)
        print(f"    Value at t=0: f(0) = {val_0:.6f}")

        # Check periodicity
        val_T = f(T)
        periodic = np.abs(val_T - val_0) < 1e-6
        print(f"    Periodic: {periodic} (|f(T) - f(0)| = {np.abs(val_T - val_0):.2e})")

        # CRITICAL: Check if DC component is non-zero
        if np.abs(avg) < 1e-6:
            print(f"    ⚠️  WARNING: Zero DC component! H_F^(1) will be zero!")
        else:
            print(f"    ✓ Non-zero DC component")

    print("\n  RECOMMENDATION: Use 'offset_sinusoidal' or 'bichromatic' for non-zero H_F^(1)")


def check_fourier_overlaps():
    """Check that Fourier overlaps are significant."""
    print_header("TEST 2: Fourier Overlap Matrix F_jk")

    T = 1.0
    K = 6

    print("\n  Testing different driving types:\n")

    for drive_type in ['sinusoidal', 'offset_sinusoidal', 'bichromatic']:
        print(f"  {drive_type}:")

        driving = floquet.create_driving_functions(K, drive_type, T, seed=42)

        # Compute overlap matrix
        F_matrix = np.zeros((K, K))
        for j in range(K):
            for k in range(j + 1, K):  # Only upper triangle
                F_jk = floquet.compute_fourier_overlap(
                    driving[j], driving[k], T, n_terms=10
                )
                F_matrix[j, k] = F_jk
                F_matrix[k, j] = -F_jk  # Antisymmetric

        max_F = np.max(np.abs(F_matrix))
        mean_F = np.mean(np.abs(F_matrix[np.triu_indices(K, k=1)]))

        print(f"    Max |F_jk|: {max_F:.6f}")
        print(f"    Mean |F_jk|: {mean_F:.6f}")

        if max_F < 0.001:
            print(f"    ⚠️  WARNING: Very small overlaps! H_F^(2) will be weak!")
        else:
            print(f"    ✓ Significant overlaps")

    print("\n  NOTE: Larger |F_jk| → stronger second-order Magnus effects")


def check_magnus_terms():
    """Check that both H_F^(1) and H_F^(2) are significant."""
    print_header("TEST 3: Magnus Expansion Terms")

    # Generate test system
    nx, ny = 2, 2
    d = 2**(nx * ny)
    K = 6

    print(f"\n  System: {nx}×{ny} lattice (d={d}), K={K} operators\n")

    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d, k=K, ensemble="GEO2", nx=nx, ny=ny, seed=42
    )
    hams = floquet.hamiltonians_to_numpy(hams_qutip)

    lambdas = np.random.randn(K) / np.sqrt(K)
    T = 1.0

    for drive_type in ['sinusoidal', 'offset_sinusoidal', 'bichromatic']:
        print(f"  {drive_type}:")

        driving = floquet.create_driving_functions(K, drive_type, T, seed=42)

        # First-order
        H_F1 = floquet.compute_floquet_hamiltonian_order1(hams, lambdas, driving, T)
        norm_1 = np.linalg.norm(H_F1)

        # Second-order
        H_F2 = floquet.compute_floquet_hamiltonian_order2(hams, lambdas, driving, T)
        norm_2 = np.linalg.norm(H_F2)

        # Full
        H_F = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, T, order=2)
        norm_full = np.linalg.norm(H_F)

        print(f"    ||H_F^(1)||: {norm_1:.6f}")
        print(f"    ||H_F^(2)||: {norm_2:.6f}")
        print(f"    ||H_F||:     {norm_full:.6f}")

        if norm_1 > 0:
            ratio = norm_2 / norm_1
            print(f"    Ratio ||H_F^(2)||/||H_F^(1)||: {ratio:.4f}")
        else:
            print(f"    ⚠️  ||H_F^(1)|| = 0! Only commutators contribute!")

        # Check Hermiticity
        is_herm = np.allclose(H_F, H_F.conj().T)
        print(f"    Hermitian: {is_herm}")

        print()


def check_commutator_structure():
    """Check that commutators generate new operator terms."""
    print_header("TEST 4: Commutator Structure")

    # Simple 3-qubit system
    n_qubits = 3
    d = 2**n_qubits

    print(f"\n  System: {n_qubits} qubits (d={d})\n")

    # Generate simple 2-body operators
    I, X, Y, Z = states.pauli_matrices()

    # Z0Z1 and X1X2
    H1 = states.tensor_product([Z, Z, I])  # Z0Z1
    H2 = states.tensor_product([I, X, X])  # X1X2

    print("  Operators:")
    print("    H1 = Z₀Z₁I₂")
    print("    H2 = I₀X₁X₂")

    # Commutator
    comm = H1 @ H2 - H2 @ H1

    print(f"\n  [H1, H2] norm: {np.linalg.norm(comm):.6f}")

    # The commutator should be proportional to Z₀Y₁X₂ (3-body!)
    Z0Y1X2 = states.tensor_product([Z, Y, X])

    # Check if commutator has component along Z0Y1X2
    overlap = np.abs(np.trace(comm.conj().T @ Z0Y1X2))
    normalized_overlap = overlap / (np.linalg.norm(comm) * np.linalg.norm(Z0Y1X2))

    print(f"  Overlap with Z₀Y₁X₂: {normalized_overlap:.6f}")

    if normalized_overlap > 0.5:
        print("  ✓ Commutator generates 3-BODY term from 2-body operators!")
        print("    This is why Floquet can reach states inaccessible to static Hamiltonians.")
    else:
        print("  ? Commutator structure unclear")

    print("\n  KEY INSIGHT: H_F^(2) ∝ [H_j, H_k] expands accessible operator space!")


def check_fidelity_comparison():
    """Check that Floquet can achieve higher fidelity than static."""
    print_header("TEST 5: Static vs Floquet Fidelity Comparison")

    # Test system
    n_qubits = 4
    d = 2**n_qubits
    K = 8

    print(f"\n  System: {n_qubits} qubits (d={d}), K={K} operators")
    print("  Initial state: |0000⟩")
    print("  Target state: GHZ = (|0000⟩ + |1111⟩)/√2\n")

    # States
    psi = states.computational_basis(n_qubits, '0' * n_qubits)
    target_states_dict = states.create_target_states(n_qubits)
    phi = target_states_dict['ghz']

    # Generate Hamiltonians
    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d, k=K, ensemble="GEO2", nx=2, ny=2, seed=42
    )
    hams = floquet.hamiltonians_to_numpy(hams_qutip)

    lambdas = np.random.randn(K) / np.sqrt(K)
    T = 1.0
    t_max = 10.0

    print("  Testing driving types:\n")

    for drive_type in ['constant', 'offset_sinusoidal', 'bichromatic']:
        print(f"  {drive_type}:")

        if drive_type == 'constant':
            # Static case
            H_static = sum(lam * H_k for lam, H_k in zip(lambdas, hams))
            fid, t_opt = optimization.optimize_fidelity(psi, phi, H_static, t_max)
            print(f"    Fidelity: {fid:.4f} (t={t_opt:.3f})")

        else:
            # Floquet case
            driving = floquet.create_driving_functions(K, drive_type, T, seed=42)

            # Compare with static
            result = optimization.compare_static_vs_floquet(
                psi, phi, hams, lambdas, driving, T, t_max, floquet_order=2
            )

            print(f"    Static fidelity:  {result['fidelity_static']:.4f}")
            print(f"    Floquet fidelity: {result['fidelity_floquet']:.4f}")
            print(f"    Improvement: {result['improvement_percent']:+.1f}%")

            if result['fidelity_floquet'] > result['fidelity_static'] + 0.01:
                print(f"    ✓ Floquet achieves HIGHER fidelity!")
            elif result['fidelity_floquet'] < result['fidelity_static'] - 0.01:
                print(f"    ⚠️  Floquet achieves LOWER fidelity (unexpected)")
            else:
                print(f"    ≈ Similar fidelity")

        print()

    print("  If Floquet fidelity > Static fidelity → implementation working correctly!")


def main():
    """Run all diagnostic tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "FLOQUET ENGINEERING DIAGNOSTICS" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")

    try:
        # Run all checks
        check_driving_functions()
        check_fourier_overlaps()
        check_magnus_terms()
        check_commutator_structure()
        check_fidelity_comparison()

        # Summary
        print("\n" + "=" * 70)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 70)
        print("\nKey Recommendations:")
        print("  1. Use 'offset_sinusoidal' or 'bichromatic' driving (NOT plain 'sinusoidal')")
        print("  2. Verify H_F^(1) ≠ 0 and H_F^(2) ≠ 0")
        print("  3. Check that Floquet fidelity ≥ static fidelity")
        print("  4. Use specific state pairs (e.g., |0000⟩ → GHZ) not random")
        print("\nIf all tests pass, proceed with production experiments!")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("DIAGNOSTICS FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
