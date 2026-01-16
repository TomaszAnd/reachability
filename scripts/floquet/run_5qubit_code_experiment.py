#!/usr/bin/env python3
"""
Test Floquet vs Static moment criterion on 5-qubit perfect code preparation.

Compares the ability of static and Floquet criteria to prove unreachability
of the logical |0_L⟩ state of the [[5,1,3]] quantum error correction code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from datetime import datetime
from reach import floquet, models, moment_criteria


def create_5qubit_code_logical_zero():
    """
    Create |0_L⟩ of the 5-qubit perfect code (Laflamme-Knill, [[5,1,3]]).

    The logical zero state is an equal superposition of 16 basis states
    with specific signs, forming the +1 eigenspace of the stabilizer group.

    Stabilizers:
        S_1 = XZZXI
        S_2 = IXZZX
        S_3 = XIXZZ
        S_4 = ZXIXZ

    Returns:
        np.ndarray: Normalized state vector of dimension 32
    """
    d = 32  # 2^5
    psi = np.zeros(d, dtype=complex)

    # Weight 0 (1 state)
    psi[0b00000] = +1

    # Weight 2 (4 states)
    psi[0b10010] = +1
    psi[0b01001] = +1
    psi[0b10100] = +1
    psi[0b01010] = +1

    # Weight 3 (10 states) - all negative
    psi[0b11011] = -1
    psi[0b00110] = -1
    psi[0b11000] = -1
    psi[0b11101] = -1
    psi[0b00011] = -1
    psi[0b11110] = -1
    psi[0b01111] = -1
    psi[0b10001] = -1
    psi[0b01100] = -1
    psi[0b10111] = -1

    # Weight 4 (1 state)
    psi[0b00101] = +1

    # Normalize
    psi /= np.linalg.norm(psi)

    return psi


def verify_5qubit_code_logical_zero(psi):
    """
    Verify that psi is in the +1 eigenspace of all stabilizer generators.

    Args:
        psi: State vector to verify

    Returns:
        bool: True if all stabilizers have +1 eigenvalue

    Raises:
        AssertionError: If any stabilizer has eigenvalue != +1
    """
    from scipy.linalg import kron

    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def tensor_product(pauli_list):
        """Compute tensor product of Pauli operators."""
        result = pauli_list[0]
        for p in pauli_list[1:]:
            result = kron(result, p)
        return result

    # Stabilizer generators
    S1 = tensor_product([X, Z, Z, X, I])  # XZZXI
    S2 = tensor_product([I, X, Z, Z, X])  # IXZZX
    S3 = tensor_product([X, I, X, Z, Z])  # XIXZZ
    S4 = tensor_product([Z, X, I, X, Z])  # ZXIXZ

    stabilizers = [('XZZXI', S1), ('IXZZX', S2), ('XIXZZ', S3), ('ZXIXZ', S4)]

    print("\nVerifying stabilizer eigenvalues:")
    all_correct = True
    for name, S in stabilizers:
        eigenvalue = np.dot(psi.conj(), S @ psi).real
        is_correct = abs(eigenvalue - 1.0) < 1e-10
        status = "✓" if is_correct else "✗"
        print(f"  {status} {name}: eigenvalue = {eigenvalue:.10f}")
        all_correct = all_correct and is_correct

    if all_correct:
        print("✓ All stabilizers verified")
    else:
        raise AssertionError("State is not a valid code state")

    return all_correct


def run_5qubit_code_experiment(n_lambda_search=100, seed=42):
    """
    Test static vs Floquet criteria on |00000⟩ → |0_L⟩ preparation.

    Args:
        n_lambda_search: Number of random λ vectors to try per K (Floquet only)
        seed: Random seed

    Returns:
        dict: Experiment results including K_c values
    """
    print("="*70)
    print("5-QUBIT PERFECT CODE REACHABILITY EXPERIMENT")
    print("="*70)
    print()
    print(f"Testing criteria on |00000⟩ → |0_L⟩ preparation")
    print(f"λ search trials (Floquet): {n_lambda_search}")
    print(f"Random seed: {seed}")
    print()

    # Setup
    n_qubits = 5
    d = 32

    # Initial state: |00000⟩
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0

    # Target state: |0_L⟩
    print("Creating |0_L⟩ state...")
    phi = create_5qubit_code_logical_zero()

    # Verify
    verify_5qubit_code_logical_zero(phi)

    # Compute overlap
    overlap = abs(np.dot(psi.conj(), phi))**2
    print(f"\nOverlap |⟨00000|0_L⟩|² = {overlap:.6f}")
    print()

    # Generate Hamiltonians (GEO2LOCAL, 1D chain)
    print("Generating GEO2LOCAL Hamiltonians (5-qubit chain)...")
    # For 1D chain: nx=5, ny=1
    # Total operators: L = 3n + 9|E| = 15 + 36 = 51
    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d, k=51, ensemble="GEO2", nx=5, ny=1, seed=seed
    )
    hams = floquet.hamiltonians_to_numpy(hams_qutip)
    print(f"Generated {len(hams)} 2-local operators")
    print()

    # Test range
    K_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    # Driving functions (bichromatic)
    def get_driving_functions(K):
        """Bichromatic driving with two frequencies."""
        omega1 = 1.0
        omega2 = np.sqrt(2)  # Incommensurate
        driving = []
        for k in range(K):
            omega = omega1 if k % 2 == 0 else omega2
            driving.append(lambda t, om=omega: np.sin(om * t))
        return driving

    period = 2 * np.pi

    print("="*70)
    print("TESTING CRITERIA AT VARIOUS K")
    print("="*70)
    print()
    print("  K | Static     | Floquet    | Comment")
    print("-----|------------|------------|" + "-"*35)

    results_by_K = []
    K_c_static = None
    K_c_floquet = None

    for K in K_values:
        # Static criterion
        print(f"\n[K={K}] Testing static criterion...", end='', flush=True)
        unreachable_static, x_opt_static, eigvals_static = \
            moment_criteria.static_moment_criterion(psi, phi, hams[:K])
        print(" done")

        # Floquet criterion (search 100 random λ)
        print(f"[K={K}] Testing Floquet criterion (O(2), {n_lambda_search} λ trials)...", end='', flush=True)
        driving_funcs = get_driving_functions(K)

        unreachable_floquet, lambdas_opt, x_opt_floquet, eigvals_floquet = \
            moment_criteria.floquet_moment_criterion_optimized(
                psi, phi, hams[:K],
                driving_functions=driving_funcs,
                period=period,
                order=2,
                n_lambda_trials=n_lambda_search,
                seed=seed + K  # Different seed for each K
            )
        print(" done")

        # Record results
        result = {
            'K': K,
            'unreachable_static': unreachable_static,
            'unreachable_floquet': unreachable_floquet,
            'x_opt_static': x_opt_static,
            'x_opt_floquet': x_opt_floquet,
            'lambdas_opt': lambdas_opt
        }
        results_by_K.append(result)

        # Determine critical K (first K where criterion succeeds)
        if unreachable_static and K_c_static is None:
            K_c_static = K

        if unreachable_floquet and K_c_floquet is None:
            K_c_floquet = K

        # Format output
        static_str = "✓ UNREACH " if unreachable_static else "✗ inconc  "
        floquet_str = "✓ UNREACH " if unreachable_floquet else "✗ inconc  "

        # Comment
        if unreachable_floquet and not unreachable_static:
            comment = "← Floquet succeeds, static fails!"
        elif unreachable_static and unreachable_floquet:
            comment = "Both succeed"
        elif unreachable_static and not unreachable_floquet:
            comment = "← Static succeeds, Floquet fails"
        else:
            comment = ""

        print(f" {K:3d} | {static_str} | {floquet_str} | {comment}")

    print()
    print("="*70)
    print("CRITICAL K VALUES")
    print("="*70)
    print()

    if K_c_static:
        print(f"Static criterion:  K_c = {K_c_static}")
        print(f"  → Proves unreachable with {K_c_static} operators or fewer")
    else:
        print("Static criterion:  K_c = None (never proved unreachable)")

    print()

    if K_c_floquet:
        print(f"Floquet criterion: K_c = {K_c_floquet}")
        print(f"  → Proves unreachable with {K_c_floquet} operators or fewer")
    else:
        print("Floquet criterion: K_c = None (never proved unreachable)")

    print()

    if K_c_static and K_c_floquet:
        improvement = K_c_floquet - K_c_static
        if improvement > 0:
            print(f"✓ Floquet advantage: Can prove unreachable with {improvement} MORE operators")
        elif improvement < 0:
            print(f"✗ Static advantage: Can prove unreachable with {-improvement} MORE operators")
        else:
            print("= Both criteria have same K_c")
    elif K_c_floquet and not K_c_static:
        print("✓ Floquet SUCCEEDS where static FAILS completely")
    elif K_c_static and not K_c_floquet:
        print("✗ Static SUCCEEDS where Floquet FAILS completely (unexpected!)")

    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()

    print("K_c (critical K) interpretation:")
    print("  → |0_L⟩ is provably UNREACHABLE with K < K_c operators")
    print("  → This provides a lower bound on resources needed for code preparation")
    print()

    if K_c_floquet and K_c_static and K_c_floquet > K_c_static:
        print("Floquet criterion is STRONGER:")
        print("  → Detects unreachability at higher K (more operators)")
        print("  → Time-dependent control enhances discriminative power")
        print("  → Commutator terms add critical information")
    elif not K_c_floquet and not K_c_static:
        print("Both criteria FAIL:")
        print("  → |0_L⟩ may be reachable with most operator subsets")
        print("  → Code state may have similar structure to generic (Haar) states")
    else:
        print("Results require further analysis")

    print()

    # Summary
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_qubits': n_qubits,
        'd': d,
        'K_values': K_values,
        'n_lambda_search': n_lambda_search,
        'seed': seed,
        'K_c_static': K_c_static,
        'K_c_floquet': K_c_floquet,
        'overlap_psi_phi': overlap,
        'results_by_K': results_by_K
    }

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Test moment criteria on 5-qubit perfect code preparation'
    )
    parser.add_argument('--n-lambda-search', type=int, default=100,
                        help='Number of random λ vectors to try per K (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Run experiment
    results = run_5qubit_code_experiment(
        n_lambda_search=args.n_lambda_search,
        seed=args.seed
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/5qubit_code_experiment_{timestamp}.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print("="*70)
    print("RESULTS SAVED")
    print("="*70)
    print()
    print(f"File: {filename}")
    print()

    # Also save summary as JSON for easy reading
    import json
    summary = {
        'timestamp': results['timestamp'],
        'K_c_static': results['K_c_static'],
        'K_c_floquet': results['K_c_floquet'],
        'overlap': results['overlap_psi_phi'],
        'K_values_tested': results['K_values']
    }

    json_filename = filename.replace('.pkl', '_summary.json')
    with open(json_filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary: {json_filename}")
    print()
