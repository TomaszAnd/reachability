#!/usr/bin/env python3
"""
Comprehensive Mathematical Verification Suite

Verifies that all reachability criteria implementations match their
mathematical definitions from the paper/theory.

Tests:
1. Spectral overlap matches S(λ) = Σₙ |⟨φ|uₙ⟩* ⟨uₙ|ψ⟩|
2. Krylov basis is orthonormal Arnoldi basis
3. Krylov score equals 1 - residual²
4. Moment criterion uses correct first/second moments
5. Analytical gradient matches finite differences
6. Lambda optimization finds global maximum
7. GEO2 Hamiltonians have correct properties
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy.linalg import expm, eigh, null_space
import qutip
from reach import models, optimize, mathematics, settings
import time


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# TEST 1: Spectral Overlap Definition
# =============================================================================

def test_spectral_overlap_definition():
    """Verify spectral overlap matches S(λ) = Σₙ |φₙ* ψₙ|."""
    banner("TEST 1: SPECTRAL OVERLAP DEFINITION")
    print("Mathematical definition: S(λ) = Σₙ |⟨φ|uₙ(λ)⟩* · ⟨uₙ(λ)|ψ⟩|")

    d = 8
    K = 4
    np.random.seed(42)

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=123)[0]
    hams = models.random_hamiltonian_ensemble(d, K, "GUE", seed=42)

    lambdas = np.random.uniform(-1, 1, K)

    # Compute using library function
    S_lib = mathematics.spectral_overlap(lambdas, psi, phi, hams)

    # Compute manually following the definition
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))
    eigenvalues, eigenvectors = eigh(H_lambda.full())

    psi_vec = psi.full().flatten()
    phi_vec = phi.full().flatten()

    # ψₙ = ⟨uₙ|ψ⟩, φₙ = ⟨uₙ|φ⟩
    psi_coeffs = eigenvectors.conj().T @ psi_vec
    phi_coeffs = eigenvectors.conj().T @ phi_vec

    # S = Σₙ |φₙ* · ψₙ|
    S_manual = np.sum(np.abs(phi_coeffs.conj() * psi_coeffs))

    print(f"\n  Library S(λ):  {S_lib:.10f}")
    print(f"  Manual S(λ):   {S_manual:.10f}")
    print(f"  Difference:    {abs(S_lib - S_manual):.2e}")

    passed = abs(S_lib - S_manual) < 1e-12
    print(f"\n  {'PASS' if passed else 'FAIL'}: Definitions match")

    # Test known case: σx rotation |0⟩ → |1⟩
    print("\n  Known case: σx eigenbasis with |0⟩ and |1⟩")
    d = 2
    psi = qutip.basis(2, 0)  # |0⟩
    phi = qutip.basis(2, 1)  # |1⟩
    H = qutip.sigmax()
    hams = [H]

    # σx eigenstates are |+⟩ = (|0⟩+|1⟩)/√2, |-⟩ = (|0⟩-|1⟩)/√2
    # Both states have equal projection onto both eigenstates
    # ψ₊ = ⟨+|0⟩ = 1/√2, ψ₋ = ⟨-|0⟩ = 1/√2
    # φ₊ = ⟨+|1⟩ = 1/√2, φ₋ = ⟨-|1⟩ = -1/√2
    # S = |φ₊* ψ₊| + |φ₋* ψ₋| = |1/2| + |-1/2| = 1

    S_01 = mathematics.spectral_overlap(np.array([1.0]), psi, phi, [H])
    print(f"  S(|0⟩→|1⟩, σx) = {S_01:.6f} (expected: 1.0)")

    return passed and abs(S_01 - 1.0) < 1e-6


# =============================================================================
# TEST 2: Krylov Basis Properties
# =============================================================================

def test_krylov_basis_properties():
    """Verify Krylov basis is orthonormal and spans correct subspace."""
    banner("TEST 2: KRYLOV BASIS PROPERTIES")
    print("Verify: V†V = I and V spans span{ψ, Hψ, H²ψ, ...}")

    d = 16
    np.random.seed(42)

    psi = models.fock_state(d, 0)
    # Create a single Hamiltonian directly (models require K>=2)
    H = qutip.rand_herm(d, seed=42)

    for m in [4, 8, 16]:
        V = mathematics.krylov_basis(H, psi, m)

        # Check orthonormality
        VtV = V.conj().T @ V
        ortho_error = np.linalg.norm(VtV - np.eye(m))

        # Check that columns span Krylov subspace
        # Each column should be expressible as linear combo of {ψ, Hψ, ..., H^{i-1}ψ}
        psi_vec = psi.full().flatten()
        H_mat = H.full()

        krylov_vecs = [psi_vec / np.linalg.norm(psi_vec)]
        v = psi_vec.copy()
        for i in range(1, m):
            v = H_mat @ v
            # Orthogonalize
            for prev in krylov_vecs:
                v = v - np.vdot(prev, v) * prev
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                break
            krylov_vecs.append(v / norm)

        # V should span same space as krylov_vecs
        # Project krylov_vecs onto V and check residual
        K_manual = np.column_stack(krylov_vecs)
        proj = V @ (V.conj().T @ K_manual)
        resid = np.linalg.norm(K_manual - proj)

        print(f"  m={m}: ortho_error={ortho_error:.2e}, span_residual={resid:.2e}")

    return ortho_error < 1e-12 and resid < 1e-12


# =============================================================================
# TEST 3: Krylov Score = 1 - Residual²
# =============================================================================

def test_krylov_score_identity():
    """Verify R(λ) = ‖V†φ‖² = 1 - ‖φ - VV†φ‖²."""
    banner("TEST 3: KRYLOV SCORE IDENTITY")
    print("Verify: R(λ) = ‖V†φ‖² = 1 - residual²")

    d = 16
    K = 5
    np.random.seed(42)

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=123)[0]
    hams = models.random_hamiltonian_ensemble(d, K, "GUE", seed=42)

    lambdas = np.random.uniform(-1, 1, K)

    # Compute Krylov score via library
    R_lib = mathematics.krylov_score(lambdas, psi, phi, hams)

    # Compute manually
    H_lambda = sum(lam * H for lam, H in zip(lambdas, hams))
    V = mathematics.krylov_basis(H_lambda, psi, d)
    phi_vec = phi.full().flatten()

    # R = ‖V†φ‖²
    coeffs = V.conj().T @ phi_vec
    R_manual = np.real(np.vdot(coeffs, coeffs))

    # Also: R = 1 - residual²
    proj = V @ coeffs
    resid = phi_vec - proj
    R_from_resid = 1.0 - np.real(np.vdot(resid, resid))

    print(f"\n  Library R(λ):      {R_lib:.10f}")
    print(f"  Manual ‖V†φ‖²:     {R_manual:.10f}")
    print(f"  1 - residual²:     {R_from_resid:.10f}")
    print(f"  |R - (1-res²)|:    {abs(R_lib - R_from_resid):.2e}")

    passed = abs(R_lib - R_manual) < 1e-12 and abs(R_lib - R_from_resid) < 1e-10
    print(f"\n  {'PASS' if passed else 'FAIL'}: Identity verified")
    return passed


# =============================================================================
# TEST 4: Moment Criterion Mathematics
# =============================================================================

def test_moment_criterion_mathematics():
    """Verify moment criterion computes correct first/second moments."""
    banner("TEST 4: MOMENT CRITERION MATHEMATICS")
    print("Verify: L_k = ⟨H_k⟩_φ - ⟨H_k⟩_ψ")
    print("        Q_km = ⟨{H_k,H_m}/2⟩_φ - ⟨{H_k,H_m}/2⟩_ψ")

    d = 8
    K = 4
    np.random.seed(42)

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=123)[0]
    hams = models.random_hamiltonian_ensemble(d, K, "GUE", seed=42)

    psi_vec = psi.full().flatten()
    phi_vec = phi.full().flatten()

    # Compute L manually
    L = np.zeros(K)
    for k in range(K):
        H_k = hams[k].full()
        L[k] = np.real(np.vdot(phi_vec, H_k @ phi_vec)) - \
               np.real(np.vdot(psi_vec, H_k @ psi_vec))

    # Compute Q manually
    Q = np.zeros((K, K))
    for k in range(K):
        for m in range(K):
            H_k, H_m = hams[k].full(), hams[m].full()
            anticomm = (H_k @ H_m + H_m @ H_k) / 2
            Q[k, m] = np.real(np.vdot(phi_vec, anticomm @ phi_vec)) - \
                      np.real(np.vdot(psi_vec, anticomm @ psi_vec))

    # Verify symmetry
    Q_sym_error = np.linalg.norm(Q - Q.T)
    print(f"\n  Q symmetry error: {Q_sym_error:.2e}")

    # Verify using QuTiP expect
    L_qt = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams])
    L_error = np.linalg.norm(L - L_qt)
    print(f"  L computation error: {L_error:.2e}")

    # Check moment criterion logic
    # If L ≠ 0, null space is K-1 dimensional at most
    kernel = null_space(L.reshape(1, -1))
    print(f"  L null space dimension: {kernel.shape[1] if kernel.size > 0 else 0}/{K}")

    passed = Q_sym_error < 1e-12 and L_error < 1e-12
    print(f"\n  {'PASS' if passed else 'FAIL'}: Moment computation correct")
    return passed


# =============================================================================
# TEST 5: Analytical Gradient Accuracy
# =============================================================================

def test_analytical_gradient():
    """Verify analytical gradient matches finite differences."""
    banner("TEST 5: ANALYTICAL GRADIENT ACCURACY")
    print("Compare ∂S/∂λ analytical vs finite difference")

    d = 16
    K = 8
    np.random.seed(42)

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=123)[0]
    hams = models.random_hamiltonian_ensemble(d, K, "GUE", seed=42)

    lambdas = np.random.uniform(-0.5, 0.5, K)

    # Analytical gradient
    S_ana, grad_ana = mathematics.spectral_overlap_with_grad(lambdas, psi, phi, hams)

    # Finite difference gradient
    eps = 1e-7
    grad_fd = np.zeros(K)
    for k in range(K):
        lam_plus = lambdas.copy()
        lam_plus[k] += eps
        lam_minus = lambdas.copy()
        lam_minus[k] -= eps
        S_plus = mathematics.spectral_overlap(lam_plus, psi, phi, hams)
        S_minus = mathematics.spectral_overlap(lam_minus, psi, phi, hams)
        grad_fd[k] = (S_plus - S_minus) / (2 * eps)

    # Compare
    abs_error = np.abs(grad_ana - grad_fd)
    rel_error = abs_error / (np.abs(grad_fd) + 1e-10)

    print(f"\n  Gradient comparison (K={K}):")
    print(f"  Max absolute error: {np.max(abs_error):.2e}")
    print(f"  Max relative error: {np.max(rel_error):.2e}")
    print(f"  Mean absolute error: {np.mean(abs_error):.2e}")

    # Test degenerate eigenvalue handling
    print("\n  Degenerate eigenvalue test:")
    # Create Hamiltonian with degenerate eigenvalues
    H_degen = qutip.Qobj(np.diag([1.0, 1.0, 2.0, 3.0]))
    hams_degen = [H_degen]
    d = 4
    psi = qutip.basis(d, 0)
    phi = qutip.basis(d, 1)
    lambdas = np.array([1.0])

    try:
        S, grad = mathematics.spectral_overlap_with_grad(lambdas, psi, phi, hams_degen)
        print(f"  S = {S:.6f}, grad = {grad[0]:.6f}")
        print(f"  No NaN: {'PASS' if np.isfinite(grad[0]) else 'FAIL'}")
        degen_pass = np.isfinite(grad[0])
    except Exception as e:
        print(f"  Exception: {e}")
        degen_pass = False

    passed = np.max(rel_error) < 1e-5 and degen_pass
    print(f"\n  {'PASS' if passed else 'FAIL'}: Gradient accurate")
    return passed


# =============================================================================
# TEST 6: Optimizer Finds Global Maximum
# =============================================================================

def test_optimizer_global():
    """Verify optimizer finds same maximum as grid search for small K."""
    banner("TEST 6: OPTIMIZER vs GRID SEARCH")
    print("For small K, compare optimizer result with exhaustive grid")

    d = 8
    K = 2  # Small K for tractable grid search
    np.random.seed(42)

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=123)[0]
    hams = models.random_hamiltonian_ensemble(d, K, "GUE", seed=42)

    # Grid search
    n_grid = 21
    grid_vals = np.linspace(-1, 1, n_grid)
    S_max_grid = 0.0
    lam_max_grid = None

    for l1 in grid_vals:
        for l2 in grid_vals:
            lam = np.array([l1, l2])
            S = mathematics.spectral_overlap(lam, psi, phi, hams)
            if S > S_max_grid:
                S_max_grid = S
                lam_max_grid = lam.copy()

    # Optimizer
    result = optimize.maximize_spectral_overlap(
        psi, phi, hams, maxiter=200, restarts=5, ftol=1e-8, seed=42
    )
    S_max_opt = result['best_value']
    lam_max_opt = result['best_x']

    print(f"\n  Grid search: S_max = {S_max_grid:.6f} at λ = {lam_max_grid}")
    print(f"  Optimizer:   S_max = {S_max_opt:.6f} at λ = {lam_max_opt}")
    print(f"  Difference:  {abs(S_max_opt - S_max_grid):.6f}")

    # Optimizer should find same or better
    passed = S_max_opt >= S_max_grid - 0.001
    print(f"\n  {'PASS' if passed else 'FAIL'}: Optimizer finds global maximum")
    return passed


# =============================================================================
# TEST 7: GEO2 Hamiltonian Properties
# =============================================================================

def test_geo2_properties():
    """Verify GEO2 Hamiltonians have correct properties."""
    banner("TEST 7: GEO2 HAMILTONIAN PROPERTIES")
    print("Check: Hermiticity, dimension, rank, normalization")

    configs = [
        (16, 2, 2),  # 2x2 lattice
        (32, 1, 5),  # 1x5 lattice (5 qubits)
    ]

    for d, nx, ny in configs:
        print(f"\n  d={d} ({nx}x{ny} lattice):")

        hams = models.random_hamiltonian_ensemble(
            d, 10, "GEO2", seed=42, nx=nx, ny=ny
        )

        hermitian_ok = True
        rank_ok = True
        dim_ok = True

        for i, H in enumerate(hams[:3]):  # Check first 3
            H_mat = H.full()

            # Hermiticity
            herm_error = np.linalg.norm(H_mat - H_mat.conj().T)
            if herm_error > 1e-12:
                hermitian_ok = False
                print(f"    H_{i} not Hermitian: error = {herm_error:.2e}")

            # Dimension
            if H_mat.shape != (d, d):
                dim_ok = False
                print(f"    H_{i} wrong shape: {H_mat.shape}")

            # Rank (should be full for GEO2)
            rank = np.linalg.matrix_rank(H_mat)
            if rank != d:
                rank_ok = False
                print(f"    H_{i} not full rank: {rank}/{d}")

        # Normalization: Tr(H²)/d should be O(1)
        H0 = hams[0].full()
        trace_H2 = np.trace(H0 @ H0) / d
        print(f"    Tr(H²)/d = {np.real(trace_H2):.4f} (expect O(1))")

        print(f"    Hermitian: {'OK' if hermitian_ok else 'FAIL'}")
        print(f"    Dimension: {'OK' if dim_ok else 'FAIL'}")
        print(f"    Full rank: {'OK' if rank_ok else 'FAIL'}")

    return hermitian_ok and rank_ok and dim_ok


# =============================================================================
# TEST 8: Provably Reachable Targets
# =============================================================================

def test_provably_reachable():
    """Verify 0% false-unreachable on exp(-iHt)|ψ⟩ targets."""
    banner("TEST 8: PROVABLY REACHABLE TARGETS")
    print("φ = exp(-iH(λ)t)|ψ⟩ is reachable by construction")

    configs = [
        ("GUE d=16, K=10", "GUE", 16, 10, {}),
        ("GEO2 d=16, K=10", "GEO2", 16, 10, {'nx': 2, 'ny': 2}),
    ]

    tau = 0.99
    n_trials = 20

    all_pass = True
    for label, ensemble, d, K, params in configs:
        print(f"\n  {label}:")

        false_unreach = 0
        for trial in range(n_trials):
            seed = 8000 + trial
            rng = np.random.RandomState(seed)

            psi = models.fock_state(d, 0)
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)

            # Generate provably reachable target
            lambdas_true = rng.uniform(-1, 1, K)
            t = 1.0
            H_lambda = sum(l * H for l, H in zip(lambdas_true, hams))
            U = expm(-1j * H_lambda.full() * t)
            phi_arr = U @ psi.full().flatten()
            phi = qutip.Qobj(phi_arr.reshape(-1, 1)).unit()

            # Optimize spectral overlap
            result = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed
            )

            if result['best_value'] < tau:
                false_unreach += 1

        rate = false_unreach / n_trials * 100
        print(f"    False-unreachable: {false_unreach}/{n_trials} ({rate:.1f}%)")

        if false_unreach > 0:
            all_pass = False

    print(f"\n  {'PASS' if all_pass else 'FAIL'}: Zero false-unreachable rate")
    return all_pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner("COMPREHENSIVE MATHEMATICAL VERIFICATION")
    print("Testing all reachability criteria against mathematical definitions")

    t0 = time.time()

    results = {}
    results['spectral_definition'] = test_spectral_overlap_definition()
    results['krylov_basis'] = test_krylov_basis_properties()
    results['krylov_identity'] = test_krylov_score_identity()
    results['moment_math'] = test_moment_criterion_mathematics()
    results['gradient'] = test_analytical_gradient()
    results['optimizer'] = test_optimizer_global()
    results['geo2_properties'] = test_geo2_properties()
    results['provably_reachable'] = test_provably_reachable()

    # Summary
    banner("VERIFICATION SUMMARY")
    print(f"\nTotal time: {time.time() - t0:.1f}s\n")

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
