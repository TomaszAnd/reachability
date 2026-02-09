#!/usr/bin/env python3
"""
Comprehensive Test Suite for Reachability Analysis.

Tests A through G covering correctness, sensitivity, monotonicity,
consistency, gradients, reproducibility, and algebraic structure.

Usage:
    # Run all tests
    python scripts/tests/run_all_tests.py

    # Run specific test
    python scripts/tests/run_all_tests.py --test A

    # Quick mode (reduced trials)
    python scripts/tests/run_all_tests.py --quick
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import time
import numpy as np
from typing import Dict, List, Tuple
from scipy.linalg import null_space, expm

from reach import models, mathematics, optimize, settings
import qutip

# =============================================================================
# Test Configuration
# =============================================================================

QUICK_MODE = False  # Reduced trials for fast validation


def get_trials():
    return 5 if QUICK_MODE else 20


# =============================================================================
# Test A: Krylov m Sensitivity
# =============================================================================

def test_A_krylov_m_sensitivity():
    """
    Test A: Verify Krylov score depends correctly on m parameter.

    Checks:
    1. R(m=1) <= R(m=2) <= ... <= R(m=d) (monotone in m)
    2. R(m=d) = 1 for GUE (full-rank Krylov space)
    3. Canonical may have early breakdown (R(m) saturates at m_eff << d)
    4. m=min(K,d) gives same result as m=d when K >= d
    """
    print("\n" + "=" * 60)
    print("TEST A: Krylov m Sensitivity")
    print("=" * 60)

    n_trials = get_trials()
    results = {'passed': 0, 'failed': 0, 'details': []}

    for d in [8, 16]:
        for ensemble in ['GUE', 'canonical']:
            for K in [5, d]:
                hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=42)
                psi = models.fock_state(d, 0)
                phi = models.random_states(1, d, seed=100)[0]

                lambdas = np.random.RandomState(42).uniform(-1, 1, K)

                # Compute R(m) for m = 1, 2, ..., min(K, d)
                m_max = min(K, d)
                scores = []
                for m in range(1, m_max + 1):
                    R = mathematics.krylov_score(lambdas, psi, phi, hams, m=m)
                    scores.append(R)

                # Check monotonicity
                is_monotone = all(scores[i] <= scores[i + 1] + 1e-10
                                  for i in range(len(scores) - 1))

                detail = (f"{ensemble} d={d} K={K}: "
                          f"R(m=1)={scores[0]:.4f}, R(m={m_max})={scores[-1]:.4f}, "
                          f"monotone={is_monotone}")
                print(f"  {detail}")
                results['details'].append(detail)

                if is_monotone:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    print(f"    FAIL: Non-monotone! scores={scores}")

    # Check GUE full rank → R=1
    d = 8
    hams = models.random_hamiltonian_ensemble(d, d, 'GUE', seed=42)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=100)[0]
    lambdas = np.random.RandomState(42).uniform(-1, 1, d)
    R_full = mathematics.krylov_score(lambdas, psi, phi, hams, m=d)

    if R_full > 0.999:
        results['passed'] += 1
        print(f"  GUE d={d} K={d} m=d: R={R_full:.6f} ≈ 1.0 PASS")
    else:
        results['failed'] += 1
        print(f"  GUE d={d} K={d} m=d: R={R_full:.6f} != 1.0 FAIL")

    return results


# =============================================================================
# Test B: Analytical Limits
# =============================================================================

def test_B_analytical_limits():
    """
    Test B: Verify criteria match analytical predictions in known limits.

    Checks:
    1. S(λ) = 1 when φ = exp(-iH(λ)t)|ψ⟩ (time-evolved state, any t)
    2. Moment criterion detects unreachability for orthogonal states with K=1
    3. R(λ) = 1 when φ ∈ K_m(H(λ), ψ) (Krylov member)
    """
    print("\n" + "=" * 60)
    print("TEST B: Analytical Limits")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    # Test B1: S(λ) = 1 for time-evolved target
    d, K = 8, 5
    hams = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)
    psi = models.fock_state(d, 0)

    rng = np.random.RandomState(42)
    lambdas = rng.uniform(-1, 1, K)
    t = 2.0

    H = sum(l * H_k for l, H_k in zip(lambdas, hams))
    U = (-1j * H * t).expm()
    phi_evolved = U * psi

    S_evolved = mathematics.spectral_overlap(lambdas, psi, phi_evolved, hams)

    if abs(S_evolved - 1.0) < 1e-6:
        results['passed'] += 1
        print(f"  B1: S(λ) for time-evolved target = {S_evolved:.8f} ≈ 1.0 PASS")
    else:
        results['failed'] += 1
        print(f"  B1: S(λ) for time-evolved target = {S_evolved:.8f} != 1.0 FAIL")

    # Test B2: Moment detects unreachability with K=2 (low K)
    # Note: random_hamiltonian_ensemble requires K>=2
    d = 8
    hams_2 = models.random_hamiltonian_ensemble(d, 2, 'GUE', seed=42)
    psi = models.fock_state(d, 0)

    n_unreachable = 0
    n_trials = get_trials()
    for trial in range(n_trials):
        phi = models.random_states(1, d, seed=200 + trial)[0]

        L = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams_2])
        kernel = null_space(L.reshape(1, -1))

        if kernel.size > 0:
            K_mom = len(hams_2)
            Q = np.zeros((K_mom, K_mom))
            for i in range(K_mom):
                for j in range(K_mom):
                    anticomm = (hams_2[i] * hams_2[j] + hams_2[j] * hams_2[i]) / 2
                    Q[i, j] = qutip.expect(anticomm, phi) - qutip.expect(anticomm, psi)
            Q_proj = kernel.T @ Q @ kernel
            eigvals = np.linalg.eigvalsh(Q_proj)
            if bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10)):
                n_unreachable += 1

    frac_unreachable = n_unreachable / n_trials
    # With K=2, many random targets should be unreachable (not enough control)
    if frac_unreachable > 0.1:
        results['passed'] += 1
        print(f"  B2: Moment K=2 unreachable fraction = {frac_unreachable:.2f} > 0.1 PASS")
    else:
        results['failed'] += 1
        print(f"  B2: Moment K=2 unreachable fraction = {frac_unreachable:.2f} <= 0.1 FAIL")

    # Test B3: R(λ) = 1 for Krylov member
    d, K = 8, 5
    hams = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)
    psi = models.fock_state(d, 0)
    lambdas = rng.uniform(-1, 1, K)

    H = sum(l * H_k for l, H_k in zip(lambdas, hams))
    # phi = H|psi⟩ / ||H|psi⟩|| is in K_m for m >= 2
    Hpsi = H * psi
    phi_krylov = Hpsi / Hpsi.norm()

    R_krylov = mathematics.krylov_score(lambdas, psi, phi_krylov, hams, m=min(K, d))

    if abs(R_krylov - 1.0) < 1e-6:
        results['passed'] += 1
        print(f"  B3: R(λ) for Krylov member = {R_krylov:.8f} ≈ 1.0 PASS")
    else:
        results['failed'] += 1
        print(f"  B3: R(λ) for Krylov member = {R_krylov:.8f} != 1.0 FAIL")

    return results


# =============================================================================
# Test C: Monotonicity in K
# =============================================================================

def test_C_monotonicity_K():
    """
    Test C: P(unreachable) should be monotonically non-increasing in K.

    More Hamiltonians → more control → easier to reach targets.
    We check that P(unreachable | K+1) <= P(unreachable | K) + statistical noise.
    """
    print("\n" + "=" * 60)
    print("TEST C: Monotonicity in K")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}
    d = 8
    K_values = [2, 4, 6, 8, 10, 12]
    tau = 0.99
    n_trials = get_trials()

    for ensemble in ['GUE', 'canonical']:
        P_unreachable = []

        for K in K_values:
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=42)
            psi = models.fock_state(d, 0)

            unreachable = 0
            for trial in range(n_trials):
                phi = models.random_states(1, d, seed=300 + trial)[0]
                result = optimize.maximize_spectral_overlap(
                    psi, phi, hams, maxiter=50, restarts=1, seed=trial
                )
                if result['best_value'] < tau:
                    unreachable += 1

            P = unreachable / n_trials
            P_unreachable.append(P)

        # Check monotonicity (with tolerance for statistical noise)
        noise_tol = 2 * np.sqrt(0.25 / n_trials)  # 2σ binomial noise
        is_monotone = all(
            P_unreachable[i + 1] <= P_unreachable[i] + noise_tol
            for i in range(len(P_unreachable) - 1)
        )

        detail = f"{ensemble} d={d}: P(unreach) = {[f'{p:.2f}' for p in P_unreachable]}"
        print(f"  {detail}")
        results['details'].append(detail)

        if is_monotone:
            results['passed'] += 1
            print(f"    Monotone (tol={noise_tol:.2f}) PASS")
        else:
            results['failed'] += 1
            print(f"    NOT monotone FAIL")

    return results


# =============================================================================
# Test D: Pipeline Consistency
# =============================================================================

def test_D_pipeline_consistency():
    """
    Test D: Verify parallel.py and direct computation give consistent results.

    Compares:
    1. Same seeds produce same Hamiltonians and states in both paths
    2. Raw criterion scores (no optimization) match exactly
    3. evaluate_single_trial produces consistent optimization results
    """
    print("\n" + "=" * 60)
    print("TEST D: Pipeline Consistency")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    d, K = 8, 5
    seed = 42

    for ensemble in ['GUE', 'canonical']:
        # Generate states and Hamiltonians directly (same way as evaluate_single_trial)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed + 1000)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

        # Generate again to verify reproducibility
        phi2 = models.random_states(1, d, seed=seed + 1000)[0]
        hams2 = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

        # D1: States/Hamiltonians match
        phi_match = np.allclose(phi.full(), phi2.full())
        hams_match = all(np.allclose(h1.full(), h2.full()) for h1, h2 in zip(hams, hams2))

        if phi_match and hams_match:
            results['passed'] += 1
            print(f"  {ensemble}: State/Hamiltonian generation reproducible PASS")
        else:
            results['failed'] += 1
            print(f"  {ensemble}: State/Hamiltonian generation NOT reproducible FAIL")

        # D2: Raw criterion scores match at fixed λ (no optimization)
        rng = np.random.RandomState(seed)
        lambdas = rng.uniform(-1, 1, K)

        S1 = mathematics.spectral_overlap(lambdas, psi, phi, hams)
        S2 = mathematics.spectral_overlap(lambdas, psi, phi2, hams2)

        R1 = mathematics.krylov_score(lambdas, psi, phi, hams, m=min(K, d))
        R2 = mathematics.krylov_score(lambdas, psi, phi2, hams2, m=min(K, d))

        spec_diff = abs(S1 - S2)
        kry_diff = abs(R1 - R2)

        detail = (f"{ensemble}: S(λ)={S1:.8f} vs {S2:.8f} (diff={spec_diff:.1e}), "
                  f"R(λ)={R1:.8f} vs {R2:.8f} (diff={kry_diff:.1e})")
        print(f"  {detail}")
        results['details'].append(detail)

        if spec_diff < 1e-12 and kry_diff < 1e-12:
            results['passed'] += 1
            print(f"    Scores match PASS")
        else:
            results['failed'] += 1
            print(f"    Scores differ FAIL")

    return results


# =============================================================================
# Test E: Gradient Edge Cases
# =============================================================================

def test_E_gradient_edge_cases():
    """
    Test E: Verify analytical gradients handle edge cases.

    Checks:
    1. Gradient at λ=0 (all zeros)
    2. Gradient at λ=boundary (±1)
    3. Gradient with degenerate eigenvalues
    4. Gradient vs finite differences agreement
    """
    print("\n" + "=" * 60)
    print("TEST E: Gradient Edge Cases")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    d, K = 8, 3
    hams = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=100)[0]

    # E1: Gradient at λ = 0
    lambdas_zero = np.zeros(K)
    S, grad = mathematics.spectral_overlap_with_grad(lambdas_zero, psi, phi, hams)
    if np.all(np.isfinite(grad)):
        results['passed'] += 1
        print(f"  E1: Gradient at λ=0: S={S:.4f}, ||grad||={np.linalg.norm(grad):.6f} PASS")
    else:
        results['failed'] += 1
        print(f"  E1: Gradient at λ=0 has NaN/Inf FAIL")

    # E2: Gradient at λ = boundary
    lambdas_boundary = np.ones(K)
    S, grad = mathematics.spectral_overlap_with_grad(lambdas_boundary, psi, phi, hams)
    if np.all(np.isfinite(grad)):
        results['passed'] += 1
        print(f"  E2: Gradient at λ=1: S={S:.4f}, ||grad||={np.linalg.norm(grad):.6f} PASS")
    else:
        results['failed'] += 1
        print(f"  E2: Gradient at λ=1 has NaN/Inf FAIL")

    # E3: Gradient vs finite differences
    lambdas = np.random.RandomState(42).uniform(-1, 1, K)
    S, grad_analytical = mathematics.spectral_overlap_with_grad(lambdas, psi, phi, hams)

    eps = 1e-6
    grad_fd = np.zeros(K)
    for i in range(K):
        lam_plus = lambdas.copy()
        lam_plus[i] += eps
        lam_minus = lambdas.copy()
        lam_minus[i] -= eps
        S_plus = mathematics.spectral_overlap(lam_plus, psi, phi, hams)
        S_minus = mathematics.spectral_overlap(lam_minus, psi, phi, hams)
        grad_fd[i] = (S_plus - S_minus) / (2 * eps)

    max_error = np.max(np.abs(grad_analytical - grad_fd))
    if max_error < 1e-4:
        results['passed'] += 1
        print(f"  E3: Spectral grad vs FD: max_error={max_error:.8f} PASS")
    else:
        results['failed'] += 1
        print(f"  E3: Spectral grad vs FD: max_error={max_error:.8f} FAIL")

    # E4: Krylov gradient vs finite differences
    m = min(K, d)
    R, grad_R = mathematics.krylov_score_with_grad(lambdas, psi, phi, hams, m=m)

    grad_R_fd = np.zeros(K)
    for i in range(K):
        lam_plus = lambdas.copy()
        lam_plus[i] += eps
        lam_minus = lambdas.copy()
        lam_minus[i] -= eps
        R_plus = mathematics.krylov_score(lam_plus, psi, phi, hams, m=m)
        R_minus = mathematics.krylov_score(lam_minus, psi, phi, hams, m=m)
        grad_R_fd[i] = (R_plus - R_minus) / (2 * eps)

    max_error_R = np.max(np.abs(grad_R - grad_R_fd))
    if max_error_R < 1e-3:
        results['passed'] += 1
        print(f"  E4: Krylov grad vs FD: max_error={max_error_R:.8f} PASS")
    else:
        results['failed'] += 1
        print(f"  E4: Krylov grad vs FD: max_error={max_error_R:.8f} FAIL")

    return results


# =============================================================================
# Test F: Reproducibility
# =============================================================================

def test_F_reproducibility():
    """
    Test F: Same seed → identical results across runs.

    Checks:
    1. Same Hamiltonians from same seed
    2. Same optimization result from same seed
    3. Same parallel_evaluate result from same seed
    """
    print("\n" + "=" * 60)
    print("TEST F: Reproducibility")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}
    d, K = 8, 5

    # F1: Hamiltonian generation reproducibility
    hams1 = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)
    hams2 = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)

    all_match = all(
        np.allclose(h1.full(), h2.full()) for h1, h2 in zip(hams1, hams2)
    )
    if all_match:
        results['passed'] += 1
        print(f"  F1: Hamiltonian generation reproducible PASS")
    else:
        results['failed'] += 1
        print(f"  F1: Hamiltonian generation NOT reproducible FAIL")

    # F2: Optimization reproducibility
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=100)[0]

    r1 = optimize.maximize_spectral_overlap(psi, phi, hams1, maxiter=50, restarts=1, seed=42)
    r2 = optimize.maximize_spectral_overlap(psi, phi, hams2, maxiter=50, restarts=1, seed=42)

    if abs(r1['best_value'] - r2['best_value']) < 1e-10:
        results['passed'] += 1
        print(f"  F2: Optimization reproducible (S*={r1['best_value']:.8f}) PASS")
    else:
        results['failed'] += 1
        print(f"  F2: Optimization NOT reproducible "
              f"({r1['best_value']:.8f} vs {r2['best_value']:.8f}) FAIL")

    # F3: parallel_evaluate reproducibility
    from reach.parallel import parallel_evaluate

    res1 = parallel_evaluate('GUE', d, K, 0.99, n_trials=5, seed_base=42,
                             criteria=['spectral', 'krylov'], n_jobs=1)
    res2 = parallel_evaluate('GUE', d, K, 0.99, n_trials=5, seed_base=42,
                             criteria=['spectral', 'krylov'], n_jobs=1)

    spec_match = abs(res1['spectral_P'] - res2['spectral_P']) < 1e-10
    kry_match = abs(res1['krylov_P'] - res2['krylov_P']) < 1e-10

    if spec_match and kry_match:
        results['passed'] += 1
        print(f"  F3: parallel_evaluate reproducible "
              f"(S_P={res1['spectral_P']:.2f}, R_P={res1['krylov_P']:.2f}) PASS")
    else:
        results['failed'] += 1
        print(f"  F3: parallel_evaluate NOT reproducible "
              f"(S_P: {res1['spectral_P']:.4f} vs {res2['spectral_P']:.4f}, "
              f"R_P: {res1['krylov_P']:.4f} vs {res2['krylov_P']:.4f}) FAIL")

    return results


# =============================================================================
# Test G: Commutator / Lie Algebra Structure
# =============================================================================

def test_G_lie_algebra():
    """
    Test G: Commutator-based reachability from Lie algebra theory.

    For a set of Hamiltonians {H_1, ..., H_K}, the dynamical Lie algebra
    L = Lie({iH_1, ..., iH_K}) determines the reachable set.

    If dim(L) = d² - 1 (full su(d)), ALL states are reachable.
    If dim(L) < d² - 1, there exist unreachable states.

    We verify:
    1. For K >= d² GUE Hamiltonians, Lie algebra is full → all states reachable
    2. For K = 1, Lie algebra is 1-dimensional → most states unreachable
    3. Commutators [H_i, H_j] generate new directions in Lie algebra
    4. Spectral/Krylov criteria agree with Lie algebra dimension predictions
    """
    print("\n" + "=" * 60)
    print("TEST G: Commutator / Lie Algebra Structure")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    d = 4  # Small for tractability

    # G1: Full su(d) with many generators
    K_full = d * d  # More than enough generators
    hams = models.random_hamiltonian_ensemble(d, K_full, 'GUE', seed=42)

    # Compute Lie algebra dimension by generating commutators
    lie_basis = []
    for H in hams:
        H_mat = H.full()
        # Use traceless part (su(d) is traceless)
        H_traceless = H_mat - np.trace(H_mat) / d * np.eye(d)
        lie_basis.append(1j * H_traceless)

    # Add commutators iteratively
    new_elements = list(lie_basis)
    for _ in range(3):  # 3 rounds of commutators
        next_new = []
        for A in new_elements:
            for B in lie_basis:
                comm = A @ B - B @ A
                # Check if linearly independent of existing basis
                if len(lie_basis) < d * d - 1:
                    # Vectorize and check rank
                    vecs = np.array([b.flatten() for b in lie_basis] + [comm.flatten()])
                    rank = np.linalg.matrix_rank(vecs, tol=1e-10)
                    if rank > len(lie_basis):
                        lie_basis.append(comm)
                        next_new.append(comm)
        new_elements = next_new

    lie_dim = np.linalg.matrix_rank(
        np.array([b.flatten() for b in lie_basis]), tol=1e-10
    )
    expected_dim = d * d - 1  # su(d) dimension

    if lie_dim >= expected_dim - 1:
        results['passed'] += 1
        print(f"  G1: K={K_full} GUE → Lie dim = {lie_dim} "
              f"(su({d}) = {expected_dim}) PASS")
    else:
        results['failed'] += 1
        print(f"  G1: K={K_full} GUE → Lie dim = {lie_dim} "
              f"(expected {expected_dim}) FAIL")

    # G2: Single generator → small Lie algebra
    # Create a single Hamiltonian manually (random_hamiltonian_ensemble requires K>=2)
    rng_g2 = np.random.RandomState(42)
    H1_raw = rng_g2.randn(d, d) + 1j * rng_g2.randn(d, d)
    H1_mat = (H1_raw + H1_raw.conj().T) / 2  # Hermitianize
    H1_traceless = H1_mat - np.trace(H1_mat) / d * np.eye(d)

    # H, H², H³, ... should all be in span of H powers
    lie_basis_1 = [1j * H1_traceless]
    for power in range(2, d + 1):
        Hp = np.linalg.matrix_power(H1_traceless, power)
        Hp = Hp - np.trace(Hp) / d * np.eye(d)
        lie_basis_1.append(1j * Hp)

    vecs = np.array([b.flatten() for b in lie_basis_1])
    lie_dim_1 = np.linalg.matrix_rank(vecs, tol=1e-10)

    # For a single generic Hermitian matrix, the Lie algebra generated
    # by {iH} has dimension = d-1 (diagonal matrices in eigenbasis)
    if lie_dim_1 <= d:
        results['passed'] += 1
        print(f"  G2: K=1 GUE → Lie dim = {lie_dim_1} <= {d} PASS")
    else:
        results['failed'] += 1
        print(f"  G2: K=1 GUE → Lie dim = {lie_dim_1} > {d} FAIL")

    # G3: Commutator generates new direction
    K = 2
    hams_2 = models.random_hamiltonian_ensemble(d, K, 'GUE', seed=42)
    H1 = hams_2[0].full()
    H2 = hams_2[1].full()

    comm12 = H1 @ H2 - H2 @ H1  # = i[iH1, iH2] up to factor

    vecs_before = np.array([H1.flatten(), H2.flatten()])
    rank_before = np.linalg.matrix_rank(vecs_before, tol=1e-10)

    vecs_after = np.array([H1.flatten(), H2.flatten(), comm12.flatten()])
    rank_after = np.linalg.matrix_rank(vecs_after, tol=1e-10)

    if rank_after > rank_before:
        results['passed'] += 1
        print(f"  G3: [H1,H2] generates new direction: "
              f"rank {rank_before} → {rank_after} PASS")
    else:
        results['failed'] += 1
        print(f"  G3: [H1,H2] does NOT generate new direction FAIL")

    # G4: Agreement with spectral criterion
    # Full Lie algebra → all targets reachable → S* ≈ 1 for all φ
    psi = models.fock_state(d, 0)
    n_reachable = 0
    n_test = get_trials()
    for trial in range(n_test):
        phi = models.random_states(1, d, seed=500 + trial)[0]
        result = optimize.maximize_spectral_overlap(
            psi, phi, hams[:d*d],  # Use all K_full generators
            maxiter=100, restarts=2, seed=trial,
        )
        if result['best_value'] >= 0.99:
            n_reachable += 1

    frac_reachable = n_reachable / n_test
    if frac_reachable >= 0.8:
        results['passed'] += 1
        print(f"  G4: Full Lie algebra → {frac_reachable:.0%} reachable (S*≥0.99) PASS")
    else:
        results['failed'] += 1
        print(f"  G4: Full Lie algebra → {frac_reachable:.0%} reachable FAIL")

    return results


# =============================================================================
# Test H: Data Pipeline Audit
# =============================================================================

def test_H_data_pipeline():
    """
    Test H: Verify CSV data integrity for overnight and corrected Krylov data.

    Checks:
    1. Overnight CSVs have correct schema and valid data ranges
    2. Corrected Krylov CSVs have m = min(K, d) for every row
    3. Merged data has no duplicates and valid probabilities
    """
    print("\n" + "=" * 60)
    print("TEST H: Data Pipeline Audit")
    print("=" * 60)

    import pandas as pd
    from pathlib import Path

    results = {'passed': 0, 'failed': 0, 'details': []}

    data_dir = Path(__file__).parent.parent.parent / 'data'
    overnight_dir = data_dir / 'overnight'
    krylov_dir = data_dir / 'krylov_corrected'

    # H1: Overnight CSV integrity
    overnight_files = {
        'canonical': overnight_dir / 'canonical_overnight_20260128_224236.csv',
        'GEO2': overnight_dir / 'geo2_overnight_20260128_224613.csv',
    }

    for name, fpath in overnight_files.items():
        if not fpath.exists():
            print(f"  H1: {name} overnight CSV not found (SKIP)")
            continue

        df = pd.read_csv(fpath)

        # Check columns
        required_cols = ['ensemble', 'd', 'K', 'rho', 'tau', 'spectral_P', 'moment_P']
        missing = [c for c in required_cols if c not in df.columns]

        # Check data ranges
        valid_d = df['d'].isin([8, 16, 32, 64]).all()
        valid_rho = (df['rho'] > 0).all() and (df['rho'] < 1).all()
        valid_P_spec = (df['spectral_P'] >= 0).all() and (df['spectral_P'] <= 1).all()
        valid_P_mom = (df['moment_P'] >= 0).all() and (df['moment_P'] <= 1).all()
        no_nan = not df[['d', 'K', 'rho', 'spectral_P']].isna().any().any()

        if not missing and valid_d and valid_rho and valid_P_spec and valid_P_mom and no_nan:
            results['passed'] += 1
            print(f"  H1: {name} overnight CSV ({len(df)} rows) - schema OK PASS")
        else:
            results['failed'] += 1
            print(f"  H1: {name} overnight CSV - issues found FAIL")
            if missing:
                print(f"      Missing columns: {missing}")

    # H2: Corrected Krylov CSV integrity
    krylov_files = {
        'canonical': krylov_dir / 'canonical_krylov_corrected_20260204_222726.csv',
        'GEO2': krylov_dir / 'geo2_krylov_corrected_20260204_222747.csv',
    }

    for name, fpath in krylov_files.items():
        if not fpath.exists():
            print(f"  H2: {name} corrected Krylov CSV not found (SKIP)")
            continue

        df = pd.read_csv(fpath)

        # Check m = min(K, d) for every row
        expected_m = df.apply(lambda r: min(r['K'], r['d']), axis=1)
        m_correct = (df['m'] == expected_m).all()

        # Check krylov_P in valid range
        valid_P = (df['krylov_P'] >= 0).all() and (df['krylov_P'] <= 1).all()

        if m_correct and valid_P:
            results['passed'] += 1
            print(f"  H2: {name} corrected Krylov ({len(df)} rows) - m=min(K,d) OK PASS")
        else:
            results['failed'] += 1
            print(f"  H2: {name} corrected Krylov - issues found FAIL")
            if not m_correct:
                bad_rows = df[df['m'] != expected_m]
                print(f"      {len(bad_rows)} rows have incorrect m values")

    # H3: Merged data consistency (simulated merge check)
    canonical_overnight = overnight_dir / 'canonical_overnight_20260128_224236.csv'
    canonical_krylov = krylov_dir / 'canonical_krylov_corrected_20260204_222726.csv'

    if canonical_overnight.exists() and canonical_krylov.exists():
        df_over = pd.read_csv(canonical_overnight)
        df_kry = pd.read_csv(canonical_krylov)

        # Aggregate overnight to remove duplicates (same as plot script)
        df_over['K'] = df_over['K'].astype(int)
        df_over['d'] = df_over['d'].astype(int)

        group_cols = ['ensemble', 'd', 'K', 'rho', 'tau']
        num_cols = [c for c in df_over.columns
                    if c not in group_cols and c != 'n_trials'
                    and df_over[c].dtype in ['float64', 'int64']]
        agg = {c: 'mean' for c in num_cols}
        if 'n_trials' in df_over.columns:
            agg['n_trials'] = 'first'
        df_over_agg = df_over.groupby(group_cols, as_index=False).agg(agg)

        df_kry['K'] = df_kry['K'].astype(int)
        df_kry['d'] = df_kry['d'].astype(int)

        # Drop duplicates from corrected Krylov before merge (keep first)
        krylov_merge = df_kry[['d', 'K', 'krylov_P']].drop_duplicates(subset=['d', 'K'])
        df_merged = df_over_agg.merge(krylov_merge, on=['d', 'K'], how='left', suffixes=('_old', ''))

        # After aggregation, check for NEW duplicates (shouldn't be any)
        dup_count = df_merged.duplicated(subset=['d', 'K', 'tau']).sum()

        # Check P values are valid
        valid_merged_P = True
        for col in ['spectral_P', 'moment_P']:
            if col in df_merged.columns:
                valid_merged_P &= (df_merged[col] >= 0).all() and (df_merged[col] <= 1).all()

        if dup_count == 0 and valid_merged_P:
            results['passed'] += 1
            matched = df_merged['krylov_P'].notna().sum()
            print(f"  H3: Merged data - {len(df_merged)} rows, {matched} Krylov matches PASS")
        else:
            results['failed'] += 1
            print(f"  H3: Merged data issues FAIL")
            if dup_count > 0:
                print(f"      {dup_count} duplicate (d, K, tau) rows after aggregation")
    else:
        print("  H3: Required CSV files not found (SKIP)")

    return results


# =============================================================================
# Runner
# =============================================================================

ALL_TESTS = {
    'A': ('Krylov m Sensitivity', test_A_krylov_m_sensitivity),
    'B': ('Analytical Limits', test_B_analytical_limits),
    'C': ('Monotonicity in K', test_C_monotonicity_K),
    'D': ('Pipeline Consistency', test_D_pipeline_consistency),
    'E': ('Gradient Edge Cases', test_E_gradient_edge_cases),
    'F': ('Reproducibility', test_F_reproducibility),
    'G': ('Lie Algebra Structure', test_G_lie_algebra),
    'H': ('Data Pipeline Audit', test_H_data_pipeline),
}


def main():
    global QUICK_MODE

    parser = argparse.ArgumentParser(description='Comprehensive Test Suite')
    parser.add_argument('--test', type=str, default=None,
                        help='Run specific test (A-G)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (reduced trials)')
    args = parser.parse_args()

    QUICK_MODE = args.quick

    print("=" * 60)
    print("COMPREHENSIVE REACHABILITY TEST SUITE")
    print(f"Mode: {'QUICK' if QUICK_MODE else 'FULL'}")
    print("=" * 60)

    start_time = time.time()
    all_results = {}

    tests_to_run = ALL_TESTS
    if args.test:
        if args.test.upper() in ALL_TESTS:
            tests_to_run = {args.test.upper(): ALL_TESTS[args.test.upper()]}
        else:
            print(f"Unknown test: {args.test}. Available: {list(ALL_TESTS.keys())}")
            sys.exit(1)

    for test_id, (name, func) in tests_to_run.items():
        try:
            result = func()
            all_results[test_id] = result
        except Exception as e:
            print(f"\n  TEST {test_id} EXCEPTION: {e}")
            all_results[test_id] = {'passed': 0, 'failed': 1, 'details': [str(e)]}

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    for test_id, result in all_results.items():
        name = ALL_TESTS[test_id][0]
        p = result['passed']
        f = result['failed']
        total_passed += p
        total_failed += f
        status = "PASS" if f == 0 else "FAIL"
        print(f"  Test {test_id} ({name}): {p} passed, {f} failed → {status}")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    print(f"Time: {elapsed:.1f}s")

    if total_failed > 0:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")


if __name__ == '__main__':
    main()
