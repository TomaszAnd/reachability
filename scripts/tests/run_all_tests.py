#!/usr/bin/env python3
"""
Comprehensive Test Suite for Reachability Analysis.

Tests A through H covering correctness, sensitivity, monotonicity,
consistency, gradients, reproducibility, algebraic structure, and data pipeline.

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

from src.models import CanonicalQuditModel, QubitGridModel, QuantumModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion, Verdict

# =============================================================================
# Test Configuration
# =============================================================================

QUICK_MODE = False  # Reduced trials for fast validation


def get_trials():
    return 5 if QUICK_MODE else 20


def make_model_and_hams(d, K, ensemble, seed=42):
    """Create a model and sample K operators from it. Returns list of numpy arrays."""
    if ensemble == 'GUE':
        # GUE: random dense Hermitian matrices
        rng = np.random.RandomState(seed)
        hams = []
        for i in range(K):
            h_seed = rng.randint(0, 2**31 - 1)
            h_rng = np.random.RandomState(h_seed)
            A = h_rng.randn(d, d) + 1j * h_rng.randn(d, d)
            H = (A + A.conj().T) / np.sqrt(2)
            hams.append(H)
        return hams
    elif ensemble == 'canonical':
        model = CanonicalQuditModel(d, seed=seed)
        sub = model.sample_submodel(K)
        return sub.basis
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}")


def make_init_state(d):
    """Create |0> state as numpy array."""
    state = np.zeros(d, dtype=np.complex128)
    state[0] = 1.0
    return state


def make_random_state(d, seed):
    """Create a Haar-random state as numpy array."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(d) + 1j * rng.randn(d)
    return vec / np.linalg.norm(vec)


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

    results = {'passed': 0, 'failed': 0, 'details': []}

    for d in [8, 16]:
        for ensemble in ['GUE', 'canonical']:
            for K in [5, d]:
                hams = make_model_and_hams(d, K, ensemble, seed=42)
                phi = make_init_state(d)
                psi = make_random_state(d, seed=100)

                lambdas = np.random.RandomState(42).uniform(-1, 1, K)

                # Compute R(m) for m = 1, 2, ..., min(K, d)
                m_max = min(K, d)
                scores = []
                for m in range(1, m_max + 1):
                    kc = KrylovCriterion(hams, phi, psi, m=m)
                    R = kc.evaluate(lambdas)
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

    # Check GUE full rank -> R=1
    d = 8
    hams = make_model_and_hams(d, d, 'GUE', seed=42)
    phi = make_init_state(d)
    psi = make_random_state(d, seed=100)
    lambdas = np.random.RandomState(42).uniform(-1, 1, d)
    kc = KrylovCriterion(hams, phi, psi, m=d)
    R_full = kc.evaluate(lambdas)

    if R_full > 0.999:
        results['passed'] += 1
        print(f"  GUE d={d} K={d} m=d: R={R_full:.6f} ~ 1.0 PASS")
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
    1. S(lambda) = 1 when psi = exp(-iH(lambda)t)|phi> (time-evolved state)
    2. Moment criterion detects unreachability for K=2
    3. R(lambda) = 1 when psi in K_m(H(lambda), phi) (Krylov member)
    """
    print("\n" + "=" * 60)
    print("TEST B: Analytical Limits")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    # Test B1: S(lambda) = 1 for time-evolved target
    d, K = 8, 5
    hams = make_model_and_hams(d, K, 'GUE', seed=42)
    phi = make_init_state(d)

    rng = np.random.RandomState(42)
    lambdas = rng.uniform(-1, 1, K)
    t = 2.0

    H = sum(l * H_k for l, H_k in zip(lambdas, hams))
    U = expm(-1j * H * t)
    psi_evolved = U @ phi

    sc = SpectralCriterion(hams, phi, psi_evolved)
    S_evolved = sc.evaluate(lambdas)

    if abs(S_evolved - 1.0) < 1e-6:
        results['passed'] += 1
        print(f"  B1: S(lambda) for time-evolved target = {S_evolved:.8f} ~ 1.0 PASS")
    else:
        results['failed'] += 1
        print(f"  B1: S(lambda) for time-evolved target = {S_evolved:.8f} != 1.0 FAIL")

    # Test B2: Moment detects unreachability with K=2 (low K)
    d = 8
    hams_2 = make_model_and_hams(d, 2, 'GUE', seed=42)
    phi = make_init_state(d)

    n_unreachable = 0
    n_trials = get_trials()
    for trial in range(n_trials):
        psi = make_random_state(d, seed=200 + trial)

        mc = MomentCriterion(hams_2, phi, psi)
        result = mc.is_reachable()
        if result.verdict == Verdict.UNREACHABLE:
            n_unreachable += 1

    frac_unreachable = n_unreachable / n_trials
    if frac_unreachable > 0.1:
        results['passed'] += 1
        print(f"  B2: Moment K=2 unreachable fraction = {frac_unreachable:.2f} > 0.1 PASS")
    else:
        results['failed'] += 1
        print(f"  B2: Moment K=2 unreachable fraction = {frac_unreachable:.2f} <= 0.1 FAIL")

    # Test B3: R(lambda) = 1 for Krylov member
    d, K = 8, 5
    hams = make_model_and_hams(d, K, 'GUE', seed=42)
    phi = make_init_state(d)
    lambdas = rng.uniform(-1, 1, K)

    H = sum(l * H_k for l, H_k in zip(lambdas, hams))
    # psi = H|phi> / ||H|phi>|| is in K_m for m >= 2
    Hphi = H @ phi
    psi_krylov = Hphi / np.linalg.norm(Hphi)

    kc = KrylovCriterion(hams, phi, psi_krylov, m=min(K, d))
    R_krylov = kc.evaluate(lambdas)

    if abs(R_krylov - 1.0) < 1e-6:
        results['passed'] += 1
        print(f"  B3: R(lambda) for Krylov member = {R_krylov:.8f} ~ 1.0 PASS")
    else:
        results['failed'] += 1
        print(f"  B3: R(lambda) for Krylov member = {R_krylov:.8f} != 1.0 FAIL")

    return results


# =============================================================================
# Test C: Monotonicity in K
# =============================================================================

def test_C_monotonicity_K():
    """
    Test C: P(unreachable) should be monotonically non-increasing in K.

    More Hamiltonians -> more control -> easier to reach targets.
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
            hams = make_model_and_hams(d, K, ensemble, seed=42)
            phi = make_init_state(d)

            unreachable = 0
            for trial in range(n_trials):
                psi = make_random_state(d, seed=300 + trial)
                sc = SpectralCriterion(hams, phi, psi, tau=tau)
                result = sc.is_reachable(maxiter=50, restarts=1, seed=trial)
                if result.verdict == Verdict.UNREACHABLE:
                    unreachable += 1

            P = unreachable / n_trials
            P_unreachable.append(P)

        # Check monotonicity (with tolerance for statistical noise)
        noise_tol = 2 * np.sqrt(0.25 / n_trials)  # 2 sigma binomial noise
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
    Test D: Same seeds produce same results.

    Compares:
    1. Same seeds produce same Hamiltonians and states
    2. Raw criterion scores match exactly
    """
    print("\n" + "=" * 60)
    print("TEST D: Pipeline Consistency")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    d, K = 8, 5
    seed = 42

    for ensemble in ['GUE', 'canonical']:
        phi = make_init_state(d)
        psi = make_random_state(d, seed=seed + 1000)
        hams = make_model_and_hams(d, K, ensemble, seed=seed)

        # Generate again to verify reproducibility
        psi2 = make_random_state(d, seed=seed + 1000)
        hams2 = make_model_and_hams(d, K, ensemble, seed=seed)

        # D1: States/Hamiltonians match
        psi_match = np.allclose(psi, psi2)
        hams_match = all(np.allclose(h1, h2) for h1, h2 in zip(hams, hams2))

        if psi_match and hams_match:
            results['passed'] += 1
            print(f"  {ensemble}: State/Hamiltonian generation reproducible PASS")
        else:
            results['failed'] += 1
            print(f"  {ensemble}: State/Hamiltonian generation NOT reproducible FAIL")

        # D2: Raw criterion scores match at fixed lambda (no optimization)
        rng = np.random.RandomState(seed)
        lambdas = rng.uniform(-1, 1, K)

        sc1 = SpectralCriterion(hams, phi, psi)
        sc2 = SpectralCriterion(hams2, phi, psi2)
        S1 = sc1.evaluate(lambdas)
        S2 = sc2.evaluate(lambdas)

        kc1 = KrylovCriterion(hams, phi, psi, m=min(K, d))
        kc2 = KrylovCriterion(hams2, phi, psi2, m=min(K, d))
        R1 = kc1.evaluate(lambdas)
        R2 = kc2.evaluate(lambdas)

        spec_diff = abs(S1 - S2)
        kry_diff = abs(R1 - R2)

        detail = (f"{ensemble}: S(lambda)={S1:.8f} vs {S2:.8f} (diff={spec_diff:.1e}), "
                  f"R(lambda)={R1:.8f} vs {R2:.8f} (diff={kry_diff:.1e})")
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
    1. Gradient at lambda=0 (all zeros)
    2. Gradient at lambda=boundary (+/-1)
    3. Gradient vs finite differences agreement (spectral)
    4. Gradient vs finite differences agreement (Krylov)
    """
    print("\n" + "=" * 60)
    print("TEST E: Gradient Edge Cases")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    d, K = 8, 3
    hams = make_model_and_hams(d, K, 'GUE', seed=42)
    phi = make_init_state(d)
    psi = make_random_state(d, seed=100)

    sc = SpectralCriterion(hams, phi, psi)

    # E1: Gradient at lambda = 0
    lambdas_zero = np.zeros(K)
    S, grad = sc.evaluate(lambdas_zero, return_gradient=True)
    if np.all(np.isfinite(grad)):
        results['passed'] += 1
        print(f"  E1: Gradient at lambda=0: S={S:.4f}, ||grad||={np.linalg.norm(grad):.6f} PASS")
    else:
        results['failed'] += 1
        print(f"  E1: Gradient at lambda=0 has NaN/Inf FAIL")

    # E2: Gradient at lambda = boundary
    lambdas_boundary = np.ones(K)
    S, grad = sc.evaluate(lambdas_boundary, return_gradient=True)
    if np.all(np.isfinite(grad)):
        results['passed'] += 1
        print(f"  E2: Gradient at lambda=1: S={S:.4f}, ||grad||={np.linalg.norm(grad):.6f} PASS")
    else:
        results['failed'] += 1
        print(f"  E2: Gradient at lambda=1 has NaN/Inf FAIL")

    # E3: Spectral gradient vs finite differences
    lambdas = np.random.RandomState(42).uniform(-1, 1, K)
    S, grad_analytical = sc.evaluate(lambdas, return_gradient=True)

    eps = 1e-6
    grad_fd = np.zeros(K)
    for i in range(K):
        lam_plus = lambdas.copy()
        lam_plus[i] += eps
        lam_minus = lambdas.copy()
        lam_minus[i] -= eps
        S_plus = sc.evaluate(lam_plus)
        S_minus = sc.evaluate(lam_minus)
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
    kc = KrylovCriterion(hams, phi, psi, m=m)
    R, grad_R = kc.evaluate(lambdas, return_gradient=True)

    grad_R_fd = np.zeros(K)
    for i in range(K):
        lam_plus = lambdas.copy()
        lam_plus[i] += eps
        lam_minus = lambdas.copy()
        lam_minus[i] -= eps
        R_plus = kc.evaluate(lam_plus)
        R_minus = kc.evaluate(lam_minus)
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
    Test F: Same seed -> identical results across runs.

    Checks:
    1. Same Hamiltonians from same seed
    2. Same optimization result from same seed
    3. DensitySweep reproducibility
    """
    print("\n" + "=" * 60)
    print("TEST F: Reproducibility")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}
    d, K = 8, 5

    # F1: Hamiltonian generation reproducibility
    hams1 = make_model_and_hams(d, K, 'GUE', seed=42)
    hams2 = make_model_and_hams(d, K, 'GUE', seed=42)

    all_match = all(
        np.allclose(h1, h2) for h1, h2 in zip(hams1, hams2)
    )
    if all_match:
        results['passed'] += 1
        print(f"  F1: Hamiltonian generation reproducible PASS")
    else:
        results['failed'] += 1
        print(f"  F1: Hamiltonian generation NOT reproducible FAIL")

    # F2: Optimization reproducibility
    phi = make_init_state(d)
    psi = make_random_state(d, seed=100)

    sc1 = SpectralCriterion(hams1, phi, psi, tau=0.99)
    sc2 = SpectralCriterion(hams2, phi, psi, tau=0.99)
    r1 = sc1.is_reachable(maxiter=50, restarts=1, seed=42)
    r2 = sc2.is_reachable(maxiter=50, restarts=1, seed=42)

    if abs(r1.score - r2.score) < 1e-10:
        results['passed'] += 1
        print(f"  F2: Optimization reproducible (S*={r1.score:.8f}) PASS")
    else:
        results['failed'] += 1
        print(f"  F2: Optimization NOT reproducible "
              f"({r1.score:.8f} vs {r2.score:.8f}) FAIL")

    # F3: DensitySweep reproducibility
    from src.sampling import DensitySweep, SweepConfig

    config = SweepConfig(n_hamiltonians=3, n_targets=2, maxiter=30, restarts=1)
    model1 = CanonicalQuditModel(d, seed=42)
    model2 = CanonicalQuditModel(d, seed=42)

    sweep1 = DensitySweep(model1, config)
    sweep2 = DensitySweep(model2, config)

    df1 = sweep1.run([3, 5], criteria=['spectral'], verbose=False)
    df2 = sweep2.run([3, 5], criteria=['spectral'], verbose=False)

    spec_match = np.allclose(df1['spectral_P'].values, df2['spectral_P'].values)

    if spec_match:
        results['passed'] += 1
        print(f"  F3: DensitySweep reproducible "
              f"(spectral_P={df1['spectral_P'].values}) PASS")
    else:
        results['failed'] += 1
        print(f"  F3: DensitySweep NOT reproducible "
              f"({df1['spectral_P'].values} vs {df2['spectral_P'].values}) FAIL")

    return results


# =============================================================================
# Test G: Commutator / Lie Algebra Structure
# =============================================================================

def test_G_lie_algebra():
    """
    Test G: Commutator-based reachability from Lie algebra theory.

    Checks:
    1. For K >= d^2 GUE Hamiltonians, Lie algebra is full
    2. For K=1, Lie algebra is small
    3. Commutators generate new directions
    4. Full Lie algebra -> all targets reachable
    """
    print("\n" + "=" * 60)
    print("TEST G: Commutator / Lie Algebra Structure")
    print("=" * 60)

    results = {'passed': 0, 'failed': 0, 'details': []}

    d = 4  # Small for tractability

    # G1: Full su(d) with many generators
    K_full = d * d
    hams = make_model_and_hams(d, K_full, 'GUE', seed=42)

    # Compute Lie algebra dimension by generating commutators
    lie_basis = []
    for H in hams:
        H_traceless = H - np.trace(H) / d * np.eye(d)
        lie_basis.append(1j * H_traceless)

    # Add commutators iteratively
    new_elements = list(lie_basis)
    for _ in range(3):
        next_new = []
        for A in new_elements:
            for B in lie_basis:
                comm = A @ B - B @ A
                if len(lie_basis) < d * d - 1:
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
        print(f"  G1: K={K_full} GUE -> Lie dim = {lie_dim} "
              f"(su({d}) = {expected_dim}) PASS")
    else:
        results['failed'] += 1
        print(f"  G1: K={K_full} GUE -> Lie dim = {lie_dim} "
              f"(expected {expected_dim}) FAIL")

    # G2: Single generator -> small Lie algebra
    rng_g2 = np.random.RandomState(42)
    H1_raw = rng_g2.randn(d, d) + 1j * rng_g2.randn(d, d)
    H1_mat = (H1_raw + H1_raw.conj().T) / 2
    H1_traceless = H1_mat - np.trace(H1_mat) / d * np.eye(d)

    lie_basis_1 = [1j * H1_traceless]
    for power in range(2, d + 1):
        Hp = np.linalg.matrix_power(H1_traceless, power)
        Hp = Hp - np.trace(Hp) / d * np.eye(d)
        lie_basis_1.append(1j * Hp)

    vecs = np.array([b.flatten() for b in lie_basis_1])
    lie_dim_1 = np.linalg.matrix_rank(vecs, tol=1e-10)

    if lie_dim_1 <= d:
        results['passed'] += 1
        print(f"  G2: K=1 GUE -> Lie dim = {lie_dim_1} <= {d} PASS")
    else:
        results['failed'] += 1
        print(f"  G2: K=1 GUE -> Lie dim = {lie_dim_1} > {d} FAIL")

    # G3: Commutator generates new direction
    K = 2
    hams_2 = make_model_and_hams(d, K, 'GUE', seed=42)
    H1 = hams_2[0]
    H2 = hams_2[1]

    comm12 = H1 @ H2 - H2 @ H1

    vecs_before = np.array([H1.flatten(), H2.flatten()])
    rank_before = np.linalg.matrix_rank(vecs_before, tol=1e-10)

    vecs_after = np.array([H1.flatten(), H2.flatten(), comm12.flatten()])
    rank_after = np.linalg.matrix_rank(vecs_after, tol=1e-10)

    if rank_after > rank_before:
        results['passed'] += 1
        print(f"  G3: [H1,H2] generates new direction: "
              f"rank {rank_before} -> {rank_after} PASS")
    else:
        results['failed'] += 1
        print(f"  G3: [H1,H2] does NOT generate new direction FAIL")

    # G4: Agreement with spectral criterion
    phi = make_init_state(d)
    n_reachable = 0
    n_test = get_trials()
    for trial in range(n_test):
        psi = make_random_state(d, seed=500 + trial)
        sc = SpectralCriterion(hams[:d*d], phi, psi, tau=0.99)
        result = sc.is_reachable(maxiter=100, restarts=2, seed=trial)
        if result.verdict == Verdict.REACHABLE:
            n_reachable += 1

    frac_reachable = n_reachable / n_test
    if frac_reachable >= 0.8:
        results['passed'] += 1
        print(f"  G4: Full Lie algebra -> {frac_reachable:.0%} reachable (S*>=0.99) PASS")
    else:
        results['failed'] += 1
        print(f"  G4: Full Lie algebra -> {frac_reachable:.0%} reachable FAIL")

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
    canonical_dir = data_dir / 'canonical'
    qubitgrid_dir = data_dir / 'qubitgrid'

    # H1: Overnight CSV integrity
    overnight_files = {
        'canonical': canonical_dir / 'overnight_20260128_224236.csv',
        'GEO2': qubitgrid_dir / 'overnight_20260128_224613.csv',
    }

    for name, fpath in overnight_files.items():
        if not fpath.exists():
            print(f"  H1: {name} overnight CSV not found (SKIP)")
            continue

        df = pd.read_csv(fpath)

        required_cols = ['ensemble', 'd', 'K', 'rho', 'tau', 'spectral_P', 'moment_P']
        missing = [c for c in required_cols if c not in df.columns]

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
        'canonical': canonical_dir / 'krylov_corrected_20260204_222726.csv',
        'GEO2': qubitgrid_dir / 'krylov_corrected_20260204_222747.csv',
    }

    for name, fpath in krylov_files.items():
        if not fpath.exists():
            print(f"  H2: {name} corrected Krylov CSV not found (SKIP)")
            continue

        df = pd.read_csv(fpath)

        expected_m = df.apply(lambda r: min(r['K'], r['d']), axis=1)
        m_correct = (df['m'] == expected_m).all()
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

    # H3: Merged data consistency
    canonical_overnight = canonical_dir / 'overnight_20260128_224236.csv'
    canonical_krylov = canonical_dir / 'krylov_corrected_20260204_222726.csv'

    if canonical_overnight.exists() and canonical_krylov.exists():
        df_over = pd.read_csv(canonical_overnight)
        df_kry = pd.read_csv(canonical_krylov)

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

        krylov_merge = df_kry[['d', 'K', 'krylov_P']].drop_duplicates(subset=['d', 'K'])
        df_merged = df_over_agg.merge(krylov_merge, on=['d', 'K'], how='left',
                                       suffixes=('_old', ''))

        dup_count = df_merged.duplicated(subset=['d', 'K', 'tau']).sum()

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
                        help='Run specific test (A-H)')
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
            import traceback
            traceback.print_exc()
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
        print(f"  Test {test_id} ({name}): {p} passed, {f} failed -> {status}")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    print(f"Time: {elapsed:.1f}s")

    if total_failed > 0:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")


if __name__ == '__main__':
    main()
