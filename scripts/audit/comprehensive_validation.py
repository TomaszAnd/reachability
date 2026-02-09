#!/usr/bin/env python3
"""
Comprehensive validation suite for all three reachability criteria.

Tests:
1. Provably reachable targets → all criteria should say reachable
2. Provably unreachable targets → at least one criterion should detect
3. Cross-criterion consistency checks
4. Moment criterion for GEO2 (debug why it always returned 1)
5. Krylov sanity check for GEO2 (verify dense Hamiltonians fill Krylov space)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from scipy.linalg import expm, null_space
import qutip
from reach import models, optimize, mathematics, settings


# ============================================================================
# Moment criterion (inline, using analysis.py's approach with QuTiP)
# ============================================================================
def moment_criterion_qutip(psi, phi, hams):
    """
    Moment criterion using QuTiP objects (matches analysis.py implementation).

    Returns True if criterion proves target is unreachable.
    """
    K = len(hams)

    # Expect values
    energies_psi = np.array([qutip.expect(H, psi) for H in hams])
    energies_phi = np.array([qutip.expect(H, phi) for H in hams])
    diff = energies_phi - energies_psi

    # Null space
    kernel = null_space(diff.reshape(1, -1))
    if kernel.size == 0:
        return False

    # Anticommutators
    anticomms = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            anticomms[i][j] = (hams[i] * hams[j] + hams[j] * hams[i]) / 2

    # Second moment expectations
    sq_psi = np.array([[qutip.expect(anticomms[i][j], psi) for j in range(K)] for i in range(K)])
    sq_phi = np.array([[qutip.expect(anticomms[i][j], phi) for j in range(K)] for i in range(K)])

    m_final = kernel.T @ (sq_phi - sq_psi) @ kernel

    # Check definiteness
    eigvals = np.linalg.eigvalsh(m_final)
    return bool(np.all(eigvals > 0) or np.all(eigvals < 0))


def moment_criterion_numpy(psi, phi, hams):
    """
    Moment criterion using numpy arrays (matches moment_criteria.py interface).

    Converts QuTiP objects to numpy first.
    """
    psi_vec = psi.full().flatten()
    phi_vec = phi.full().flatten()
    hams_np = [H.full() for H in hams]

    K = len(hams_np)

    # L vector
    L = np.zeros(K)
    for k in range(K):
        L[k] = np.real(phi_vec.conj() @ hams_np[k] @ phi_vec) - \
               np.real(psi_vec.conj() @ hams_np[k] @ psi_vec)

    # Q matrix
    Q = np.zeros((K, K))
    for k in range(K):
        for m in range(K):
            anticomm = (hams_np[k] @ hams_np[m] + hams_np[m] @ hams_np[k]) / 2
            Q[k, m] = np.real(phi_vec.conj() @ anticomm @ phi_vec) - \
                      np.real(psi_vec.conj() @ anticomm @ psi_vec)

    # Search for x such that Q + x LL^T > 0
    L_outer = np.outer(L, L)
    for x in np.linspace(-10, 10, 1000):
        M = Q + x * L_outer
        eigvals = np.linalg.eigvalsh(M)
        if np.all(eigvals > 1e-10):
            return True

    return False


# ============================================================================
# Helper: generate reachable target
# ============================================================================
def generate_reachable_target(psi, hams, rng, t=1.0):
    """Generate target = exp(-iH(λ)t)|ψ> (provably reachable)."""
    K = len(hams)
    lambdas = rng.uniform(-1, 1, K)
    H_lambda = sum(l * H for l, H in zip(lambdas, hams))
    H_matrix = H_lambda.full()
    psi_arr = psi.full().flatten()
    U = expm(-1j * H_matrix * t)
    target_arr = U @ psi_arr
    target = qutip.Qobj(target_arr.reshape(-1, 1)).unit()
    return target, lambdas


# ============================================================================
# Test 1: Provably Reachable Targets
# ============================================================================
def test_provably_reachable():
    """All criteria should classify exp(-iHt)|ψ⟩ as reachable."""
    print("=" * 70)
    print("TEST 1: PROVABLY REACHABLE TARGETS")
    print("Target: φ = exp(-iH(λ)t)|ψ⟩ → all criteria must say reachable")
    print("=" * 70)

    configs = [
        ("Canonical d=16, K=13", "canonical", 16, 13, {}),
        ("GUE d=16, K=13", "GUE", 16, 13, {}),
        ("GEO2 d=16, K=13", "GEO2", 16, 13, {'nx': 2, 'ny': 2}),
        ("GEO2 d=32, K=20", "GEO2", 32, 20, {'nx': 1, 'ny': 5}),
    ]

    tau = 0.99
    n_trials = 20

    for label, ensemble, d, K, params in configs:
        print(f"\n--- {label} ---")
        rng = np.random.RandomState(42)

        spectral_false_unreach = 0
        krylov_false_unreach = 0
        moment_false_unreach_qt = 0
        moment_false_unreach_np = 0

        for trial in range(n_trials):
            seed = rng.randint(0, 2**31 - 1)
            psi = models.fock_state(d, 0)
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)
            phi, true_lam = generate_reachable_target(psi, hams, rng)

            # Spectral (with analytical gradient)
            res_s = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
            )
            if res_s['best_value'] < tau:
                spectral_false_unreach += 1

            # Krylov
            res_k = optimize.maximize_krylov_score(
                psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
            )
            if res_k['best_value'] < tau:
                krylov_false_unreach += 1

            # Moment (QuTiP approach)
            if moment_criterion_qutip(psi, phi, hams):
                moment_false_unreach_qt += 1

            # Moment (numpy approach)
            if moment_criterion_numpy(psi, phi, hams):
                moment_false_unreach_np += 1

            if trial < 3:
                print(f"  Trial {trial}: S*={res_s['best_value']:.4f}, "
                      f"R*={res_k['best_value']:.4f}, "
                      f"Moment_qt={'UNREACH' if moment_criterion_qutip(psi, phi, hams) else 'reach'}, "
                      f"Moment_np={'UNREACH' if moment_criterion_numpy(psi, phi, hams) else 'reach'}")

        print(f"\n  False-unreachable rates (should all be 0%):")
        print(f"    Spectral:    {spectral_false_unreach}/{n_trials} ({100*spectral_false_unreach/n_trials:.1f}%)")
        print(f"    Krylov:      {krylov_false_unreach}/{n_trials} ({100*krylov_false_unreach/n_trials:.1f}%)")
        print(f"    Moment (qt): {moment_false_unreach_qt}/{n_trials} ({100*moment_false_unreach_qt/n_trials:.1f}%)")
        print(f"    Moment (np): {moment_false_unreach_np}/{n_trials} ({100*moment_false_unreach_np/n_trials:.1f}%)")

        all_ok = (spectral_false_unreach == 0 and krylov_false_unreach == 0 and
                  moment_false_unreach_qt == 0 and moment_false_unreach_np == 0)
        print(f"  {'PASS' if all_ok else 'FAIL'}")


# ============================================================================
# Test 2: Random Targets with Known Physics
# ============================================================================
def test_random_targets_physics():
    """Test behavior on random targets at different densities."""
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM TARGETS - Criterion Behavior vs Density")
    print("Verify Krylov ≤ Spectral ≤ Moment ordering")
    print("=" * 70)

    configs = [
        ("GEO2 d=16", "GEO2", 16, {'nx': 2, 'ny': 2}),
        ("Canonical d=16", "canonical", 16, {}),
    ]

    K_values = [2, 5, 10, 13]
    tau = 0.99
    n_trials = 30

    for label, ensemble, d, params in configs:
        print(f"\n--- {label} ---")

        for K in K_values:
            rho = K / d**2
            s_unreach = 0
            k_unreach = 0
            m_unreach = 0
            s_scores = []
            k_scores = []

            for trial in range(n_trials):
                seed = 3000 + K * 100 + trial
                psi = models.fock_state(d, 0)
                phi = models.random_states(1, d, seed=seed + 500)[0]
                hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)

                # Spectral
                res_s = optimize.maximize_spectral_overlap(
                    psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
                )
                s_scores.append(res_s['best_value'])
                if res_s['best_value'] < tau:
                    s_unreach += 1

                # Krylov
                res_k = optimize.maximize_krylov_score(
                    psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
                )
                k_scores.append(res_k['best_value'])
                if res_k['best_value'] < tau:
                    k_unreach += 1

                # Moment
                if moment_criterion_qutip(psi, phi, hams):
                    m_unreach += 1

            print(f"  K={K:3d} (ρ={rho:.4f}): "
                  f"P_s={s_unreach/n_trials:.2f} (<S*>={np.mean(s_scores):.4f}), "
                  f"P_k={k_unreach/n_trials:.2f} (<R*>={np.mean(k_scores):.4f}), "
                  f"P_m={m_unreach/n_trials:.2f}")


# ============================================================================
# Test 3: Krylov Space Dimension Check for GEO2
# ============================================================================
def test_krylov_space_dimension():
    """Check if GEO2 Krylov space fills entire Hilbert space."""
    print("\n" + "=" * 70)
    print("TEST 3: KRYLOV SPACE DIMENSION FOR GEO2")
    print("Why is P(krylov unreach) = 0 everywhere for GEO2?")
    print("=" * 70)

    d = 16
    psi = models.fock_state(d, 0)
    rng = np.random.RandomState(42)

    for K in [2, 5, 13]:
        hams = models.random_hamiltonian_ensemble(d, K, "GEO2", seed=42, nx=2, ny=2)

        # Construct H(λ) for random λ
        lambdas = rng.uniform(-1, 1, K)
        H_lambda = sum(l * H for l, H in zip(lambdas, hams))
        H_mat = H_lambda.full()
        psi_vec = psi.full().flatten()

        # Build Krylov basis explicitly: {ψ, Hψ, H²ψ, ...}
        krylov_vecs = [psi_vec]
        v = psi_vec.copy()
        for m in range(1, d):
            v = H_mat @ v
            # Orthogonalize
            for prev in krylov_vecs:
                v = v - np.dot(prev.conj(), v) * prev
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                break
            v = v / norm
            krylov_vecs.append(v)

        krylov_dim = len(krylov_vecs)

        # Also check rank of each H_i
        ranks = [np.linalg.matrix_rank(H.full()) for H in hams[:3]]

        print(f"  K={K}: Krylov dim = {krylov_dim}/{d}, "
              f"H_i ranks: {ranks}")

    # Compare with canonical
    print("\n  --- Canonical d=16 for comparison ---")
    for K in [2, 5, 13]:
        hams = models.random_hamiltonian_ensemble(d, K, "canonical", seed=42)

        lambdas = rng.uniform(-1, 1, K)
        H_lambda = sum(l * H for l, H in zip(lambdas, hams))
        H_mat = H_lambda.full()
        psi_vec = psi.full().flatten()

        krylov_vecs = [psi_vec]
        v = psi_vec.copy()
        for m in range(1, d):
            v = H_mat @ v
            for prev in krylov_vecs:
                v = v - np.dot(prev.conj(), v) * prev
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                break
            v = v / norm
            krylov_vecs.append(v)

        krylov_dim = len(krylov_vecs)
        ranks = [np.linalg.matrix_rank(H.full()) for H in hams[:3]]

        print(f"  K={K}: Krylov dim = {krylov_dim}/{d}, "
              f"H_i ranks: {ranks}")


# ============================================================================
# Test 4: Moment Criterion Diagnostics for GEO2
# ============================================================================
def test_moment_diagnostics():
    """Deep dive into why moment criterion may not work for GEO2."""
    print("\n" + "=" * 70)
    print("TEST 4: MOMENT CRITERION DIAGNOSTICS")
    print("Investigate Q matrix structure for GEO2 vs Canonical")
    print("=" * 70)

    d = 16
    K = 13
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=777)[0]

    for ensemble, params in [("canonical", {}), ("GEO2", {'nx': 2, 'ny': 2})]:
        print(f"\n--- {ensemble} d={d}, K={K} ---")
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=42, **params)

        # Compute L and Q (numpy approach)
        psi_vec = psi.full().flatten()
        phi_vec = phi.full().flatten()
        hams_np = [H.full() for H in hams]

        L_vec = np.array([
            np.real(phi_vec.conj() @ H @ phi_vec) - np.real(psi_vec.conj() @ H @ psi_vec)
            for H in hams_np
        ])

        Q = np.zeros((K, K))
        for k in range(K):
            for m in range(K):
                ac = (hams_np[k] @ hams_np[m] + hams_np[m] @ hams_np[k]) / 2
                Q[k, m] = np.real(phi_vec.conj() @ ac @ phi_vec) - \
                          np.real(psi_vec.conj() @ ac @ psi_vec)

        Q_eigvals = np.linalg.eigvalsh(Q)

        print(f"  ||L|| = {np.linalg.norm(L_vec):.4f}")
        print(f"  Q eigenvalues: min={Q_eigvals[0]:.4f}, max={Q_eigvals[-1]:.4f}")
        print(f"  Q has {np.sum(Q_eigvals > 1e-10)} positive, "
              f"{np.sum(Q_eigvals < -1e-10)} negative, "
              f"{np.sum(np.abs(Q_eigvals) < 1e-10)} zero eigenvalues")

        # Check if Q alone is positive definite (x=0)
        if np.all(Q_eigvals > 1e-10):
            print(f"  Q is positive definite → unreachable at x=0")
        elif np.all(Q_eigvals < -1e-10):
            print(f"  Q is negative definite → unreachable at x=0 (negated)")
        else:
            print(f"  Q is indefinite → need nonzero x")

        # Check analysis.py approach (null space)
        energies_psi = np.array([qutip.expect(H, psi) for H in hams])
        energies_phi = np.array([qutip.expect(H, phi) for H in hams])
        diff = energies_phi - energies_psi
        kernel = null_space(diff.reshape(1, -1))
        print(f"  Energy diff null space dim: {kernel.shape[1] if kernel.size > 0 else 0}/{K}")
        print(f"  ||energy_diff|| = {np.linalg.norm(diff):.4f}")

        # Check moment via both approaches
        unreach_np = moment_criterion_numpy(psi, phi, hams)
        unreach_qt = moment_criterion_qutip(psi, phi, hams)
        print(f"  Moment (numpy): {'UNREACHABLE' if unreach_np else 'inconclusive'}")
        print(f"  Moment (qutip): {'UNREACHABLE' if unreach_qt else 'inconclusive'}")


# ============================================================================
# Test 5: GEO2 optimize_weights mode comparison
# ============================================================================
def test_geo2_optimize_weights_mode():
    """Compare GEO2 dense (default) vs sparse (optimize_weights=True) modes."""
    print("\n" + "=" * 70)
    print("TEST 5: GEO2 DENSE vs SPARSE MODE")
    print("geo2_optimize_weights=False (dense) vs True (sparse)")
    print("=" * 70)

    d = 16
    K_values = [5, 10, 13, 20]
    tau = 0.99
    n_trials = 20

    for mode_name, opt_weights in [("Dense (default)", False), ("Sparse (opt_weights)", True)]:
        print(f"\n--- {mode_name} ---")

        for K in K_values:
            s_unreach = 0
            k_unreach = 0
            m_unreach = 0

            for trial in range(n_trials):
                seed = 4000 + K * 100 + trial
                psi = models.fock_state(d, 0)
                phi = models.random_states(1, d, seed=seed + 500)[0]

                try:
                    hams = models.random_hamiltonian_ensemble(
                        d, K, "GEO2", seed=seed, nx=2, ny=2,
                        geo2_optimize_weights=opt_weights
                    )
                except ValueError as e:
                    print(f"  K={K}: {e}")
                    break

                # Spectral
                res_s = optimize.maximize_spectral_overlap(
                    psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
                )
                if res_s['best_value'] < tau:
                    s_unreach += 1

                # Krylov
                res_k = optimize.maximize_krylov_score(
                    psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
                )
                if res_k['best_value'] < tau:
                    k_unreach += 1

                # Moment
                if moment_criterion_qutip(psi, phi, hams):
                    m_unreach += 1
            else:
                print(f"  K={K:3d} (ρ={K/d**2:.4f}): "
                      f"P_s={s_unreach/n_trials:.2f}, "
                      f"P_k={k_unreach/n_trials:.2f}, "
                      f"P_m={m_unreach/n_trials:.2f}")


# ============================================================================
# Test 6: Cross-criterion consistency
# ============================================================================
def test_cross_criterion_consistency():
    """Check that criterion ordering is consistent."""
    print("\n" + "=" * 70)
    print("TEST 6: CROSS-CRITERION CONSISTENCY")
    print("If spectral says unreachable, Krylov should also (stronger criterion)")
    print("=" * 70)

    configs = [
        ("Canonical d=16, K=5", "canonical", 16, 5, {}),
        ("GUE d=16, K=5", "GUE", 16, 5, {}),
    ]

    tau = 0.99
    n_trials = 50

    for label, ensemble, d, K, params in configs:
        print(f"\n--- {label} ---")

        violations = 0
        s_unreach_k_reach = 0

        for trial in range(n_trials):
            seed = 6000 + trial
            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed + 500)[0]
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **params)

            res_s = optimize.maximize_spectral_overlap(
                psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
            )
            res_k = optimize.maximize_krylov_score(
                psi, phi, hams, maxiter=100, restarts=3, ftol=1e-6, seed=seed,
            )

            s_unreach = res_s['best_value'] < tau
            k_unreach = res_k['best_value'] < tau

            # If spectral says reachable but Krylov says unreachable → unexpected
            if not s_unreach and k_unreach:
                violations += 1
                print(f"  VIOLATION trial {trial}: S*={res_s['best_value']:.4f}, "
                      f"R*={res_k['best_value']:.4f}")

            if s_unreach and not k_unreach:
                s_unreach_k_reach += 1

        print(f"  Spectral-unreach & Krylov-reach (expected): {s_unreach_k_reach}/{n_trials}")
        print(f"  Spectral-reach & Krylov-unreach (violations): {violations}/{n_trials}")
        print(f"  {'PASS' if violations == 0 else 'INVESTIGATE'}")


def main():
    t0 = time.time()

    test_provably_reachable()
    test_random_targets_physics()
    test_krylov_space_dimension()
    test_moment_diagnostics()
    test_geo2_optimize_weights_mode()
    test_cross_criterion_consistency()

    print(f"\n{'='*70}")
    print(f"Total validation time: {time.time() - t0:.0f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
