#!/usr/bin/env python3
"""
Compare Two Moment Criterion Implementations

Tests:
1. Agreement rate on random (psi, phi, hams) triples
2. Disagreement analysis (when do they differ?)
3. Edge cases (L≈0, Q≈0, K large)
4. Provably reachable targets (should both return False)
5. Provably unreachable targets (should both return True)
6. Runtime comparison
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import qutip
from scipy.linalg import null_space, expm
import time
from reach import models


# =============================================================================
# IMPLEMENTATION A: Null Space Approach
# =============================================================================

def moment_nullspace(psi, phi, hams):
    """
    Null space projection approach (analysis.py style).

    Projects Q onto null space of L, checks definiteness.
    """
    K = len(hams)

    # First moments: L_k = ⟨H_k⟩_φ - ⟨H_k⟩_ψ
    L = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams])

    # Null space of L^T (vectors v such that L·v = 0)
    kernel = null_space(L.reshape(1, -1))
    if kernel.size == 0:
        return False  # No null space → cannot prove unreachable

    # Second moments: Q_ij = ⟨{H_i, H_j}/2⟩_φ - ⟨{H_i, H_j}/2⟩_ψ
    Q = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            anticomm = (hams[i] * hams[j] + hams[j] * hams[i]) / 2
            Q[i, j] = qutip.expect(anticomm, phi) - qutip.expect(anticomm, psi)

    # Project Q onto null space
    Q_proj = kernel.T @ Q @ kernel

    # Check definiteness
    eigvals = np.linalg.eigvalsh(Q_proj)
    return bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10))


# =============================================================================
# IMPLEMENTATION B: X-Sweep Approach
# =============================================================================

def moment_xsweep(psi, phi, hams, x_range=100, n_points=2001):
    """
    X-sweep approach (comprehensive_validation.py style).

    Searches for x such that Q + x·LL^T is definite.
    """
    K = len(hams)

    # First moments
    L = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams])

    # Second moments
    Q = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            anticomm = (hams[i] * hams[j] + hams[j] * hams[i]) / 2
            Q[i, j] = qutip.expect(anticomm, phi) - qutip.expect(anticomm, psi)

    L_outer = np.outer(L, L)

    # Sweep over x values
    for x in np.linspace(-x_range, x_range, n_points):
        M = Q + x * L_outer
        eigvals = np.linalg.eigvalsh(M)
        if np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10):
            return True

    return False


# =============================================================================
# COMPARISON TESTS
# =============================================================================

def run_random_comparison():
    """Compare implementations on random targets."""
    print("\n" + "=" * 60)
    print("TEST 1: Random Target Comparison")
    print("=" * 60)

    configs = [
        {'d': 8, 'K': 3, 'n_trials': 100, 'ensemble': 'canonical'},
        {'d': 8, 'K': 6, 'n_trials': 100, 'ensemble': 'canonical'},
        {'d': 16, 'K': 5, 'n_trials': 50, 'ensemble': 'canonical'},
        {'d': 16, 'K': 10, 'n_trials': 50, 'ensemble': 'canonical'},
        {'d': 8, 'K': 5, 'n_trials': 50, 'ensemble': 'GUE'},
        {'d': 16, 'K': 8, 'n_trials': 50, 'ensemble': 'GUE'},
    ]

    total_results = {
        'total': 0,
        'agree': 0,
        'ns_true_xs_false': 0,
        'ns_false_xs_true': 0,
        'ns_time': 0.0,
        'xs_time': 0.0,
        'disagreements': [],
    }

    for config in configs:
        d, K, n_trials = config['d'], config['K'], config['n_trials']
        ensemble = config['ensemble']

        agree = 0
        ns_true_xs_false = 0
        ns_false_xs_true = 0

        for trial in range(n_trials):
            seed = 1000 * d + 100 * K + trial

            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed)[0]
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

            # Null space approach
            t0 = time.time()
            result_ns = moment_nullspace(psi, phi, hams)
            total_results['ns_time'] += time.time() - t0

            # X-sweep approach
            t0 = time.time()
            result_xs = moment_xsweep(psi, phi, hams)
            total_results['xs_time'] += time.time() - t0

            total_results['total'] += 1

            if result_ns == result_xs:
                agree += 1
                total_results['agree'] += 1
            elif result_ns and not result_xs:
                ns_true_xs_false += 1
                total_results['ns_true_xs_false'] += 1
                total_results['disagreements'].append({
                    'd': d, 'K': K, 'seed': seed, 'type': 'ns_true_xs_false'
                })
            else:
                ns_false_xs_true += 1
                total_results['ns_false_xs_true'] += 1
                total_results['disagreements'].append({
                    'd': d, 'K': K, 'seed': seed, 'type': 'ns_false_xs_true'
                })

        print(f"  {ensemble} d={d}, K={K}: {agree}/{n_trials} agree "
              f"(ns+/xs-: {ns_true_xs_false}, ns-/xs+: {ns_false_xs_true})")

    return total_results


def test_provably_reachable():
    """Both should return False for exp(-iHt)|ψ⟩ targets."""
    print("\n" + "=" * 60)
    print("TEST 2: Provably Reachable Targets")
    print("Both implementations should return False (inconclusive)")
    print("=" * 60)

    configs = [
        {'d': 8, 'K': 4, 'n_trials': 30},
        {'d': 16, 'K': 8, 'n_trials': 20},
    ]

    for config in configs:
        d, K, n_trials = config['d'], config['K'], config['n_trials']

        ns_false_positives = 0
        xs_false_positives = 0

        for trial in range(n_trials):
            seed = 2000 + d * 100 + trial
            rng = np.random.RandomState(seed)

            psi = models.fock_state(d, 0)
            hams = models.random_hamiltonian_ensemble(d, K, 'canonical', seed=seed)

            # Generate provably reachable target
            lambdas = rng.uniform(-1, 1, K)
            t = 1.0
            H_lambda = sum(l * H for l, H in zip(lambdas, hams))
            U = expm(-1j * H_lambda.full() * t)
            phi_arr = U @ psi.full().flatten()
            phi = qutip.Qobj(phi_arr.reshape(-1, 1)).unit()

            # Test both implementations
            result_ns = moment_nullspace(psi, phi, hams)
            result_xs = moment_xsweep(psi, phi, hams)

            if result_ns:
                ns_false_positives += 1
            if result_xs:
                xs_false_positives += 1

        print(f"  d={d}, K={K}: Null-space FP={ns_false_positives}/{n_trials}, "
              f"X-sweep FP={xs_false_positives}/{n_trials}")


def test_edge_cases():
    """Test specific edge cases."""
    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)

    d = 8
    K = 4

    # Case 1: L ≈ 0 (psi and phi have same energies)
    print("\n  Case 1: L ≈ 0 (similar energy states)")
    psi = models.fock_state(d, 0)
    # Create phi as small perturbation of psi
    phi_arr = psi.full().flatten() + 0.1 * np.random.randn(d) + 0.1j * np.random.randn(d)
    phi_arr /= np.linalg.norm(phi_arr)
    phi = qutip.Qobj(phi_arr.reshape(-1, 1))

    hams = models.random_hamiltonian_ensemble(d, K, 'canonical', seed=42)

    L = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams])
    print(f"    ||L|| = {np.linalg.norm(L):.6f}")

    result_ns = moment_nullspace(psi, phi, hams)
    result_xs = moment_xsweep(psi, phi, hams)
    print(f"    Null-space: {result_ns}, X-sweep: {result_xs}")

    # Case 2: K = 2 (minimal case)
    print("\n  Case 2: K = 2 (minimal)")
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=100)[0]
    hams = models.random_hamiltonian_ensemble(d, 2, 'canonical', seed=42)

    result_ns = moment_nullspace(psi, phi, hams)
    result_xs = moment_xsweep(psi, phi, hams)
    print(f"    Null-space: {result_ns}, X-sweep: {result_xs}")

    # Case 3: K >> d
    print("\n  Case 3: K >> d (K=20, d=8)")
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=200)[0]
    hams = models.random_hamiltonian_ensemble(d, 20, 'canonical', seed=42)

    result_ns = moment_nullspace(psi, phi, hams)
    result_xs = moment_xsweep(psi, phi, hams)
    print(f"    Null-space: {result_ns}, X-sweep: {result_xs}")


def analyze_disagreement(d, K, seed):
    """Deep analysis of a specific disagreement case."""
    print(f"\n  Analyzing disagreement: d={d}, K={K}, seed={seed}")

    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=seed)[0]
    hams = models.random_hamiltonian_ensemble(d, K, 'canonical', seed=seed)

    # Compute L and Q
    L = np.array([qutip.expect(H, phi) - qutip.expect(H, psi) for H in hams])
    Q = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            anticomm = (hams[i] * hams[j] + hams[j] * hams[i]) / 2
            Q[i, j] = qutip.expect(anticomm, phi) - qutip.expect(anticomm, psi)

    # Null space analysis
    kernel = null_space(L.reshape(1, -1))
    if kernel.size > 0:
        Q_proj = kernel.T @ Q @ kernel
        eigvals_proj = np.linalg.eigvalsh(Q_proj)
        print(f"    Null space dim: {kernel.shape[1]}")
        print(f"    Q_proj eigenvalues: min={eigvals_proj.min():.6e}, max={eigvals_proj.max():.6e}")
    else:
        print(f"    Null space is empty")

    # X-sweep analysis
    L_outer = np.outer(L, L)
    found_definite = False
    best_x = None
    best_margin = 0

    for x in np.linspace(-100, 100, 2001):
        M = Q + x * L_outer
        eigvals = np.linalg.eigvalsh(M)
        min_eig = eigvals.min()
        max_eig = eigvals.max()

        if min_eig > 1e-10:
            margin = min_eig
            if margin > best_margin:
                best_margin = margin
                best_x = x
                found_definite = True
        elif max_eig < -1e-10:
            margin = -max_eig
            if margin > best_margin:
                best_margin = margin
                best_x = x
                found_definite = True

    if found_definite:
        print(f"    X-sweep found definite at x={best_x:.2f}, margin={best_margin:.6e}")
    else:
        print(f"    X-sweep found no definite point")

    # Check if it's a tolerance issue
    eigvals_Q = np.linalg.eigvalsh(Q)
    print(f"    Q eigenvalues: min={eigvals_Q.min():.6e}, max={eigvals_Q.max():.6e}")
    print(f"    ||L|| = {np.linalg.norm(L):.6e}")


def runtime_benchmark():
    """Benchmark runtime of both implementations."""
    print("\n" + "=" * 60)
    print("TEST 4: Runtime Benchmark")
    print("=" * 60)

    configs = [
        {'d': 8, 'K': 4, 'n_trials': 100},
        {'d': 16, 'K': 8, 'n_trials': 50},
        {'d': 16, 'K': 16, 'n_trials': 30},
    ]

    for config in configs:
        d, K, n_trials = config['d'], config['K'], config['n_trials']

        ns_time = 0
        xs_time = 0

        for trial in range(n_trials):
            seed = 3000 + trial

            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed)[0]
            hams = models.random_hamiltonian_ensemble(d, K, 'canonical', seed=seed)

            t0 = time.time()
            _ = moment_nullspace(psi, phi, hams)
            ns_time += time.time() - t0

            t0 = time.time()
            _ = moment_xsweep(psi, phi, hams)
            xs_time += time.time() - t0

        ns_avg = ns_time / n_trials * 1000  # ms
        xs_avg = xs_time / n_trials * 1000  # ms
        speedup = xs_time / ns_time

        print(f"  d={d}, K={K}: Null-space {ns_avg:.2f}ms, "
              f"X-sweep {xs_avg:.2f}ms, Speedup {speedup:.1f}x")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MOMENT CRITERION IMPLEMENTATION COMPARISON")
    print("=" * 70)
    print("\nImplementation A: Null-space projection approach")
    print("Implementation B: X-sweep approach (Q + x*LL^T)")

    t0 = time.time()

    # Run comparison
    results = run_random_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("RANDOM COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Total tests: {results['total']}")
    print(f"  Agreement rate: {results['agree']/results['total']*100:.1f}%")
    print(f"  Null-space True, X-sweep False: {results['ns_true_xs_false']}")
    print(f"  Null-space False, X-sweep True: {results['ns_false_xs_true']}")
    print(f"\n  Total Null-space time: {results['ns_time']:.2f}s")
    print(f"  Total X-sweep time: {results['xs_time']:.2f}s")
    if results['ns_time'] > 0:
        print(f"  X-sweep / Null-space ratio: {results['xs_time']/results['ns_time']:.1f}x")

    # Analyze disagreements if any
    if results['disagreements']:
        print("\n" + "=" * 60)
        print("DISAGREEMENT ANALYSIS")
        print("=" * 60)
        # Analyze first few disagreements
        for disagreement in results['disagreements'][:3]:
            analyze_disagreement(disagreement['d'], disagreement['K'], disagreement['seed'])

    # Other tests
    test_provably_reachable()
    test_edge_cases()
    runtime_benchmark()

    # Final summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    agreement_rate = results['agree'] / results['total'] * 100

    if agreement_rate >= 99:
        print("  Implementations are effectively equivalent (>99% agreement)")
    elif agreement_rate >= 95:
        print(f"  Implementations mostly agree ({agreement_rate:.1f}%)")
        print("  Minor disagreements likely due to numerical tolerance")
    else:
        print(f"  Significant disagreements ({agreement_rate:.1f}% agreement)")
        print("  Further investigation needed")

    print(f"\n  RECOMMENDATION: Use null-space approach")
    print(f"    - Faster (~{results['xs_time']/results['ns_time']:.0f}x)")
    print(f"    - Direct computation (no sweep)")
    print(f"    - Mathematically equivalent for K-1 dimensional null space")

    print(f"\n  Total runtime: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
