#!/usr/bin/env python3
"""
Quick Heuristic Test: Can ANY λ make Floquet beat Static?

Tests whether there exist coupling coefficients λ such that Floquet O2
achieves higher fidelity than the best static fidelity.

This avoids implementing full parameterized optimization for Floquet
by simply trying many random λ values and seeing if any work.

If YES → Worth implementing proper optimization
If NO → Hypothesis likely rejected
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from reach import floquet, models, optimization, states

def test_lambda_search(K, n_qubits=4, n_lambda_trials=50, seed=42):
    """
    Try multiple random λ values for Floquet, compare with optimized static.

    Args:
        K: Number of operators
        n_qubits: System size
        n_lambda_trials: How many random λ to try
        seed: Random seed

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"FLOQUET λ SEARCH: Can we beat static with K={K}?")
    print(f"{'='*70}\n")

    # States
    d = 2**n_qubits
    psi = states.computational_basis(n_qubits, '0' * n_qubits)
    target_states_dict = states.create_target_states(n_qubits)
    phi = target_states_dict['ghz']

    classical_overlap = np.abs(phi.conj() @ psi)**2
    print(f"State pair: |0000⟩ → GHZ")
    print(f"Classical overlap: {classical_overlap:.4f}")
    print(f"Number of λ trials: {n_lambda_trials}\n")

    # Generate Hamiltonians (use same set for all trials)
    hams_qutip = models.random_hamiltonian_ensemble(
        dim=d, k=K, ensemble="GEO2", nx=2, ny=2, seed=seed
    )
    hams = floquet.hamiltonians_to_numpy(hams_qutip)

    # BASELINE: Optimized static
    print("BASELINE: Static with optimized λ")
    print("-" * 70)
    fid_static_opt, lambdas_static_opt, t_static = optimization.optimize_fidelity_parameterized(
        psi, phi, hams, t_max=50.0, n_trials=10, seed=seed
    )
    print(f"Best static fidelity: {fid_static_opt:.6f} (t={t_static:.3f})")
    print(f"Optimal λ: {lambdas_static_opt}\n")

    # TEST: Floquet with multiple random λ
    print("TEST: Floquet O2 with random λ values")
    print("-" * 70)

    T = 1.0
    t_max = 50.0
    driving = floquet.create_driving_functions(K, 'bichromatic', T, seed=seed)

    best_fid_f2 = 0.0
    best_lambdas_f2 = None
    best_t_f2 = 0.0

    fidelities_f2 = []
    rng = np.random.RandomState(seed)

    for trial in range(n_lambda_trials):
        # Random λ
        lambdas_trial = rng.randn(K) / np.sqrt(K)

        # Compute Floquet O2
        H_F2 = floquet.compute_floquet_hamiltonian(hams, lambdas_trial, driving, T, order=2)

        # Optimize time
        fid_f2, t_f2 = optimization.optimize_fidelity(psi, phi, H_F2, t_max)
        fidelities_f2.append(fid_f2)

        if fid_f2 > best_fid_f2:
            best_fid_f2 = fid_f2
            best_lambdas_f2 = lambdas_trial.copy()
            best_t_f2 = t_f2

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{n_lambda_trials}: Best so far = {best_fid_f2:.6f}")

    print(f"\nBest Floquet O2 fidelity: {best_fid_f2:.6f} (t={best_t_f2:.3f})")
    print(f"Best λ: {best_lambdas_f2}")

    # Statistics
    mean_f2 = np.mean(fidelities_f2)
    std_f2 = np.std(fidelities_f2)
    max_f2 = np.max(fidelities_f2)

    print(f"\nFloquet O2 statistics:")
    print(f"  Mean: {mean_f2:.6f} ± {std_f2:.6f}")
    print(f"  Max:  {max_f2:.6f}")
    print(f"  Min:  {np.min(fidelities_f2):.6f}")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}\n")

    improvement = best_fid_f2 - fid_static_opt
    improvement_pct = 100 * improvement / max(fid_static_opt, 1e-10)

    print(f"Static (optimized λ): {fid_static_opt:.6f}")
    print(f"Floquet O2 (best λ):  {best_fid_f2:.6f}")
    print(f"Difference:           {improvement:+.6f} ({improvement_pct:+.1f}%)\n")

    # Verdict
    print(f"{'='*70}")
    print("VERDICT")
    print(f"{'='*70}\n")

    if best_fid_f2 > fid_static_opt + 0.05:
        print("✅ FLOQUET WINS!")
        print(f"   Floquet O2 achieved {100*improvement:.1f}% higher fidelity")
        print("   Hypothesis is VIABLE - worth implementing full optimization")
        verdict = "FLOQUET_WINS"
    elif best_fid_f2 > fid_static_opt - 0.05:
        print("≈ TIE")
        print(f"   Difference is small ({100*abs(improvement):.1f}%)")
        print("   May need proper optimization to determine winner")
        verdict = "TIE"
    else:
        print("❌ STATIC WINS")
        print(f"   Even best Floquet λ is {-100*improvement:.1f}% worse")
        print("   Hypothesis likely REJECTED for this state pair")
        print("   BUT: Should test other state pairs before giving up")
        verdict = "STATIC_WINS"

    # Additional insights
    print(f"\nAdditional insights:")
    fraction_better_than_classical = np.sum(np.array(fidelities_f2) > classical_overlap) / len(fidelities_f2)
    print(f"  - {100*fraction_better_than_classical:.0f}% of random λ beat classical overlap")

    if mean_f2 < classical_overlap + 0.05:
        print(f"  ⚠️  Most random λ stuck near classical overlap")
        print(f"  → Suggests Floquet H_F with random λ can't reach GHZ easily")

    return {
        'K': K,
        'fidelity_static_optimized': fid_static_opt,
        'fidelity_floquet_best': best_fid_f2,
        'fidelity_floquet_mean': mean_f2,
        'fidelity_floquet_std': std_f2,
        'improvement': improvement,
        'improvement_percent': improvement_pct,
        'verdict': verdict,
        'lambdas_static': lambdas_static_opt,
        'lambdas_floquet_best': best_lambdas_f2,
        'all_fidelities_floquet': fidelities_f2,
    }


def main():
    """Run λ search for K=16."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "FLOQUET λ SEARCH TEST" + " " * 28 + "║")
    print("╚" + "═" * 68 + "╝")

    # Test at K=16 where static achieved 0.946
    results = test_lambda_search(K=16, n_lambda_trials=50, seed=42)

    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*70}\n")

    if results['verdict'] == "FLOQUET_WINS":
        print("📊 Implement optimize_floquet_fidelity_parameterized()")
        print("   Full λ optimization for Floquet will likely show clear advantage")
    elif results['verdict'] == "TIE":
        print("🤔 Consider implementing full optimization OR test other states")
        print("   Current results inconclusive - need more rigorous test")
    else:
        print("🔬 Test different state pairs first")
        print("   Try |0000⟩ → W-state or cluster before giving up")
        print("   May be that GHZ specifically doesn't benefit from Floquet")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
