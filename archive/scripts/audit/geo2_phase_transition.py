#!/usr/bin/env python3
"""
Find the actual GEO2 phase transition at ultra-low densities.

With proper optimizer settings (maxiter≥50, restarts≥2), P(unreachable)≈0
at ρ=0.05 for all dimensions. The real transition must occur at much lower ρ.

This script sweeps ultra-low densities to find ρ_c for each dimension,
using all three criteria (Spectral, Krylov, Moment).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
import csv
from scipy.linalg import null_space
import qutip
from reach import models, optimize, mathematics, settings


def moment_criterion_qutip(psi, phi, hams):
    """Moment criterion using QuTiP (matches analysis.py implementation)."""
    K = len(hams)
    energies_psi = np.array([qutip.expect(H, psi) for H in hams])
    energies_phi = np.array([qutip.expect(H, phi) for H in hams])
    diff = energies_phi - energies_psi
    kernel = null_space(diff.reshape(1, -1))
    if kernel.size == 0:
        return False
    anticomms = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            anticomms[i][j] = (hams[i] * hams[j] + hams[j] * hams[i]) / 2
    sq_psi = np.array([[qutip.expect(anticomms[i][j], psi) for j in range(K)] for i in range(K)])
    sq_phi = np.array([[qutip.expect(anticomms[i][j], phi) for j in range(K)] for i in range(K)])
    m_final = kernel.T @ (sq_phi - sq_psi) @ kernel
    eigvals = np.linalg.eigvalsh(m_final)
    return bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10))


ACCURATE_SETTINGS = {
    'maxiter': 100,
    'restarts': 2,
    'ftol': 1e-6,
}


def run_density_sweep_accurate(d, K_values, ensemble, n_trials=50, tau=0.99, **model_params):
    """
    Sweep over K values (corresponding to different densities) with accurate
    optimizer settings and all three criteria.

    Returns list of dicts with results per K.
    """
    results = []

    for K in K_values:
        if K < 2:
            continue

        rho = K / d**2
        t0 = time.time()

        spectral_unreach = 0
        krylov_unreach = 0
        moment_unreach = 0
        spectral_scores = []
        krylov_scores = []

        for trial in range(n_trials):
            seed = 1000 * K + trial
            psi = models.fock_state(d, 0)
            phi = models.random_states(1, d, seed=seed + 500)[0]
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **model_params)

            # Spectral criterion (optimized)
            res_s = optimize.maximize_spectral_overlap(
                psi, phi, hams,
                seed=seed + 100,
                **ACCURATE_SETTINGS,
            )
            s_star = res_s['best_value']
            spectral_scores.append(s_star)
            if s_star < tau:
                spectral_unreach += 1

            # Krylov criterion (optimized)
            res_k = optimize.maximize_krylov_score(
                psi, phi, hams,
                seed=seed + 200,
                **ACCURATE_SETTINGS,
            )
            r_star = res_k['best_value']
            krylov_scores.append(r_star)
            if r_star < tau:
                krylov_unreach += 1

            # Moment criterion (no optimization needed)
            if moment_criterion_qutip(psi, phi, hams):
                moment_unreach += 1

        elapsed = time.time() - t0

        row = {
            'd': d,
            'K': K,
            'rho': rho,
            'n_trials': n_trials,
            'tau': tau,
            'spectral_P_unreach': spectral_unreach / n_trials,
            'krylov_P_unreach': krylov_unreach / n_trials,
            'moment_P_unreach': moment_unreach / n_trials,
            'spectral_mean': np.mean(spectral_scores),
            'krylov_mean': np.mean(krylov_scores),
            'time_s': elapsed,
        }
        results.append(row)

        print(f"  d={d:3d}, K={K:3d} (ρ={rho:.4f}): "
              f"P_s={row['spectral_P_unreach']:.2f}, "
              f"P_k={row['krylov_P_unreach']:.2f}, "
              f"P_m={row['moment_P_unreach']:.2f}, "
              f"<S*>={row['spectral_mean']:.4f}, "
              f"<R*>={row['krylov_mean']:.4f}, "
              f"[{elapsed:.1f}s]", flush=True)

    return results


def main():
    print("=" * 70)
    print("GEO2 PHASE TRANSITION AT ULTRA-LOW DENSITIES")
    print("Accurate optimizer settings: maxiter=100, restarts=2, ftol=1e-6")
    print("=" * 70)

    # Configurations: (label, d, K_values, model_params)
    # d=16: d²=256, so K=2..26 gives ρ=0.008..0.10
    # d=32: d²=1024, so K=3..51 gives ρ=0.003..0.05
    configs = [
        ("GEO2 d=16 (2x2 lattice)", 16,
         [2, 3, 4, 5, 6, 7, 8, 10, 13, 16, 20, 26],
         {'nx': 2, 'ny': 2}),
        ("GEO2 d=32 (1x5 lattice)", 32,
         [3, 5, 8, 10, 13, 16, 20, 26, 32, 40, 51],
         {'nx': 1, 'ny': 5}),
    ]

    all_results = []

    for label, d, K_values, model_params in configs:
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"K values: {K_values}")
        print(f"ρ range: {min(K_values)/d**2:.4f} to {max(K_values)/d**2:.4f}")
        print(f"{'=' * 70}")

        results = run_density_sweep_accurate(
            d, K_values, "GEO2", n_trials=30, tau=0.99, **model_params
        )
        all_results.extend(results)

    # Write CSV
    outdir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'geo2_phase_transition.csv')

    fieldnames = ['d', 'K', 'rho', 'n_trials', 'tau',
                  'spectral_P_unreach', 'krylov_P_unreach', 'moment_P_unreach',
                  'spectral_mean', 'krylov_mean', 'time_s']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Estimated critical densities (P_unreach drops below 0.5)")
    print("=" * 70)

    for d_val in [16, 32]:
        d_results = [r for r in all_results if r['d'] == d_val]
        if not d_results:
            continue

        print(f"\n  d={d_val}:")
        for criterion in ['spectral', 'krylov', 'moment']:
            key = f'{criterion}_P_unreach'
            # Find where P drops below 0.5
            below_half = [r for r in d_results if r[key] < 0.5]
            if below_half:
                rho_c = min(r['rho'] for r in below_half)
                K_c = min(r['K'] for r in below_half)
                print(f"    {criterion:10s}: ρ_c ≈ {rho_c:.4f} (K_c ≈ {K_c})")
            else:
                max_rho = max(r['rho'] for r in d_results)
                print(f"    {criterion:10s}: ρ_c > {max_rho:.4f} (not reached)")


if __name__ == "__main__":
    main()
