#!/usr/bin/env python3
"""
Quick GEO2 production run with K-cap to avoid expensive tail points.

The spectral transition for GEO2 occurs at rho ~ 0.02-0.05, so K_cap=100
captures all interesting physics while avoiding K>100 points that are
computationally expensive and scientifically uninformative (all P=0).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout = sys.stderr  # Unbuffered

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.linalg import null_space
import qutip
import time

from reach import models, optimize, mathematics

# ============================================================================
# CONFIGURATION
# ============================================================================

DIMENSIONS = [8, 16, 32, 64]
TAU_VALUES = [0.99]
N_TRIALS = 50
K_CAP = 100

# Dense sampling in transition region
RHO_VALUES = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04,
              0.05, 0.06, 0.08, 0.10]

GEO2_LATTICES = {
    8:  {'nx': 1, 'ny': 3},
    16: {'nx': 2, 'ny': 2},
    32: {'nx': 1, 'ny': 5},
    64: {'nx': 2, 'ny': 3},
}


def get_settings(d):
    if d <= 32:
        return {'maxiter': 100, 'restarts': 3, 'ftol': 1e-6}
    else:
        return {'maxiter': 150, 'restarts': 3, 'ftol': 1e-6}


def moment_criterion_qutip(psi, phi, hams):
    K = len(hams)
    energies_psi = np.array([qutip.expect(H, psi) for H in hams])
    energies_phi = np.array([qutip.expect(H, phi) for H in hams])
    diff = energies_phi - energies_psi
    kernel = null_space(diff.reshape(1, -1))
    if kernel.size == 0:
        return False
    anticomms = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(i, K):
            ac = (hams[i] * hams[j] + hams[j] * hams[i]) / 2
            anticomms[i][j] = ac
            anticomms[j][i] = ac
    sq_psi = np.array([[qutip.expect(anticomms[i][j], psi)
                        for j in range(K)] for i in range(K)])
    sq_phi = np.array([[qutip.expect(anticomms[i][j], phi)
                        for j in range(K)] for i in range(K)])
    m_final = kernel.T @ (sq_phi - sq_psi) @ kernel
    eigvals = np.linalg.eigvalsh(m_final)
    return bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10))


def main():
    output_dir = Path("data/production")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print("=" * 70)
    print("QUICK GEO2 PRODUCTION RUN (K-capped)")
    print(f"K_cap = {K_CAP}, n_trials = {N_TRIALS}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for d in DIMENSIONS:
        lattice = GEO2_LATTICES[d]
        opt = get_settings(d)

        print(f"\n--- d={d} ({lattice['nx']}x{lattice['ny']} lattice), "
              f"maxiter={opt['maxiter']} ---")

        for rho in RHO_VALUES:
            K = max(2, round(rho * d**2))

            if K > K_CAP:
                print(f"  d={d}, rho={rho:.4f}, K={K}: SKIPPED (K > {K_CAP})")
                continue

            t0 = time.time()

            spectral_scores = []
            krylov_scores = []
            moment_unreach = 0

            for trial in range(N_TRIALS):
                seed = 2000 * K + trial

                psi = models.fock_state(d, 0)
                phi = models.random_states(1, d, seed=seed + 500)[0]
                hams = models.random_hamiltonian_ensemble(
                    d, K, "GEO2", seed=seed, **lattice
                )

                # Spectral
                res_s = optimize.maximize_spectral_overlap(
                    psi, phi, hams, seed=seed + 100, **opt
                )
                spectral_scores.append(res_s['best_value'])

                # Krylov
                m = min(K, d)
                res_k = optimize.maximize_krylov_score(
                    psi, phi, hams, m=m, seed=seed + 200, **opt
                )
                krylov_scores.append(res_k['best_value'])

                # Moment
                try:
                    if moment_criterion_qutip(psi, phi, hams):
                        moment_unreach += 1
                except Exception:
                    pass

            elapsed = time.time() - t0

            for tau in TAU_VALUES:
                spectral_unreach = sum(1 for s in spectral_scores if s < tau)
                krylov_unreach = sum(1 for s in krylov_scores if s < tau)

                results.append({
                    'd': d, 'K': K, 'rho': rho, 'tau': tau,
                    'n_trials': N_TRIALS,
                    'spectral_P': spectral_unreach / N_TRIALS,
                    'krylov_P': krylov_unreach / N_TRIALS,
                    'moment_P': moment_unreach / N_TRIALS,
                    'spectral_mean': float(np.mean(spectral_scores)),
                    'spectral_std': float(np.std(spectral_scores)),
                    'krylov_mean': float(np.mean(krylov_scores)),
                    'krylov_std': float(np.std(krylov_scores)),
                    'ensemble': 'GEO2',
                })

            r = results[-1]
            print(f"  d={d}, rho={rho:.4f}, K={K}: "
                  f"P_s={r['spectral_P']:.2f}, "
                  f"P_k={r['krylov_P']:.2f}, "
                  f"P_m={r['moment_P']:.2f} "
                  f"[{elapsed:.1f}s]")

        # Checkpoint
        df_tmp = pd.DataFrame(results)
        checkpoint = output_dir / f"GEO2_quick_checkpoint_d{d}_{timestamp}.pkl"
        with open(checkpoint, 'wb') as f:
            pickle.dump({'results': results, 'd': d}, f)
        print(f"  Checkpoint: {checkpoint}")

    # Save final
    df = pd.DataFrame(results)
    csv_file = output_dir / f"GEO2_quick_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    print(f"\n{'='*70}")
    print(f"COMPLETE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Saved: {csv_file}")
    print(f"Total data points: {len(df)}")
    for d in sorted(df['d'].unique()):
        dd = df[df['d'] == d]
        print(f"  d={d}: {len(dd)} points, rho=[{dd['rho'].min():.4f}, {dd['rho'].max():.4f}]")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
