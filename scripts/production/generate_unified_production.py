#!/usr/bin/env python3
"""
Unified Production Data Generation for Canonical and GEO2 Ensembles.

Generates comparable datasets using IDENTICAL optimizer settings for fair
scientific comparison between ensemble types.

Key Features:
- Universal optimizer settings (same for both ensembles)
- Analytical gradient for spectral criterion
- Dimension-adaptive maxiter for large K
- Checkpointing after each (ensemble, dimension) combination
- CSV output for incremental plotting

Expected Runtime (with analytical gradient):
- Canonical d=8,16,32,64:  ~2-3 hours total
- GEO2 d=8,16,32,64:       ~3-5 hours total
- Combined:                ~5-8 hours

Output:
- data/production/canonical_unified_YYYYMMDD.pkl
- data/production/geo2_unified_YYYYMMDD.pkl
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.stdout = sys.stderr  # Unbuffered output

import pickle
import time as time_module
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import traceback
from scipy.linalg import null_space
import qutip

from reach import models, optimize, mathematics, settings as reach_settings

# ============================================================================
# UNIVERSAL PRODUCTION SETTINGS
# ============================================================================

def get_universal_settings(d):
    """
    Get universal optimizer settings that work for ALL ensembles.

    These settings were validated on GEO2 d=64 (K=205) with 0% false-unreachable
    rate. Since canonical converges faster (sparse operators), these settings
    are MORE than sufficient for canonical.
    """
    if d <= 32:
        maxiter = 100
    elif d <= 64:
        maxiter = 150
    else:
        maxiter = 200

    return {
        'method': 'L-BFGS-B',
        'maxiter': maxiter,
        'restarts': 3,
        'ftol': 1e-6,
    }

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

# Power-of-2 dimensions for both ensembles (enables direct comparison)
DIMENSIONS = [8, 16, 32, 64]

# Tau thresholds
TAU_VALUES = [0.99, 0.95, 0.90]

# Density range
RHO_MIN = 0.01
RHO_MAX_CANONICAL = 0.40  # Canonical transitions later
RHO_MAX_GEO2 = 0.20       # GEO2 transitions earlier
N_RHO_POINTS = 20

# Monte Carlo sampling
N_TRIALS = 100  # Per (d, rho) point

# GEO2 lattice configurations
GEO2_LATTICES = {
    8:  {'nx': 1, 'ny': 3},   # 3 qubits
    16: {'nx': 2, 'ny': 2},   # 4 qubits
    32: {'nx': 1, 'ny': 5},   # 5 qubits
    64: {'nx': 2, 'ny': 3},   # 6 qubits
}

# ============================================================================
# MOMENT CRITERION (QuTiP implementation)
# ============================================================================

def moment_criterion_qutip(psi, phi, hams):
    """
    Moment criterion using QuTiP (matches analysis.py implementation).

    Returns True if criterion PROVES target is unreachable.
    """
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

# ============================================================================
# SINGLE POINT COMPUTATION
# ============================================================================

def compute_single_point(d, rho, tau_values, ensemble, n_trials, **model_params):
    """
    Compute P(unreachable) at a single (d, rho) point for all three criteria
    and all tau values simultaneously.

    Returns list of result dicts (one per tau).
    """
    K = max(2, round(rho * d**2))
    opt = get_universal_settings(d)

    spectral_scores = []
    krylov_scores = []
    moment_unreach = 0

    for trial in range(n_trials):
        seed = 1000 * K + trial

        # Generate system
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed + 500)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **model_params)

        # Spectral criterion (with analytical gradient)
        res_s = optimize.maximize_spectral_overlap(
            psi, phi, hams, seed=seed + 100,
            maxiter=opt['maxiter'], restarts=opt['restarts'], ftol=opt['ftol']
        )
        spectral_scores.append(res_s['best_value'])

        # Krylov criterion
        m = min(K, d)
        res_k = optimize.maximize_krylov_score(
            psi, phi, hams, m=m, seed=seed + 200,
            maxiter=opt['maxiter'], restarts=opt['restarts'], ftol=opt['ftol']
        )
        krylov_scores.append(res_k['best_value'])

        # Moment criterion (no optimization)
        try:
            if moment_criterion_qutip(psi, phi, hams):
                moment_unreach += 1
        except Exception:
            pass  # Moment may fail for some configs; count as reachable

    # Compute results for each tau
    results = []
    for tau in tau_values:
        spectral_unreach = sum(1 for s in spectral_scores if s < tau)
        krylov_unreach = sum(1 for s in krylov_scores if s < tau)

        results.append({
            'd': d,
            'K': K,
            'rho': rho,
            'tau': tau,
            'n_trials': n_trials,
            'spectral_P': spectral_unreach / n_trials,
            'krylov_P': krylov_unreach / n_trials,
            'moment_P': moment_unreach / n_trials,
            'spectral_mean': float(np.mean(spectral_scores)),
            'spectral_std': float(np.std(spectral_scores)),
            'krylov_mean': float(np.mean(krylov_scores)),
            'krylov_std': float(np.std(krylov_scores)),
        })

    return results

# ============================================================================
# ENSEMBLE SWEEP
# ============================================================================

def run_ensemble_sweep(ensemble, dimensions, rho_range, tau_values, n_trials,
                       output_dir, model_params_func):
    """
    Run full sweep for one ensemble.
    """
    rho_min, rho_max, n_points = rho_range
    rho_values = np.logspace(np.log10(rho_min), np.log10(rho_max), n_points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    print(f"\n{'='*70}")
    print(f"ENSEMBLE: {ensemble.upper()}")
    print(f"Dimensions: {dimensions}")
    print(f"rho range: {rho_min:.3f} to {rho_max:.3f} ({n_points} points, log-spaced)")
    print(f"tau values: {tau_values}")
    print(f"Trials per point: {n_trials}")
    print(f"{'='*70}\n")

    total_points = len(dimensions) * len(rho_values)
    completed = 0

    for d in dimensions:
        model_params = model_params_func(d)
        opt = get_universal_settings(d)

        print(f"\n--- d={d}, maxiter={opt['maxiter']} ---")

        for rho in rho_values:
            K = max(2, round(rho * d**2))

            t0 = time_module.time()

            results = compute_single_point(
                d, rho, tau_values, ensemble, n_trials, **model_params
            )
            for r in results:
                r['ensemble'] = ensemble
            all_results.extend(results)

            elapsed = time_module.time() - t0
            completed += 1

            # Show result for tau=0.99 as representative
            r99 = next((r for r in results if r['tau'] == 0.99), results[0])
            print(f"  d={d}, rho={rho:.4f}, K={K}: "
                  f"P_s={r99['spectral_P']:.2f}, "
                  f"P_k={r99['krylov_P']:.2f}, "
                  f"P_m={r99['moment_P']:.2f} "
                  f"[{elapsed:.1f}s] ({completed}/{total_points})")

        # Checkpoint after each dimension
        checkpoint_file = output_dir / f"{ensemble}_checkpoint_d{d}_{timestamp}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'ensemble': ensemble,
                'timestamp': timestamp,
                'completed_dim': d,
                'results': all_results,
                'settings': {dd: get_universal_settings(dd) for dd in dimensions},
            }, f)
        print(f"  Checkpoint: {checkpoint_file}")

    # Save final results
    df = pd.DataFrame(all_results)
    csv_file = output_dir / f"{ensemble}_unified_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    pkl_file = output_dir / f"{ensemble}_unified_{timestamp}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump({
            'ensemble': ensemble,
            'timestamp': timestamp,
            'dimensions': dimensions,
            'rho_range': rho_range,
            'tau_values': tau_values,
            'n_trials': n_trials,
            'universal_settings': {d: get_universal_settings(d) for d in dimensions},
            'results': all_results,
            'dataframe': df,
        }, f)

    print(f"\nSaved: {csv_file}")
    print(f"Saved: {pkl_file}")

    return df, all_results

# ============================================================================
# MAIN
# ============================================================================

def main():
    output_dir = Path("data/production")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("UNIFIED PRODUCTION DATA GENERATION")
    print("Canonical vs GEO2 with Universal Optimizer Settings")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {output_dir}")
    print()

    # Show universal settings
    print("Universal Settings by Dimension:")
    for d in DIMENSIONS:
        opt = get_universal_settings(d)
        print(f"  d={d}: maxiter={opt['maxiter']}, restarts={opt['restarts']}, ftol={opt['ftol']}")
    print()

    # ========== CANONICAL ==========
    print("\n" + "=" * 70)
    print("PHASE 1: CANONICAL ENSEMBLE")
    print("=" * 70)

    df_canonical, results_canonical = run_ensemble_sweep(
        ensemble='canonical',
        dimensions=DIMENSIONS,
        rho_range=(RHO_MIN, RHO_MAX_CANONICAL, N_RHO_POINTS),
        tau_values=TAU_VALUES,
        n_trials=N_TRIALS,
        output_dir=output_dir,
        model_params_func=lambda d: {},
    )

    # ========== GEO2 ==========
    print("\n" + "=" * 70)
    print("PHASE 2: GEO2 ENSEMBLE")
    print("=" * 70)

    df_geo2, results_geo2 = run_ensemble_sweep(
        ensemble='GEO2',
        dimensions=DIMENSIONS,
        rho_range=(RHO_MIN, RHO_MAX_GEO2, N_RHO_POINTS),
        tau_values=TAU_VALUES,
        n_trials=N_TRIALS,
        output_dir=output_dir,
        model_params_func=lambda d: GEO2_LATTICES.get(d, {}),
    )

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("PRODUCTION RUN COMPLETE")
    print("=" * 70)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nCanonical: {len(df_canonical)} data points")
    print(f"GEO2: {len(df_geo2)} data points")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
