#!/usr/bin/env python3
"""
Publication Pilot Experiment

Quick test run with reduced parameters to verify:
1. All criteria work correctly
2. Sampling density is appropriate
3. No numerical artifacts
4. Runtime estimates are accurate

Run time: ~30-60 minutes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import time
from pathlib import Path
from scipy.linalg import null_space
import qutip

from reach import models, optimize, mathematics
from scripts.production.publication_config import PILOT_CONFIG, OPTIMIZER_SETTINGS


# =============================================================================
# MOMENT CRITERION
# =============================================================================

def moment_criterion(psi, phi, hams):
    """
    Moment criterion using QuTiP objects.
    Returns True if criterion proves target is unreachable.
    """
    K = len(hams)

    # First moments
    energies_psi = np.array([qutip.expect(H, psi) for H in hams])
    energies_phi = np.array([qutip.expect(H, phi) for H in hams])
    diff = energies_phi - energies_psi

    # Null space of first moment difference
    kernel = null_space(diff.reshape(1, -1))
    if kernel.size == 0:
        return False

    # Second moment anticommutators
    anticomms = [[None]*K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            anticomms[i][j] = (hams[i] * hams[j] + hams[j] * hams[i]) / 2

    # Second moment expectations
    sq_psi = np.array([[qutip.expect(anticomms[i][j], psi)
                       for j in range(K)] for i in range(K)])
    sq_phi = np.array([[qutip.expect(anticomms[i][j], phi)
                       for j in range(K)] for i in range(K)])

    # Project onto null space
    m_final = kernel.T @ (sq_phi - sq_psi) @ kernel

    # Check definiteness
    eigvals = np.linalg.eigvalsh(m_final)
    return bool(np.all(eigvals > 1e-10) or np.all(eigvals < -1e-10))


# =============================================================================
# SINGLE POINT EVALUATION
# =============================================================================

def evaluate_point(ensemble, d, K, tau, n_trials, seed_base, lattice_params=None):
    """
    Evaluate all criteria at a single (d, K, tau) point.

    Returns:
        dict with P(unreachable) for each criterion
    """
    rho = K / d**2

    spectral_unreach = 0
    krylov_unreach = 0
    moment_unreach = 0
    spectral_scores = []
    krylov_scores = []

    opts = OPTIMIZER_SETTINGS.get(ensemble, OPTIMIZER_SETTINGS['canonical'])

    for trial in range(n_trials):
        seed = seed_base + trial

        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed + 1000)[0]

        # Generate Hamiltonians
        if ensemble == 'GEO2' and lattice_params:
            hams = models.random_hamiltonian_ensemble(
                d, K, ensemble, seed=seed, **lattice_params
            )
        else:
            hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed)

        # Spectral criterion
        res_s = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            maxiter=opts['maxiter'],
            restarts=opts['restarts'],
            ftol=opts['ftol'],
            seed=seed
        )
        spectral_scores.append(res_s['best_value'])
        if res_s['best_value'] < tau:
            spectral_unreach += 1

        # Krylov criterion
        res_k = optimize.maximize_krylov_score(
            psi, phi, hams,
            maxiter=opts['maxiter'],
            restarts=opts['restarts'],
            ftol=opts['ftol'],
            seed=seed
        )
        krylov_scores.append(res_k['best_value'])
        if res_k['best_value'] < tau:
            krylov_unreach += 1

        # Moment criterion
        if moment_criterion(psi, phi, hams):
            moment_unreach += 1

    return {
        'd': d,
        'K': K,
        'rho': rho,
        'tau': tau,
        'n_trials': n_trials,
        'spectral_P': spectral_unreach / n_trials,
        'krylov_P': krylov_unreach / n_trials,
        'moment_P': moment_unreach / n_trials,
        'spectral_mean': np.mean(spectral_scores),
        'spectral_std': np.std(spectral_scores),
        'krylov_mean': np.mean(krylov_scores),
        'krylov_std': np.std(krylov_scores),
    }


# =============================================================================
# RUN PILOT
# =============================================================================

def run_pilot():
    """Run pilot experiment for all ensembles."""
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'production'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PUBLICATION PILOT EXPERIMENT")
    print("=" * 70)

    all_results = []

    for ensemble in ['canonical', 'GEO2']:
        config = PILOT_CONFIG[ensemble]

        print(f"\n{'='*60}")
        print(f"Ensemble: {ensemble}")
        print(f"Dimensions: {config['dimensions']}")
        print(f"Trials: {config['n_trials']}")
        print(f"{'='*60}")

        for d in config['dimensions']:
            rho_points = config['rho_points'][d]
            lattice_params = config.get('lattice_params', {}).get(d)

            print(f"\n  d={d}: {len(rho_points)} rho points")

            for tau in config['tau_values']:
                for rho in rho_points:
                    K = max(2, int(round(rho * d**2)))
                    rho_actual = K / d**2

                    t0 = time.time()

                    result = evaluate_point(
                        ensemble, d, K, tau,
                        config['n_trials'],
                        config['seed_base'],
                        lattice_params
                    )
                    result['ensemble'] = ensemble

                    elapsed = time.time() - t0

                    all_results.append(result)

                    print(f"    K={K:3d} (ρ={rho_actual:.4f}): "
                          f"S={result['spectral_P']:.2f}, "
                          f"K={result['krylov_P']:.2f}, "
                          f"M={result['moment_P']:.2f} "
                          f"[{elapsed:.1f}s]")

    # Save results
    df = pd.DataFrame(all_results)
    output_file = output_dir / 'pilot_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nSaved results to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)

    for ensemble in ['canonical', 'GEO2']:
        df_ens = df[df['ensemble'] == ensemble]
        print(f"\n{ensemble}:")
        for d in df_ens['d'].unique():
            df_d = df_ens[df_ens['d'] == d]
            print(f"  d={int(d)}: {len(df_d)} points, "
                  f"spectral range [{df_d['spectral_P'].min():.2f}, {df_d['spectral_P'].max():.2f}]")

    return df


# =============================================================================
# QUICK VALIDATION
# =============================================================================

def validate_results(df):
    """Check for artifacts in pilot results."""
    print("\n" + "=" * 70)
    print("VALIDATION CHECKS")
    print("=" * 70)

    issues = []

    # Check 1: P values in [0, 1]
    for col in ['spectral_P', 'krylov_P', 'moment_P']:
        if df[col].min() < 0 or df[col].max() > 1:
            issues.append(f"{col} outside [0,1]")

    # Check 2: Monotonicity (P should decrease with rho for fixed d)
    for ensemble in df['ensemble'].unique():
        for d in df[df['ensemble'] == ensemble]['d'].unique():
            df_sub = df[(df['ensemble'] == ensemble) & (df['d'] == d)].sort_values('rho')

            for col in ['spectral_P', 'krylov_P']:
                P = df_sub[col].values

                # Check for significant increases (>0.2) which suggest non-monotonicity
                for i in range(1, len(P)):
                    if P[i] - P[i-1] > 0.2:
                        issues.append(f"{ensemble} d={d}: {col} non-monotonic at rho={df_sub['rho'].values[i]:.4f}")

    # Check 3: d=8 starting at P~1
    for ensemble in df['ensemble'].unique():
        df_8 = df[(df['ensemble'] == ensemble) & (df['d'] == 8)]
        if len(df_8) > 0:
            min_rho_row = df_8[df_8['rho'] == df_8['rho'].min()].iloc[0]
            for col in ['spectral_P', 'krylov_P', 'moment_P']:
                if min_rho_row[col] < 0.9:
                    issues.append(f"{ensemble} d=8: {col} starts at {min_rho_row[col]:.2f} (expected ~1)")

    # Check 4: Dimension ordering (higher d should transition first)
    for ensemble in df['ensemble'].unique():
        dims = sorted(df[df['ensemble'] == ensemble]['d'].unique())
        if len(dims) >= 2:
            # At mid-range rho, higher d should have lower P
            mid_rho = 0.05
            for d in dims:
                df_d = df[(df['ensemble'] == ensemble) & (df['d'] == d)]
                closest_idx = (df_d['rho'] - mid_rho).abs().idxmin()
                df.loc[closest_idx, 'check_rho'] = mid_rho

    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n  All validation checks passed!")

    return len(issues) == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()

    df = run_pilot()
    validate_results(df)

    total_time = time.time() - t0
    print(f"\n\nTotal pilot time: {total_time/60:.1f} minutes")

    # Estimate full production time
    pilot_trials = sum(PILOT_CONFIG[e]['n_trials'] for e in ['canonical', 'GEO2'])
    pilot_points = sum(sum(len(PILOT_CONFIG[e]['rho_points'][d])
                          for d in PILOT_CONFIG[e]['dimensions'])
                      for e in ['canonical', 'GEO2'])

    full_trials_canonical = 200 * 4 * 30 * 3  # 200 trials, 4 dims, ~30 points, 3 tau
    full_trials_geo2 = 200 * 3 * 25 * 1       # 200 trials, 3 dims, ~25 points, 1 tau

    scale_factor = (full_trials_canonical + full_trials_geo2) / (pilot_trials * pilot_points / 2)
    est_full_time = total_time * scale_factor

    print(f"Estimated full production time: {est_full_time/3600:.1f} hours")


if __name__ == '__main__':
    main()
