#!/usr/bin/env python3
"""
Verify optimizer actually converges (gradient norm small at termination).

For each (d, K, ensemble):
1. Run optimization with analytical gradient
2. Record final gradient norm
3. Check if ||∇S|| < 1e-4 at termination
4. Report convergence quality statistics
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time
from scipy.optimize import minimize
from reach import models, mathematics


def check_convergence(d, K, ensemble, n_trials=20, maxiter=100, restarts=3, **model_params):
    """Check gradient norms at termination for each trial."""
    results = []

    for trial in range(n_trials):
        seed = 8000 + trial
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=seed + 500)[0]
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=seed, **model_params)
        hams_np = [H.full() for H in hams]
        bounds = [(-1.0, 1.0)] * K

        rng = np.random.RandomState(seed + 100)

        best_val = 0.0
        best_grad_norm = None
        best_success = False

        for r in range(restarts):
            x0 = rng.uniform(-1, 1, K)

            def obj_grad(x):
                S, g = mathematics.spectral_overlap_with_grad(x, psi, phi, hams, hams_np=hams_np)
                return -S, -g.astype(np.float64)

            res = minimize(obj_grad, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-6})

            val = -res.fun
            if val > best_val:
                best_val = val
                best_success = res.success
                # Compute gradient at final point
                _, grad_at_opt = mathematics.spectral_overlap_with_grad(
                    res.x, psi, phi, hams, hams_np=hams_np
                )
                best_grad_norm = np.linalg.norm(grad_at_opt)

        results.append({
            'trial': trial,
            'S_star': best_val,
            'grad_norm': best_grad_norm,
            'success': best_success,
        })

    return results


def main():
    print("=" * 70)
    print("OPTIMIZER CONVERGENCE VERIFICATION")
    print("Check that ||∇S|| < 1e-4 at termination")
    print("=" * 70)

    configs = [
        ("Canonical d=16, K=13", "canonical", 16, 13, {}),
        ("GEO2 d=16, K=13", "GEO2", 16, 13, {'nx': 2, 'ny': 2}),
        ("GEO2 d=32, K=51", "GEO2", 32, 51, {'nx': 1, 'ny': 5}),
    ]

    for label, ensemble, d, K, params in configs:
        print(f"\n--- {label} ---")
        results = check_convergence(d, K, ensemble, n_trials=15, **params)

        grad_norms = [r['grad_norm'] for r in results if r['grad_norm'] is not None]
        s_stars = [r['S_star'] for r in results]
        successes = sum(1 for r in results if r['success'])

        print(f"  S* range: [{min(s_stars):.4f}, {max(s_stars):.4f}], "
              f"mean={np.mean(s_stars):.4f}")
        print(f"  ||∇S|| range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}], "
              f"mean={np.mean(grad_norms):.2e}")
        print(f"  Converged (||∇S|| < 1e-4): {sum(1 for g in grad_norms if g < 1e-4)}/{len(grad_norms)}")
        print(f"  Scipy success: {successes}/{len(results)}")

        # Show worst cases
        worst = sorted(results, key=lambda r: -(r['grad_norm'] or 0))[:3]
        for w in worst:
            status = "OK" if (w['grad_norm'] or 0) < 1e-4 else "LARGE GRAD"
            print(f"    Trial {w['trial']}: S*={w['S_star']:.4f}, "
                  f"||∇S||={w['grad_norm']:.2e} [{status}]")


if __name__ == "__main__":
    main()
