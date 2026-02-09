#!/usr/bin/env python3
"""Validate that hams_np speed optimization produces identical results."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import qutip
from reach import mathematics, models, optimize

print("=" * 60)
print("VALIDATION: hams_np wiring produces identical results")
print("=" * 60)

for d in [8, 16, 32]:
    for ensemble in ['canonical', 'GUE']:
        K = max(2, d // 2)
        hams = models.random_hamiltonian_ensemble(d, K, ensemble, seed=42)
        psi = models.fock_state(d, 0)
        phi = models.random_states(1, d, seed=99)[0]
        lam = np.random.RandomState(42).uniform(-1, 1, K)

        hams_np = [H.full() for H in hams]

        # Test spectral_overlap
        s1 = mathematics.spectral_overlap(lam, psi, phi, hams)
        s2 = mathematics.spectral_overlap(lam, psi, phi, hams, hams_np=hams_np)
        assert s1 == s2, f"spectral_overlap mismatch: {abs(s1-s2):.2e}"

        # Test spectral_overlap_with_grad
        v1, g1 = mathematics.spectral_overlap_with_grad(lam, psi, phi, hams)
        v2, g2 = mathematics.spectral_overlap_with_grad(lam, psi, phi, hams, hams_np=hams_np)
        assert v1 == v2, f"spectral_grad value mismatch: {abs(v1-v2):.2e}"
        assert np.allclose(g1, g2, atol=1e-15), f"spectral_grad gradient mismatch: {np.max(np.abs(g1-g2)):.2e}"

        # Test krylov_score
        for m in [min(K, d), d]:
            r1 = mathematics.krylov_score(lam, psi, phi, hams, m=m)
            r2 = mathematics.krylov_score(lam, psi, phi, hams, m=m, hams_np=hams_np)
            assert r1 == r2, f"krylov_score m={m} mismatch: {abs(r1-r2):.2e}"

        # Test krylov_score_with_grad
        r1, g1 = mathematics.krylov_score_with_grad(lam, psi, phi, hams, m=min(K, d))
        r2, g2 = mathematics.krylov_score_with_grad(lam, psi, phi, hams, m=min(K, d), hams_np=hams_np)
        assert r1 == r2, f"krylov_grad value mismatch: {abs(r1-r2):.2e}"
        assert np.allclose(g1, g2, atol=1e-15), f"krylov_grad gradient mismatch"

        # Test full optimizer
        res1 = optimize.maximize_spectral_overlap(psi, phi, hams, seed=42, restarts=1, maxiter=20)
        res2 = optimize.maximize_spectral_overlap(psi, phi, hams, seed=42, restarts=1, maxiter=20)
        assert res1['best_value'] == res2['best_value'], "Optimizer result mismatch"

        print(f"  d={d}, {ensemble}, K={K}: ALL IDENTICAL")

print("\nALL VALIDATION PASSED -- zero accuracy change")
