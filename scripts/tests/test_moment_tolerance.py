#!/usr/bin/env python3
"""Test Moment criterion sensitivity to tolerance and gamma values.

Investigates whether different tolerance/gamma choices affect the
non-monotonicity behavior observed in v81.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.models import CanonicalQuditModel


def test_tolerance_and_gamma(d=16, K_values=None, n_hamiltonians=50,
                             n_targets=10, seed=42):
    """Test Moment detection rate at different tolerance/gamma settings."""
    if K_values is None:
        K_values = [4, 6, 8, 10, 12, 14, 16]

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    n_trials = n_hamiltonians * n_targets

    configs = [
        ("tol=1e-10, g=1e3", 1e-10, [1000.0, -1000.0]),
        ("tol=1e-10, g=1e4", 1e-10, [10000.0, -10000.0]),
        ("tol=1e-10, g=1e6", 1e-10, [1e6, -1e6]),
        ("tol=1e-8,  g=1e3", 1e-8, [1000.0, -1000.0]),
        ("tol=1e-12, g=1e3", 1e-12, [1000.0, -1000.0]),
    ]

    print(f"Moment Tolerance/Gamma Test: d={d}, {n_trials} trials per K")
    header = f"{'K':>4}"
    for label, _, _ in configs:
        header += f"  {label:>20}"
    print(header)
    print("-" * (4 + 22 * len(configs)))

    for K in K_values:
        if K > model.K:
            continue

        results = {label: 0 for label, _, _ in configs}
        rng = np.random.default_rng(seed + K)

        for h in range(n_hamiltonians):
            sub = model.sample_submodel(K, seed=int(rng.integers(0, 2**31)))
            hams = np.stack(sub.basis)

            for t in range(n_targets):
                psi = sub.random_state(rng=rng)

                # Compute Q and L once
                Hpsi = np.einsum('kij,j->ki', hams, psi)
                Hphi = np.einsum('kij,j->ki', hams, phi)
                L = np.real(np.einsum('d,kd->k', psi.conj(), Hpsi)
                            - np.einsum('d,kd->k', phi.conj(), Hphi))
                Q = 2 * (np.real(Hpsi.conj() @ Hpsi.T)
                         - np.real(Hphi.conj() @ Hphi.T))
                L_outer = np.outer(L, L)

                for label, tol, gammas in configs:
                    found = False
                    for gamma in gammas:
                        M = Q + gamma * L_outer
                        eigvals = np.linalg.eigvalsh(M)
                        if np.all(eigvals > tol) or np.all(eigvals < -tol):
                            found = True
                            break
                    if found:
                        results[label] += 1

        row = f"{K:4d}"
        for label, _, _ in configs:
            P = results[label] / n_trials
            row += f"  {P:20.3f}"
        print(row)


def test_monotonicity_per_config(d=16, K_pairs=None, n_hamiltonians=50,
                                 n_targets=10, seed=42):
    """Paired monotonicity test at different settings."""
    if K_pairs is None:
        K_pairs = [(6, 8), (8, 10), (10, 12)]

    model = CanonicalQuditModel(dim=d, seed=seed)
    phi = model.init_state()
    n_trials = n_hamiltonians * n_targets

    configs = [
        ("default (g=1e3)", 1e-10, [1000.0, -1000.0]),
        ("g=1e6", 1e-10, [1e6, -1e6]),
        ("tol=1e-8", 1e-8, [1000.0, -1000.0]),
    ]

    print(f"\nPaired Monotonicity by Config: d={d}, {n_trials} trials")

    for label, tol, gammas in configs:
        violations = {f"{Ks}->{Kl}": 0 for Ks, Kl in K_pairs}
        rng = np.random.default_rng(seed)

        for h in range(n_hamiltonians):
            K_max = max(k for _, k in K_pairs)
            sub = model.sample_submodel(K_max, seed=int(rng.integers(0, 2**31)))
            hams = np.stack(sub.basis)

            for t in range(n_targets):
                psi = sub.random_state(rng=rng)
                Hpsi = np.einsum('kij,j->ki', hams, psi)
                Hphi = np.einsum('kij,j->ki', hams, phi)

                verdicts = {}
                for K in sorted(set(k for pair in K_pairs for k in pair)):
                    L = np.real(np.einsum('d,kd->k', psi.conj(), Hpsi[:K])
                                - np.einsum('d,kd->k', phi.conj(), Hphi[:K]))
                    Q = 2 * (np.real(Hpsi[:K].conj() @ Hpsi[:K].T)
                             - np.real(Hphi[:K].conj() @ Hphi[:K].T))
                    L_outer = np.outer(L, L)

                    found = False
                    for gamma in gammas:
                        M = Q + gamma * L_outer
                        eigvals = np.linalg.eigvalsh(M)
                        if np.all(eigvals > tol) or np.all(eigvals < -tol):
                            found = True
                            break
                    verdicts[K] = found

                for Ks, Kl in K_pairs:
                    if verdicts[Ks] and not verdicts[Kl]:
                        violations[f"{Ks}->{Kl}"] += 1

        print(f"\n  {label}:")
        for key, v in violations.items():
            print(f"    {key}: {v} violations ({v/n_trials*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        test_tolerance_and_gamma(n_hamiltonians=20, n_targets=5)
        test_monotonicity_per_config(n_hamiltonians=20, n_targets=5)
    else:
        test_tolerance_and_gamma()
        test_monotonicity_per_config()
