#!/usr/bin/env python3
"""
Extract and analyze optimal λ* vectors from Floquet moment criterion.

This script finds the coupling vectors λ* that maximize the discriminative
power of the Floquet criterion, then analyzes their structure to understand
which commutator combinations are most important.

Key insight: λ* encodes the essential physics for QEC code preparation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from datetime import datetime
from scipy.optimize import minimize, differential_evolution
from reach import floquet, models, moment_criteria


def compute_min_eigenvalue_with_x_search(L_F, Q_F, x_range=(-10, 10), n_points=1000):
    """
    Find the minimum eigenvalue of Q_F + x L_F L_F^T over all x.

    If max_x min_eig(Q_F + x L_F L_F^T) > 0, criterion succeeds.

    Returns:
        best_min_eig: Maximum achievable minimum eigenvalue
        best_x: Value of x achieving it
    """
    L_F_outer = np.outer(L_F, L_F)
    x_values = np.linspace(x_range[0], x_range[1], n_points)

    best_min_eig = -np.inf
    best_x = None

    for x in x_values:
        M = Q_F + x * L_F_outer
        min_eig = np.min(np.linalg.eigvalsh(M))

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_x = x

    return best_min_eig, best_x


def optimize_lambda_for_criterion(
    psi, phi, hams,
    driving_functions, period,
    order=2,
    method='differential_evolution',
    n_restarts=10,
    seed=42
):
    """
    Find λ* that maximizes the minimum eigenvalue of Q_F + x L_F L_F^T.

    This is the λ that gives the strongest unreachability proof.

    Args:
        psi: Initial state
        phi: Target state
        hams: List of Hamiltonians
        driving_functions: Time-periodic driving functions
        period: Driving period
        order: Magnus expansion order
        method: 'differential_evolution' (global) or 'BFGS' (local)
        n_restarts: Number of random initializations for local methods
        seed: Random seed

    Returns:
        lambda_star: Optimal coupling vector
        max_min_eigenvalue: Best achievable minimum eigenvalue
        optimization_history: Trajectory of optimizer
    """
    np.random.seed(seed)
    K = len(hams)

    def objective(lam):
        """Negative of min eigenvalue (we want to maximize min eigenvalue)."""
        # Compute Floquet moments for this λ
        from reach.moment_criteria import floquet_moment_criterion

        # We need the L_F and Q_F matrices, not just success/failure
        # Let me compute them directly

        # Compute derivatives ∂H_F/∂λ_k
        dH_F_dlambda = []
        for k in range(K):
            lambda_bar_k = floquet.compute_time_average(driving_functions[k], period)
            derivative = lambda_bar_k * hams[k]

            if order >= 2:
                for j in range(K):
                    if j != k:
                        F_jk = floquet.compute_fourier_overlap(
                            driving_functions[j], driving_functions[k], period
                        )
                        commutator = hams[j] @ hams[k] - hams[k] @ hams[j]
                        derivative += lam[j] * F_jk * commutator / (2 * 1j)

            derivative = (derivative + derivative.conj().T) / 2
            dH_F_dlambda.append(derivative)

        # Compute L_F
        L_F = np.zeros(K)
        for k in range(K):
            exp_val_phi = np.real(phi.conj() @ dH_F_dlambda[k] @ phi)
            exp_val_psi = np.real(psi.conj() @ dH_F_dlambda[k] @ psi)
            L_F[k] = exp_val_phi - exp_val_psi

        # Compute Q_F
        Q_F = np.zeros((K, K))
        for k in range(K):
            for m in range(K):
                anticomm = (dH_F_dlambda[k] @ dH_F_dlambda[m] +
                           dH_F_dlambda[m] @ dH_F_dlambda[k]) / 2
                exp_val_phi = np.real(phi.conj() @ anticomm @ phi)
                exp_val_psi = np.real(psi.conj() @ anticomm @ psi)
                Q_F[k, m] = exp_val_phi - exp_val_psi

        # Find best x for this λ
        min_eig, _ = compute_min_eigenvalue_with_x_search(L_F, Q_F)

        # Return negative (scipy minimizes)
        return -min_eig

    # Global optimization
    if method == 'differential_evolution':
        bounds = [(-5, 5) for _ in range(K)]
        result = differential_evolution(
            objective, bounds, seed=seed, maxiter=1000, atol=1e-6
        )
        lambda_star = result.x
        max_min_eig = -result.fun
        history = None

    # Local optimization with multiple restarts
    elif method == 'BFGS':
        best_lambda = None
        best_value = -np.inf
        history = []

        for restart in range(n_restarts):
            lambda_init = np.random.randn(K)

            result = minimize(
                objective, lambda_init, method='BFGS',
                options={'maxiter': 500, 'disp': False}
            )

            if -result.fun > best_value:
                best_value = -result.fun
                best_lambda = result.x

            history.append({
                'restart': restart,
                'init': lambda_init,
                'final': result.x,
                'value': -result.fun,
                'success': result.success
            })

        lambda_star = best_lambda
        max_min_eig = best_value

    else:
        raise ValueError(f"Unknown method: {method}")

    return lambda_star, max_min_eig, history


def analyze_lambda_star(lambda_star, hams, driving_functions, period, order=2):
    """
    Analyze the structure of optimal λ*.

    Returns:
        analysis: Dictionary containing:
            - operator_weights: Magnitude of each λ_k*
            - commutator_contributions: Weight of each [H_j, H_k] term
            - dominant_commutators: Top 10 commutators by contribution
            - locality_distribution: Distribution of operator localities
    """
    K = len(lambda_star)

    # Operator weights
    operator_weights = {
        k: {
            'lambda_k': lambda_star[k],
            'magnitude': abs(lambda_star[k]),
            'hamiltonian_norm': np.linalg.norm(hams[k])
        }
        for k in range(K)
    }

    # Commutator contributions (only for order >= 2)
    commutator_contributions = []
    if order >= 2:
        for j in range(K):
            for k in range(j+1, K):
                # Compute F_jk
                F_jk = floquet.compute_fourier_overlap(
                    driving_functions[j], driving_functions[k], period
                )

                # Commutator
                comm = hams[j] @ hams[k] - hams[k] @ hams[j]
                comm_norm = np.linalg.norm(comm)

                # Weight in effective Hamiltonian
                weight = abs(lambda_star[j] * lambda_star[k] * F_jk)

                # Effective contribution
                effective_contribution = weight * comm_norm

                commutator_contributions.append({
                    'j': j,
                    'k': k,
                    'lambda_j': lambda_star[j],
                    'lambda_k': lambda_star[k],
                    'F_jk': F_jk,
                    'weight': weight,
                    'commutator_norm': comm_norm,
                    'effective_contribution': effective_contribution
                })

        # Sort by effective contribution
        commutator_contributions.sort(
            key=lambda x: x['effective_contribution'], reverse=True
        )

    # Top 10 commutators
    dominant_commutators = commutator_contributions[:10] if commutator_contributions else []

    analysis = {
        'operator_weights': operator_weights,
        'commutator_contributions': commutator_contributions,
        'dominant_commutators': dominant_commutators,
        'lambda_norm': np.linalg.norm(lambda_star),
        'lambda_distribution': {
            'mean': np.mean(lambda_star),
            'std': np.std(lambda_star),
            'max': np.max(lambda_star),
            'min': np.min(lambda_star)
        }
    }

    return analysis


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract optimal λ* from Floquet moment criterion'
    )
    parser.add_argument('--target', type=str, default='5qubit',
                        choices=['5qubit', 'haar'],
                        help='Target state type')
    parser.add_argument('--K', type=int, default=4,
                        help='Number of operators')
    parser.add_argument('--order', type=int, default=2,
                        help='Magnus expansion order')
    parser.add_argument('--method', type=str, default='differential_evolution',
                        choices=['differential_evolution', 'BFGS'],
                        help='Optimization method')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    print("="*70)
    print("OPTIMAL λ* EXTRACTION FOR FLOQUET MOMENT CRITERION")
    print("="*70)
    print()
    print(f"Target: {args.target}")
    print(f"K: {args.K}")
    print(f"Magnus order: {args.order}")
    print(f"Optimization method: {args.method}")
    print()

    # Setup
    if args.target == '5qubit':
        from scripts.run_5qubit_code_experiment import create_5qubit_code_logical_zero

        n = 5
        d = 32
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0  # |00000⟩
        phi = create_5qubit_code_logical_zero()

        # GEO2LOCAL 1D chain
        hams_qutip = models.random_hamiltonian_ensemble(
            dim=d, k=51, ensemble="GEO2", nx=5, ny=1, seed=args.seed
        )
        hams = floquet.hamiltonians_to_numpy(hams_qutip)

    elif args.target == 'haar':
        n = 4
        d = 16
        psi = np.zeros(d, dtype=complex)
        psi[0] = 1.0  # |0000⟩

        # Random Haar state
        np.random.seed(args.seed)
        phi = models.random_state_haar(d)

        # GEO2LOCAL 2x2 lattice
        hams_qutip = models.random_hamiltonian_ensemble(
            dim=d, k=48, ensemble="GEO2", nx=2, ny=2, seed=args.seed
        )
        hams = floquet.hamiltonians_to_numpy(hams_qutip)

    # Driving functions (bichromatic)
    omega1 = 1.0
    omega2 = np.sqrt(2)
    period = 2 * np.pi

    driving_functions = []
    for k in range(args.K):
        omega = omega1 if k % 2 == 0 else omega2
        driving_functions.append(lambda t, om=omega: np.sin(om * t))

    # Optimize λ*
    print("Optimizing λ* to maximize criterion discriminative power...")
    print(f"(This may take several minutes with {args.method})")
    print()

    lambda_star, max_min_eig, history = optimize_lambda_for_criterion(
        psi, phi, hams[:args.K],
        driving_functions, period,
        order=args.order,
        method=args.method,
        n_restarts=10,
        seed=args.seed
    )

    print("="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print()
    print(f"λ* = {lambda_star}")
    print(f"||λ*|| = {np.linalg.norm(lambda_star):.4f}")
    print(f"Max min eigenvalue = {max_min_eig:.6f}")
    print(f"Criterion succeeds: {max_min_eig > 1e-10}")
    print()

    # Analyze λ*
    print("="*70)
    print("ANALYZING λ* STRUCTURE")
    print("="*70)
    print()

    analysis = analyze_lambda_star(
        lambda_star, hams[:args.K], driving_functions, period, order=args.order
    )

    print("Top 5 operator weights:")
    sorted_weights = sorted(
        analysis['operator_weights'].items(),
        key=lambda x: x[1]['magnitude'],
        reverse=True
    )
    for k, data in sorted_weights[:5]:
        print(f"  k={k}: λ*_k={data['lambda_k']:+.4f}, "
              f"|λ*_k|={data['magnitude']:.4f}, "
              f"||H_k||={data['hamiltonian_norm']:.4f}")
    print()

    if args.order >= 2 and analysis['dominant_commutators']:
        print("Top 5 commutator contributions:")
        for i, comm in enumerate(analysis['dominant_commutators'][:5]):
            print(f"  {i+1}. [H_{comm['j']}, H_{comm['k']}]: "
                  f"weight={comm['weight']:.4f}, "
                  f"||[H_j,H_k]||={comm['commutator_norm']:.4f}, "
                  f"contribution={comm['effective_contribution']:.4f}")
        print()

    print("λ* distribution statistics:")
    for key, value in analysis['lambda_distribution'].items():
        print(f"  {key}: {value:.4f}")
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/lambda_star_{args.target}_K{args.K}_o{args.order}_{timestamp}.pkl'

    results = {
        'timestamp': datetime.now().isoformat(),
        'target': args.target,
        'K': args.K,
        'order': args.order,
        'method': args.method,
        'seed': args.seed,
        'lambda_star': lambda_star,
        'max_min_eigenvalue': max_min_eig,
        'analysis': analysis,
        'optimization_history': history
    }

    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print("="*70)
    print("RESULTS SAVED")
    print("="*70)
    print()
    print(f"File: {filename}")
    print()


if __name__ == '__main__':
    main()
