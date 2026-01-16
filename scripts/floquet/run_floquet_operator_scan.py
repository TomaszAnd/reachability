#!/usr/bin/env python3
"""
Floquet Operator Scan Experiment

Tests the hypothesis: Floquet engineering requires fewer operators K to reach
a target state with high fidelity than static Hamiltonians.

This is the REDESIGNED experiment that addresses the fundamental design flaw
of the original approach. Instead of testing whether the moment criterion can
prove unreachability, we directly compare fidelity optimization results.

Key idea:
- For each K (number of operators), optimize fidelity for both:
  * Static: H = Σ λ_k H_k
  * Floquet: H_F = H_F^(1) + H_F^(2)
- Plot fidelity vs K to find critical operator number K_c
- Hypothesis: K_c(Floquet) < K_c(Static) due to commutator-generated 3-body terms

Usage:
    python3 scripts/run_floquet_operator_scan.py \
        --state-pair product_0-ghz \
        --K-values 4 6 8 10 12 14 16 \
        --driving-type bichromatic \
        --n-trials 5 \
        --t-max 20.0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime

from reach import floquet, models, optimization, states


def parse_state_pair(state_pair_str):
    """
    Parse state pair string like 'product_0-ghz' into (initial, target).

    Format: {initial_name}-{target_name}

    Examples:
        'product_0-ghz' -> (|0000⟩, GHZ)
        'product_+-cluster' -> (|++++⟩, Cluster)
        'neel-ghz' -> (Neel, GHZ)
    """
    parts = state_pair_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid state pair format: {state_pair_str}. Use 'initial-target'")

    initial_name, target_name = parts
    return initial_name, target_name


def get_state_by_name(name, n_qubits):
    """Get state vector by name."""
    initial_states = states.create_initial_states(n_qubits)
    target_states = states.create_target_states(n_qubits)

    # Try initial states first
    if name in initial_states:
        return initial_states[name]

    # Then target states
    if name in target_states:
        return target_states[name]

    raise ValueError(f"Unknown state name: {name}. Available: {list(initial_states.keys()) + list(target_states.keys())}")


def run_operator_scan(
    initial_name,
    target_name,
    K_values,
    driving_type='bichromatic',
    n_trials=5,
    t_max=20.0,
    n_qubits=4,
    period=1.0,
    seed=42,
    optimize_lambdas=True
):
    """
    Run operator number scan for a single state pair.

    Args:
        initial_name: Name of initial state (e.g., 'product_0')
        target_name: Name of target state (e.g., 'ghz')
        K_values: List of operator numbers to test
        driving_type: Type of driving function
        n_trials: Number of random Hamiltonian samples per K
        t_max: Maximum evolution time
        n_qubits: Number of qubits
        period: Driving period
        seed: Random seed
        optimize_lambdas: If True, optimize over coupling coefficients λ (slower but better)

    Returns:
        Dictionary with results for each K
    """
    print(f"\n{'='*70}")
    print(f"OPERATOR SCAN: {initial_name} → {target_name}")
    print(f"{'='*70}\n")

    # Get states
    psi = get_state_by_name(initial_name, n_qubits)
    phi = get_state_by_name(target_name, n_qubits)

    # Classical overlap
    classical_overlap = np.abs(phi.conj() @ psi)**2
    print(f"Classical overlap |⟨{target_name}|{initial_name}⟩|² = {classical_overlap:.4f}")
    print(f"Driving type: {driving_type}")
    print(f"Number of trials per K: {n_trials}")
    print(f"Maximum evolution time: {t_max}")
    print(f"Period: {period}")
    print(f"Optimize λ: {optimize_lambdas}")
    if optimize_lambdas:
        print("  (Slower but finds optimal Hamiltonian in operator span)")
    else:
        print("  (Faster but uses fixed random λ - may miss reachable states!)")
    print()

    d = 2**n_qubits
    nx, ny = 2, 2  # 2x2 lattice for 4 qubits

    results = {
        'initial_name': initial_name,
        'target_name': target_name,
        'classical_overlap': classical_overlap,
        'K_values': K_values,
        'driving_type': driving_type,
        'n_trials': n_trials,
        't_max': t_max,
        'period': period,
        'n_qubits': n_qubits,
        'fidelity_static': [],
        'fidelity_floquet_o1': [],
        'fidelity_floquet_o2': [],
        'std_static': [],
        'std_floquet_o1': [],
        'std_floquet_o2': [],
        'time_static': [],
        'time_floquet_o1': [],
        'time_floquet_o2': [],
    }

    for K in K_values:
        print(f"Testing K = {K} operators...")

        fid_static_trials = []
        fid_f1_trials = []
        fid_f2_trials = []
        time_static_trials = []
        time_f1_trials = []
        time_f2_trials = []

        for trial in range(n_trials):
            trial_seed = seed + trial

            # Generate random GEO2 Hamiltonians
            hams_qutip = models.random_hamiltonian_ensemble(
                dim=d, k=K, ensemble="GEO2", nx=nx, ny=ny, seed=trial_seed
            )
            hams = floquet.hamiltonians_to_numpy(hams_qutip)

            if optimize_lambdas:
                # PARAMETERIZED OPTIMIZATION: Optimize over both λ and t
                # This is SLOWER but finds the best possible Hamiltonian in the span

                # Static: max_{λ,t} |⟨φ|exp(-i(Σλ_k H_k)t)|ψ⟩|²
                fid_s, lambdas_s, t_s = optimization.optimize_fidelity_parameterized(
                    psi, phi, hams, t_max=t_max, n_trials=10, seed=trial_seed
                )
                fid_static_trials.append(fid_s)
                time_static_trials.append(t_s)

                # Floquet order 1: max_{λ,t} |⟨φ|exp(-iH_F^(1)t)|ψ⟩|²
                # Need to create Floquet Hamiltonians for different λ
                # For now, use same λ from static optimization as approximation
                driving = floquet.create_driving_functions(K, driving_type, period, seed=trial_seed)
                H_F1 = floquet.compute_floquet_hamiltonian(hams, lambdas_s, driving, period, order=1)
                fid_f1, t_f1 = optimization.optimize_fidelity(psi, phi, H_F1, t_max)
                fid_f1_trials.append(fid_f1)
                time_f1_trials.append(t_f1)

                # Floquet order 2: max_{λ,t} |⟨φ|exp(-iH_F^(2)t)|ψ⟩|²
                H_F2 = floquet.compute_floquet_hamiltonian(hams, lambdas_s, driving, period, order=2)
                fid_f2, t_f2 = optimization.optimize_fidelity(psi, phi, H_F2, t_max)
                fid_f2_trials.append(fid_f2)
                time_f2_trials.append(t_f2)

            else:
                # FIXED RANDOM λ: Only optimize over time t (faster but may miss reachable states)

                rng = np.random.RandomState(trial_seed)
                lambdas = rng.randn(K) / np.sqrt(K)
                driving = floquet.create_driving_functions(K, driving_type, period, seed=trial_seed)

                # Static Hamiltonian
                H_static = sum(lam * H_k for lam, H_k in zip(lambdas, hams))
                fid_s, t_s = optimization.optimize_fidelity(psi, phi, H_static, t_max)
                fid_static_trials.append(fid_s)
                time_static_trials.append(t_s)

                # Floquet order 1 (time-averaged only)
                H_F1 = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, period, order=1)
                fid_f1, t_f1 = optimization.optimize_fidelity(psi, phi, H_F1, t_max)
                fid_f1_trials.append(fid_f1)
                time_f1_trials.append(t_f1)

                # Floquet order 2 (time-averaged + commutators)
                H_F2 = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, period, order=2)
                fid_f2, t_f2 = optimization.optimize_fidelity(psi, phi, H_F2, t_max)
                fid_f2_trials.append(fid_f2)
                time_f2_trials.append(t_f2)

            if (trial + 1) % max(1, n_trials // 5) == 0:
                print(f"  Trial {trial+1}/{n_trials}: Static={fid_s:.3f}, F1={fid_f1:.3f}, F2={fid_f2:.3f}")

        # Compute statistics
        mean_static = np.mean(fid_static_trials)
        mean_f1 = np.mean(fid_f1_trials)
        mean_f2 = np.mean(fid_f2_trials)
        std_static = np.std(fid_static_trials)
        std_f1 = np.std(fid_f1_trials)
        std_f2 = np.std(fid_f2_trials)

        results['fidelity_static'].append(mean_static)
        results['fidelity_floquet_o1'].append(mean_f1)
        results['fidelity_floquet_o2'].append(mean_f2)
        results['std_static'].append(std_static)
        results['std_floquet_o1'].append(std_f1)
        results['std_floquet_o2'].append(std_f2)
        results['time_static'].append(np.mean(time_static_trials))
        results['time_floquet_o1'].append(np.mean(time_f1_trials))
        results['time_floquet_o2'].append(np.mean(time_f2_trials))

        print(f"  K={K}: Static={mean_static:.4f}±{std_static:.4f}, "
              f"F1={mean_f1:.4f}±{std_f1:.4f}, F2={mean_f2:.4f}±{std_f2:.4f}")

        # Check for improvement
        if mean_f2 > mean_static + 0.05:
            print(f"  ✓ Floquet O2 shows improvement! (+{100*(mean_f2-mean_static):.1f}%)")
        elif mean_f2 < mean_static - 0.05:
            print(f"  ⚠️  Floquet O2 worse than static (-{100*(mean_static-mean_f2):.1f}%)")

        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"State pair: {initial_name} → {target_name}")
    print(f"\nFidelity Results (mean ± std):\n")
    print(f"{'K':>4s} | {'Static':>12s} | {'Floquet O1':>12s} | {'Floquet O2':>12s} | {'Improvement':>12s}")
    print("-" * 70)

    for i, K in enumerate(K_values):
        s = results['fidelity_static'][i]
        s_std = results['std_static'][i]
        f1 = results['fidelity_floquet_o1'][i]
        f1_std = results['std_floquet_o1'][i]
        f2 = results['fidelity_floquet_o2'][i]
        f2_std = results['std_floquet_o2'][i]
        improvement = (f2 - s) / max(s, 1e-10) * 100

        print(f"{K:4d} | {s:.3f}±{s_std:.3f} | {f1:.3f}±{f1_std:.3f} | {f2:.3f}±{f2_std:.3f} | {improvement:+8.1f}%")

    # Find critical K (fidelity > 0.9)
    critical_static = None
    critical_f2 = None

    for i, K in enumerate(K_values):
        if results['fidelity_static'][i] > 0.9 and critical_static is None:
            critical_static = K
        if results['fidelity_floquet_o2'][i] > 0.9 and critical_f2 is None:
            critical_f2 = K

    print(f"\nCritical K (fidelity > 0.9):")
    print(f"  Static: K_c = {critical_static if critical_static else '>'+str(K_values[-1])}")
    print(f"  Floquet O2: K_c = {critical_f2 if critical_f2 else '>'+str(K_values[-1])}")

    if critical_static and critical_f2 and critical_f2 < critical_static:
        reduction = 100 * (critical_static - critical_f2) / critical_static
        print(f"  ✓ Floquet reduces operator requirement by {reduction:.0f}%!")
    elif critical_static and critical_f2 and critical_f2 > critical_static:
        print(f"  ⚠️  Floquet requires MORE operators (unexpected)")
    else:
        print(f"  Need to test higher K values to find critical points")

    return results


def save_results(results, output_path):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            results_serializable[key] = [v.tolist() for v in value]
        else:
            results_serializable[key] = value

    # Add metadata
    results_serializable['timestamp'] = datetime.now().isoformat()
    results_serializable['script'] = 'run_floquet_operator_scan.py'

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Floquet operator number scan experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test for |0000⟩ → GHZ
  python3 scripts/run_floquet_operator_scan.py \\
      --state-pair product_0-ghz \\
      --K-values 4 6 8 10 12 14 16 \\
      --driving-type bichromatic \\
      --n-trials 5

  # Full scan with more trials
  python3 scripts/run_floquet_operator_scan.py \\
      --state-pair neel-ghz \\
      --K-values 4 6 8 10 12 14 16 20 \\
      --driving-type bichromatic \\
      --n-trials 20 \\
      --t-max 30.0
        """
    )

    parser.add_argument(
        '--state-pair',
        type=str,
        required=True,
        help='State pair in format "initial-target" (e.g., "product_0-ghz")'
    )
    parser.add_argument(
        '--K-values',
        type=int,
        nargs='+',
        required=True,
        help='List of operator numbers to test (e.g., 4 6 8 10 12 14 16)'
    )
    parser.add_argument(
        '--driving-type',
        type=str,
        default='bichromatic',
        choices=['constant', 'sinusoidal', 'offset_sinusoidal', 'bichromatic'],
        help='Type of driving function (default: bichromatic)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=5,
        help='Number of random Hamiltonian samples per K (default: 5)'
    )
    parser.add_argument(
        '--t-max',
        type=float,
        default=20.0,
        help='Maximum evolution time (default: 20.0)'
    )
    parser.add_argument(
        '--n-qubits',
        type=int,
        default=4,
        help='Number of qubits (default: 4)'
    )
    parser.add_argument(
        '--period',
        type=float,
        default=1.0,
        help='Driving period T (default: 1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file (default: auto-generated based on state pair)'
    )
    parser.add_argument(
        '--no-optimize-lambdas',
        action='store_true',
        help='Use fixed random λ instead of optimizing (faster but may miss reachable states)'
    )

    args = parser.parse_args()

    # Parse state pair
    initial_name, target_name = parse_state_pair(args.state_pair)

    # Run scan
    results = run_operator_scan(
        initial_name=initial_name,
        target_name=target_name,
        K_values=args.K_values,
        driving_type=args.driving_type,
        n_trials=args.n_trials,
        t_max=args.t_max,
        n_qubits=args.n_qubits,
        period=args.period,
        seed=args.seed,
        optimize_lambdas=not args.no_optimize_lambdas
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/floquet_operator_scan_{args.state_pair}_{args.driving_type}_{timestamp}.json"

    # Create results directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    save_results(results, output_path)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
