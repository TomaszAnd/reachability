#!/usr/bin/env python3
"""
Test connectivity hypothesis for criterion ordering flip.

HYPOTHESIS: The GEO2/Canonical ordering difference comes from connectivity constraints,
not integrability per se.

- GEO2: Lattice-constrained (nearest-neighbor on 2D grid)
- Canonical: All-to-all (any pair j,k can couple)

TESTS:
1. GEO2-style operators with all-to-all connectivity
2. Canonical-style operators restricted to chain topology
3. Interpolated connectivity (chain → complete graph)

PREDICTION: If connectivity is the key factor:
- Chain connectivity → Krylov wins (GEO2-like ordering)
- Complete graph → Spectral wins (Canonical-like ordering)

Usage:
    python scripts/integrability/connectivity_test.py

Author: Claude Code (research exploration)
Date: 2026-01-13
"""

import numpy as np
import qutip
from pathlib import Path
import pickle
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reach import optimize, mathematics, settings


# =============================================================================
# CONNECTIVITY-CONTROLLED HAMILTONIAN GENERATORS
# =============================================================================

def build_pauli_operator(n_qubits: int, sites: List[int], paulis: List[qutip.Qobj]) -> qutip.Qobj:
    """
    Build tensor product of Pauli operators at specified sites.

    Args:
        n_qubits: Total number of qubits
        sites: List of site indices where Paulis act
        paulis: List of Pauli operators (same length as sites)

    Returns:
        Tensor product operator with flattened dims
    """
    d = 2 ** n_qubits
    I = qutip.qeye(2)
    ops = [I] * n_qubits
    for site, pauli in zip(sites, paulis):
        ops[site] = pauli
    term = qutip.tensor(ops)
    term.dims = [[d], [d]]
    return term


def generate_edges(n_qubits: int, connectivity: str, connectivity_param: float = 0.0) -> List[Tuple[int, int]]:
    """
    Generate edge list based on connectivity type.

    Args:
        n_qubits: Number of qubits
        connectivity: "chain", "complete", "2d_lattice", or "interpolated"
        connectivity_param: For "interpolated", probability of adding non-nearest edges

    Returns:
        List of (i, j) edges with i < j
    """
    edges = []

    if connectivity == "chain":
        # 1D chain: only nearest neighbors
        for i in range(n_qubits - 1):
            edges.append((i, i + 1))

    elif connectivity == "complete":
        # Complete graph: all pairs
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                edges.append((i, j))

    elif connectivity == "2d_lattice":
        # 2D square lattice (approximate for general n)
        # Try to make as square as possible
        ny = int(np.sqrt(n_qubits))
        nx = n_qubits // ny
        while nx * ny < n_qubits:
            ny -= 1
            nx = n_qubits // ny

        for y in range(ny):
            for x in range(nx):
                site = y * nx + x
                if site >= n_qubits:
                    continue
                # Right neighbor
                if x + 1 < nx and site + 1 < n_qubits:
                    edges.append((site, site + 1))
                # Down neighbor
                if y + 1 < ny and site + nx < n_qubits:
                    edges.append((site, site + nx))

    elif connectivity == "interpolated":
        # Start with chain
        for i in range(n_qubits - 1):
            edges.append((i, i + 1))
        # Add non-nearest edges with given probability
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        for i in range(n_qubits):
            for j in range(i + 2, n_qubits):  # Skip nearest neighbors
                if rng.random() < connectivity_param:
                    edges.append((i, j))
    else:
        raise ValueError(f"Unknown connectivity: {connectivity}")

    return edges


def generate_pauli_basis(n_qubits: int, connectivity: str,
                         connectivity_param: float = 0.0) -> List[qutip.Qobj]:
    """
    Generate Pauli basis operators with specified connectivity.

    Structure similar to GEO2: 1-local {X,Y,Z} per site + 2-local per edge

    Args:
        n_qubits: Number of qubits
        connectivity: Edge connectivity type
        connectivity_param: Parameter for interpolated connectivity

    Returns:
        List of Hermitian Pauli operators
    """
    basis = []
    paulis = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]

    # 1-local terms: X_i, Y_i, Z_i for each site
    for i in range(n_qubits):
        for pauli in paulis:
            op = build_pauli_operator(n_qubits, [i], [pauli])
            basis.append(op)

    # 2-local terms: P_i ⊗ P_j for each edge
    edges = generate_edges(n_qubits, connectivity, connectivity_param)
    for i, j in edges:
        for pauli_i in paulis:
            for pauli_j in paulis:
                op = build_pauli_operator(n_qubits, [i, j], [pauli_i, pauli_j])
                basis.append(op)

    return basis


def sample_from_basis(basis: List[qutip.Qobj], k: int,
                      rng: np.random.RandomState) -> List[qutip.Qobj]:
    """Sample k operators from basis without replacement."""
    if k > len(basis):
        raise ValueError(f"Cannot sample {k} from basis of size {len(basis)}")
    indices = rng.choice(len(basis), size=k, replace=False)
    return [basis[i] for i in indices]


# =============================================================================
# LEVEL SPACING STATISTICS
# =============================================================================

def compute_level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    """Compute mean r-ratio for level spacing statistics."""
    E = np.sort(np.real(eigenvalues))
    gaps = np.diff(E)
    mask = gaps > 1e-12
    gaps = gaps[mask]
    if len(gaps) < 2:
        return np.nan
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return np.mean(r)


def classify_spectrum(r_mean: float) -> str:
    """Classify spectrum type from r-ratio."""
    if np.isnan(r_mean):
        return "undefined"
    elif r_mean < 0.45:
        return "Poisson"
    elif r_mean < 0.56:
        return "GOE"
    else:
        return "GUE"


# =============================================================================
# CRITERION SWEEP
# =============================================================================

def fermi_dirac(rho, rho_c, delta):
    """Fermi-Dirac fit function."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))


def fit_fermi_dirac(rho: np.ndarray, P: np.ndarray) -> Optional[Dict]:
    """Fit Fermi-Dirac to P(rho) data."""
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            mask = np.ones(len(P), dtype=bool)
        popt, _ = curve_fit(
            fermi_dirac, rho[mask], P[mask],
            p0=[np.median(rho[mask]), 0.02],
            bounds=([0, 0.001], [1.0, 0.5]),
            maxfev=10000
        )
        y_pred = fermi_dirac(rho[mask], *popt)
        ss_res = np.sum((P[mask] - y_pred)**2)
        ss_tot = np.sum((P[mask] - np.mean(P[mask]))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"rho_c": popt[0], "delta": popt[1], "R2": r2}
    except Exception:
        return None


def run_connectivity_sweep(
    connectivity: str,
    n_qubits: int,
    k_values: np.ndarray,
    trials: int,
    tau: float,
    rng_seed: int,
    connectivity_param: float = 0.0
) -> Dict:
    """
    Run criterion sweep for given connectivity type.

    Args:
        connectivity: "chain", "complete", "2d_lattice", or "interpolated"
        n_qubits: Number of qubits
        k_values: Array of K values to test
        trials: Number of trials per K
        tau: Threshold for criteria
        rng_seed: Random seed
        connectivity_param: For interpolated connectivity

    Returns:
        Dictionary with results
    """
    d = 2 ** n_qubits
    rng = np.random.RandomState(rng_seed)

    # Generate basis once
    basis = generate_pauli_basis(n_qubits, connectivity, connectivity_param)
    L = len(basis)
    edges = generate_edges(n_qubits, connectivity, connectivity_param)

    results = {
        "connectivity": connectivity,
        "connectivity_param": connectivity_param,
        "n_qubits": n_qubits,
        "d": d,
        "L": L,
        "n_edges": len(edges),
        "tau": tau,
        "k_values": k_values.tolist(),
        "spectral": {"P": [], "sem": []},
        "krylov": {"P": [], "sem": []},
        "r_ratios": [],
    }

    print(f"\n{'='*60}")
    print(f"Connectivity: {connectivity} (param={connectivity_param})")
    print(f"n_qubits={n_qubits}, d={d}, L={L}, edges={len(edges)}")
    print(f"K values: {k_values}")
    print(f"{'='*60}")

    for k in k_values:
        if k > L:
            print(f"  K={k} > L={L}, skipping")
            results["spectral"]["P"].append(np.nan)
            results["spectral"]["sem"].append(np.nan)
            results["krylov"]["P"].append(np.nan)
            results["krylov"]["sem"].append(np.nan)
            results["r_ratios"].append(np.nan)
            continue

        print(f"\n  K={k} (ρ={k/d**2:.4f}, K/L={k/L:.2f})...")

        spectral_unreachable = 0
        krylov_unreachable = 0
        r_ratio_samples = []

        for trial in range(trials):
            trial_rng = np.random.RandomState(rng.randint(0, 2**31))

            # Sample K operators from basis
            hams = sample_from_basis(basis, k, trial_rng)

            # Compute level spacing for random H(λ)
            if trial < 10:
                lambdas = trial_rng.randn(k)
                lambdas = lambdas / np.linalg.norm(lambdas)
                H_combined = sum(lam * H for lam, H in zip(lambdas, hams))
                eigenvalues = np.linalg.eigvalsh(H_combined.full())
                r = compute_level_spacing_ratio(eigenvalues)
                if not np.isnan(r):
                    r_ratio_samples.append(r)

            # Generate random states
            psi = qutip.rand_ket(d)
            phi = qutip.rand_ket(d)

            # Spectral criterion
            try:
                result_s = optimize.maximize_spectral_overlap(
                    psi, phi, hams,
                    restarts=settings.GEO2_RESTARTS,
                    maxiter=settings.GEO2_MAXITER,
                    seed=trial_rng.randint(0, 2**31)
                )
                if result_s["best_value"] < tau:
                    spectral_unreachable += 1
            except Exception:
                pass

            # Krylov criterion
            try:
                m = min(k, d)
                result_r = optimize.maximize_krylov_score(
                    psi, phi, hams, m=m,
                    restarts=settings.GEO2_RESTARTS,
                    maxiter=settings.GEO2_MAXITER,
                    seed=trial_rng.randint(0, 2**31)
                )
                if result_r["best_value"] < tau:
                    krylov_unreachable += 1
            except Exception:
                pass

        P_spectral = spectral_unreachable / trials
        P_krylov = krylov_unreachable / trials

        results["spectral"]["P"].append(P_spectral)
        results["spectral"]["sem"].append(np.sqrt(P_spectral * (1 - P_spectral) / trials))
        results["krylov"]["P"].append(P_krylov)
        results["krylov"]["sem"].append(np.sqrt(P_krylov * (1 - P_krylov) / trials))

        if r_ratio_samples:
            results["r_ratios"].append(np.mean(r_ratio_samples))
        else:
            results["r_ratios"].append(np.nan)

        r_str = f"{np.mean(r_ratio_samples):.3f}" if r_ratio_samples else "N/A"
        print(f"    P(unreach): Spectral={P_spectral:.3f}, Krylov={P_krylov:.3f}, <r>={r_str}")

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_connectivity_experiment(
    n_qubits: int = 4,
    k_max: int = 30,
    trials: int = 50,
    tau: float = 0.99,
    seed: int = 42
) -> Dict:
    """
    Run full connectivity comparison experiment.
    """
    d = 2 ** n_qubits

    print("=" * 70)
    print("CONNECTIVITY HYPOTHESIS TEST")
    print("=" * 70)
    print(f"n_qubits = {n_qubits}, d = {d}")
    print(f"trials = {trials}, tau = {tau}")
    print("=" * 70)

    all_results = {
        "metadata": {
            "n_qubits": n_qubits,
            "d": d,
            "k_max": k_max,
            "trials": trials,
            "tau": tau,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        },
        "connectivity_tests": {}
    }

    # Test different connectivities
    connectivity_configs = [
        ("chain", 0.0),
        ("2d_lattice", 0.0),
        ("interpolated_25", 0.25),
        ("interpolated_50", 0.50),
        ("interpolated_75", 0.75),
        ("complete", 0.0),
    ]

    for config_name, param in connectivity_configs:
        if config_name.startswith("interpolated"):
            connectivity = "interpolated"
        else:
            connectivity = config_name

        # Determine K range based on basis size
        test_basis = generate_pauli_basis(n_qubits, connectivity, param)
        L = len(test_basis)
        k_values = np.arange(2, min(k_max + 1, L + 1, d + 1), 2)

        results = run_connectivity_sweep(
            connectivity=connectivity,
            n_qubits=n_qubits,
            k_values=k_values,
            trials=trials,
            tau=tau,
            rng_seed=seed,
            connectivity_param=param
        )
        all_results["connectivity_tests"][config_name] = results

    return all_results


def plot_connectivity_results(results: Dict, output_dir: Path):
    """Generate comparison plots for connectivity experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    d = results["metadata"]["d"]

    for idx, (config_name, data) in enumerate(results["connectivity_tests"].items()):
        if idx >= 6:
            break

        ax = axes[idx]
        k_values = np.array(data["k_values"])
        rho = k_values / d**2

        P_s = np.array(data["spectral"]["P"])
        P_k = np.array(data["krylov"]["P"])
        sem_s = np.array(data["spectral"]["sem"])
        sem_k = np.array(data["krylov"]["sem"])

        # Filter NaN
        valid = ~np.isnan(P_s) & ~np.isnan(P_k)

        ax.errorbar(rho[valid], P_s[valid], yerr=sem_s[valid],
                    fmt='o-', label='Spectral', color='C0', capsize=3)
        ax.errorbar(rho[valid], P_k[valid], yerr=sem_k[valid],
                    fmt='s--', label='Krylov', color='C1', capsize=3)

        # Fit and annotate
        fit_s = fit_fermi_dirac(rho[valid], P_s[valid])
        fit_k = fit_fermi_dirac(rho[valid], P_k[valid])

        if fit_s and fit_k:
            if fit_k['rho_c'] < fit_s['rho_c'] * 0.95:
                ordering = "K<S (GEO2-like)"
                color = 'green'
            elif fit_s['rho_c'] < fit_k['rho_c'] * 0.95:
                ordering = "S<K (reversed)"
                color = 'red'
            else:
                ordering = "S≈K"
                color = 'gray'

            title = f"{config_name}\n{ordering}"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')

            info = f"ρc(S)={fit_s['rho_c']:.3f}\nρc(K)={fit_k['rho_c']:.3f}"
            ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.set_title(config_name, fontsize=10)

        ax.set_xlabel('ρ = K/d²')
        ax.set_ylabel('P(unreachable)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Connectivity Hypothesis Test (d={d})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "connectivity_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def analyze_connectivity_results(results: Dict):
    """Analyze criterion ordering for each connectivity type."""
    print("\n" + "=" * 70)
    print("CONNECTIVITY ANALYSIS")
    print("=" * 70)

    d = results["metadata"]["d"]

    ordering_summary = []

    for config_name, data in results["connectivity_tests"].items():
        print(f"\n### {config_name.upper()} ###")
        print(f"  L = {data['L']} operators, edges = {data['n_edges']}")

        k_values = np.array(data["k_values"])
        rho = k_values / d**2
        P_s = np.array(data["spectral"]["P"])
        P_k = np.array(data["krylov"]["P"])

        valid = ~np.isnan(P_s) & ~np.isnan(P_k)

        r_mean = np.nanmean(data["r_ratios"])
        print(f"  <r> = {r_mean:.3f} [{classify_spectrum(r_mean)}]")

        fit_s = fit_fermi_dirac(rho[valid], P_s[valid])
        fit_k = fit_fermi_dirac(rho[valid], P_k[valid])

        if fit_s:
            print(f"  Spectral: ρc={fit_s['rho_c']:.4f}, Δ={fit_s['delta']:.4f}, R²={fit_s['R2']:.3f}")
        else:
            print(f"  Spectral: fit failed")

        if fit_k:
            print(f"  Krylov:   ρc={fit_k['rho_c']:.4f}, Δ={fit_k['delta']:.4f}, R²={fit_k['R2']:.3f}")
        else:
            print(f"  Krylov:   fit failed")

        if fit_s and fit_k:
            ratio = fit_k['rho_c'] / fit_s['rho_c']
            if ratio < 0.9:
                ordering = "Krylov < Spectral"
            elif ratio > 1.1:
                ordering = "Spectral < Krylov"
            else:
                ordering = "Spectral ≈ Krylov"
            print(f"  ORDERING: {ordering} (ratio={ratio:.2f})")
            ordering_summary.append((config_name, data['n_edges'], ratio, r_mean))

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Connectivity vs Ordering")
    print("=" * 70)
    print(f"{'Config':<20} {'Edges':<8} {'ρc(K)/ρc(S)':<12} {'<r>':<8} Ordering")
    print("-" * 60)
    for config, edges, ratio, r in ordering_summary:
        if ratio < 0.9:
            ord_str = "K < S (GEO2-like)"
        elif ratio > 1.1:
            ord_str = "S < K"
        else:
            ord_str = "≈ equal"
        print(f"{config:<20} {edges:<8} {ratio:<12.2f} {r:<8.3f} {ord_str}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run experiment
    results = run_connectivity_experiment(
        n_qubits=4,      # d=16
        k_max=25,
        trials=30,       # Quick test (increase for publication)
        tau=0.99,
        seed=42
    )

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"connectivity_test_{timestamp}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results: {output_file}")

    # Generate plots
    fig_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
    plot_connectivity_results(results, fig_dir)

    # Analyze ordering
    analyze_connectivity_results(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
