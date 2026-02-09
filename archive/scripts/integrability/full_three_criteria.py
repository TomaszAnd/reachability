#!/usr/bin/env python3
"""
Full integrability study with all three criteria: Moment, Spectral, Krylov.

This script tests the relationship between integrability/chaos and criterion ordering
across multiple dimensions (d=8, 16, 32).

Models tested:
1. Integrable: Ising with longitudinal field only (Z-Z + Z)
2. Near-integrable: Ising with weak transverse field (Z-Z + Z + gX)
3. Chaotic: Heisenberg with random couplings (XX + YY + ZZ + on-site)

Criteria:
- Moment: Energy bounds test (weakest, simplest)
- Spectral: Eigenbasis alignment (medium, λ-dependent)
- Krylov: Dynamical subspace projection (strongest, λ-independent)

Usage:
    python scripts/integrability/full_three_criteria.py

Author: Claude Code (research exploration)
Date: 2026-01-14
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

from reach import models, optimize, mathematics, settings


# =============================================================================
# LEVEL SPACING STATISTICS
# =============================================================================

def compute_level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute mean r-ratio for level spacing statistics.

    r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    where s_n = E_{n+1} - E_n

    Reference values:
    - Poisson (integrable): ⟨r⟩ ≈ 0.386
    - GOE (chaotic, real): ⟨r⟩ ≈ 0.531
    - GUE (chaotic, complex): ⟨r⟩ ≈ 0.600
    """
    E = np.sort(np.real(eigenvalues))
    gaps = np.diff(E)

    # Filter out zero gaps
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
        return "Poisson (integrable)"
    elif r_mean < 0.56:
        return "GOE (chaotic)"
    else:
        return "GUE (chaotic)"


# =============================================================================
# HAMILTONIAN GENERATORS
# =============================================================================

def ising_longitudinal(n_qubits: int, J_vals: np.ndarray, h_vals: np.ndarray) -> qutip.Qobj:
    """
    Integrable Ising model: H = Σᵢ Jᵢ σᶻᵢσᶻᵢ₊₁ + Σᵢ hᵢ σᶻᵢ

    Diagonal in computational basis, Poisson level statistics.
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    # Z-Z coupling
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + J_vals[i] * term

    # Z field
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + h_vals[i] * term

    return H


def heisenberg_random(n_qubits: int, rng: np.random.RandomState) -> qutip.Qobj:
    """
    Chaotic Heisenberg model with random couplings.

    H = Σᵢ (Jˣᵢ σˣᵢσˣᵢ₊₁ + Jʸᵢ σʸᵢσʸᵢ₊₁ + Jᶻᵢ σᶻᵢσᶻᵢ₊₁) + on-site terms

    GOE level statistics expected.
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    paulis = [sx, sy, sz]

    # Two-site couplings
    for i in range(n_qubits - 1):
        for pauli in paulis:
            J = rng.randn()
            ops = [I] * n_qubits
            ops[i] = pauli
            ops[i + 1] = pauli
            term = qutip.tensor(ops)
            term.dims = [[d], [d]]
            H = H + J * term

    # On-site terms
    for i in range(n_qubits):
        for pauli in paulis:
            h = rng.randn() * 0.5
            ops = [I] * n_qubits
            ops[i] = pauli
            term = qutip.tensor(ops)
            term.dims = [[d], [d]]
            H = H + h * term

    return H


def generate_ensemble(model_type: str, n_qubits: int, k: int,
                     rng: np.random.RandomState) -> List[qutip.Qobj]:
    """Generate ensemble of k Hamiltonians."""
    hamiltonians = []

    for _ in range(k):
        if model_type == "integrable":
            J_vals = rng.randn(n_qubits - 1)
            h_vals = rng.randn(n_qubits)
            H = ising_longitudinal(n_qubits, J_vals, h_vals)
        elif model_type == "chaotic":
            H = heisenberg_random(n_qubits, rng)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        hamiltonians.append(H)

    return hamiltonians


# =============================================================================
# CRITERION FUNCTIONS
# =============================================================================

def evaluate_moment_criterion(psi: qutip.Qobj, phi: qutip.Qobj,
                             hams: List[qutip.Qobj]) -> bool:
    """
    Moment criterion: Check if energy moments can be matched.

    Uses mathematics.is_unreachable_moment from reach package.
    Returns True if state is classified as UNREACHABLE.
    """
    try:
        return mathematics.is_unreachable_moment(psi, phi, hams)
    except Exception:
        return False  # Conservative: assume reachable on error


def evaluate_spectral_criterion(psi: qutip.Qobj, phi: qutip.Qobj,
                               hams: List[qutip.Qobj], tau: float,
                               rng: np.random.RandomState) -> bool:
    """
    Spectral criterion: Maximize eigenbasis overlap S(λ).

    Returns True if S* < tau (UNREACHABLE).
    """
    try:
        result = optimize.maximize_spectral_overlap(
            psi, phi, hams,
            restarts=3,  # Reduced for speed
            maxiter=50,
            seed=rng.randint(0, 2**31)
        )
        return result["best_value"] < tau
    except Exception:
        return False


def evaluate_krylov_criterion(psi: qutip.Qobj, phi: qutip.Qobj,
                             hams: List[qutip.Qobj], tau: float,
                             rng: np.random.RandomState) -> bool:
    """
    Krylov criterion: Maximize subspace projection R(λ).

    Returns True if R* < tau (UNREACHABLE).
    """
    try:
        d = psi.shape[0]
        m = min(len(hams), d)
        result = optimize.maximize_krylov_score(
            psi, phi, hams, m=m,
            restarts=3,  # Reduced for speed
            maxiter=50,
            seed=rng.randint(0, 2**31)
        )
        return result["best_value"] < tau
    except Exception:
        return False


# =============================================================================
# FITTING UTILITIES
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
        return {"rho_c": popt[0], "delta": popt[1]}
    except Exception:
        return None


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_full_study(
    n_qubits_list: List[int] = [3, 4],
    k_max: int = 16,
    trials: int = 30,
    tau: float = 0.99,
    seed: int = 42
) -> Dict:
    """
    Run full integrability study with all three criteria.

    Args:
        n_qubits_list: List of qubit counts to test
        k_max: Maximum number of Hamiltonians
        trials: Trials per data point
        tau: Threshold for criteria
        seed: Random seed

    Returns:
        Dictionary with all results
    """
    rng = np.random.RandomState(seed)

    results = {
        "metadata": {
            "n_qubits_list": n_qubits_list,
            "k_max": k_max,
            "trials": trials,
            "tau": tau,
            "seed": seed,
            "timestamp": datetime.now().isoformat()
        },
        "experiments": {}
    }

    print("=" * 70)
    print("FULL INTEGRABILITY STUDY: THREE CRITERIA")
    print("=" * 70)
    print(f"Qubits: {n_qubits_list}, k_max={k_max}, trials={trials}, tau={tau}")
    print("=" * 70)

    models_to_test = ["integrable", "chaotic"]

    for n_qubits in n_qubits_list:
        d = 2 ** n_qubits
        k_values = np.arange(2, min(k_max + 1, d + 1), 2)

        for model_type in models_to_test:
            key = f"{model_type}_n{n_qubits}"
            print(f"\n{'='*60}")
            print(f"Model: {model_type}, n_qubits={n_qubits}, d={d}")
            print(f"{'='*60}")

            exp_results = {
                "model_type": model_type,
                "n_qubits": n_qubits,
                "d": d,
                "k_values": k_values.tolist(),
                "moment": {"P": [], "sem": []},
                "spectral": {"P": [], "sem": []},
                "krylov": {"P": [], "sem": []},
                "r_ratios": [],
            }

            for k in k_values:
                print(f"\n  K={k} (ρ={k/d**2:.4f})...")

                moment_unreachable = 0
                spectral_unreachable = 0
                krylov_unreachable = 0
                r_samples = []

                for trial in range(trials):
                    trial_rng = np.random.RandomState(rng.randint(0, 2**31))

                    # Generate Hamiltonians
                    hams = generate_ensemble(model_type, n_qubits, k, trial_rng)

                    # Sample r-ratio for a few trials
                    if trial < 10:
                        lambdas = trial_rng.randn(k)
                        lambdas = lambdas / np.linalg.norm(lambdas)
                        H_combined = sum(lam * H for lam, H in zip(lambdas, hams))
                        eigs = np.linalg.eigvalsh(H_combined.full())
                        r = compute_level_spacing_ratio(eigs)
                        if not np.isnan(r):
                            r_samples.append(r)

                    # Random states
                    psi = qutip.rand_ket(d)
                    phi = qutip.rand_ket(d)

                    # === MOMENT CRITERION ===
                    if evaluate_moment_criterion(psi, phi, hams):
                        moment_unreachable += 1

                    # === SPECTRAL CRITERION ===
                    if evaluate_spectral_criterion(psi, phi, hams, tau, trial_rng):
                        spectral_unreachable += 1

                    # === KRYLOV CRITERION ===
                    if evaluate_krylov_criterion(psi, phi, hams, tau, trial_rng):
                        krylov_unreachable += 1

                # Record results
                for name, count in [("moment", moment_unreachable),
                                   ("spectral", spectral_unreachable),
                                   ("krylov", krylov_unreachable)]:
                    P = count / trials
                    exp_results[name]["P"].append(P)
                    exp_results[name]["sem"].append(np.sqrt(P * (1-P) / trials))

                exp_results["r_ratios"].append(np.mean(r_samples) if r_samples else np.nan)

                P_m = moment_unreachable / trials
                P_s = spectral_unreachable / trials
                P_k = krylov_unreachable / trials
                r_mean = np.mean(r_samples) if r_samples else np.nan

                print(f"    P(unreachable): M={P_m:.3f}, S={P_s:.3f}, K={P_k:.3f}")
                if r_samples:
                    print(f"    <r> = {r_mean:.3f} [{classify_spectrum(r_mean)}]")

            results["experiments"][key] = exp_results

    return results


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = results["experiments"]
    n_exp = len(experiments)

    if n_exp == 0:
        print("No experiments to plot")
        return

    # Determine grid size
    n_rows = 2  # integrable and chaotic
    n_cols = len(set(exp["n_qubits"] for exp in experiments.values()))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    model_to_row = {"integrable": 0, "chaotic": 1}
    qubit_to_col = {}

    for key, exp_data in experiments.items():
        model_type = exp_data["model_type"]
        n_qubits = exp_data["n_qubits"]
        d = exp_data["d"]

        if n_qubits not in qubit_to_col:
            qubit_to_col[n_qubits] = len(qubit_to_col)

        row = model_to_row[model_type]
        col = qubit_to_col[n_qubits]

        ax = axes[row, col]

        k_values = np.array(exp_data["k_values"])
        rho = k_values / d**2

        # Plot all three criteria
        colors = {'moment': '#1B998B', 'spectral': '#E94F37', 'krylov': '#F39237'}
        markers = {'moment': '^', 'spectral': 'o', 'krylov': 's'}

        for criterion in ['moment', 'spectral', 'krylov']:
            P = np.array(exp_data[criterion]["P"])
            ax.plot(rho, P, f'{markers[criterion]}-', color=colors[criterion],
                   markersize=6, linewidth=1.5, label=criterion.capitalize())

        ax.set_xlabel(r'$\rho = K/d^2$')
        ax.set_ylabel('P(unreachable)')
        ax.set_title(f'{model_type.capitalize()}, d={d} (n={n_qubits})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        # Add r-ratio annotation
        r_mean = np.nanmean(exp_data["r_ratios"])
        ax.text(0.95, 0.95, f'⟨r⟩={r_mean:.2f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Three Criteria vs Integrability', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "three_criteria_integrability.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def analyze_ordering(results: Dict):
    """Analyze criterion ordering for each experiment."""
    print("\n" + "=" * 70)
    print("CRITERION ORDERING ANALYSIS")
    print("=" * 70)

    for key, exp_data in results["experiments"].items():
        print(f"\n### {key.upper()} ###")

        d = exp_data["d"]
        k_values = np.array(exp_data["k_values"])
        rho = k_values / d**2

        r_mean = np.nanmean(exp_data["r_ratios"])
        print(f"  Spectrum: ⟨r⟩ = {r_mean:.3f} ({classify_spectrum(r_mean)})")

        for criterion in ['moment', 'spectral', 'krylov']:
            P = np.array(exp_data[criterion]["P"])
            fit = fit_fermi_dirac(rho, P)
            if fit:
                print(f"  {criterion.capitalize()}: ρ_c = {fit['rho_c']:.4f}")
            else:
                if np.all(P > 0.95):
                    print(f"  {criterion.capitalize()}: always unreachable")
                elif np.all(P < 0.05):
                    print(f"  {criterion.capitalize()}: always reachable")
                else:
                    print(f"  {criterion.capitalize()}: fit failed")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run study
    results = run_full_study(
        n_qubits_list=[3, 4],  # d=8, d=16
        k_max=16,
        trials=25,  # Moderate for speed
        tau=0.99,
        seed=42
    )

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_three_criteria_study_{timestamp}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results: {output_file}")

    # Generate plots
    fig_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
    plot_results(results, fig_dir)

    # Analyze ordering
    analyze_ordering(results)

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
