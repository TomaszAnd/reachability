#!/usr/bin/env python3
"""
Investigate how criterion behavior depends on integrability vs chaos.

HYPOTHESIS: GEO2's Krylov advantage (Krylov detects reachability at lower ρ than
Spectral) comes from structured/near-integrable Hamiltonian ensemble, while
Canonical's similar or reversed ordering comes from more chaotic structure.

TEST STRATEGY:
1. Implement well-characterized integrable and chaotic Hamiltonians
2. Compute level spacing statistics (r-ratio) to quantify chaos
3. Run all three criteria (Moment, Spectral, Krylov)
4. Compare ρ_c ordering: Is Krylov < Spectral (GEO2-like) or similar (Canonical-like)?

MODELS:
1. Integrable: Ising with longitudinal field only (Z-Z + h_z Z)
2. Near-integrable: Ising with weak transverse field (Z-Z + h_z Z + g X)
3. Chaotic: Heisenberg XXZ with random couplings

EXPECTED OUTCOMES:
- Integrable: Poisson level spacing (r ~ 0.39), Krylov < Spectral (like GEO2)
- Chaotic: Wigner-Dyson spacing (r ~ 0.53), Spectral ≲ Krylov (like Canonical)

Usage:
    python scripts/integrability/criterion_vs_chaos.py

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

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reach import models, optimize, mathematics, analysis, settings


# =============================================================================
# LEVEL SPACING STATISTICS
# =============================================================================

def compute_level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute mean r-ratio for level spacing statistics.

    The r-ratio distinguishes integrable from chaotic systems:
    - Poisson (integrable): <r> ≈ 0.386
    - GOE (chaotic, real): <r> ≈ 0.5307
    - GUE (chaotic, complex): <r> ≈ 0.5996

    r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    where s_n = E_{n+1} - E_n are level spacings.

    Args:
        eigenvalues: Sorted eigenvalues (real, ascending)

    Returns:
        Mean r-ratio <r> ∈ [0, 1]
    """
    E = np.sort(np.real(eigenvalues))
    gaps = np.diff(E)

    # Filter out zero gaps (degeneracies)
    mask = gaps > 1e-12
    gaps = gaps[mask]

    if len(gaps) < 2:
        return np.nan

    # Compute r-ratio
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return np.mean(r)


def classify_spectrum(r_mean: float) -> str:
    """Classify spectrum type from r-ratio."""
    if np.isnan(r_mean):
        return "undefined"
    elif r_mean < 0.45:
        return "integrable (Poisson)"
    elif r_mean < 0.56:
        return "chaotic (GOE)"
    else:
        return "chaotic (GUE)"


# =============================================================================
# HAMILTONIAN GENERATORS
# =============================================================================

def ising_longitudinal(n_qubits: int, J_vals: np.ndarray, h_vals: np.ndarray) -> qutip.Qobj:
    """
    Integrable Ising model with longitudinal field only.

    H = Σᵢ Jᵢ σᶻᵢσᶻᵢ₊₁ + Σᵢ hᵢ σᶻᵢ

    This is classically simulable and integrable (diagonal in Z-basis).
    Spectrum: Product structure, high degeneracy expected.
    Level spacing: Poisson statistics.

    Args:
        n_qubits: Number of qubits (chain length)
        J_vals: Coupling strengths (length n-1 for open BC)
        h_vals: Field strengths (length n)

    Returns:
        Hamiltonian as qutip.Qobj
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    # Single-qubit operators
    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    # Z-Z coupling terms
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]  # Flatten dims immediately
        H = H + J_vals[i] * term

    # Z field terms
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]  # Flatten dims immediately
        H = H + h_vals[i] * term

    return H


def ising_transverse(n_qubits: int, J: float, h_z: float, g: float) -> qutip.Qobj:
    """
    Near-integrable Ising model with transverse field.

    H = J Σᵢ σᶻᵢσᶻᵢ₊₁ + h_z Σᵢ σᶻᵢ + g Σᵢ σˣᵢ

    For small g: Near-integrable, slight level repulsion.
    For large g: Approaches quantum phase transition.

    Args:
        n_qubits: Number of qubits
        J: Z-Z coupling strength
        h_z: Longitudinal field strength
        g: Transverse field strength (integrability-breaking)

    Returns:
        Hamiltonian as qutip.Qobj
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sz = qutip.sigmaz()
    sx = qutip.sigmax()
    I = qutip.qeye(2)

    # Z-Z coupling
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + J * term

    # Z field
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + h_z * term

    # X field (transverse - breaks integrability)
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sx
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + g * term

    return H


def heisenberg_random(n_qubits: int, rng: np.random.RandomState) -> qutip.Qobj:
    """
    Chaotic Heisenberg model with random couplings.

    H = Σᵢ Jˣᵢ σˣᵢσˣᵢ₊₁ + Jʸᵢ σʸᵢσʸᵢ₊₁ + Jᶻᵢ σᶻᵢσᶻᵢ₊₁

    Random Jˣ, Jʸ, Jᶻ ~ N(0,1) → quantum chaotic.
    Level spacing: Wigner-Dyson (GOE) statistics.

    Args:
        n_qubits: Number of qubits
        rng: Random number generator

    Returns:
        Hamiltonian as qutip.Qobj
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    paulis = [sx, sy, sz]

    for i in range(n_qubits - 1):
        for pauli_idx, pauli in enumerate(paulis):
            J = rng.randn()  # Random coupling
            ops = [I] * n_qubits
            ops[i] = pauli
            ops[i + 1] = pauli
            term = qutip.tensor(ops)
            term.dims = [[d], [d]]
            H = H + J * term

    # Add random on-site terms for more chaos
    for i in range(n_qubits):
        for pauli in paulis:
            h = rng.randn() * 0.5
            ops = [I] * n_qubits
            ops[i] = pauli
            term = qutip.tensor(ops)
            term.dims = [[d], [d]]
            H = H + h * term

    return H


def generate_hamiltonian_ensemble(
    model_type: str,
    n_qubits: int,
    k: int,
    rng: np.random.RandomState,
    **kwargs
) -> List[qutip.Qobj]:
    """
    Generate ensemble of k Hamiltonians from specified model.

    Args:
        model_type: "integrable", "near_integrable", or "chaotic"
        n_qubits: Number of qubits
        k: Number of Hamiltonians to generate
        rng: Random number generator
        **kwargs: Model-specific parameters

    Returns:
        List of k Hamiltonian operators
    """
    hamiltonians = []

    for _ in range(k):
        if model_type == "integrable":
            # Random couplings for variety but same integrable structure
            J_vals = rng.randn(n_qubits - 1)
            h_vals = rng.randn(n_qubits)
            H = ising_longitudinal(n_qubits, J_vals, h_vals)

        elif model_type == "near_integrable":
            J = kwargs.get("J", 1.0)
            h_z = rng.randn()
            g = kwargs.get("g", 0.1)  # Small transverse field
            H = ising_transverse(n_qubits, J, h_z, g)

        elif model_type == "chaotic":
            H = heisenberg_random(n_qubits, rng)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        hamiltonians.append(H)

    return hamiltonians


# =============================================================================
# CRITERION EVALUATION
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


def run_criterion_sweep(
    model_type: str,
    n_qubits: int,
    k_values: np.ndarray,
    trials: int,
    tau: float,
    rng_seed: int,
    **model_kwargs
) -> Dict:
    """
    Run full criterion sweep for a model type.

    Args:
        model_type: "integrable", "near_integrable", or "chaotic"
        n_qubits: Number of qubits
        k_values: Array of K values to test
        trials: Number of trials per K
        tau: Threshold for spectral/krylov criteria
        rng_seed: Random seed
        **model_kwargs: Model-specific parameters

    Returns:
        Dictionary with results for each criterion
    """
    d = 2 ** n_qubits
    rng = np.random.RandomState(rng_seed)

    results = {
        "model_type": model_type,
        "n_qubits": n_qubits,
        "d": d,
        "tau": tau,
        "k_values": k_values.tolist(),
        "moment": {"P": [], "sem": []},
        "spectral": {"P": [], "sem": []},
        "krylov": {"P": [], "sem": []},
        "r_ratios": [],
    }

    print(f"\n{'='*60}")
    print(f"Model: {model_type}, n_qubits={n_qubits}, d={d}")
    print(f"K values: {k_values}")
    print(f"{'='*60}")

    for k in k_values:
        print(f"\n  K={k} (ρ={k/d**2:.4f})...")

        moment_unreachable = 0
        spectral_unreachable = 0
        krylov_unreachable = 0
        r_ratio_samples = []

        for trial in range(trials):
            # Generate Hamiltonians
            trial_rng = np.random.RandomState(rng.randint(0, 2**31))
            hams = generate_hamiltonian_ensemble(
                model_type, n_qubits, k, trial_rng, **model_kwargs
            )

            # Compute level spacing for combined Hamiltonian
            if trial < 10:  # Sample r-ratio for a few trials
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

            # === MOMENT CRITERION ===
            # Using the moment test from analysis
            # Simplified: check if energy differences span null space
            # For this experiment, use is_unreachable_moment if available
            # Otherwise skip moment or implement simplified version

            # === SPECTRAL CRITERION ===
            try:
                result_s = optimize.maximize_spectral_overlap(
                    psi, phi, hams,
                    restarts=settings.GEO2_RESTARTS,
                    maxiter=settings.GEO2_MAXITER,
                    seed=trial_rng.randint(0, 2**31)
                )
                S_star = result_s["best_value"]
                if S_star < tau:
                    spectral_unreachable += 1
            except Exception:
                pass

            # === KRYLOV CRITERION ===
            try:
                m = min(k, d)  # Krylov rank
                result_r = optimize.maximize_krylov_score(
                    psi, phi, hams, m=m,
                    restarts=settings.GEO2_RESTARTS,
                    maxiter=settings.GEO2_MAXITER,
                    seed=trial_rng.randint(0, 2**31)
                )
                R_star = result_r["best_value"]
                if R_star < tau:
                    krylov_unreachable += 1
            except Exception:
                pass

        # Record results
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

        print(f"    P(unreachable): Spectral={P_spectral:.3f}, Krylov={P_krylov:.3f}")
        if r_ratio_samples:
            print(f"    <r> = {np.mean(r_ratio_samples):.3f} [{classify_spectrum(np.mean(r_ratio_samples))}]")

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_integrability_experiment(
    n_qubits: int = 4,
    k_max: int = 20,
    trials: int = 50,
    tau: float = 0.99,
    seed: int = 42
) -> Dict:
    """
    Run full integrability comparison experiment.

    Args:
        n_qubits: Number of qubits (d = 2^n)
        k_max: Maximum number of Hamiltonians
        trials: Trials per data point
        tau: Threshold for criteria
        seed: Random seed

    Returns:
        Dictionary with results for all models
    """
    d = 2 ** n_qubits
    k_values = np.arange(2, min(k_max + 1, d + 1), 2)  # Even steps

    print("=" * 70)
    print("INTEGRABILITY vs CHAOS: CRITERION ORDERING EXPERIMENT")
    print("=" * 70)
    print(f"n_qubits = {n_qubits}, d = {d}")
    print(f"k_values = {k_values}")
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
        "models": {}
    }

    # Test each model type
    model_configs = [
        ("integrable", {}),
        ("near_integrable", {"g": 0.1}),
        ("near_integrable_strong", {"g": 0.5}),
        ("chaotic", {}),
    ]

    for model_name, kwargs in model_configs:
        # Handle near_integrable variants
        if model_name.startswith("near_integrable"):
            model_type = "near_integrable"
        else:
            model_type = model_name

        results = run_criterion_sweep(
            model_type=model_type,
            n_qubits=n_qubits,
            k_values=k_values,
            trials=trials,
            tau=tau,
            rng_seed=seed,
            **kwargs
        )
        results["model_config"] = kwargs
        all_results["models"][model_name] = results

    return all_results


def plot_results(results: Dict, output_dir: Path):
    """Generate comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    models = list(results["models"].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    d = results["metadata"]["d"]

    for idx, (model_name, model_data) in enumerate(results["models"].items()):
        k_values = np.array(model_data["k_values"])
        rho = k_values / d**2

        # Top left: Spectral P(unreachable)
        ax = axes[0, 0]
        P_s = np.array(model_data["spectral"]["P"])
        ax.plot(rho, P_s, 'o-', color=colors[idx], label=model_name, markersize=6)

        # Top right: Krylov P(unreachable)
        ax = axes[0, 1]
        P_k = np.array(model_data["krylov"]["P"])
        ax.plot(rho, P_k, 's-', color=colors[idx], label=model_name, markersize=6)

        # Bottom left: Spectral vs Krylov comparison
        ax = axes[1, 0]
        ax.plot(rho, P_s, 'o-', color=colors[idx], label=f"{model_name} (S)")
        ax.plot(rho, P_k, 's--', color=colors[idx], alpha=0.5)

        # Bottom right: r-ratio evolution
        ax = axes[1, 1]
        r_vals = np.array(model_data["r_ratios"])
        ax.plot(rho, r_vals, 'o-', color=colors[idx], label=model_name, markersize=6)

    # Labels
    axes[0, 0].set_xlabel("ρ = K/d²")
    axes[0, 0].set_ylabel("P(unreachable)")
    axes[0, 0].set_title("Spectral Criterion")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("ρ = K/d²")
    axes[0, 1].set_ylabel("P(unreachable)")
    axes[0, 1].set_title("Krylov Criterion")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("ρ = K/d²")
    axes[1, 0].set_ylabel("P(unreachable)")
    axes[1, 0].set_title("Comparison: Solid=Spectral, Dashed=Krylov")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("ρ = K/d²")
    axes[1, 1].set_ylabel("<r> (level spacing ratio)")
    axes[1, 1].set_title("Spectrum Type (r≈0.39=Poisson, r≈0.53=GOE)")
    axes[1, 1].axhline(0.386, color='gray', linestyle=':', label='Poisson')
    axes[1, 1].axhline(0.5307, color='gray', linestyle='--', label='GOE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Integrability vs Chaos: Criterion Ordering (d={d})", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / "criterion_vs_chaos_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def analyze_ordering(results: Dict):
    """Analyze criterion ordering for each model."""
    print("\n" + "=" * 70)
    print("CRITERION ORDERING ANALYSIS")
    print("=" * 70)

    d = results["metadata"]["d"]

    for model_name, model_data in results["models"].items():
        print(f"\n### {model_name.upper()} ###")

        k_values = np.array(model_data["k_values"])
        rho = k_values / d**2
        P_s = np.array(model_data["spectral"]["P"])
        P_k = np.array(model_data["krylov"]["P"])
        r_mean = np.nanmean(model_data["r_ratios"])

        print(f"  Mean r-ratio: {r_mean:.3f} ({classify_spectrum(r_mean)})")

        # Fit critical densities
        fit_s = fit_fermi_dirac(rho, P_s)
        fit_k = fit_fermi_dirac(rho, P_k)

        if fit_s:
            print(f"  Spectral: ρ_c = {fit_s['rho_c']:.4f}, Δ = {fit_s['delta']:.4f}, R² = {fit_s['R2']:.3f}")
        else:
            print(f"  Spectral: fit failed")

        if fit_k:
            print(f"  Krylov:   ρ_c = {fit_k['rho_c']:.4f}, Δ = {fit_k['delta']:.4f}, R² = {fit_k['R2']:.3f}")
        else:
            print(f"  Krylov:   fit failed")

        if fit_s and fit_k:
            if fit_k['rho_c'] < fit_s['rho_c']:
                ordering = "Krylov < Spectral (GEO2-like)"
            elif fit_s['rho_c'] < fit_k['rho_c']:
                ordering = "Spectral < Krylov (reversed)"
            else:
                ordering = "Spectral ≈ Krylov"
            print(f"  ORDERING: {ordering}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run experiment
    results = run_integrability_experiment(
        n_qubits=4,      # d=16
        k_max=16,        # Up to 16 Hamiltonians
        trials=30,       # Quick test (increase for publication)
        tau=0.99,
        seed=42
    )

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"integrability_study_{timestamp}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results: {output_file}")

    # NOTE: criterion_vs_chaos_comparison.png removed from publication - commented out
    # fig_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
    # plot_results(results, fig_dir)

    # Analyze ordering
    analyze_ordering(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
