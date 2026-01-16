#!/usr/bin/env python3
"""
Three-model integrability comparison with detailed equations and analysis.

Models:
1. Integrable Ising: H = Σ Jᵢ σᶻᵢσᶻᵢ₊₁ + Σ hᵢ σᶻᵢ
   - Diagonal in Z-basis, Poisson statistics (⟨r⟩ ≈ 0.39)

2. Near-Integrable: H = J Σ σᶻᵢσᶻᵢ₊₁ + h Σ σᶻᵢ + g Σ σˣᵢ
   - Transverse field breaks integrability, intermediate ⟨r⟩

3. Chaotic Heisenberg: H = Σ (Jˣᵢσˣᵢσˣᵢ₊₁ + Jʸᵢσʸᵢσʸᵢ₊₁ + Jᶻᵢσᶻᵢσᶻᵢ₊₁)
   - Random couplings, GOE statistics (⟨r⟩ ≈ 0.53)

Usage:
    python scripts/integrability/three_models_comparison.py

Author: Claude Code (research exploration)
Date: 2026-01-14
"""

import numpy as np
import qutip
from pathlib import Path
import pickle
from datetime import datetime
from typing import List, Dict, Optional
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reach import optimize, mathematics


# =============================================================================
# PUBLICATION STYLE
# =============================================================================

def set_style():
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

COLORS = {
    'moment': '#1B998B',
    'spectral': '#E94F37',
    'krylov': '#F39237',
}


# =============================================================================
# LEVEL SPACING
# =============================================================================

def compute_r_ratio(eigenvalues: np.ndarray) -> float:
    """Compute mean r-ratio for level spacing statistics."""
    E = np.sort(np.real(eigenvalues))
    gaps = np.diff(E)
    mask = gaps > 1e-12
    gaps = gaps[mask]
    if len(gaps) < 2:
        return np.nan
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return np.mean(r)


# =============================================================================
# HAMILTONIAN MODELS
# =============================================================================

def integrable_ising(n_qubits: int, rng: np.random.RandomState) -> qutip.Qobj:
    """
    Integrable Ising: H = Σᵢ Jᵢ σᶻᵢσᶻᵢ₊₁ + Σᵢ hᵢ σᶻᵢ

    Diagonal in computational basis → Poisson level statistics
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    # Z-Z couplings with random J
    for i in range(n_qubits - 1):
        J = rng.randn()
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + J * term

    # Z fields with random h
    for i in range(n_qubits):
        h = rng.randn()
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + h * term

    return H


def near_integrable_ising(n_qubits: int, g: float, rng: np.random.RandomState) -> qutip.Qobj:
    """
    Near-Integrable: H = J Σ σᶻᵢσᶻᵢ₊₁ + h Σ σᶻᵢ + g Σ σˣᵢ

    Transverse field g breaks integrability. g=0 → integrable, g→∞ → also integrable
    Maximum chaos around g ≈ J.
    """
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    J = 1.0  # Fixed coupling
    h = rng.randn() * 0.5  # Random longitudinal field

    # Z-Z couplings
    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + J * term

    # Z fields
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + h * term

    # X fields (transverse - breaks integrability)
    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sx
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + g * term

    return H


def chaotic_heisenberg(n_qubits: int, rng: np.random.RandomState) -> qutip.Qobj:
    """
    Chaotic Heisenberg: H = Σᵢ (Jˣᵢσˣᵢσˣᵢ₊₁ + Jʸᵢσʸᵢσʸᵢ₊₁ + Jᶻᵢσᶻᵢσᶻᵢ₊₁) + on-site

    All random couplings → GOE level statistics
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


def generate_ensemble(model: str, n_qubits: int, k: int,
                     rng: np.random.RandomState, **kwargs) -> List[qutip.Qobj]:
    """Generate ensemble of K Hamiltonians."""
    hams = []
    for _ in range(k):
        if model == "integrable":
            H = integrable_ising(n_qubits, rng)
        elif model == "near_integrable":
            g = kwargs.get("g", 0.3)
            H = near_integrable_ising(n_qubits, g, rng)
        elif model == "chaotic":
            H = chaotic_heisenberg(n_qubits, rng)
        else:
            raise ValueError(f"Unknown model: {model}")
        hams.append(H)
    return hams


# =============================================================================
# CRITERIA
# =============================================================================

def eval_moment(psi, phi, hams) -> bool:
    """Moment criterion. Returns True if UNREACHABLE."""
    try:
        return mathematics.is_unreachable_moment(psi, phi, hams)
    except:
        return False


def eval_spectral(psi, phi, hams, tau, rng) -> bool:
    """Spectral criterion. Returns True if UNREACHABLE."""
    try:
        result = optimize.maximize_spectral_overlap(
            psi, phi, hams, restarts=3, maxiter=50,
            seed=rng.randint(0, 2**31)
        )
        return result["best_value"] < tau
    except:
        return False


def eval_krylov(psi, phi, hams, tau, rng) -> bool:
    """Krylov criterion. Returns True if UNREACHABLE."""
    try:
        d = psi.shape[0]
        m = min(len(hams), d)
        result = optimize.maximize_krylov_score(
            psi, phi, hams, m=m, restarts=3, maxiter=50,
            seed=rng.randint(0, 2**31)
        )
        return result["best_value"] < tau
    except:
        return False


# =============================================================================
# FITTING
# =============================================================================

def fermi_dirac(rho, rho_c, delta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = (rho - rho_c) / delta
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(x))


def fit_rho_c(rho, P):
    try:
        mask = (P > 0.02) & (P < 0.98)
        if np.sum(mask) < 3:
            return None
        popt, _ = curve_fit(fermi_dirac, rho[mask], P[mask],
                           p0=[np.median(rho[mask]), 0.02],
                           bounds=([0, 0.001], [1.0, 0.5]), maxfev=10000)
        return popt[0]
    except:
        return None


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_three_models(n_qubits_list=[3, 4], k_max=16, trials=25, tau=0.99, seed=42):
    """Run all three models across dimensions."""
    rng = np.random.RandomState(seed)

    models = [
        ("integrable", {}, r"$H = \sum_i J_i \sigma^z_i\sigma^z_{i+1} + \sum_i h_i \sigma^z_i$"),
        ("near_integrable", {"g": 0.3}, r"$H = J\sum_i \sigma^z_i\sigma^z_{i+1} + h\sum_i \sigma^z_i + g\sum_i \sigma^x_i$"),
        ("chaotic", {}, r"$H = \sum_i (J^x_i \sigma^x_i\sigma^x_{i+1} + J^y_i \sigma^y_i\sigma^y_{i+1} + J^z_i \sigma^z_i\sigma^z_{i+1})$"),
    ]

    results = {
        "metadata": {
            "n_qubits_list": n_qubits_list,
            "k_max": k_max,
            "trials": trials,
            "tau": tau,
            "models": [(m[0], m[1], m[2]) for m in models],
            "timestamp": datetime.now().isoformat()
        },
        "data": {}
    }

    print("=" * 70)
    print("THREE-MODEL INTEGRABILITY COMPARISON")
    print("=" * 70)

    for n_qubits in n_qubits_list:
        d = 2 ** n_qubits
        k_values = np.arange(2, min(k_max + 1, d + 1), 2)

        for model_name, model_kwargs, model_eq in models:
            key = f"{model_name}_n{n_qubits}"
            print(f"\n{'='*60}")
            print(f"{model_name.upper()}, n={n_qubits}, d={d}")
            print(f"Equation: {model_eq}")
            print(f"{'='*60}")

            exp = {
                "model": model_name,
                "equation": model_eq,
                "n_qubits": n_qubits,
                "d": d,
                "k_values": k_values.tolist(),
                "moment": {"P": []},
                "spectral": {"P": []},
                "krylov": {"P": []},
                "r_ratios": [],
            }

            for k in k_values:
                print(f"  K={k} (rho={k/d**2:.4f})...", end=" ")

                counts = {"moment": 0, "spectral": 0, "krylov": 0}
                r_samples = []

                for trial in range(trials):
                    trial_rng = np.random.RandomState(rng.randint(0, 2**31))
                    hams = generate_ensemble(model_name, n_qubits, k, trial_rng, **model_kwargs)

                    # Sample r-ratio
                    if trial < 10:
                        lambdas = trial_rng.randn(k)
                        lambdas = lambdas / np.linalg.norm(lambdas)
                        H_comb = sum(l * H for l, H in zip(lambdas, hams))
                        eigs = np.linalg.eigvalsh(H_comb.full())
                        r = compute_r_ratio(eigs)
                        if not np.isnan(r):
                            r_samples.append(r)

                    psi = qutip.rand_ket(d)
                    phi = qutip.rand_ket(d)

                    if eval_moment(psi, phi, hams):
                        counts["moment"] += 1
                    if eval_spectral(psi, phi, hams, tau, trial_rng):
                        counts["spectral"] += 1
                    if eval_krylov(psi, phi, hams, tau, trial_rng):
                        counts["krylov"] += 1

                for crit in ["moment", "spectral", "krylov"]:
                    exp[crit]["P"].append(counts[crit] / trials)
                exp["r_ratios"].append(np.mean(r_samples) if r_samples else np.nan)

                r_str = f"r={np.mean(r_samples):.2f}" if r_samples else "r=?"
                print(f"M={counts['moment']/trials:.2f}, S={counts['spectral']/trials:.2f}, K={counts['krylov']/trials:.2f} [{r_str}]")

            results["data"][key] = exp

    return results


def plot_three_models(results: Dict, output_dir: Path):
    """Create publication-quality 2×3 figure."""
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_qubits_list = results["metadata"]["n_qubits_list"]
    model_names = ["integrable", "near_integrable", "chaotic"]
    model_labels = ["Integrable", "Near-Integrable", "Chaotic"]

    fig, axes = plt.subplots(len(n_qubits_list), 3, figsize=(12, 3.5*len(n_qubits_list)))
    if len(n_qubits_list) == 1:
        axes = axes.reshape(1, -1)

    for row, n_qubits in enumerate(n_qubits_list):
        d = 2 ** n_qubits

        for col, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
            ax = axes[row, col]
            key = f"{model_name}_n{n_qubits}"

            if key not in results["data"]:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            exp = results["data"][key]
            k_values = np.array(exp["k_values"])
            rho = k_values / d**2

            # Plot criteria
            for crit, color, marker in [("moment", COLORS["moment"], "^"),
                                        ("spectral", COLORS["spectral"], "o"),
                                        ("krylov", COLORS["krylov"], "s")]:
                P = np.array(exp[crit]["P"])
                ax.plot(rho, P, f'{marker}-', color=color, markersize=5,
                       label=crit.capitalize())

            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel(r'$\rho = K/d^2$')
            ax.set_ylabel('P(unreachable)')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.2)

            # Title with r-ratio
            r_mean = np.nanmean(exp["r_ratios"])
            r_class = "Poisson" if r_mean < 0.45 else ("GOE" if r_mean < 0.56 else "GUE")
            ax.set_title(f"{model_label}, d={d}\n" + r"$\langle r \rangle$" + f"={r_mean:.2f} ({r_class})", fontsize=9)

    # Add column equations at top
    equations = [
        r"$H = \sum_i J_i \sigma^z_i\sigma^z_{i+1} + h_i \sigma^z_i$",
        r"$H = J\sum_i \sigma^z\sigma^z + h\sum_i \sigma^z + g\sum_i \sigma^x$",
        r"$H = \sum_i (J^x\sigma^x\sigma^x + J^y\sigma^y\sigma^y + J^z\sigma^z\sigma^z)$"
    ]

    for col, eq in enumerate(equations):
        fig.text(0.17 + col*0.28, 0.98, eq, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Three-Criterion Analysis Across Integrability Levels',
                fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()

    output_file = output_dir / "three_models_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def create_rho_c_vs_r_ratio_plot(results: Dict, output_dir: Path):
    """Create plot showing ρ_c vs r-ratio (integrability correlation)."""
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Collect data points
    spectral_data = []  # (r, rho_c, label)
    krylov_data = []

    for key, exp in results["data"].items():
        d = exp["d"]
        r_mean = np.nanmean(exp["r_ratios"])

        k_values = np.array(exp["k_values"])
        rho = k_values / d**2

        # Fit ρ_c for spectral
        P_s = np.array(exp["spectral"]["P"])
        rho_c_s = fit_rho_c(rho, P_s)
        if rho_c_s is None and np.all(np.array(P_s) > 0.9):
            rho_c_s = 1.0  # Spectral never transitions

        # Fit ρ_c for krylov
        P_k = np.array(exp["krylov"]["P"])
        rho_c_k = fit_rho_c(rho, P_k)

        if not np.isnan(r_mean):
            label = f"{exp['model'][:3]}_d{d}"
            if rho_c_s is not None:
                spectral_data.append((r_mean, rho_c_s, label))
            if rho_c_k is not None:
                krylov_data.append((r_mean, rho_c_k, label))

    # Plot spectral
    if spectral_data:
        r_s, rho_s, labels_s = zip(*spectral_data)
        ax.scatter(r_s, rho_s, s=100, c=COLORS['spectral'], marker='o',
                  label='Spectral', edgecolors='black', linewidths=0.5, alpha=0.8)
        for r, rho, lab in spectral_data:
            if rho < 0.5:  # Don't label the "failed" points
                ax.annotate(lab, (r, rho), xytext=(5, 5), textcoords='offset points', fontsize=7)

    # Plot krylov
    if krylov_data:
        r_k, rho_k, labels_k = zip(*krylov_data)
        ax.scatter(r_k, rho_k, s=100, c=COLORS['krylov'], marker='s',
                  label='Krylov', edgecolors='black', linewidths=0.5, alpha=0.8)
        for r, rho, lab in krylov_data:
            ax.annotate(lab, (r, rho), xytext=(5, -10), textcoords='offset points', fontsize=7)

    # Reference lines
    ax.axvline(0.386, color='gray', linestyle='--', alpha=0.6, label='Poisson (integrable)')
    ax.axvline(0.531, color='gray', linestyle=':', alpha=0.6, label='GOE (chaotic)')

    # Shade regions
    ax.axvspan(0.0, 0.45, alpha=0.1, color='blue', label='_nolegend_')
    ax.axvspan(0.45, 0.7, alpha=0.1, color='red', label='_nolegend_')
    ax.text(0.30, 0.02, 'Integrable', fontsize=9, ha='center', transform=ax.get_xaxis_transform())
    ax.text(0.55, 0.02, 'Chaotic', fontsize=9, ha='center', transform=ax.get_xaxis_transform())

    ax.set_xlabel(r'Mean r-ratio $\langle r \rangle$', fontsize=11)
    ax.set_ylabel(r'Critical density $\rho_c$', fontsize=11)
    ax.set_title('Integrability vs Criterion Performance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(0.2, 0.7)
    ax.set_ylim(0, 0.15)
    ax.grid(True, alpha=0.2)

    # Add annotation
    ax.annotate('Spectral fails\n(ρ_c → ∞)', xy=(0.38, 0.12), fontsize=8,
               ha='center', style='italic', color=COLORS['spectral'])

    plt.tight_layout()

    output_file = output_dir / "rho_c_vs_r_ratio.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_file}")


def print_summary(results: Dict):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY: ρ_c VALUES BY MODEL AND CRITERION")
    print("=" * 70)

    print(f"\n{'Model':<20} {'d':>4} {'⟨r⟩':>6} {'ρ_c(M)':>8} {'ρ_c(S)':>8} {'ρ_c(K)':>8}")
    print("-" * 60)

    for key, exp in results["data"].items():
        d = exp["d"]
        r_mean = np.nanmean(exp["r_ratios"])

        k_values = np.array(exp["k_values"])
        rho = k_values / d**2

        rho_c_m = fit_rho_c(rho, np.array(exp["moment"]["P"]))
        rho_c_s = fit_rho_c(rho, np.array(exp["spectral"]["P"]))
        rho_c_k = fit_rho_c(rho, np.array(exp["krylov"]["P"]))

        # Handle special cases
        P_s = np.array(exp["spectral"]["P"])
        if rho_c_s is None and np.all(P_s > 0.9):
            rho_c_s_str = "∞ (fails)"
        elif rho_c_s is not None:
            rho_c_s_str = f"{rho_c_s:.4f}"
        else:
            rho_c_s_str = "N/A"

        rho_c_m_str = f"{rho_c_m:.4f}" if rho_c_m else "N/A"
        rho_c_k_str = f"{rho_c_k:.4f}" if rho_c_k else "N/A"

        print(f"{exp['model']:<20} {d:>4} {r_mean:>6.3f} {rho_c_m_str:>8} {rho_c_s_str:>8} {rho_c_k_str:>8}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run experiment
    results = run_three_models(
        n_qubits_list=[3, 4],
        k_max=16,
        trials=25,
        tau=0.99,
        seed=42
    )

    # Save
    output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"three_models_study_{timestamp}.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved results")

    # Plot
    fig_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
    plot_three_models(results, fig_dir)
    # NOTE: rho_c_vs_r_ratio.png removed from publication - commented out
    # create_rho_c_vs_r_ratio_plot(results, fig_dir)

    # Summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
