#!/usr/bin/env python3
"""
Extended integrability study with larger dimensions and g-sweep.

Extensions from three_models_comparison.py:
1. Includes d=32 (n_qubits=5) for larger scale testing
2. Sweeps g values: 0.1, 0.3, 0.5, 0.7, 1.0 for near-integrable model
3. Uses 50 trials (instead of 25) for better statistics
4. Creates comprehensive g-sweep visualization

Usage:
    python scripts/integrability/extended_integrability_study.py

Expected runtime: ~2-4 hours (depending on hardware)

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
# STYLE
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
    """Integrable Ising: H = Σᵢ Jᵢ σᶻᵢσᶻᵢ₊₁ + Σᵢ hᵢ σᶻᵢ"""
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    for i in range(n_qubits - 1):
        J = rng.randn()
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + J * term

    for i in range(n_qubits):
        h = rng.randn()
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + h * term

    return H


def near_integrable_ising(n_qubits: int, g: float, rng: np.random.RandomState) -> qutip.Qobj:
    """Near-Integrable: H = J Σ σᶻᵢσᶻᵢ₊₁ + h Σ σᶻᵢ + g Σ σˣᵢ"""
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sx = qutip.sigmax()
    sz = qutip.sigmaz()
    I = qutip.qeye(2)

    J = 1.0
    h = rng.randn() * 0.5

    for i in range(n_qubits - 1):
        ops = [I] * n_qubits
        ops[i] = sz
        ops[i + 1] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + J * term

    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sz
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + h * term

    for i in range(n_qubits):
        ops = [I] * n_qubits
        ops[i] = sx
        term = qutip.tensor(ops)
        term.dims = [[d], [d]]
        H = H + g * term

    return H


def chaotic_heisenberg(n_qubits: int, rng: np.random.RandomState) -> qutip.Qobj:
    """Chaotic Heisenberg with random couplings."""
    d = 2 ** n_qubits
    H = qutip.Qobj(np.zeros((d, d), dtype=complex))

    sx = qutip.sigmax()
    sy = qutip.sigmay()
    sz = qutip.sigmaz()
    I = qutip.qeye(2)
    paulis = [sx, sy, sz]

    for i in range(n_qubits - 1):
        for pauli in paulis:
            J = rng.randn()
            ops = [I] * n_qubits
            ops[i] = pauli
            ops[i + 1] = pauli
            term = qutip.tensor(ops)
            term.dims = [[d], [d]]
            H = H + J * term

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
# EXTENDED EXPERIMENTS
# =============================================================================

def run_extended_study(n_qubits_list=[3, 4, 5], k_max=20, trials=50, tau=0.99, seed=42):
    """Run extended study with d=32 and more trials."""
    rng = np.random.RandomState(seed)

    models = [
        ("integrable", {}, r"$H = \sum_i J_i \sigma^z_i\sigma^z_{i+1} + h_i \sigma^z_i$"),
        ("near_integrable", {"g": 0.5}, r"$H = J\sum \sigma^z\sigma^z + h\sum \sigma^z + g\sum \sigma^x$"),
        ("chaotic", {}, r"$H = \sum_i (J^x\sigma^x\sigma^x + J^y\sigma^y\sigma^y + J^z\sigma^z\sigma^z)$"),
    ]

    results = {
        "metadata": {
            "n_qubits_list": n_qubits_list,
            "k_max": k_max,
            "trials": trials,
            "tau": tau,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "extended"
        },
        "data": {}
    }

    print("=" * 70)
    print("EXTENDED INTEGRABILITY STUDY")
    print(f"Dimensions: d = {[2**n for n in n_qubits_list]}")
    print(f"Trials: {trials}")
    print("=" * 70)

    for n_qubits in n_qubits_list:
        d = 2 ** n_qubits
        k_values = np.arange(2, min(k_max + 1, d + 1), 2)

        for model_name, model_kwargs, model_eq in models:
            key = f"{model_name}_n{n_qubits}"
            print(f"\n{'='*60}")
            print(f"{model_name.upper()}, n={n_qubits}, d={d}")
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
                print(f"  K={k} (rho={k/d**2:.4f})...", end=" ", flush=True)

                counts = {"moment": 0, "spectral": 0, "krylov": 0}
                r_samples = []

                for trial in range(trials):
                    trial_rng = np.random.RandomState(rng.randint(0, 2**31))
                    hams = generate_ensemble(model_name, n_qubits, k, trial_rng, **model_kwargs)

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


def run_g_sweep(n_qubits=4, g_values=[0.1, 0.3, 0.5, 0.7, 1.0], k_max=16, trials=50, tau=0.99, seed=42):
    """
    Sweep transverse field g in near-integrable model.

    g controls the integrability:
    - g=0: Fully integrable (diagonal)
    - g≈J: Maximum chaos
    - g→∞: Another integrable limit (X-basis diagonal)
    """
    rng = np.random.RandomState(seed)

    d = 2 ** n_qubits
    k_values = np.arange(2, min(k_max + 1, d + 1), 2)

    results = {
        "metadata": {
            "n_qubits": n_qubits,
            "d": d,
            "g_values": g_values,
            "k_max": k_max,
            "trials": trials,
            "tau": tau,
            "timestamp": datetime.now().isoformat(),
            "experiment_type": "g_sweep"
        },
        "data": {}
    }

    print("\n" + "=" * 70)
    print("G-SWEEP: NEAR-INTEGRABLE MODEL")
    print(f"n_qubits={n_qubits}, d={d}")
    print(f"g values: {g_values}")
    print("=" * 70)

    for g in g_values:
        key = f"g{g:.2f}"
        print(f"\n--- g = {g} ---")

        exp = {
            "g": g,
            "k_values": k_values.tolist(),
            "moment": {"P": []},
            "spectral": {"P": []},
            "krylov": {"P": []},
            "r_ratios": [],
        }

        for k in k_values:
            print(f"  K={k}...", end=" ", flush=True)

            counts = {"moment": 0, "spectral": 0, "krylov": 0}
            r_samples = []

            for trial in range(trials):
                trial_rng = np.random.RandomState(rng.randint(0, 2**31))
                hams = generate_ensemble("near_integrable", n_qubits, k, trial_rng, g=g)

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
            print(f"S={counts['spectral']/trials:.2f}, K={counts['krylov']/trials:.2f} [{r_str}]")

        results["data"][key] = exp

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_g_sweep(results: Dict, output_dir: Path):
    """Create g-sweep visualization."""
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    d = results["metadata"]["d"]
    g_values = results["metadata"]["g_values"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: P(unreachable) vs ρ for different g
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(g_values)))

    for g, color in zip(g_values, colors):
        key = f"g{g:.2f}"
        if key not in results["data"]:
            continue
        exp = results["data"][key]
        k_values = np.array(exp["k_values"])
        rho = k_values / d**2

        # Spectral
        ax.plot(rho, exp["spectral"]["P"], 'o-', color=color,
               alpha=0.7, label=f'Spectral g={g}')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\rho = K/d^2$')
    ax.set_ylabel('P(unreachable)')
    ax.set_title('Spectral Criterion vs g')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel 2: Krylov (same format)
    ax = axes[1]
    for g, color in zip(g_values, colors):
        key = f"g{g:.2f}"
        if key not in results["data"]:
            continue
        exp = results["data"][key]
        k_values = np.array(exp["k_values"])
        rho = k_values / d**2

        ax.plot(rho, exp["krylov"]["P"], 's-', color=color,
               alpha=0.7, label=f'Krylov g={g}')

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\rho = K/d^2$')
    ax.set_ylabel('P(unreachable)')
    ax.set_title('Krylov Criterion vs g')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel 3: r-ratio and ρ_c vs g
    ax = axes[2]
    r_means = []
    rho_c_spectral = []
    rho_c_krylov = []

    for g in g_values:
        key = f"g{g:.2f}"
        if key not in results["data"]:
            continue
        exp = results["data"][key]
        r_means.append(np.nanmean(exp["r_ratios"]))

        k_values = np.array(exp["k_values"])
        rho = k_values / d**2

        P_s = np.array(exp["spectral"]["P"])
        rho_c_s = fit_rho_c(rho, P_s)
        if rho_c_s is None and np.all(P_s > 0.9):
            rho_c_s = np.nan  # Failed
        rho_c_spectral.append(rho_c_s if rho_c_s else np.nan)

        P_k = np.array(exp["krylov"]["P"])
        rho_c_k = fit_rho_c(rho, P_k)
        rho_c_krylov.append(rho_c_k if rho_c_k else np.nan)

    ax2 = ax.twinx()
    ax.plot(g_values, r_means, 'ko-', markersize=8, label=r'$\langle r \rangle$')
    ax.axhline(0.386, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.531, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Transverse field g')
    ax.set_ylabel(r'Mean r-ratio $\langle r \rangle$')

    ax2.plot(g_values, rho_c_spectral, 'o--', color=COLORS['spectral'],
            markersize=6, label=r'$\rho_c$ Spectral')
    ax2.plot(g_values, rho_c_krylov, 's--', color=COLORS['krylov'],
            markersize=6, label=r'$\rho_c$ Krylov')
    ax2.set_ylabel(r'Critical density $\rho_c$')

    ax.set_title(f'Integrability vs g (d={d})')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    plt.tight_layout()

    output_file = output_dir / "g_sweep_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def plot_extended_comparison(results: Dict, output_dir: Path):
    """
    Create publication-quality extended integrability comparison figure.

    Layout: 1 row x 3 columns (one per model)
    All dimensions shown with consistent colors across subplots.
    Spectral: solid lines, Krylov: dashed lines
    """
    set_style()
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
    })
    output_dir.mkdir(parents=True, exist_ok=True)

    n_qubits_list = results["metadata"]["n_qubits_list"]
    model_names = ["integrable", "near_integrable", "chaotic"]
    model_labels = ["Integrable Ising", "Near-Integrable", "Chaotic Heisenberg"]

    # Model equations for display
    model_equations = [
        r"$H = \sum_i J_i \sigma^z_i\sigma^z_{i+1} + h_i \sigma^z_i$",
        r"$H = J\sum \sigma^z\sigma^z + h\sum \sigma^z + g\sum \sigma^x$",
        r"$H = \sum_i J^\alpha_{ij} \sigma^\alpha_i\sigma^\alpha_j$"
    ]

    # Additional info
    model_info = [
        r"Poisson, $\langle r \rangle \approx 0.39$",
        r"Intermediate chaos",
        r"GOE, $\langle r \rangle \approx 0.53$"
    ]

    # Dimension colors - consistent across all subplots
    dim_colors = {
        8: '#2E86AB',    # Blue
        16: '#E94F37',   # Red
        32: '#F39237',   # Orange
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for col, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
        ax = axes[col]

        # Collect data across all dimensions for this model
        has_data = False
        for n_qubits in n_qubits_list:
            d = 2 ** n_qubits
            key = f"{model_name}_n{n_qubits}"

            if key not in results["data"]:
                continue
            has_data = True

            exp = results["data"][key]
            k_values = np.array(exp["k_values"])
            rho = k_values / d**2
            color = dim_colors.get(d, 'gray')

            # Plot Spectral (solid line)
            P_spectral = np.array(exp["spectral"]["P"])
            ax.plot(rho, P_spectral, 'o-', color=color, markersize=5,
                   linewidth=1.8, label=f'd={d} Spectral')

            # Plot Krylov (dashed line)
            P_krylov = np.array(exp["krylov"]["P"])
            ax.plot(rho, P_krylov, 's--', color=color, markersize=5,
                   linewidth=1.8, alpha=0.8, label=f'd={d} Krylov')

        if not has_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            continue

        # Reference line
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Add model equation in wheat-colored box
        eq_text = model_equations[col]
        ax.text(0.5, 0.97, eq_text, transform=ax.transAxes, fontsize=9,
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9, edgecolor='tan'))

        # Labels
        ax.set_xlabel(r'Control density $\rho = K/d^2$')
        if col == 0:
            ax.set_ylabel('P(unreachable)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        # Title with model name and spectral class
        ax.set_title(f"{model_label}\n{model_info[col]}", fontsize=10, fontweight='bold')

        # Legend - simplified
        if col == 2:  # Only on rightmost panel
            # Create legend entries for line styles and dimensions
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='-', marker='o', markersize=4, label='Spectral'),
                Line2D([0], [0], color='black', linestyle='--', marker='s', markersize=4, label='Krylov'),
                Line2D([0], [0], color='white', label=''),  # spacer
            ]
            for d, color in dim_colors.items():
                legend_elements.append(Line2D([0], [0], color=color, linewidth=3, label=f'd={d}'))
            ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

    fig.suptitle('Criterion Performance Across Integrability Levels',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / "extended_integrability_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_file}")


def print_extended_summary(results: Dict):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("EXTENDED STUDY SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<15} {'d':>4} {'⟨r⟩':>6} {'ρ_c(S)':>10} {'ρ_c(K)':>10}")
    print("-" * 50)

    for key, exp in results["data"].items():
        d = exp["d"]
        r_mean = np.nanmean(exp["r_ratios"])

        k_values = np.array(exp["k_values"])
        rho = k_values / d**2

        rho_c_s = fit_rho_c(rho, np.array(exp["spectral"]["P"]))
        rho_c_k = fit_rho_c(rho, np.array(exp["krylov"]["P"]))

        P_s = np.array(exp["spectral"]["P"])
        if rho_c_s is None and np.all(P_s > 0.9):
            rho_c_s_str = "∞ (fails)"
        elif rho_c_s is not None:
            rho_c_s_str = f"{rho_c_s:.4f}"
        else:
            rho_c_s_str = "N/A"

        rho_c_k_str = f"{rho_c_k:.4f}" if rho_c_k else "N/A"

        print(f"{exp['model']:<15} {d:>4} {r_mean:>6.3f} {rho_c_s_str:>10} {rho_c_k_str:>10}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended integrability study")
    parser.add_argument("--mode", choices=["extended", "gsweep", "both"], default="both",
                       help="Which experiment to run")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"
    fig_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ["extended", "both"]:
        print("\n" + "="*70)
        print("RUNNING EXTENDED STUDY (d=8, 16, 32)")
        print("="*70)

        results_ext = run_extended_study(
            n_qubits_list=[3, 4, 5],  # d=8, 16, 32
            k_max=20,
            trials=args.trials,
            tau=0.99,
            seed=args.seed
        )

        with open(output_dir / f"extended_integrability_{timestamp}.pkl", 'wb') as f:
            pickle.dump(results_ext, f)
        print(f"\n✓ Saved results to extended_integrability_{timestamp}.pkl")

        plot_extended_comparison(results_ext, fig_dir)
        print_extended_summary(results_ext)

    if args.mode in ["gsweep", "both"]:
        print("\n" + "="*70)
        print("RUNNING G-SWEEP")
        print("="*70)

        results_g = run_g_sweep(
            n_qubits=4,
            g_values=[0.1, 0.3, 0.5, 0.7, 1.0],
            k_max=16,
            trials=args.trials,
            tau=0.99,
            seed=args.seed + 1000  # Different seed
        )

        with open(output_dir / f"g_sweep_{timestamp}.pkl", 'wb') as f:
            pickle.dump(results_g, f)
        print(f"\n✓ Saved results to g_sweep_{timestamp}.pkl")

        # NOTE: g_sweep_analysis.png removed from publication - commented out
        # plot_g_sweep(results_g, fig_dir)

    print("\n" + "="*70)
    print("EXTENDED INTEGRABILITY STUDY COMPLETE")
    print("="*70)
