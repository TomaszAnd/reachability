#!/usr/bin/env python3
"""
Generate publication-ready integrability comparison figure.

Uses existing data from three_models_study to create a figure showing:
- Three columns: Integrable Ising, Near-Integrable, Chaotic Heisenberg
- All available dimensions with consistent colors
- Model equations in wheat-colored boxes
- Different line styles: Spectral (solid), Krylov (dashed)

Author: Claude Code
Date: 2026-01-15
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def load_data():
    """Load integrability study data."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw_logs"

    # Try to load three_models data first (has all three models)
    three_models_files = sorted(data_dir.glob("three_models_study_*.pkl"))
    if three_models_files:
        with open(three_models_files[-1], 'rb') as f:
            return pickle.load(f)

    # Fallback to extended_integrability
    ext_files = sorted(data_dir.glob("extended_integrability_*.pkl"))
    if ext_files:
        with open(ext_files[-1], 'rb') as f:
            return pickle.load(f)

    return None


def generate_figure():
    """Generate the publication integrability comparison figure."""
    set_publication_style()

    data = load_data()
    if data is None:
        print("ERROR: No integrability data found!")
        return

    n_qubits_list = data["metadata"]["n_qubits_list"]
    print(f"Loaded data with n_qubits: {n_qubits_list} -> d = {[2**n for n in n_qubits_list]}")
    print(f"Available models: {list(data['data'].keys())}")

    # Model configuration
    model_names = ["integrable", "near_integrable", "chaotic"]
    model_labels = ["Integrable Ising", "Near-Integrable", "Chaotic Heisenberg"]

    # Model equations - compact versions for figure
    model_equations = [
        r"$H = \sum_i J_i \sigma^z_i\sigma^z_{i+1} + h_i \sigma^z_i$",
        r"$H = J\sum \sigma^z\sigma^z + h\sum \sigma^z + g\sum \sigma^x$",
        r"$H = \sum_{ij} (J^x\sigma^x\sigma^x + J^y\sigma^y\sigma^y + J^z\sigma^z\sigma^z)$"
    ]

    # Model spectral class
    model_info = [
        r"Poisson, $\langle r \rangle \approx 0.39$",
        r"Intermediate",
        r"GOE, $\langle r \rangle \approx 0.53$"
    ]

    # Dimension colors - colorblind friendly
    dim_colors = {
        8: '#2E86AB',    # Blue
        16: '#E94F37',   # Red
        32: '#F39237',   # Orange
        64: '#1B998B',   # Teal
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for col, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
        ax = axes[col]

        # Collect data across all dimensions for this model
        has_data = False
        r_ratios_all = []

        for n_qubits in n_qubits_list:
            d = 2 ** n_qubits
            key = f"{model_name}_n{n_qubits}"

            if key not in data["data"]:
                continue
            has_data = True

            exp = data["data"][key]
            k_values = np.array(exp["k_values"])
            rho = k_values / d**2
            color = dim_colors.get(d, 'gray')

            # Collect r-ratios for averaging
            if "r_ratios" in exp:
                r_ratios_all.extend([r for r in exp["r_ratios"] if not np.isnan(r)])

            # Plot Spectral (solid line, circle markers)
            P_spectral = np.array(exp["spectral"]["P"])
            ax.plot(rho, P_spectral, 'o-', color=color, markersize=5,
                   linewidth=1.8, label=f'd={d} Spectral')

            # Plot Krylov (dashed line, square markers)
            P_krylov = np.array(exp["krylov"]["P"])
            ax.plot(rho, P_krylov, 's--', color=color, markersize=5,
                   linewidth=1.8, alpha=0.85, label=f'd={d} Krylov')

        if not has_data:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            continue

        # Reference line at P=0.5
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Add model equation in wheat-colored box
        eq_text = model_equations[col]
        ax.text(0.5, 0.97, eq_text, transform=ax.transAxes, fontsize=8,
               ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                        alpha=0.9, edgecolor='tan'))

        # Calculate mean r-ratio for this model
        r_mean = np.mean(r_ratios_all) if r_ratios_all else np.nan
        r_class = "Poisson" if r_mean < 0.45 else ("GOE" if r_mean < 0.56 else "GUE")

        # Labels
        ax.set_xlabel(r'Control density $\rho = K/d^2$')
        if col == 0:
            ax.set_ylabel('P(unreachable)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        # Title with model name and measured spectral class
        if not np.isnan(r_mean):
            title = f"{model_label}\n" + r"$\langle r \rangle$" + f" = {r_mean:.2f} ({r_class})"
        else:
            title = f"{model_label}\n{model_info[col]}"
        ax.set_title(title, fontsize=10, fontweight='bold')

        # Legend on rightmost panel
        if col == 2:
            # Create legend entries for line styles and dimensions
            legend_elements = [
                Line2D([0], [0], color='black', linestyle='-', marker='o',
                      markersize=4, linewidth=1.5, label='Spectral'),
                Line2D([0], [0], color='black', linestyle='--', marker='s',
                      markersize=4, linewidth=1.5, label='Krylov'),
                Line2D([0], [0], color='white', linewidth=0, label=''),  # spacer
            ]
            # Add dimension colors
            for d in sorted(dim_colors.keys()):
                if any(f"_n{int(np.log2(d))}" in k for k in data["data"].keys()):
                    legend_elements.append(
                        Line2D([0], [0], color=dim_colors[d], linewidth=3, label=f'd={d}')
                    )
            ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
                     framealpha=0.95, frameon=True)

    fig.suptitle('Criterion Performance Across Integrability Levels',
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent.parent / "fig" / "integrability"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "extended_integrability_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_file}")


if __name__ == "__main__":
    generate_figure()
