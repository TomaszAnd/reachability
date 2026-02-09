#!/usr/bin/env python3
"""
[[5,1,3]] Perfect Code: Floquet Driving Discovery

Unlike the toric plaquette (where we know the optimal answer), for the [[5,1,3]] code
we use commutator structure analysis to DISCOVER the optimal driving structure.

The [[5,1,3]] code:
- 5 physical qubits, 1 logical qubit, distance 3
- 4 stabilizer generators (all weight-4)
- Encodes 1 logical qubit

Stabilizers (cyclic form):
  S1 = XZZXI
  S2 = IXZZX
  S3 = XIXZZ
  S4 = ZXIXZ

Methodology:
1. Build Hamiltonian operators from available 2-local terms (XY couplings on a ring)
2. Compute commutator structure ||[H_j, H_k]||_F for all pairs
3. Identify which pairs have large commutator norms
4. Use graph coloring to assign frequencies (connected nodes get different frequencies)
5. Validate via Fourier overlap analysis

Usage:
    python scripts/five_qubit_code_discovery.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed, using simple graph coloring")

from reach import floquet


# =============================================================================
# PAULI MATRICES
# =============================================================================

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor(*ops):
    """Compute tensor product of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# =============================================================================
# [[5,1,3]] CODE SYSTEM
# =============================================================================

def create_5qubit_xy_operators():
    """
    Create XY coupling operators for 5-qubit ring geometry.

    Ring geometry:
        1 --- 2
       /       \
      5         3
       \       /
        4 ----

    Available 2-local operators (nearest-neighbor):
    - H_12: X1X2 + Y1Y2
    - H_23: X2X3 + Y2Y3
    - H_34: X3X4 + Y3Y4
    - H_45: X4X5 + Y4Y5
    - H_51: X5X1 + Y5Y1

    Plus single-qubit fields:
    - H_k: X_k for k=1,...,5

    Returns:
        dict: Operator name -> numpy array
    """
    # Helper to create XY coupling between qubits i and j (1-indexed)
    def xy_coupling(i, j, n=5):
        """Create (X_i X_j + Y_i Y_j) on n qubits."""
        ops_xx = [I] * n
        ops_yy = [I] * n
        ops_xx[i-1] = X
        ops_xx[j-1] = X
        ops_yy[i-1] = Y
        ops_yy[j-1] = Y
        return tensor(*ops_xx) + tensor(*ops_yy)

    # Helper for single-qubit X field
    def x_field(i, n=5):
        """Create X_i on n qubits."""
        ops = [I] * n
        ops[i-1] = X
        return tensor(*ops)

    operators = {}

    # XY couplings (ring)
    operators['H_12'] = xy_coupling(1, 2)
    operators['H_23'] = xy_coupling(2, 3)
    operators['H_34'] = xy_coupling(3, 4)
    operators['H_45'] = xy_coupling(4, 5)
    operators['H_51'] = xy_coupling(5, 1)

    # Single-qubit X fields
    operators['H_1'] = x_field(1)
    operators['H_2'] = x_field(2)
    operators['H_3'] = x_field(3)
    operators['H_4'] = x_field(4)
    operators['H_5'] = x_field(5)

    return operators


def create_5qubit_logical_zero():
    """
    Create |0_L> of the [[5,1,3]] code.

    Returns:
        np.ndarray: Normalized state vector of dimension 32
    """
    d = 32  # 2^5
    psi = np.zeros(d, dtype=complex)

    # Weight 0 (1 state)
    psi[0b00000] = +1

    # Weight 2 (4 states)
    psi[0b10010] = +1
    psi[0b01001] = +1
    psi[0b10100] = +1
    psi[0b01010] = +1

    # Weight 3 (10 states) - all negative
    psi[0b11011] = -1
    psi[0b00110] = -1
    psi[0b11000] = -1
    psi[0b11101] = -1
    psi[0b00011] = -1
    psi[0b11110] = -1
    psi[0b01111] = -1
    psi[0b10001] = -1
    psi[0b01100] = -1
    psi[0b10111] = -1

    # Weight 4 (1 state)
    psi[0b00101] = +1

    # Normalize
    psi /= np.linalg.norm(psi)

    return psi


def create_5qubit_initial_state():
    """Create initial product state |00000>."""
    d = 32
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    return psi


# =============================================================================
# COMMUTATOR ANALYSIS
# =============================================================================

def analyze_commutator_structure(operators, operator_names):
    """
    Compute ||[H_j, H_k]||_F for all pairs.

    Args:
        operators: List of Hamiltonian matrices
        operator_names: List of operator names

    Returns:
        comm_matrix: K×K matrix of commutator norms
        large_pairs: List of (name_j, name_k, norm) for pairs above threshold
    """
    K = len(operators)
    comm_matrix = np.zeros((K, K))

    for j in range(K):
        for k in range(j+1, K):
            comm = operators[j] @ operators[k] - operators[k] @ operators[j]
            norm = np.linalg.norm(comm, 'fro')
            comm_matrix[j, k] = norm
            comm_matrix[k, j] = norm

    # Identify large commutator pairs (threshold: 10% of max)
    max_norm = np.max(comm_matrix)
    threshold = 0.1 * max_norm if max_norm > 0 else 0

    large_pairs = []
    for j in range(K):
        for k in range(j+1, K):
            if comm_matrix[j, k] > threshold:
                large_pairs.append((operator_names[j], operator_names[k], comm_matrix[j, k]))

    # Sort by norm (descending)
    large_pairs.sort(key=lambda x: -x[2])

    return comm_matrix, large_pairs


# =============================================================================
# FREQUENCY ASSIGNMENT
# =============================================================================

def propose_frequency_groupings(large_pairs, operator_names):
    """
    Use graph coloring to assign frequencies.

    Build conflict graph:
    - Nodes = operators
    - Edges = large commutator pairs

    Color the graph:
    - Each color = one frequency group
    - Connected nodes must have different colors

    Returns:
        groups: Dict mapping frequency label to list of operators
        coloring: Dict mapping operator name to color (int)
    """
    if HAS_NETWORKX:
        G = nx.Graph()
        G.add_nodes_from(operator_names)

        for name_j, name_k, _ in large_pairs:
            G.add_edge(name_j, name_k)

        # Greedy coloring
        coloring = nx.greedy_color(G, strategy='largest_first')
    else:
        # Simple greedy coloring without networkx
        coloring = {}
        for name in operator_names:
            # Find colors of neighbors
            neighbor_colors = set()
            for name_j, name_k, _ in large_pairs:
                if name_j == name and name_k in coloring:
                    neighbor_colors.add(coloring[name_k])
                elif name_k == name and name_j in coloring:
                    neighbor_colors.add(coloring[name_j])

            # Assign smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            coloring[name] = color

    # Convert to frequency groups
    n_colors = max(coloring.values()) + 1 if coloring else 1
    groups = {f'freq_{i+1}': [] for i in range(n_colors)}

    for op, color in coloring.items():
        groups[f'freq_{color+1}'].append(op)

    return groups, coloring


def create_driving_functions_from_coloring(operator_names, coloring, omega=1.0):
    """
    Create driving functions based on frequency coloring.

    Operators in color k get frequency (k+1)*omega.
    """
    driving_funcs = []
    for name in operator_names:
        freq_mult = coloring[name] + 1  # freq = (color+1)*ω
        driving_funcs.append(lambda t, m=freq_mult, w=omega: 1.0 + np.cos(m * w * t))

    return driving_funcs


def validate_frequency_assignment(operators, operator_names, coloring, large_pairs, omega=1.0):
    """
    Verify that the proposed frequency assignment minimizes F_jk for large commutator pairs.

    Returns:
        F_matrix: Full Fourier overlap matrix
        validation_results: List of (pair, F_jk, same_group, status)
    """
    K = len(operators)
    period = 2 * np.pi / omega

    # Create driving functions
    driving_funcs = create_driving_functions_from_coloring(operator_names, coloring, omega)

    # Compute F_jk matrix
    F_matrix = np.zeros((K, K))
    for j in range(K):
        for k in range(j+1, K):
            F_jk = floquet.compute_fourier_overlap(driving_funcs[j], driving_funcs[k], period)
            F_matrix[j, k] = F_jk
            F_matrix[k, j] = -F_jk

    # Validate large commutator pairs
    validation_results = []
    for name_j, name_k, comm_norm in large_pairs:
        j = operator_names.index(name_j)
        k = operator_names.index(name_k)

        F_jk = abs(F_matrix[j, k])
        same_group = (coloring[name_j] == coloring[name_k])

        if same_group:
            status = "WARNING" if F_jk > 0.01 else "OK (same freq, low F_jk)"
        else:
            status = "OK" if F_jk < 0.01 else "ISSUE"

        validation_results.append({
            'pair': (name_j, name_k),
            'comm_norm': comm_norm,
            'F_jk': F_jk,
            'same_group': same_group,
            'freq_j': coloring[name_j] + 1,
            'freq_k': coloring[name_k] + 1,
            'status': status
        })

    return F_matrix, validation_results


# =============================================================================
# MAIN DISCOVERY
# =============================================================================

def run_discovery_experiment(verbose=True):
    """
    Run the full [[5,1,3]] code driving discovery experiment.
    """
    if verbose:
        print("="*70)
        print("[[5,1,3]] PERFECT CODE: FLOQUET DRIVING DISCOVERY")
        print("="*70)
        print()

    # Setup system
    operators_dict = create_5qubit_xy_operators()

    # Focus on XY coupling operators only (exclude single-qubit fields for commutator analysis)
    xy_names = ['H_12', 'H_23', 'H_34', 'H_45', 'H_51']
    xy_operators = [operators_dict[name] for name in xy_names]

    # Also include single-qubit fields for completeness
    all_names = xy_names + ['H_1', 'H_2', 'H_3', 'H_4', 'H_5']
    all_operators = [operators_dict[name] for name in all_names]

    psi = create_5qubit_initial_state()
    phi = create_5qubit_logical_zero()

    if verbose:
        print("System setup:")
        print(f"  Initial state: |00000>")
        print(f"  Target: |0_L> of [[5,1,3]] code")
        print(f"  XY operators: {xy_names}")
        print(f"  Single-qubit fields: ['H_1', ..., 'H_5']")
        print(f"  Dimension: {len(psi)}")
        print()

    # Step 1: Analyze commutator structure (XY operators only)
    if verbose:
        print("="*70)
        print("STEP 1: COMMUTATOR STRUCTURE ANALYSIS")
        print("="*70)
        print()

    comm_matrix, large_pairs = analyze_commutator_structure(xy_operators, xy_names)

    if verbose:
        print("Commutator norm matrix ||[H_j, H_k]||_F:")
        print()
        print(f"{'':8}", end='')
        for name in xy_names:
            print(f"{name:>8}", end='')
        print()

        for j, name_j in enumerate(xy_names):
            print(f"{name_j:8}", end='')
            for k in range(len(xy_names)):
                if j == k:
                    print(f"{'--':>8}", end='')
                else:
                    print(f"{comm_matrix[j,k]:>8.2f}", end='')
            print()
        print()

        print(f"Large commutator pairs (threshold > 10% of max):")
        for name_j, name_k, norm in large_pairs:
            print(f"  {name_j} - {name_k}: ||[H_j, H_k]|| = {norm:.2f}")
        print()

    # Step 2: Propose frequency groupings
    if verbose:
        print("="*70)
        print("STEP 2: FREQUENCY ASSIGNMENT (Graph Coloring)")
        print("="*70)
        print()

    groups, coloring = propose_frequency_groupings(large_pairs, xy_names)

    if verbose:
        print("Proposed frequency groups:")
        for freq_label, ops in sorted(groups.items()):
            if ops:
                freq_num = int(freq_label.split('_')[1])
                print(f"  {freq_label} ({freq_num}ω): {ops}")
        print()

        print("Operator -> Frequency mapping:")
        for name in xy_names:
            freq = coloring[name] + 1
            print(f"  {name}: {freq}ω")
        print()

    # Step 3: Validate with Fourier overlap
    if verbose:
        print("="*70)
        print("STEP 3: VALIDATION (Fourier Overlap Analysis)")
        print("="*70)
        print()

    F_matrix, validation_results = validate_frequency_assignment(
        xy_operators, xy_names, coloring, large_pairs, omega=1.0
    )

    if verbose:
        print("Validation of large commutator pairs:")
        print()
        for result in validation_results:
            name_j, name_k = result['pair']
            print(f"  {name_j} - {name_k}:")
            print(f"    ||[H_j,H_k]|| = {result['comm_norm']:.2f}")
            print(f"    Frequencies: {result['freq_j']}ω vs {result['freq_k']}ω")
            print(f"    |F_jk| = {result['F_jk']:.6f}")
            print(f"    Status: {result['status']}")
            print()

    # Summary
    n_issues = sum(1 for r in validation_results if 'ISSUE' in r['status'] or 'WARNING' in r['status'])

    if verbose:
        print("="*70)
        print("DISCOVERY SUMMARY")
        print("="*70)
        print()

        n_freq_groups = len([g for g in groups.values() if g])
        print(f"Number of frequency groups needed: {n_freq_groups}")
        print()

        print("Discovered frequency assignment:")
        for freq_label, ops in sorted(groups.items()):
            if ops:
                print(f"  {freq_label}: {', '.join(ops)}")
        print()

        if n_issues == 0:
            print("VALIDATION: All large commutator pairs have |F_jk| ≈ 0")
            print("The discovered frequency assignment minimizes interference!")
        else:
            print(f"VALIDATION: {n_issues} pairs need attention")
        print()

    return {
        'operators': operators_dict,
        'xy_names': xy_names,
        'all_names': all_names,
        'comm_matrix': comm_matrix,
        'large_pairs': large_pairs,
        'groups': groups,
        'coloring': coloring,
        'F_matrix': F_matrix,
        'validation_results': validation_results
    }


def generate_discovery_plots(data, outdir='fig/floquet'):
    """Generate discovery visualization."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xy_names = data['xy_names']
    comm_matrix = data['comm_matrix']
    coloring = data['coloring']
    groups = data['groups']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # =========================================================================
    # Panel A: Commutator Norms Heatmap
    # =========================================================================
    ax_comm = axes[0]
    K = len(xy_names)

    im = ax_comm.imshow(comm_matrix, cmap='Blues', aspect='auto')
    ax_comm.set_xticks(range(K))
    ax_comm.set_yticks(range(K))
    ax_comm.set_xticklabels(xy_names, fontsize=10, rotation=45, ha='right')
    ax_comm.set_yticklabels(xy_names, fontsize=10)
    ax_comm.set_title('(A) Commutator Norms\n||[H_j, H_k]||_F', fontsize=12, fontweight='bold')

    # Add value annotations
    for j in range(K):
        for k in range(K):
            color = 'white' if comm_matrix[j, k] > 6 else 'black'
            ax_comm.text(k, j, f'{comm_matrix[j,k]:.1f}', ha='center', va='center',
                        fontsize=10, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_comm)
    cbar.set_label('Frobenius Norm', fontsize=10)

    # =========================================================================
    # Panel B: Ring Geometry with Frequency Colors
    # =========================================================================
    ax_ring = axes[1]

    # Ring positions (5 qubits)
    angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2  # Start from top
    ring_x = np.cos(angles)
    ring_y = np.sin(angles)

    # Frequency colors
    freq_colors = ['royalblue', 'darkorange', 'forestgreen', 'crimson', 'purple']

    # Draw edges (XY couplings)
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    edge_names = ['H_12', 'H_23', 'H_34', 'H_45', 'H_51']

    for (i, j), name in zip(edge_pairs, edge_names):
        color_idx = coloring.get(name, 0)
        mid_x = (ring_x[i] + ring_x[j]) / 2
        mid_y = (ring_y[i] + ring_y[j]) / 2

        ax_ring.plot([ring_x[i], ring_x[j]], [ring_y[i], ring_y[j]],
                    color=freq_colors[color_idx], linewidth=4, alpha=0.7, zorder=1)

        # Label the edge
        ax_ring.text(mid_x * 1.3, mid_y * 1.3, name, ha='center', va='center',
                    fontsize=9, color=freq_colors[color_idx], fontweight='bold')

    # Draw qubit nodes
    for i in range(5):
        ax_ring.scatter([ring_x[i]], [ring_y[i]], s=400, c='lightgray',
                       edgecolors='black', linewidth=2, zorder=2)
        ax_ring.text(ring_x[i], ring_y[i], str(i+1), ha='center', va='center',
                    fontsize=12, fontweight='bold')

    ax_ring.set_xlim(-1.8, 1.8)
    ax_ring.set_ylim(-1.8, 1.8)
    ax_ring.set_aspect('equal')
    ax_ring.axis('off')
    ax_ring.set_title('(B) Ring Geometry with\nDiscovered Frequencies', fontsize=12, fontweight='bold')

    # Legend
    for i, (freq_label, ops) in enumerate(sorted(groups.items())):
        if ops:
            freq_num = int(freq_label.split('_')[1])
            ax_ring.plot([], [], color=freq_colors[i], linewidth=4, label=f'{freq_num}ω: {", ".join(ops)}')
    ax_ring.legend(loc='lower center', fontsize=8, ncol=1)

    # =========================================================================
    # Panel C: Frequency Assignment Summary
    # =========================================================================
    ax_summary = axes[2]
    ax_summary.axis('off')

    # Create text summary
    summary_text = "Discovered Frequency Assignment\n"
    summary_text += "=" * 30 + "\n\n"

    for freq_label, ops in sorted(groups.items()):
        if ops:
            freq_num = int(freq_label.split('_')[1])
            summary_text += f"Frequency {freq_num}ω:\n"
            for op in ops:
                summary_text += f"  • {op}\n"
            summary_text += "\n"

    summary_text += "Design Principle:\n"
    summary_text += "Operators with large ||[H_j,H_k]||\n"
    summary_text += "should be in DIFFERENT frequency groups\n"
    summary_text += "to minimize interference in H_F^(2)."

    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, family='monospace', verticalalignment='top')

    ax_summary.set_title('(C) Summary', fontsize=12, fontweight='bold')

    plt.suptitle('[[5,1,3]] Perfect Code: Floquet Driving Discovery', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = outdir / 'five_qubit_discovery.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved discovery figure: {fig_path}")
    return fig_path


def generate_discovery_report(data, outdir='results'):
    """Generate discovery report in markdown."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# [[5,1,3]] Perfect Code: Floquet Driving Discovery Report\n")
    report.append(f"Generated: {datetime.now().isoformat()}\n")
    report.append("")

    report.append("## Overview\n")
    report.append("This report documents the discovery of optimal Floquet driving frequencies")
    report.append("for preparing the logical |0_L> state of the [[5,1,3]] perfect code.\n")
    report.append("")

    report.append("## System Configuration\n")
    report.append("**Ring geometry (5 qubits):**")
    report.append("```")
    report.append("    1 --- 2")
    report.append("   /       \\")
    report.append("  5         3")
    report.append("   \\       /")
    report.append("    4 ----")
    report.append("```\n")
    report.append("")

    report.append("**XY Coupling Operators:**")
    for name in data['xy_names']:
        report.append(f"- {name}")
    report.append("")

    report.append("## Commutator Analysis\n")
    report.append("**Large commutator pairs (||[H_j, H_k]||_F > 10% of max):**\n")
    report.append("| Pair | ||[H_j, H_k]||_F |")
    report.append("|------|-----------------|")
    for name_j, name_k, norm in data['large_pairs']:
        report.append(f"| {name_j} - {name_k} | {norm:.2f} |")
    report.append("")

    report.append("## Discovered Frequency Assignment\n")
    report.append("Using graph coloring on the conflict graph (edges = large commutator pairs):\n")
    for freq_label, ops in sorted(data['groups'].items()):
        if ops:
            freq_num = int(freq_label.split('_')[1])
            report.append(f"**Frequency {freq_num}ω:** {', '.join(ops)}")
    report.append("")

    report.append("## Validation\n")
    report.append("Fourier overlap |F_jk| for large commutator pairs:\n")
    report.append("| Pair | ||[H_j,H_k]|| | Freq j | Freq k | |F_jk| | Status |")
    report.append("|------|-------------|--------|--------|-------|--------|")
    for r in data['validation_results']:
        name_j, name_k = r['pair']
        report.append(f"| {name_j}-{name_k} | {r['comm_norm']:.2f} | {r['freq_j']}ω | {r['freq_k']}ω | {r['F_jk']:.4f} | {r['status']} |")
    report.append("")

    report.append("## Comparison with Toric Plaquette\n")
    report.append("| Property | Toric (4 qubits) | [[5,1,3]] (5 qubits) |")
    report.append("|----------|------------------|----------------------|")
    report.append(f"| XY Operators | 3 | {len(data['xy_names'])} |")
    report.append(f"| Large commutator pairs | 2 | {len(data['large_pairs'])} |")
    n_freq = len([g for g in data['groups'].values() if g])
    report.append(f"| Frequency groups needed | 2 | {n_freq} |")
    report.append("")

    report.append("## Conclusion\n")
    n_issues = sum(1 for r in data['validation_results'] if 'ISSUE' in r['status'])
    if n_issues == 0:
        report.append("The discovered frequency assignment successfully minimizes Fourier overlap")
        report.append("for all large commutator pairs, validating the graph coloring approach.\n")
    else:
        report.append(f"The assignment has {n_issues} pairs that may need further optimization.\n")

    report_text = "\n".join(report)

    report_path = outdir / '5QUBIT_DRIVING_DISCOVERY.md'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"Saved discovery report: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Run discovery
    data = run_discovery_experiment(verbose=True)

    # Generate plots
    print()
    generate_discovery_plots(data, outdir='fig/floquet')

    # Generate report
    generate_discovery_report(data, outdir='results')

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = Path('results') / f'5qubit_driving_discovery_{timestamp}.pkl'

    # Convert numpy arrays for pickling
    save_data = {
        'xy_names': data['xy_names'],
        'all_names': data['all_names'],
        'comm_matrix': data['comm_matrix'],
        'large_pairs': data['large_pairs'],
        'groups': data['groups'],
        'coloring': data['coloring'],
        'F_matrix': data['F_matrix'],
        'validation_results': data['validation_results']
    }

    with open(data_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nSaved data: {data_path}")

    print()
    print("="*70)
    print("DISCOVERY COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
