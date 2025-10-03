"""
Visualization functions for quantum reachability analysis (render only).

Pipeline Role:
This module provides PURE RENDERING functions that consume numerical data from
analysis.py and produce publication-quality figures. Strict separation of
concerns: NO computation allowed here.

Mathematical Context (for plot labels/annotations):
- **Eigendecomposition**: H(λ) = U(λ) diag(E₁,...,Eₐ) U†(λ)
- **Spectral overlap**: S(λ) = Σₙ |φₙ*(λ) ψₙ(λ)| ∈ [0,1]
- **Maximized overlap**: S* = max_{λ∈[-1,1]ᴷ} S(λ)
- **Unreachability probability**: P_unreach(d,K;τ) = Pr[S* < τ]
- **Binomial SEM**: SEM(p) = √(p(1-p)/N) for error bars

Plot Types & Exact Filenames:
1. **Rank comparison**: unreachability_vs_rank_old_vs_new_{ensemble}_tau{τ}.png
   - Compares old (τ-free) vs new (τ-based) criterion
   - Annotates τ value and criterion difference

2. **Threshold histograms**: tau_hist_{ensemble}.png
   - Grouped bars: P(unreachability) vs τ for multiple dimensions
   - Shows sensitivity to threshold choice

3. **Optimizer comparison**: optimizer_overlap_hist_{ensemble}.png
   - Mean ± SEM of S* across different optimization methods
   - Helps validate optimizer choice

4. **Iteration sweep**: iter_sweep_prob_{ensemble}.png
   - Dual panel: P(unreachability) vs iterations + runtime scaling
   - Convergence analysis for iteration count tuning

5. **Landscape 2D**: landscape_S2D_{ensemble}_d{d}_k{k}.png
   - Heatmap of S(λ₁, λ₂) over parameter grid

6. **Landscape 3D**: landscape_S3D_{ensemble}_d{d}_k{k}.png
   - Surface plot of S(λ₁, λ₂)

All plots use settings.DEFAULT_DPI, consistent colormaps, and annotate key
parameters (d, k, τ, grid size) for traceability.

Floor Handling:
Values ≤ settings.DISPLAY_FLOOR shown as isolated markers without connecting
lines to avoid vertical artifacts in log plots (see apply_masked_connections).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import settings

logger = logging.getLogger(__name__)

# Configure matplotlib for consistent output
plt.rcParams["figure.dpi"] = settings.DEFAULT_DPI
plt.rcParams["savefig.dpi"] = settings.DEFAULT_DPI
plt.rcParams["savefig.bbox"] = "tight"


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def apply_masked_connections(
    ax, x: np.ndarray, y: np.ndarray, floor: float = settings.DISPLAY_FLOOR, **kwargs
) -> None:
    """
    Plot data with masked connections to avoid vertical artifacts.

    Points at or below floor are shown as isolated markers without connecting
    lines to prevent vertical drops in log plots.

    Args:
        ax: Matplotlib axes object
        x: X-axis data (must be sorted)
        y: Y-axis data
        floor: Display floor for masking
        **kwargs: Line/marker styling arguments
    """
    # Apply floor for display
    y_display = np.maximum(y, floor)
    y_log = np.log(y_display)

    # Identify floored points
    is_floored = y <= floor

    # Plot contiguous segments of non-floored points
    i = 0
    while i < len(x):
        if is_floored[i]:
            # Plot single floored point as marker only
            ax.plot(
                x[i],
                y_log[i],
                marker=kwargs.get("marker", "o"),
                color=kwargs.get("color"),
                markersize=kwargs.get("markersize", 6),
                linestyle="none",
                alpha=0.5,
            )
            i += 1
        else:
            # Find end of non-floored segment
            j = i
            while j < len(x) and not is_floored[j]:
                j += 1

            # Plot segment with line
            ax.plot(x[i:j], y_log[i:j], **kwargs)
            i = j


def plot_landscape_S2D(
    L1: np.ndarray,
    L2: np.ndarray,
    S: np.ndarray,
    d: int,
    k: int,
    ensemble: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot spectral overlap landscape S(λ₁,λ₂) as 2D heatmap.

    Creates a 2D heatmap showing the spectral overlap function over parameter
    space (λ₁, λ₂) with other parameters fixed at zero.

    Args:
        L1: Meshgrid for λ₁ values
        L2: Meshgrid for λ₂ values
        S: Spectral overlap values S(λ₁,λ₂) ∈ [0,1]
        d: Hilbert space dimension
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        output_dir: Directory for saving figure (default: settings.FIG_SUMMARY_DIR)

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    # Ensure S is in valid range [0,1]
    S_clipped = np.clip(S, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    # Create heatmap
    im = ax.pcolormesh(L1, L2, S_clipped, cmap=settings.DEFAULT_COLORMAP, shading="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("S(λ₁,λ₂)", fontsize=12)

    # Labels and title
    ax.set_xlabel("λ₁", fontsize=12)
    ax.set_ylabel("λ₂", fontsize=12)
    ax.set_title(f"S(λ₁,λ₂) — {ensemble}, d={d}, k={k} (2D)", fontsize=14)

    # Annotate min/max values and grid info
    s_min, s_max = np.min(S_clipped), np.max(S_clipped)
    grid_size = L1.shape[0]
    ax.text(
        0.02,
        0.98,
        f"S ∈ [{s_min:.3f}, {s_max:.3f}]\nGrid: {grid_size}×{grid_size}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    # Grid for better readability
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with 2D-specific filename
    filename = f"landscape_S2D_{ensemble}_d{d}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 2D landscape: {filepath}")
    return filepath


# Backward compatibility alias for CLI
def plot_landscape_S(
    L1: np.ndarray,
    L2: np.ndarray,
    S: np.ndarray,
    d: int,
    k: int,
    ensemble: str,
    output_dir: Optional[str] = None,
) -> str:
    """Alias for plot_landscape_S2D (backward compatibility)."""
    return plot_landscape_S2D(L1, L2, S, d, k, ensemble, output_dir)


def plot_landscape_S3D(
    L1: np.ndarray,
    L2: np.ndarray,
    S: np.ndarray,
    d: int,
    k: int,
    ensemble: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot spectral overlap landscape S(λ₁,λ₂) as 3D surface.

    Creates a 3D surface plot showing the spectral overlap function over
    parameter space (λ₁, λ₂) with mathematical background:
    - Eigendecomposition: U(λ)†H(λ)U(λ) = diag(E₁,...,E_d)
    - Spectral overlap: S(λ) = Σₙ |φₙ(λ)*ψₙ(λ)|

    Args:
        L1: Meshgrid for λ₁ values
        L2: Meshgrid for λ₂ values
        S: Spectral overlap values S(λ₁,λ₂) ∈ [0,1]
        d: Hilbert space dimension
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        output_dir: Directory for saving figure (default: settings.FIG_SUMMARY_DIR)

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    # Ensure S is in valid range [0,1]
    S_clipped = np.clip(S, 0.0, 1.0)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Create 3D surface
    surf = ax.plot_surface(
        L1, L2, S_clipped, cmap=settings.DEFAULT_COLORMAP, alpha=0.9, linewidth=0, antialiased=True
    )

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15)
    cbar.set_label(r"$S(\lambda_1,\lambda_2)$", fontsize=12)

    # Labels and title with LaTeX
    ax.set_xlabel(r"$\lambda_1$", fontsize=12)
    ax.set_ylabel(r"$\lambda_2$", fontsize=12)
    ax.set_zlabel(r"$S(\lambda_1,\lambda_2)$", fontsize=12)
    ax.set_title(rf"$S(\lambda_1,\lambda_2)$ — {ensemble}, d={d}, k={k} (3D)", fontsize=14)

    # Set view angle for better visualization
    ax.view_init(elev=30, azim=45)
    ax.set_zlim(0, 1)

    # Annotate min/max values and grid info
    s_min, s_max = np.min(S_clipped), np.max(S_clipped)
    grid_size = L1.shape[0]
    ax.text2D(
        0.02,
        0.98,
        f"S ∈ [{s_min:.3f}, {s_max:.3f}]\nGrid: {grid_size}×{grid_size}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()

    # Save with exact filename (3D variant)
    filename = f"landscape_S3D_{ensemble}_d{d}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 3D landscape: {filepath}")
    return filepath


# Backward compatibility alias for CLI
def plot_landscape_S_3d(
    L1: np.ndarray,
    L2: np.ndarray,
    S: np.ndarray,
    d: int,
    k: int,
    ensemble: str,
    output_dir: Optional[str] = None,
) -> str:
    """Alias for plot_landscape_S3D (backward compatibility)."""
    return plot_landscape_S3D(L1, L2, S, d, k, ensemble, output_dir)


def plot_tau_histograms(
    data: Dict[int, Dict[str, np.ndarray]], ensemble: str, output_dir: Optional[str] = None
) -> List[str]:
    """
    Plot unreachability probability vs τ threshold as grouped histograms.

    Creates a single grouped bar plot showing P_unreach(τ) for different dimensions.
    Only generates one file per ensemble: tau_hist_{ensemble}.png

    Mathematical background:
    - Eigendecomposition: U(λ)†H(λ)U(λ) = diag(E₁,...,E_d)
    - Spectral overlap: S(λ) = Σₙ |φₙ(λ)*ψₙ(λ)|

    Args:
        data: Dictionary {d: {'tau': taus, 'p': probs, 'err': errors}}
        ensemble: "GOE" or "GUE"
        output_dir: Directory for saving figures

    Returns:
        List containing single path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    if not data:
        logger.warning("No data provided for tau histograms")
        return []

    dims = sorted(data.keys())

    # Create single aggregated figure with grouped bars
    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    # Get tau values (should be same for all dimensions)
    taus = data[dims[0]]["tau"]
    n_taus = len(taus)
    n_dims = len(dims)

    # Setup grouped bars
    width = 0.8 / n_dims  # Bar width
    x = np.arange(n_taus)  # X positions for tau values

    for i, d in enumerate(dims):
        probs = data[d]["p"]
        errors = data[d]["err"]

        # Position bars for this dimension
        x_pos = x + (i - n_dims / 2 + 0.5) * width

        bars = ax.bar(
            x_pos, probs, width, label=f"d={d}", color=colors[i], yerr=errors, capsize=3, alpha=0.8
        )

    # Formatting
    ax.set_xlabel("τ (threshold)", fontsize=12)
    ax.set_ylabel("P(unreachability)", fontsize=12)
    ax.set_title(f"Threshold Sensitivity — {ensemble}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{tau:.2f}" for tau in taus])
    ax.legend(ncol=min(3, len(dims)), fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save single aggregated figure (required filename)
    filename = f"tau_hist_{ensemble}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved tau histogram: {filepath}")
    return [filepath]


def plot_optimizer_comparison(
    data: Dict[str, Dict[int, Dict[str, Any]]], ensemble: str, output_dir: Optional[str] = None
) -> str:
    """
    Plot optimizer comparison showing mean S* (max spectral overlap) distribution.

    Creates grouped bar chart showing mean±SEM(S*) for each optimization method,
    with different colors representing different dimensions.

    Statistic plotted: S* = max spectral overlap across trials (NOT P(unreachability))

    Args:
        data: Dictionary {method: {d: {'mean_S': float, 'sem_S': float}}}
        ensemble: "GOE" or "GUE"
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    if not data:
        logger.warning("No data provided for optimizer comparison")
        return ""

    # Extract methods and dimensions
    methods = sorted(data.keys())
    all_dims = set()
    for method_data in data.values():
        all_dims.update(method_data.keys())
    dims = sorted(all_dims)

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    # Colors for dimensions
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    # Calculate bar width and positions
    n_dims = len(dims)
    bar_width = 0.8 / n_dims
    x_positions = np.arange(len(methods))

    for i, d in enumerate(dims):
        means = []
        sems = []

        for method in methods:
            if d in data[method]:
                means.append(data[method][d]["mean_S"])
                sems.append(data[method][d]["sem_S"])
            else:
                means.append(0.0)
                sems.append(0.0)

        # Bar positions for this dimension
        x_pos = x_positions + (i - n_dims / 2 + 0.5) * bar_width

        ax.bar(
            x_pos,
            means,
            yerr=sems,
            width=bar_width,
            color=colors[i],
            label=f"d={d}",
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1},
        )

    # Formatting
    ax.set_xlabel("Optimization Method", fontsize=12)
    ax.set_ylabel("Mean S* ± SEM", fontsize=12)
    ax.set_title(f"Optimizer Comparison: Max Spectral Overlap — {ensemble}", fontsize=14)
    ax.legend(ncol=2, fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(0, 1.05)

    # Add caption annotation
    ax.text(
        0.98,
        0.02,
        "Statistic = max spectral overlap (S*) across trials",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
        ha="right",
        va="bottom",
    )

    plt.tight_layout()

    # Save with corrected filename
    filename = f"optimizer_overlap_hist_{ensemble}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved optimizer comparison: {filepath}")
    return filepath


def plot_punreach_heatmaps(
    data: Dict[str, np.ndarray], ensemble: str, epsilon: float, output_dir: Optional[str] = None
) -> str:
    """
    Plot P(unreachability) vs (dimension, K) as heatmap.

    Shows unreachability probability over parameter space for fixed ε threshold.

    Args:
        data: Dictionary from analysis.punreach_vs_dimension_K
        ensemble: "GOE" or "GUE"
        epsilon: Threshold value for unreachability
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    eps_key = f"eps_{epsilon:.2f}".replace(".", "_")

    if eps_key not in data:
        logger.warning(f"No data for epsilon={epsilon}")
        return ""

    d_vals = data["d_vals"]
    K_vals = data["K_vals"]
    P_grid = data[eps_key]

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    # Create heatmap
    im = ax.imshow(
        P_grid,
        cmap=settings.DEFAULT_COLORMAP,
        aspect="auto",
        origin="lower",
        extent=[K_vals[0], K_vals[-1], d_vals[0], d_vals[-1]],
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(unreachability)", fontsize=12)

    # Labels and title
    ax.set_xlabel("K (number of Hamiltonians)", fontsize=12)
    ax.set_ylabel("d (dimension)", fontsize=12)
    ax.set_title(f"P(unreachability) vs (d,K) — {ensemble}, ε={epsilon}", fontsize=14)

    # Set ticks
    ax.set_xticks(K_vals)
    ax.set_yticks(d_vals)

    plt.tight_layout()

    # Save with exact filename
    filename = f"punreach_vs_dk_{ensemble}_eps{epsilon:.2f}".replace(".", "_") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved P(unreachability) heatmap: {filepath}")
    return filepath


def plot_iteration_sweep(
    data: Dict[str, np.ndarray],
    d: int,
    k: int,
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot convergence analysis: P(unreachability) and runtime vs iterations.

    Shows how optimization convergence affects unreachability detection.

    Args:
        data: Dictionary from analysis.probability_vs_iterations
        d: Hilbert space dimension
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    iters = data["iterations"]
    probs = data["probabilities"]
    errors = data["errors"]
    runtimes = data["runtimes"]

    # Left subplot: P(unreachability) vs iterations
    ax1.errorbar(
        iters, probs, yerr=errors, fmt="bo-", linewidth=2, markersize=8, capsize=5, capthick=2
    )
    ax1.set_xlabel("Max Iterations", fontsize=12)
    ax1.set_ylabel("P(unreachability)", fontsize=12)
    ax1.set_title(f"Convergence — {ensemble}, d={d}, k={k}, τ={tau}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Right subplot: Runtime vs iterations
    ax2.semilogy(iters, runtimes, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Max Iterations", fontsize=12)
    ax2.set_ylabel("Runtime (s)", fontsize=12)
    ax2.set_title("Computational Cost", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("L-BFGS-B Iteration Analysis", fontsize=14)
    plt.tight_layout()

    # Save with exact filename
    filename = f"iter_sweep_prob_{ensemble}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved iteration sweep: {filepath}")
    return filepath


def plot_iteration_sweep_multi_dk(
    data_dict: Dict[Tuple[int, int], Dict[str, np.ndarray]],
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot convergence analysis for multiple (d,k) pairs on same figure.

    Shows how optimization convergence affects unreachability detection across
    different (d,k) configurations with overlaid curves.

    Args:
        data_dict: Dictionary mapping (d,k) → iteration sweep data
                  {(d1,k1): {'iterations': ..., 'probabilities': ..., 'errors': ..., 'runtimes': ...},
                   (d2,k2): {...}, ...}
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Use distinct colors for each (d,k) pair
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_dict)))

    for idx, ((d, k), data) in enumerate(sorted(data_dict.items())):
        iters = data["iterations"]
        probs = data["probabilities"]
        errors = data["errors"]
        runtimes = data["runtimes"]

        # Left subplot: P(unreachability) vs iterations
        ax1.errorbar(
            iters,
            probs,
            yerr=errors,
            fmt="o-",
            linewidth=2,
            markersize=6,
            capsize=4,
            capthick=1.5,
            color=colors[idx],
            label=f"d={d},k={k}",
            alpha=0.8,
        )

        # Right subplot: Runtime vs iterations
        ax2.semilogy(
            iters,
            runtimes,
            "o-",
            linewidth=2,
            markersize=6,
            color=colors[idx],
            label=f"d={d},k={k}",
            alpha=0.8,
        )

    # Configure left subplot
    ax1.set_xlabel("Max Iterations", fontsize=12)
    ax1.set_ylabel("P(unreachability)", fontsize=12)
    ax1.set_title(f"Convergence — {ensemble}, τ={tau}", fontsize=13)
    ax1.legend(ncol=1, fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Configure right subplot
    ax2.set_xlabel("Max Iterations", fontsize=12)
    ax2.set_ylabel("Runtime (s)", fontsize=12)
    ax2.set_title("Computational Cost", fontsize=13)
    ax2.legend(ncol=1, fontsize=10, loc="best")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Multi-(d,k) L-BFGS-B Iteration Analysis", fontsize=14)
    plt.tight_layout()

    # Save figure
    filename = f"iter_sweep_prob_{ensemble}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved multi-(d,k) iteration sweep: {filepath}")
    return filepath


def plot_rank_comparison(
    old_results: Dict[Tuple[int, int], float],
    new_results: Dict[Tuple[int, int], float],
    dims: List[int],
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot old vs new criterion comparison with τ annotations.

    Compares moment-based (old, τ-free) and spectral overlap (new, τ-based)
    reachability criteria with proper display floor handling.

    IMPORTANT: Old criterion is τ-free (definiteness check), new uses τ threshold.

    Args:
        old_results: Dictionary mapping (d,k) → probability (old criterion, τ-free)
        new_results: Dictionary mapping (d,k) → probability (new criterion, uses τ)
        dims: List of dimensions to plot
        ensemble: "GOE" or "GUE"
        tau: Threshold value used by new criterion
        output_dir: Directory for saving figures

    Returns:
        Path to saved figure (only old_vs_new comparison)
    """
    output_dir = output_dir or settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    # Log τ audit information
    logger.info("[rank-compare] old_uses_tau=False")
    logger.info(f"[rank-compare] new_tau={tau}")

    # Create single comparison figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    # Plot old and new criteria together
    for idx, d in enumerate(dims):
        # Old criterion (τ-free)
        points_old = [(k, p) for (dim, k), p in old_results.items() if dim == d]
        if points_old:
            points_old.sort(key=lambda x: x[0])
            ks, ps = zip(*points_old)
            apply_masked_connections(
                ax,
                np.array(ks),
                np.array(ps),
                color=colors[idx],
                label=f"d={d} (old)",
                linewidth=2,
                marker="o",
                markersize=6,
            )

        # New criterion (uses τ)
        points_new = [(k, p) for (dim, k), p in new_results.items() if dim == d]
        if points_new:
            points_new.sort(key=lambda x: x[0])
            ks, ps = zip(*points_new)
            apply_masked_connections(
                ax,
                np.array(ks),
                np.array(ps),
                color=colors[idx],
                label=f"d={d} (new)",
                linewidth=1.5,
                marker="s",
                markersize=5,
                linestyle="--",
                alpha=0.7,
            )

    # Compute actual k range from data
    ks_all = sorted({k for (_, k) in old_results.keys()} | {k for (_, k) in new_results.keys()})

    ax.set_xlabel("k (Number of Hamiltonians)", fontsize=12)
    ax.set_ylabel("log(P(detected unreachability))", fontsize=12)
    ax.set_title(f"Old vs New Criterion — {ensemble} (τ={tau:.2f})", fontsize=14)
    ax.legend(ncol=3, fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3)

    # Set x-axis limits and ticks based on actual data
    if ks_all:
        ax.set_xlim(min(ks_all) - 0.2, max(ks_all) + 0.2)
        ax.set_xticks(ks_all)

    ax.set_ylim(bottom=-12, top=1)

    # Add floor annotation
    ax.text(
        0.02,
        0.02,
        f"Display floor = {settings.DISPLAY_FLOOR}; floored points not connected",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.7,
    )

    # Add τ annotation (critical: old is τ-free, new uses τ)
    ax.text(
        0.98,
        0.98,
        f"old: τ-free; new: τ={tau:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        ha="right",
        va="top",
    )

    plt.tight_layout()

    # Save with tau in filename
    filename = f"unreachability_vs_rank_old_vs_new_{ensemble}_tau{tau:.2f}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved rank comparison: {filepath}")
    return filepath


def plot_tau_histograms_multiD(
    tau_data: Dict[int, Dict[str, np.ndarray]],
    k: int,
    ensemble: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot tau histograms for multiple dimensions in a single figure.

    Creates overlaid tau distribution histograms for different dimensions,
    allowing comparison of unreachability threshold patterns across dimensions.

    Args:
        tau_data: Dictionary mapping dimension → tau histogram data
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    output_dir = output_dir or settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    fig, ax = plt.subplots(1, 1, figsize=settings.DEFAULT_FIGSIZE)

    # Use distinct colors for each dimension
    colors = plt.cm.viridis(np.linspace(0, 1, len(tau_data)))

    for idx, (d, data) in enumerate(sorted(tau_data.items())):
        tau_bins = data["tau_bins"]
        frequencies = data["frequencies"]

        # Plot histogram as step plot for better overlay visualization
        ax.step(
            tau_bins[:-1],
            frequencies,
            where="post",
            color=colors[idx],
            linewidth=2,
            alpha=0.8,
            label=f"d={d}",
        )

    ax.set_xlabel("Spectral Overlap τ", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Tau Distributions — {ensemble}, k={k}", fontsize=14)
    ax.legend(ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()

    # Save figure
    filename = f"tau_hist_multiD_{ensemble}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved multi-D tau histograms: {filepath}")
    return filepath


def plot_landscape_summary(
    L1: np.ndarray,
    L2: np.ndarray,
    S: np.ndarray,
    d: int,
    k: int,
    ensemble: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot 2-panel landscape summary: 2D and 3D views side by side.

    Creates a comprehensive landscape visualization combining both
    2D heatmap and 3D surface representations in a single figure.

    Args:
        L1: First parameter meshgrid
        L2: Second parameter meshgrid
        S: Spectral overlap values
        d: Hilbert space dimension
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    output_dir = output_dir or settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    fig = plt.figure(figsize=settings.LANDSCAPE_FIGSIZE)

    # Left panel: 2D heatmap
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(
        S,
        extent=[L1.min(), L1.max(), L2.min(), L2.max()],
        origin="lower",
        cmap=settings.DEFAULT_COLORMAP,
        aspect="auto",
    )
    ax1.set_xlabel("λ₁", fontsize=12)
    ax1.set_ylabel("λ₂", fontsize=12)
    ax1.set_title(f"2D Landscape — {ensemble}, d={d}, k={k}", fontsize=13)

    # Add colorbar for 2D plot
    cbar1 = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar1.set_label("Spectral Overlap S(λ)", fontsize=11)

    # Right panel: 3D surface
    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(
        L1, L2, S, cmap=settings.DEFAULT_COLORMAP, alpha=0.9, linewidth=0, antialiased=True
    )
    ax2.set_xlabel("λ₁", fontsize=11)
    ax2.set_ylabel("λ₂", fontsize=11)
    ax2.set_zlabel("S(λ)", fontsize=11)
    ax2.set_title(f"3D Surface — {ensemble}, d={d}, k={k}", fontsize=13)

    # Add colorbar for 3D plot
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label("Spectral Overlap S(λ)", fontsize=10)

    plt.tight_layout()

    # Save figure
    filename = f"landscape_summary_{ensemble}_d{d}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved landscape summary: {filepath}")
    return filepath


def plot_overlap_hist_pdf(
    Sstar_by_d: Dict[int, np.ndarray],
    ensemble: str,
    bins: np.ndarray = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot PDF histogram of S* with bin width 0.01 in [0.90,1.00].

    Creates grouped bar chart showing probability distribution of S* (max spectral overlap)
    with X-axis = bins, bars = P_i for each d ∈ {12,16,20,24,30}.

    Args:
        Sstar_by_d: Dictionary mapping dimension → array of S* values
        ensemble: "GOE" or "GUE"
        bins: Bin edges (default: np.arange(0.90, 1.00 + 1e-9, 0.01))
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    if bins is None:
        bins = np.arange(0.90, 1.00 + 1e-9, 0.01)

    # Sort dimensions
    dims = sorted(Sstar_by_d.keys())
    n_dims = len(dims)

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    # Colors for dimensions
    colors = plt.cm.viridis(np.linspace(0, 1, n_dims))

    # Setup grouped bars
    n_bins = len(bins) - 1
    width = 0.8 / n_dims  # Bar width
    x = np.arange(n_bins)  # X positions for bins

    # Track min/max probabilities for logging
    all_probs = []

    for i, d in enumerate(dims):
        # Filter values within the range [0.90, 1.00]
        values = Sstar_by_d[d]
        values_in_range = values[(values >= bins[0]) & (values <= bins[-1])]

        # Log fraction outside range
        if len(values) > 0:
            frac_outside = 1.0 - len(values_in_range) / len(values)
            if frac_outside > 0:
                logger.info(
                    f"  d={d}: {frac_outside*100:.2f}% of samples outside [{bins[0]:.2f},{bins[-1]:.2f}]"
                )

        # Compute histogram counts
        counts, _ = np.histogram(values_in_range, bins=bins)

        # Normalize to probabilities (sum to 1)
        total = np.sum(counts)
        if total > 0:
            probs = counts / total
        else:
            probs = np.zeros(n_bins)
            logger.warning(f"  d={d}: no samples in range")

        all_probs.extend(probs)

        # Position bars for this dimension
        x_pos = x + (i - n_dims / 2 + 0.5) * width

        # Plot bars
        ax.bar(x_pos, probs, width, label=f"d={d}", color=colors[i], alpha=0.8)

    # Formatting
    ax.set_xlabel("S* bins (width 0.01)", fontsize=12)
    ax.set_ylabel("P_i = P(S* in bin)", fontsize=12)
    ax.set_title(f'Histogram S* — {ensemble}, d ∈ {{{",".join(map(str, dims))}}}', fontsize=14)

    # Set x-axis ticks to show bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.set_xticks(x[::2])  # Show every other tick to avoid crowding
    ax.set_xticklabels(
        [f"{bin_centers[i]:.2f}" for i in range(0, n_bins, 2)], rotation=45, ha="right"
    )

    ax.legend(ncol=min(3, n_dims), fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Set y-axis limits
    if all_probs:
        y_max = max(all_probs)
        ax.set_ylim(0, y_max * 1.1)

    plt.tight_layout()

    # Save with exact filename
    filename = f"overlap_hist_pdf_{ensemble}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved overlap histogram PDF: {filepath}")
    return filepath


def plot_iter_sweep_multiD(
    iter_data: Dict[int, Dict[str, np.ndarray]],
    k: int,
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot iteration sweep comparison for multiple dimensions in a single figure.

    Creates overlaid convergence analysis showing how optimization convergence
    affects unreachability detection across different dimensions.

    Args:
        iter_data: Dictionary mapping dimension → iteration sweep data
        k: Number of Hamiltonians
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        output_dir: Directory for saving figure

    Returns:
        Path to saved figure
    """
    output_dir = output_dir or settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Use distinct colors for each dimension
    colors = plt.cm.viridis(np.linspace(0, 1, len(iter_data)))

    for idx, (d, data) in enumerate(sorted(iter_data.items())):
        iters = data["iterations"]
        probs = data["probabilities"]
        errors = data["errors"]
        runtimes = data["runtimes"]

        # Left subplot: P(unreachability) vs iterations
        ax1.errorbar(
            iters,
            probs,
            yerr=errors,
            fmt="o-",
            linewidth=2,
            markersize=6,
            capsize=4,
            capthick=1.5,
            color=colors[idx],
            label=f"d={d}",
            alpha=0.8,
        )

        # Right subplot: Runtime vs iterations
        ax2.semilogy(
            iters,
            runtimes,
            "o-",
            linewidth=2,
            markersize=6,
            color=colors[idx],
            label=f"d={d}",
            alpha=0.8,
        )

    # Configure left subplot
    ax1.set_xlabel("Max Iterations", fontsize=12)
    ax1.set_ylabel("P(unreachability)", fontsize=12)
    ax1.set_title(f"Convergence — {ensemble}, k={k}, τ={tau}", fontsize=13)
    ax1.legend(ncol=2, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Configure right subplot
    ax2.set_xlabel("Max Iterations", fontsize=12)
    ax2.set_ylabel("Runtime (s)", fontsize=12)
    ax2.set_title("Computational Cost", fontsize=13)
    ax2.legend(ncol=2, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Multi-D L-BFGS-B Iteration Analysis", fontsize=14)
    plt.tight_layout()

    # Save figure
    filename = f"iter_sweep_multiD_{ensemble}_k{k}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved multi-D iteration sweeps: {filepath}")
    return filepath
