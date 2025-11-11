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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import settings

logger = logging.getLogger(__name__)

# Configure matplotlib for consistent output
plt.rcParams["figure.dpi"] = settings.DEFAULT_DPI
plt.rcParams["savefig.dpi"] = settings.DEFAULT_DPI
plt.rcParams["savefig.bbox"] = "tight"


def _create_floor_masked_array(y_values: np.ndarray, floor: float) -> np.ma.MaskedArray:
    """
    Create a masked array that breaks line segments at floor values.

    This prevents matplotlib from drawing vertical "cliff" lines when values
    hit the display floor. Floored points will be plotted separately as
    faded markers without connecting lines.

    Args:
        y_values: Array of y values (probabilities)
        floor: Display floor value

    Returns:
        Masked array where floor values are masked
    """
    # Mask values at or near the floor (within 1% tolerance)
    is_floored = np.abs(y_values - floor) < floor * 0.01
    return np.ma.masked_where(is_floored, y_values)


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def _wilson_interval(k: int, n: int, z: float = 1.0) -> Tuple[float, float]:
    """
    Compute Wilson score interval for binomial proportion k/n.

    The Wilson interval provides better coverage near boundaries (p ≈ 0 or p ≈ 1)
    compared to the normal approximation (Wald interval). For z=1.0, this
    approximates a 68% confidence interval (≈1σ).

    Mathematical formula:
        p̂ = k/n
        denom = 1 + z²/n
        center = (p̂ + z²/(2n)) / denom
        margin = (z/denom) × √(p̂(1-p̂)/n + z²/(4n²))
        interval = [max(0, center - margin), min(1, center + margin)]

    Args:
        k: Number of successes (0 ≤ k ≤ n)
        n: Number of trials (n > 0)
        z: Z-score for confidence level (default: 1.0 for ≈68% CI)

    Returns:
        (lower_bound, upper_bound) both in [0, 1]

    References:
        Wilson, E.B. (1927). "Probable inference, the law of succession, and
        statistical inference". Journal of the American Statistical Association.
    """
    if n == 0:
        return 0.0, 0.0

    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper


def _compute_asymmetric_errorbars(
    p: np.ndarray, n: int, floor: float = settings.DISPLAY_FLOOR, z: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute asymmetric error bars for probabilities using Wilson intervals.

    Error bars are floor-aware and hidden when meaningless (k=0, k=n, or p≤floor).

    Args:
        p: Array of probabilities (each in [0, 1])
        n: Total number of trials (same for all p values)
        floor: Display floor for log plots (default: settings.DISPLAY_FLOOR)
        z: Z-score for Wilson interval (default: 1.0)

    Returns:
        (err_lower, err_upper): Arrays of asymmetric error bar sizes
        where err_lower[i] = p[i] - lower[i] and err_upper[i] = upper[i] - p[i]
        Hidden bars (p≤floor, k=0, k=n) have err_lower=err_upper=0
    """
    err_lower = np.zeros_like(p)
    err_upper = np.zeros_like(p)

    for i, prob in enumerate(p):
        # Infer success count from probability
        k = int(round(prob * n))

        # Hide error bars when meaningless
        if prob <= floor or k == 0 or k == n or n < 5:
            continue  # Leave as zero (no error bar)

        # Compute Wilson interval
        lo, hi = _wilson_interval(k, n, z=z)

        # Floor-aware clipping: lower bound cannot go below floor
        lo = max(lo, floor)

        # Asymmetric error bar sizes
        err_lower[i] = prob - lo
        err_upper[i] = hi - prob

    return err_lower, err_upper


def _to_nines(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert probability to "nines" scale: y = -log₁₀(1 - p).

    This scale makes values close to 1 more visible. For example:
    - p = 0.9 → y ≈ 1 (one nine)
    - p = 0.99 → y ≈ 2 (two nines)
    - p = 0.999 → y ≈ 3 (three nines)

    Args:
        p: Probability array (values in [0, 1])
        eps: Epsilon floor to avoid inf when p=1 (default: 1e-8)

    Returns:
        Transformed values on nines scale
    """
    return -np.log10(np.clip(1.0 - p, eps, 1.0))


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

    # Extract axis limits from meshgrid
    lambda_min = L1[0, 0]
    lambda_max = L1[0, -1]

    # Create heatmap with imshow (no interpolation for exact pixel rendering)
    im = ax.imshow(
        S_clipped,
        cmap=settings.DEFAULT_COLORMAP,
        interpolation="none",
        origin="lower",
        extent=[lambda_min, lambda_max, lambda_min, lambda_max],
        aspect="equal",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("S(λ₁,λ₂)", fontsize=12)

    # Add crosshairs at λ₁=0 and λ₂=0
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=10)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=10)

    # Labels and title
    ax.set_xlabel("λ₁", fontsize=12)
    ax.set_ylabel("λ₂", fontsize=12)
    ax.set_title(f"S(λ₁,λ₂) — {ensemble}, d={d}, k={k} (2D)", fontsize=14)

    # Set symmetric ticks
    tick_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    tick_vals_in_range = [t for t in tick_vals if lambda_min <= t <= lambda_max]
    ax.set_xticks(tick_vals_in_range)
    ax.set_yticks(tick_vals_in_range)

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
    ax.grid(True, alpha=0.3, linewidth=0.5)

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
    parameter space (λ₁, λ₂):
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

    # Create 3D surface with raw mesh (no interpolation)
    surf = ax.plot_surface(
        L1,
        L2,
        S_clipped,
        cmap=settings.DEFAULT_COLORMAP,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
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

        _ = ax.bar(  # plotted bars; result unused (silence F841)
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


def plot_unreach_vs_k_single_d(
    data: Dict[str, np.ndarray],
    d: int,
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
    y_floor: float = 1e-8,
) -> str:
    """
    Plot P(unreachability) vs K for a single fixed dimension.

    Creates a line plot with error bars showing unreachability probability as a
    function of K (number of Hamiltonians) for a fixed dimension d and threshold τ.

    Uses masked connections to avoid vertical artifacts when probabilities hit the floor.

    Args:
        data: Dictionary with keys 'k' (K values), 'p' (probabilities), 'err' (SEM)
        d: Hilbert space dimension (fixed)
        ensemble: "GOE" or "GUE"
        tau: Unreachability threshold
        output_dir: Directory for saving figure
        y_floor: Display floor for log scale (default: 1e-8)

    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    ks = data["k"]
    probs = data["p"]
    errs = data["err"]

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    # Clamp probabilities to [y_floor, 1] for display
    probs_display = np.clip(probs, y_floor, 1.0)

    # Use log scale for y-axis
    probs_log = np.log10(probs_display)

    # Plot with masked connections (don't connect floored points)
    apply_masked_connections(
        ax,
        ks,
        probs,
        floor=y_floor,
        color="steelblue",
        linewidth=2.5,
        marker="o",
        markersize=7,
        label=f"d={d}, τ={tau}",
    )

    # Add error bars separately (only for non-floored points)
    is_floored = probs <= y_floor
    for i, k in enumerate(ks):
        if not is_floored[i]:
            # Convert SEM to log space for error bars
            p_lower = max(probs[i] - errs[i], y_floor)
            p_upper = min(probs[i] + errs[i], 1.0)
            err_lower = np.log10(probs[i]) - np.log10(p_lower)
            err_upper = np.log10(p_upper) - np.log10(probs[i])
            ax.errorbar(
                k,
                probs_log[i],
                yerr=[[err_lower], [err_upper]],
                fmt="none",
                color="steelblue",
                capsize=4,
                capthick=1.5,
                alpha=0.7,
            )

    # Configure axes
    ax.set_xlabel("K (Number of Hamiltonians)", fontsize=12)
    ax.set_ylabel("log₁₀(P(unreachability))", fontsize=12)
    ax.set_title(f"P(unreachability) vs K — {ensemble}, d={d}, τ={tau}", fontsize=14)

    # Set y-axis limits and ticks for log scale
    ax.set_ylim(np.log10(y_floor), 0)  # [log(y_floor), log(1)] = [log(y_floor), 0]
    y_ticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]
    y_ticks_in_range = [yt for yt in y_ticks if y_floor <= yt <= 1.0]
    ax.set_yticks([np.log10(yt) for yt in y_ticks_in_range])
    ax.set_yticklabels([f"{yt:.0e}" if yt < 1 else "1" for yt in y_ticks_in_range])

    # Set x-axis ticks to integer K values
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)

    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    # Annotate floor value
    ax.text(
        0.98,
        0.02,
        f"Floor: {y_floor:.0e}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()

    # Save with exact filename
    filename = f"unreachability_vs_k_single_d_{ensemble}_d{d}_tau{tau:.2f}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved single-d K sweep: {filepath}")
    return filepath


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
    moment_results: Dict[Tuple[int, int], float],
    spectral_results: Dict[Tuple[int, int], float],
    dims: List[int],
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
) -> str:
    """
    Plot moment vs spectral criterion comparison with τ annotations.

    Compares moment-based (τ-free) and spectral overlap (τ-based)
    reachability criteria with proper display floor handling.

    IMPORTANT: Moment criterion is τ-free (definiteness check), spectral uses τ threshold.

    Args:
        moment_results: Dictionary mapping (d,k) → probability (moment criterion, τ-free)
        spectral_results: Dictionary mapping (d,k) → probability (spectral criterion, uses τ)
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
    logger.info("[rank-compare] moment_uses_tau=False")
    logger.info(f"[rank-compare] spectral_tau={tau}")

    # Create single comparison figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    # Plot moment and spectral criteria together
    for idx, d in enumerate(dims):
        # Moment criterion (τ-free)
        points_moment = [(k, p) for (dim, k), p in moment_results.items() if dim == d]
        if points_moment:
            points_moment.sort(key=lambda x: x[0])
            ks, ps = zip(*points_moment)
            apply_masked_connections(
                ax,
                np.array(ks),
                np.array(ps),
                color=colors[idx],
                label=f"d={d} (Moment)",
                linewidth=2,
                marker="o",
                markersize=6,
            )

        # Spectral criterion (uses τ)
        points_spectral = [(k, p) for (dim, k), p in spectral_results.items() if dim == d]
        if points_spectral:
            points_spectral.sort(key=lambda x: x[0])
            ks, ps = zip(*points_spectral)
            apply_masked_connections(
                ax,
                np.array(ks),
                np.array(ps),
                color=colors[idx],
                label=f"d={d} (Spectral)",
                linewidth=1.5,
                marker="s",
                markersize=5,
                linestyle="--",
                alpha=0.7,
            )

    # Compute actual k range from data
    ks_all = sorted({k for (_, k) in moment_results.keys()} | {k for (_, k) in spectral_results.keys()})

    ax.set_xlabel("k (Number of Hamiltonians)", fontsize=12)
    ax.set_ylabel("log(P(detected unreachability))", fontsize=12)
    ax.set_title(f"Moment vs Spectral Criterion — {ensemble} (τ={tau:.2f})", fontsize=14)
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


def plot_rank_comparison_with_inset(
    moment_results: Dict[Tuple[int, int], float],
    spectral_results: Dict[Tuple[int, int], float],
    dims: List[int],
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
    y_floor: float = 1e-8,
    zoom_xlim: Tuple[float, float] = (1, 6),
    zoom_ylim: Tuple[float, float] = (1e-8, 3e-2),
) -> str:
    """
    Plot moment vs spectral criterion comparison with zoomed inset.

    Creates a main plot with full data range and adds a zoomed inset focusing on
    low-k, low-probability region. Includes a box indicator on the main plot showing
    the inset region.

    Args:
        moment_results: Dictionary mapping (d,k) → probability (moment criterion, τ-free)
        spectral_results: Dictionary mapping (d,k) → probability (spectral criterion, uses τ)
        dims: List of dimensions to plot
        ensemble: "GOE" or "GUE"
        tau: Threshold value used by new criterion
        output_dir: Directory for saving figures
        y_floor: Display floor for log scale in inset (default: 1e-8)
        zoom_xlim: X-axis limits for inset zoom (default: (1, 6))
        zoom_ylim: Y-axis limits for inset zoom in log scale (default: (1e-8, 3e-2))

    Returns:
        Path to saved figure
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    output_dir = output_dir or settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    # Log τ audit information
    logger.info("[rank-compare-zoom] old_uses_tau=False")
    logger.info(f"[rank-compare-zoom] new_tau={tau}")

    # Create main figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    # Plot old and new criteria on main axes
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
                floor=y_floor,
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
                floor=y_floor,
                color=colors[idx],
                label=f"d={d} (new)",
                linewidth=1.5,
                marker="s",
                markersize=5,
                linestyle="--",
                alpha=0.7,
            )

    # Configure main axes
    ks_all = sorted({k for (_, k) in old_results.keys()} | {k for (_, k) in new_results.keys()})
    ax.set_xlabel("k (Number of Hamiltonians)", fontsize=12)
    ax.set_ylabel("log₁₀(P(detected unreachability))", fontsize=12)
    ax.set_title(f"Old vs New Criterion with Zoom — {ensemble} (τ={tau:.3f})", fontsize=14)
    ax.legend(ncol=3, fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(True, alpha=0.3)

    if ks_all:
        ax.set_xlim(min(ks_all) - 0.2, max(ks_all) + 0.2)
        ax.set_xticks(ks_all)

    # Main y-axis: log scale from floor to 1
    ax.set_ylim(np.log10(y_floor), 0)
    y_ticks_main = [1e-8, 1e-6, 1e-4, 1e-2, 1e0]
    y_ticks_main_in_range = [yt for yt in y_ticks_main if y_floor <= yt <= 1.0]
    ax.set_yticks([np.log10(yt) for yt in y_ticks_main_in_range])
    ax.set_yticklabels([f"{yt:.0e}" if yt < 1 else "1" for yt in y_ticks_main_in_range])

    # Add τ annotation
    ax.text(
        0.02,
        0.98,
        f"old: τ-free; new: τ={tau:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        ha="left",
        va="top",
    )

    # Create inset axes (positioned in lower left)
    axins = inset_axes(ax, width="35%", height="35%", loc="lower left", borderpad=2.5)

    # Plot data on inset with same styling
    for idx, d in enumerate(dims):
        # Moment criterion
        points_moment = [(k, p) for (dim, k), p in moment_results.items() if dim == d]
        if points_moment:
            points_moment.sort(key=lambda x: x[0])
            ks, ps = zip(*points_moment)
            apply_masked_connections(
                axins,
                np.array(ks),
                np.array(ps),
                floor=y_floor,
                color=colors[idx],
                linewidth=2,
                marker="o",
                markersize=5,
            )

        # Spectral criterion
        points_spectral = [(k, p) for (dim, k), p in spectral_results.items() if dim == d]
        if points_spectral:
            points_spectral.sort(key=lambda x: x[0])
            ks, ps = zip(*points_spectral)
            apply_masked_connections(
                axins,
                np.array(ks),
                np.array(ps),
                floor=y_floor,
                color=colors[idx],
                linewidth=1.5,
                marker="s",
                markersize=4,
                linestyle="--",
                alpha=0.7,
            )

    # Configure inset axes (zoomed region)
    axins.set_xlim(zoom_xlim)
    axins.set_ylim(np.log10(zoom_ylim[0]), np.log10(zoom_ylim[1]))

    # Inset y-ticks
    y_ticks_inset = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    y_ticks_inset_in_range = [
        yt for yt in y_ticks_inset if zoom_ylim[0] <= yt <= zoom_ylim[1]
    ]
    axins.set_yticks([np.log10(yt) for yt in y_ticks_inset_in_range])
    axins.set_yticklabels([f"{yt:.0e}" for yt in y_ticks_inset_in_range], fontsize=8)

    # Inset x-ticks
    x_ticks_inset = list(range(int(zoom_xlim[0]), int(zoom_xlim[1]) + 1))
    axins.set_xticks(x_ticks_inset)
    axins.set_xticklabels([str(xt) for xt in x_ticks_inset], fontsize=8)

    axins.grid(True, alpha=0.3, linewidth=0.5)
    axins.tick_params(labelsize=8)

    # Draw box on main plot showing zoom region
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", linestyle="--", linewidth=1.5)

    plt.tight_layout()

    # Save with exact filename including _zoom suffix
    filename = f"unreachability_vs_rank_old_vs_new_{ensemble}_tau{tau:.3f}_zoom.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved rank comparison with zoom: {filepath}")
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


def plot_rank_comparison_rescaled(
    moment_results: Dict[Tuple[int, int], float],
    spectral_results: Dict[Tuple[int, int], float],
    dims: List[int],
    ensemble: str,
    tau: float,
    output_dir: Optional[str] = None,
    eps_floor: float = 1e-9,
    legend_loc: str = "lower left",
    hide_floored: bool = True,
) -> str:
    """
    Plot moment vs spectral criterion comparison using log10(P) scale (no inset).

    Y-axis dynamically trimmed to data range. Legend placed at bottom to avoid
    occluding data.

    Args:
        moment_results: Dictionary mapping (d,k) → probability (moment criterion, τ-free)
        spectral_results: Dictionary mapping (d,k) → probability (spectral criterion, uses τ)
        dims: List of dimensions to plot
        ensemble: "GOE" or "GUE"
        tau: Threshold value used by new criterion
        output_dir: Directory for saving figures (default: fig_summary)
        eps_floor: Epsilon floor to avoid log10(0) (default: 1e-9)
        legend_loc: Legend location (default: "lower left")
        hide_floored: If True, hide points with p <= eps_floor (default: True)

    Returns:
        Path to saved figure

    Note:
        Filename format: unreachability_vs_rank_old_vs_new_{ensemble}_tau{τ:.3f}.png
        (no _zoom suffix, unlike plot_rank_comparison_with_inset)
    """
    output_dir = output_dir or settings.FIG_SUMMARY_DIR
    ensure_dir(output_dir)

    # Log τ audit information
    logger.info("[rank-compare-rescaled] old_uses_tau=False")
    logger.info(f"[rank-compare-rescaled] new_tau={tau}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(dims)))

    # Collect all log10 values from visible (non-hidden) points to determine y-axis range
    all_log_values = []
    hidden_count_old = 0
    hidden_count_new = 0

    # Plot old and new criteria
    for idx, d in enumerate(dims):
        # Old criterion (τ-free)
        points_old = [(k, p) for (dim, k), p in old_results.items() if dim == d]
        if points_old:
            points_old.sort(key=lambda x: x[0])
            ks, ps = zip(*points_old)
            ks = np.array(ks)
            ps = np.array(ps)

            # Filter out floored points if requested
            if hide_floored:
                mask = ps > eps_floor
                ks_visible = ks[mask]
                ps_visible = ps[mask]
                hidden_count_old += np.sum(~mask)
            else:
                ks_visible = ks
                ps_visible = np.maximum(ps, eps_floor)

            if len(ks_visible) > 0:
                y_old = np.log10(ps_visible)
                all_log_values.extend(y_old)

                # Plot with line segments broken at hidden points
                if hide_floored and np.sum(~mask) > 0:
                    # Plot segments separately to break lines at hidden points
                    segments = []
                    current_seg_k = []
                    current_seg_y = []
                    for i, visible in enumerate(mask):
                        if visible:
                            current_seg_k.append(ks[i])
                            current_seg_y.append(np.log10(ps[i]))
                        else:
                            if current_seg_k:
                                segments.append((np.array(current_seg_k), np.array(current_seg_y)))
                                current_seg_k = []
                                current_seg_y = []
                    if current_seg_k:
                        segments.append((np.array(current_seg_k), np.array(current_seg_y)))

                    for seg_idx, (seg_k, seg_y) in enumerate(segments):
                        ax.plot(seg_k, seg_y, color=colors[idx],
                               label=f"d={d} (old)" if seg_idx == 0 else None,
                               linewidth=2, marker="o", markersize=6, linestyle="-")
                else:
                    ax.plot(ks_visible, y_old, color=colors[idx], label=f"d={d} (old)",
                           linewidth=2, marker="o", markersize=6, linestyle="-")

        # New criterion (uses τ)
        points_new = [(k, p) for (dim, k), p in new_results.items() if dim == d]
        if points_new:
            points_new.sort(key=lambda x: x[0])
            ks, ps = zip(*points_new)
            ks = np.array(ks)
            ps = np.array(ps)

            # Filter out floored points if requested
            if hide_floored:
                mask = ps > eps_floor
                ks_visible = ks[mask]
                ps_visible = ps[mask]
                hidden_count_new += np.sum(~mask)
            else:
                ks_visible = ks
                ps_visible = np.maximum(ps, eps_floor)

            if len(ks_visible) > 0:
                y_new = np.log10(ps_visible)
                all_log_values.extend(y_new)

                # Plot with line segments broken at hidden points
                if hide_floored and np.sum(~mask) > 0:
                    # Plot segments separately to break lines at hidden points
                    segments = []
                    current_seg_k = []
                    current_seg_y = []
                    for i, visible in enumerate(mask):
                        if visible:
                            current_seg_k.append(ks[i])
                            current_seg_y.append(np.log10(ps[i]))
                        else:
                            if current_seg_k:
                                segments.append((np.array(current_seg_k), np.array(current_seg_y)))
                                current_seg_k = []
                                current_seg_y = []
                    if current_seg_k:
                        segments.append((np.array(current_seg_k), np.array(current_seg_y)))

                    for seg_idx, (seg_k, seg_y) in enumerate(segments):
                        ax.plot(seg_k, seg_y, color=colors[idx],
                               label=f"d={d} (new)" if seg_idx == 0 else None,
                               linewidth=1.5, marker="s", markersize=5, linestyle="--", alpha=0.7)
                else:
                    ax.plot(ks_visible, y_new, color=colors[idx], label=f"d={d} (new)",
                           linewidth=1.5, marker="s", markersize=5, linestyle="--", alpha=0.7)

    # Log and annotate hidden points
    hidden_total = hidden_count_old + hidden_count_new
    if hide_floored and hidden_total > 0:
        logger.info(
            f"[rank-compare] hidden {hidden_total} floored points @ eps_floor={eps_floor} "
            f"(old={hidden_count_old}, new={hidden_count_new})"
        )

    # Configure axes
    ks_all = sorted({k for (_, k) in old_results.keys()} | {k for (_, k) in new_results.keys()})
    ax.set_xlabel("k (Number of Hamiltonians)", fontsize=12)
    ax.set_ylabel("log₁₀(P(detected unreachability))", fontsize=12)
    ax.set_title(f"Old vs New Criterion — {ensemble} (τ={tau:.3f})", fontsize=14)
    ax.grid(True, alpha=0.3)

    # X-axis: integer ticks for k
    if ks_all:
        ax.set_xlim(min(ks_all) - 0.2, max(ks_all) + 0.2)
        ax.set_xticks(ks_all)

    # Y-axis: dynamically trim based on data
    # ymin = max(10^floor(log10(min_p)), 1e-9) → ymin_log = max(floor(log10(min_p)), -9)
    if all_log_values:
        y_min_data = min(all_log_values)
        y_min = max(np.floor(y_min_data), -9)
        # Y-max: small headroom above max visible probability
        y_max = min(0, max(all_log_values) + 0.05)
    else:
        y_min = -9
        y_max = 0

    ax.set_ylim(y_min, y_max)

    # Y-axis ticks: use scientific notation
    y_ticks = []
    y_tick_labels = []
    for exp in range(int(np.floor(y_min)), 1):
        y_ticks.append(exp)
        if exp == 0:
            y_tick_labels.append("1")
        else:
            y_tick_labels.append(f"10^{exp}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)

    # Legend at specified location
    ax.legend(ncol=2, fontsize=9, loc=legend_loc, framealpha=0.85)

    # Add annotation in top-left corner
    ax.text(
        0.02,
        0.98,
        f"old: τ-free; new: τ={tau:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        ha="left",
        va="top",
    )

    # Add hidden points warning if any were hidden
    if hide_floored and hidden_total > 0:
        ax.text(
            0.02,
            0.92,
            f"⚠ hidden {hidden_total} floored points (see logs)",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.6),
            ha="left",
            va="top",
        )

    # Save figure
    filename = f"unreachability_vs_rank_old_vs_new_{ensemble}_tau{tau:.3f}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved rank comparison (rescaled): {filepath}")
    return filepath


def plot_unreachability_three_criteria_vs_m(
    data: Dict[str, np.ndarray],
    ensemble: str,
    d: int,
    K: int,
    tau: float,
    outdir: str = ".",
    floor: float = settings.DISPLAY_FLOOR,
    trials: Optional[int] = None,
) -> str:
    """
    Plot P(unreachability) vs Krylov rank m for 3 criteria (overlay).

    Creates log-scale plot with:
    - X-axis: Krylov rank m
    - Y-axis: P(unreachability) (log scale)
    - 3 curves with floor-aware asymmetric error bars (spectral, old, Krylov)
    - Floor-masked connections for DISPLAY_FLOOR values

    Args:
        data: Output from monte_carlo_unreachability_vs_m
        ensemble: "GOE" or "GUE"
        d: Hilbert space dimension
        K: Number of Hamiltonians
        tau: Threshold (for spectral only)
        outdir: Output directory
        floor: Display floor for log plot masking
        trials: Total number of trials (nks × nst) for error bar computation

    Returns:
        Path to saved figure

    Styling:
    - Markers: 'o' (spectral), 's' (old), '^' (Krylov)
    - Colors: From default palette (C0, C1, C2)
    - Line style: '--' (dashed)
    - Legend: "Spectral overlap (τ=X)", "Old criterion", "Krylov rank"
    - Title: f"P(unreachability) vs Krylov rank m | {ensemble}, d={d}, K={K}"
    - Filename: f"unreachability_vs_rank_three_{ensemble}_d{d}_K{K}_tau{tau:.2f}.png"
    """
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    m = data["m"]

    # Define styling for each criterion
    styles = {
        "spectral": {"marker": "o", "color": "C0", "label": f"Spectral overlap (τ={tau:.2f})"},
        "moment": {"marker": "s", "color": "C1", "label": "Moment criterion"},
        "krylov": {"marker": "^", "color": "C2", "label": "Krylov rank"},
    }

    # Plot each criterion if present
    for criterion, style in styles.items():
        p_key = f"p_{criterion}"
        err_key = f"err_{criterion}"

        if p_key in data and err_key in data:
            p = data[p_key]

            # Plot line with markers (no error bars yet)
            ax.plot(
                m,
                p,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle="--",
                markersize=6,
                linewidth=1.5,
            )

            # Add asymmetric Wilson error bars if trials is provided
            if trials is not None and trials > 0:
                err_lower, err_upper = _compute_asymmetric_errorbars(p, trials, floor=floor)
                yerr = np.vstack([err_lower, err_upper])

                # Only plot error bars where they are non-zero
                has_errbar = (err_lower > 0) | (err_upper > 0)
                if np.any(has_errbar):
                    ax.errorbar(
                        m[has_errbar],
                        p[has_errbar],
                        yerr=yerr[:, has_errbar],
                        fmt="none",
                        color=style["color"],
                        capsize=3,
                        linewidth=1,
                        alpha=0.7,
                    )

    ax.set_yscale("log")
    ax.set_xlabel("Krylov rank $m$", fontsize=12)
    ax.set_ylabel(r"$\log_{10} P(\mathrm{unreachable})$", fontsize=12)
    ax.set_title(
        f"P(unreachability) vs Krylov rank $m$ | {ensemble}, $d={d}$, $K={K}$", fontsize=13
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation box with parameters
    annotation_text = f"Ensemble: {ensemble}\n$d={d}$, $K={K}$\n$\\tau={tau:.2f}$ (spectral only)"
    ax.text(
        0.98,
        0.02,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        ha="right",
        va="bottom",
    )

    # Save figure
    filename = f"unreachability_vs_rank_three_{ensemble}_d{d}_K{K}_tau{tau:.2f}.png"
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved three-criteria rank sweep: {filepath}")
    return filepath


def plot_unreachability_three_criteria_vs_K(
    data: Dict[str, np.ndarray],
    ensemble: str,
    d: int,
    tau: float,
    outdir: str = ".",
    floor: float = settings.DISPLAY_FLOOR,
    trials: Optional[int] = None,
) -> str:
    """
    Plot P(unreachability) vs K for 3 criteria (replica of single-d K-sweep).

    Similar to plot_unreachability_three_criteria_vs_m, but:
    - X-axis: K (number of Hamiltonians)
    - Annotation includes Krylov m strategy from data['m_label']
    - Filename: f"unreachability_vs_k_three_{ensemble}_d{d}_tau{tau:.2f}.png"

    Args:
        data: Output from monte_carlo_unreachability_vs_K_three
        ensemble: "GOE" or "GUE"
        d: Hilbert space dimension
        tau: Threshold (for spectral only)
        outdir: Output directory
        floor: Display floor for log plot masking
        trials: Total number of trials (nks × nst) for error bar computation

    Returns:
        Path to saved figure
    """
    ensure_dir(outdir)

    fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)

    k = data["k"]
    m_label = data.get("m_label", "m = K")

    # Define styling for each criterion
    styles = {
        "spectral": {"marker": "o", "color": "C0", "label": f"Spectral overlap (τ={tau:.2f})"},
        "moment": {"marker": "s", "color": "C1", "label": "Moment criterion"},
        "krylov": {"marker": "^", "color": "C2", "label": f"Krylov rank ({m_label})"},
    }

    # Plot each criterion if present
    for criterion, style in styles.items():
        p_key = f"p_{criterion}"
        err_key = f"err_{criterion}"

        if p_key in data and err_key in data:
            p = data[p_key]

            # Plot line with markers (no error bars yet)
            ax.plot(
                k,
                p,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linestyle="--",
                markersize=6,
                linewidth=1.5,
            )

            # Add asymmetric Wilson error bars if trials is provided
            if trials is not None and trials > 0:
                err_lower, err_upper = _compute_asymmetric_errorbars(p, trials, floor=floor)
                yerr = np.vstack([err_lower, err_upper])

                # Only plot error bars where they are non-zero
                has_errbar = (err_lower > 0) | (err_upper > 0)
                if np.any(has_errbar):
                    ax.errorbar(
                        k[has_errbar],
                        p[has_errbar],
                        yerr=yerr[:, has_errbar],
                        fmt="none",
                        color=style["color"],
                        capsize=3,
                        linewidth=1,
                        alpha=0.7,
                    )

    ax.set_yscale("log")
    ax.set_xlabel("Number of Hamiltonians $K$", fontsize=12)
    ax.set_ylabel(r"$\log_{10} P(\mathrm{unreachable})$", fontsize=12)
    ax.set_title(f"P(unreachability) vs $K$ | {ensemble}, $d={d}$", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation box with parameters
    annotation_text = f"Ensemble: {ensemble}\n$d={d}$\n$\\tau={tau:.2f}$ (spectral only)\nKrylov: {m_label}"
    ax.text(
        0.98,
        0.02,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        ha="right",
        va="bottom",
    )

    # Save figure
    filename = f"unreachability_vs_k_three_{ensemble}_d{d}_tau{tau:.2f}.png"
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=settings.DEFAULT_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved three-criteria K sweep: {filepath}")
    return filepath


def plot_unreachability_three_criteria_vs_density(
    data: Dict[Any, Any],
    ensemble: str,
    outdir: str = ".",
    floor: float = settings.DISPLAY_FLOOR,
    trials: Optional[int] = None,
    y_axis: str = "unreachable",
) -> List[str]:
    """
    Plot P(unreachability) vs density ρ=K/d² for 3 criteria across multiple dimensions.

    Publication-ready single-axes plot per τ: all criteria and all dimensions overlaid,
    distinguished by line style (criteria) and color (dimensions).

    Floor handling: Points at the display floor are shown as faded markers without
    connecting line segments, preventing misleading vertical "cliff" drops.

    Args:
        data: Output from monte_carlo_unreachability_vs_density
        ensemble: "GOE" or "GUE"
        outdir: Output directory
        floor: Display floor for log plot masking
        trials: Total number of trials (for error bar computation)
        y_axis: "unreachable" (default) or "reachable" (plots 1 - p_unreach)

    Returns:
        List of paths to saved figures (one per τ)

    Styling:
        - Single axes per τ
        - Figure size: 14×10 inches, DPI 200
        - X-axis: K/d² (normalized control density ρ)
        - Y-axis: log₁₀ P (unreachable or reachable depending on y_axis)
        - Criteria distinguished by line style/marker
        - d values distinguished by color
        - Legend: "Spectral (τ=0.95) • d=20", "Old • d=30", etc.
        - Filename: three_criteria_vs_density_{ensemble}_tau{tau:.2f}_{y_axis}.png
    """
    ensure_dir(outdir)

    dims = data["dims"]
    taus = data["taus"]
    saved_paths = []

    # Define colors for each dimension (use distinct colors)
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    dim_colors = {d: color_palette[i % len(color_palette)] for i, d in enumerate(dims)}

    # Define line styles and markers for each criterion
    criterion_styles = {
        "spectral": {"linestyle": "-", "marker": "o", "markersize": 6},
        "moment": {"linestyle": "--", "marker": "s", "markersize": 6},
        "krylov": {"linestyle": ":", "marker": "^", "markersize": 6},
    }
    criterion_labels = {
        "spectral": "Spectral",
        "moment": "Moment",
        "krylov": "Krylov (m=min(K,d))",
    }

    for tau in taus:
        # Publication-ready figure size: 14×10 inches, DPI 200
        fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

        criteria = ["spectral", "moment", "krylov"]

        # Plot all combinations of (criterion, d)
        for criterion in criteria:
            for d in dims:
                key = (d, tau, criterion)
                if key not in data:
                    continue

                result = data[key]
                rho = result["rho"]
                p = result["p"]

                # Convert to reachable if requested
                if y_axis == "reachable":
                    p = 1.0 - p
                    # Floor handling for reachable: p_reach near 1 → don't floor
                    p = np.maximum(p, floor)

                # Get color and style
                color = dim_colors[d]
                style = criterion_styles[criterion]

                # Build legend label
                if criterion == "spectral":
                    label = f"{criterion_labels[criterion]} (τ={tau:.2f}) • d={d}"
                else:
                    label = f"{criterion_labels[criterion]} • d={d}"

                # Detect floor-clipped points
                is_floored = np.abs(p - floor) < floor * 0.01

                # Plot non-floored points with connecting lines (using masked array)
                p_masked = _create_floor_masked_array(p, floor)
                ax.plot(
                    rho,
                    p_masked,
                    marker=style["marker"],
                    color=color,
                    label=label,
                    linestyle=style["linestyle"],
                    markersize=style["markersize"],
                    linewidth=2.0,
                    alpha=0.9,
                    markeredgewidth=0.5,
                    markeredgecolor='white',
                )

                # Plot floored points separately as faded markers (no connecting lines)
                if np.any(is_floored):
                    ax.plot(
                        rho[is_floored],
                        p[is_floored],
                        marker=style["marker"],
                        color=color,
                        linestyle='none',
                        markersize=style["markersize"],
                        alpha=0.3,  # Faded
                        markeredgewidth=0,
                    )

                # Add asymmetric Wilson error bars if trials is provided
                if trials is not None and trials > 0:
                    p_orig = result["p"]  # Always use unreachable for error bars
                    err_lower, err_upper = _compute_asymmetric_errorbars(
                        p_orig, trials, floor=floor
                    )

                    # If y_axis is reachable, transform error bars
                    if y_axis == "reachable":
                        # For p_reach = 1 - p_unreach, errors flip
                        err_lower, err_upper = err_upper, err_lower

                    yerr = np.vstack([err_lower, err_upper])

                    # Only plot error bars where they are non-zero
                    has_errbar = (err_lower > 0) | (err_upper > 0)
                    if np.any(has_errbar):
                        ax.errorbar(
                            rho[has_errbar],
                            p[has_errbar],
                            yerr=yerr[:, has_errbar],
                            fmt="none",
                            color=color,
                            capsize=2,
                            linewidth=1,
                            alpha=0.5,
                        )

        ax.set_yscale("log")
        ax.set_xlabel(r"$K/d^2$", fontsize=16, fontweight='bold')

        if y_axis == "reachable":
            ax.set_ylabel(r"$\log_{10} P(\mathrm{reachable})$", fontsize=16, fontweight='bold')
            title_y = "P(reachable)"
        else:
            ax.set_ylabel(r"$\log_{10} P(\mathrm{unreachable})$", fontsize=16, fontweight='bold')
            title_y = "P(unreachable)"

        ax.set_title(
            f"{title_y} vs density $K/d^2$ | {ensemble}, $\\tau={tau:.2f}$",
            fontsize=18,
            fontweight='bold',
            pad=20,
        )
        ax.legend(loc="best", fontsize=11, ncol=2, framealpha=0.95,
                  edgecolor='gray', fancybox=True)
        ax.grid(True, alpha=0.25, linewidth=0.5, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=13)

        # Add floor annotation if any points are floored
        floor_hit = False
        for criterion in criteria:
            for d in dims:
                key = (d, tau, criterion)
                if key in data:
                    p_check = data[key]["p"]
                    if np.any(np.abs(p_check - floor) < floor * 0.01):
                        floor_hit = True
                        break
            if floor_hit:
                break

        if floor_hit:
            ax.text(0.02, 0.02, f"Display floor: {floor:.0e}",
                   transform=ax.transAxes, fontsize=10, style='italic',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Save figure with higher DPI
        filename = f"three_criteria_vs_density_{ensemble}_tau{tau:.2f}_{y_axis}.png"
        filepath = os.path.join(outdir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved density plot for τ={tau:.2f}: {filepath}")
        saved_paths.append(filepath)

    return saved_paths


def plot_unreachability_K_multi_tau(
    data: Dict[Any, Any],
    ensemble: str,
    outdir: str = ".",
    floor: float = settings.DISPLAY_FLOOR,
    trials: Optional[int] = None,
    y_type: str = "unreachable",
) -> str:
    """
    Plot P(unreachability) vs K with multiple τ for spectral (gradient) + old + krylov.

    Single-axes plot showing 5 curves:
    - Spectral at 3 τ values (light→dark gradient of same base color)
    - Old criterion
    - Krylov criterion

    Floor-aware plotting:
    - Points at display floor are NOT connected by lines (prevents vertical "cliffs")
    - Floored points shown as faded markers (alpha=0.3)
    - Line segments are broken at floor using masked arrays

    Args:
        data: Output from monte_carlo_unreachability_vs_K_multi_tau
        ensemble: "GOE" or "GUE"
        outdir: Output directory
        floor: Display floor for log plot masking
        trials: Total number of trials (for error bar computation)
        y_type: "unreachable" or "reachable" (plots P(unreachable) or P(reachable))

    Returns:
        Path to saved figure

    Styling:
        - Single axes, publication-ready (14×10 inches, DPI 200)
        - X-axis: K (number of Hamiltonians)
        - Y-axis: log₁₀ P(unreachable) or log₁₀ P(reachable)
        - Spectral: gradient from light to dark for increasing τ
        - Old: dashed line, distinct color
        - Krylov: dotted line, distinct color
    """
    ensure_dir(outdir)

    k = data["k"]
    taus = data["taus"]
    d = data["d"]

    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

    # Define base color for spectral (use blue gradient)
    spectral_base = np.array([0.2, 0.4, 0.8])  # Blue RGB
    # Create gradient: light to dark for increasing tau
    if len(taus) == 1:
        spectral_colors = [spectral_base]  # Single color for single tau
    else:
        spectral_colors = [spectral_base * (0.4 + 0.6 * (i / (len(taus) - 1))) for i in range(len(taus))]

    # Plot spectral for each tau with gradient (floor-aware)
    for i, tau in enumerate(taus):
        key = (tau, "spectral")
        result = data[key]
        p_unreach = result["p"]

        # Transform for reachable if needed
        if y_type == "reachable":
            p = 1.0 - p_unreach
            # Clip to floor for numerical stability
            p = np.maximum(p, floor)
        else:
            p = p_unreach

        color = spectral_colors[i]

        label = f"Spectral (τ={tau:.2f})"

        # Detect floor-clipped points
        is_floored = np.abs(p - floor) < floor * 0.01

        # Plot non-floored points with masked array (breaks line segments at floor)
        p_masked = _create_floor_masked_array(p, floor)
        ax.plot(
            k,
            p_masked,
            marker="o",
            color=color,
            label=label,
            linestyle="-",
            markersize=6,
            linewidth=2.0,
            alpha=0.9,
        )

        # Plot floored points as faded markers (no connecting lines)
        if np.any(is_floored):
            ax.plot(
                k[is_floored],
                p[is_floored],
                marker="o",
                color=color,
                linestyle="none",
                markersize=6,
                alpha=0.3,
            )

        # Add error bars (only for non-floored points)
        if trials is not None and trials > 0:
            err_lower, err_upper = _compute_asymmetric_errorbars(p, trials, floor=floor)
            yerr = np.vstack([err_lower, err_upper])

            has_errbar = (err_lower > 0) | (err_upper > 0) & ~is_floored
            if np.any(has_errbar):
                ax.errorbar(
                    k[has_errbar],
                    p[has_errbar],
                    yerr=yerr[:, has_errbar],
                    fmt="none",
                    color=color,
                    capsize=3,
                    linewidth=1.2,
                    alpha=0.6,
                )

    # Plot moment criterion (floor-aware)
    p_moment_unreach = data["moment"]["p"]

    # Transform for reachable if needed
    if y_type == "reachable":
        p_moment = 1.0 - p_moment_unreach
        p_moment = np.maximum(p_moment, floor)
    else:
        p_moment = p_moment_unreach

    is_floored_moment = np.abs(p_moment - floor) < floor * 0.01

    # Plot non-floored with masked array
    p_moment_masked = _create_floor_masked_array(p_moment, floor)
    ax.plot(
        k,
        p_moment_masked,
        marker="s",
        color="C1",
        label="Moment criterion",
        linestyle="--",
        markersize=6,
        linewidth=2.0,
        alpha=0.9,
    )

    # Plot floored points as faded markers
    if np.any(is_floored_moment):
        ax.plot(
            k[is_floored_moment],
            p_moment[is_floored_moment],
            marker="s",
            color="C1",
            linestyle="none",
            markersize=6,
            alpha=0.3,
        )

    if trials is not None and trials > 0:
        err_lower, err_upper = _compute_asymmetric_errorbars(p_moment, trials, floor=floor)
        yerr = np.vstack([err_lower, err_upper])
        has_errbar = (err_lower > 0) | (err_upper > 0) & ~is_floored_moment
        if np.any(has_errbar):
            ax.errorbar(
                k[has_errbar],
                p_moment[has_errbar],
                yerr=yerr[:, has_errbar],
                fmt="none",
                color="C1",
                capsize=3,
                linewidth=1.2,
                alpha=0.6,
            )

    # Plot krylov criterion (floor-aware)
    p_krylov_unreach = data["krylov"]["p"]

    # Transform for reachable if needed
    if y_type == "reachable":
        p_krylov = 1.0 - p_krylov_unreach
        p_krylov = np.maximum(p_krylov, floor)
    else:
        p_krylov = p_krylov_unreach

    is_floored_krylov = np.abs(p_krylov - floor) < floor * 0.01

    # Plot non-floored with masked array
    p_krylov_masked = _create_floor_masked_array(p_krylov, floor)
    ax.plot(
        k,
        p_krylov_masked,
        marker="^",
        color="C2",
        label="Krylov (m=min(K,d))",
        linestyle=":",
        markersize=6,
        linewidth=2.0,
        alpha=0.9,
    )

    # Plot floored points as faded markers
    if np.any(is_floored_krylov):
        ax.plot(
            k[is_floored_krylov],
            p_krylov[is_floored_krylov],
            marker="^",
            color="C2",
            linestyle="none",
            markersize=6,
            alpha=0.3,
        )

    if trials is not None and trials > 0:
        err_lower, err_upper = _compute_asymmetric_errorbars(p_krylov, trials, floor=floor)
        yerr = np.vstack([err_lower, err_upper])
        has_errbar = (err_lower > 0) | (err_upper > 0) & ~is_floored_krylov
        if np.any(has_errbar):
            ax.errorbar(
                k[has_errbar],
                p_krylov[has_errbar],
                yerr=yerr[:, has_errbar],
                fmt="none",
                color="C2",
                capsize=3,
                linewidth=1.2,
                alpha=0.6,
            )

    # Enhanced axis formatting
    ax.set_yscale("log")
    ax.set_xlabel("Number of Hamiltonians $K$", fontsize=16, fontweight='bold')

    # Set labels based on y_type
    if y_type == "reachable":
        ax.set_ylabel(r"$\log_{10} P(\mathrm{reachable})$", fontsize=16, fontweight='bold')
        ax.set_title(
            f"P(reachable) vs $K$ | {ensemble}, $d={d}$",
            fontsize=18,
            fontweight='bold',
        )
    else:
        ax.set_ylabel(r"$\log_{10} P(\mathrm{unreachable})$", fontsize=16, fontweight='bold')
        ax.set_title(
            f"P(unreachable) vs $K$ | {ensemble}, $d={d}$",
            fontsize=18,
            fontweight='bold',
        )
    ax.legend(loc="best", fontsize=12, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25, which='both')
    ax.tick_params(labelsize=14)

    # Add floor annotation if any curve hits the floor
    # Check if any points across all curves hit the floor
    any_floored = is_floored_moment.any() or is_floored_krylov.any()
    # Also check spectral curves
    for i, tau in enumerate(taus):
        key = (tau, "spectral")
        p = data[key]["p"]
        if np.any(np.abs(p - floor) < floor * 0.01):
            any_floored = True
            break

    if any_floored:
        ax.text(
            0.02, 0.02,
            f"Display floor: {floor:.0e}\n(faded markers show floored values)",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

    # Save figure at higher DPI
    tau_str = "_".join([f"{t:.2f}" for t in taus])
    filename = f"K_sweep_multi_tau_{ensemble}_d{d}_taus{tau_str}_{y_type}.png"
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved K-sweep multi-tau plot ({y_type}): {filepath}")
    return filepath


# ==============================================================================
# CSV-BASED PLOTTING FUNCTIONS (merged from viz_csv.py)
# ==============================================================================
# These functions enable plotting from partial/incomplete CSV files for
# resumable workflows and mid-run progress visualization.

# Required dimensions for density plots (matches CLI validation)
REQUIRED_DENSITY_DIMS = {20, 30, 40, 50}


def _load_and_filter_csv(
    csv_path: str,
    ensemble: str,
    taus: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Load CSV and filter by ensemble and optional tau values.

    Args:
        csv_path: Path to CSV file
        ensemble: Filter by ensemble ("GOE", "GUE", or "GEO2")
        taus: Optional list of tau values to filter (default: all)

    Returns:
        Filtered DataFrame

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If CSV is empty, missing required columns, or contains mixed ensembles
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Check for mixed ensembles before filtering
    unique_ensembles = df["ensemble"].unique()
    if len(unique_ensembles) > 1:
        raise ValueError(
            f"CSV contains mixed ensembles: {list(unique_ensembles)}. "
            f"Please use separate CSVs for each ensemble or filter manually. "
            f"Requested ensemble: {ensemble}"
        )

    # Filter by ensemble
    df = df[df["ensemble"] == ensemble].copy()

    if df.empty:
        raise ValueError(
            f"No data found for ensemble={ensemble} in {csv_path}. "
            f"Available ensemble: {unique_ensembles[0] if len(unique_ensembles) > 0 else 'none'}. "
            f"Ensure --ensemble matches the CSV data."
        )

    # Check for mixed dimensions (warn for GEO2)
    unique_dims = df["d"].unique()
    if ensemble == "GEO2" and len(unique_dims) > 1:
        logger.warning(
            f"CSV contains multiple dimensions for GEO2: {sorted(unique_dims)}. "
            f"Ensure consistent lattice parameters (--nx, --ny) for proper plotting."
        )

    # Filter by tau if specified
    if taus is not None:
        # For tau filtering, keep rows where tau matches OR tau is empty (for moment/krylov)
        mask = df["tau"].isin(taus) | (df["tau"] == "") | df["tau"].isna()
        df = df[mask].copy()

    logger.info(f"Loaded {len(df)} rows from {csv_path} (ensemble={ensemble})")
    return df


def plot_density_from_csv(
    csv_path: str,
    ensemble: str,
    y_axis: str = "unreachable",
    outdir: str = "fig/comparison/",
    taus: Optional[List[float]] = None,
) -> List[str]:
    """
    Generate density plots from CSV data.

    Reads CSV and generates one plot per tau value, showing:
    - All dimensions (20, 30, 40, 50) with different colors
    - All three criteria (spectral, old, krylov) with different line styles
    - Floor-aware rendering (faded markers at floor, broken lines)

    Args:
        csv_path: Path to CSV file (e.g., "fig_summary/density_gue.csv")
        ensemble: Random matrix ensemble ("GOE" or "GUE")
        y_axis: "unreachable" or "reachable"
        outdir: Output directory for plots
        taus: Optional list of tau values to plot (default: all in CSV)

    Returns:
        List of generated file paths

    Raises:
        ValueError: If required dimensions are missing or data is invalid
    """
    df = _load_and_filter_csv(csv_path, ensemble, taus)

    if df.empty:
        logger.warning("No data to plot")
        return []

    # Get unique tau values from spectral criterion
    df_spectral = df[df["criterion"] == "spectral"]
    available_taus = sorted(df_spectral["tau"].dropna().unique())

    if not available_taus:
        logger.warning("No spectral tau values found in CSV")
        return []

    # Check dimensions
    available_dims = sorted(df["d"].unique())
    missing_dims = REQUIRED_DENSITY_DIMS - set(available_dims)
    if missing_dims:
        logger.warning(
            f"Missing required dimensions: {sorted(missing_dims)}. "
            f"Only plotting available dimensions: {available_dims}"
        )

    # Use available dimensions that are in required set
    plot_dims = sorted(set(available_dims) & REQUIRED_DENSITY_DIMS)

    if not plot_dims:
        raise ValueError(
            f"No valid dimensions found. Required: {sorted(REQUIRED_DENSITY_DIMS)}, "
            f"Found: {sorted(available_dims)}"
        )

    # Generate plot for each tau
    filepaths = []
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for tau in available_taus:
        filepath = _plot_density_single_tau(
            df, tau, ensemble, plot_dims, y_axis, outdir
        )
        filepaths.append(filepath)

    return filepaths


def _plot_density_single_tau(
    df: pd.DataFrame,
    tau: float,
    ensemble: str,
    dims: List[int],
    y_axis: str,
    outdir: str,
) -> str:
    """Generate a single density plot for given tau."""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

    # Colors for dimensions
    colors = {20: "#1f77b4", 30: "#ff7f0e", 40: "#2ca02c", 50: "#d62728"}

    # Line styles for criteria
    styles = {
        "spectral": {"linestyle": "-", "marker": "o", "markersize": 6},
        "moment": {"linestyle": "--", "marker": "s", "markersize": 5},
        "krylov": {"linestyle": ":", "marker": "^", "markersize": 5},
    }

    floor = settings.DISPLAY_FLOOR

    # Plot each criterion and dimension
    for criterion in ["spectral", "moment", "krylov"]:
        for d in dims:
            # Filter data
            if criterion == "spectral":
                mask = (df["criterion"] == criterion) & (df["tau"] == tau) & (df["d"] == d)
            else:
                mask = (df["criterion"] == criterion) & (df["d"] == d)

            data = df[mask].sort_values("rho_K_over_d2")

            if data.empty:
                logger.debug(f"No data for {criterion}, tau={tau}, d={d}")
                continue

            rho = data["rho_K_over_d2"].values
            p = data["p_unreach"].values

            # Transform for reachable if needed
            if y_axis == "reachable":
                p = 1.0 - p

            # Convert to log scale
            p_log = np.log10(np.maximum(p, floor))

            # Mask floor values
            p_masked = _create_floor_masked_array(p_log, np.log10(floor))

            # Plot line
            label = _format_legend_label(criterion, tau, d)
            ax.plot(
                rho,
                p_masked,
                label=label,
                color=colors[d],
                linewidth=2,
                **styles[criterion],
            )

            # Plot floored points as faded markers
            is_floored = np.abs(p_log - np.log10(floor)) < np.log10(floor) * 0.01
            if np.any(is_floored):
                ax.plot(
                    rho[is_floored],
                    p_log[is_floored],
                    marker=styles[criterion]["marker"],
                    markersize=styles[criterion]["markersize"],
                    color=colors[d],
                    alpha=0.3,
                    linestyle="none",
                )

    # Labels and styling
    y_label = f"P({'reachable' if y_axis == 'reachable' else 'unreachable'})"
    ax.set_xlabel("K/d²", fontsize=16, fontweight="bold")
    ax.set_ylabel(f"log₁₀ {y_label}", fontsize=16, fontweight="bold")
    ax.set_title(
        f"{y_label} vs density K/d² | {ensemble}, τ = {tau:.2f}",
        fontsize=18,
        fontweight="bold",
    )

    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Add floor annotation
    ax.text(
        0.02,
        0.02,
        f"Display floor: {floor:.0e}",
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.6,
    )

    plt.tight_layout()

    # Save
    filename = f"three_criteria_vs_density_{ensemble}_tau{tau:.2f}_{y_axis}.png"
    filepath = str(Path(outdir) / filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved density plot: {filepath}")
    return filepath


def plot_k_multi_tau_from_csv(
    csv_path: str,
    ensemble: str,
    y_type: str = "unreachable",
    outdir: str = "fig/comparison/",
    taus: Optional[List[float]] = None,
) -> List[str]:
    """
    Generate K-sweep multi-tau plots from CSV data.

    Reads CSV and generates plots showing:
    - Spectral criterion at multiple tau values (blue gradient)
    - Old and Krylov criteria (tau-independent)
    - Floor-aware rendering

    Args:
        csv_path: Path to CSV file (e.g., "fig_summary/k30_gue.csv")
        ensemble: Random matrix ensemble ("GOE" or "GUE")
        y_type: "unreachable" or "reachable"
        outdir: Output directory for plots
        taus: Optional list of tau values to plot (default: all in CSV)

    Returns:
        List of generated file paths
    """
    df = _load_and_filter_csv(csv_path, ensemble, taus)

    if df.empty:
        logger.warning("No data to plot")
        return []

    # Get unique d value (should be single dimension for K-sweep)
    dims = df["d"].unique()
    if len(dims) != 1:
        logger.warning(f"Expected single dimension, found {len(dims)}: {dims}")
        d = int(dims[0])  # Use first dimension
    else:
        d = int(dims[0])

    # Get tau values from spectral criterion
    df_spectral = df[df["criterion"] == "spectral"]
    available_taus = sorted(df_spectral["tau"].dropna().unique())

    if not available_taus:
        logger.warning("No spectral tau values found")
        return []

    # Generate plot
    Path(outdir).mkdir(parents=True, exist_ok=True)
    filepath = _plot_k_multi_tau_single(
        df, d, available_taus, ensemble, y_type, outdir
    )

    return [filepath]


def _plot_k_multi_tau_single(
    df: pd.DataFrame,
    d: int,
    taus: List[float],
    ensemble: str,
    y_type: str,
    outdir: str,
) -> str:
    """Generate a single K-sweep multi-tau plot."""
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)

    floor = settings.DISPLAY_FLOOR

    # Blue gradient for spectral tau values
    n_taus = len(taus)
    blues = plt.cm.Blues(np.linspace(0.4, 0.9, n_taus))

    # Plot spectral for each tau
    for i, tau in enumerate(taus):
        mask = (df["criterion"] == "spectral") & (df["tau"] == tau)
        data = df[mask].sort_values("K")

        if data.empty:
            logger.debug(f"No data for spectral, tau={tau}")
            continue

        K = data["K"].values
        p = data["p_unreach"].values

        # Transform for reachable if needed
        if y_type == "reachable":
            p = 1.0 - p

        # Convert to log scale
        p_log = np.log10(np.maximum(p, floor))

        # Mask floor values
        p_masked = _create_floor_masked_array(p_log, np.log10(floor))

        # Plot line
        label = f"Spectral (τ={tau:.2f})"
        ax.plot(
            K,
            p_masked,
            label=label,
            color=blues[i],
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=6,
        )

        # Plot floored points
        is_floored = np.abs(p_log - np.log10(floor)) < np.log10(floor) * 0.01
        if np.any(is_floored):
            ax.plot(
                K[is_floored],
                p_log[is_floored],
                marker="o",
                markersize=6,
                color=blues[i],
                alpha=0.3,
                linestyle="none",
            )

    # Plot moment criterion
    mask = df["criterion"] == "moment"
    data = df[mask].sort_values("K")
    if not data.empty:
        K = data["K"].values
        p = data["p_unreach"].values

        if y_type == "reachable":
            p = 1.0 - p

        p_log = np.log10(np.maximum(p, floor))
        p_masked = _create_floor_masked_array(p_log, np.log10(floor))

        ax.plot(
            K,
            p_masked,
            label="Moment criterion",
            color="purple",
            linestyle="--",
            linewidth=2,
            marker="s",
            markersize=5,
        )

        is_floored = np.abs(p_log - np.log10(floor)) < np.log10(floor) * 0.01
        if np.any(is_floored):
            ax.plot(
                K[is_floored],
                p_log[is_floored],
                marker="s",
                markersize=5,
                color="purple",
                alpha=0.3,
                linestyle="none",
            )

    # Plot krylov criterion
    mask = df["criterion"] == "krylov"
    data = df[mask].sort_values("K")
    if not data.empty:
        K = data["K"].values
        p = data["p_unreach"].values

        if y_type == "reachable":
            p = 1.0 - p

        p_log = np.log10(np.maximum(p, floor))
        p_masked = _create_floor_masked_array(p_log, np.log10(floor))

        ax.plot(
            K,
            p_masked,
            label="Krylov (m=min(K,d))",
            color="orange",
            linestyle=":",
            linewidth=2,
            marker="^",
            markersize=5,
        )

        is_floored = np.abs(p_log - np.log10(floor)) < np.log10(floor) * 0.01
        if np.any(is_floored):
            ax.plot(
                K[is_floored],
                p_log[is_floored],
                marker="^",
                markersize=5,
                color="orange",
                alpha=0.3,
                linestyle="none",
            )

    # Labels and styling
    y_label = f"P({'reachable' if y_type == 'reachable' else 'unreachability'})"
    ax.set_xlabel("Number of Hamiltonians K", fontsize=16, fontweight="bold")
    ax.set_ylabel(f"log₁₀ {y_label}", fontsize=16, fontweight="bold")
    ax.set_title(
        f"{y_label} vs K | {ensemble}, d = {d}",
        fontsize=18,
        fontweight="bold",
    )

    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Add floor annotation
    ax.text(
        0.02,
        0.02,
        f"Display floor: {floor:.0e}",
        transform=ax.transAxes,
        fontsize=10,
        alpha=0.6,
    )

    plt.tight_layout()

    # Save
    tau_str = "_".join(f"{t:.2f}" for t in taus)
    filename = f"K_sweep_multi_tau_{ensemble}_d{d}_taus{tau_str}_{y_type}.png"
    filepath = str(Path(outdir) / filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved K-sweep plot: {filepath}")
    return filepath


def _format_legend_label(criterion: str, tau: float, d: int) -> str:
    """Format legend label for density plots."""
    if criterion == "spectral":
        return f"Spectral (τ={tau:.2f}) • d={d}"
    elif criterion == "moment":
        return f"Moment • d={d}"
    elif criterion == "krylov":
        return f"Krylov (m=min(K,d)) • d={d}"
    else:
        return f"{criterion} • d={d}"
