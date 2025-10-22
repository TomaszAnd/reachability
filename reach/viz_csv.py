"""
Plotting functions for reading CSV data and generating publication-ready plots.

This module enables:
- Plotting from partial/incomplete CSV files (useful during long runs)
- Resumable workflows (re-plot from existing data without recomputation)
- Mid-run progress visualization

All plots maintain floor-aware rendering and publication styling from viz.py.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import settings

logger = logging.getLogger(__name__)

# Required dimensions for density plots (matches CLI validation)
REQUIRED_DENSITY_DIMS = {20, 30, 40, 50}


def _create_floor_masked_array(y_values: np.ndarray, floor: float) -> np.ma.MaskedArray:
    """
    Create masked array that breaks line segments at floor values.

    This prevents vertical "cliff" lines when values hit display floor.

    Args:
        y_values: Array of probability values
        floor: Display floor threshold

    Returns:
        Masked array with floor values masked out
    """
    is_floored = np.abs(y_values - floor) < floor * 0.01
    return np.ma.masked_where(is_floored, y_values)


def _load_and_filter_csv(
    csv_path: str,
    ensemble: str,
    taus: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Load CSV and filter by ensemble and optional tau values.

    Args:
        csv_path: Path to CSV file
        ensemble: Filter by ensemble ("GOE" or "GUE")
        taus: Optional list of tau values to filter (default: all)

    Returns:
        Filtered DataFrame

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If CSV is empty or missing required columns
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    # Filter by ensemble
    df = df[df["ensemble"] == ensemble].copy()

    if df.empty:
        logger.warning(f"No data found for ensemble={ensemble}")
        return df

    # Filter by tau if specified
    if taus is not None:
        # For tau filtering, keep rows where tau matches OR tau is empty (for old/krylov)
        mask = df["tau"].isin(taus) | (df["tau"] == "") | df["tau"].isna()
        df = df[mask].copy()

    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def plot_density_from_csv(
    csv_path: str,
    ensemble: str,
    y_axis: str = "unreachable",
    outdir: str = "fig_summary/",
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
        "old": {"linestyle": "--", "marker": "s", "markersize": 5},
        "krylov": {"linestyle": ":", "marker": "^", "markersize": 5},
    }

    floor = settings.DISPLAY_FLOOR

    # Plot each criterion and dimension
    for criterion in ["spectral", "old", "krylov"]:
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
    outdir: str = "fig_summary/",
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

    # Plot old criterion
    mask = df["criterion"] == "old"
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
            label="Old criterion",
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
    elif criterion == "old":
        return f"Old • d={d}"
    elif criterion == "krylov":
        return f"Krylov (m=min(K,d)) • d={d}"
    else:
        return f"{criterion} • d={d}"
