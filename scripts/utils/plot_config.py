"""
Consistent plot configuration for all reachability figures.

IMPORTANT: DO NOT use artificial clipping (e.g., np.clip(P, epsilon, 1-epsilon))
for linearized fits. See docs/clipping_methodology.md and CLAUDE.md for details.

Physical data treatment:
- P=0 and P=1 are genuine binomial outcomes
- Use Wilson score intervals for proper uncertainty
- Fit only transition regions (0 < k < N)
- Report explicit quality metrics (N_trans, frac_trans, quality flags)
"""

# =============================================================================
# Color Schemes (ColorBrewer Set1, colorblind-safe)
# =============================================================================

# Consistent color map for dimensions
DIMENSION_COLORS = {
    8: '#a65628',   # Brown (legacy, if needed)
    10: '#e41a1c',  # Red
    12: '#984ea3',  # Purple (legacy, if needed)
    14: '#377eb8',  # Blue
    16: '#ff7f00',  # Orange (legacy, if needed)
    18: '#4daf4a',  # Green
    22: '#984ea3',  # Purple
    26: '#ff7f00',  # Orange
}

# Marker styles per criterion
CRITERION_MARKERS = {
    'moment': 'o',     # Circle
    'spectral': 's',   # Square
    'krylov': '^',     # Triangle
}

# Quality status colors (for linearized fits)
QUALITY_COLORS = {
    'good': '#2ca02c',          # Green
    'marginal': '#ff7f0e',      # Orange
    'insufficient': '#d62728',  # Red
}

# =============================================================================
# Typography
# =============================================================================

# Font sizes
FONTSIZE_TITLE = 18
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_LEGEND = 10
FONTSIZE_TEXTBOX = 11
FONTSIZE_TICK = 11

# Font weights
FONTWEIGHT_LABEL = 'bold'
FONTWEIGHT_TITLE = 'bold'

# =============================================================================
# Figure Dimensions
# =============================================================================

# Standard figure sizes (width, height) in inches
FIGSIZE_SINGLE_PANEL = (10, 7)
FIGSIZE_TWO_PANEL = (16, 7)
FIGSIZE_THREE_PANEL = (22, 7)
FIGSIZE_FOUR_PANEL = (22, 14)

# =============================================================================
# Data Point Styling
# =============================================================================

# Error bar styling for main data points
ERRORBAR_KWARGS = {
    'markersize': 8,
    'capsize': 4,
    'capthick': 2,
    'elinewidth': 2,
    'alpha': 0.85,
    'zorder': 3
}

# Boundary points (k=0 or k=N) - faded for linearized fits
BOUNDARY_POINT_KWARGS = {
    'markersize': 4,
    'alpha': 0.25,
    'capsize': 2,
    'elinewidth': 1,
    'zorder': 1
}

# Transition points (0 < k < N) - prominent for linearized fits
TRANSITION_POINT_KWARGS = {
    'markersize': 6,
    'alpha': 0.9,
    'capsize': 3,
    'elinewidth': 2,
    'zorder': 3
}

# =============================================================================
# Line Styling
# =============================================================================

# Fit lines (linear regression)
FIT_LINE_KWARGS = {
    'linestyle': '-',
    'linewidth': 2.5,
    'alpha': 0.8,
    'zorder': 2
}

# Reference lines (horizontal/vertical guides)
REFERENCE_LINE_KWARGS = {
    'color': 'gray',
    'linestyle': ':',
    'linewidth': 1.5,
    'alpha': 0.5,
    'zorder': 0
}

# Grid styling
GRID_KWARGS = {
    'alpha': 0.3,
    'linestyle': '-',
    'linewidth': 0.5
}

# =============================================================================
# Figure Export Settings
# =============================================================================

# Standard export (for quick review)
FIGURE_KWARGS = {
    'dpi': 150,
    'bbox_inches': 'tight',
    'facecolor': 'white'
}

# Publication quality (for papers/presentations)
PUBLICATION_KWARGS = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'facecolor': 'white'
}

# =============================================================================
# Quality Assessment (Linearized Fits)
# =============================================================================

# Thresholds for data quality classification
QUALITY_THRESHOLDS = {
    'good': {'n_trans_min': 10, 'frac_trans_min': 0.20},
    'marginal': {'n_trans_min': 5, 'frac_trans_min': 0.10},
    # Below marginal = insufficient
}

# Quality flag symbols
QUALITY_SYMBOLS = {
    'good': '',
    'marginal': '⚠',
    'insufficient': '⚠️'
}

# =============================================================================
# Helper Functions
# =============================================================================

def apply_axis_styling(ax, xlabel=None, ylabel=None, title=None):
    """
    Apply consistent axis styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style
    xlabel, ylabel, title : str, optional
        Axis labels and title
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_LABEL)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_LABEL)
    if title:
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight=FONTWEIGHT_TITLE)

    ax.tick_params(labelsize=FONTSIZE_TICK)
    ax.grid(**GRID_KWARGS)
    ax.legend(fontsize=FONTSIZE_LEGEND)


def classify_data_quality(n_trans, frac_trans):
    """
    Classify data quality based on number and fraction of transition points.

    Parameters
    ----------
    n_trans : int
        Number of transition points (0 < k < N)
    frac_trans : float
        Fraction of data in transition region

    Returns
    -------
    str : 'good', 'marginal', or 'insufficient'
    """
    if (n_trans >= QUALITY_THRESHOLDS['good']['n_trans_min'] and
        frac_trans >= QUALITY_THRESHOLDS['good']['frac_trans_min']):
        return 'good'
    elif (n_trans >= QUALITY_THRESHOLDS['marginal']['n_trans_min'] and
          frac_trans >= QUALITY_THRESHOLDS['marginal']['frac_trans_min']):
        return 'marginal'
    else:
        return 'insufficient'
