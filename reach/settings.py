"""
Global settings and configuration for the reach package.

Pipeline Role:
**SINGLE SOURCE OF TRUTH** for all configuration, constants, and defaults.
All other modules import from here to ensure consistency and reproducibility.

Configuration Philosophy:
- Deterministic seeding (SEED=42) for full reproducibility
- Fast/full sampling modes for quick validation vs production quality
- Centralized constants eliminate magic numbers
- Easy tuning of experiment parameters without editing code

Key Configuration Sections:
1. Reproducibility: SEED, AUTO_TIDYUP
2. Optimization: DEFAULT_METHOD, DEFAULT_BOUNDS, convergence tolerances
3. Analysis: DEFAULT_TAU, FAST_SAMPLING, FULL_SAMPLING
4. Visualization: DEFAULT_DPI, colormaps, figure sizes, DISPLAY_FLOOR
5. Experiment Configs: FAST_DIMS, FAST_TAUS, RANK_DIMS, etc.

All sampling counts (nks, nst), iteration limits, and thresholds defined here.
"""

from typing import List, Tuple

# ============================================================================
# REPRODUCIBILITY SETTINGS
# ============================================================================

#: Global random seed for all stochastic operations
SEED: int = 42

#: Whether to disable QuTiP's automatic cleanup (for reproducibility)
AUTO_TIDYUP: bool = False

# ============================================================================
# OPTIMIZATION DEFAULTS
# ============================================================================

#: Default parameter bounds for optimization: λ ∈ [-1,1]^K
DEFAULT_BOUNDS: List[Tuple[float, float]] = [(-1.0, 1.0)]

#: Default optimization method
DEFAULT_METHOD: str = "L-BFGS-B"

#: Default number of optimization restarts
DEFAULT_RESTARTS: int = 2

#: Default maximum iterations per optimization
DEFAULT_MAXITER: int = 200

#: Default function tolerance for optimization convergence
DEFAULT_FTOL: float = 1e-8

# ----------------------------------------------------------------------------
# Ensemble-Specific Optimization Settings
# ----------------------------------------------------------------------------

#: GEO2-specific optimization settings
#: Previous "aggressive" settings (maxiter=20, restarts=1) caused 70% false-unreachable
#: rate on provably reachable targets. See scripts/audit/AUDIT_FINDINGS.md for details.
#: Settings below achieve ≤1% false-unreachable rate across d=16..32 (sweep 2026-01-27).
#: With analytical gradient (spectral_overlap_with_grad), these are ~5-11× faster than
#: the old finite-difference approach at equivalent maxiter.
GEO2_RESTARTS: int = 3      # 3 restarts needed for d=16 (K=13); d=32 works with 1
GEO2_MAXITER: int = 100     # 75 minimum, 100 for safety at d=32 with low K
GEO2_FTOL: float = 1e-6     # Tighter than before (1e-4) for reliable convergence

# ----------------------------------------------------------------------------
# Production Settings (for fair cross-ensemble comparison)
# ----------------------------------------------------------------------------
#: Production restarts (same for all ensembles)
PRODUCTION_RESTARTS: int = 3

#: Production maxiter base value (adapted by dimension)
PRODUCTION_MAXITER: int = 150

#: Production ftol
PRODUCTION_FTOL: float = 1e-6


def get_optimization_settings(ensemble: str) -> dict:
    """
    Get ensemble-specific optimization settings.

    Parameters
    ----------
    ensemble : str
        Ensemble type ('GUE', 'GOE', 'canonical', 'GEO2')

    Returns
    -------
    dict
        Dictionary with keys: 'restarts', 'maxiter', 'ftol', 'method', 'bounds'

    Notes
    -----
    GEO2 uses maxiter=100, restarts=3 based on sweep results (2026-01-27).
    With analytical gradient, this achieves 0% false-unreachable rate on
    provably reachable targets at d=16 (K=13) and d=32 (K=51).
    See scripts/audit/AUDIT_FINDINGS.md for full analysis.
    """
    if ensemble.upper() == 'GEO2':
        return {
            'restarts': GEO2_RESTARTS,
            'maxiter': GEO2_MAXITER,
            'ftol': GEO2_FTOL,
            'method': DEFAULT_METHOD,
            'bounds': DEFAULT_BOUNDS,
        }
    else:
        # GUE, GOE, canonical use default settings
        return {
            'restarts': DEFAULT_RESTARTS,
            'maxiter': DEFAULT_MAXITER,
            'ftol': DEFAULT_FTOL,
            'method': DEFAULT_METHOD,
            'bounds': DEFAULT_BOUNDS,
        }


def get_production_settings(d: int, ensemble: str = 'GEO2') -> dict:
    """
    Get production-quality optimization settings with dimension-adaptive maxiter.

    These settings ensure ≤1% false-unreachable rate across all tested
    configurations. Uses analytical gradient for speed.

    Parameters
    ----------
    d : int
        Hilbert space dimension
    ensemble : str
        Ensemble type

    Returns
    -------
    dict
        Dictionary with keys: 'restarts', 'maxiter', 'ftol', 'method', 'bounds'

    Notes
    -----
    Maxiter scales with dimension to accommodate larger K = O(d²) parameter spaces:
    - d ≤ 16: maxiter=100 (K ≤ 13 at ρ=0.05)
    - d ≤ 32: maxiter=100 (K ≤ 51 at ρ=0.05)
    - d ≤ 64: maxiter=150 (K ≤ 205 at ρ=0.05)
    - d > 64: maxiter=200 (safety margin for very large K)
    """
    if d <= 32:
        maxiter = 100
    elif d <= 64:
        maxiter = PRODUCTION_MAXITER  # 150
    else:
        maxiter = 200

    return {
        'restarts': PRODUCTION_RESTARTS,
        'maxiter': maxiter,
        'ftol': PRODUCTION_FTOL,
        'method': DEFAULT_METHOD,
        'bounds': DEFAULT_BOUNDS,
    }


# ============================================================================
# ANALYSIS DEFAULTS
# ============================================================================

#: Default threshold τ for unreachability classification
DEFAULT_TAU: float = 0.95

#: Fast mode sampling: (nks, nst) = (hamiltonians, states per hamiltonian)
FAST_SAMPLING: Tuple[int, int] = (80, 20)

#: Full mode sampling: (nks, nst) = (hamiltonians, states per hamiltonian)
FULL_SAMPLING: Tuple[int, int] = (150, 30)

#: Default grid size for landscape plots
DEFAULT_GRID_SIZE: int = 81

#: Number of target states for landscape analysis
DEFAULT_LANDSCAPE_TARGETS: int = 40

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

#: Display floor for log plots (values ≤ floor shown as isolated points)
DISPLAY_FLOOR: float = 1e-12

#: Default figure DPI for saved plots
DEFAULT_DPI: int = 150

#: Default colormap for landscapes and heatmaps
DEFAULT_COLORMAP: str = "viridis"

#: Default figure size for single plots
DEFAULT_FIGSIZE: Tuple[float, float] = (10.0, 7.0)

#: Default figure size for landscape plots
LANDSCAPE_FIGSIZE: Tuple[float, float] = (16.0, 7.0)

# ============================================================================
# MATHEMATICAL CONSTANTS
# ============================================================================

#: Tolerance for spectral overlap bounds checking
OVERLAP_TOLERANCE: float = 1e-6

#: Minimum condition number for eigendecomposition stability
MIN_CONDITION_NUMBER: float = 1e-12

#: Gaussian smoothing sigma for landscape interpolation
LANDSCAPE_SMOOTH_SIGMA: float = 1.0

#: Fine grid multiplier for landscape interpolation (81 → 120)
LANDSCAPE_FINE_MULTIPLIER: float = 1.5

# ============================================================================
# EXPERIMENT CONFIGURATION (FAST MODE DEFAULTS)
# ============================================================================

#: Fast mode dimensions for experiments
FAST_DIMS: List[int] = [6, 10, 14, 18]

#: Fast mode tau thresholds for experiments
FAST_TAUS: List[float] = [0.90, 0.95, 0.99]

#: Fast mode k values for experiments
FAST_K_VALUES: List[int] = [3, 4]

#: Fast mode iteration counts for convergence analysis
FAST_ITERS: List[int] = [10, 20, 50, 100]

#: Fast mode Hamiltonian sample count
FAST_NKS: int = 80

#: Fast mode target states per Hamiltonian
FAST_NST: int = 20

#: Landscape plot configuration (dimension, k) pairs
LANDSCAPE_CONFIG: List[Tuple[int, int]] = [(10, 3)]

#: Iteration sweep configuration (dimension, k) pairs for multi-curve plots
ITER_SWEEP_CONFIG: List[Tuple[int, int]] = [(6, 3), (10, 3), (14, 4)]

# ============================================================================
# BIG SUITES (for comprehensive multi-tau and large-scale analyses)
# ============================================================================

#: Big dimension suite for rank comparisons
RANK_DIMS: List[int] = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30]

#: Big dimension suite for other comprehensive analyses
BIG_DIMS_5: List[int] = [12, 16, 20, 24, 30]

#: Tau set for rank comparisons (3 values)
RANK_TAUS_3: List[float] = [0.90, 0.95, 0.99]

#: Tau set for threshold sensitivity analysis (5 values)
TAUS_5: List[float] = [0.90, 0.92, 0.95, 0.97, 0.99]

#: Sampling for big figures (tune only if too slow)
BIG_NKS: int = 60
BIG_NST: int = 15

#: Iter sweep iterations (reduced if runtime heavy)
ITER_SWEEP_ITERS: List[int] = [10, 20, 50]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

#: New organized output directories
FIG_COMPARISON_DIR: str = "fig/comparison"
FIG_SPECTRAL_DIR: str = "fig/spectral_criterion"
FIG_KRYLOV_DIR: str = "fig/krylov_criterion"
FIG_MOMENT_DIR: str = "fig/moment_criterion"

#: Backward compatibility (deprecated - use FIG_COMPARISON_DIR instead)
FIG_SUMMARY_DIR: str = FIG_COMPARISON_DIR

# ============================================================================
# KRYLOV CRITERION SETTINGS
# ============================================================================

#: Breakdown tolerance for Arnoldi iteration (detect rank saturation)
KRYLOV_BREAKDOWN_TOL: float = 1e-14

#: Rank tolerance for matrix rank comparisons in Krylov criterion
KRYLOV_RANK_TOL: float = 1e-8

#: Default Krylov rank values for m-sweep experiments (None = use range(1, d+1))
DEFAULT_KRYLOV_M_VALUES: List[int] | None = None

#: Default Krylov m strategy when sweeping K ("K" means m=K dynamically)
DEFAULT_KRYLOV_M_STRATEGY: str = "K"
