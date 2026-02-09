"""
Global constants for the reach package.

These are defaults used by the legacy mathematics.py and optimize.py modules.
New code should use SweepConfig from sampling.py or pass values directly.
"""

# Reproducibility
SEED: int = 42
AUTO_TIDYUP: bool = False

# Optimization defaults
DEFAULT_BOUNDS = [(-1.0, 1.0)]
DEFAULT_METHOD: str = "L-BFGS-B"
DEFAULT_RESTARTS: int = 2
DEFAULT_MAXITER: int = 200
DEFAULT_FTOL: float = 1e-8

# Mathematical constants
OVERLAP_TOLERANCE: float = 1e-6
MIN_CONDITION_NUMBER: float = 1e-12

# Krylov
KRYLOV_BREAKDOWN_TOL: float = 1e-14
KRYLOV_RANK_TOL: float = 1e-8

# Analysis
DEFAULT_TAU: float = 0.95
