"""
Quantum reachability analysis.

Class-based API:
    from src.models import QuantumModel, CanonicalQuditModel, QubitGridModel
    from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion, Verdict
    from src.sampling import DensitySweep, SweepConfig
    from src import plotting
"""

from . import (
    models,
    criteria,
    sampling,
    math_utils,
    plotting,
)

__all__ = [
    "models",
    "criteria",
    "sampling",
    "math_utils",
    "plotting",
]
__version__ = "0.3.0"
