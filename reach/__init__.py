"""
Reach: time-free quantum reachability analysis.

New class-based API:
    from reach.models import QuantumModel, CanonicalModel, GeometricTwoLocalModel
    from reach.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion, Verdict
    from reach.sampling import DensitySweep, SweepConfig
    from reach import plotting

Legacy API (for backward compatibility with benchmark scripts):
    from reach import models, mathematics, optimize
"""

from . import (
    models,
    mathematics,
    optimize,
    criteria,
    sampling,
    math_utils,
    plotting,
    settings,
    legacy,
)

__all__ = [
    "models",
    "mathematics",
    "optimize",
    "criteria",
    "sampling",
    "math_utils",
    "plotting",
    "settings",
    "legacy",
]
__version__ = "0.2.0"
