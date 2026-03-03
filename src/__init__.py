"""
Quantum reachability analysis.

Class-based API:
    from src.models import QuantumModel, CanonicalQuditModel, QubitGridModel
    from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion, Verdict
    from src.sampling import DensitySweep, SweepConfig, AdaptiveSweep, AdaptiveSweepConfig
    from src import plotting

Optional JAX backend:
    import os
    os.environ['REACHABILITY_BACKEND'] = 'jax'
    from src.criteria_jax import SpectralCriterionJAX, KrylovCriterionJAX
"""

from . import (
    models,
    criteria,
    sampling,
    math_utils,
    plotting,
    backend,
)

__all__ = [
    "models",
    "criteria",
    "sampling",
    "math_utils",
    "plotting",
    "backend",
]
__version__ = "0.4.2"
