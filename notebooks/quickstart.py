#!/usr/bin/env python3
"""
Quick-start example for the reach package (class-based API).

Demonstrates:
1. Creating a Hamiltonian family (CanonicalModel)
2. Sampling a submodel with K operators
3. Evaluating all three criteria
4. Running a mini density sweep
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from reach.models import CanonicalModel, GeometricTwoLocalModel
from reach.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion, Verdict
from reach.sampling import DensitySweep, SweepConfig

# ============================================================================
# 1. Create a Hamiltonian family
# ============================================================================
print("=" * 60)
print("1. Create a Canonical Hamiltonian family (d=8)")
print("=" * 60)

model = CanonicalModel(dim=8, seed=42)
print(f"   Dimension: {model.dim}")
print(f"   Full basis size (L = d^2): {model.basis_size}")
print(f"   Operator shape: {model.basis[0].shape}")

# ============================================================================
# 2. Sample a submodel with K operators
# ============================================================================
print(f"\n2. Sample submodel with K=10 operators")
sub = model.sample_submodel(10)
print(f"   Submodel K: {sub.K}")

# States
psi = model.init_state()
phi = model.random_state()
print(f"   Initial state |0>: norm = {np.linalg.norm(psi):.4f}")
print(f"   Random target: norm = {np.linalg.norm(phi):.4f}")

# ============================================================================
# 3. Evaluate all three criteria (passing model to constructors)
# ============================================================================
print(f"\n3. Evaluate reachability criteria")

# Spectral
sc = SpectralCriterion(sub, psi, phi, tau=0.99)
spec_result = sc.is_reachable(maxiter=100, restarts=2)
print(f"   Spectral:  verdict={spec_result.verdict.value:12s}  S*={spec_result.score:.4f}")

# Krylov
kc = KrylovCriterion(sub, psi, phi, tau=0.99, m=min(10, 8))
kryl_result = kc.is_reachable(maxiter=100, restarts=2)
print(f"   Krylov:    verdict={kryl_result.verdict.value:12s}  R*={kryl_result.score:.4f}")

# Moment
mc = MomentCriterion(sub, psi, phi, tau=0.99)
mom_result = mc.is_reachable()
print(f"   Moment:    verdict={mom_result.verdict.value:12s}  ", end="")
if mom_result.certificate:
    print(f"certificate: {mom_result.certificate}")
else:
    print("no certificate found")

# ============================================================================
# 4. Quick density sweep
# ============================================================================
print(f"\n4. Quick density sweep (d=8, K=2..15)")

config = SweepConfig.fast()
config.tau = 0.99
sweep = DensitySweep(model, config)
K_values = [2, 4, 6, 8, 10, 12, 15]
df = sweep.run(K_values, criteria=['moment', 'spectral'], verbose=True)

print(f"\nResults DataFrame ({len(df)} rows):")
print(df[['K', 'rho', 'moment_P', 'spectral_P']].to_string(index=False))

# ============================================================================
# 5. GeometricTwoLocal example
# ============================================================================
print(f"\n5. GeometricTwoLocal family (d=16, 2x2 lattice)")
geo = GeometricTwoLocalModel(dim=16, nx=2, ny=2, seed=42)
print(f"   Dimension: {geo.dim}")
print(f"   Basis size: {geo.basis_size} (3*4 + 9*4 = 48)")
sub_geo = geo.sample_submodel(5)
print(f"   Submodel K: {sub_geo.K}")

print("\nDone!")
