"""
Monte Carlo sweep over operator count K.

DensitySweep runs Monte Carlo experiments varying K (the number of control
operators), evaluating reachability criteria at each K value.

K is the primary parameter. Density rho = K/d^2 is computed for output only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .models import QuantumModel
from .criteria import (
    SpectralCriterion, KrylovCriterion, MomentCriterion,
    Verdict, ReachabilityResult,
)
from .math_utils import compute_binomial_sem


@dataclass
class SweepConfig:
    """Configuration for a density sweep experiment."""
    n_hamiltonians: int = 150
    n_targets: int = 30
    tau: float = 0.99
    maxiter: int = 100
    restarts: int = 3
    method: str = 'L-BFGS-B'
    krylov_m: Optional[int] = None  # None = use full dimension d
    save_raw_scores: bool = True

    @classmethod
    def fast(cls) -> 'SweepConfig':
        """Quick validation config."""
        return cls(n_hamiltonians=20, n_targets=10, maxiter=50, restarts=2)

    @classmethod
    def production(cls, dim: Optional[int] = None) -> 'SweepConfig':
        """Publication-quality config with dimension-adaptive maxiter."""
        maxiter = 100 if dim is None or dim <= 32 else 150
        return cls(n_hamiltonians=150, n_targets=30, maxiter=maxiter, restarts=3)


class DensitySweep:
    """
    Monte Carlo sweep over operator count K.

    For each K value:
    1. Sample n_hamiltonians submodels from the full basis
    2. For each submodel, test n_targets random target states
    3. Compute P(unreachable) for each criterion

    Results stored as DataFrame with columns:
    - K, d, rho (= K/d^2), tau
    - moment_P, spectral_P, krylov_P (fraction unreachable)
    - moment_sem, spectral_sem, krylov_sem
    - n_trials
    """

    def __init__(self, model: QuantumModel, config: Optional[SweepConfig] = None):
        self.model = model
        self.config = config or SweepConfig()
        self._results: List[Dict] = []

    def run(
        self,
        K_values: List[int],
        criteria: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the sweep over given K values.

        Args:
            K_values: List of operator counts to test
            criteria: Which criteria to evaluate ('moment', 'spectral', 'krylov').
                      Default: all three.
            verbose: Print progress

        Returns:
            DataFrame with results
        """
        if criteria is None:
            criteria = ['moment', 'spectral', 'krylov']

        cfg = self.config
        d = self.model.dim
        psi = self.model.init_state()

        for K in K_values:
            if K > self.model.basis_size:
                if verbose:
                    print(f"  K={K} > basis_size={self.model.basis_size}, skipping")
                continue

            if verbose:
                print(f"  K={K}/{K_values[-1]}, rho={K/d**2:.4f}...", end=" ",
                      flush=True)

            n_trials = cfg.n_hamiltonians * cfg.n_targets
            counts = {c: 0 for c in criteria}
            raw_scores = {c: [] for c in criteria}

            # Spawn child RNGs for this K
            child_seeds = self.model._seed_seq.spawn(cfg.n_hamiltonians)

            for h_idx in range(cfg.n_hamiltonians):
                sub_seed = child_seeds[h_idx].entropy
                sub_model = self.model.sample_submodel(K, seed=sub_seed)

                for t_idx in range(cfg.n_targets):
                    phi = sub_model.random_state()
                    # Krylov subspace dimension: full d by default (m<d gives trivially low scores)
                    m = cfg.krylov_m if cfg.krylov_m is not None else d

                    if 'moment' in criteria:
                        mc = MomentCriterion(sub_model, psi, phi, tau=cfg.tau)
                        result = mc.is_reachable()
                        if result.verdict == Verdict.UNREACHABLE:
                            counts['moment'] += 1
                        raw_scores['moment'].append(result.score)

                    if 'spectral' in criteria:
                        sc = SpectralCriterion(sub_model, psi, phi, tau=cfg.tau)
                        result = sc.is_reachable(
                            maxiter=cfg.maxiter, restarts=cfg.restarts,
                            method=cfg.method)
                        if result.verdict == Verdict.UNREACHABLE:
                            counts['spectral'] += 1
                        raw_scores['spectral'].append(result.score)

                    if 'krylov' in criteria:
                        kc = KrylovCriterion(sub_model, psi, phi, tau=cfg.tau, m=m)
                        result = kc.is_reachable(
                            maxiter=cfg.maxiter, restarts=cfg.restarts,
                            method=cfg.method)
                        if result.verdict == Verdict.UNREACHABLE:
                            counts['krylov'] += 1
                        raw_scores['krylov'].append(result.score)

            row = {
                'K': K,
                'd': d,
                'rho': K / d**2,
                'tau': cfg.tau,
                'n_trials': n_trials,
            }

            for c in criteria:
                P = counts[c] / n_trials if n_trials > 0 else 0.0
                row[f'{c}_P'] = P
                row[f'{c}_sem'] = compute_binomial_sem(P, n_trials)
                if cfg.save_raw_scores:
                    row[f'{c}_mean'] = np.mean(raw_scores[c]) if raw_scores[c] else 0.0
                    row[f'{c}_std'] = np.std(raw_scores[c]) if raw_scores[c] else 0.0

            self._results.append(row)

            if verbose:
                parts = []
                for c in criteria:
                    parts.append(f"{c}={row[f'{c}_P']:.2f}")
                print(", ".join(parts))

        return pd.DataFrame(self._results)

    def save(self, path: Path) -> None:
        """Save results to CSV."""
        df = pd.DataFrame(self._results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
