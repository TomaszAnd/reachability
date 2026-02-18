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
        early_stop_zeros: int = 0,
    ) -> pd.DataFrame:
        """
        Run the sweep over given K values.

        Args:
            K_values: List of operator counts to test
            criteria: Which criteria to evaluate ('moment', 'spectral', 'krylov').
                      Default: all three.
            verbose: Print progress
            early_stop_zeros: Stop after this many consecutive K values with P=0
                              for ALL criteria. 0 = disabled.

        Returns:
            DataFrame with results
        """
        if criteria is None:
            criteria = ['moment', 'spectral', 'krylov']

        cfg = self.config
        d = self.model.dim
        phi = self.model.init_state()
        consecutive_zeros = 0

        for K in K_values:
            if K > self.model.K:
                if verbose:
                    print(f"  K={K} > K_max={self.model.K}, skipping")
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
                sub_seed = int(child_seeds[h_idx].generate_state(1)[0])
                sub_model = self.model.sample_submodel(K, seed=sub_seed)

                for t_idx in range(cfg.n_targets):
                    psi = sub_model.random_state()
                    # Krylov subspace dimension: full d by default (m<d gives trivially low scores)
                    m = cfg.krylov_m if cfg.krylov_m is not None else d

                    if 'moment' in criteria:
                        mc = MomentCriterion(sub_model, phi, psi, tau=cfg.tau)
                        result = mc.is_reachable()
                        if result.verdict == Verdict.UNREACHABLE:
                            counts['moment'] += 1
                        raw_scores['moment'].append(result.score)

                    if 'spectral' in criteria:
                        sc = SpectralCriterion(sub_model, phi, psi, tau=cfg.tau)
                        result = sc.is_reachable(
                            maxiter=cfg.maxiter, restarts=cfg.restarts,
                            method=cfg.method)
                        if result.verdict == Verdict.UNREACHABLE:
                            counts['spectral'] += 1
                        raw_scores['spectral'].append(result.score)

                    if 'krylov' in criteria:
                        kc = KrylovCriterion(sub_model, phi, psi, tau=cfg.tau, m=m)
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

            # Early stopping: all criteria at P=0 for N consecutive K values
            if early_stop_zeros > 0:
                all_zero = all(row[f'{c}_P'] == 0 for c in criteria)
                if all_zero:
                    consecutive_zeros += 1
                    if consecutive_zeros >= early_stop_zeros:
                        if verbose:
                            print(f"  Early stop: {early_stop_zeros} consecutive zeros")
                        break
                else:
                    consecutive_zeros = 0

        return pd.DataFrame(self._results)

    def run_parallel(
        self,
        K_values: List[int],
        criteria: Optional[List[str]] = None,
        verbose: bool = True,
        early_stop_zeros: int = 0,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Parallel version of run() using joblib.

        Parallelizes across Hamiltonian samples (n_hamiltonians) for each K.
        Falls back to sequential run() if joblib is not installed.

        Args:
            K_values: List of operator counts to test
            criteria: Which criteria to evaluate. Default: all three.
            verbose: Print progress
            early_stop_zeros: Stop after N consecutive all-zero K values. 0 = disabled.
            n_jobs: Number of parallel workers. -1 = all cores.

        Returns:
            DataFrame with results (identical format to run())
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            if verbose:
                print("joblib not installed, falling back to sequential run()")
            return self.run(K_values, criteria, verbose, early_stop_zeros)

        if criteria is None:
            criteria = ['moment', 'spectral', 'krylov']

        cfg = self.config
        d = self.model.dim
        phi = self.model.init_state()
        consecutive_zeros = 0

        def _evaluate_submodel(sub_seed, K, phi, criteria, cfg, d):
            """Evaluate all targets for one submodel. Runs in worker process."""
            sub_model = self.model.sample_submodel(K, seed=sub_seed)
            m = cfg.krylov_m if cfg.krylov_m is not None else d

            local_counts = {c: 0 for c in criteria}
            local_scores = {c: [] for c in criteria}

            for t_idx in range(cfg.n_targets):
                psi = sub_model.random_state()

                if 'moment' in criteria:
                    mc = MomentCriterion(sub_model, phi, psi, tau=cfg.tau)
                    result = mc.is_reachable()
                    if result.verdict == Verdict.UNREACHABLE:
                        local_counts['moment'] += 1
                    local_scores['moment'].append(result.score)

                if 'spectral' in criteria:
                    sc = SpectralCriterion(sub_model, phi, psi, tau=cfg.tau)
                    result = sc.is_reachable(
                        maxiter=cfg.maxiter, restarts=cfg.restarts,
                        method=cfg.method)
                    if result.verdict == Verdict.UNREACHABLE:
                        local_counts['spectral'] += 1
                    local_scores['spectral'].append(result.score)

                if 'krylov' in criteria:
                    kc = KrylovCriterion(sub_model, phi, psi, tau=cfg.tau, m=m)
                    result = kc.is_reachable(
                        maxiter=cfg.maxiter, restarts=cfg.restarts,
                        method=cfg.method)
                    if result.verdict == Verdict.UNREACHABLE:
                        local_counts['krylov'] += 1
                    local_scores['krylov'].append(result.score)

            return local_counts, local_scores

        for K in K_values:
            if K > self.model.K:
                if verbose:
                    print(f"  K={K} > K_max={self.model.K}, skipping")
                continue

            if verbose:
                print(f"  K={K}/{K_values[-1]}, rho={K/d**2:.4f}...", end=" ",
                      flush=True)

            n_trials = cfg.n_hamiltonians * cfg.n_targets

            # Spawn child RNGs for this K
            child_seeds = self.model._seed_seq.spawn(cfg.n_hamiltonians)
            sub_seeds = [int(cs.generate_state(1)[0]) for cs in child_seeds]

            # Parallel over submodels
            results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_evaluate_submodel)(seed, K, phi, criteria, cfg, d)
                for seed in sub_seeds
            )

            # Aggregate results
            counts = {c: 0 for c in criteria}
            raw_scores = {c: [] for c in criteria}
            for local_counts, local_scores in results_list:
                for c in criteria:
                    counts[c] += local_counts[c]
                    raw_scores[c].extend(local_scores[c])

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

            # Early stopping
            if early_stop_zeros > 0:
                all_zero = all(row[f'{c}_P'] == 0 for c in criteria)
                if all_zero:
                    consecutive_zeros += 1
                    if consecutive_zeros >= early_stop_zeros:
                        if verbose:
                            print(f"  Early stop: {early_stop_zeros} consecutive zeros")
                        break
                else:
                    consecutive_zeros = 0

        return pd.DataFrame(self._results)

    def save(self, path: Path) -> None:
        """Save results to CSV."""
        df = pd.DataFrame(self._results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
