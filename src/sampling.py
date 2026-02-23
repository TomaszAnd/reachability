"""
Monte Carlo sweep over operator count K.

DensitySweep runs Monte Carlo experiments varying K (the number of control
operators), evaluating reachability criteria at each K value.

K is the primary parameter. Density rho = K/d^2 is computed for output only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import QuantumModel
from .criteria import (
    SpectralCriterion, KrylovCriterion, MomentCriterion,
    Verdict,
)
from .math_utils import compute_binomial_sem


def adaptive_K_values(
    d: int,
    rho_c: float = 0.15,
    rho_min: float = 0.02,
    rho_max: float = 0.40,
    n_total: int = 15,
    K_max: Optional[int] = None,
) -> List[int]:
    """Generate K values with denser sampling near the phase transition rho_c.

    Places ~half the points in [rho_c - 2*delta, rho_c + 2*delta] and the rest
    spread across the tails, where delta = (rho_max - rho_min) / n_total.

    Args:
        d: Hilbert space dimension.
        rho_c: Estimated critical density for the phase transition.
        rho_min: Minimum rho value.
        rho_max: Maximum rho value.
        n_total: Total number of K values to generate.
        K_max: Maximum allowed K (default: d^2 - 1).

    Returns:
        Sorted list of unique K values (integers >= 2).
    """
    if K_max is None:
        K_max = d * d - 1

    # Half the points in the dense region near rho_c
    n_dense = n_total // 2
    n_sparse = n_total - n_dense

    delta = (rho_max - rho_min) / n_total
    rho_lo = max(rho_min, rho_c - 2 * delta)
    rho_hi = min(rho_max, rho_c + 2 * delta)

    # Dense region: uniform in [rho_lo, rho_hi]
    rho_dense = np.linspace(rho_lo, rho_hi, n_dense)

    # Sparse region: spread across tails
    rho_left = np.linspace(rho_min, rho_lo, n_sparse // 2 + 1)[:-1]
    rho_right = np.linspace(rho_hi, rho_max, n_sparse - n_sparse // 2 + 1)[1:]
    rho_sparse = np.concatenate([rho_left, rho_right])

    rho_all = np.concatenate([rho_dense, rho_sparse])
    K_values = sorted(set(max(2, int(rho * d**2)) for rho in rho_all))
    K_values = [k for k in K_values if k <= K_max]
    return K_values


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


def _check_early_stop(row: Dict, criteria: List[str],
                      early_stop_zeros: int,
                      consecutive_zeros: int,
                      verbose: bool) -> tuple:
    """Check early-stopping condition for density sweeps.

    Returns (should_stop, updated_consecutive_zeros).
    """
    if early_stop_zeros <= 0:
        return False, consecutive_zeros
    all_zero = all(row[f'{c}_P'] == 0 for c in criteria)
    if all_zero:
        consecutive_zeros += 1
        if consecutive_zeros >= early_stop_zeros:
            if verbose:
                print(f"  Early stop: {early_stop_zeros} consecutive zeros")
            return True, consecutive_zeros
    else:
        consecutive_zeros = 0
    return False, consecutive_zeros


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

    def _evaluate_submodel(
        self,
        sub_seed: int,
        K: int,
        phi: np.ndarray,
        criteria: List[str],
    ) -> Tuple[Dict[str, int], Dict[str, list]]:
        """Evaluate all target states for one submodel.

        Returns (counts, scores) where counts[c] is the number of UNREACHABLE
        verdicts and scores[c] is a list of raw scores, for each criterion c.
        """
        cfg = self.config
        d = self.model.dim
        sub_model = self.model.sample_submodel(K, seed=sub_seed)
        m = cfg.krylov_m if cfg.krylov_m is not None else d

        counts = {c: 0 for c in criteria}
        scores = {c: [] for c in criteria}

        for _ in range(cfg.n_targets):
            psi = sub_model.random_state()

            if 'moment' in criteria:
                mc = MomentCriterion(sub_model, phi, psi, tau=cfg.tau)
                result = mc.is_reachable()
                if result.verdict == Verdict.UNREACHABLE:
                    counts['moment'] += 1
                scores['moment'].append(result.score)

            if 'spectral' in criteria:
                sc = SpectralCriterion(sub_model, phi, psi, tau=cfg.tau)
                result = sc.is_reachable(
                    maxiter=cfg.maxiter, restarts=cfg.restarts,
                    method=cfg.method)
                if result.verdict == Verdict.UNREACHABLE:
                    counts['spectral'] += 1
                scores['spectral'].append(result.score)

            if 'krylov' in criteria:
                kc = KrylovCriterion(sub_model, phi, psi, tau=cfg.tau, m=m)
                result = kc.is_reachable(
                    maxiter=cfg.maxiter, restarts=cfg.restarts,
                    method=cfg.method)
                if result.verdict == Verdict.UNREACHABLE:
                    counts['krylov'] += 1
                scores['krylov'].append(result.score)

        return counts, scores

    def _build_row(
        self,
        K: int,
        criteria: List[str],
        counts: Dict[str, int],
        raw_scores: Dict[str, list],
        n_trials: int,
    ) -> Dict:
        """Build a results row for one K value."""
        cfg = self.config
        d = self.model.dim
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
        return row

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

            child_seeds = self.model._seed_seq.spawn(cfg.n_hamiltonians)

            for h_idx in range(cfg.n_hamiltonians):
                sub_seed = int(child_seeds[h_idx].generate_state(1)[0])
                local_counts, local_scores = self._evaluate_submodel(
                    sub_seed, K, phi, criteria)
                for c in criteria:
                    counts[c] += local_counts[c]
                    raw_scores[c].extend(local_scores[c])

            row = self._build_row(K, criteria, counts, raw_scores, n_trials)
            self._results.append(row)

            if verbose:
                parts = [f"{c}={row[f'{c}_P']:.2f}" for c in criteria]
                print(", ".join(parts))

            should_stop, consecutive_zeros = _check_early_stop(
                row, criteria, early_stop_zeros, consecutive_zeros, verbose)
            if should_stop:
                break

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

        for K in K_values:
            if K > self.model.K:
                if verbose:
                    print(f"  K={K} > K_max={self.model.K}, skipping")
                continue

            if verbose:
                print(f"  K={K}/{K_values[-1]}, rho={K/d**2:.4f}...", end=" ",
                      flush=True)

            n_trials = cfg.n_hamiltonians * cfg.n_targets

            child_seeds = self.model._seed_seq.spawn(cfg.n_hamiltonians)
            sub_seeds = [int(cs.generate_state(1)[0]) for cs in child_seeds]

            results_list = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._evaluate_submodel)(seed, K, phi, criteria)
                for seed in sub_seeds
            )

            # Aggregate
            counts = {c: 0 for c in criteria}
            raw_scores = {c: [] for c in criteria}
            for local_counts, local_scores in results_list:
                for c in criteria:
                    counts[c] += local_counts[c]
                    raw_scores[c].extend(local_scores[c])

            row = self._build_row(K, criteria, counts, raw_scores, n_trials)
            self._results.append(row)

            if verbose:
                parts = [f"{c}={row[f'{c}_P']:.2f}" for c in criteria]
                print(", ".join(parts))

            should_stop, consecutive_zeros = _check_early_stop(
                row, criteria, early_stop_zeros, consecutive_zeros, verbose)
            if should_stop:
                break

        return pd.DataFrame(self._results)

    def save(self, path: Path) -> None:
        """Save results to CSV."""
        df = pd.DataFrame(self._results)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


@dataclass
class AdaptiveSweepConfig(SweepConfig):
    """Configuration for adaptive density sweep with early termination.

    Instead of fixed n_hamiltonians * n_targets trials per K, adaptively
    stops early when statistical confidence is achieved or when P is
    clearly near 0 or 1.
    """
    min_hamiltonians: int = 20
    max_hamiltonians: int = 150
    target_sem: float = 0.02
    batch_size: int = 10
    zero_threshold: float = 0.01

    @classmethod
    def fast(cls) -> 'AdaptiveSweepConfig':
        """Quick validation config."""
        return cls(min_hamiltonians=10, max_hamiltonians=30,
                   n_targets=5, maxiter=50, restarts=2,
                   target_sem=0.05, batch_size=5)

    @classmethod
    def production(cls, dim: Optional[int] = None) -> 'AdaptiveSweepConfig':
        """Publication-quality config."""
        maxiter = 100 if dim is None or dim <= 32 else 150
        return cls(min_hamiltonians=30, max_hamiltonians=200,
                   n_targets=20, maxiter=maxiter, restarts=3,
                   target_sem=0.015, batch_size=10)


class AdaptiveSweep(DensitySweep):
    """
    Adaptive Monte Carlo sweep with early termination.

    Like DensitySweep but adaptively adjusts the number of Hamiltonian
    samples per K value based on statistical confidence:
    - Stops early when SEM < target_sem for all criteria
    - Stops early when all criteria show P < zero_threshold
    - Uses at least min_hamiltonians and at most max_hamiltonians
    """

    def __init__(self, model: QuantumModel, config: Optional[AdaptiveSweepConfig] = None):
        cfg = config or AdaptiveSweepConfig()
        super().__init__(model, cfg)

    def run(
        self,
        K_values: List[int],
        criteria: Optional[List[str]] = None,
        verbose: bool = True,
        early_stop_zeros: int = 0,
    ) -> pd.DataFrame:
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

            counts = {c: 0 for c in criteria}
            raw_scores = {c: [] for c in criteria}
            n_hams_done = 0

            child_seeds = self.model._seed_seq.spawn(cfg.max_hamiltonians)

            while n_hams_done < cfg.max_hamiltonians:
                # Process a batch
                batch_end = min(n_hams_done + cfg.batch_size, cfg.max_hamiltonians)
                for h_idx in range(n_hams_done, batch_end):
                    sub_seed = int(child_seeds[h_idx].generate_state(1)[0])
                    local_counts, local_scores = self._evaluate_submodel(
                        sub_seed, K, phi, criteria)
                    for c in criteria:
                        counts[c] += local_counts[c]
                        raw_scores[c].extend(local_scores[c])

                n_hams_done = batch_end
                n_trials = n_hams_done * cfg.n_targets

                # Don't check early stopping until minimum reached
                if n_hams_done < cfg.min_hamiltonians:
                    continue

                # Check if all criteria are confident enough
                all_confident = True
                all_near_zero = True
                for c in criteria:
                    P = counts[c] / n_trials if n_trials > 0 else 0.0
                    sem = compute_binomial_sem(P, n_trials)
                    if P > cfg.zero_threshold:
                        all_near_zero = False
                    if P > cfg.zero_threshold and P < 1 - cfg.zero_threshold:
                        if sem > cfg.target_sem:
                            all_confident = False

                if all_near_zero or all_confident:
                    break

            n_trials = n_hams_done * cfg.n_targets
            row = self._build_row(K, criteria, counts, raw_scores, n_trials)
            row['n_hamiltonians_used'] = n_hams_done
            self._results.append(row)

            if verbose:
                parts = [f"{c}={row[f'{c}_P']:.2f}" for c in criteria]
                parts.append(f"n_h={n_hams_done}")
                print(", ".join(parts))

            should_stop, consecutive_zeros = _check_early_stop(
                row, criteria, early_stop_zeros, consecutive_zeros, verbose)
            if should_stop:
                break

        return pd.DataFrame(self._results)
