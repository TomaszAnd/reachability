#!/usr/bin/env python3
"""Estimate sweep time for publication runs.

Runs a small sample of trials and projects to full sweep durations.

Usage:
    python scripts/estimate_sweep_time.py
    python scripts/estimate_sweep_time.py --dim 128 --trials 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.models import QubitGridModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion


def estimate_sweep_time(d: int = 256, K: int = 60, n_trials: int = 10,
                        maxiter: int = 50, restarts: int = 1):
    """Run timing tests and project to full sweep."""
    model = QubitGridModel(dim=d, seed=42)
    sub = model.sample_submodel(K, seed=0)
    phi = sub.init_state()

    times = {"spectral": [], "krylov": [], "moment": []}

    for i in range(n_trials):
        rng = np.random.default_rng(i)
        psi = sub.random_state(rng=rng)

        # Spectral
        sc = SpectralCriterion(sub, phi, psi, tau=0.99)
        t0 = time.perf_counter()
        sc.is_reachable(maxiter=maxiter, restarts=restarts)
        times["spectral"].append(time.perf_counter() - t0)

        # Krylov
        kc = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)
        t0 = time.perf_counter()
        kc.is_reachable(maxiter=maxiter, restarts=restarts)
        times["krylov"].append(time.perf_counter() - t0)

        # Moment
        mc = MomentCriterion(sub, phi, psi, tau=0.99)
        t0 = time.perf_counter()
        mc.is_reachable()
        times["moment"].append(time.perf_counter() - t0)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_trials} done")

    print(f"\n{'='*60}")
    print(f"d={d} K={K} SWEEP TIME ESTIMATION (maxiter={maxiter}, restarts={restarts})")
    print(f"{'='*60}")

    configs = [
        ("Quick test (20h x 3t x 5K)", 20 * 3 * 5),
        ("Publication (50h x 5t x 10K)", 50 * 5 * 10),
        ("Full (100h x 10t x 10K)", 100 * 10 * 10),
    ]

    for criterion, t_list in times.items():
        avg = np.mean(t_list)
        std = np.std(t_list)
        print(f"\n{criterion.upper()}:")
        print(f"  Per trial: {avg*1000:.1f} +/- {std*1000:.1f} ms")
        for label, n in configs:
            hours = n * avg / 3600
            print(f"  {label}: {hours:.1f} hours")

    total_avg = sum(np.mean(t) for t in times.values())
    print(f"\nALL CRITERIA COMBINED:")
    for label, n in configs:
        hours = n * total_avg / 3600
        print(f"  {label}: {hours:.1f} hours ({hours/24:.1f} days)")


def main():
    parser = argparse.ArgumentParser(description="Sweep time estimation")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=1)
    args = parser.parse_args()

    estimate_sweep_time(args.dim, args.K, args.trials,
                        args.maxiter, args.restarts)


if __name__ == "__main__":
    main()
