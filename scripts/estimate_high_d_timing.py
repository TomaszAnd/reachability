"""Estimate timing for higher dimensions based on measured scaling."""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import CanonicalQuditModel
from src.criteria import SpectralCriterion, KrylovCriterion, MomentCriterion


def measure_single_trial(d, K, n_repeats=3):
    """Measure average time for single trial at dimension d."""
    model = CanonicalQuditModel(dim=d, seed=42)
    sub = model.sample_submodel(K, seed=123)
    phi = model.init_state()
    psi = sub.random_state()

    times = {'moment': [], 'spectral': [], 'krylov': []}

    for _ in range(n_repeats):
        mc = MomentCriterion(sub, phi, psi, tau=0.99)
        t0 = time.perf_counter()
        mc.is_reachable()
        times['moment'].append(time.perf_counter() - t0)

        sc = SpectralCriterion(sub, phi, psi, tau=0.99)
        t0 = time.perf_counter()
        sc.is_reachable(maxiter=50, restarts=2)
        times['spectral'].append(time.perf_counter() - t0)

        kc = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)
        t0 = time.perf_counter()
        kc.is_reachable(maxiter=50, restarts=2)
        times['krylov'].append(time.perf_counter() - t0)

    return {k: np.median(v) for k, v in times.items()}


def estimate_scaling():
    """Measure at multiple d values and fit scaling."""
    measured_dims = [8, 16, 32]
    target_dims = [64, 128]

    print("Measuring timing at reference dimensions (median of 3 repeats)...")
    measurements = {}
    for d in measured_dims:
        K = d * d // 4  # rho = 0.25
        times = measure_single_trial(d, K)
        measurements[d] = times
        print(f"  d={d}, K={K}: moment={times['moment']*1000:.1f}ms, "
              f"spectral={times['spectral']*1000:.1f}ms, "
              f"krylov={times['krylov']*1000:.1f}ms")

    # Fit scaling exponents
    ds = np.array(measured_dims, dtype=float)
    for crit in ['spectral', 'krylov']:
        ts = np.array([measurements[d][crit] for d in measured_dims])
        log_fit = np.polyfit(np.log(ds), np.log(ts), 1)
        print(f"\n{crit.capitalize()} scaling: t ~ d^{log_fit[0]:.2f}")

    # Estimate for all dimensions
    print(f"\n{'d':>6} | {'K (rho=0.25)':>12} | {'Spectral':>12} | {'Krylov':>12} | "
          f"{'Est. 200 trials':>16} | {'Est. 1600 trials':>16}")
    print("-"*90)

    for d in measured_dims + target_dims:
        K = d * d // 4
        if d in measurements:
            t_spec = measurements[d]['spectral']
            t_krylov = measurements[d]['krylov']
        else:
            # Extrapolate from measured data
            spec_times = np.array([measurements[dd]['spectral'] for dd in measured_dims])
            kry_times = np.array([measurements[dd]['krylov'] for dd in measured_dims])
            spec_fit = np.polyfit(np.log(ds), np.log(spec_times), 1)
            kry_fit = np.polyfit(np.log(ds), np.log(kry_times), 1)
            t_spec = np.exp(spec_fit[1]) * (d ** spec_fit[0])
            t_krylov = np.exp(kry_fit[1]) * (d ** kry_fit[0])

        t_trial = t_spec + t_krylov
        t_200 = 200 * t_trial  # 200 trials, 8 K values per dim
        t_1600 = 1600 * t_trial  # 1600 trials (8 K values x 200)

        def fmt_time(s):
            if s < 60:
                return f"{s:.0f}s"
            elif s < 3600:
                return f"{s/60:.1f} min"
            else:
                return f"{s/3600:.1f} hr"

        print(f"{d:>6} | {K:>12} | {t_spec*1000:>10.1f}ms | {t_krylov*1000:>10.1f}ms | "
              f"{fmt_time(t_200):>16} | {fmt_time(t_1600):>16}")


if __name__ == '__main__':
    estimate_scaling()
