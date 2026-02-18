"""
Generate accuracy baseline for v51 optimization validation.
Saves exact scores and verdicts for comparison after changes.

Usage:
    python scripts/tests/accuracy_baseline.py           # Generate baseline
    python scripts/tests/accuracy_baseline.py verify    # Verify against baseline
"""
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models import CanonicalQuditModel, QubitGridModel
from src.criteria import MomentCriterion, SpectralCriterion, KrylovCriterion


BASELINE_PATH = os.path.join(os.path.dirname(__file__), 'accuracy_baseline.json')

TEST_CASES = [
    ('canonical', 8, 5),
    ('canonical', 8, 15),
    ('canonical', 16, 10),
    ('canonical', 16, 30),
    ('qubitgrid', 8, 5),
    ('qubitgrid', 8, 15),
]


def generate_baseline():
    results = {}

    for model_type, d, K in TEST_CASES:
        key = f"{model_type}_d{d}_K{K}"

        if model_type == 'canonical':
            model = CanonicalQuditModel(dim=d, seed=42)
        else:
            model = QubitGridModel(dim=d, seed=42)

        sub = model.sample_submodel(K, seed=123)
        phi = model.init_state()
        psi = sub.random_state()

        mc = MomentCriterion(sub, phi, psi, tau=0.99)
        sc = SpectralCriterion(sub, phi, psi, tau=0.99)
        kc = KrylovCriterion(sub, phi, psi, tau=0.99, m=d)

        m_result = mc.is_reachable()
        s_result = sc.is_reachable(maxiter=100, restarts=3, ftol=1e-8)
        k_result = kc.is_reachable(maxiter=100, restarts=3, ftol=1e-8)

        results[key] = {
            'moment_score': float(m_result.score),
            'moment_verdict': m_result.verdict.value,
            'spectral_score': float(s_result.score),
            'spectral_verdict': s_result.verdict.value,
            'krylov_score': float(k_result.score),
            'krylov_verdict': k_result.verdict.value,
        }

        print(f"{key}: M={m_result.verdict.value}, "
              f"S={s_result.score:.6f}({s_result.verdict.value}), "
              f"K={k_result.score:.6f}({k_result.verdict.value})")

    with open(BASELINE_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline saved to {BASELINE_PATH}")
    return results


def verify_against_baseline(tolerance=1e-6):
    """Run after optimization to verify accuracy preserved."""
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)

    print("Regenerating with current code...\n")
    current = generate_baseline()

    print("\n--- Comparison ---")
    all_match = True
    for key in baseline:
        for field in ['moment_verdict', 'spectral_verdict', 'krylov_verdict']:
            if baseline[key][field] != current[key][field]:
                print(f"VERDICT MISMATCH: {key}.{field}: "
                      f"{baseline[key][field]} -> {current[key][field]}")
                all_match = False

        for field in ['moment_score', 'spectral_score', 'krylov_score']:
            diff = abs(baseline[key][field] - current[key][field])
            if diff > tolerance:
                print(f"SCORE DRIFT: {key}.{field}: "
                      f"{baseline[key][field]:.6f} -> {current[key][field]:.6f} "
                      f"(diff={diff:.2e})")

    if all_match:
        print("\nAll verdicts match baseline")
    else:
        print("\nVERDICTS CHANGED - ROLLBACK REQUIRED")

    return all_match


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'verify':
        verify_against_baseline()
    else:
        generate_baseline()
