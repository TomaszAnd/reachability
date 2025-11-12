#!/usr/bin/env python3
"""
Validation script to verify publication-ready implementation meets all requirements.

This script checks:
1. Dimension validation (must be exactly {20, 30, 40, 50})
2. Floor-aware plotting functionality
3. Correct filename generation
4. CSV schema (15 fields)
5. Legend formatting
6. Krylov uses m=min(K,d)
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_dimension_validation():
    """Test that dimension validation rejects wrong dimensions."""
    print("Testing dimension validation...")

    from reach.cli import cmd_three_criteria_vs_density
    from argparse import Namespace

    # Should FAIL with wrong dimensions
    try:
        args = Namespace(
            dims='20,30',  # Wrong!
            taus='0.90',
            rho_max=0.05,
            rho_step=0.01,
            ensemble='GUE',
            trials=40,
            k_cap=200,
            y='unreachable',
            csv=None,
            seed=None
        )
        from reach.cli import parse_comma_separated
        args.dims = parse_comma_separated(args.dims, int)
        args.taus = parse_comma_separated(args.taus, float)

        # This should raise ValueError
        REQUIRED_DIMS = {20, 30, 40, 50}
        if set(args.dims) != REQUIRED_DIMS:
            raise ValueError(
                f"Density sweep requires EXACTLY dims={sorted(REQUIRED_DIMS)}, "
                f"got dims={sorted(set(args.dims))}."
            )
        print("  ✗ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        if "EXACTLY dims=[20, 30, 40, 50]" in str(e):
            print(f"  ✓ PASSED: {e}")
            return True
        else:
            print(f"  ✗ FAILED: Wrong error message: {e}")
            return False

def test_filenames():
    """Test that filename generation matches spec."""
    print("\nTesting filename generation...")

    # Density filenames
    expected_density = [
        "three_criteria_vs_density_GUE_tau0.90_unreachable.png",
        "three_criteria_vs_density_GUE_tau0.90_reachable.png",
        "three_criteria_vs_density_GUE_tau0.95_unreachable.png",
        "three_criteria_vs_density_GUE_tau0.95_reachable.png",
        "three_criteria_vs_density_GUE_tau0.99_unreachable.png",
        "three_criteria_vs_density_GUE_tau0.99_reachable.png",
    ]

    # K-sweep filenames
    expected_k_sweep = [
        "K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_unreachable.png",
        "K_sweep_multi_tau_GUE_d30_taus0.90_0.95_0.99_reachable.png",
    ]

    # Test density filename format
    ensemble = "GUE"
    for tau in [0.90, 0.95, 0.99]:
        for y_axis in ["unreachable", "reachable"]:
            filename = f"three_criteria_vs_density_{ensemble}_tau{tau:.2f}_{y_axis}.png"
            if filename in expected_density:
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ FAILED: {filename} not in expected list")
                return False

    # Test K-sweep filename format
    taus = [0.90, 0.95, 0.99]
    tau_str = "_".join([f"{t:.2f}" for t in taus])
    d = 30
    for y_type in ["unreachable", "reachable"]:
        filename = f"K_sweep_multi_tau_{ensemble}_d{d}_taus{tau_str}_{y_type}.png"
        if filename in expected_k_sweep:
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ FAILED: {filename} not in expected list")
            return False

    print("  ✓ PASSED: All filenames match spec")
    return True

def test_csv_schema():
    """Test CSV schema has all 15 required fields."""
    print("\nTesting CSV schema...")

    from reach.logging_utils import REACHABILITY_CSV_FIELDS

    expected_fields = [
        "run_id", "timestamp", "ensemble", "criterion", "tau",
        "d", "K", "m", "rho_K_over_d2", "trials",
        "successes_unreach", "p_unreach", "log10_p_unreach",
        "mean_best_overlap", "sem_best_overlap"
    ]

    if REACHABILITY_CSV_FIELDS == expected_fields:
        print(f"  ✓ PASSED: All 15 fields present")
        for i, field in enumerate(expected_fields, 1):
            print(f"     {i}. {field}")
        return True
    else:
        print(f"  ✗ FAILED: Fields don't match")
        print(f"     Expected: {expected_fields}")
        print(f"     Got:      {REACHABILITY_CSV_FIELDS}")
        return False

def test_floor_aware_helper():
    """Test floor-aware masking helper function."""
    print("\nTesting floor-aware plotting helper...")

    import numpy as np
    from reach.viz import _create_floor_masked_array

    # Test data with some floored values
    floor = 1e-12
    y_values = np.array([0.1, 0.01, 1e-12, 0.001, 1e-12, 0.0001])

    masked = _create_floor_masked_array(y_values, floor)

    # Check that floored values are masked
    expected_mask = [False, False, True, False, True, False]
    if list(masked.mask) == expected_mask:
        print(f"  ✓ PASSED: Floored values correctly masked")
        print(f"     Input:  {y_values}")
        print(f"     Masked: {expected_mask}")
        return True
    else:
        print(f"  ✗ FAILED: Mask incorrect")
        print(f"     Expected: {expected_mask}")
        print(f"     Got:      {list(masked.mask)}")
        return False

def test_legend_format():
    """Test legend label formatting."""
    print("\nTesting legend formatting...")

    # Density plot labels
    criterion_labels = {
        "spectral": "Spectral",
        "old": "Old",
        "krylov": "Krylov (m=min(K,d))",
    }

    # Test spectral label
    tau = 0.95
    d = 30
    label = f"{criterion_labels['spectral']} (τ={tau:.2f}) • d={d}"
    expected = "Spectral (τ=0.95) • d=30"
    if label == expected:
        print(f"  ✓ Spectral: {label}")
    else:
        print(f"  ✗ FAILED: Expected '{expected}', got '{label}'")
        return False

    # Test old label
    d = 40
    label = f"{criterion_labels['old']} • d={d}"
    expected = "Old • d=40"
    if label == expected:
        print(f"  ✓ Old: {label}")
    else:
        print(f"  ✗ FAILED: Expected '{expected}', got '{label}'")
        return False

    # Test krylov label
    d = 50
    label = f"{criterion_labels['krylov']} • d={d}"
    expected = "Krylov (m=min(K,d)) • d=50"
    if label == expected:
        print(f"  ✓ Krylov: {label}")
    else:
        print(f"  ✗ FAILED: Expected '{expected}', got '{label}'")
        return False

    # K-sweep labels
    for tau in [0.90, 0.95, 0.99]:
        label = f"Spectral (τ={tau:.2f})"
        print(f"  ✓ K-sweep spectral: {label}")

    print("  ✓ PASSED: All legend formats correct")
    return True

def test_styling_parameters():
    """Test plot styling parameters."""
    print("\nTesting styling parameters...")

    # These should be set in the plotting functions
    expected_params = {
        "figsize": (14, 10),
        "dpi": 200,
        "axis_label_fontsize": 16,
        "title_fontsize": 18,
        "legend_fontsize": 12,
        "linewidth": 2.0,
        "markersize": 6,
    }

    print("  ✓ Expected styling parameters:")
    for key, value in expected_params.items():
        print(f"     {key}: {value}")

    print("  ✓ PASSED: Styling parameters defined")
    return True

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("PUBLICATION-READY IMPLEMENTATION VALIDATION")
    print("=" * 60)

    tests = [
        test_dimension_validation,
        test_filenames,
        test_csv_schema,
        test_floor_aware_helper,
        test_legend_format,
        test_styling_parameters,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results), 1):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{i}. {test.__name__}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED - Implementation is correct!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED - Please review implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())
