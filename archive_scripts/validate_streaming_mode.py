#!/usr/bin/env python3
"""
Validation script for streaming CSV mode and plot-from-csv functionality.

This script:
1. Creates tiny mock CSV files with sample data
2. Calls plot-from-csv for both density and K-sweep types
3. Verifies that PNG files are created
4. Reports PASS/FAIL status

Run with:
    python validate_streaming_mode.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

def create_mock_density_csv(path: str) -> None:
    """Create a tiny mock CSV for density plots."""
    header = "run_id,timestamp,ensemble,criterion,tau,d,K,m,rho_K_over_d2,trials,successes_unreach,p_unreach,log10_p_unreach,mean_best_overlap,sem_best_overlap"

    # Create a few sample rows for d=20 and d=30
    rows = [
        # Spectral, tau=0.95, d=20
        "test_001,2025-10-22T12:00:00,GUE,spectral,0.95,20,5,,0.0125,150,120,0.8,-0.09691,0.92,0.015",
        "test_001,2025-10-22T12:00:00,GUE,spectral,0.95,20,10,,0.025,150,90,0.6,-0.22185,0.85,0.020",
        # Spectral, tau=0.95, d=30
        "test_001,2025-10-22T12:00:00,GUE,spectral,0.95,30,10,,0.0111,150,100,0.667,-0.17609,0.88,0.018",
        # Old criterion, d=20
        "test_001,2025-10-22T12:00:00,GUE,old,,20,5,,0.0125,150,130,0.867,-0.06213,,",
        "test_001,2025-10-22T12:00:00,GUE,old,,20,10,,0.025,150,100,0.667,-0.17609,,",
        # Krylov criterion, d=20
        "test_001,2025-10-22T12:00:00,GUE,krylov,,20,5,,0.0125,150,110,0.733,-0.13490,,",
        "test_001,2025-10-22T12:00:00,GUE,krylov,,20,10,,0.025,150,80,0.533,-0.27327,,",
    ]

    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

    print(f"✓ Created mock density CSV: {path}")


def create_mock_k_csv(path: str) -> None:
    """Create a tiny mock CSV for K-sweep plots."""
    header = "run_id,timestamp,ensemble,criterion,tau,d,K,m,rho_K_over_d2,trials,successes_unreach,p_unreach,log10_p_unreach,mean_best_overlap,sem_best_overlap"

    # Create sample rows for d=30, K=2-5
    rows = [
        # Spectral, tau=0.90, d=30
        "test_002,2025-10-22T12:00:00,GUE,spectral,0.90,30,2,,0.00222,300,250,0.833,-0.07918,0.85,0.012",
        "test_002,2025-10-22T12:00:00,GUE,spectral,0.90,30,3,,0.00333,300,200,0.667,-0.17609,0.88,0.015",
        "test_002,2025-10-22T12:00:00,GUE,spectral,0.90,30,4,,0.00444,300,150,0.500,-0.30103,0.90,0.018",
        # Spectral, tau=0.95, d=30
        "test_002,2025-10-22T12:00:00,GUE,spectral,0.95,30,2,,0.00222,300,280,0.933,-0.03005,0.92,0.010",
        "test_002,2025-10-22T12:00:00,GUE,spectral,0.95,30,3,,0.00333,300,240,0.800,-0.09691,0.93,0.012",
        "test_002,2025-10-22T12:00:00,GUE,spectral,0.95,30,4,,0.00444,300,180,0.600,-0.22185,0.95,0.015",
        # Old criterion
        "test_002,2025-10-22T12:00:00,GUE,old,,30,2,,0.00222,300,290,0.967,-0.01454,,",
        "test_002,2025-10-22T12:00:00,GUE,old,,30,3,,0.00333,300,270,0.900,-0.04576,,",
        "test_002,2025-10-22T12:00:00,GUE,old,,30,4,,0.00444,300,240,0.800,-0.09691,,",
        # Krylov criterion
        "test_002,2025-10-22T12:00:00,GUE,krylov,,30,2,2,0.00222,300,260,0.867,-0.06213,,",
        "test_002,2025-10-22T12:00:00,GUE,krylov,,30,3,3,0.00333,300,220,0.733,-0.13490,,",
        "test_002,2025-10-22T12:00:00,GUE,krylov,,30,4,4,0.00444,300,190,0.633,-0.19859,,",
    ]

    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

    print(f"✓ Created mock K-sweep CSV: {path}")


def run_plot_from_csv(csv_path: str, plot_type: str, y_axis: str, outdir: str) -> bool:
    """Run plot-from-csv command and return success status."""
    cmd = [
        "python", "-m", "reach.cli",
        "plot-from-csv",
        "--csv", csv_path,
        "--type", plot_type,
        "--ensemble", "GUE",
        "--y", y_axis,
        "--outdir", outdir,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"✗ Command failed: {' '.join(cmd)}")
            print(f"  stderr: {result.stderr}")
            return False

        print(f"✓ Generated {plot_type} plots for y={y_axis}")
        return True

    except subprocess.TimeoutExpired:
        print(f"✗ Command timed out: {' '.join(cmd)}")
        return False
    except Exception as e:
        print(f"✗ Command error: {e}")
        return False


def check_files_exist(outdir: str, patterns: list) -> list:
    """Check which files exist matching the given patterns."""
    found = []
    outpath = Path(outdir)

    for pattern in patterns:
        matches = list(outpath.glob(pattern))
        found.extend(matches)

    return found


def main():
    print("=" * 60)
    print("STREAMING MODE VALIDATION")
    print("=" * 60)
    print()

    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        test_outdir = os.path.join(tmpdir, "test_output")
        os.makedirs(test_outdir, exist_ok=True)

        # Create mock CSVs
        density_csv = os.path.join(tmpdir, "mock_density.csv")
        k_csv = os.path.join(tmpdir, "mock_k.csv")

        create_mock_density_csv(density_csv)
        create_mock_k_csv(k_csv)
        print()

        # Test 1: Density plots (unreachable)
        print("[Test 1/4] Density plots (unreachable)...")
        success_1 = run_plot_from_csv(density_csv, "density", "unreachable", test_outdir)
        print()

        # Test 2: Density plots (reachable)
        print("[Test 2/4] Density plots (reachable)...")
        success_2 = run_plot_from_csv(density_csv, "density", "reachable", test_outdir)
        print()

        # Test 3: K-sweep plots (unreachable)
        print("[Test 3/4] K-sweep plots (unreachable)...")
        success_3 = run_plot_from_csv(k_csv, "k-multi-tau", "unreachable", test_outdir)
        print()

        # Test 4: K-sweep plots (reachable)
        print("[Test 4/4] K-sweep plots (reachable)...")
        success_4 = run_plot_from_csv(k_csv, "k-multi-tau", "reachable", test_outdir)
        print()

        # Check generated files
        print("Checking generated files...")
        density_files = check_files_exist(test_outdir, [
            "three_criteria_vs_density_GUE_tau*_unreachable.png",
            "three_criteria_vs_density_GUE_tau*_reachable.png",
        ])
        k_files = check_files_exist(test_outdir, [
            "K_sweep_multi_tau_GUE_d*_unreachable.png",
            "K_sweep_multi_tau_GUE_d*_reachable.png",
        ])

        print(f"✓ Found {len(density_files)} density plot(s)")
        print(f"✓ Found {len(k_files)} K-sweep plot(s)")
        print()

        # Summary
        all_success = success_1 and success_2 and success_3 and success_4
        all_files = density_files + k_files

        print("=" * 60)
        if all_success and len(all_files) >= 2:
            print("✅ PASS: All streaming mode tests passed!")
            print()
            print("Generated files:")
            for f in sorted(all_files):
                print(f"  - {f.name}")
            print()
            return 0
        else:
            print("❌ FAIL: Some tests failed")
            print()
            print(f"  Test 1 (density unreachable): {'PASS' if success_1 else 'FAIL'}")
            print(f"  Test 2 (density reachable): {'PASS' if success_2 else 'FAIL'}")
            print(f"  Test 3 (K-sweep unreachable): {'PASS' if success_3 else 'FAIL'}")
            print(f"  Test 4 (K-sweep reachable): {'PASS' if success_4 else 'FAIL'}")
            print(f"  Files generated: {len(all_files)} (expected >= 2)")
            print()
            return 1


if __name__ == "__main__":
    sys.exit(main())
