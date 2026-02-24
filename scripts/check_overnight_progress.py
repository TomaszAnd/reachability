#!/usr/bin/env python3
"""Check progress of overnight sweep."""
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("data/overnight_v87")
FIG_DIR = Path("fig/overnight_v87")


def check_progress():
    print(f"Checking at {datetime.now()}")
    print("=" * 50)

    # Check CSV files
    csvs = sorted(OUTPUT_DIR.glob("*.csv"))
    print(f"\nData files ({len(csvs)}):")
    for csv in csvs:
        size = csv.stat().st_size / 1024
        print(f"  {csv.name}: {size:.1f} KB")

    # Check figures
    figs = sorted(FIG_DIR.glob("*.png"))
    print(f"\nFigures ({len(figs)}):")
    for fig in figs:
        print(f"  {fig.name}")

    # Check timing log
    timing_path = OUTPUT_DIR / "timing_log.csv"
    if timing_path.exists():
        import pandas as pd
        timing = pd.read_csv(timing_path)
        print(f"\nCompleted sweeps:")
        print(timing.to_string(index=False))
        total = timing['runtime_min'].sum()
        print(f"\nTotal runtime so far: {total / 60:.2f} hours")
    else:
        print("\nNo timing log yet (sweep may still be starting)")


if __name__ == "__main__":
    check_progress()
