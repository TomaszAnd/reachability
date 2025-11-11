"""
CSV logging utilities for quantum reachability analysis.

This module provides simple CSV logging helpers with no external dependencies
beyond Python's standard library (csv, pathlib).

Key Features:
- Automatic header creation on first write
- Append-mode for incremental data collection
- Field ordering preservation for consistent column layout
- Parent directory creation as needed

Schema for reachability CSV logs:
    run_id: Unique identifier for this experimental run (UUID or timestamp-based)
    timestamp: ISO 8601 timestamp when row was written
    ensemble: Random matrix ensemble ("GOE" or "GUE")
    criterion: Reachability criterion ("spectral", "moment", or "krylov")
    tau: Spectral-overlap threshold (filled for spectral only; empty for others)
    d: Hilbert space dimension
    K: Number of Hamiltonians (control parameters)
    m: Krylov rank (filled for m-sweeps; empty for K-sweeps)
    rho_K_over_d2: Normalized control density K/d² (for density plots)
    trials: Total number of Monte Carlo trials (nks × nst)
    successes_unreach: Count of "unreachable" outcomes in MC batch
    p_unreach: Probability of unreachability (successes/trials)
    log10_p_unreach: log₁₀(max(p_unreach, DISPLAY_FLOOR))
    mean_best_overlap: Mean of best spectral overlap values (spectral only)
    sem_best_overlap: SEM of best spectral overlap values (spectral only)

Glossary:
    d: Hilbert space dimension
    K: Number of Hamiltonians (control parameters) for H(λ) = Σᵢ λᵢHᵢ
    n: For CSV consistency with literature figures, set n := K (alias only)
    ρ = K/d²: Normalized control density (replaces n/D² from literature)
    τ: Spectral-overlap threshold for spectral criterion
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str) -> None:
    """
    Ensure parent directory of a file path exists, creating it if necessary.

    Args:
        path: File path (not directory path)

    Example:
        ensure_parent_dir("data/output/results.csv")  # Creates data/output/ if needed
    """
    parent = Path(path).parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured parent directory exists: {parent}")


def append_rows_csv(
    path: str,
    rows: List[Dict[str, Any]],
    field_order: List[str],
) -> None:
    """
    Append rows to CSV file, creating it with header if it doesn't exist.

    This function is idempotent: safe to call multiple times on the same file.
    - On first call: creates file with header + data rows
    - On subsequent calls: appends only data rows

    Args:
        path: Path to CSV file (parent directory will be created if needed)
        rows: List of dictionaries with data to append
        field_order: Ordered list of field names (defines column order)

    Raises:
        ValueError: If any row has keys not in field_order

    Example:
        >>> field_order = ["run_id", "timestamp", "ensemble", "p_unreach"]
        >>> rows = [
        ...     {"run_id": "abc123", "timestamp": "2025-01-15T10:30:00",
        ...      "ensemble": "GOE", "p_unreach": 0.123},
        ... ]
        >>> append_rows_csv("results.csv", rows, field_order)
    """
    if not rows:
        logger.debug(f"No rows to write to {path}")
        return

    # Validate rows have only expected fields
    for i, row in enumerate(rows):
        extra_keys = set(row.keys()) - set(field_order)
        if extra_keys:
            raise ValueError(
                f"Row {i} has unexpected keys not in field_order: {extra_keys}"
            )

    # Ensure parent directory exists
    ensure_parent_dir(path)

    # Check if file exists (determines whether to write header)
    file_exists = os.path.isfile(path)

    # Write rows
    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()
            logger.info(f"Created CSV with header: {path}")

        writer.writerows(rows)
        logger.debug(f"Appended {len(rows)} row(s) to {path}")


# Standard field order for reachability CSV logs
REACHABILITY_CSV_FIELDS = [
    "run_id",
    "timestamp",
    "ensemble",
    "criterion",
    "tau",
    "d",
    "K",
    "m",
    "rho_K_over_d2",
    "trials",
    "successes_unreach",
    "p_unreach",
    "log10_p_unreach",
    "mean_best_overlap",
    "sem_best_overlap",
]


class StreamingCSVWriter:
    """
    Buffered CSV writer for streaming mode with periodic flushing.

    This class enables:
    - Incremental CSV writing during long computations
    - Automatic flushing every N rows
    - Graceful handling of interrupts (flush remaining rows on cleanup)
    - Resumable runs (appends to existing files)

    Usage:
        with StreamingCSVWriter("output.csv", flush_every=10) as writer:
            for data_point in computation():
                row = {...}
                writer.write_row(row)
        # Remaining buffer automatically flushed on exit
    """

    def __init__(
        self,
        path: str,
        field_order: List[str] = REACHABILITY_CSV_FIELDS,
        flush_every: int = 10,
    ):
        """
        Initialize streaming CSV writer.

        Args:
            path: Path to CSV file (parent directory will be created if needed)
            field_order: Ordered list of field names (defines column order)
            flush_every: Flush buffer to disk every N rows (default: 10)
        """
        self.path = path
        self.field_order = field_order
        self.flush_every = flush_every
        self.buffer = []

        # Ensure parent directory exists
        ensure_parent_dir(path)

    def write_row(self, row: Dict[str, Any]) -> None:
        """
        Write a single row to buffer (flushes if buffer reaches threshold).

        Args:
            row: Dictionary with data to append
        """
        self.buffer.append(row)

        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Flush buffered rows to disk."""
        if not self.buffer:
            return

        append_rows_csv(self.path, self.buffer, self.field_order)
        logger.debug(f"Flushed {len(self.buffer)} row(s) to {self.path}")
        self.buffer.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush remaining buffer."""
        if self.buffer:
            logger.info(f"Flushing remaining {len(self.buffer)} row(s) to {self.path}")
            self.flush()
        return False  # Don't suppress exceptions
