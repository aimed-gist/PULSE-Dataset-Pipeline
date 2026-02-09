"""
Step 5: Valid Participant Filtering

This module filters participants based on consecutive valid days.
A valid participant must have at least N consecutive valid days (default: 7).
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from .config import Step5Config
from .utils import (
    ensure_dir,
    load_csv_safe,
    validate_required_columns,
    longest_consecutive_run,
    save_dataframe,
    print_summary_stats
)


def run_step5(cfg: Step5Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 5: Filter valid participants based on consecutive valid days.

    This step:
    1. Loads daily adherence data from Step 4
    2. Finds longest consecutive run of valid days per subject
    3. Applies minimum consecutive day threshold (default: 7 days)
    4. Splits subjects into PASS and FAIL groups
    5. Filters daily data to only valid participants

    Args:
        cfg: Step5Config with input/output paths and parameters

    Returns:
        Tuple of (summary, daily_pass, subjects_pass, subjects_fail)
            - summary: All subjects with consecutive run statistics
            - daily_pass: All days for valid participants only
            - subjects_pass: Valid participants (>= min consecutive days)
            - subjects_fail: Excluded participants (< min consecutive days)
    """
    print("\n" + "=" * 70)
    print("STEP 5: Valid Participant Filtering")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load daily adherence data from Step 4
    print(f"\nLoading input: {cfg.input_csv}")
    daily = load_csv_safe(cfg.input_csv, dtype="str")

    required_cols = ["subject_id", "day_index", "is_valid_day"]
    validate_required_columns(daily, required_cols, str(cfg.input_csv))

    # Convert types
    daily["day_index"] = pd.to_numeric(daily["day_index"], errors="coerce").astype("Int64")
    daily["is_valid_day"] = daily["is_valid_day"].astype(str).str.lower().isin(["true", "1"])

    print(f"  Total patient-days: {len(daily):,}")
    print(f"  Number of subjects: {daily['subject_id'].nunique():,}")

    # Optionally filter to day range
    if cfg.day_start is not None or cfg.day_end is not None:
        print("\nFiltering to specified day range...")
        if cfg.day_start is not None:
            daily = daily[daily["day_index"] >= cfg.day_start]
        if cfg.day_end is not None:
            daily = daily[daily["day_index"] <= cfg.day_end]
        print(f"  Remaining patient-days: {len(daily):,}")

    # Find longest consecutive run of valid days per subject
    print(f"\nFinding longest consecutive run of valid days per subject...")
    print(f"  Minimum consecutive days threshold: {cfg.min_consecutive_days}")

    results = []
    for subject_id, group in daily.groupby("subject_id"):
        # Get valid days sorted by day_index
        valid_days = group[group["is_valid_day"]].sort_values("day_index")
        valid_day_indices = valid_days["day_index"].values

        # Find longest consecutive run
        max_len, start_day, end_day = longest_consecutive_run(valid_day_indices)

        results.append({
            "subject_id": subject_id,
            "max_run_len": max_len,
            "best_run_start_day": start_day,
            "best_run_end_day": end_day,
            "pass": max_len >= cfg.min_consecutive_days
        })

    summary = pd.DataFrame(results)
    summary = summary.sort_values("subject_id").reset_index(drop=True)

    # Split into pass and fail
    subjects_pass = summary[summary["pass"]].copy().reset_index(drop=True)
    subjects_fail = summary[~summary["pass"]].copy().reset_index(drop=True)

    n_pass = len(subjects_pass)
    n_fail = len(subjects_fail)
    n_total = len(summary)

    print_summary_stats("Valid participants", n_total, n_pass)
    print(f"    - Failed participants: {n_fail:,}")

    # Filter daily data to only pass subjects
    pass_subject_ids = set(subjects_pass["subject_id"])
    daily_pass = daily[daily["subject_id"].isin(pass_subject_ids)].copy()
    daily_pass = daily_pass.sort_values(["subject_id", "day_index"]).reset_index(drop=True)

    print(f"\n  Patient-days for valid participants: {len(daily_pass):,}")

    # Calculate additional summary statistics
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(f"  Total subjects: {n_total:,}")
    print(f"  Valid participants (>= {cfg.min_consecutive_days} consecutive days): {n_pass:,}")
    print(f"  Excluded participants: {n_fail:,}")
    print(f"  Pass rate: {n_pass / n_total * 100:.2f}%")

    if n_pass > 0:
        print(f"\n  Consecutive run statistics for valid participants:")
        print(f"    - Mean: {subjects_pass['max_run_len'].mean():.2f} days")
        print(f"    - Median: {subjects_pass['max_run_len'].median():.0f} days")
        print(f"    - Min: {subjects_pass['max_run_len'].min():.0f} days")
        print(f"    - Max: {subjects_pass['max_run_len'].max():.0f} days")

    # Save outputs
    print("\nSaving outputs...")
    save_dataframe(subjects_pass[["subject_id"]], cfg.outdir / "subjects_pass.csv")
    save_dataframe(subjects_fail[["subject_id"]], cfg.outdir / "subjects_fail.csv")
    save_dataframe(daily_pass, cfg.outdir / "daily_adherence_pass_subjects.csv")

    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED")
    print("=" * 70)

    return summary, daily_pass, subjects_pass, subjects_fail


def main():
    """
    Example usage of Step 5.

    This is a template for running Step 5. Modify the paths below to match
    your data location before running.
    """
    from pathlib import Path

    # =========================================================================
    # CONFIGURATION - Modify these paths for your environment
    # =========================================================================
    input_csv = Path("output/step4/daily_adherence.csv")  # Input: daily adherence data from Step 4
    outdir = Path("output/step5")                          # Output directory

    # Valid participant criteria
    min_consecutive_days = 7                               # Minimum consecutive valid days required
    day_start = None                                       # Optional: Starting day index (None = use all)
    day_end = None                                         # Optional: Ending day index (None = use all)
    # =========================================================================

    # Create configuration
    config = Step5Config(
        input_csv=input_csv,
        outdir=outdir,
        min_consecutive_days=min_consecutive_days,
        day_start=day_start,
        day_end=day_end
    )

    # Run step
    summary, daily_pass, subjects_pass, subjects_fail = run_step5(config)
    return summary, daily_pass, subjects_pass, subjects_fail


if __name__ == "__main__":
    main()
