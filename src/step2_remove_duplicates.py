"""
Step 2: Duplicate Removal

This module identifies and removes duplicate 1-minute heart rate records.
Duplicates are identified by (subject_id, date, time) key.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

from .config import Step2Config, REQUIRED_COLS
from .utils import (
    ensure_dir,
    load_input,
    build_subject_day_grid,
    save_dataframe
)


def run_step2(cfg: Step2Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 2: Remove duplicate heart rate records.

    Returns:
        Tuple of (df_deduped, summary)
            - df_deduped: Deduplicated HR data
            - summary: Summary statistics
    """
    print("\n" + "=" * 70)
    print("STEP 2: Duplicate Removal")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load input data
    print(f"\nLoading input: {cfg.input_csv}")
    df = load_input(cfg.input_csv, REQUIRED_COLS)
    print(f"  Total rows: {len(df):,}")

    # Apply same filter as Step 1
    print("\nApplying Step 1 filters (valid datetime + numeric HR != 0)...")
    valid_mask = df["ok_datetime"] & ~df["hr_num"].isna() & (df["hr_num"] != 0)
    df_filtered = df[valid_mask].copy()
    print(f"  Valid rows: {len(df_filtered):,}")

    # Remove duplicates (keep first or last)
    print(f"\nRemoving duplicates by {cfg.dedup_key} (keeping {cfg.keep_policy})...")
    dup_removed = df_filtered.duplicated(subset=cfg.dedup_key, keep=cfg.keep_policy)
    n_removed = int(dup_removed.sum())

    df_deduped = df_filtered[~dup_removed].copy()
    print(f"  Removed: {n_removed:,} duplicate records")
    print(f"  Remaining: {len(df_deduped):,} unique records")

    # Calculate non-wear days
    print("\nCalculating non-wear days...")

    # Build complete patient-day grid
    subject_maxday = df_deduped.groupby("subject_id")["day_index"].max()
    grid = build_subject_day_grid(subject_maxday, day_start=0, day_end=None)
    total_patient_days = len(grid)

    # Count days with actual records
    days_with_records = df_deduped[["subject_id", "day_index"]].drop_duplicates()
    n_days_with_records = len(days_with_records)

    # Non-wear days = total days - days with records
    non_wear_days = total_patient_days - n_days_with_records

    print(f"  Total patient-days: {total_patient_days:,}")
    print(f"  Days with records: {n_days_with_records:,}")
    print(f"  Non-wear days: {non_wear_days:,}")

    # Create summary
    summary = pd.DataFrame([{
        "filtered_rows": len(df_deduped),
        "removed_duplicates_rows": n_removed,
        "non_wear_days": non_wear_days
    }])

    # Select output columns
    output_cols = ["subject_id", "date", "time", "hr"]
    df_deduped = df_deduped[output_cols].reset_index(drop=True)

    # Print summary
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(f"  Filtered rows (after deduplication): {len(df_deduped):,}")
    print(f"  Removed duplicates rows: {n_removed:,}")
    print(f"  Non-wear days: {non_wear_days:,}")

    # Save outputs
    print("\nSaving outputs...")
    save_dataframe(df_deduped, cfg.outdir / "hr_1min_deduped.csv")
    save_dataframe(summary, cfg.outdir / "step2_summary.csv")

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED")
    print("=" * 70)

    return df_deduped, summary


def main():
    """
    Example usage of Step 2.

    This is a template for running Step 2. Modify the paths below to match
    your data location before running.
    """
    from pathlib import Path

    # =========================================================================
    # CONFIGURATION - Modify these paths for your environment
    # =========================================================================
    input_csv = Path("data/heartrate_minute.csv")  # Input: 1-minute HR data (raw or from Step 1)
    outdir = Path("output/step2")                   # Output directory

    # Deduplication settings
    dedup_key = ["subject_id", "date", "time"]     # Columns to identify duplicates
    keep_policy = "last"                            # Keep 'first' or 'last' duplicate
    normalize_time = True                           # Normalize time format to HH:MM:SS
    # =========================================================================

    # Create configuration
    config = Step2Config(
        input_csv=input_csv,
        outdir=outdir,
        dedup_key=dedup_key,
        keep_policy=keep_policy,
        normalize_time=normalize_time
    )

    # Run step
    df_deduped, summary = run_step2(config)
    return df_deduped, summary


if __name__ == "__main__":
    main()
