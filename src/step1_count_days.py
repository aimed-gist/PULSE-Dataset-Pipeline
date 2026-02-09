"""
Step 1: Count Total Collection Days

This module counts the total patient-days, expected heart rate minutes,
and recorded heart rate minutes from the raw data.

A recorded minute is defined as:
    - Valid datetime (valid 'dayN' format AND valid 'HH:MM:SS' time)
    - Numeric HR value (not NaN)
    - HR value != 0
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from .config import Step1Config, REQUIRED_COLS, EXPECTED_MINUTES_PER_DAY
from .utils import (
    ensure_dir,
    load_input,
    build_subject_day_grid,
    save_dataframe
)


def run_step1(cfg: Step1Config) -> pd.DataFrame:
    """
    Execute Step 1: Count total collection days.

    This step:
    1. Loads raw 1-minute heart rate data
    2. Filters to valid datetime + numeric HR != 0
    3. Counts total patient-days (including missing days)
    4. Calculates total expected and recorded minutes

    Args:
        cfg: Step1Config with input/output paths and parameters

    Returns:
        DataFrame with global counts (n_subjects, total_patient_days,
                                     total_expected_minutes, total_recorded_minutes)
    """
    print("\n" + "=" * 70)
    print("STEP 1: Count Total Collection Days")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load input data
    print(f"\nLoading input: {cfg.input_csv}")
    df = load_input(cfg.input_csv, REQUIRED_COLS)
    print(f"  Total rows: {len(df):,}")

    # Filter to valid minutes (ok_datetime AND hr_num not NaN AND hr_num != 0)
    valid_mask = df["ok_datetime"] & ~df["hr_num"].isna() & (df["hr_num"] != 0)
    df_valid = df[valid_mask].copy()
    print(f"  Valid recorded minutes: {len(df_valid):,}")

    # Count recorded minutes per patient-day
    print("\nCounting recorded minutes per patient-day...")
    recorded = (
        df_valid.groupby(["subject_id", "day_index"])
        .size()
        .reset_index(name="recorded_minutes")
    )

    # Build complete patient-day grid
    print("Building complete patient-day grid...")
    subject_maxday = df_valid.groupby("subject_id")["day_index"].max()
    grid = build_subject_day_grid(subject_maxday, cfg.day_start, cfg.day_end)

    # Merge grid with recorded counts
    patient_day = grid.merge(recorded, on=["subject_id", "day_index"], how="left")
    patient_day["recorded_minutes"] = patient_day["recorded_minutes"].fillna(0).astype(int)

    # Calculate global summary
    print("Calculating global summary...")
    n_subjects = patient_day["subject_id"].nunique()
    total_patient_days = len(patient_day)
    total_expected_minutes = total_patient_days * EXPECTED_MINUTES_PER_DAY
    total_recorded_minutes = int(patient_day["recorded_minutes"].sum())

    global_counts = pd.DataFrame([{
        "n_subjects": n_subjects,
        "total_patient_days": total_patient_days,
        "total_expected_minutes": total_expected_minutes,
        "total_recorded_minutes": total_recorded_minutes
    }])

    # Print summary
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(f"  Number of subjects: {n_subjects:,}")
    print(f"  Total patient-days: {total_patient_days:,}")
    print(f"  Total expected minutes: {total_expected_minutes:,}")
    print(f"  Total recorded minutes: {total_recorded_minutes:,}")
    recording_ratio = total_recorded_minutes / total_expected_minutes if total_expected_minutes > 0 else 0
    print(f"  Overall recording ratio: {recording_ratio:.4f}")

    # Save output
    print("\nSaving output...")
    save_dataframe(global_counts, cfg.outdir / "global_counts.csv")

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETED")
    print("=" * 70)

    return global_counts


def main():
    """
    Example usage of Step 1.

    This is a template for running Step 1. Modify the paths below to match
    your data location before running.
    """
    from pathlib import Path

    # =========================================================================
    # CONFIGURATION - Modify these paths for your environment
    # =========================================================================
    input_csv = Path("data/heartrate_minute.csv")  # Input: 1-minute HR data
    outdir = Path("output/step1")                   # Output directory

    # Optional: Set day range (None = use all available days)
    day_start = 0      # Starting day index (default: 0)
    day_end = None     # Ending day index (default: None = max observed day)
    # =========================================================================

    # Create configuration
    config = Step1Config(
        input_csv=input_csv,
        outdir=outdir,
        day_start=day_start,
        day_end=day_end
    )

    # Run step
    global_counts = run_step1(config)
    return global_counts


if __name__ == "__main__":
    main()
