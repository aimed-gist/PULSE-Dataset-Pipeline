"""
Step 4: Valid Day Filtering

This module calculates daily wear adherence and identifies valid days.
A valid day is defined as having >= 80% wear adherence (>= 1152 minutes).
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

from .config import Step4Config, REQUIRED_COLS
from .utils import (
    ensure_dir,
    load_csv_safe,
    validate_required_columns,
    normalize_time_str,
    parse_day_index,
    time_to_minute_of_day,
    build_subject_day_grid,
    save_dataframe,
    print_summary_stats
)


def run_step4(cfg: Step4Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 4: Filter valid days based on wear adherence.

    This step:
    1. Loads outlier-cleaned heart rate data from Step 3
    2. Calculates worn minutes per day (HR > 0, unique minutes)
    3. Creates complete day grid including missing days
    4. Applies adherence threshold (>= 80% or >= 1152 minutes)
    5. Marks each day as valid or invalid

    Args:
        cfg: Step4Config with input/output paths and parameters

    Returns:
        Tuple of (daily_adherence, summary)
            - daily_adherence: All days with adherence metrics
            - summary: Cohort-level summary statistics
    """
    print("\n" + "=" * 70)
    print("STEP 4: Valid Day Filtering")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load outlier-cleaned data from Step 3
    print(f"\nLoading input: {cfg.input_csv}")
    df = load_csv_safe(cfg.input_csv, dtype="str")
    validate_required_columns(df, REQUIRED_COLS, str(cfg.input_csv))
    print(f"  Total rows: {len(df):,}")

    # Normalize time if requested
    if cfg.normalize_time:
        print("Normalizing time strings to HH:MM:SS format...")
        df["time"] = normalize_time_str(df["time"])

    # Parse day index and minute of day
    print("\nParsing datetime fields...")
    df["day_index"], ok_day = parse_day_index(df["date"])
    df["minute_of_day"], ok_time = time_to_minute_of_day(df["time"])
    df["ok_datetime"] = ok_day & ok_time

    # Convert HR to numeric
    df["hr_num"] = pd.to_numeric(df["hr"], errors="coerce").astype("Float64")

    # Filter to valid datetime
    df_valid = df[df["ok_datetime"]].copy()
    print(f"  Rows with valid datetime: {len(df_valid):,}")

    # Calculate worn minutes per day (HR > 0, unique minutes)
    print("\nCalculating worn minutes per day...")
    worn = df_valid[df_valid["hr_num"] > 0].copy()

    # Count unique minutes per day (vectorized - much faster than .apply())
    worn_counts = (
        worn.groupby(["subject_id", "day_index"])["minute_of_day"]
        .nunique()
        .reset_index(name="worn_minutes")
    )

    # Get original date for each day
    day_dates = df_valid[["subject_id", "day_index", "date"]].drop_duplicates(
        subset=["subject_id", "day_index"], keep="first"
    )

    # Build complete day grid
    print("Building complete patient-day grid...")
    subject_maxday = df_valid.groupby("subject_id")["day_index"].max()

    # Optionally force minimum day to 0
    if cfg.force_min_day_to_zero:
        day_start = 0
    else:
        day_start = df_valid["day_index"].min()

    grid = build_subject_day_grid(subject_maxday, day_start=day_start, day_end=None)

    # Merge grid with dates and worn counts
    daily = grid.merge(day_dates, on=["subject_id", "day_index"], how="left")
    daily = daily.merge(worn_counts, on=["subject_id", "day_index"], how="left")

    # Fill missing worn_minutes with 0
    daily["worn_minutes"] = daily["worn_minutes"].fillna(0).astype(int)
    daily["expected_minutes"] = cfg.expected_minutes_per_day

    # Calculate adherence ratio
    daily["adherence_ratio"] = daily["worn_minutes"] / daily["expected_minutes"]

    # Determine threshold
    if cfg.threshold_minutes is not None:
        threshold_min = cfg.threshold_minutes
    else:
        threshold_min = int(cfg.expected_minutes_per_day * cfg.threshold_ratio)

    print(f"  Adherence threshold: {threshold_min} minutes ({cfg.threshold_ratio:.0%})")

    # Mark valid days
    daily["is_valid_day"] = daily["worn_minutes"] >= threshold_min

    # Sort by subject and day
    daily = daily.sort_values(["subject_id", "day_index"]).reset_index(drop=True)

    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    n_subjects = daily["subject_id"].nunique()
    total_patient_days = len(daily)
    total_valid_days = int(daily["is_valid_day"].sum())
    valid_day_ratio = total_valid_days / total_patient_days if total_patient_days > 0 else 0

    summary = pd.DataFrame([{
        "n_subjects": n_subjects,
        "total_patient_days": total_patient_days,
        "total_valid_days": total_valid_days,
        "valid_day_ratio": valid_day_ratio,
        "threshold_minutes": threshold_min,
        "threshold_ratio": cfg.threshold_ratio
    }])

    # Print summary
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    s = summary.iloc[0]
    print(f"  Number of subjects: {s['n_subjects']:,}")
    print(f"  Total patient-days: {s['total_patient_days']:,}")
    print_summary_stats("Valid days", int(s['total_patient_days']), int(s['total_valid_days']))
    print(f"  Valid day ratio: {s['valid_day_ratio']:.4f}")
    print(f"  Threshold: {s['threshold_minutes']:,} minutes ({s['threshold_ratio']:.0%})")

    # Save outputs
    print("\nSaving outputs...")
    save_dataframe(daily, cfg.outdir / "daily_adherence.csv")
    save_dataframe(summary, cfg.outdir / "step4_summary.csv")

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETED")
    print("=" * 70)

    return daily, summary


def main():
    """Example usage of Step 4."""
    from pathlib import Path

    # Configure paths
    base_dir = Path("/app/ai_worker/data_anonymization")
    input_csv = base_dir / "10-code/Processed_wearable_dataset/step3/hr_1min_outlier_cleaned.csv"
    outdir = base_dir / "10-code/Processed_wearable_dataset/step4"

    # Create configuration
    config = Step4Config(
        input_csv=input_csv,
        outdir=outdir,
        expected_minutes_per_day=1440,
        threshold_ratio=0.80,
        threshold_minutes=1152,
        force_min_day_to_zero=True,
        normalize_time=True
    )

    # Run step
    daily, summary = run_step4(config)


if __name__ == "__main__":
    main()
