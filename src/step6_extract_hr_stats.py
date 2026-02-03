"""
Step 6: HR Statistics Extraction

This module extracts daily heart rate summary statistics (min, max, mean)
for valid days only.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from typing import Tuple
from .config import Step6Config, REQUIRED_COLS
from .utils import (
    ensure_dir,
    load_csv_safe,
    validate_required_columns,
    parse_day_index,
    save_dataframe
)


def load_valid_day_keys(key_csv: Path, filter_using_is_valid_day: bool = True) -> pd.DataFrame:
    """
    Load valid day keys from Step 4 output.

    Args:
        key_csv: Path to daily_adherence CSV from Step 4
        filter_using_is_valid_day: If True, filter to is_valid_day == True

    Returns:
        DataFrame with subject_id, day_index, date for valid days
    """
    keys = load_csv_safe(key_csv, dtype="str")

    required_cols = ["subject_id", "day_index", "date"]
    if filter_using_is_valid_day:
        required_cols.append("is_valid_day")

    validate_required_columns(keys, required_cols, str(key_csv))

    # Convert types
    keys["day_index"] = pd.to_numeric(keys["day_index"], errors="coerce").astype("Int64")

    if filter_using_is_valid_day:
        keys["is_valid_day"] = keys["is_valid_day"].astype(str).str.lower().isin(["true", "1"])
        keys = keys[keys["is_valid_day"]]

    return keys[["subject_id", "day_index", "date"]]


def load_hr_1min(hr_1min_csv: Path) -> pd.DataFrame:
    """
    Load 1-minute heart rate data from Step 3 output.

    Args:
        hr_1min_csv: Path to outlier-cleaned HR data

    Returns:
        DataFrame with subject_id, day_index, hr_num
    """
    hr = load_csv_safe(hr_1min_csv, dtype="str")
    validate_required_columns(hr, REQUIRED_COLS, str(hr_1min_csv))

    # Parse day index
    hr["day_index"], ok_day = parse_day_index(hr["date"])

    # Convert HR to numeric
    hr["hr_num"] = pd.to_numeric(hr["hr"], errors="coerce").astype("Float64")

    # Filter to valid rows (ok day and HR > 0)
    hr = hr[ok_day & (hr["hr_num"] > 0)].copy()

    return hr[["subject_id", "day_index", "hr_num"]]


def run_step6(cfg: Step6Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 6: Extract daily HR statistics for valid days.

    This step:
    1. Loads valid day keys from Step 4
    2. Loads 1-minute HR data from Step 3
    3. Filters HR data to valid days only
    4. Calculates daily statistics: mean, max, min
    5. Computes summary statistics with SD
    6. Exports daily HR stats and summary

    Args:
        cfg: Step6Config with input/output paths and parameters

    Returns:
        Tuple of (daily_stats, summary)
            - daily_stats: Daily HR statistics for each valid day
            - summary: Overall summary statistics with mean and SD
    """
    print("\n" + "=" * 70)
    print("STEP 6: HR Statistics Extraction")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load valid day keys
    print(f"\nLoading valid day keys: {cfg.key_csv}")
    keys = load_valid_day_keys(cfg.key_csv, cfg.filter_using_is_valid_day)
    print(f"  Valid days: {len(keys):,}")

    # Load 1-minute HR data
    print(f"\nLoading 1-minute HR data: {cfg.hr_1min_csv}")
    hr = load_hr_1min(cfg.hr_1min_csv)
    print(f"  Total HR minutes (HR > 0): {len(hr):,}")

    # Merge with valid day keys to filter to valid days only
    print("\nFiltering to valid days only...")
    hr_valid = hr.merge(
        keys[["subject_id", "day_index"]],
        on=["subject_id", "day_index"],
        how="inner"
    )
    print(f"  HR minutes on valid days: {len(hr_valid):,}")

    # Calculate daily statistics
    print("\nCalculating daily HR statistics (mean, max, min)...")
    daily_stats = hr_valid.groupby(["subject_id", "day_index"])["hr_num"].agg([
        ("daily_mean", "mean"),
        ("daily_max", "max"),
        ("daily_min", "min")
    ]).reset_index()

    # Merge with keys to get date
    daily_stats = keys.merge(daily_stats, on=["subject_id", "day_index"], how="left")

    # Sort by subject and day
    daily_stats = daily_stats.sort_values(["subject_id", "day_index"]).reset_index(drop=True)

    # Reorder columns
    daily_stats = daily_stats[["subject_id", "date", "daily_mean", "daily_max", "daily_min"]]

    # Calculate summary statistics (mean and SD)
    print("\nCalculating summary statistics...")
    valid_stats = daily_stats.dropna(subset=["daily_mean"])

    n_subjects = keys["subject_id"].nunique()
    n_valid_days = len(keys)
    n_days_with_hr = len(valid_stats)
    n_missing_hr = n_valid_days - n_days_with_hr

    summary = pd.DataFrame([{
        "n_subjects": n_subjects,
        "n_valid_days": n_valid_days,
        "n_days_with_hr_data": n_days_with_hr,
        "n_days_missing_hr_data": n_missing_hr,

        "mean_daily_mean_hr": round(valid_stats["daily_mean"].mean(), 1),
        "sd_daily_mean_hr": round(valid_stats["daily_mean"].std(ddof=1), 1),

        "mean_daily_max_hr": round(valid_stats["daily_max"].mean(), 1),
        "sd_daily_max_hr": round(valid_stats["daily_max"].std(ddof=1), 1),

        "mean_daily_min_hr": round(valid_stats["daily_min"].mean(), 1),
        "sd_daily_min_hr": round(valid_stats["daily_min"].std(ddof=1), 1),
    }])

    # Print summary
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(f"  Number of subjects: {n_subjects:,}")
    print(f"  Valid days (total): {n_valid_days:,}")
    print(f"  Valid days with HR data: {n_days_with_hr:,}")
    print(f"  Valid days missing HR data: {n_missing_hr:,}")

    if n_days_with_hr > 0:
        print(f"\n  Daily Mean HR:")
        print(f"    - Mean: {summary['mean_daily_mean_hr'].iloc[0]:.1f} bpm")
        print(f"    - SD: {summary['sd_daily_mean_hr'].iloc[0]:.1f} bpm")

        print(f"\n  Daily Max HR:")
        print(f"    - Mean: {summary['mean_daily_max_hr'].iloc[0]:.1f} bpm")
        print(f"    - SD: {summary['sd_daily_max_hr'].iloc[0]:.1f} bpm")

        print(f"\n  Daily Min HR:")
        print(f"    - Mean: {summary['mean_daily_min_hr'].iloc[0]:.1f} bpm")
        print(f"    - SD: {summary['sd_daily_min_hr'].iloc[0]:.1f} bpm")

    # Save outputs
    print("\nSaving outputs...")
    daily_output_path = cfg.outdir / cfg.out_csv_name
    summary_output_path = cfg.outdir / "step6_summary.csv"

    save_dataframe(daily_stats, daily_output_path)
    save_dataframe(summary, summary_output_path)

    print("\n" + "=" * 70)
    print("STEP 6 COMPLETED")
    print("=" * 70)

    return daily_stats, summary


def main():
    """Example usage of Step 6."""
    from pathlib import Path

    # Configure paths
    base_dir = Path("/app/ai_worker/data_anonymization")
    key_csv = base_dir / "10-code/Processed_wearable_dataset/step4/daily_adherence.csv"
    hr_1min_csv = base_dir / "10-code/Processed_wearable_dataset/step3/hr_1min_outlier_cleaned.csv"
    outdir = base_dir / "10-code/Processed_wearable_dataset/step6"

    # Create configuration
    config = Step6Config(
        key_csv=key_csv,
        hr_1min_csv=hr_1min_csv,
        outdir=outdir,
        out_csv_name="daily_hr_stats_valid_days.csv",
        filter_using_is_valid_day=True
    )

    # Run step
    daily_stats, summary = run_step6(config)


if __name__ == "__main__":
    main()
