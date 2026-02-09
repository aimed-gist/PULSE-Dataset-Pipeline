"""
Step 3: Outlier Removal

This module identifies and removes physiologically implausible heart rate values.
Outliers are defined as HR < min_hr (default: 40) or HR > max_hr (default: 163).
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

from .config import Step3Config, REQUIRED_COLS
from .utils import (
    ensure_dir,
    load_csv_safe,
    validate_required_columns,
    normalize_time_str,
    save_dataframe,
    print_summary_stats
)


def run_step3(cfg: Step3Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 3: Remove outlier heart rate values.

    This step:
    1. Loads deduplicated heart rate data from Step 2
    2. Converts HR to numeric (Float64)
    3. Identifies outliers: HR < min_hr OR HR > max_hr
    4. Either drops outliers or replaces with 0 based on outlier_mode
    5. Exports cleaned data

    Args:
        cfg: Step3Config with input/output paths and parameters

    Returns:
        Tuple of (cleaned, summary)
            - cleaned: Heart rate data with outliers removed/replaced
            - summary: Outlier statistics
    """
    print("\n" + "=" * 70)
    print("STEP 3: Outlier Removal")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load deduplicated data from Step 2
    print(f"\nLoading input: {cfg.input_csv}")
    df = load_csv_safe(cfg.input_csv, dtype="str")
    validate_required_columns(df, REQUIRED_COLS, str(cfg.input_csv))
    print(f"  Total rows: {len(df):,}")

    # Normalize time if requested
    if cfg.normalize_time:
        print("Normalizing time strings to HH:MM:SS format...")
        df["time"] = normalize_time_str(df["time"])

    # Convert HR to numeric
    print("\nConverting HR to numeric (Float64)...")
    df["hr_num"] = pd.to_numeric(df["hr"], errors="coerce").astype("Float64")

    # Identify outliers
    print(f"\nIdentifying outliers (HR < {cfg.min_hr} or HR > {cfg.max_hr})...")

    # Create outlier flags
    df["is_too_low"] = df["hr_num"] < cfg.min_hr
    df["is_too_high"] = df["hr_num"] > cfg.max_hr
    df["is_outlier"] = df["is_too_low"] | df["is_too_high"]

    n_outliers = df["is_outlier"].sum()
    n_too_low = df["is_too_low"].sum()
    n_too_high = df["is_too_high"].sum()

    print_summary_stats("Outliers detected", len(df), n_outliers)
    print(f"    - Too low (< {cfg.min_hr}): {n_too_low:,}")
    print(f"    - Too high (> {cfg.max_hr}): {n_too_high:,}")

    # Handle outliers based on mode
    print(f"\nHandling outliers (mode: {cfg.outlier_mode})...")
    df_cleaned = df.copy()

    if cfg.outlier_mode == "drop":
        # Remove outlier rows entirely
        df_cleaned = df_cleaned[~df_cleaned["is_outlier"]].copy()
        print(f"  Dropped {n_outliers:,} outlier rows")

    elif cfg.outlier_mode == "replace_zero":
        # Replace outlier values with 0
        df_cleaned.loc[df_cleaned["is_outlier"], "hr_num"] = 0.0
        df_cleaned.loc[df_cleaned["is_outlier"], "hr"] = "0"
        print(f"  Replaced {n_outliers:,} outlier values with 0")

    # Select output columns
    output_cols = ["subject_id", "date", "time", "hr"]
    df_cleaned = df_cleaned[output_cols].copy()

    # Update hr column to match hr_num for replaced values
    if cfg.outlier_mode == "replace_zero":
        # Ensure hr column reflects the cleaned values
        pass  # Already updated above

    df_cleaned = df_cleaned.reset_index(drop=True)

    # Create summary
    summary = pd.DataFrame([{
        "input_rows": len(df),
        "outliers_detected": n_outliers,
        "outliers_too_low": n_too_low,
        "outliers_too_high": n_too_high
    }])

    # Print summary
    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(f"  Input rows: {len(df):,}")
    print(f"  Outliers detected: {n_outliers:,}")
    print(f"  Output rows: {len(df_cleaned):,}")

    if cfg.outlier_mode == "drop":
        print(f"  Rows dropped: {len(df) - len(df_cleaned):,}")
    else:
        print(f"  Values replaced with 0: {n_outliers:,}")

    # Save outputs
    print("\nSaving outputs...")
    save_dataframe(df_cleaned, cfg.outdir / "hr_1min_outlier_cleaned.csv")
    save_dataframe(summary, cfg.outdir / "step3_summary.csv")

    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED")
    print("=" * 70)

    return df_cleaned, summary


def main():
    """
    Example usage of Step 3.

    This is a template for running Step 3. Modify the paths below to match
    your data location before running.
    """
    from pathlib import Path

    # =========================================================================
    # CONFIGURATION - Modify these paths for your environment
    # =========================================================================
    input_csv = Path("output/step2/hr_1min_deduped.csv")  # Input: deduplicated data from Step 2
    outdir = Path("output/step3")                          # Output directory

    # Outlier detection settings
    min_hr = 40.0                                          # Minimum valid HR (bpm)
    max_hr = 163.0                                         # Maximum valid HR (bpm)
    outlier_mode = "replace_zero"                          # 'drop' or 'replace_zero'
    normalize_time = True                                  # Normalize time format to HH:MM:SS
    # =========================================================================

    # Create configuration
    config = Step3Config(
        input_csv=input_csv,
        outdir=outdir,
        min_hr=min_hr,
        max_hr=max_hr,
        outlier_mode=outlier_mode,
        normalize_time=normalize_time
    )

    # Run step
    cleaned, summary = run_step3(config)
    return cleaned, summary


if __name__ == "__main__":
    main()
