"""
Step 7: Missingness Analysis

This module merges all wearable-derived metrics and assesses missing data
across different metric types for valid subject-days.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd

from .config import Step7Config
from .utils import (
    ensure_dir,
    load_csv_safe,
    validate_required_columns,
    parse_day_index,
    save_dataframe
)


def load_valid_keys(key_csv: Path, filter_valid_days: bool = True) -> pd.DataFrame:
    """
    Load valid subject-day keys from Step 5 output.

    Args:
        key_csv: Path to daily adherence data (from Step 5)
        filter_valid_days: If True, filter to is_valid_day == True

    Returns:
        DataFrame with subject_id, date, day_index
    """
    keys = load_csv_safe(key_csv, dtype="str")

    required_cols = ["subject_id", "date"]
    if filter_valid_days and "is_valid_day" in keys.columns:
        keys["is_valid_day"] = keys["is_valid_day"].astype(str).str.lower().isin(["true", "1"])
        keys = keys[keys["is_valid_day"]]

    return keys[["subject_id", "date"]].drop_duplicates()


def collapse_daily_duplicates(df: pd.DataFrame, value_cols: List[str], collapse_by_mean: bool = True) -> pd.DataFrame:
    """
    Handle duplicate subject-date rows by collapsing with mean or first.

    Args:
        df: DataFrame with potential duplicates
        value_cols: Columns containing metric values
        collapse_by_mean: If True, use mean; else use first

    Returns:
        DataFrame with no duplicate (subject_id, date) pairs
    """
    # Check for duplicates
    dup_mask = df.duplicated(subset=["subject_id", "date"], keep=False)

    if not dup_mask.any():
        return df

    # Separate duplicates and non-duplicates
    df_unique = df[~dup_mask].copy()
    df_dup = df[dup_mask].copy()

    # Convert value columns to numeric
    for col in value_cols:
        if col in df_dup.columns:
            df_dup[col] = pd.to_numeric(df_dup[col], errors="coerce")

    # Collapse duplicates
    if collapse_by_mean:
        agg_dict = {col: "mean" for col in value_cols if col in df_dup.columns}
    else:
        agg_dict = {col: "first" for col in value_cols if col in df_dup.columns}

    df_collapsed = df_dup.groupby(["subject_id", "date"]).agg(agg_dict).reset_index()

    # Combine back
    result = pd.concat([df_unique, df_collapsed], ignore_index=True)
    return result.sort_values(["subject_id", "date"]).reset_index(drop=True)


def load_activity(activity_csv: Path, collapse_by_mean: bool = True) -> pd.DataFrame:
    """
    Load activity daily metrics (steps, distance, calories).

    Args:
        activity_csv: Path to activity_daily.csv
        collapse_by_mean: If True, collapse duplicates by mean

    Returns:
        DataFrame with subject_id, date, steps, distance, calories
    """
    df = load_csv_safe(activity_csv, dtype="str")

    # Expected columns
    metric_cols = ["steps", "distance", "calories"]
    available_cols = [col for col in metric_cols if col in df.columns]

    if not available_cols:
        # Return empty dataframe with expected structure
        return pd.DataFrame(columns=["subject_id", "date"] + metric_cols)

    df = df[["subject_id", "date"] + available_cols].copy()

    # Collapse duplicates if needed
    df = collapse_daily_duplicates(df, available_cols, collapse_by_mean)

    # Ensure all metric columns exist (add as NaN if missing)
    for col in metric_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[["subject_id", "date"] + metric_cols]


def load_resting_hr(resting_csv: Path, collapse_by_mean: bool = True) -> pd.DataFrame:
    """
    Load resting heart rate daily metrics.

    Args:
        resting_csv: Path to heartrate_resting_daily.csv
        collapse_by_mean: If True, collapse duplicates by mean

    Returns:
        DataFrame with subject_id, date, resting_hr
    """
    df = load_csv_safe(resting_csv, dtype="str")

    metric_cols = ["resting_hr"]
    available_cols = [col for col in metric_cols if col in df.columns]

    if not available_cols:
        return pd.DataFrame(columns=["subject_id", "date"] + metric_cols)

    df = df[["subject_id", "date"] + available_cols].copy()
    df = collapse_daily_duplicates(df, available_cols, collapse_by_mean)

    for col in metric_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[["subject_id", "date"] + metric_cols]


def load_daily_hr(daily_hr_csv: Path, collapse_by_mean: bool = True) -> pd.DataFrame:
    """
    Load daily HR summary statistics (from Step 6 or original source).

    Args:
        daily_hr_csv: Path to daily HR stats CSV
        collapse_by_mean: If True, collapse duplicates by mean

    Returns:
        DataFrame with subject_id, date, daily_mean, daily_max, daily_min
    """
    df = load_csv_safe(daily_hr_csv, dtype="str")

    metric_cols = ["daily_mean", "daily_max", "daily_min"]
    available_cols = [col for col in metric_cols if col in df.columns]

    if not available_cols:
        return pd.DataFrame(columns=["subject_id", "date"] + metric_cols)

    df = df[["subject_id", "date"] + available_cols].copy()
    df = collapse_daily_duplicates(df, available_cols, collapse_by_mean)

    for col in metric_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[["subject_id", "date"] + metric_cols]


def load_sleep(sleep_csv: Path, collapse_by_mean: bool = True) -> pd.DataFrame:
    """
    Load sleep daily metrics.

    Args:
        sleep_csv: Path to sleep_summary_daily.csv
        collapse_by_mean: If True, collapse duplicates by mean

    Returns:
        DataFrame with subject_id, date, and sleep metrics
    """
    df = load_csv_safe(sleep_csv, dtype="str")

    metric_cols = [
        "total_minutes_asleep", "total_time_in_bed",
        "stages_deep", "stages_light", "stages_rem", "stages_wake",
        "stages_asleep", "stages_restless", "stages_awake",
        "cnt_deep", "cnt_light", "cnt_rem", "cnt_wake",
        "cnt_asleep", "cnt_restless", "cnt_awake"
    ]

    available_cols = [col for col in metric_cols if col in df.columns]

    if not available_cols:
        return pd.DataFrame(columns=["subject_id", "date"] + metric_cols)

    df = df[["subject_id", "date"] + available_cols].copy()
    df = collapse_daily_duplicates(df, available_cols, collapse_by_mean)

    for col in metric_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df[["subject_id", "date"] + metric_cols]


def missingness_report(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """
    Generate missingness summary for specified metrics.

    Args:
        df: DataFrame with metrics
        metrics: List of metric column names to analyze

    Returns:
        DataFrame with missingness statistics per metric
    """
    results = []

    total_keys = len(df)

    for metric in metrics:
        if metric not in df.columns:
            results.append({
                "metric": metric,
                "missing_count": total_keys,
                "missing_rate": 1.0,
                "total_keys": total_keys
            })
        else:
            missing_count = df[metric].isna().sum()
            missing_rate = missing_count / total_keys if total_keys > 0 else 0

            results.append({
                "metric": metric,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "total_keys": total_keys
            })

    report = pd.DataFrame(results)

    return report


def run_step7(cfg: Step7Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Step 7: Merge metrics and assess missingness.

    This step:
    1. Loads valid subject-day keys from Step 5
    2. Loads all daily metric sources (activity, resting HR, daily HR, sleep)
    3. Merges all metrics onto valid keys
    4. Calculates missingness statistics for each metric
    5. Exports merged data and missingness report

    Args:
        cfg: Step7Config with input/output paths and parameters

    Returns:
        Tuple of (merged_metrics, missingness_report)
            - merged_metrics: All metrics merged on valid subject-days
            - missingness_report: Summary of missing data
    """
    print("\n" + "=" * 70)
    print("STEP 7: Missingness Analysis")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    # Load valid keys
    print(f"\nLoading valid subject-day keys: {cfg.key_csv}")
    keys = load_valid_keys(cfg.key_csv, cfg.filter_valid_days)
    print(f"  Valid subject-days: {len(keys):,}")
    print(f"  Number of subjects: {keys['subject_id'].nunique():,}")

    # Load all metric sources
    print("\nLoading metric sources...")

    print(f"  Loading activity: {cfg.activity_csv}")
    activity = load_activity(cfg.activity_csv, cfg.collapse_duplicates_by_mean)

    print(f"  Loading resting HR: {cfg.resting_csv}")
    resting = load_resting_hr(cfg.resting_csv, cfg.collapse_duplicates_by_mean)

    print(f"  Loading daily HR stats: {cfg.daily_hr_csv}")
    daily_hr = load_daily_hr(cfg.daily_hr_csv, cfg.collapse_duplicates_by_mean)

    print(f"  Loading sleep: {cfg.sleep_csv}")
    sleep = load_sleep(cfg.sleep_csv, cfg.collapse_duplicates_by_mean)

    # Merge all metrics onto valid keys
    print("\nMerging all metrics onto valid keys...")
    merged = keys.copy()

    merged = merged.merge(activity, on=["subject_id", "date"], how="left")
    merged = merged.merge(resting, on=["subject_id", "date"], how="left")
    merged = merged.merge(daily_hr, on=["subject_id", "date"], how="left")
    merged = merged.merge(sleep, on=["subject_id", "date"], how="left")

    # Sort by subject and date
    merged = merged.sort_values(["subject_id", "date"]).reset_index(drop=True)

    print(f"  Merged rows: {len(merged):,}")

    # Generate missingness report
    print("\nGenerating missingness report...")
    report = missingness_report(merged, cfg.metrics)

    # Print summary
    print("\n" + "-" * 70)
    print("Missingness Summary:")
    print("-" * 70)
    for _, row in report.head(10).iterrows():
        print(f"  {row['metric']:30s}: {row['missing_count']:6,} / {row['total_keys']:6,} ({row['missing_rate']:6.2%})")

    if len(report) > 10:
        print(f"  ... and {len(report) - 10} more metrics")

    # Save outputs
    print("\nSaving outputs...")
    save_dataframe(report, cfg.outdir / "step7_summary.csv")

    print("\n" + "=" * 70)
    print("STEP 7 COMPLETED")
    print("=" * 70)

    return merged, report


def main():
    """
    Example usage of Step 7.

    This is a template for running Step 7. Modify the paths below to match
    your data location before running.
    """
    from pathlib import Path

    # =========================================================================
    # CONFIGURATION - Modify these paths for your environment
    # =========================================================================
    # Input from previous pipeline steps
    key_csv = Path("output/step5/daily_adherence_pass_subjects.csv")     # Valid participants from Step 5
    daily_hr_csv = Path("output/step6/daily_hr_stats__valid_days.csv")   # HR stats from Step 6

    # Additional data sources (daily activity and sleep data)
    activity_csv = Path("data/activity_daily.csv")                        # Daily activity data
    resting_csv = Path("data/heartrate_resting_daily.csv")                # Daily resting HR data
    sleep_csv = Path("data/sleep_summary_daily.csv")                      # Daily sleep summary data

    # Output directory
    outdir = Path("output/step7")                                          # Output directory

    # Metrics to analyze for missingness
    metrics = [
        'steps', 'distance', 'calories', 'resting_hr',
        'daily_mean', 'daily_max', 'daily_min',
        'total_minutes_asleep', 'total_time_in_bed',
        'stages_deep', 'stages_light', 'stages_rem', 'stages_wake',
        'stages_asleep', 'stages_restless', 'stages_awake',
        'cnt_deep', 'cnt_light', 'cnt_rem', 'cnt_wake',
        'cnt_asleep', 'cnt_restless', 'cnt_awake'
    ]

    # Analysis settings
    filter_valid_days = True                                               # Filter to valid days only
    collapse_duplicates_by_mean = True                                     # Average duplicate records
    # =========================================================================

    # Create configuration
    config = Step7Config(
        key_csv=key_csv,
        activity_csv=activity_csv,
        resting_csv=resting_csv,
        daily_hr_csv=daily_hr_csv,
        sleep_csv=sleep_csv,
        outdir=outdir,
        metrics=metrics,
        filter_valid_days=filter_valid_days,
        collapse_duplicates_by_mean=collapse_duplicates_by_mean
    )

    # Run step
    merged, report = run_step7(config)
    return merged, report


if __name__ == "__main__":
    main()
