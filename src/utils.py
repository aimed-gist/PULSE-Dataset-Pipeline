"""
Utility functions shared across the heart rate data pipeline.

This module contains helper functions for data loading, parsing,
and common operations used throughout the pipeline.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def normalize_time_str(s: pd.Series) -> pd.Series:
    """
    Normalize time strings to HH:MM:SS format.

    Examples:
        "09:30" -> "09:30:00"
        "09:30:00" -> "09:30:00"

    Args:
        s: Series of time strings

    Returns:
        Series of normalized time strings
    """
    # Vectorized approach: much faster than .apply() for large datasets
    s_str = s.astype(str).str.strip()

    # Count colons in each string
    colon_count = s_str.str.count(":")

    # If 2 colons (HH:MM:SS format), keep as is
    # If 1 colon (HH:MM format), add :00
    # Otherwise keep as is
    mask_two_parts = (colon_count == 1)
    result = s_str.copy()
    result[mask_two_parts] = s_str[mask_two_parts] + ":00"

    # Restore NaN values
    result[s.isna()] = pd.NA

    return result


def parse_day_index(date_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Parse day index from date strings in format 'dayN'.

    Args:
        date_series: Series of date strings (e.g., 'day0', 'day1', 'day2')

    Returns:
        Tuple of (day_index series as Int64, ok_day boolean series)
            - day_index: Numeric day index (0, 1, 2, ...)
            - ok_day: True if format is valid 'dayN'
    """
    # Vectorized approach
    s_str = date_series.astype(str).str.strip().str.lower()

    # Check if string starts with "day"
    starts_with_day = s_str.str.startswith("day")

    # Extract the numeric part after "day"
    day_index = pd.Series(pd.NA, index=date_series.index, dtype="Int64")

    if starts_with_day.any():
        # Extract substring after "day" (first 3 characters)
        numeric_part = s_str[starts_with_day].str[3:]

        # Convert to numeric, coerce errors to NA
        day_index[starts_with_day] = pd.to_numeric(numeric_part, errors="coerce").astype("Int64")

    # Handle original NaN values
    day_index[date_series.isna()] = pd.NA

    ok_day = ~pd.isna(day_index)

    return day_index, ok_day


def time_to_minute_of_day(time_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Convert HH:MM:SS time strings to minute-of-day (0-1439).

    Args:
        time_series: Series of time strings in HH:MM:SS format

    Returns:
        Tuple of (minute_of_day series as Int64, ok_time boolean series)
            - minute_of_day: Minutes since midnight (0-1439)
            - ok_time: True if time could be parsed
    """
    # Vectorized approach: convert all at once
    s_str = time_series.astype(str).str.strip()

    # Parse as datetime with format HH:MM:SS
    ts = pd.to_datetime(s_str, format="%H:%M:%S", errors="coerce")

    # Calculate minute of day: hour * 60 + minute
    minute_of_day = pd.Series(pd.NA, index=time_series.index, dtype="Int64")

    # Only calculate for valid parsed times
    valid_mask = ~ts.isna()
    if valid_mask.any():
        minute_of_day[valid_mask] = (ts[valid_mask].dt.hour * 60 + ts[valid_mask].dt.minute).astype("Int64")

    # Preserve original NaN values
    minute_of_day[time_series.isna()] = pd.NA

    ok_time = ~pd.isna(minute_of_day)

    return minute_of_day, ok_time


def load_csv_safe(csv_path: Path, dtype: str = "str") -> pd.DataFrame:
    """
    Load CSV file with safe defaults.

    Args:
        csv_path: Path to CSV file
        dtype: Default dtype for columns (default: 'str')

    Returns:
        DataFrame with lowercased column names

    Raises:
        FileNotFoundError: If CSV file doesn't exist
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=dtype)
    df.columns = df.columns.str.lower().str.strip()

    return df


def validate_required_columns(df: pd.DataFrame, required_cols: List[str], file_name: str = "Input") -> None:
    """
    Validate that DataFrame contains all required columns.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        file_name: Name of file for error message

    Raises:
        ValueError: If any required columns are missing
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{file_name} missing required columns: {missing}")


def load_input(input_csv: Path, required_cols: List[str]) -> pd.DataFrame:
    """
    Load and preprocess input heart rate CSV file.

    This function loads the CSV and adds derived columns:
        - day_index: Numeric day index parsed from date
        - minute_of_day: Minute of day (0-1439) parsed from time
        - ok_datetime: Boolean indicating valid date AND time
        - hr_num: Numeric HR value (Float64)
        - ok_hr_validminute: Boolean for valid HR minute (hr_num not NaN and != 0)

    Args:
        input_csv: Path to input CSV file
        required_cols: List of required column names

    Returns:
        DataFrame with original and derived columns

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required columns are missing
    """
    df = load_csv_safe(input_csv, dtype="str")
    validate_required_columns(df, required_cols, str(input_csv))

    # ✅ Normalize time BEFORE parsing minute_of_day so ok_datetime is correct
    # (e.g., "09:30" -> "09:30:00")
    df["time"] = normalize_time_str(df["time"])

    # Parse day index
    day_index, ok_day = parse_day_index(df["date"])
    df["day_index"] = day_index

    # Parse minute of day
    minute_of_day, ok_time = time_to_minute_of_day(df["time"])
    df["minute_of_day"] = minute_of_day

    # Combined datetime validity
    df["ok_datetime"] = ok_day & ok_time

    # Convert HR to numeric
    df["hr_num"] = pd.to_numeric(df["hr"], errors="coerce").astype("Float64")

    # Valid HR minute: not NaN and != 0
    df["ok_hr_validminute"] = ~df["hr_num"].isna() & (df["hr_num"] != 0)

    return df



def build_subject_day_grid(
    subject_maxday: pd.Series,
    day_start: int = 0,
    day_end: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a complete grid of all (subject_id, day_index) combinations.

    This ensures that missing days (with no data) are included in the analysis.

    Args:
        subject_maxday: Series with subject_id as index and max day_index as values
        day_start: Starting day index (default: 0)
        day_end: Optional ending day index (if None, use max observed day per subject)

    Returns:
        DataFrame with columns: subject_id, day_index
    """
    rows = []
    for subject_id, max_day in subject_maxday.items():
        end_day = day_end if day_end is not None else max_day
        for day_idx in range(day_start, end_day + 1):
            rows.append({"subject_id": subject_id, "day_index": day_idx})

    return pd.DataFrame(rows)


def longest_consecutive_run(days: np.ndarray) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Find the longest consecutive sequence in a sorted array of day indices.

    Args:
        days: Sorted array of day indices

    Returns:
        Tuple of (max_length, start_day, end_day)
            - max_length: Length of longest consecutive run
            - start_day: Starting day of longest run (None if no days)
            - end_day: Ending day of longest run (None if no days)
    """
    if len(days) == 0:
        return 0, None, None

    max_len = 1
    best_start = days[0]
    best_end = days[0]

    current_len = 1
    current_start = days[0]

    for i in range(1, len(days)):
        if days[i] == days[i - 1] + 1:
            # Continue the run
            current_len += 1
        else:
            # Run broken, check if it's the longest
            if current_len > max_len:
                max_len = current_len
                best_start = current_start
                best_end = days[i - 1]

            # Start new run
            current_len = 1
            current_start = days[i]

    # Check final run
    if current_len > max_len:
        max_len = current_len
        best_start = current_start
        best_end = days[-1]

    return max_len, best_start, best_end


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save DataFrame to CSV with UTF-8-sig encoding for Excel compatibility.

    Args:
        df: DataFrame to save
        output_path: Path to output CSV file
    """
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Saved: {output_path}")


def print_summary_stats(label: str, total: int, count: int) -> None:
    """
    Print summary statistics with percentage.

    Args:
        label: Label for the statistic
        total: Total count
        count: Specific count
    """
    pct = (count / total * 100) if total > 0 else 0
    print(f"  {label}: {count:,} / {total:,} ({pct:.2f}%)")
