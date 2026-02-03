"""
Configuration classes and constants for the heart rate data pipeline.

This module contains all configuration dataclasses and constants used across
the data processing pipeline for wearable heart rate data.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


# ============================================================================
# Constants
# ============================================================================

# Column requirements
REQUIRED_COLS = ["subject_id", "date", "time", "hr"]

# Time and day constants
EXPECTED_MINUTES_PER_DAY = 1440  # 24 hours * 60 minutes

# Heart rate thresholds
MIN_HR = 40.0  # Minimum valid HR (bpm)
MAX_HR = 163.0  # Maximum valid HR (bpm)

# Adherence thresholds
ADHERENCE_THRESHOLD_RATIO = 0.80  # 80% wear adherence
ADHERENCE_THRESHOLD_MINUTES = 1152  # 0.80 * 1440 = 1152 (exact 80%)

# Participant filtering
MIN_CONSECUTIVE_DAYS = 7  # Minimum consecutive valid days for valid participant

# Duplicate handling
DEDUP_KEY = ["subject_id", "date", "time"]
KEEP_POLICY = "last"  # Keep last duplicate

# Outlier handling
OUTLIER_MODE = "replace_zero"  # Replace outliers with 0 (alternative: "drop")


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass(frozen=True)
class Step1Config:
    """Configuration for Step 1: Count total collection days."""

    input_csv: Path
    outdir: Path
    day_start: int = 0
    day_end: Optional[int] = None


@dataclass(frozen=True)
class Step2Config:
    """Configuration for Step 2: Duplicate removal."""

    input_csv: Path
    outdir: Path
    dedup_key: List[str]
    keep_policy: Literal["first", "last"] = "last"
    normalize_time: bool = True


@dataclass(frozen=True)
class Step3Config:
    """Configuration for Step 3: Outlier removal."""

    input_csv: Path
    outdir: Path
    min_hr: float = 40.0
    max_hr: float = 163.0
    outlier_mode: Literal["drop", "replace_zero"] = "replace_zero"
    normalize_time: bool = True


@dataclass(frozen=True)
class Step4Config:
    """Configuration for Step 4: Valid day filtering."""

    input_csv: Path
    outdir: Path
    expected_minutes_per_day: int = 1440
    threshold_ratio: float = 0.80
    threshold_minutes: Optional[int] = None
    force_min_day_to_zero: bool = True
    normalize_time: bool = True


@dataclass(frozen=True)
class Step5Config:
    """Configuration for Step 5: Valid participant filtering."""

    input_csv: Path
    outdir: Path
    min_consecutive_days: int = 7
    day_start: Optional[int] = None
    day_end: Optional[int] = None


@dataclass(frozen=True)
class Step6Config:
    """Configuration for Step 6: HR statistics extraction."""

    key_csv: Path
    hr_1min_csv: Path
    outdir: Path
    out_csv_name: str = "daily_hr_stats__valid_days.csv"
    filter_using_is_valid_day: bool = True


@dataclass(frozen=True)
class Step7Config:
    """Configuration for Step 7: Missingness analysis."""

    key_csv: Path
    activity_csv: Path
    resting_csv: Path
    daily_hr_csv: Path
    sleep_csv: Path
    outdir: Path
    metrics: List[str]
    filter_valid_days: bool = True
    collapse_duplicates_by_mean: bool = True


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for wear adherence visualization."""

    input_csv: Path
    outdir: Path
    normalize_time: bool = True
    expected_minutes_per_day: int = 1440
    threshold_ratio: float = 0.80


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the complete pipeline."""

    # Input data paths
    input_hr_minute_csv: Path
    input_activity_csv: Path
    input_resting_hr_csv: Path
    input_daily_hr_csv: Path
    input_sleep_csv: Path

    # Output directory
    output_base_dir: Path

    # Pipeline parameters
    day_start: int = 0
    day_end: Optional[int] = None
    min_hr: float = MIN_HR
    max_hr: float = MAX_HR
    adherence_threshold_ratio: float = ADHERENCE_THRESHOLD_RATIO
    adherence_threshold_minutes: Optional[int] = ADHERENCE_THRESHOLD_MINUTES
    min_consecutive_days: int = MIN_CONSECUTIVE_DAYS

    # Missingness metrics to analyze
    missingness_metrics: List[str] = None

    def __post_init__(self):
        """Initialize default missingness metrics if not provided."""
        if self.missingness_metrics is None:
            # Default metrics to analyze
            object.__setattr__(self, 'missingness_metrics', [
                'steps', 'distance', 'calories', 'resting_hr',
                'daily_mean', 'daily_max', 'daily_min',
                'total_minutes_asleep', 'total_time_in_bed',
                'stages_deep', 'stages_light', 'stages_rem', 'stages_wake',
                'stages_asleep', 'stages_restless', 'stages_awake',
                'cnt_deep', 'cnt_light', 'cnt_rem', 'cnt_wake',
                'cnt_asleep', 'cnt_restless', 'cnt_awake'
            ])
