# Heart Rate Data Processing Pipeline

A comprehensive, modular pipeline for processing and quality-controlling wearable heart rate data for research publication.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This pipeline processes raw 1-minute heart rate data from wearable devices through multiple quality control steps to produce a clean, validated dataset suitable for scientific publication. It implements best practices for wearable data processing including duplicate removal, outlier detection, wear adherence filtering, and missingness analysis.

## Features

- **Step 1: Collection Day Counting** - Establishes baseline patient-days and recording statistics
- **Step 2: Duplicate Removal** - Identifies and removes duplicate 1-minute records
- **Step 3: Outlier Removal** - Removes physiologically implausible heart rate values
- **Step 4: Valid Day Filtering** - Filters days based on wear adherence (≥80% or ≥1140 minutes)
- **Step 5: Valid Participant Filtering** - Retains participants with ≥7 consecutive valid days
- **Step 6: HR Statistics Extraction** - Calculates daily mean, max, min heart rate
- **Step 7: Missingness Analysis** - Assesses missing data across all wearable metrics
- **Step 8: Visualization** - Generates wear adherence distribution histogram

## Installation

### Requirements

- Python 3.8 or higher
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0 (optional, for visualization)

### Setup

1. Clone or download this repository:
```bash
cd /path/to/your/project
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib
```

Or using conda:
```bash
conda install pandas numpy matplotlib
```

3. Verify installation:
```bash
python -c "import pandas; import numpy; print('Installation successful!')"
```

## Quick Start

### Option 1: Run Complete Pipeline

Run all steps sequentially with a single command:

```bash
python -m src.run_pipeline \
  --input-hr /path/to/heartrate_minute.csv \
  --input-activity /path/to/activity_daily.csv \
  --input-resting-hr /path/to/heartrate_resting_daily.csv \
  --input-daily-hr /path/to/heartrate_summary_daily.csv \
  --input-sleep /path/to/sleep_summary_daily.csv \
  --output /path/to/output
```

### Option 2: Run Individual Steps

Execute steps one at a time:

```python
from pathlib import Path
from src.step1_count_days import run_step1
from src.config import Step1Config

config = Step1Config(
    input_csv=Path("data/heartrate_minute.csv"),
    outdir=Path("output/step1")
)

patient_day, patient_counts, global_counts = run_step1(config)
```

### Option 3: Use Provided Script

Modify and run the provided example script:

```bash
python scripts/run_all_steps.py
```

## Input Data Format

### Required Input: 1-Minute Heart Rate Data

**File:** `heartrate_minute.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| subject_id | string | Unique participant identifier | "S001" |
| date | string | Day index in format "dayN" | "day0", "day1" |
| time | string | Time in HH:MM:SS format | "09:30:00" |
| hr | numeric | Heart rate in beats per minute | "72" |

**Example:**
```csv
subject_id,date,time,hr
S001,day0,00:00:00,72
S001,day0,00:01:00,74
S001,day0,00:02:00,71
```

### Optional Inputs: Daily Metric Sources

**Activity Daily** (`activity_daily.csv`):
- Columns: subject_id, date, steps, distance, calories

**Resting HR Daily** (`heartrate_resting_daily.csv`):
- Columns: subject_id, date, resting_hr

**HR Summary Daily** (`heartrate_summary_daily.csv`):
- Columns: subject_id, date, daily_mean, daily_max, daily_min

**Sleep Summary Daily** (`sleep_summary_daily.csv`):
- Columns: subject_id, date, total_minutes_asleep, total_time_in_bed, stages_*, cnt_*

## Pipeline Steps

### Step 1: Count Total Collection Days

**Purpose:** Establish baseline patient-days and count recorded minutes.

**Definition of Recorded Minute:**
- Valid datetime (valid "dayN" format AND valid "HH:MM:SS" time)
- Numeric HR value (not NaN)
- HR value ≠ 0

**Outputs:**
- `patient_day_counts.csv` - Recording stats per patient-day
- `patient_counts.csv` - Summary per patient
- `global_counts.csv` - Overall cohort statistics

**Usage:**
```python
from src.step1_count_days import run_step1, Step1Config

config = Step1Config(
    input_csv=Path("data/heartrate_minute.csv"),
    outdir=Path("output/step1"),
    day_start=0,
    day_end=None  # Use None for max observed day
)

patient_day, patient_counts, global_counts = run_step1(config)
```

### Step 2: Remove Duplicates

**Purpose:** Identify and remove duplicate 1-minute heart rate records.

**Duplicate Definition:**
- Records with identical (subject_id, date, time)

**Outputs:**
- `hr_1min_deduped.csv` - Deduplicated dataset
- `duplicates_all.csv` - All duplicate records
- `duplicates_extra.csv` - Removed duplicates
- `step2_input_summary.csv` - Filtering statistics

**Usage:**
```python
from src.step2_remove_duplicates import run_step2, Step2Config

config = Step2Config(
    input_csv=Path("data/heartrate_minute.csv"),
    outdir=Path("output/step2"),
    dedup_key=["subject_id", "date", "time"],
    keep_policy="last",  # or "first"
    normalize_time=True
)

deduped, duplicates_all, duplicates_extra, summary = run_step2(config)
```

### Step 3: Remove Outliers

**Purpose:** Remove or replace physiologically implausible heart rate values.

**Outlier Criteria:**
- HR < 40 bpm (too low)
- HR > 163 bpm (too high)

**Modes:**
- `replace_zero`: Replace outliers with 0 (default)
- `drop`: Remove outlier rows entirely

**Outputs:**
- `hr_1min_outlier_cleaned.csv` - Cleaned dataset
- `hr_1min_outliers.csv` - Outlier records with reasons
- `outlier_summary.csv` - Outlier statistics

**Usage:**
```python
from src.step3_remove_outliers import run_step3, Step3Config

config = Step3Config(
    input_csv=Path("output/step2/hr_1min_deduped.csv"),
    outdir=Path("output/step3"),
    min_hr=40.0,
    max_hr=163.0,
    outlier_mode="replace_zero",
    normalize_time=True
)

cleaned, outliers, summary = run_step3(config)
```

### Step 4: Filter Valid Days

**Purpose:** Identify days with sufficient wear adherence.

**Valid Day Criteria:**
- Worn minutes ≥ 1140 (80% of 1440 minutes per day)
- Worn minute: HR > 0, unique (subject_id, day_index, minute_of_day)

**Outputs:**
- `daily_adherence.csv` - All days with adherence metrics
- `daily_adherence__valid_days.csv` - Valid days only
- `valid_day_counts_by_subject.csv` - Count per subject
- `valid_day_summary.csv` - Cohort summary

**Usage:**
```python
from src.step4_filter_valid_days import run_step4, Step4Config

config = Step4Config(
    input_csv=Path("output/step3/hr_1min_outlier_cleaned.csv"),
    outdir=Path("output/step4"),
    expected_minutes_per_day=1440,
    threshold_ratio=0.80,
    threshold_minutes=1140,
    force_min_day_to_zero=True,
    normalize_time=True
)

daily, valid_days, valid_counts, summary = run_step4(config)
```

### Step 5: Filter Valid Participants

**Purpose:** Keep only participants with ≥7 consecutive valid days.

**Valid Participant Criteria:**
- Longest consecutive run of valid days ≥ 7 days
- Consecutive = day_index increases by 1 each day (no gaps)

**Outputs:**
- `subject_consecutive_summary.csv` - All subjects with run lengths
- `subjects_pass.csv` - Valid subjects only
- `subjects_fail.csv` - Excluded subjects
- `daily_adherence__pass_subjects.csv` - All days for valid subjects

**Usage:**
```python
from src.step5_filter_valid_participants import run_step5, Step5Config

config = Step5Config(
    input_csv=Path("output/step4/daily_adherence.csv"),
    outdir=Path("output/step5"),
    min_consecutive_days=7,
    day_start=None,
    day_end=None
)

summary, daily_pass, subjects_pass, subjects_fail = run_step5(config)
```

### Step 6: Extract HR Statistics

**Purpose:** Calculate daily heart rate summary statistics for valid days.

**Statistics:**
- `daily_mean` - Mean HR
- `daily_max` - Maximum HR
- `daily_min` - Minimum HR

**Outputs:**
- `daily_hr_stats__valid_days.csv`

**Usage:**
```python
from src.step6_extract_hr_stats import run_step6, Step6Config

config = Step6Config(
    key_csv=Path("output/step4/daily_adherence__valid_days.csv"),
    hr_1min_csv=Path("output/step3/hr_1min_outlier_cleaned.csv"),
    outdir=Path("output/step6"),
    out_csv_name="daily_hr_stats__valid_days.csv",
    filter_using_is_valid_day=True
)

daily_stats = run_step6(config)
```

### Step 7: Missingness Analysis

**Purpose:** Merge all wearable metrics and assess missing data.

**Missingness Definition:**
- NaN = missing
- Numeric 0 ≠ missing (valid measurement)

**Outputs:**
- `merged_metrics.csv` - All metrics merged on valid subject-days
- `missingness_report.csv` - Summary table
- `missing_by_metric/missing_rows__<metric>.csv` - Per-metric missing rows

**Usage:**
```python
from src.step7_missingness_analysis import run_step7, Step7Config

config = Step7Config(
    key_csv=Path("output/step5/daily_adherence__pass_subjects.csv"),
    activity_csv=Path("data/activity_daily.csv"),
    resting_csv=Path("data/heartrate_resting_daily.csv"),
    daily_hr_csv=Path("output/step6/daily_hr_stats__valid_days.csv"),
    sleep_csv=Path("data/sleep_summary_daily.csv"),
    outdir=Path("output/step7"),
    metrics=['steps', 'resting_hr', 'daily_mean', 'total_minutes_asleep'],
    filter_valid_days=True,
    collapse_duplicates_by_mean=True
)

merged, report = run_step7(config)
```

### Step 8: Visualization

**Purpose:** Generate wear adherence distribution histogram.

**Outputs:**
- `hist_daily_compliance.pdf/svg/png` - Histogram plots
- `daily_compliance.csv` - Daily compliance data

**Usage:**
```python
from src.step8_visualize_adherence import run_visualization, VisualizationConfig

config = VisualizationConfig(
    input_csv=Path("output/step3/hr_1min_outlier_cleaned.csv"),
    outdir=Path("output/visualization"),
    normalize_time=True,
    expected_minutes_per_day=1440,
    threshold_ratio=0.80
)

compliance_data = run_visualization(config)
```

## Configuration

### Default Parameters

```python
# Heart rate thresholds
MIN_HR = 40.0  # bpm
MAX_HR = 163.0  # bpm

# Adherence thresholds
ADHERENCE_THRESHOLD_RATIO = 0.80  # 80%
ADHERENCE_THRESHOLD_MINUTES = 1140  # 0.80 * 1440

# Participant filtering
MIN_CONSECUTIVE_DAYS = 7

# Duplicate handling
DEDUP_KEY = ["subject_id", "date", "time"]
KEEP_POLICY = "last"  # Keep last duplicate

# Outlier handling
OUTLIER_MODE = "replace_zero"  # Replace outliers with 0
```

### Customizing Parameters

Parameters can be customized via:

1. **Command-line arguments:**
```bash
python -m src.run_pipeline \
  --min-hr 35 \
  --max-hr 180 \
  --adherence-threshold 0.75 \
  --min-consecutive-days 5
```

2. **Config objects:**
```python
config = Step3Config(
    min_hr=35.0,
    max_hr=180.0,
    outlier_mode="drop"
)
```

3. **Modifying `config.py`:**
Edit default values in `src/config.py`.

## Output Structure

```
output/
├── step1/
│   ├── patient_day_counts.csv
│   ├── patient_counts.csv
│   └── global_counts.csv
├── step2/
│   ├── hr_1min_deduped.csv
│   ├── duplicates_all.csv
│   ├── duplicates_extra.csv
│   └── step2_input_summary.csv
├── step3/
│   ├── hr_1min_outlier_cleaned.csv
│   ├── hr_1min_outliers.csv
│   └── outlier_summary.csv
├── step4/
│   ├── daily_adherence.csv
│   ├── daily_adherence__valid_days.csv
│   ├── valid_day_counts_by_subject.csv
│   └── valid_day_summary.csv
├── step5/
│   ├── subject_consecutive_summary.csv
│   ├── subjects_pass.csv
│   ├── subjects_fail.csv
│   └── daily_adherence__pass_subjects.csv
├── step6/
│   └── daily_hr_stats__valid_days.csv
├── step7/
│   ├── merged_metrics.csv
│   ├── missingness_report.csv
│   └── missing_by_metric/
│       ├── missing_rows__steps.csv
│       └── ...
└── visualization/
    ├── hist_daily_compliance.pdf
    ├── hist_daily_compliance.svg
    ├── hist_daily_compliance.png
    └── daily_compliance.csv
```

## Data Quality Checks

The pipeline implements comprehensive quality checks:

1. **Column Validation** - Verifies required columns exist
2. **Type Safety** - Converts data types with error handling
3. **Format Validation** - Validates date ("dayN") and time ("HH:MM:SS") formats
4. **Duplicate Detection** - Explicit duplicate tracking with configurable policy
5. **Range Validation** - Heart rate outlier detection (40-163 bpm)
6. **Completeness Tracking** - Recording ratios, adherence ratios, missing counts
7. **Reproducibility** - UTF-8-sig encoding for Excel compatibility

## Best Practices

### For Research Publication

1. **Document Parameters** - Record all threshold values used
2. **Report Exclusions** - Include counts of duplicates, outliers, excluded days/participants
3. **Visualize Quality** - Use Step 8 histogram to show adherence distribution
4. **Archive Raw Data** - Keep original input files unchanged
5. **Version Control** - Track pipeline version and any customizations

### Performance Tips

1. **Large Datasets** - Process in chunks if memory issues occur
2. **Parallel Processing** - Steps 1-3 can be run independently if needed
3. **Skip Optional Steps** - Use `--skip-visualization` or `--skip-missingness` flags
4. **Intermediate Outputs** - Each step saves outputs for checkpoint recovery

## Troubleshooting

### Common Issues

**Issue: "FileNotFoundError: Input file not found"**
- Solution: Check that input file paths are correct and files exist

**Issue: "ValueError: Missing required columns"**
- Solution: Verify input CSV has required columns (subject_id, date, time, hr)

**Issue: "Memory Error"**
- Solution: Process data in smaller batches or increase available RAM

**Issue: "matplotlib not available" during visualization**
- Solution: Install matplotlib: `pip install matplotlib`

**Issue: Incorrect date format**
- Solution: Ensure date column uses "day0", "day1", etc. format

**Issue: Incorrect time format**
- Solution: Ensure time column uses "HH:MM:SS" format (e.g., "09:30:00")

## Citation

If you use this pipeline in your research, please cite:

```
[Your Citation Here]
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [mass3758@gmail.com, gunwoof1234@gmail.com]

## Acknowledgments

This pipeline was developed for processing wearable heart rate data for scientific publication in the Scientific Data journal.

## Version History

### Version 1.0.0 (2024)
- Initial release
- Complete pipeline implementation
- All 8 processing steps
- Comprehensive documentation


