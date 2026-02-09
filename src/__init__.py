"""
Heart Rate Data Processing Pipeline

A modular pipeline for processing and quality-controlling wearable heart rate data.

Modules:
    - config: Configuration classes and constants
    - utils: Shared utility functions
    - step1_count_days: Count total collection days
    - step2_remove_duplicates: Remove duplicate records
    - step3_remove_outliers: Remove outlier heart rate values
    - step4_filter_valid_days: Filter days by wear adherence
    - step5_filter_valid_participants: Filter participants by consecutive valid days
    - step6_extract_hr_stats: Extract daily HR statistics
    - step7_missingness_analysis: Analyze missing data across metrics
    - step8_visualize_adherence: Visualize wear adherence distribution
"""

__version__ = "1.0.0"
__author__ = "Heart Rate Data Pipeline Team"

from .config import (
    Step1Config,
    Step2Config,
    Step3Config,
    Step4Config,
    Step5Config,
    Step6Config,
    Step7Config,
    VisualizationConfig,
    PipelineConfig
)

from .step1_count_days import run_step1
from .step2_remove_duplicates import run_step2
from .step3_remove_outliers import run_step3
from .step4_filter_valid_days import run_step4
from .step5_filter_valid_participants import run_step5
from .step6_extract_hr_stats import run_step6
from .step7_missingness_analysis import run_step7
from .step8_visualize_adherence import run_visualization


__all__ = [
    # Config classes
    "Step1Config",
    "Step2Config",
    "Step3Config",
    "Step4Config",
    "Step5Config",
    "Step6Config",
    "Step7Config",
    "VisualizationConfig",
    "PipelineConfig",
    # Step functions
    "run_step1",
    "run_step2",
    "run_step3",
    "run_step4",
    "run_step5",
    "run_step6",
    "run_step7",
    "run_visualization",

]
