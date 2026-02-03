"""
Step 8: Wear Adherence Visualization

This module creates a histogram of daily wear adherence (compliance)
with reference lines for median and threshold values.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .config import VisualizationConfig, REQUIRED_COLS
from .utils import (
    ensure_dir,
    load_csv_safe,
    validate_required_columns,
    normalize_time_str,
    parse_day_index,
    time_to_minute_of_day,
    build_subject_day_grid,
    save_dataframe
)


def set_academic_mpl_style() -> None:
    """Simple paper-friendly matplotlib styling + vector font embedding."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,

        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        "savefig.dpi": 300,
        "figure.dpi": 120,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_histogram(
    data: np.ndarray,
    bins: int,
    xlabel: str,
    out_base: Path,
    line_specs: Optional[List[Tuple[float, str, str]]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    alpha: float = 0.35
) -> None:
    """
    Histogram with frequency(count) on y-axis.
    line_specs: list of (x, linestyle, label)
    Saves: .pdf, .svg, .png

    Args:
        data: Data to plot
        bins: Number of histogram bins
        xlabel: X-axis label
        out_base: Base path for output files (without extension)
        line_specs: List of (x, linestyle, label) for vertical reference lines
        xlim: Optional (xmin, xmax) for x-axis limits
        alpha: Transparency for histogram bars
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Skipping visualization.")
        return

    if data.size == 0:
        print(f"[warn] no data for plot")
        return

    set_academic_mpl_style()

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.hist(
        data,
        bins=bins,
        density=False,          # frequency
        alpha=alpha,            # lighter bars
        edgecolor="black",
        linewidth=0.6,
    )

    # Add reference lines
    if line_specs:
        for x, ls, lab in line_specs:
            ax.axvline(x, linestyle=ls, linewidth=2.2, label=lab)

    # ax.set_title(title)  # title is optional; keep commented if you prefer clean figure
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000])
    ax.set_ylim(0, 10000)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(str(out_base) + ".png", dpi=300)
    plt.close(fig)

    print(f"[ok] saved: {out_base}.png")


def run_visualization(cfg: VisualizationConfig) -> pd.DataFrame:
    """
    Execute visualization: Create wear adherence histogram.

    This step:
    1. Loads outlier-cleaned heart rate data from Step 3
    2. Calculates daily wear compliance (worn_minutes / 1440)
    3. Creates frequency histogram with median and threshold lines
    4. Saves plots in PDF, SVG, and PNG formats
    5. Exports daily compliance data

    Args:
        cfg: VisualizationConfig with input/output paths and parameters

    Returns:
        DataFrame with daily compliance values
    """
    print("\n" + "=" * 70)
    print("VISUALIZATION: Wear Adherence Histogram")
    print("=" * 70)

    ensure_dir(cfg.outdir)

    if not MATPLOTLIB_AVAILABLE:
        print("\nWarning: matplotlib is not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
        print("Only daily compliance CSV will be generated.\n")

    # Load outlier-cleaned data
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

    # Calculate worn minutes per day (HR > 0, unique minutes)
    print("\nCalculating daily compliance...")
    worn = df_valid[df_valid["hr_num"] > 0].copy()

    # Count worn minutes per day
    worn_counts = (
        worn.groupby(["subject_id", "day_index"])["minute_of_day"]
        .nunique()
        .reset_index(name="worn_minutes")
    )

    # Build complete subject-day grid (includes ALL days, even those with no data)
    print("\nBuilding complete patient-day grid...")
    subject_maxday = df_valid.groupby("subject_id")["day_index"].max()
    grid = build_subject_day_grid(subject_maxday, day_start=0, day_end=None)

    print(f"  Total patient-days (grid): {len(grid):,}")
    print(f"  Patient-days with worn data: {len(worn_counts):,}")

    # Merge grid with worn counts (missing days → 0 worn_minutes)
    daily_compliance = grid.merge(worn_counts, on=["subject_id", "day_index"], how="left")
    daily_compliance["worn_minutes"] = daily_compliance["worn_minutes"].fillna(0).astype(int)

    # Calculate compliance ratio (missing days → 0% compliance)
    daily_compliance["compliance"] = daily_compliance["worn_minutes"] / cfg.expected_minutes_per_day

    # Add date column for reference
    daily_compliance["date"] = "day" + daily_compliance["day_index"].astype(str)

    # Reorder columns
    daily_compliance = daily_compliance[["subject_id", "day_index", "date", "worn_minutes", "compliance"]]
    daily_compliance = daily_compliance.sort_values(["subject_id", "day_index"]).reset_index(drop=True)

    print(f"\n  Total patient-days (all): {len(daily_compliance):,}")
    print(f"  Mean compliance: {daily_compliance['compliance'].mean():.2%}")
    print(f"  Median compliance: {daily_compliance['compliance'].median():.2%}")
    print(f"  Days with 0% compliance: {(daily_compliance['compliance'] == 0).sum():,}")

    # Create histogram
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating histogram...")

        median_compliance = float(daily_compliance["compliance"].median())
        threshold_compliance = cfg.threshold_ratio

        line_specs = [
            (median_compliance, "--", f"Median = {median_compliance:.3f}"),
            (threshold_compliance, ":", f"Threshold = {threshold_compliance:.2f}"),
        ]

        out_base = cfg.outdir / "hist_daily_compliance"

        save_histogram(
            data=daily_compliance["compliance"].values,
            bins=25,
            xlabel="Mean daily adherence (%)",
            out_base=out_base,
            line_specs=line_specs,
            xlim=(0.0, 1.0),
            alpha=0.35
        )

    # Save compliance data
    print("\nSaving compliance data...")
    save_dataframe(daily_compliance, cfg.outdir / "daily_compliance.csv")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)

    return daily_compliance


def main():
    """Example usage of visualization."""
    from pathlib import Path

    # Configure paths
    base_dir = Path("/app/ai_worker/data_anonymization")
    input_csv = base_dir / "10-code/Processed_wearable_dataset/step3/hr_1min_outlier_cleaned.csv"
    outdir = base_dir / "10-code/Processed_wearable_dataset/visualization"

    # Create configuration
    config = VisualizationConfig(
        input_csv=input_csv,
        outdir=outdir,
        normalize_time=True,
        expected_minutes_per_day=1440,
        threshold_ratio=0.80
    )

    # Run visualization
    compliance_data = run_visualization(config)


if __name__ == "__main__":
    main()
