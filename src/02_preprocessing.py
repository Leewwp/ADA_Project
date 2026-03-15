"""
Data Preprocessing Script
=========================
Handles missing values, outliers, and basic physical validation for the
raw AQI dataset. Each run also writes a preprocessing audit report.

Default behavior is conservative: outliers are flagged for review but not
automatically clipped, because pollution spikes may reflect real events.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_RAW_DATA_FILE = os.path.join(DATA_DIR, "raw_aqi_data.csv")
DEFAULT_PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_aqi_data.csv")
DEFAULT_REPORT_FILE = os.path.join(DATA_DIR, "preprocessing_report.json")

NON_NEGATIVE_COLUMNS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "wind_speed_10m",
    "surface_pressure",
    "precipitation",
]

BOUNDED_COLUMNS = {
    "relative_humidity_2m": (0, 100),
    "cloud_cover": (0, 100),
    "wind_direction_10m": (0, 360),
}

OUTLIER_REVIEW_COLUMNS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "surface_pressure",
]


def json_default(value):
    """Convert numpy and timestamp objects into JSON-safe values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(payload: dict, path: str) -> None:
    """Write a JSON artifact using UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=json_default)


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact dataset summary."""
    missing_by_column = {column: int(count) for column, count in df.isna().sum().items() if int(count) > 0}
    numeric_df = df.select_dtypes(include=[np.number])

    return {
        "records": int(len(df)),
        "column_count": int(len(df.columns)),
        "date_start": df.index.min(),
        "date_end": df.index.max(),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "negative_numeric_values": int((numeric_df < 0).sum().sum()) if not numeric_df.empty else 0,
    }


def summarize_numeric_ranges(df: pd.DataFrame, columns: List[str]) -> dict:
    """Summarize percentile ranges for selected columns."""
    summary = {}
    for column in columns:
        if column not in df.columns:
            continue

        series = df[column].dropna()
        if series.empty:
            continue

        summary[column] = {
            "min": round(float(series.min()), 4),
            "p01": round(float(series.quantile(0.01)), 4),
            "p25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "p75": round(float(series.quantile(0.75)), 4),
            "p99": round(float(series.quantile(0.99)), 4),
            "max": round(float(series.max()), 4),
        }

    return summary


def load_data(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """Load raw AQI data from CSV and normalize timestamp order."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index.name = "datetime"

    duplicate_count = int(df.index.duplicated().sum())
    if duplicate_count:
        df = df[~df.index.duplicated(keep="first")]

    df = df.sort_index()

    load_report = {
        "input_file": filepath,
        "loaded_records": int(len(df)),
        "duplicates_removed": duplicate_count,
    }

    logger.info("Loaded %s records from %s", len(df), filepath)
    return df, load_report


def check_missing_values(df: pd.DataFrame) -> Dict[str, dict]:
    """Check and report missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_summary = {
        column: {
            "missing_count": int(missing[column]),
            "missing_pct": float(missing_pct[column]),
        }
        for column in df.columns
        if int(missing[column]) > 0
    }

    if missing_summary:
        logger.info("Missing values detected: %s", missing_summary)
    else:
        logger.info("No missing values detected in input data.")

    return missing_summary


def handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Handle missing values using time-aware interpolation.

    Linear interpolation is applied first, followed by forward/backward fill for
    leading or trailing gaps.
    """
    df_clean = df.copy()
    before_missing = int(df_clean.isnull().sum().sum())

    df_clean = df_clean.interpolate(method="time")
    df_clean = df_clean.ffill().bfill()

    after_missing = int(df_clean.isnull().sum().sum())
    summary = {
        "method": "time_interpolation_then_ffill_bfill",
        "missing_before": before_missing,
        "missing_after": after_missing,
    }

    return df_clean, summary


def analyze_outliers(df: pd.DataFrame, columns: List[str]) -> dict:
    """Analyze outliers in numeric columns using the IQR rule."""
    outlier_summary = {}

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = int(outliers.sum())

        outlier_summary[col] = {
            "count": outlier_count,
            "percentage": round(outlier_count / len(df) * 100, 2),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "min": round(float(df[col].min()), 4),
            "p01": round(float(df[col].quantile(0.01)), 4),
            "p99": round(float(df[col].quantile(0.99)), 4),
            "max": round(float(df[col].max()), 4),
        }

    return outlier_summary


def handle_outliers(df: pd.DataFrame, method: str = "flag") -> Tuple[pd.DataFrame, dict]:
    """Handle outliers in numeric columns using the IQR rule."""
    df_clean = df.copy()
    outlier_summary = analyze_outliers(df_clean, OUTLIER_REVIEW_COLUMNS)

    if method == "flag":
        return df_clean, {
            "mode": "flag_only",
            "reason": "Pollution spikes may represent real air-quality episodes, so they are preserved by default.",
            "columns_reviewed": list(outlier_summary.keys()),
            "details": outlier_summary,
        }

    if method not in {"clip", "remove"}:
        raise ValueError("method must be one of: flag, clip, remove")

    for col in [column for column in OUTLIER_REVIEW_COLUMNS if column in df_clean.columns]:
        if col not in outlier_summary:
            continue

        lower_bound = outlier_summary[col]["lower_bound"]
        upper_bound = outlier_summary[col]["upper_bound"]
        outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)

        if method == "clip":
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == "remove":
            df_clean = df_clean.loc[~outliers]

    return df_clean, {
        "mode": method,
        "reason": "IQR-based adjustment applied explicitly by user request.",
        "columns_reviewed": list(outlier_summary.keys()),
        "details": outlier_summary,
    }


def handle_infinity_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Replace infinity values and backfill any resulting gaps."""
    df_clean = df.copy()
    inf_count = int(np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum())

    if not inf_count:
        return df_clean, {
            "infinity_values_found": 0,
            "infinity_values_remaining": 0,
        }

    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.interpolate(method="time").ffill().bfill()

    for col in df_clean.columns:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    remaining_inf = int(np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum())
    return df_clean, {
        "infinity_values_found": inf_count,
        "infinity_values_remaining": remaining_inf,
    }


def enforce_physical_bounds(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clip clearly impossible values to domain-safe bounds."""
    df_clean = df.copy()
    adjustments = {}

    for column in NON_NEGATIVE_COLUMNS:
        if column not in df_clean.columns:
            continue
        mask = df_clean[column] < 0
        if mask.any():
            adjustments[column] = adjustments.get(column, {})
            adjustments[column]["set_to_zero"] = int(mask.sum())
            df_clean.loc[mask, column] = 0

    for column, (lower, upper) in BOUNDED_COLUMNS.items():
        if column not in df_clean.columns:
            continue

        below_mask = df_clean[column] < lower
        above_mask = df_clean[column] > upper

        if below_mask.any() or above_mask.any():
            adjustments[column] = adjustments.get(column, {})
            if below_mask.any():
                adjustments[column]["clipped_below_lower_bound"] = int(below_mask.sum())
            if above_mask.any():
                adjustments[column]["clipped_above_upper_bound"] = int(above_mask.sum())

            df_clean[column] = df_clean[column].clip(lower=lower, upper=upper)

    return df_clean, adjustments


def validate_data(df: pd.DataFrame) -> dict:
    """Validate the cleaned dataset."""
    numeric_df = df.select_dtypes(include=[np.number])
    validation = {
        "records": int(len(df)),
        "date_start": df.index.min(),
        "date_end": df.index.max(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "negative_numeric_values": int((numeric_df < 0).sum().sum()) if not numeric_df.empty else 0,
    }

    return validation


def preprocess_data(input_path: str, output_path: str, report_path: str, outlier_mode: str) -> Tuple[pd.DataFrame, dict]:
    """Run the full preprocessing pipeline and save artifacts."""
    logger.info("Starting preprocessing pipeline")

    df, load_report = load_data(input_path)
    input_summary = summarize_dataframe(df)
    missing_summary = check_missing_values(df)

    df, missing_report = handle_missing_values(df)
    pre_bounds_ranges = summarize_numeric_ranges(df, OUTLIER_REVIEW_COLUMNS)
    df, outlier_report = handle_outliers(df, method=outlier_mode)
    df, infinity_report = handle_infinity_values(df)
    df, bounds_report = enforce_physical_bounds(df)
    post_bounds_ranges = summarize_numeric_ranges(df, OUTLIER_REVIEW_COLUMNS)

    validation = validate_data(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    report = {
        "run_timestamp_utc": utc_now_iso(),
        "input_file": input_path,
        "output_file": output_path,
        "load_report": load_report,
        "input_summary": input_summary,
        "missing_summary_before_processing": missing_summary,
        "processing_steps": {
            "missing_values": missing_report,
            "outliers": outlier_report,
            "infinity_values": infinity_report,
            "physical_bounds": bounds_report,
        },
        "distribution_snapshot": {
            "before_processing": pre_bounds_ranges,
            "after_processing": post_bounds_ranges,
        },
        "output_validation": validation,
    }

    write_json(report, report_path)

    logger.info("Processed data saved to: %s", output_path)
    logger.info("Preprocessing report saved to: %s", report_path)

    return df, report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess Hong Kong AQI data.")
    parser.add_argument("--input", default=DEFAULT_RAW_DATA_FILE, help="Path to the raw CSV input file.")
    parser.add_argument("--output", default=DEFAULT_PROCESSED_DATA_FILE, help="Path to the processed CSV output file.")
    parser.add_argument("--report", default=DEFAULT_REPORT_FILE, help="Path to the preprocessing JSON report.")
    parser.add_argument(
        "--outlier-mode",
        choices=["flag", "clip", "remove"],
        default="flag",
        help="How to handle IQR-detected outliers. Default is 'flag' to preserve real pollution spikes.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for preprocessing."""
    args = parse_args()

    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)

    df, report = preprocess_data(args.input, args.output, args.report, args.outlier_mode)

    print("\nValidation Summary:")
    print("-" * 40)
    print(f"Records: {report['output_validation']['records']}")
    print(f"Date range: {report['output_validation']['date_start']} to {report['output_validation']['date_end']}")
    print(f"Missing values: {report['output_validation']['missing_values']}")
    print(f"Duplicate timestamps: {report['output_validation']['duplicate_timestamps']}")
    print(f"Negative numeric values: {report['output_validation']['negative_numeric_values']}")
    print(f"Outlier mode: {report['processing_steps']['outliers']['mode']}")
    print("\nPreprocessing complete!")
    print(df.head())


if __name__ == "__main__":
    main()
