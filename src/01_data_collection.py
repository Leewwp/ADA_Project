"""
Data Collection Script
======================
Collects hourly air-quality and weather data for Hong Kong.

Key improvements:
1. Live API collection is the default and fails loudly on error.
2. Sample data can still be generated, but only when explicitly requested.
3. Every run writes collection metadata for provenance and auditability.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_RAW_DATA_FILE = "raw_aqi_data.csv"
DEFAULT_METADATA_FILE = "collection_metadata.json"

LATITUDE = 22.2855
LONGITUDE = 114.1577
LOCATION_NAME = "Hong Kong"
TARGET_TIMEZONE = "Asia/Hong_Kong"

DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_CHUNK_MONTHS = 6

AQI_API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

AQI_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
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


def build_date_chunks(start_date: str, end_date: str, months_per_chunk: int) -> List[Tuple[str, str]]:
    """Split a date range into month-sized collection windows."""
    if months_per_chunk <= 0:
        raise ValueError("months_per_chunk must be a positive integer")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if start_ts > end_ts:
        raise ValueError("start_date must be earlier than or equal to end_date")

    chunks: List[Tuple[str, str]] = []
    current = start_ts

    while current <= end_ts:
        next_start = current + pd.DateOffset(months=months_per_chunk)
        chunk_end = min(next_start - pd.Timedelta(days=1), end_ts)
        chunks.append((current.date().isoformat(), chunk_end.date().isoformat()))
        current = chunk_end + pd.Timedelta(days=1)

    return chunks


def fetch_with_retry(url: str, params: dict, max_retries: int = 3, pause_seconds: int = 1) -> Tuple[Optional[dict], dict]:
    """Fetch data from an API with retry tracking."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json(), {
                "success": True,
                "attempts": attempt,
                "status_code": response.status_code,
                "error": None,
            }
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            logger.warning("Attempt %s/%s failed for %s: %s", attempt, max_retries, url, exc)
            if attempt < max_retries:
                time.sleep(pause_seconds * attempt)

    return None, {
        "success": False,
        "attempts": max_retries,
        "status_code": None,
        "error": last_error,
    }


def fetch_aqi_data(start_date: str, end_date: str) -> Tuple[Optional[dict], dict]:
    """Fetch AQI-related pollutant data from Open-Meteo."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(AQI_VARIABLES),
        "timezone": TARGET_TIMEZONE,
    }
    logger.info("Fetching AQI data from %s to %s", start_date, end_date)
    return fetch_with_retry(AQI_API_URL, params)


def fetch_weather_data(start_date: str, end_date: str) -> Tuple[Optional[dict], dict]:
    """Fetch weather data from the Open-Meteo archive API."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": TARGET_TIMEZONE,
    }
    logger.info("Fetching weather data from %s to %s", start_date, end_date)
    return fetch_with_retry(WEATHER_API_URL, params)


def parse_hourly_response(data: Optional[dict], variables: List[str], source_name: str) -> pd.DataFrame:
    """Parse an hourly API response into a timestamp-indexed DataFrame."""
    if not data or "hourly" not in data or "time" not in data["hourly"]:
        logger.warning("No usable hourly data found in %s response", source_name)
        return pd.DataFrame(columns=variables)

    hourly = data["hourly"]
    df = pd.DataFrame(index=pd.to_datetime(hourly["time"]))
    df.index.name = "datetime"

    for variable in variables:
        df[variable] = hourly.get(variable, [np.nan] * len(df.index))

    logger.info("Parsed %s %s records", len(df), source_name)
    return df


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Ensure a DataFrame contains the requested columns."""
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


def merge_aqi_weather(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather data onto AQI timestamps.

    A left join is used because AQI timestamps define the modeling timeline.
    """
    aqi_df = ensure_columns(aqi_df, AQI_VARIABLES)

    if weather_df.empty:
        weather_df = pd.DataFrame(index=aqi_df.index)

    weather_df = ensure_columns(weather_df, WEATHER_VARIABLES)
    merged = aqi_df.join(weather_df[WEATHER_VARIABLES], how="left")

    missing_weather = int(merged[WEATHER_VARIABLES].isna().sum().sum())
    if missing_weather:
        logger.warning("Weather data still has %s missing values after merge", missing_weather)

    logger.info("Merged data shape: %s", merged.shape)
    return merged


def fill_missing_with_generated_values(df: pd.DataFrame, column: str, values: np.ndarray, lower: float = None, upper: float = None) -> int:
    """Fill only missing entries for a column using generated proxy values."""
    if column not in df.columns:
        df[column] = np.nan

    mask = df[column].isna()
    if not mask.any():
        return 0

    fill_values = values.copy()
    if lower is not None:
        fill_values = np.maximum(fill_values, lower)
    if upper is not None:
        fill_values = np.minimum(fill_values, upper)

    df.loc[mask, column] = fill_values[mask]
    return int(mask.sum())


def generate_realistic_weather(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Fill missing weather values with climate-based proxy data.

    This is only used to fill weather gaps. AQI values are never synthesized here.
    """
    df = ensure_columns(df.copy(), WEATHER_VARIABLES)
    np.random.seed(42)

    month_factor = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    hour_factor = np.sin(2 * np.pi * df.index.hour / 24)

    synthetic_summary = {}

    temperature = 22 + 10 * month_factor + 3 * hour_factor + np.random.randn(len(df)) * 2
    synthetic_summary["temperature_2m"] = fill_missing_with_generated_values(df, "temperature_2m", temperature)

    wind_speed = 15 + 8 * hour_factor + np.random.randn(len(df)) * 8
    synthetic_summary["wind_speed_10m"] = fill_missing_with_generated_values(df, "wind_speed_10m", wind_speed, lower=0)

    wind_direction = (45 + 90 * month_factor + np.random.randn(len(df)) * 30) % 360
    synthetic_summary["wind_direction_10m"] = fill_missing_with_generated_values(
        df, "wind_direction_10m", wind_direction, lower=0, upper=360
    )

    humidity = 78 + 12 * hour_factor + np.random.randn(len(df)) * 10
    synthetic_summary["relative_humidity_2m"] = fill_missing_with_generated_values(
        df, "relative_humidity_2m", humidity, lower=0, upper=100
    )

    pressure = 1013 + 5 * month_factor + np.random.randn(len(df)) * 5
    synthetic_summary["surface_pressure"] = fill_missing_with_generated_values(df, "surface_pressure", pressure)

    rain_prob = 0.1 + 0.2 * np.maximum(0, month_factor)
    precipitation = np.random.exponential(2, len(df)) * (np.random.rand(len(df)) < rain_prob)
    synthetic_summary["precipitation"] = fill_missing_with_generated_values(
        df, "precipitation", precipitation, lower=0
    )

    cloud_cover = 50 + 30 * month_factor + 20 * hour_factor + np.random.randn(len(df)) * 15
    synthetic_summary["cloud_cover"] = fill_missing_with_generated_values(
        df, "cloud_cover", cloud_cover, lower=0, upper=100
    )

    filled_total = sum(synthetic_summary.values())
    if filled_total:
        logger.warning(
            "Filled %s missing weather values with synthetic proxy data. Review collection_metadata.json before reporting results.",
            filled_total,
        )

    return df, {
        "used": filled_total > 0,
        "filled_cells": filled_total,
        "filled_by_column": synthetic_summary,
    }


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact quality summary for a dataset."""
    missing_by_column = {column: int(count) for column, count in df.isna().sum().items() if int(count) > 0}

    return {
        "records": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "date_start": df.index.min(),
        "date_end": df.index.max(),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_column,
    }


def save_collection_artifacts(df: pd.DataFrame, metadata: dict, output_dir: str, raw_filename: str, metadata_filename: str) -> None:
    """Write raw data and metadata files to disk."""
    os.makedirs(output_dir, exist_ok=True)

    raw_data_path = os.path.join(output_dir, raw_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    df.to_csv(raw_data_path)
    write_json(metadata, metadata_path)

    logger.info("Data saved to: %s", raw_data_path)
    logger.info("Collection metadata saved to: %s", metadata_path)


def collect_all_data(
    start_date: str,
    end_date: str,
    months_per_chunk: int,
    output_dir: str,
    raw_filename: str,
    metadata_filename: str,
) -> Tuple[pd.DataFrame, dict]:
    """Collect real AQI and weather data from APIs."""
    chunks = build_date_chunks(start_date, end_date, months_per_chunk)
    all_aqi = []
    all_weather = []
    chunk_reports = []

    for chunk_start, chunk_end in chunks:
        aqi_json, aqi_fetch = fetch_aqi_data(chunk_start, chunk_end)
        aqi_df = parse_hourly_response(aqi_json, AQI_VARIABLES, "AQI")
        if not aqi_df.empty:
            all_aqi.append(aqi_df)

        time.sleep(1)

        weather_json, weather_fetch = fetch_weather_data(chunk_start, chunk_end)
        weather_df = parse_hourly_response(weather_json, WEATHER_VARIABLES, "weather")
        if not weather_df.empty:
            all_weather.append(weather_df)

        time.sleep(1)

        chunk_reports.append({
            "start_date": chunk_start,
            "end_date": chunk_end,
            "aqi_fetch": aqi_fetch,
            "weather_fetch": weather_fetch,
            "aqi_records": int(len(aqi_df)),
            "weather_records": int(len(weather_df)),
        })

    if not all_aqi:
        raise RuntimeError("AQI collection failed for all chunks. No raw dataset was produced.")

    aqi_combined = pd.concat(all_aqi).sort_index()
    aqi_combined = aqi_combined[~aqi_combined.index.duplicated(keep="first")]

    if all_weather:
        weather_combined = pd.concat(all_weather).sort_index()
        weather_combined = weather_combined[~weather_combined.index.duplicated(keep="first")]
    else:
        weather_combined = pd.DataFrame(index=aqi_combined.index)

    combined_df = merge_aqi_weather(aqi_combined, weather_combined)

    missing_weather_before_fill = {
        column: int(count)
        for column, count in combined_df[WEATHER_VARIABLES].isna().sum().items()
    }
    combined_df, synthetic_fill = generate_realistic_weather(combined_df)
    missing_weather_after_fill = {
        column: int(count)
        for column, count in combined_df[WEATHER_VARIABLES].isna().sum().items()
    }

    metadata = {
        "run_timestamp_utc": utc_now_iso(),
        "mode_requested": "live",
        "source_mode": "live_api",
        "fallback_to_sample": False,
        "location": {
            "name": LOCATION_NAME,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
        "timezone": TARGET_TIMEZONE,
        "requested_date_range": {
            "start_date": start_date,
            "end_date": end_date,
            "months_per_chunk": months_per_chunk,
        },
        "source_urls": {
            "air_quality_api": AQI_API_URL,
            "weather_api": WEATHER_API_URL,
        },
        "aqi_variables": AQI_VARIABLES,
        "weather_variables": WEATHER_VARIABLES,
        "chunk_reports": chunk_reports,
        "weather_synthetic_fill": {
            "missing_before_fill": missing_weather_before_fill,
            "missing_after_fill": missing_weather_after_fill,
            **synthetic_fill,
        },
        "raw_data_summary": summarize_dataframe(combined_df),
        "notes": [
            "AQI values are sourced from the air-quality API; only weather gaps may be filled with synthetic proxy values.",
            "Sample data is never used unless explicitly requested from the command line.",
        ],
    }

    save_collection_artifacts(combined_df, metadata, output_dir, raw_filename, metadata_filename)
    return combined_df, metadata


def generate_sample_data(
    start_date: str,
    end_date: str,
    output_dir: str,
    raw_filename: str,
    metadata_filename: str,
) -> Tuple[pd.DataFrame, dict]:
    """Generate synthetic data for offline debugging only."""
    logger.warning("Generating sample data. Do not present this dataset as real collection output.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    dates = pd.date_range(start=start_ts, end=end_ts, freq="h")

    n = len(dates)
    np.random.seed(42)

    month_factor = np.sin(2 * np.pi * (dates.month - 1) / 12)
    hour_factor = np.sin(2 * np.pi * dates.hour / 24)

    pm25 = np.maximum(30 + 15 * hour_factor + 10 * month_factor + np.random.randn(n) * 10, 0)
    pm10 = np.maximum(45 + 20 * hour_factor + 15 * month_factor + np.random.randn(n) * 12, 0)
    no2 = np.maximum(40 + 10 * hour_factor + 5 * month_factor + np.random.randn(n) * 8, 0)
    ozone = np.maximum(50 - 10 * hour_factor + 15 * month_factor + np.random.randn(n) * 10, 0)

    df = pd.DataFrame({
        "pm2_5": pm25,
        "pm10": pm10,
        "nitrogen_dioxide": no2,
        "ozone": ozone,
        "temperature_2m": 22 + 10 * month_factor + 3 * hour_factor + np.random.randn(n) * 2,
        "relative_humidity_2m": np.clip(78 + 12 * hour_factor + np.random.randn(n) * 10, 0, 100),
        "wind_speed_10m": np.maximum(15 + np.random.randn(n) * 8, 0),
        "wind_direction_10m": (45 + 90 * month_factor + np.random.randn(n) * 30) % 360,
        "surface_pressure": 1013 + 5 * month_factor + np.random.randn(n) * 5,
        "precipitation": np.maximum(np.random.exponential(0.5, n), 0),
        "cloud_cover": np.clip(50 + 30 * month_factor + np.random.randn(n) * 15, 0, 100),
    }, index=dates)
    df.index.name = "datetime"

    metadata = {
        "run_timestamp_utc": utc_now_iso(),
        "mode_requested": "sample",
        "source_mode": "sample_generator",
        "fallback_to_sample": False,
        "location": {
            "name": LOCATION_NAME,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
        },
        "timezone": TARGET_TIMEZONE,
        "requested_date_range": {
            "start_date": start_date,
            "end_date": end_date,
            "months_per_chunk": None,
        },
        "source_urls": {},
        "aqi_variables": AQI_VARIABLES,
        "weather_variables": WEATHER_VARIABLES,
        "chunk_reports": [],
        "weather_synthetic_fill": {
            "used": True,
            "filled_cells": int(len(df) * len(WEATHER_VARIABLES)),
            "filled_by_column": {column: int(len(df)) for column in WEATHER_VARIABLES},
            "missing_before_fill": {column: int(len(df)) for column in WEATHER_VARIABLES},
            "missing_after_fill": {column: 0 for column in WEATHER_VARIABLES},
        },
        "raw_data_summary": summarize_dataframe(df),
        "notes": [
            "This dataset is synthetic and is intended only for debugging or offline walkthroughs.",
        ],
    }

    save_collection_artifacts(df, metadata, output_dir, raw_filename, metadata_filename)
    return df, metadata


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect Hong Kong AQI and weather data.")
    parser.add_argument("--mode", choices=["live", "sample"], default="live", help="Data source mode.")
    parser.add_argument("--allow-sample-fallback", action="store_true", help="Fall back to sample data if live collection fails.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Collection start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Collection end date in YYYY-MM-DD format.")
    parser.add_argument("--chunk-months", type=int, default=DEFAULT_CHUNK_MONTHS, help="Months per API collection chunk.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for raw data and metadata artifacts.")
    parser.add_argument("--raw-filename", default=DEFAULT_RAW_DATA_FILE, help="Output CSV filename.")
    parser.add_argument("--metadata-filename", default=DEFAULT_METADATA_FILE, help="Output metadata JSON filename.")
    return parser.parse_args()


def print_summary(df: pd.DataFrame, metadata: dict) -> None:
    """Print a concise run summary to stdout."""
    summary = metadata["raw_data_summary"]

    print("\nData Summary:")
    print("-" * 40)
    print(f"Source mode: {metadata['source_mode']}")
    print(f"Total records: {summary['records']}")
    print(f"Date range: {summary['date_start']} to {summary['date_end']}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['missing_by_column'] if summary['missing_by_column'] else 'None'}")

    synthetic_fill = metadata["weather_synthetic_fill"]
    if synthetic_fill["used"]:
        print(f"Synthetic weather fill used: {synthetic_fill['filled_cells']} cells")
    else:
        print("Synthetic weather fill used: No")


def main() -> None:
    """Entry point for data collection."""
    args = parse_args()

    print("=" * 60)
    print("Hong Kong AQI & Weather Data Collection")
    print("=" * 60)

    try:
        if args.mode == "sample":
            df, metadata = generate_sample_data(
                args.start_date,
                args.end_date,
                args.output_dir,
                args.raw_filename,
                args.metadata_filename,
            )
        else:
            try:
                df, metadata = collect_all_data(
                    args.start_date,
                    args.end_date,
                    args.chunk_months,
                    args.output_dir,
                    args.raw_filename,
                    args.metadata_filename,
                )
            except Exception as exc:
                if not args.allow_sample_fallback:
                    raise

                logger.error("Live collection failed: %s", exc)
                logger.warning("Falling back to sample data because --allow-sample-fallback was provided.")
                df, metadata = generate_sample_data(
                    args.start_date,
                    args.end_date,
                    args.output_dir,
                    args.raw_filename,
                    args.metadata_filename,
                )
                metadata["fallback_to_sample"] = True
                metadata["notes"].append(f"Live collection failed before fallback: {exc}")
                write_json(metadata, os.path.join(args.output_dir, args.metadata_filename))

        print_summary(df, metadata)
        print("\nData collection complete!")
    except Exception as exc:
        logger.error("Data collection failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
