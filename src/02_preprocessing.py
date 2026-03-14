"""
Data Preprocessing Script
=========================
Handles missing values and outliers in the AQI dataset.
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_FILE = os.path.join(DATA_DIR, 'raw_aqi_data.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_aqi_data.csv')


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw AQI data from CSV."""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} records from {filepath}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Check and report missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print("\nMissing Values Analysis:")
    print(missing_df[missing_df['Missing Count'] > 0])
    return missing_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using linear interpolation.

    For time series data, linear interpolation is appropriate
    as it assumes a linear relationship between known values.
    """
    df_clean = df.copy()

    # Use linear interpolation for missing values
    df_clean = df_clean.interpolate(method='linear')

    # For any remaining NaN at the start/end, use forward/backward fill
    df_clean = df_clean.ffill().bfill()

    missing_after = df_clean.isnull().sum().sum()
    print(f"\nMissing values after interpolation: {missing_after}")

    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Args:
        df: DataFrame
        column: Column name to check for outliers
        threshold: IQR multiplier (default 1.5)

    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers


def handle_outliers(df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
    """
    Handle outliers in the dataset using IQR method.

    Args:
        df: DataFrame with potential outliers
        method: 'clip' to cap values at bounds, 'remove' to drop rows

    Returns:
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    outlier_summary = {}

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_count = outliers.sum()

        if outlier_count > 0:
            outlier_summary[col] = {
                'count': outlier_count,
                'percentage': round(outlier_count / len(df_clean) * 100, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2)
            }

            if method == 'clip':
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            elif method == 'remove':
                df_clean = df_clean[~outliers]

    if outlier_summary:
        print("\nOutlier Handling Summary:")
        print("-" * 60)
        for col, stats in outlier_summary.items():
            print(f"{col}:")
            print(f"  Count: {stats['count']} ({stats['percentage']}%)")
            print(f"  Bounds: [{stats['lower_bound']}, {stats['upper_bound']}]")
    else:
        print("\nNo significant outliers detected.")

    return df_clean


def handle_infinity_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle infinity and extremely large values in the dataset.
    These can occur from division operations (pct_change, division by zero).
    """
    df_clean = df.copy()

    # Replace infinity with NaN, then interpolate
    inf_count = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()

    if inf_count > 0:
        print(f"\nHandling {inf_count} infinity values...")

        # Replace inf with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Linear interpolate
        df_clean = df_clean.interpolate(method='linear')

        # Fill remaining NaN
        df_clean = df_clean.ffill().bfill()

        # If still have NaN, fill with column median
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the cleaned dataset.

    Returns:
        Dictionary with validation results
    """
    validation = {
        'total_records': len(df),
        'date_range': (df.index.min(), df.index.max()),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_records': df.index.duplicated().sum(),
        'negative_values': (df.select_dtypes(include=[np.number]) < 0).sum().sum()
    }

    print("\n" + "=" * 60)
    print("Data Validation Results")
    print("=" * 60)
    print(f"Total Records: {validation['total_records']}")
    print(f"Date Range: {validation['date_range'][0]} to {validation['date_range'][1]}")
    print(f"Missing Values: {validation['missing_values']}")
    print(f"Duplicate Timestamps: {validation['duplicate_records']}")
    print(f"Negative Values: {validation['negative_values']}")

    return validation


def preprocess_data():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)

    # Load raw data
    df = load_data(RAW_DATA_FILE)

    # Check missing values
    check_missing_values(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Handle outliers
    df = handle_outliers(df, method='clip')

    # Handle infinity values
    df = handle_infinity_values(df)

    # Validate
    validate_data(df)

    # Save processed data
    df.to_csv(PROCESSED_DATA_FILE)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_FILE}")

    return df


if __name__ == "__main__":
    df = preprocess_data()
    print("\nPreprocessing complete!")
    print(df.head())
