"""
Feature Engineering Script
==========================
Advanced feature engineering for AQI prediction with:
1. Extended time features
2. Meteorological interaction features
3. Atmospheric stability indicators
4. Pollution accumulation indices
5. Seasonal decomposition features
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_aqi_data.csv')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')


def load_processed_data() -> pd.DataFrame:
    """Load preprocessed AQI data."""
    df = pd.read_csv(PROCESSED_DATA_FILE, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive time-based features.
    """
    df = df.copy()

    # Basic time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Time of day categories
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    # Rush hour indicator (7-9 AM and 5-8 PM)
    df['is_rush_hour'] = (((df['hour'] >= 7) & (df['hour'] <= 9)) |
                          ((df['hour'] >= 17) & (df['hour'] <= 20))).astype(int)

    # Weekend/holiday indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Season encoding
    df['season'] = df['month'].map({
        12: 0, 1: 0, 2: 0,  # Winter (dry)
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer (wet)
        9: 3, 10: 3, 11: 3  # Autumn
    })

    # Cyclical encoding for periodic features
    # Hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Day of week (7-day cycle)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Day of year (365-day cycle)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    print(f"Created time features")
    return df


def create_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from meteorological variables.
    """
    df = df.copy()

    # Wind features
    if 'wind_speed_10m' in df.columns:
        # Wind speed categories
        df['wind_calm'] = (df['wind_speed_10m'] < 5).astype(int)
        df['wind_moderate'] = ((df['wind_speed_10m'] >= 5) & (df['wind_speed_10m'] < 15)).astype(int)
        df['wind_strong'] = (df['wind_speed_10m'] >= 15).astype(int)

        # Wind power (proportional to v^3)
        df['wind_power'] = df['wind_speed_10m'] ** 3 / 1000

    # Wind direction features (if available)
    if 'wind_direction_10m' in df.columns:
        # Convert wind direction to compass sectors
        df['wind_dir_sector'] = (df['wind_direction_10m'] / 45).astype(int) % 8

        # Wind direction components (for pollution transport)
        df['wind_u'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
        df['wind_v'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))

    # Temperature features
    if 'temperature_2m' in df.columns:
        # Temperature categories
        df['temp_cold'] = (df['temperature_2m'] < 15).astype(int)
        df['temp_moderate'] = ((df['temperature_2m'] >= 15) & (df['temperature_2m'] < 25)).astype(int)
        df['temp_hot'] = (df['temperature_2m'] >= 25).astype(int)

        # Temperature deviation from daily mean (will be computed after rolling)
        df['temp_deviation'] = df['temperature_2m'] - df['temperature_2m'].rolling(24, min_periods=1).mean()

    # Humidity features
    if 'relative_humidity_2m' in df.columns:
        # Humidity categories
        df['humidity_low'] = (df['relative_humidity_2m'] < 60).astype(int)
        df['humidity_high'] = (df['relative_humidity_2m'] >= 80).astype(int)

        # Humidity-PM interaction (high humidity = more particle bound PM)
        if 'pm2_5' in df.columns:
            df['humidity_pm25_interaction'] = df['relative_humidity_2m'] * df['pm2_5'] / 100

    # Pressure features (if available)
    if 'surface_pressure' in df.columns:
        # Pressure change (indicator of weather systems)
        df['pressure_change_3h'] = df['surface_pressure'].diff(3)
        df['pressure_change_24h'] = df['surface_pressure'].diff(24)

        # Low pressure system (often associated with poor dispersion)
        df['low_pressure'] = (df['surface_pressure'] < 1010).astype(int)

    # Precipitation features (if available)
    if 'precipitation' in df.columns:
        df['is_raining'] = (df['precipitation'] > 0.1).astype(int)
        df['heavy_rain'] = (df['precipitation'] > 5).astype(int)

    # Cloud cover features (if available)
    if 'cloud_cover' in df.columns:
        df['cloudy'] = (df['cloud_cover'] > 70).astype(int)
        df['clear_sky'] = (df['cloud_cover'] < 30).astype(int)

    print(f"Created meteorological features")
    return df


def create_atmospheric_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to atmospheric stability.
    These affect pollution accumulation/dispersion.
    """
    df = df.copy()

    # Temperature inversion indicator (stable atmosphere = poor dispersion)
    # Simplified: cold night + high pressure = likely inversion
    if 'temperature_2m' in df.columns and 'surface_pressure' in df.columns:
        # Night time temperature inversion potential
        night_temp = df[df['hour'].isin([0, 1, 2, 3, 4, 5])]['temperature_2m']
        day_temp = df[df['hour'].isin([12, 13, 14, 15])]['temperature_2m']

        # Simplified inversion proxy
        df['inversion_potential'] = (
            (df['hour'].isin([0, 1, 2, 3, 4, 5])) &
            (df['surface_pressure'] > 1015) &
            (df['temperature_2m'] > df['temperature_2m'].shift(12))
        ).astype(int)

    # Ventilation index proxy (higher = better dispersion)
    if 'wind_speed_10m' in df.columns and 'boundary_layer_height' not in df.columns:
        # Simplified ventilation: wind speed * mixing height proxy
        # Use temperature as proxy for mixing height
        if 'temperature_2m' in df.columns:
            mixing_proxy = np.clip(df['temperature_2m'] + 10, 0, 100)  #简化
            df['ventilation_index'] = df['wind_speed_10m'] * mixing_proxy

    # Atmospheric stability based on wind speed and time of day
    if 'wind_speed_10m' in df.columns:
        # Stable: low wind + night
        # Unstable: higher wind + day
        df['stability_index'] = np.where(
            (df['wind_speed_10m'] < 5) & (df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])),
            3,  # Very stable
            np.where(
                (df['wind_speed_10m'] < 10) & (df['hour'].isin([6, 7, 8, 9, 20, 21])),
                2,  # Stable
                np.where(
                    (df['wind_speed_10m'] > 15) & (df['hour'].isin([10, 11, 12, 13, 14, 15, 16, 17])),
                    0,  # Unstable
                    1   # Neutral
                )
            )
        )

    print(f"Created atmospheric stability features")
    return df


def create_pollution_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pollution-specific indices.
    """
    df = df.copy()

    # Primary pollutants
    pollutants = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone']

    # PM2.5/PM10 ratio (indicates pollution source)
    if 'pm2_5' in df.columns and 'pm10' in df.columns:
        df['pm25_pm10_ratio'] = df['pm2_5'] / (df['pm10'] + 0.1)
        # High ratio = more secondary/vehicular, Low ratio = more crustal/dust

    # NOx reactivity (ozone formation potential)
    if 'nitrogen_dioxide' in df.columns and 'ozone' in df.columns:
        df['nox_ozone_interaction'] = df['nitrogen_dioxide'] * df['ozone'] / 100

    # Ozone exceedance indicator
    if 'ozone' in df.columns:
        df['ozone_high'] = (df['ozone'] > 100).astype(int)

    # PM2.5 exceedance indicator
    if 'pm2_5' in df.columns:
        df['pm25_high'] = (df['pm2_5'] > 35).astype(int)
        df['pm25_very_high'] = (df['pm2_5'] > 75).astype(int)

    # Pollution accumulation index (simplified)
    if all(p in df.columns for p in pollutants):
        # Weighted sum normalized
        df['pollution_index'] = (
            df['pm2_5'] / 35 * 0.3 +
            df['pm10'] / 50 * 0.2 +
            df['nitrogen_dioxide'] / 40 * 0.2 +
            df['ozone'] / 100 * 0.3
        )

    print(f"Created pollution indices")
    return df


def create_lag_features(df: pd.DataFrame, target_columns: list = None) -> pd.DataFrame:
    """
    Create lag features with more comprehensive coverage.
    """
    df = df.copy()

    if target_columns is None:
        target_columns = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone']

    # Extended lag periods
    lag_periods = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # Added 168 (1 week)

    for col in target_columns:
        if col in df.columns:
            for lag in lag_periods:
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

    print(f"Created lag features for {len(target_columns)} columns")
    return df


def create_rolling_features(df: pd.DataFrame, target_columns: list = None) -> pd.DataFrame:
    """
    Create rolling window statistics with multiple window sizes.
    """
    df = df.copy()

    if target_columns is None:
        target_columns = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'temperature_2m', 'wind_speed_10m']

    # Various window sizes
    windows = [3, 6, 12, 24, 48, 72, 168]  # Added weekly window

    for col in target_columns:
        if col in df.columns:
            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()

                # Rolling standard deviation (volatility)
                df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()

                # Rolling min/max range
                df[f'{col}_rolling_range_{window}h'] = (
                    df[col].rolling(window=window, min_periods=1).max() -
                    df[col].rolling(window=window, min_periods=1).min()
                )

    print(f"Created rolling features for {len(target_columns)} columns")
    return df


def create_diff_features(df: pd.DataFrame, target_columns: list = None) -> pd.DataFrame:
    """
    Create difference and rate of change features.
    """
    df = df.copy()

    if target_columns is None:
        target_columns = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone']

    for col in target_columns:
        if col in df.columns:
            # 1st order differences
            df[f'{col}_diff_1h'] = df[col].diff(1)
            df[f'{col}_diff_3h'] = df[col].diff(3)
            df[f'{col}_diff_6h'] = df[col].diff(6)
            df[f'{col}_diff_24h'] = df[col].diff(24)

            # Safe percentage change (avoid division by zero)
            prev_val = df[col].shift(1)
            df[f'{col}_pct_change_1h'] = (df[col] - prev_val) / (prev_val.abs() + 0.1)
            prev_val_24 = df[col].shift(24)
            df[f'{col}_pct_change_24h'] = (df[col] - prev_val_24) / (prev_val_24.abs() + 0.1)

            # Acceleration (2nd derivative)
            df[f'{col}_accel_1h'] = df[f'{col}_diff_1h'].diff(1)

    print(f"Created difference features")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between variables.
    """
    df = df.copy()

    # Temperature-Wind interaction (affects dispersion)
    if 'temperature_2m' in df.columns and 'wind_speed_10m' in df.columns:
        df['temp_wind_interaction'] = df['temperature_2m'] * df['wind_speed_10m']

    # Humidity-Wind interaction
    if 'relative_humidity_2m' in df.columns and 'wind_speed_10m' in df.columns:
        df['humidity_wind_interaction'] = df['relative_humidity_2m'] * df['wind_speed_10m']

    # Pressure-Altitude proxy (higher pressure = more stable)
    if 'surface_pressure' in df.columns and 'wind_speed_10m' in df.columns:
        df['pressure_wind_ratio'] = df['surface_pressure'] / (df['wind_speed_10m'] + 1)

    # Temperature-Humidity (heat index proxy)
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['heat_index'] = df['temperature_2m'] + 0.5 * df['relative_humidity_2m']

    print(f"Created interaction features")
    return df


def create_aqi_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create AQI target variables for different prediction horizons.
    Also creates AQI autoregressive features (lags and rolling stats) for better prediction.
    """
    df = df.copy()

    # Calculate AQI from PM2.5 (24-hour average)
    pm25 = df['pm2_5']

    def pm25_to_aqi(pm):
        if pd.isna(pm):
            return np.nan
        elif pm <= 35:
            return (50 / 35) * pm
        elif pm <= 75:
            return 50 + (50 / 40) * (pm - 35)
        elif pm <= 115:
            return 100 + (50 / 40) * (pm - 75)
        elif pm <= 150:
            return 150 + (50 / 35) * (pm - 115)
        elif pm <= 250:
            return 200 + (100 / 100) * (pm - 150)
        else:
            return 300 + (100 / 100) * (pm - 250)

    df['aqi'] = pm25.apply(pm25_to_aqi)

    # AQI category (1-6)
    df['aqi_category'] = pd.cut(
        df['aqi'],
        bins=[-np.inf, 50, 100, 150, 200, 300, np.inf],
        labels=[1, 2, 3, 4, 5, 6]
    ).astype(int)

    # ============================================
    # ADD AQI AUTOREGRESSIVE FEATURES (KEY FOR PREDICTION)
    # AQI has strong autocorrelation - its history is the most powerful predictor
    # ============================================

    # AQI lag features
    lag_periods = [1, 3, 6, 12, 24, 48, 72, 168]  # hours
    for lag in lag_periods:
        df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)

    # AQI rolling statistics
    rolling_windows = [6, 12, 24, 48]
    for window in rolling_windows:
        df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).mean()
        df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).std()
        # AQI change over window
        df[f'aqi_change_{window}h'] = df['aqi'] - df['aqi'].shift(window)

    # AQI momentum (rate of change)
    df['aqi_momentum_1h'] = df['aqi'].diff(1)
    df['aqi_momentum_6h'] = df['aqi'].diff(6)
    df['aqi_momentum_24h'] = df['aqi'].diff(24)

    # AQI percentile (relative position)
    df['aqi_percentile_24h'] = df['aqi'].rolling(window=24, min_periods=1).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 0.1) if len(x) > 1 else 0.5
    )

    print(f"Created AQI autoregressive features")

    # Future AQI targets (for 1-3 day prediction)
    for hours in [24, 48, 72]:
        df[f'aqi_target_{hours}h'] = df['aqi'].shift(-hours)

    # Also create targets for individual pollutants
    for pollutant in ['pm2_5', 'pm10', 'ozone']:
        if pollutant in df.columns:
            for hours in [24, 48]:
                df[f'{pollutant}_target_{hours}h'] = df[pollutant].shift(-hours)

    print(f"Created AQI target variables")
    return df


def create_all_features():
    """Main feature engineering pipeline."""
    print("=" * 60)
    print("Advanced Feature Engineering Pipeline")
    print("=" * 60)

    # Load processed data
    df = load_processed_data()

    original_cols = len(df.columns)

    # Create all feature groups
    # IMPORTANT: create_aqi_target() is called BEFORE lag/rolling features
    # This ensures AQI autoregressive features (lags, rolling stats) are created
    # from the historical AQI values, not the future targets
    df = create_time_features(df)
    df = create_meteorological_features(df)
    df = create_atmospheric_stability_features(df)
    df = create_pollution_indices(df)
    df = create_aqi_target(df)  # Create AQI and its autoregressive features FIRST
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_diff_features(df)
    df = create_interaction_features(df)

    # Clean up infinity values
    # Replace infinity with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN from lag/diff operations
    initial_len = len(df)
    df = df.dropna()
    print(f"\nDropped {initial_len - len(df)} rows with NaN values")

    # Final check: if any NaN/inf remain, fill with median
    if df.isna().any().any():
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # Ensure no infinity values remain
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            median_val = df[col].replace([np.inf, -np.inf], np.nan()).median()
            df[col] = df[col].replace([np.inf, -np.inf], median_val)

    # Summary
    new_cols = len(df.columns) - original_cols
    print(f"\nTotal features: {len(df.columns)} (new: {new_cols})")
    print(f"Final dataset shape: {df.shape}")
    print(f"Any NaN: {df.isna().any().any()}, Any Inf: {np.isinf(df.select_dtypes(include=[np.number])).any().any()}")

    # Save featured data
    df.to_csv(FEATURED_DATA_FILE)
    print(f"Featured data saved to: {FEATURED_DATA_FILE}")

    # Print feature categories
    print("\n" + "=" * 60)
    print("Feature Categories Created:")
    print("=" * 60)
    print("1. Time features (hour, day, month, cyclical encoding)")
    print("2. Meteorological features (wind, temp, humidity categories)")
    print("3. Atmospheric stability (inversion, ventilation)")
    print("4. Pollution indices (PM ratio, exceedance indicators)")
    print("5. Lag features (1h to 168h)")
    print("6. Rolling statistics (mean, std, range)")
    print("7. Difference features (1h to 24h changes)")
    print("8. Interaction features (temp*wind, humidity*wind)")
    print("9. Target variables (AQI for 24h, 48h, 72h)")

    return df


if __name__ == "__main__":
    df = create_all_features()
    print("\nFeature engineering complete!")
    print(f"Total features: {len(df.columns)}")
