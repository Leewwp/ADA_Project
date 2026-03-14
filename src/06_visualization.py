"""
Visualization Script
====================
Generates visualizations for EDA and model results.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load data for visualization."""
    processed = pd.read_csv(
        os.path.join(DATA_DIR, 'processed_aqi_data.csv'),
        index_col=0, parse_dates=True
    )
    featured = pd.read_csv(
        os.path.join(DATA_DIR, 'featured_aqi_data.csv'),
        index_col=0, parse_dates=True
    )
    return processed, featured


def plot_time_series(df: pd.DataFrame, save: bool = True):
    """Plot AQI time series."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # PM2.5
    axes[0].plot(df.index, df['pm2_5'], alpha=0.7, linewidth=0.5)
    axes[0].set_title('PM2.5 Concentration Over Time')
    axes[0].set_ylabel('PM2.5 (μg/m³)')
    axes[0].grid(True, alpha=0.3)

    # PM10
    axes[1].plot(df.index, df['pm10'], alpha=0.7, linewidth=0.5, color='orange')
    axes[1].set_title('PM10 Concentration Over Time')
    axes[1].set_ylabel('PM10 (μg/m³)')
    axes[1].grid(True, alpha=0.3)

    # Ozone
    axes[2].plot(df.index, df['ozone'], alpha=0.7, linewidth=0.5, color='green')
    axes[2].set_title('Ozone Concentration Over Time')
    axes[2].set_ylabel('Ozone (μg/m³)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'time_series.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


def plot_daily_pattern(df: pd.DataFrame, save: bool = True):
    """Plot daily pollution patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PM2.5 by hour
    hourly_pm25 = df.groupby(df.index.hour)['pm2_5'].mean()
    axes[0, 0].bar(hourly_pm25.index, hourly_pm25.values, color='steelblue')
    axes[0, 0].set_title('Average PM2.5 by Hour')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('PM2.5 (μg/m³)')

    # PM2.5 by month
    monthly_pm25 = df.groupby(df.index.month)['pm2_5'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    axes[0, 1].bar(range(1, 13), monthly_pm25.values, color='coral')
    axes[0, 1].set_title('Average PM2.5 by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('PM2.5 (μg/m³)')
    axes[0, 1].set_xticks(range(1, 13))
    axes[0, 1].set_xticklabels(months, rotation=45)

    # PM2.5 by day of week
    dow_pm25 = df.groupby(df.index.dayofweek)['pm2_5'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), dow_pm25.values, color='seagreen')
    axes[1, 0].set_title('Average PM2.5 by Day of Week')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('PM2.5 (μg/m³)')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(days)

    # Temperature vs PM2.5
    axes[1, 1].scatter(df['temperature_2m'], df['pm2_5'], alpha=0.3, s=5)
    axes[1, 1].set_title('Temperature vs PM2.5')
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('PM2.5 (μg/m³)')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'daily_pattern.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, save: bool = True):
    """Plot correlation matrix."""
    # Select numeric columns
    numeric_cols = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone',
                    'temperature_2m', 'wind_speed_10m', 'relative_humidity_2m']

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Air Quality Variables')
    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'correlation_matrix.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


def plot_distribution(df: pd.DataFrame, save: bool = True):
    """Plot distribution of key variables."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    variables = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'temperature_2m', 'wind_speed_10m']

    for i, var in enumerate(variables):
        ax = axes[i // 3, i % 3]
        ax.hist(df[var].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{var} Distribution')
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency')

        # Add statistics
        mean_val = df[var].mean()
        std_val = df[var].std()
        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
        ax.legend(fontsize=8)

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'distribution.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    """Plot model comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    models = results_df['Model'].values
    x = np.arange(len(models))
    width = 0.6

    # MAE
    axes[0].bar(x, results_df['MAE'], width, color='steelblue')
    axes[0].set_title('Mean Absolute Error (MAE)')
    axes[0].set_ylabel('MAE')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')

    # RMSE
    axes[1].bar(x, results_df['RMSE'], width, color='coral')
    axes[1].set_title('Root Mean Squared Error (RMSE)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')

    # R2
    axes[2].bar(x, results_df['R2'], width, color='seagreen')
    axes[2].set_title('R-squared (R²)')
    axes[2].set_ylabel('R²')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


def plot_aqi_calendar(df: pd.DataFrame, save: bool = True):
    """Plot AQI as a calendar heatmap."""
    # Calculate daily AQI
    daily_aqi = df['pm2_5'].resample('D').mean()

    # Create pivot table for heatmap
    df_temp = pd.DataFrame({'date': daily_aqi.index, 'aqi': daily_aqi.values})
    df_temp['week'] = df_temp['date'].dt.isocalendar().week
    df_temp['dayofweek'] = df_temp['date'].dt.dayofweek

    # Pivot
    pivot = df_temp.pivot_table(values='aqi', index='week', columns='dayofweek', aggfunc='mean')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.diverging_palette(150, 10, as_cmap=True)
    sns.heatmap(pivot, cmap='RdYlGn_r', annot=False, ax=ax, cbar_kws={'label': 'PM2.5'})
    ax.set_title('Daily PM2.5 Heatmap')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Week')

    # Add day labels
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticklabels(days)

    plt.tight_layout()

    if save:
        filepath = os.path.join(OUTPUT_DIR, 'aqi_calendar.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    plt.close()


def generate_all_visualizations():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    # Load data
    processed, featured = load_data()

    print("\nGenerating time series plot...")
    plot_time_series(processed)

    print("Generating daily pattern plot...")
    plot_daily_pattern(processed)

    print("Generating correlation matrix...")
    plot_correlation_matrix(processed)

    print("Generating distribution plot...")
    plot_distribution(processed)

    print("Generating AQI calendar...")
    plot_aqi_calendar(processed)

    # Try to load evaluation results for model comparison
    try:
        results_file = os.path.join(os.path.dirname(__file__), '..', 'report', 'evaluation_results.csv')
        results_df = pd.read_csv(results_file)
        print("Generating model comparison plot...")
        plot_model_comparison(results_df)
    except Exception as e:
        print(f"Skipping model comparison: {e}")

    print("\nAll visualizations saved!")


if __name__ == "__main__":
    generate_all_visualizations()
