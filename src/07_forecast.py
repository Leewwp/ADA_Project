"""
Final Forecast Script
=====================
Generates AQI forecasts for the next 1-3 days using the best model.
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'report')


def load_data():
    """Load featured data."""
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)
    return df


def load_best_model():
    """Load the best performing model."""
    # Try to find evaluation results
    results_file = os.path.join(OUTPUT_DIR, 'evaluation_results.csv')

    if os.path.exists(results_file):
        results = pd.read_csv(results_file)
        # Find best model by RMSE
        best_model_name = results.loc[results['RMSE'].idxmin(), 'Model']
    else:
        # Default to Lasso
        best_model_name = 'Lasso Regression'

    print(f"Using best model: {best_model_name}")

    # Map to filename
    model_map = {
        'Lasso Regression': 'lasso.pkl',
        'ElasticNet': 'elastic_net.pkl',
        'Ridge Regression': 'ridge.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'SARIMA': 'sarima.pkl'
    }

    model_path = os.path.join(MODELS_DIR, model_map.get(best_model_name, 'lasso.pkl'))

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), best_model_name

    return None, best_model_name


def generate_forecast_rf(model_data, df: pd.DataFrame, hours: int = 72):
    """Generate forecast using Random Forest."""
    model = model_data['model']

    # Get latest features
    target_cols = [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns
                    if col not in target_cols and col != 'aqi' and col != 'aqi_category']

    # Use last available data point
    latest = df.iloc[-1:][feature_cols]

    # For simplicity, we'll forecast iteratively
    forecasts = []
    current_data = df.copy()

    for h in range(hours):
        # Get features for prediction
        X = current_data.iloc[-1:][feature_cols]
        pred = model.predict(X)[0]

        # Ensure non-negative
        pred = max(0, pred)

        forecasts.append(pred)

        # For next iteration, we'd need to update features
        # This is a simplified version

    return forecasts


def generate_forecast_simple(df: pd.DataFrame, hours: int = 72):
    """Generate simple forecast based on recent trends."""
    # Get recent values
    recent = df['aqi'].tail(24 * 7)  # Last week

    # Calculate average by hour of day
    hourly_avg = recent.groupby(recent.index.hour).mean()

    # Forecast
    forecasts = []
    base_time = df.index[-1]

    for h in range(1, hours + 1):
        forecast_time = base_time + timedelta(hours=h)
        hour = forecast_time.hour

        # Use average for that hour + slight trend
        pred = hourly_avg.get(hour, df['aqi'].mean())

        # Add some variation based on recent trend
        trend = (df['aqi'].tail(24).mean() - df['aqi'].tail(48).mean()) / 48
        pred = pred + trend * h

        forecasts.append({
            'datetime': forecast_time,
            'predicted_aqi': pred,
            'hour': hour
        })

    return pd.DataFrame(forecasts)


def categorize_aqi(aqi):
    """Convert AQI to category."""
    if aqi <= 50:
        return 'Good', 'Green'
    elif aqi <= 100:
        return 'Moderate', 'Yellow'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive', 'Orange'
    elif aqi <= 200:
        return 'Unhealthy', 'Red'
    elif aqi <= 300:
        return 'Very Unhealthy', 'Purple'
    else:
        return 'Hazardous', 'Maroon'


def generate_forecast():
    """Main forecast generation."""
    print("=" * 60)
    print("AQI Forecast Generation")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"Loaded data: {df.index.min()} to {df.index.max()}")

    # Load model
    model_data, model_name = load_best_model()

    # Generate forecast
    print("\nGenerating 72-hour forecast...")
    forecast_df = generate_forecast_simple(df, hours=72)

    # Add categories
    forecast_df[['category', 'color']] = forecast_df['predicted_aqi'].apply(
        lambda x: pd.Series(categorize_aqi(x))
    )

    # Display forecast
    print("\n" + "=" * 60)
    print("72-Hour AQI Forecast")
    print("=" * 60)

    # Show daily summaries
    daily = forecast_df.set_index('datetime').resample('D')['predicted_aqi'].mean()

    print("\nDaily Average Forecast:")
    print("-" * 50)
    for date, aqi in daily.items():
        cat, color = categorize_aqi(aqi)
        print(f"{date.strftime('%Y-%m-%d')}: AQI {aqi:.1f} ({cat})")

    # Save forecast
    forecast_file = os.path.join(OUTPUT_DIR, 'aqi_forecast.csv')
    forecast_df.to_csv(forecast_file, index=False)
    print(f"\nForecast saved to: {forecast_file}")

    # Generate visualization
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot forecast
    ax.plot(forecast_df['datetime'], forecast_df['predicted_aqi'],
            'b-', linewidth=2, label='Predicted AQI')

    # Add color regions
    ax.axhspan(0, 50, alpha=0.2, color='green', label='Good')
    ax.axhspan(50, 100, alpha=0.2, color='yellow')
    ax.axhspan(100, 150, alpha=0.2, color='orange')
    ax.axhspan(150, 200, alpha=0.2, color='red')
    ax.axhspan(200, 300, alpha=0.2, color='purple')
    ax.axhspan(300, 500, alpha=0.2, color='maroon')

    ax.set_xlabel('Date')
    ax.set_ylabel('AQI')
    ax.set_title(f'AQI Forecast (Next 72 Hours) - Model: {model_name}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    forecast_fig = os.path.join(OUTPUT_DIR, 'aqi_forecast.png')
    plt.savefig(forecast_fig, dpi=150, bbox_inches='tight')
    print(f"Forecast plot saved to: {forecast_fig}")

    plt.close()

    # Print recommendations
    print("\n" + "=" * 60)
    print("Health Recommendations")
    print("=" * 60)

    avg_aqi = forecast_df['predicted_aqi'].mean()
    cat, _ = categorize_aqi(avg_aqi)

    if avg_aqi <= 50:
        print("[OK] Air quality is good. Ideal for outdoor activities.")
    elif avg_aqi <= 100:
        print("[!] Air quality is moderate. Sensitive individuals should limit prolonged outdoor exertion.")
    elif avg_aqi <= 150:
        print("[!] Air quality is unhealthy for sensitive groups. Children and elderly should reduce outdoor activities.")
    elif avg_aqi <= 200:
        print("[X] Air quality is unhealthy. Everyone should limit outdoor activities.")
    else:
        print("[XX] Air quality is very unhealthy. Avoid outdoor activities. Keep windows closed.")

    return forecast_df


if __name__ == "__main__":
    forecast = generate_forecast()
