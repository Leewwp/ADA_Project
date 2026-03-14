"""
Model Evaluation Script - Optimized
===================================
Evaluates all models using multiple metrics with proper handling.
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from typing import Dict, Tuple

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'report', 'evaluation_results.csv')

warnings.filterwarnings('ignore')


def load_featured_data() -> pd.DataFrame:
    """Load featured AQI data."""
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)
    return df


def load_models() -> Dict:
    """Load trained models."""
    models = {}
    model_names = ['ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost', 'sarima']

    for name in model_names:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)

    return models


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def prepare_test_data(df: pd.DataFrame) -> Tuple:
    """Prepare test data for evaluation."""
    target_cols = [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns
                   if col not in target_cols and col not in ['aqi', 'aqi_category']]

    # Test set: 2025 Q2
    test_mask = (df.index.year == 2025) & (df.index.month > 3)
    target_col = 'aqi_target_24h'
    test_mask = test_mask & df[target_col].notna()

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, target_col]

    return X_test, y_test


def evaluate_ridge(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate Ridge model."""
    model_data = models.get('ridge')
    if not model_data:
        return None

    model = model_data['model']
    scaler = model_data['scaler']

    try:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['Model'] = 'Ridge Regression'
        return metrics
    except Exception as e:
        print(f"  Ridge evaluation error: {e}")
        return None


def evaluate_lasso(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate Lasso model."""
    model_data = models.get('lasso')
    if not model_data:
        return None

    model = model_data['model']
    scaler = model_data['scaler']

    try:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['Model'] = 'Lasso Regression'
        return metrics
    except Exception as e:
        print(f"  Lasso evaluation error: {e}")
        return None


def evaluate_elastic_net(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate ElasticNet model."""
    model_data = models.get('elastic_net')
    if not model_data:
        return None

    model = model_data['model']
    scaler = model_data['scaler']

    try:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['Model'] = 'ElasticNet'
        return metrics
    except Exception as e:
        print(f"  ElasticNet evaluation error: {e}")
        return None


def evaluate_random_forest(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate Random Forest model."""
    model_data = models.get('random_forest')
    if not model_data:
        return None

    model = model_data['model']

    try:
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['Model'] = 'Random Forest'
        return metrics
    except Exception as e:
        print(f"  Random Forest evaluation error: {e}")
        return None


def evaluate_xgboost(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """Evaluate XGBoost model."""
    model_data = models.get('xgboost')
    if not model_data:
        return None

    model = model_data['model']

    try:
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics['Model'] = 'XGBoost'
        return metrics
    except Exception as e:
        print(f"  XGBoost evaluation error: {e}")
        return None


def evaluate_sarima(models: Dict, df: pd.DataFrame) -> Dict:
    """Evaluate SARIMA model."""
    model_data = models.get('sarima')
    if not model_data or model_data['model'] is None:
        return None

    model = model_data['model']

    test_mask = (df.index.year == 2025) & (df.index.month > 3)
    test_data = df.loc[test_mask, 'aqi']

    try:
        forecast = model.forecast(steps=len(test_data))
        metrics = calculate_metrics(test_data.values, forecast.values)
        metrics['Model'] = 'SARIMA'
        return metrics
    except Exception as e:
        print(f"  SARIMA evaluation error: {e}")
        return None


def evaluate_all_models():
    """Evaluate all models."""
    print("=" * 60)
    print("Optimized Model Evaluation")
    print("=" * 60)

    df = load_featured_data()
    models = load_models()
    print(f"Loaded {len(models)} models")

    X_test, y_test = prepare_test_data(df)
    print(f"Test set: {len(X_test)} samples")

    results = []

    print("\nEvaluating models...")

    # Ridge
    print("  Ridge Regression...", end=" ")
    m = evaluate_ridge(models, X_test, y_test)
    if m:
        results.append(m)
        print(f"RMSE: {m['RMSE']:.2f}")
    else:
        print("Failed")

    # Lasso
    print("  Lasso Regression...", end=" ")
    m = evaluate_lasso(models, X_test, y_test)
    if m:
        results.append(m)
        print(f"RMSE: {m['RMSE']:.2f}")
    else:
        print("Failed")

    # ElasticNet
    print("  ElasticNet...", end=" ")
    m = evaluate_elastic_net(models, X_test, y_test)
    if m:
        results.append(m)
        print(f"RMSE: {m['RMSE']:.2f}")
    else:
        print("Failed")

    # Random Forest
    print("  Random Forest...", end=" ")
    m = evaluate_random_forest(models, X_test, y_test)
    if m:
        results.append(m)
        print(f"RMSE: {m['RMSE']:.2f}")
    else:
        print("Failed")

    # XGBoost
    print("  XGBoost...", end=" ")
    m = evaluate_xgboost(models, X_test, y_test)
    if m:
        results.append(m)
        print(f"RMSE: {m['RMSE']:.2f}")
    else:
        print("Failed")

    # SARIMA
    print("  SARIMA...", end=" ")
    m = evaluate_sarima(models, df)
    if m:
        results.append(m)
        print(f"RMSE: {m['RMSE']:.2f}")
    else:
        print("Failed")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    cols = ['Model', 'MAE', 'RMSE', 'MAPE', 'R2']
    results_df = results_df[cols]
    results_df = results_df.sort_values('RMSE')

    print("\n" + "=" * 60)
    print("Evaluation Results (Test Set: 2025 Q2)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Best model
    best = results_df.iloc[0]
    print(f"\nBest Model: {best['Model']} (RMSE: {best['RMSE']:.2f}, R2: {best['R2']:.4f})")

    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nResults saved to: {RESULTS_FILE}")

    return results_df


if __name__ == "__main__":
    results = evaluate_all_models()
