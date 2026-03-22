"""
Final Model Evaluation - Fixed
=============================
"""

import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, Tuple, List

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'report', 'evaluation_results.csv')

import warnings
warnings.filterwarnings('ignore')


def load_all_features() -> Tuple[pd.DataFrame, List[str]]:
    """Load data with all features."""
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)

    # All features used in original models
    target_cols = [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns
                   if col not in target_cols and col not in ['aqi', 'aqi_category']]

    return df, feature_cols


def load_selected_features() -> Tuple[pd.DataFrame, List[str]]:
    """Load data with optimized feature selection."""
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)

    # Original selected features PLUS more AQI autoregressive features
    # This combination was found to perform better than dynamic selection
    original_features = [
        'pm2_5', 'month', 'surface_pressure', 'pm2_5_lag_48h', 'nitrogen_dioxide',
        'is_afternoon', 'pm25_pm10_ratio', 'heat_index', 'temperature_2m_rolling_mean_12h',
        'pm10_rolling_range_168h', 'temperature_2m_rolling_std_168h',
        'nitrogen_dioxide_rolling_range_12h', 'pm2_5_lag_168h', 'pm10_rolling_range_48h',
        'pm10_rolling_range_72h', 'pm10', 'ozone'
    ]

    # Add more AQI autoregressive features for better prediction
    aqi_features = ['aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h', 'aqi_lag_12h',
                    'aqi_lag_24h', 'aqi_lag_48h', 'aqi_lag_72h', 'aqi_lag_168h',
                    'aqi_rolling_mean_6h', 'aqi_rolling_mean_12h', 'aqi_rolling_mean_24h', 'aqi_rolling_mean_48h',
                    'aqi_rolling_std_6h', 'aqi_rolling_std_12h', 'aqi_rolling_std_24h', 'aqi_rolling_std_48h',
                    'aqi_momentum_1h', 'aqi_momentum_6h', 'aqi_momentum_24h',
                    'aqi_change_6h', 'aqi_change_12h', 'aqi_change_24h', 'aqi_change_48h']
    feature_cols = original_features + [f for f in aqi_features if f in df.columns]

    return df, feature_cols


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics including direction accuracy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Direction accuracy: Did we correctly predict AQI going up/down?
    if len(y_true) > 1:
        true_direction = np.diff(y_true)
        pred_direction = np.diff(y_pred)
        direction_correct = np.sign(true_direction) == np.sign(pred_direction)
        direction_accuracy = np.mean(direction_correct) * 100
    else:
        direction_accuracy = np.nan

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'Direction Accuracy': direction_accuracy}


def evaluate_models():
    print("="*60)
    print("Final Model Evaluation (Optimized)")
    print("="*60)

    results = []

    # Load data with selected features
    print("\n--- Loading Data with Selected Features ---")
    df, feature_cols_sel = load_selected_features()
    target_col = 'aqi_target_24h'

    test_mask = (df.index.year == 2025) & (df.index.month > 3)
    test_mask = test_mask & df[target_col].notna()
    train_mask = df.index.year == 2024
    val_mask = (df.index.year == 2025) & (df.index.month <= 3)
    val_mask = val_mask & df[target_col].notna()

    X_train = df.loc[train_mask, feature_cols_sel]
    X_val = df.loc[val_mask, feature_cols_sel]
    X_test = df.loc[test_mask, feature_cols_sel]
    y_train = df.loc[train_mask, target_col]
    y_val = df.loc[val_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {len(feature_cols_sel)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Training Models ---")

    # Lasso
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    pred = lasso.predict(X_test_scaled)
    metrics = calculate_metrics(y_test, pred)
    metrics['Model'] = 'Lasso'
    results.append(metrics)
    print(f"  Lasso: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    pred = ridge.predict(X_test_scaled)
    metrics = calculate_metrics(y_test, pred)
    metrics['Model'] = 'Ridge'
    results.append(metrics)
    print(f"  Ridge: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=10,
                              min_samples_leaf=5, n_jobs=1, random_state=42)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    metrics = calculate_metrics(y_test, pred)
    metrics['Model'] = 'Random Forest'
    results.append(metrics)
    print(f"  Random Forest: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")

    # Gradient Boosting
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    pred = gb.predict(X_test)
    metrics = calculate_metrics(y_test, pred)
    metrics['Model'] = 'Gradient Boosting'
    results.append(metrics)
    print(f"  Gradient Boosting: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")

    # XGBoost
    xgb_rmse = None
    xgb_model = None
    try:
        from xgboost import XGBRegressor
        xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        pred = xgb_model.predict(X_test)
        metrics = calculate_metrics(y_test, pred)
        metrics['Model'] = 'XGBoost'
        results.append(metrics)
        print(f"  XGBoost: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")
        xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val)))
    except Exception as e:
        print(f"  XGBoost error: {e}")

    # LightGBM
    lgbm_rmse = None
    lgbm_model = None
    try:
        import lightgbm as lgb
        lgbm_model = lgb.LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8, n_jobs=1, random_state=42, verbose=-1)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        pred = lgbm_model.predict(X_test)
        metrics = calculate_metrics(y_test, pred)
        metrics['Model'] = 'LightGBM'
        results.append(metrics)
        print(f"  LightGBM: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")
        lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_model.predict(X_val)))
    except Exception as e:
        print(f"  LightGBM error: {e}")

    # Ensemble
    print("\n--- Ensemble (Weighted Average) ---")
    lasso_pred = lasso.predict(X_test_scaled)
    ridge_pred = ridge.predict(X_test_scaled)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)

    lasso_val = lasso.predict(X_val_scaled)
    ridge_val = ridge.predict(X_val_scaled)
    rf_val = rf.predict(X_val)
    gb_val = gb.predict(X_val)

    lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_val))
    ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_val))
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_val))
    gb_rmse = np.sqrt(mean_squared_error(y_val, gb_val))

    total_inv = 1/lasso_rmse + 1/ridge_rmse + 1/rf_rmse + 1/gb_rmse
    if xgb_rmse:
        total_inv += 1/xgb_rmse
    if lgbm_rmse:
        total_inv += 1/lgbm_rmse

    w_lasso = (1/lasso_rmse) / total_inv
    w_ridge = (1/ridge_rmse) / total_inv
    w_rf = (1/rf_rmse) / total_inv
    w_gb = (1/gb_rmse) / total_inv
    w_xgb = (1/xgb_rmse) / total_inv if xgb_rmse else 0
    w_lgbm = (1/lgbm_rmse) / total_inv if lgbm_rmse else 0

    xgb_pred = xgb_model.predict(X_test) if xgb_model else 0
    lgbm_pred = lgbm_model.predict(X_test) if lgbm_model else 0

    ensemble_pred = (w_lasso * lasso_pred + w_ridge * ridge_pred +
                     w_rf * rf_pred + w_gb * gb_pred +
                     w_xgb * xgb_pred + w_lgbm * lgbm_pred)

    metrics = calculate_metrics(y_test, ensemble_pred)
    metrics['Model'] = 'Ensemble (Weighted)'
    results.append(metrics)
    print(f"  Ensemble: RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}")
    print(f"    Weights: L={w_lasso:.3f}, R={w_ridge:.3f}, RF={w_rf:.3f}, GB={w_gb:.3f}" +
          (f", XGB={w_xgb:.3f}" if xgb_rmse else "") +
          (f", LGBM={w_lgbm:.3f}" if lgbm_rmse else ""))

    # Results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')

    print("\n" + "="*60)
    print("FINAL RANKING (Test Set)")
    print("="*60)
    display_cols = ['Model', 'MAE', 'RMSE', 'R2']
    if 'Direction Accuracy' in results_df.columns:
        display_cols.append('Direction Accuracy')
    print(results_df[display_cols].to_string(index=False))

    best = results_df.iloc[0]
    print(f"\n*** BEST: {best['Model']} ***")
    print(f"    RMSE: {best['RMSE']:.2f}")
    print(f"    MAE:  {best['MAE']:.2f}")
    print(f"    R2:   {best['R2']:.4f}")
    if 'Direction Accuracy' in best and pd.notna(best.get('Direction Accuracy')):
        print(f"    Direction Accuracy: {best['Direction Accuracy']:.1f}%")

    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved to: {RESULTS_FILE}")

    return results_df


if __name__ == "__main__":
    results = evaluate_models()
