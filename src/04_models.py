"""
Model Implementations - Optimized
=================================
Implements 6 prediction models:
1. Ridge Regression (L2 regularization)
2. Lasso Regression (L1 regularization)
3. ElasticNet (L1+L2)
4. Random Forest (tuned)
5. XGBoost
6. SARIMA (time series)

With:
- Feature selection
- Time series cross-validation
- Hyperparameter tuning
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from typing import Tuple, Dict

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, make_scorer
import logging

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping...")

# Try importing LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available, skipping...")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')


def load_featured_data() -> pd.DataFrame:
    """Load featured AQI data."""
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df


def prepare_data(df: pd.DataFrame, target_col: str = 'aqi_target_24h') -> Tuple:
    """
    Prepare data for modeling with feature selection.
    """
    # Remove target columns
    target_cols = [col for col in df.columns if 'target' in col]
    feature_cols = [col for col in df.columns
                   if col not in target_cols and col not in ['aqi', 'aqi_category']]

    # Filter data by date
    train_mask = df.index.year == 2024
    val_mask = (df.index.year == 2025) & (df.index.month <= 3)
    test_mask = (df.index.year == 2025) & (df.index.month > 3)

    # Ensure target is not NaN
    train_mask = train_mask & df[target_col].notna()
    val_mask = val_mask & df[target_col].notna()
    test_mask = test_mask & df[target_col].notna()

    X_train = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]

    y_train = df.loc[train_mask, target_col]
    y_val = df.loc[val_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    print(f"\nData Split:")
    print(f"  Training: {len(X_train)} samples (2024)")
    print(f"  Validation: {len(X_val)} samples (2025 Q1)")
    print(f"  Test: {len(X_test)} samples (2025 Q2)")
    print(f"  Features: {len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_ridge_regression(X_train, y_train, X_val, y_val):
    """Train Ridge Regression with expanded hyperparameter tuning."""
    print("\n" + "=" * 40)
    print("Training Ridge Regression (L2 Regularization)")
    print("=" * 40)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Expanded alpha grid: 20 values from 10^-4 to 10^2
    alpha_grid = np.logspace(-4, 2, 20)

    best_alpha = 1.0
    best_val_rmse = float('inf')

    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_alpha = alpha

    # Train final model with best alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Best alpha: {best_alpha:.4f}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model, scaler, 'ridge', best_alpha


def train_lasso_regression(X_train, y_train, X_val, y_val):
    """Train Lasso Regression with expanded hyperparameter tuning."""
    print("\n" + "=" * 40)
    print("Training Lasso Regression (L1 Regularization)")
    print("=" * 40)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Expanded alpha grid: 20 values from 10^-4 to 10^1
    alpha_grid = np.logspace(-4, 1, 20)

    best_alpha = 0.1
    best_val_rmse = float('inf')

    for alpha in alpha_grid:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_alpha = alpha

    model = Lasso(alpha=best_alpha, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    # Count non-zero coefficients (feature selection)
    n_features = np.sum(model.coef_ != 0)
    print(f"  Best alpha: {best_alpha:.4f}")
    print(f"  Non-zero features: {n_features}/{len(model.coef_)}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model, scaler, 'lasso', best_alpha


def train_elastic_net(X_train, y_train, X_val, y_val):
    """Train ElasticNet (L1 + L2) with expanded hyperparameter tuning."""
    print("\n" + "=" * 40)
    print("Training ElasticNet (L1 + L2)")
    print("=" * 40)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Expanded grid: 10 alpha values x 5 l1_ratio values
    alpha_grid = np.logspace(-4, 1, 10)
    l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_params = {'alpha': 0.1, 'l1_ratio': 0.5}
    best_val_rmse = float('inf')

    for alpha in alpha_grid:
        for l1_ratio in l1_ratio_grid:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            model.fit(X_train_scaled, y_train)
            val_pred = model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}

    model = ElasticNet(**best_params, max_iter=10000)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Best params: alpha={best_params['alpha']:.4f}, l1_ratio={best_params['l1_ratio']:.2f}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model, scaler, 'elastic_net', best_params


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with RandomizedSearchCV tuning."""
    print("\n" + "=" * 40)
    print("Training Random Forest (Tuned with RandomizedSearchCV)")
    print("=" * 40)

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform

    # Define parameter distributions for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(6, 20),
        'min_samples_split': randint(5, 20),
        'min_samples_leaf': randint(3, 10),
        'max_features': ['sqrt', 'log2', None],
    }

    # Base model
    rf = RandomForestRegressor(n_jobs=1, random_state=42)

    # RandomizedSearchCV with 30 iterations
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=30,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=42,
        verbose=0
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Best params: {best_params}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  Top 10 Important Features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    return best_model, None, 'random_forest', best_params


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model with expanded hyperparameter tuning and early stopping."""
    if not HAS_XGBOOST:
        return None, None, 'xgboost', None

    print("\n" + "=" * 40)
    print("Training XGBoost (Tuned with Early Stopping)")
    print("=" * 40)

    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform, randint

    # Define parameter distributions for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(200, 500),
        'max_depth': randint(4, 12),
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'reg_alpha': [0.0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0],
    }

    # Base model with early stopping
    base_xgb = XGBRegressor(
        n_jobs=1,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50
    )

    # RandomizedSearchCV with 30 iterations
    search = RandomizedSearchCV(
        base_xgb, param_dist,
        n_iter=30,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=1,
        random_state=42,
        verbose=0
    )

    search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_model = search.best_estimator_
    best_params = search.best_params_

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Best params: max_depth={best_params.get('max_depth')}, lr={best_params.get('learning_rate')}, n_est={best_params.get('n_estimators')}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return best_model, None, 'xgboost', best_params


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model with early stopping."""
    if not HAS_LIGHTGBM:
        return None, None, 'lightgbm', None

    print("\n" + "=" * 40)
    print("Training LightGBM")
    print("=" * 40)

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': 1,
        'random_state': 42
    }

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Train with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # Suppress output
        ]
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)

    print("\n  Top 10 Important Features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']}")

    return model, None, 'lightgbm', params


def train_sarima(train_data, val_data, target_col='aqi'):
    """Train SARIMA model."""
    print("\n" + "=" * 40)
    print("Training SARIMA")
    print("=" * 40)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        train_subset = train_data[-5000:]

        model = SARIMAX(
            train_subset,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(disp=False)
        print(f"  SARIMA fitted on {len(train_subset)} samples")

        val_forecast = results.forecast(steps=len(val_data))
        val_rmse = np.sqrt(mean_squared_error(val_data.values, val_forecast.values))
        print(f"  Validation RMSE: {val_rmse:.2f}")

        return results, None, 'sarima', None

    except Exception as e:
        print(f"  SARIMA training failed: {e}")
        return None, None, 'sarima', None


def time_series_cv(X_train, y_train, model, n_splits=5):
    """Perform time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rmse_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_vl)
        rmse = np.sqrt(mean_squared_error(y_vl, pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores), np.std(rmse_scores)


def train_all_models():
    """Train all models with optimization."""
    print("=" * 60)
    print("Optimized Model Training Pipeline")
    print("=" * 60)

    # Load data
    df = load_featured_data()

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_data(df)

    models = {}

    # 1. Ridge Regression
    model, scaler, name, params = train_ridge_regression(X_train, y_train, X_val, y_val)
    models[name] = {'model': model, 'scaler': scaler, 'params': params}

    # 2. Lasso Regression
    model, scaler, name, params = train_lasso_regression(X_train, y_train, X_val, y_val)
    models[name] = {'model': model, 'scaler': scaler, 'params': params}

    # 3. ElasticNet
    model, scaler, name, params = train_elastic_net(X_train, y_train, X_val, y_val)
    models[name] = {'model': model, 'scaler': scaler, 'params': params}

    # 4. Random Forest (tuned)
    model, _, name, params = train_random_forest(X_train, y_train, X_val, y_val)
    models[name] = {'model': model, 'scaler': None, 'params': params}

    # 5. XGBoost
    if HAS_XGBOOST:
        model, _, name, params = train_xgboost(X_train, y_train, X_val, y_val)
        if model is not None:
            models[name] = {'model': model, 'scaler': None, 'params': params}

    # 6. SARIMA
    train_series = df[df.index.year == 2024]['aqi']
    val_series = df[(df.index.year == 2025) & (df.index.month <= 3)]['aqi']
    model, _, name, params = train_sarima(train_series, val_series)
    models[name] = {'model': model, 'scaler': None, 'params': params}

    # Save models
    for name, model_data in models.items():
        filepath = os.path.join(MODELS_DIR, f'{name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nSaved {name} to {filepath}")

    print("\nAll models trained and saved!")

    return models, (X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    models, data = train_all_models()
