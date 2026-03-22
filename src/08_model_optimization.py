"""
Model Optimization - Advanced
============================
1. Feature selection based on Lasso importance
2. Ensemble model (weighted average)
3. MLP Neural Network
4. Stacking ensemble
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from typing import Tuple, Dict, List

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FEATURED_DATA_FILE = os.path.join(DATA_DIR, 'featured_aqi_data.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

warnings.filterwarnings('ignore')


def load_featured_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} records")
    return df


def prepare_data(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple:
    """Prepare data for modeling."""
    target_cols = [col for col in df.columns if 'target' in col]

    if feature_cols is None:
        feature_cols = [col for col in df.columns
                       if col not in target_cols and col not in ['aqi', 'aqi_category']]

    train_mask = df.index.year == 2024
    val_mask = (df.index.year == 2025) & (df.index.month <= 3)
    test_mask = (df.index.year == 2025) & (df.index.month > 3)

    target_col = 'aqi_target_24h'
    train_mask = train_mask & df[target_col].notna()
    val_mask = val_mask & df[target_col].notna()
    test_mask = test_mask & df[target_col].notna()

    X_train = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    y_val = df.loc[val_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def select_features(df: pd.DataFrame, n_features: int = 40) -> List[str]:
    """
    Select top N features using combined approach:
    1. Lasso with lower alpha (less aggressive)
    2. Mutual Information scoring
    3. Tree-based importance from Random Forest

    Takes union of top features from each method for robust selection.
    """
    print(f"\n{'='*40}")
    print(f"Advanced Feature Selection (Top {n_features})")
    print('='*40)

    from sklearn.feature_selection import mutual_info_regression

    # Prepare data
    target_cols = [col for col in df.columns if 'target' in col]
    all_features = [col for col in df.columns
                   if col not in target_cols and col not in ['aqi', 'aqi_category']]

    train_mask = df.index.year == 2024
    target_col = 'aqi_target_24h'
    train_mask = train_mask & df[target_col].notna()

    X = df.loc[train_mask, all_features]
    y = df.loc[train_mask, target_col]

    # Remove any NaN columns
    valid_cols = [col for col in all_features if not X[col].isna().any()]
    X = X[valid_cols]

    print(f"Using {len(valid_cols)} valid features")

    # ============================================
    # Method 1: Lasso with lower alpha (less aggressive)
    # ============================================
    print("\n[1/3] Running Lasso (alpha=0.1)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_scaled, y)

    lasso_importance = pd.DataFrame({
        'feature': valid_cols,
        'lasso_score': np.abs(lasso.coef_)
    }).sort_values('lasso_score', ascending=False)

    lasso_top = set(lasso_importance.head(n_features)['feature'].tolist())
    print(f"  Lasso selected: {len(lasso_top)} features")

    # ============================================
    # Method 2: Mutual Information
    # ============================================
    print("[2/3] Computing Mutual Information...")
    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    mi_importance = pd.DataFrame({
        'feature': valid_cols,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    mi_top = set(mi_importance.head(n_features)['feature'].tolist())
    print(f"  MI selected: {len(mi_top)} features")

    # ============================================
    # Method 3: Random Forest importance
    # ============================================
    print("[3/3] Computing Random Forest importance...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, random_state=42)
    rf.fit(X, y)

    rf_importance = pd.DataFrame({
        'feature': valid_cols,
        'rf_score': rf.feature_importances_
    }).sort_values('rf_score', ascending=False)

    rf_top = set(rf_importance.head(n_features)['feature'].tolist())
    print(f"  RF selected: {len(rf_top)} features")

    # ============================================
    # Union of all methods
    # ============================================
    combined_features = lasso_top | mi_top | rf_top
    print(f"\nCombined (union): {len(combined_features)} features")

    # If too many, prioritize by average rank
    if len(combined_features) > n_features:
        # Compute average rank
        all_ranks = []
        for feat in combined_features:
            lasso_rank = list(lasso_importance['feature']).index(feat) if feat in lasso_importance['feature'].values else len(valid_cols)
            mi_rank = list(mi_importance['feature']).index(feat) if feat in mi_importance['feature'].values else len(valid_cols)
            rf_rank = list(rf_importance['feature']).index(feat) if feat in rf_importance['feature'].values else len(valid_cols)
            avg_rank = (lasso_rank + mi_rank + rf_rank) / 3
            all_ranks.append((feat, avg_rank))

        all_ranks.sort(key=lambda x: x[1])
        top_features = [f for f, _ in all_ranks[:n_features]]
    else:
        top_features = list(combined_features)

    # Print top features
    print(f"\nSelected {len(top_features)} features:")
    for i, f in enumerate(top_features[:15], 1):
        print(f"  {i}. {f}")
    if len(top_features) > 15:
        print(f"  ... and {len(top_features)-15} more")

    return top_features


def train_mlp(X_train, y_train, X_val, y_val):
    """Train MLP Neural Network."""
    print(f"\n{'='*40}")
    print("Training MLP Neural Network")
    print('='*40)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Try different architectures
    best_val_rmse = float('inf')
    best_config = None

    configs = [
        (64, 32),
        (128, 64),
        (128, 64, 32),
        (256, 128, 64),
    ]

    for config in configs:
        mlp = MLPRegressor(
            hidden_layer_sizes=config,
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)

        val_pred = mlp.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_config = config

    # Train final with best config
    mlp = MLPRegressor(
        hidden_layer_sizes=best_config,
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)

    train_pred = mlp.predict(X_train_scaled)
    val_pred = mlp.predict(X_val_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Best config: {best_config}")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return mlp, scaler, 'mlp', best_config


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting."""
    print(f"\n{'='*40}")
    print("Training Gradient Boosting")
    print('='*40)

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model, None, 'gradient_boosting', None


def train_xgboost_tuned(X_train, y_train, X_val, y_val):
    """Train XGBoost with better tuning."""
    if not HAS_XGBOOST:
        return None, None, 'xgboost_tuned', None

    print(f"\n{'='*40}")
    print("Training XGBoost (Tuned)")
    print('='*40)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model, None, 'xgboost_tuned', None


def train_ensemble(models: Dict, X_train, y_train, X_val, y_val):
    """Train ensemble - includes both weighted average and stacking."""
    print(f"\n{'='*40}")
    print("Training Ensemble (Weighted + Stacking)")
    print('='*40)

    # Get predictions from each model
    preds_train = {}
    preds_val = {}

    for name, model_data in models.items():
        model = model_data['model']
        scaler = model_data.get('scaler')

        if scaler is not None:
            X_tr = scaler.transform(X_train)
            X_vl = scaler.transform(X_val)
        else:
            X_tr = X_train
            X_vl = X_val

        preds_train[name] = model.predict(X_tr)
        preds_val[name] = model.predict(X_vl)

    # ============================================
    # Part 1: Weighted Average (existing approach)
    # ============================================
    print("\n--- Weighted Average ---")
    best_rmse = float('inf')
    best_weights = None

    model_names = list(preds_train.keys())

    if len(model_names) >= 2:
        # Simple equal weights first
        weights = {name: 1.0/len(model_names) for name in model_names}
        ensemble_val = sum(weights[name] * preds_val[name] for name in model_names)
        val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
        best_rmse = val_rmse
        best_weights = weights.copy()
        print(f"  Equal weights RMSE: {val_rmse:.2f}")

        # Inverse RMSE weighted
        rmses = {name: np.sqrt(mean_squared_error(y_val, preds_val[name])) for name in model_names}
        total_inv = sum(1/r for r in rmses.values())
        weights = {name: (1/rmses[name])/total_inv for name in model_names}
        ensemble_val = sum(weights[name] * preds_val[name] for name in model_names)
        val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_weights = weights.copy()
        print(f"  Inverse-RMSE weights RMSE: {val_rmse:.2f}")

    # ============================================
    # Part 2: Stacking Ensemble (NEW)
    # ============================================
    print("\n--- Stacking Ensemble ---")

    # Create meta-features from base model predictions
    meta_train = np.column_stack([preds_train[name] for name in model_names])
    meta_val = np.column_stack([preds_val[name] for name in model_names])

    # Train meta-learner (Ridge regression)
    from sklearn.linear_model import Ridge
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_train, y_train)

    # Predict on validation
    stacking_val_pred = meta_learner.predict(meta_val)
    stacking_rmse = np.sqrt(mean_squared_error(y_val, stacking_val_pred))

    print(f"  Stacking RMSE: {stacking_rmse:.2f}")
    print(f"  Meta-learner coefficients: {dict(zip(model_names, meta_learner.coef_.round(3)))}")

    # Choose best approach
    if stacking_rmse < best_rmse:
        print(f"\n  Stacking wins! ({stacking_rmse:.2f} vs {best_rmse:.2f})")
        return {
            'type': 'stacking',
            'meta_learner': meta_learner,
            'base_models': model_names,
            'stacking_rmse': stacking_rmse,
            'weights': best_weights,
            'weighted_rmse': best_rmse
        }, 'ensemble', stacking_rmse
    else:
        print(f"\n  Weighted average wins! ({best_rmse:.2f} vs {stacking_rmse:.2f})")
        return {
            'type': 'weighted',
            'weights': best_weights,
            'base_models': model_names,
            'rmse': best_rmse
        }, 'ensemble', best_rmse


def optimize_all():
    """Run full optimization pipeline."""
    print("="*60)
    print("Advanced Model Optimization")
    print("="*60)

    # Load data
    df = load_featured_data()

    # Feature selection
    selected_features = select_features(df, n_features=30)

    # Prepare data with selected features
    X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_data(df, selected_features)

    models = {}

    # 1. Train Lasso with selected features
    print(f"\n{'='*40}")
    print("Training Lasso (Selected Features)")
    print('='*40)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lasso = Lasso(alpha=0.5, max_iter=5000)
    lasso.fit(X_train_scaled, y_train)

    lasso_val_pred = lasso.predict(X_val_scaled)
    lasso_val_rmse = np.sqrt(mean_squared_error(y_val, lasso_val_pred))
    print(f"  Lasso Val RMSE: {lasso_val_rmse:.2f}")

    models['lasso'] = {'model': lasso, 'scaler': scaler}

    # 2. Train Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled, y_train)
    ridge_val_pred = ridge.predict(X_val_scaled)
    print(f"  Ridge Val RMSE: {np.sqrt(mean_squared_error(y_val, ridge_val_pred)):.2f}")
    models['ridge'] = {'model': ridge, 'scaler': scaler}

    # 3. Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)
    print(f"  RF Val RMSE: {np.sqrt(mean_squared_error(y_val, rf_val_pred)):.2f}")
    models['rf'] = {'model': rf, 'scaler': None}

    # 4. XGBoost Tuned
    if HAS_XGBOOST:
        xgb, _, name, _ = train_xgboost_tuned(X_train, y_train, X_val, y_val)
        if xgb:
            models[name] = {'model': xgb, 'scaler': None}

    # 5. Gradient Boosting
    gb, _, name, _ = train_gradient_boosting(X_train, y_train, X_val, y_val)
    models[name] = {'model': gb, 'scaler': None}

    # 6. MLP
    mlp, scaler_mlp, name, _ = train_mlp(X_train, y_train, X_val, y_val)
    models[name] = {'model': mlp, 'scaler': scaler_mlp}

    # 7. Ensemble
    weights, name, _ = train_ensemble(models, X_train, y_train, X_val, y_val)
    models[name] = {'weights': weights, 'base_models': list(weights.keys())}

    # Save models
    for name, model_data in models.items():
        filepath = os.path.join(MODELS_DIR, f'optimized_{name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    print("\nAll optimized models saved!")

    return models, (X_train, X_val, X_test, y_train, y_val, y_test), selected_features


if __name__ == "__main__":
    models, data, features = optimize_all()
