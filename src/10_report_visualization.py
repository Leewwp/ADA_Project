"""
Report Visualization Script
===========================
Generates report-ready model-effect figures for the 24-hour AQI task.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURED_DATA_FILE = os.path.join(DATA_DIR, "featured_aqi_data.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")
PREDICTIONS_FILE = os.path.join(REPORT_DIR, "model_predictions_test.csv")

SELECTED_FEATURES = [
    "pm2_5",
    "month",
    "surface_pressure",
    "pm2_5_lag_48h",
    "nitrogen_dioxide",
    "is_afternoon",
    "pm25_pm10_ratio",
    "heat_index",
    "temperature_2m_rolling_mean_12h",
    "pm10_rolling_range_168h",
    "temperature_2m_rolling_std_168h",
    "nitrogen_dioxide_rolling_range_12h",
    "pm2_5_lag_168h",
    "pm10_rolling_range_48h",
    "pm10_rolling_range_72h",
    "pm10",
    "ozone",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_lag_72h",
    "aqi_lag_168h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_12h",
    "aqi_rolling_mean_24h",
    "aqi_rolling_mean_48h",
    "aqi_rolling_std_6h",
    "aqi_rolling_std_12h",
    "aqi_rolling_std_24h",
    "aqi_rolling_std_48h",
    "aqi_momentum_1h",
    "aqi_momentum_6h",
    "aqi_momentum_24h",
    "aqi_change_6h",
    "aqi_change_12h",
    "aqi_change_24h",
    "aqi_change_48h",
]


@dataclass
class ModelArtifact:
    name: str
    model: object
    scaler: Optional[StandardScaler]
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    val_rmse: float
    metrics: Dict[str, float]


class PredictorWrapper:
    """Adapter that accepts a DataFrame and applies scaling when needed."""

    def __init__(self, model: object, scaler: Optional[StandardScaler] = None):
        self.model = model
        self.scaler = scaler

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_input = self.scaler.transform(X) if self.scaler is not None else X
        return np.asarray(self.model.predict(X_input), dtype=float)


class WeightedEnsemblePredictor:
    """Inverse-validation-RMSE weighted ensemble."""

    def __init__(self, predictors: Dict[str, PredictorWrapper], weights: Dict[str, float]):
        self.predictors = predictors
        self.weights = weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        combined = np.zeros(len(X), dtype=float)
        for name, weight in self.weights.items():
            combined += weight * self.predictors[name].predict(X)
        return combined


def ensure_output_dirs() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_selected_features() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURED_DATA_FILE, index_col=0, parse_dates=True)
    feature_cols = [feature for feature in SELECTED_FEATURES if feature in df.columns]
    return df, feature_cols


def prepare_splits(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, ...]:
    target_col = "aqi_target_24h"

    train_mask = (df.index.year == 2024) & df[target_col].notna()
    val_mask = (df.index.year == 2025) & (df.index.month <= 3) & df[target_col].notna()
    test_mask = (df.index.year == 2025) & (df.index.month > 3) & df[target_col].notna()

    X_train = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    y_val = df.loc[val_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = y_pred - y_true.to_numpy()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    direction_accuracy = np.mean(
        np.sign(np.diff(y_true.to_numpy())) == np.sign(np.diff(y_pred))
    ) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Direction Accuracy": float(direction_accuracy),
        "Residual Mean": float(np.mean(residuals)),
        "Residual Std": float(np.std(residuals)),
    }


def train_linear_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, ModelArtifact]:
    models: Dict[str, ModelArtifact] = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    for name, model in {
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
        "Ridge": Ridge(alpha=1.0),
    }.items():
        model.fit(X_train_scaled, y_train)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        models[name] = ModelArtifact(
            name=name,
            model=model,
            scaler=scaler,
            val_predictions=np.asarray(val_pred, dtype=float),
            test_predictions=np.asarray(test_pred, dtype=float),
            val_rmse=float(np.sqrt(mean_squared_error(y_val, val_pred))),
            metrics=calculate_metrics(y_test, test_pred),
        )

    return models


def train_tree_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, ModelArtifact]:
    models: Dict[str, ModelArtifact] = {}

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_val = rf.predict(X_val)
    rf_test = rf.predict(X_test)
    models["Random Forest"] = ModelArtifact(
        name="Random Forest",
        model=rf,
        scaler=None,
        val_predictions=np.asarray(rf_val, dtype=float),
        test_predictions=np.asarray(rf_test, dtype=float),
        val_rmse=float(np.sqrt(mean_squared_error(y_val, rf_val))),
        metrics=calculate_metrics(y_test, rf_test),
    )

    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_val = gb.predict(X_val)
    gb_test = gb.predict(X_test)
    models["Gradient Boosting"] = ModelArtifact(
        name="Gradient Boosting",
        model=gb,
        scaler=None,
        val_predictions=np.asarray(gb_val, dtype=float),
        test_predictions=np.asarray(gb_test, dtype=float),
        val_rmse=float(np.sqrt(mean_squared_error(y_val, gb_val))),
        metrics=calculate_metrics(y_test, gb_test),
    )

    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_val = xgb.predict(X_val)
        xgb_test = xgb.predict(X_test)
        models["XGBoost"] = ModelArtifact(
            name="XGBoost",
            model=xgb,
            scaler=None,
            val_predictions=np.asarray(xgb_val, dtype=float),
            test_predictions=np.asarray(xgb_test, dtype=float),
            val_rmse=float(np.sqrt(mean_squared_error(y_val, xgb_val))),
            metrics=calculate_metrics(y_test, xgb_test),
        )
    except Exception:
        pass

    try:
        import lightgbm as lgb

        lgbm = lgb.LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        lgbm_val = lgbm.predict(X_val)
        lgbm_test = lgbm.predict(X_test)
        models["LightGBM"] = ModelArtifact(
            name="LightGBM",
            model=lgbm,
            scaler=None,
            val_predictions=np.asarray(lgbm_val, dtype=float),
            test_predictions=np.asarray(lgbm_test, dtype=float),
            val_rmse=float(np.sqrt(mean_squared_error(y_val, lgbm_val))),
            metrics=calculate_metrics(y_test, lgbm_test),
        )
    except Exception:
        pass

    return models


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, ModelArtifact]:
    models = {}
    models.update(train_linear_models(X_train, y_train, X_val, y_val, X_test, y_test))
    models.update(train_tree_models(X_train, y_train, X_val, y_val, X_test, y_test))
    return models


def build_weighted_ensemble(
    models: Dict[str, ModelArtifact],
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple[WeightedEnsemblePredictor, Dict[str, float], np.ndarray, Dict[str, float]]:
    inverse_total = sum(1.0 / artifact.val_rmse for artifact in models.values())
    weights = {
        name: (1.0 / artifact.val_rmse) / inverse_total
        for name, artifact in models.items()
    }

    predictors = {
        name: PredictorWrapper(artifact.model, artifact.scaler)
        for name, artifact in models.items()
    }
    ensemble = WeightedEnsemblePredictor(predictors, weights)

    val_pred = np.zeros(len(y_val), dtype=float)
    test_pred = np.zeros(len(y_test), dtype=float)
    for name, artifact in models.items():
        val_pred += weights[name] * artifact.val_predictions
        test_pred += weights[name] * artifact.test_predictions

    metrics = calculate_metrics(y_test, test_pred)
    return ensemble, weights, test_pred, metrics


def create_predictions_frame(
    test_index: pd.Index,
    y_test: pd.Series,
    models: Dict[str, ModelArtifact],
    ensemble_pred: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=test_index)
    frame.index.name = "datetime"
    frame["actual_aqi"] = y_test.to_numpy()

    for name, artifact in models.items():
        safe_name = name.lower().replace(" ", "_")
        frame[f"pred_{safe_name}"] = artifact.test_predictions

    frame["pred_ensemble_weighted"] = ensemble_pred
    frame["residual_ensemble_weighted"] = frame["pred_ensemble_weighted"] - frame["actual_aqi"]
    return frame.reset_index()


def plot_prediction_overview(
    test_index: pd.Index,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    metrics: Dict[str, float],
) -> str:
    recent_hours = min(24 * 14, len(test_index))
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharey=False)

    axes[0].plot(test_index, y_test, label="Actual AQI", linewidth=1.4, color="#1f3b73")
    axes[0].plot(test_index, y_pred, label="Predicted AQI", linewidth=1.2, color="#d95f02", alpha=0.85)
    axes[0].set_title(f"{model_name}: prediction over the full test period")
    axes[0].set_ylabel("AQI proxy")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].text(
        0.01,
        0.97,
        f"RMSE={metrics['RMSE']:.2f}  MAE={metrics['MAE']:.2f}  R2={metrics['R2']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    recent_index = test_index[-recent_hours:]
    axes[1].plot(recent_index, y_test.iloc[-recent_hours:], linewidth=1.8, color="#1f3b73")
    axes[1].plot(recent_index, y_pred[-recent_hours:], linewidth=1.6, color="#d95f02", alpha=0.9)
    axes[1].set_title("Recent 14-day zoom")
    axes[1].set_ylabel("AQI proxy")
    axes[1].set_xlabel("Datetime")
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=max(1, recent_hours // (24 * 7))))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "best_model_prediction_overview.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_actual_vs_predicted(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lower = float(min(y_test.min(), y_pred.min()))
    upper = float(max(y_test.max(), y_pred.max()))

    axes[0].scatter(y_test, y_pred, s=14, alpha=0.28, color="#2a9d8f", edgecolors="none")
    axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="#444444", linewidth=1.2)
    axes[0].set_title(f"{model_name}: actual vs predicted")
    axes[0].set_xlabel("Actual AQI proxy")
    axes[0].set_ylabel("Predicted AQI proxy")
    axes[0].grid(True, alpha=0.2)

    bins = np.linspace(lower, upper, 35)
    axes[1].hist(y_test, bins=bins, alpha=0.65, color="#1f77b4", label="Actual")
    axes[1].hist(y_pred, bins=bins, alpha=0.55, color="#ff7f0e", label="Predicted")
    axes[1].set_title("Distribution alignment")
    axes[1].set_xlabel("AQI proxy")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "best_model_actual_vs_predicted.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_residual_diagnostics(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> str:
    residuals = y_pred - y_test.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(residuals, bins=40, kde=True, ax=axes[0], color="#6a4c93")
    axes[0].axvline(0, color="#444444", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"{model_name}: residual distribution")
    axes[0].set_xlabel("Prediction - actual")
    axes[0].set_ylabel("Count")

    axes[1].scatter(y_test, residuals, s=14, alpha=0.28, color="#bc4749", edgecolors="none")
    axes[1].axhline(0, color="#444444", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residuals vs actual AQI")
    axes[1].set_xlabel("Actual AQI proxy")
    axes[1].set_ylabel("Prediction - actual")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "best_model_residual_diagnostics.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_permutation_importance(
    predictor: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    baseline_rmse = np.sqrt(mean_squared_error(y_test, predictor.predict(X_test)))
    rng = np.random.default_rng(random_state)
    importance_rows = []

    for column in X_test.columns:
        deltas = []
        for _ in range(repeats):
            shuffled = X_test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            shuffled_rmse = np.sqrt(mean_squared_error(y_test, predictor.predict(shuffled)))
            deltas.append(float(shuffled_rmse - baseline_rmse))

        importance_rows.append(
            {
                "feature": column,
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
            }
        )

    importance = pd.DataFrame(importance_rows)
    return importance.sort_values("importance_mean", ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
) -> str:
    top_features = importance_df.head(15).sort_values("importance_mean")

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(top_features["feature"], top_features["importance_mean"], xerr=top_features["importance_std"], color="#4c956c")
    ax.set_title(f"{model_name}: top 15 permutation importances")
    ax.set_xlabel("RMSE increase after shuffling feature")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "best_model_feature_importance.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ensemble_weights(weights: Dict[str, float]) -> str:
    weights_df = (
        pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())})
        .sort_values("weight", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=weights_df, x="model", y="weight", hue="model", palette="crest", legend=False, ax=ax)
    ax.set_title("Weighted ensemble contribution by base model")
    ax.set_xlabel("Base model")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, max(weights_df["weight"]) * 1.2)
    ax.grid(True, axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "ensemble_weights.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")

    print("=" * 60)
    print("Report Visualization Pipeline")
    print("=" * 60)

    df, feature_cols = load_selected_features()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(df, feature_cols)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features used: {len(feature_cols)}")

    models = train_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    ensemble, weights, ensemble_pred, ensemble_metrics = build_weighted_ensemble(models, y_val, y_test)

    print("\nModel RMSE on test set:")
    for name, artifact in sorted(models.items(), key=lambda item: item[1].metrics["RMSE"]):
        print(f"  {name}: {artifact.metrics['RMSE']:.2f}")
    print(f"  Ensemble (Weighted): {ensemble_metrics['RMSE']:.2f}")

    best_model_name = "Ensemble (Weighted)"
    best_pred = ensemble_pred
    best_predictor = ensemble
    best_metrics = ensemble_metrics

    best_single = min(models.values(), key=lambda artifact: artifact.metrics["RMSE"])
    if best_single.metrics["RMSE"] < ensemble_metrics["RMSE"]:
        best_model_name = best_single.name
        best_pred = best_single.test_predictions
        best_predictor = PredictorWrapper(best_single.model, best_single.scaler)
        best_metrics = best_single.metrics

    predictions = create_predictions_frame(X_test.index, y_test, models, ensemble_pred)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    figure_paths = [
        plot_prediction_overview(X_test.index, y_test, best_pred, best_model_name, best_metrics),
        plot_actual_vs_predicted(y_test, best_pred, best_model_name),
        plot_residual_diagnostics(y_test, best_pred, best_model_name),
        plot_ensemble_weights(weights),
    ]

    importance_df = compute_permutation_importance(best_predictor, X_test, y_test)
    importance_path = os.path.join(REPORT_DIR, "best_model_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    figure_paths.append(plot_feature_importance(importance_df, best_model_name))

    print(f"\nBest model for plots: {best_model_name}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Feature importance saved to: {importance_path}")
    print("\nGenerated figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
