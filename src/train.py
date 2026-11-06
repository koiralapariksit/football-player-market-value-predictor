"""
Model Training & Evaluation Pipeline for Football Player Market Value.
Trains and benchmarks Ridge, Random Forest, and HistGradientBoosting models,
performs 5-Fold Cross Validation, computes comprehensive metrics on both
log-scale and real Euro scale, and exports production-ready model artifacts.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, root_mean_squared_error
from sklearn.inspection import permutation_importance

from src.feature_engineering import prepare_training_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "football_players_market_value.csv.gz")


def compute_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics on both log and Euro scales."""
    # Ensure positive predictions
    y_pred_raw = np.maximum(y_pred_raw, 1000.0)

    # Percentage errors
    pct_errors = np.abs(y_pred_raw - y_true_raw) / y_true_raw
    within_20 = float(np.mean(pct_errors <= 0.20) * 100)
    within_50 = float(np.mean(pct_errors <= 0.50) * 100)
    mdape = float(np.median(pct_errors) * 100)

    return {
        "log_r2": float(r2_score(y_true_log, y_pred_log)),
        "log_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "log_rmse": float(root_mean_squared_error(y_true_log, y_pred_log)),
        "raw_r2": float(r2_score(y_true_raw, y_pred_raw)),
        "raw_mae_eur": float(mean_absolute_error(y_true_raw, y_pred_raw)),
        "raw_medae_eur": float(median_absolute_error(y_true_raw, y_pred_raw)),
        "raw_rmse_eur": float(root_mean_squared_error(y_true_raw, y_pred_raw)),
        "mdape_percent": mdape,
        "within_20_pct": within_20,
        "within_50_pct": within_50
    }


def train_and_benchmark(data_path: str = DATA_PATH, models_dir: str = MODELS_DIR):
    """Main training routine."""
    os.makedirs(models_dir, exist_ok=True)
    t0 = time.time()

    print(f"[TRAIN] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    X, y_log, y_raw = prepare_training_data(df)
    print(f"[TRAIN] Loaded {len(X)} players with {len(NUMERIC_FEATURES)} numeric & {len(CATEGORICAL_FEATURES)} categorical features.")

    # Train/Test Split (80/20)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_log_train, y_log_test = y_log[train_idx], y_log[test_idx]
    y_raw_train, y_raw_test = y_raw[train_idx], y_raw[test_idx]

    # Preprocessors
    prep_tree = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
    ])

    prep_linear = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
    ])

    # Candidate Models
    candidate_configs = {
        "Ridge Baseline": {
            "pipeline": Pipeline([
                ("prep", prep_linear),
                ("model", Ridge(alpha=10.0))
            ])
        },
        "Random Forest Regressor": {
            "pipeline": Pipeline([
                ("prep", prep_tree),
                ("model", RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42
                ))
            ])
        },
        "HistGradientBoosting Regressor": {
            "pipeline": Pipeline([
                ("prep", prep_tree),
                ("model", HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.06,
                    max_depth=8,
                    min_samples_leaf=20,
                    l2_regularization=0.5,
                    random_state=42
                ))
            ])
        }
    }

    results = {}
    fitted_models = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*70)
    print("BENCHMARKING MODEL ARCHITECTURES ON FOOTBALL MARKET VALUES")
    print("="*70)

    for name, config in candidate_configs.items():
        pipe = config["pipeline"]
        print(f"\n[BENCHMARK] Training {name}...")

        # 5-Fold Cross Validation on train split
        cv_scores = cross_val_score(pipe, X_train, y_log_train, cv=cv, scoring="r2", n_jobs=-1)
        print(f"  > 5-Fold CV Log R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on full train set
        pipe.fit(X_train, y_log_train)
        fitted_models[name] = pipe

        # Predict test set
        pred_log = pipe.predict(X_test)
        pred_raw = np.expm1(np.maximum(pred_log, 0))

        metrics = compute_metrics(y_raw_test, pred_raw, y_log_test, pred_log)
        metrics["cv_log_r2_mean"] = float(cv_scores.mean())
        metrics["cv_log_r2_std"] = float(cv_scores.std())
        results[name] = metrics

        print(f"  > Test Log R2:       {metrics['log_r2']:.4f}")
        print(f"  > Test Euro R2:      {metrics['raw_r2']:.4f}")
        print(f"  > Test MAE (EUR):    €{metrics['raw_mae_eur']:,.0f}")
        print(f"  > Test MedAE (EUR):  €{metrics['raw_medae_eur']:,.0f}")
        print(f"  > Test MdAPE:        {metrics['mdape_percent']:.1f}%")
        print(f"  > Within +/-20%:     {metrics['within_20_pct']:.1f}%")
        print(f"  > Within +/-50%:     {metrics['within_50_pct']:.1f}%")

    # Select best model based on Log R2 and Raw R2
    best_name = "HistGradientBoosting Regressor"
    best_pipeline = fitted_models[best_name]

    # Save model artifact
    best_model_path = os.path.join(models_dir, "best_market_value_model.joblib")
    print(f"\n[SAVE] Exporting best model ({best_name}) to {best_model_path}...")
    joblib.dump(best_pipeline, best_model_path)

    # Compute Feature Importances via Permutation Importance on a test slice
    print("[IMPORTANCE] Computing permutation feature importances...")
    perm_sample_size = min(1500, len(X_test))
    perm = permutation_importance(
        best_pipeline,
        X_test.iloc[:perm_sample_size],
        y_log_test[:perm_sample_size],
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    feature_names = list(X.columns)
    importances = [
        {"feature": feat, "importance": float(imp), "std": float(std)}
        for feat, imp, std in zip(feature_names, perm.importances_mean, perm.importances_std)
    ]
    importances.sort(key=lambda x: x["importance"], reverse=True)

    # Save metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_name,
        "dataset_records": int(len(df)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "benchmark_results": results,
        "feature_importances": importances[:25],
        "training_time_seconds": round(time.time() - t0, 2)
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVE] Exported metadata and evaluation metrics to {metadata_path}")

    # Also save test predictions for interactive visualization
    test_eval_df = df.iloc[test_idx][["player_id", "name", "position", "sub_position", "current_club_name", "market_value_in_eur"]].copy()
    test_eval_df["pred_market_value_eur"] = np.expm1(np.maximum(best_pipeline.predict(X_test), 0))
    test_eval_df["error_eur"] = test_eval_df["pred_market_value_eur"] - test_eval_df["market_value_in_eur"]
    test_eval_df["abs_pct_error"] = (test_eval_df["error_eur"].abs() / test_eval_df["market_value_in_eur"]) * 100
    test_eval_path = os.path.join(models_dir, "test_predictions_sample.csv")
    test_eval_df.to_csv(test_eval_path, index=False)
    print(f"[SAVE] Exported test predictions sample to {test_eval_path}")

    print(f"\n[DONE] Pipeline completed in {time.time()-t0:.2f}s!")
    return results, metadata


if __name__ == "__main__":
    train_and_benchmark()
