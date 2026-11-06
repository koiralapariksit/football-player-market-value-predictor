"""
Stacked Ensemble Model for Football Player Market Value Prediction.

Architecture:
  Level-0: HistGradientBoosting, LightGBM, XGBoost, RandomForest
  Level-1: Ridge meta-learner on out-of-fold predictions
"""

import sys
import os
import json
import time
import warnings

sys.path.insert(0, r"d:\ml")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, root_mean_squared_error

import lightgbm as lgb
from xgboost import XGBRegressor

from src.feature_engineering import prepare_training_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor():
    """Build ColumnTransformer for numeric passthrough + categorical one-hot."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_base_learners():
    """Return list of (name, model) tuples for Level-0."""
    return [
        (
            "HistGBR",
            HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.04,
                max_depth=8,
                min_samples_leaf=15,
                l2_regularization=0.3,
                random_state=42,
            ),
        ),
        (
            "LightGBM",
            lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.04,
                max_depth=8,
                num_leaves=63,
                min_child_samples=20,
                reg_lambda=0.5,
                reg_alpha=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1,
            ),
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.04,
                max_depth=8,
                min_child_weight=15,
                reg_lambda=1.0,
                reg_alpha=0.1,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=0,
            ),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]


def compute_metrics(y_log_true, y_log_pred, y_raw_true):
    """Compute comprehensive evaluation metrics."""
    y_raw_pred = np.expm1(y_log_pred)
    y_raw_pred = np.maximum(y_raw_pred, 0)

    # Log-space metrics
    log_r2 = r2_score(y_log_true, y_log_pred)
    log_mae = mean_absolute_error(y_log_true, y_log_pred)
    log_rmse = root_mean_squared_error(y_log_true, y_log_pred)

    # Raw EUR metrics
    raw_r2 = r2_score(y_raw_true, y_raw_pred)
    raw_mae = mean_absolute_error(y_raw_true, y_raw_pred)
    raw_medae = median_absolute_error(y_raw_true, y_raw_pred)

    # Percentage-based metrics
    safe_raw = np.maximum(y_raw_true, 1.0)
    abs_pct_error = np.abs(y_raw_pred - y_raw_true) / safe_raw
    mdape = float(np.median(abs_pct_error) * 100)
    within_20 = float(np.mean(abs_pct_error <= 0.20) * 100)
    within_50 = float(np.mean(abs_pct_error <= 0.50) * 100)

    return {
        "log_r2": float(log_r2),
        "log_mae": float(log_mae),
        "log_rmse": float(log_rmse),
        "raw_r2": float(raw_r2),
        "raw_mae": float(raw_mae),
        "raw_medae": float(raw_medae),
        "mdape_pct": mdape,
        "within_20_pct": within_20,
        "within_50_pct": within_50,
    }


def print_metrics(name, metrics):
    """Pretty-print metrics for a model."""
    print(f"\n  {'='*55}")
    print(f"  {name}")
    print(f"  {'='*55}")
    print(f"  Log-space:  R²={metrics['log_r2']:.4f}  MAE={metrics['log_mae']:.4f}  RMSE={metrics['log_rmse']:.4f}")
    print(f"  Raw EUR:    R²={metrics['raw_r2']:.4f}  MAE=€{metrics['raw_mae']:,.0f}  MedAE=€{metrics['raw_medae']:,.0f}")
    print(f"  MdAPE={metrics['mdape_pct']:.1f}%  Within±20%={metrics['within_20_pct']:.1f}%  Within±50%={metrics['within_50_pct']:.1f}%")


def main():
    start_time = time.time()
    print("=" * 65)
    print(" STACKED ENSEMBLE - Football Player Market Value Prediction")
    print("=" * 65)

    # ---- Load data ----
    print("\n[1/6] Loading data...")
    df = pd.read_csv(r"d:\ml\data\processed\football_players_market_value.csv.gz")
    X, y_log, y_raw = prepare_training_data(df)
    print(f"  Dataset: {X.shape[0]} players, {X.shape[1]} features")

    # ---- Train/test split ----
    print("\n[2/6] Splitting data (80/20, random_state=42)...")
    X_train, X_test, y_log_train, y_log_test, y_raw_train, y_raw_test = train_test_split(
        X, y_log, y_raw, test_size=0.20, random_state=42
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # ---- Preprocessing ----
    print("\n[3/6] Fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"  Processed feature dim: {X_train_proc.shape[1]}")

    # ---- Generate OOF predictions (5-fold CV) ----
    print("\n[4/6] Generating out-of-fold predictions (5-fold CV)...")
    base_learners = build_base_learners()
    n_models = len(base_learners)
    n_train = X_train_proc.shape[0]
    n_test = X_test_proc.shape[0]

    oof_preds = np.zeros((n_train, n_models))
    test_preds = np.zeros((n_test, n_models))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    individual_metrics = {}

    for i, (name, model) in enumerate(base_learners):
        print(f"\n  --- {name} ---")
        fold_test_preds = np.zeros((n_test, 5))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_proc)):
            X_fold_train = X_train_proc[train_idx]
            y_fold_train = y_log_train[train_idx]
            X_fold_val = X_train_proc[val_idx]

            fold_model = _clone_model(base_learners[i][1])
            fold_model.fit(X_fold_train, y_fold_train)

            oof_preds[val_idx, i] = fold_model.predict(X_fold_val)
            fold_test_preds[:, fold_idx] = fold_model.predict(X_test_proc)

            print(f"    Fold {fold_idx + 1}/5 done")

        # Average test predictions across folds (for diagnostics)
        test_preds[:, i] = fold_test_preds.mean(axis=1)

        # Compute OOF metrics
        oof_metrics = compute_metrics(y_log_train, oof_preds[:, i], y_raw_train)
        print(f"    OOF Log R²: {oof_metrics['log_r2']:.4f}")

    # ---- Fit base learners on FULL training data ----
    print("\n[5/6] Fitting base learners on full training set...")
    fitted_base_learners = []
    for i, (name, model) in enumerate(base_learners):
        print(f"  Fitting {name}...")
        model.fit(X_train_proc, y_log_train)
        fitted_base_learners.append(model)

        # Generate test predictions from full-train model
        test_preds[:, i] = model.predict(X_test_proc)

        # Individual model metrics on test set
        metrics = compute_metrics(y_log_test, test_preds[:, i], y_raw_test)
        individual_metrics[name] = metrics
        print_metrics(f"{name} (Individual)", metrics)

    # ---- Train meta-learner ----
    print("\n[6/6] Training Ridge meta-learner...")
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(oof_preds, y_log_train)

    # Meta-learner weights
    print(f"\n  Meta-learner coefficients:")
    for i, (name, _) in enumerate(base_learners):
        print(f"    {name}: {meta_learner.coef_[i]:.4f}")
    print(f"    Intercept: {meta_learner.intercept_:.4f}")

    # ---- Final ensemble prediction ----
    ensemble_test_preds = meta_learner.predict(test_preds)
    ensemble_metrics = compute_metrics(y_log_test, ensemble_test_preds, y_raw_test)

    print("\n" + "=" * 65)
    print(" FINAL RESULTS")
    print("=" * 65)

    for name, metrics in individual_metrics.items():
        print_metrics(name, metrics)

    print_metrics("STACKED ENSEMBLE", ensemble_metrics)

    # ---- Comparison with baseline ----
    print(f"\n  {'='*55}")
    print(f"  IMPROVEMENT OVER BASELINE")
    print(f"  {'='*55}")
    baseline_log_r2 = 0.7276
    baseline_raw_r2 = 0.8004
    print(f"  Baseline HistGBR:     Log R²={baseline_log_r2:.4f}  Raw R²={baseline_raw_r2:.4f}")
    print(f"  Stacked Ensemble:     Log R²={ensemble_metrics['log_r2']:.4f}  Raw R²={ensemble_metrics['raw_r2']:.4f}")
    print(f"  Δ Log R²: {ensemble_metrics['log_r2'] - baseline_log_r2:+.4f}")
    print(f"  Δ Raw R²: {ensemble_metrics['raw_r2'] - baseline_raw_r2:+.4f}")

    # ---- Save artifacts ----
    models_dir = r"d:\ml\models"
    os.makedirs(models_dir, exist_ok=True)

    # Save model bundle
    model_bundle = {
        "preprocessor": preprocessor,
        "base_learners": fitted_base_learners,
        "meta_learner": meta_learner,
        "base_learner_names": [name for name, _ in base_learners],
    }
    model_path = os.path.join(models_dir, "ensemble_model.joblib")
    joblib.dump(model_bundle, model_path, compress=3)
    print(f"\n  Model saved: {model_path}")

    # Feature importances (from tree-based models that support it)
    feature_importances = {}
    # Get feature names after preprocessing
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names

    for i, (name, _) in enumerate(base_learners):
        model = fitted_base_learners[i]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            if len(imp) == len(all_feature_names):
                sorted_idx = np.argsort(imp)[::-1][:15]
                feature_importances[name] = {
                    all_feature_names[j]: float(imp[j]) for j in sorted_idx
                }

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": int(X.shape[0]),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_processed": int(X_train_proc.shape[1]),
        "base_learner_names": [name for name, _ in base_learners],
        "meta_learner_coefficients": {
            name: float(meta_learner.coef_[i]) for i, (name, _) in enumerate(base_learners)
        },
        "meta_learner_intercept": float(meta_learner.intercept_),
        "individual_metrics": individual_metrics,
        "ensemble_metrics": ensemble_metrics,
        "feature_importances": feature_importances,
        "baseline_comparison": {
            "baseline_log_r2": baseline_log_r2,
            "baseline_raw_r2": baseline_raw_r2,
            "ensemble_log_r2": ensemble_metrics["log_r2"],
            "ensemble_raw_r2": ensemble_metrics["raw_r2"],
            "delta_log_r2": ensemble_metrics["log_r2"] - baseline_log_r2,
            "delta_raw_r2": ensemble_metrics["raw_r2"] - baseline_raw_r2,
        },
        "training_time_seconds": round(time.time() - start_time, 1),
    }
    metadata_path = os.path.join(models_dir, "ensemble_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")

    elapsed = time.time() - start_time
    print(f"\n  Total training time: {elapsed:.1f}s")
    print("=" * 65)


def _clone_model(model):
    """Create a fresh clone of a scikit-learn-compatible estimator."""
    from sklearn.base import clone
    return clone(model)


if __name__ == "__main__":
    main()
