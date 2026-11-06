"""
Super Ensemble Model Training Pipeline for Football Player Market Value.
Combines LightGBM, XGBoost, and HistGradientBoosting with advanced domain
features: club prestige indexing, wonderkid interactions, and position-specific
production rates.
"""

import os
import sys
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, root_mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

from src.ensemble_pipeline import FootballEnsemblePipeline
from src.feature_engineering import (
    prepare_training_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_and_save_club_prestige_map
)


MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "football_players_market_value.csv.gz")

def train_super_ensemble():
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    print("[TRAIN] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    valid = df[df["market_value_in_eur"].notnull() & (df["market_value_in_eur"] > 0)].copy()

    # Build and cache prestige map
    prestige_map = build_and_save_club_prestige_map(valid)
    print(f"[TRAIN] Built club prestige map ({len(prestige_map)} entities mapped)")

    X, y_log, y_raw = prepare_training_data(valid)
    print(f"[TRAIN] Features shape: {X.shape} (Numeric: {len(NUMERIC_FEATURES)}, Categorical: {len(CATEGORICAL_FEATURES)})")

    indices = np.arange(len(valid))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_log_train, y_log_test = y_log[train_idx], y_log[test_idx]
    y_raw_train, y_raw_test = y_raw[train_idx], y_raw[test_idx]

    num_cols = [c for c in X.columns if c not in CATEGORICAL_FEATURES]
    cat_cols = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    m_lgb = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=50,
        max_depth=8,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )

    m_xgb = xgb.XGBRegressor(
        n_estimators=450,
        learning_rate=0.035,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )

    m_hgb = HistGradientBoostingRegressor(
        max_iter=450,
        learning_rate=0.04,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=42
    )

    print("\n[TRAIN] Training Super-Ensemble (LightGBM + XGBoost + HistGradientBoosting)...")
    ensemble = FootballEnsemblePipeline(
        preprocessor=preprocessor,
        m_lgb=m_lgb,
        m_xgb=m_xgb,
        m_hgb=m_hgb
    )

    ensemble.fit(X_train, y_log_train)

    # Evaluate on held-out test split
    print("[EVAL] Evaluating on held-out test set (5,237 players)...")
    pred_log = ensemble.predict(X_test)
    pred_raw = np.expm1(np.maximum(pred_log, 0))

    pct_errors = np.abs(pred_raw - y_raw_test) / y_raw_test
    metrics = {
        "log_r2": float(r2_score(y_log_test, pred_log)),
        "log_mae": float(mean_absolute_error(y_log_test, pred_log)),
        "log_rmse": float(root_mean_squared_error(y_log_test, pred_log)),
        "raw_r2": float(r2_score(y_raw_test, pred_raw)),
        "raw_mae_eur": float(mean_absolute_error(y_raw_test, pred_raw)),
        "raw_medae_eur": float(median_absolute_error(y_raw_test, pred_raw)),
        "mdape_percent": float(np.median(pct_errors) * 100),
        "within_20_pct": float(np.mean(pct_errors <= 0.20) * 100),
        "within_50_pct": float(np.mean(pct_errors <= 0.50) * 100)
    }

    print("=" * 60)
    print("SUPER-ENSEMBLE EVALUATION RESULTS:")
    print("=" * 60)
    print(f"Log R2:           {metrics['log_r2']:.4f}")
    print(f"Raw Euro R2:      {metrics['raw_r2']:.4f}")
    print(f"MAE (EUR):        €{metrics['raw_mae_eur']:,.0f}")
    print(f"MedAE (EUR):      €{metrics['raw_medae_eur']:,.0f}")
    print(f"MdAPE:            {metrics['mdape_percent']:.1f}%")
    print(f"Within +/-20%:    {metrics['within_20_pct']:.1f}%")
    print(f"Within +/-50%:    {metrics['within_50_pct']:.1f}%")

    # Save to primary model path
    best_model_path = os.path.join(MODELS_DIR, "best_market_value_model.joblib")
    super_model_path = os.path.join(MODELS_DIR, "super_ensemble_model.joblib")
    print(f"\n[SAVE] Exporting model to {best_model_path}...")
    joblib.dump(ensemble, best_model_path)
    joblib.dump(ensemble, super_model_path)

    # Save metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "Super Ensemble (LightGBM 40% + XGBoost 35% + HistGradientBoosting 25%)",
        "dataset_records": int(len(valid)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "metrics": metrics,
        "training_time_seconds": round(time.time() - t0, 2)
    }

    meta_path = os.path.join(MODELS_DIR, "super_ensemble_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Export test predictions sample
    test_df = valid.iloc[test_idx][["player_id", "name", "position", "sub_position", "current_club_name", "market_value_in_eur", "image_url"]].copy()
    test_df["pred_market_value_eur"] = pred_raw
    test_df["error_eur"] = pred_raw - y_raw_test
    test_df["abs_pct_error"] = pct_errors * 100
    test_df.to_csv(os.path.join(MODELS_DIR, "test_predictions_sample.csv"), index=False)

    print(f"[DONE] Completed in {time.time()-t0:.2f}s!")
    return ensemble, metrics


if __name__ == "__main__":
    train_super_ensemble()
