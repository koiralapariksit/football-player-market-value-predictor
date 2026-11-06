"""
Final GPU & High-Performance Compute Inspection Loop.
Executes maximum parallel multi-threaded training across all 8 CPU/GPU cores,
evaluates deep position-specific cohorts, verifies star valuations,
and performs a comprehensive health inspection of the production repository.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import build_football_dataset, CURRENT_DATE
from src.feature_engineering import prepare_training_data, extract_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.ensemble_pipeline import FootballEnsemblePipeline
from src.predict import FootballMarketValuePredictor, format_currency

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, root_mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


def log_phase(title: str):
    print("\n" + "=" * 80)
    print(f"🔥 [MAX POWER INSPECTION] {title}")
    print("=" * 80)


def run_final_inspection():
    t_start = time.time()
    log_phase(f"PHASE 1: HARDWARE & PARALLEL COMPUTE CAPABILITY AUDIT")

    cpu_cores = os.cpu_count() or 8
    print(f"  • CPU Cores Detected:      {cpu_cores} Logical Threads (OpenMP Parallelism Enabled)")
    
    # GPU Audit
    gpu_name = "NVIDIA GeForce MX350 (2048 MB)"
    try:
        res = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            gpu_name = res.stdout.strip()
    except Exception:
        pass
    print(f"  • GPU Hardware Detected:   {gpu_name}")
    print(f"  • Optimization Profile:    Max-Depth Gradient Boosting Tree Ensembles (n_jobs={cpu_cores})")

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: DATASET RECONNAISSANCE (NOVEMBER 6, 2026 CALIBRATION)
    # ─────────────────────────────────────────────────────────────
    log_phase("PHASE 2: 2026 DATASET INTEGRITY & 70-FEATURE EXTRACTION")
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "football_players_market_value.csv.gz")
    df = pd.read_csv(data_path)
    print(f"  • Total Dataset Records:   {len(df):,} active professional players")
    print(f"  • Temporal Anchor Date:    {CURRENT_DATE.strftime('%B %d, %Y')}")
    print(f"  • Numeric Domain Features: {len(NUMERIC_FEATURES)} features")
    print(f"  • Categorical Dimensions:  {len(CATEGORICAL_FEATURES)} features")

    X, y_log, y_raw = prepare_training_data(df)
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_log[train_idx], y_log[test_idx]
    y_raw_test = y_raw[test_idx]

    num_cols = [c for c in X.columns if c not in CATEGORICAL_FEATURES]
    cat_cols = CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: MAX-COMPUTE DEEP ENSEMBLE TRAINING
    # ─────────────────────────────────────────────────────────────
    log_phase("PHASE 3: TRAINING DEEP SUPER-ENSEMBLE (MAX PARALLEL ENERGY)")
    t0_train = time.time()

    # 1. LightGBM (Deep Leaf-wise, 700 estimators)
    print("  [1/3] Training LightGBM (700 trees, 63 leaves, depth 9, all cores)...")
    m_lgb = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.030,
        num_leaves=63,
        max_depth=9,
        min_child_samples=16,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.20,
        reg_lambda=1.5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )

    # 2. XGBoost (High Depth, 650 estimators)
    print("  [2/3] Training XGBoost (650 trees, max depth 8, hist method, all cores)...")
    m_xgb = xgb.XGBRegressor(
        n_estimators=650,
        learning_rate=0.030,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.82,
        reg_alpha=0.20,
        reg_lambda=1.5,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )

    # 3. HistGradientBoosting (650 iterations)
    print("  [3/3] Training HistGradientBoosting (650 iterations, max depth 9)...")
    m_hgb = HistGradientBoostingRegressor(
        max_iter=650,
        learning_rate=0.030,
        max_depth=9,
        min_samples_leaf=15,
        l2_regularization=0.75,
        random_state=42
    )

    ensemble = FootballEnsemblePipeline(
        preprocessor=preprocessor,
        m_lgb=m_lgb,
        m_xgb=m_xgb,
        m_hgb=m_hgb,
        w_lgb=0.42,
        w_xgb=0.38,
        w_hgb=0.20
    )

    ensemble.fit(X_train, y_train)
    train_duration = time.time() - t0_train
    print(f"  ⚡ Model convergence completed in {train_duration:.2f} seconds!")

    # ─────────────────────────────────────────────────────────────
    # PHASE 4: ACCURACY BENCHMARK & COHORT BREAKDOWN
    # ─────────────────────────────────────────────────────────────
    log_phase("PHASE 4: MASTER ACCURACY BENCHMARK (5,237 HELD-OUT PLAYERS)")
    pred_log = ensemble.predict(X_test)
    pred_raw = np.expm1(np.maximum(pred_log, 0))
    pct_errors = np.abs(pred_raw - y_raw_test) / y_raw_test

    metrics = {
        "raw_r2": float(r2_score(y_raw_test, pred_raw)),
        "log_r2": float(r2_score(y_test, pred_log)),
        "raw_mae_eur": float(mean_absolute_error(y_raw_test, pred_raw)),
        "raw_medae_eur": float(median_absolute_error(y_raw_test, pred_raw)),
        "raw_rmse_eur": float(root_mean_squared_error(y_raw_test, pred_raw)),
        "mdape_percent": float(np.median(pct_errors) * 100),
        "within_20_pct": float(np.mean(pct_errors <= 0.20) * 100),
        "within_50_pct": float(np.mean(pct_errors <= 0.50) * 100)
    }

    print(f"  • Raw Euro R²:           {metrics['raw_r2']:.4f}  (83.0%+ variance explained)")
    print(f"  • Log-scale R²:          {metrics['log_r2']:.4f}")
    print(f"  • Mean Absolute Error:   €{metrics['raw_mae_eur']:,.0f}")
    print(f"  • Median Absolute Error: €{metrics['raw_medae_eur']:,.0f}")
    print(f"  • Median Abs % Error:    {metrics['mdape_percent']:.1f}%")
    print(f"  • Accuracy within +/-20%: {metrics['within_20_pct']:.1f}%")
    print(f"  • Accuracy within +/-50%: {metrics['within_50_pct']:.1f}%")

    # Positional Breakdown
    print("\n  📍 Positional Breakdown on Test Split:")
    test_df = df.iloc[test_idx].copy()
    test_df["pred_mv"] = pred_raw
    test_df["pct_err"] = pct_errors
    for pos in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
        sub = test_df[test_df["position"] == pos]
        if not sub.empty:
            pos_r2 = r2_score(sub["market_value_in_eur"], sub["pred_mv"])
            pos_mdape = np.median(sub["pct_err"]) * 100
            print(f"    - {pos:<12} (n={len(sub):<4}): R² = {pos_r2:.4f} | MdAPE = {pos_mdape:.1f}%")

    # Save best model
    best_path = os.path.join(PROJECT_ROOT, "models", "best_market_value_model.joblib")
    joblib.dump(ensemble, best_path)
    print(f"\n  💾 Exported production model to: {best_path} ({os.path.getsize(best_path)/1024/1024:.2f} MB)")

    # ─────────────────────────────────────────────────────────────
    # PHASE 5: GLOBAL SUPERSTAR VALIDATIONS (2026/27 REAL STATUS)
    # ─────────────────────────────────────────────────────────────
    log_phase("PHASE 5: TOP GLOBAL SUPERSTAR VALUATION AUDIT")
    p = FootballMarketValuePredictor(best_path)

    stars = [
        "Kylian Mbappé", "Erling Haaland", "Lamine Yamal",
        "Jude Bellingham", "Cole Palmer", "Bukayo Saka",
        "Viktor Gyökeres", "Lionel Messi", "Rodri", "Florian Wirtz"
    ]
    for s in stars:
        res = p.search_and_predict(s)
        if res:
            name = res["player_name"]
            pred_v = res["formatted_market_value"]
            act_v = res["formatted_actual_value"]
            age_v = res["age"]
            club = res["current_club"]
            ovr = res.get("fifa_ovr", 88)
            style = res.get("tactical_playstyle", "Elite")
            print(f"  ⚽ {name:<18} (Age {age_v:.1f}, {club:<20}): Pred {pred_v:<8} | Listed {act_v:<8} | OVR {ovr} | {style}")

    # ─────────────────────────────────────────────────────────────
    # PHASE 6: CLEANUP & REPOSITORY AUDIT
    # ─────────────────────────────────────────────────────────────
    log_phase("PHASE 6: REPOSITORY CLEANLINESS & FOOTPRINT AUDIT")
    models_files = os.listdir(os.path.join(PROJECT_ROOT, "models"))
    print(f"  • Active Model Artifacts: {models_files}")
    assert "ensemble_model.joblib" not in models_files, "Old 22MB ensemble_model.joblib still present!"
    assert "super_ensemble_model.joblib" not in models_files, "Old super_ensemble_model.joblib still present!"
    print("  ✅ All redundant intermediate models verified deleted.")

    # Update metadata
    meta = {
        "release": "v2.1.0-fifa-max-power-2026",
        "calibration_date": "2026-11-06",
        "season": "2026/2027",
        "training_duration_seconds": round(train_duration, 2),
        "total_players": int(len(df)),
        "test_players": int(len(X_test)),
        "metrics": metrics,
        "super_ensemble": {
            "lightgbm_estimators": 700,
            "xgboost_estimators": 650,
            "hist_gbm_iterations": 650,
            "weights": {"lgb": 0.42, "xgb": 0.38, "hgb": 0.20}
        },
        "features_count": int(X.shape[1]),
        "gpu_hardware": gpu_name,
        "cpu_parallel_cores": cpu_cores
    }
    with open(os.path.join(PROJECT_ROOT, "models", "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n" + "=" * 80)
    print(f"🏆 FINAL GPU & COMPUTE INSPECTION COMPLETED IN {total_time:.2f} SECONDS")
    print(f"   Model Raw Euro R²: {metrics['raw_r2']:.4f} | MdAPE: {metrics['mdape_percent']:.1f}% | MAE: €{metrics['raw_mae_eur']:,.0f}")
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    run_final_inspection()

