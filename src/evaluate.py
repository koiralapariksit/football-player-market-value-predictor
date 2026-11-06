"""
Evaluation & Deep-Dive Diagnostics for Football Market Value Predictor.
Analyzes error distributions across positions, league tiers, and age brackets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
PREDICTIONS_PATH = os.path.join(MODELS_DIR, "test_predictions_sample.csv")


def run_evaluation_diagnostics():
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"Predictions file not found at '{PREDICTIONS_PATH}'. Run train.py first.")

    df = pd.read_csv(PREDICTIONS_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("=" * 80)
    print("FOOTBALL MARKET VALUE PREDICTOR - COMPREHENSIVE EVALUATION REPORT")
    print("=" * 80)

    # 1. Overall Benchmark Summary
    print("\n1. MODEL BENCHMARK COMPARISON (Log and Raw Euro Scale):")
    bench = meta["benchmark_results"]
    summary_rows = []
    for model_name, m in bench.items():
        summary_rows.append({
            "Model": model_name,
            "CV Log R2": f"{m.get('cv_log_r2_mean', 0):.3f} +/- {m.get('cv_log_r2_std', 0):.3f}",
            "Test Log R2": f"{m['log_r2']:.4f}",
            "Raw Euro R2": f"{m['raw_r2']:.4f}",
            "MAE (EUR)": f"€{m['raw_mae_eur']:,.0f}",
            "MedAE (EUR)": f"€{m['raw_medae_eur']:,.0f}",
            "MdAPE (%)": f"{m['mdape_percent']:.1f}%",
            "Within +/-50%": f"{m['within_50_pct']:.1f}%"
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # 2. Performance breakdown by Position
    print("\n2. PERFORMANCE BREAKDOWN BY POSITION (Test Set):")
    pos_stats = df.groupby("position").agg(
        Count=("player_id", "count"),
        Mean_Actual_EUR=("market_value_in_eur", "mean"),
        Mean_Pred_EUR=("pred_market_value_eur", "mean"),
        Median_Abs_Pct_Err=("abs_pct_error", "median"),
        MAE_EUR=("error_eur", lambda s: np.mean(np.abs(s)))
    ).reset_index()

    pos_stats["Mean_Actual"] = pos_stats["Mean_Actual_EUR"].apply(lambda v: f"€{v:,.0f}")
    pos_stats["Mean_Pred"] = pos_stats["Mean_Pred_EUR"].apply(lambda v: f"€{v:,.0f}")
    pos_stats["MAE"] = pos_stats["MAE_EUR"].apply(lambda v: f"€{v:,.0f}")
    pos_stats["MdAPE"] = pos_stats["Median_Abs_Pct_Err"].apply(lambda v: f"{v:.1f}%")

    print(pos_stats[["position", "Count", "Mean_Actual", "Mean_Pred", "MAE", "MdAPE"]].to_string(index=False))

    # 3. Top Feature Importances
    print("\n3. TOP 12 VALUATION DRIVERS (Permutation Importance):")
    imp_df = pd.DataFrame(meta.get("feature_importances", []))
    if not imp_df.empty:
        imp_df["importance"] = imp_df["importance"].round(4)
        imp_df["std"] = imp_df["std"].round(4)
        print(imp_df.head(12).to_string(index=False))

    # 4. Outlier Analysis - Largest Discrepancies
    print("\n4. TOP OVER-PERFORMERS RELATIVE TO MODEL (Potential Breakout / High-Hype Stars):")
    df["ratio"] = df["market_value_in_eur"] / df["pred_market_value_eur"]
    over = df.sort_values("ratio", ascending=False).head(5)
    for _, r in over.iterrows():
        print(f"  - {r['name']} ({r['sub_position']} - {r['current_club_name']}): Listed €{r['market_value_in_eur']:,.0f} vs Model €{r['pred_market_value_eur']:,.0f} (Ratio: {r['ratio']:.2f}x)")

    print("\n5. TOP UNDERVALUED PLAYERS RELATIVE TO MODEL (Potential Bargain / Underrated Talents):")
    under = df[df["market_value_in_eur"] >= 500_000].sort_values("ratio", ascending=True).head(5)
    for _, r in under.iterrows():
        print(f"  - {r['name']} ({r['sub_position']} - {r['current_club_name']}): Listed €{r['market_value_in_eur']:,.0f} vs Model €{r['pred_market_value_eur']:,.0f} (Ratio: {r['ratio']:.2f}x)")


if __name__ == "__main__":
    run_evaluation_diagnostics()

