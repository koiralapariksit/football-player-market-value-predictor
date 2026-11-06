"""
Inference & Prediction Module for Football Player Market Value.
Provides a clean high-level API to estimate market value from either:
1) An existing player in the database by name (e.g. 'Erling Haaland')
2) Custom player stats provided as a Python dict or CLI arguments
Includes valuation confidence intervals and key driving factor explanations.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import extract_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.ensemble_pipeline import FootballEnsemblePipeline

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_market_value_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "football_players_market_value.csv.gz")


def format_currency(value_eur: float) -> str:
    """Formats euro value into human readable notation (e.g., €125.0M, €850K)."""
    if value_eur >= 1_000_000:
        return f"€{value_eur / 1_000_000:.2f}M"
    elif value_eur >= 1_000:
        return f"€{value_eur / 1_000:.0f}K"
    else:
        return f"€{value_eur:.0f}"


def categorize_tier(value_eur: float) -> str:
    """Classifies player market value into descriptive market tiers."""
    if value_eur >= 80_000_000:
        return "Global Superstar / Elite Asset"
    elif value_eur >= 35_000_000:
        return "Top European Club Starter"
    elif value_eur >= 15_000_000:
        return "Established First-Tier Professional"
    elif value_eur >= 5_000_000:
        return "Solid Division 1 Rotation / Key Starter"
    elif value_eur >= 1_000_000:
        return "Emerging Prospect / Second Tier Regular"
    else:
        return "Developmental / Lower League Talent"



def compute_fifa_ovr(value_eur: float, age: float, is_top_5: bool) -> int:
    """Computes EA Sports FC / FIFA official proxy rating (48-94)."""
    if value_eur <= 0: return 52
    log_v = np.log10(max(value_eur, 50000))
    ovr = 52.0 + (log_v - 4.7) * (40.0 / 3.6)
    if is_top_5: ovr += 1.5
    if age <= 21 and value_eur > 50_000_000: ovr = max(ovr, 86.0)
    return int(np.clip(round(ovr), 48, 94))

class FootballMarketValuePredictor:
    """High-level predictor for professional football player market values."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, metadata_path: str = DEFAULT_METADATA_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run src/train.py first.")
        self.model = joblib.load(model_path)

        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Standard error on log scale for 80% confidence interval
        log_rmse = 0.85
        if "benchmark_results" in self.metadata:
            best_key = self.metadata.get("best_model", "HistGradientBoosting Regressor")
            if best_key in self.metadata["benchmark_results"]:
                log_rmse = self.metadata["benchmark_results"][best_key].get("log_rmse", 0.85)
        self.log_rmse = log_rmse

    def predict_from_features_df(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs pipeline and returns estimated market value with intervals."""
        log_preds = self.model.predict(features_df)
        raw_preds = np.expm1(np.maximum(log_preds, 0))

        # 80% confidence interval z-score ~ 1.28
        lower_log = np.maximum(log_preds - 1.28 * self.log_rmse, 0)
        upper_log = log_preds + 1.28 * self.log_rmse

        lower_raw = np.expm1(lower_log)
        upper_raw = np.expm1(upper_log)

        return {
            "predicted_log": float(log_preds[0]),
            "predicted_market_value_eur": float(raw_preds[0]),
            "formatted_market_value": format_currency(raw_preds[0]),
            "confidence_80_lower_eur": float(lower_raw[0]),
            "confidence_80_upper_eur": float(upper_raw[0]),
            "formatted_confidence_range": f"{format_currency(lower_raw[0])} - {format_currency(upper_raw[0])}",
            "market_tier": categorize_tier(raw_preds[0]),
            "fifa_ovr": compute_fifa_ovr(raw_preds[0], 25.0, True)
        }

    def predict_player_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw player profile dictionary, engineers features, and predicts market value.
        """
        df_input = pd.DataFrame([profile])
        X = extract_features(df_input, is_training=False)
        result = self.predict_from_features_df(X)

        # Add profile echo
        result["player_name"] = profile.get("name", "Unknown Player")
        result["position"] = profile.get("position", "Attack")
        result["age"] = float(X["age"].iloc[0])
        result["is_top_5_league"] = bool(X["is_top_5_league"].iloc[0])

        # Explain key drivers
        drivers = []
        age_val = result["age"]
        if age_val <= 21:
            drivers.append(f"High Young Talent Ceiling (Age {age_val:.0f})")
        elif 24 <= age_val <= 28:
            drivers.append(f"Prime Athletic Age Bracket (Age {age_val:.0f})")
        elif age_val >= 32:
            drivers.append(f"Veteran Age Depreciation (Age {age_val:.0f})")

        rec_g = float(X["recent_goals"].iloc[0])
        rec_a = float(X["recent_assists"].iloc[0])
        if rec_g + rec_a >= 20:
            drivers.append(f"Elite Recent Production ({rec_g:.0f}G + {rec_a:.0f}A in last 2 seasons)")
        elif rec_g + rec_a >= 10:
            drivers.append(f"Strong Attacking Contribution ({rec_g:.0f}G + {rec_a:.0f}A)")

        if X["is_top_5_league"].iloc[0] == 1.0:
            drivers.append("Top 5 European League Commercial Multiplier")

        cont = float(X["continental_appearances"].iloc[0])
        if cont >= 15:
            drivers.append(f"Proven UEFA European Continental Pedigree ({cont:.0f} apps)")

        intl = float(X["international_caps"].iloc[0])
        if intl >= 20:
            drivers.append(f"Regular Senior International Experience ({intl:.0f} caps)")

        contract = float(X["contract_years_left"].iloc[0])
        if contract >= 3.0:
            drivers.append(f"Strong Club Transfer Leverage ({contract:.1f} contract years remaining)")
        elif contract <= 1.0:
            drivers.append("Expiring Contract Leverage Discount (<1 year left)")

        result["valuation_drivers"] = drivers
        result["driver_attributions"] = [{"factor": d, "impact": "High" if idx < 2 else "Moderate"} for idx, d in enumerate(drivers[:4])]
        return result

    def search_and_predict(self, query_name: str, dataset_path: str = DEFAULT_DATA_PATH) -> Optional[Dict[str, Any]]:
        """
        Finds a real player in the dataset by name, feeds their exact statistics into the model,
        and compares predicted valuation against Transfermarkt recorded valuation.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}'")

        df = pd.read_csv(dataset_path)
        matches = df[df["name"].str.contains(query_name, case=False, na=False)]

        if matches.empty:
            return None

        # Pick best match (highest market value if multiple matches, e.g. 'Ronaldo')
        player_row = matches.sort_values("market_value_in_eur", ascending=False).iloc[0]
        player_dict = player_row.to_dict()

        pred_result = self.predict_player_profile(player_dict)
        pred_result["actual_market_value_eur"] = float(player_row["market_value_in_eur"])
        pred_result["formatted_actual_value"] = format_currency(player_row["market_value_in_eur"])
        pred_result["current_club"] = player_row.get("current_club_name", "Unknown")
        pred_result["sub_position"] = player_row.get("sub_position", "Unknown")

        # Ratio & divergence
        ratio = pred_result["predicted_market_value_eur"] / pred_result["actual_market_value_eur"]
        diff_pct = (ratio - 1.0) * 100
        pred_result["prediction_to_actual_ratio"] = round(ratio, 2)
        if abs(diff_pct) <= 15:
            pred_result["alignment"] = "Highly Aligned (+/- 15%)"
        elif diff_pct > 15:
            pred_result["alignment"] = f"Model Evaluates Higher (+{diff_pct:.1f}% vs listed)"
        else:
            pred_result["alignment"] = f"Model Evaluates Lower ({diff_pct:.1f}% vs listed)"

        return pred_result


if __name__ == "__main__":
    predictor = FootballMarketValuePredictor()

    test_players = ["Erling Haaland", "Lamine Yamal", "Bukayo Saka", "Vinicius Junior", "Cole Palmer"]
    print("\n" + "="*75)
    print("DEMO: TESTING HIGH LEVEL FOOTBALL MARKET VALUE MODEL ON TOP PLAYERS")
    print("="*75)

    for p in test_players:
        res = predictor.search_and_predict(p)
        if res:
            print(f"\nPlayer: {res['player_name']} ({res['sub_position']} - {res['current_club']})")
            print(f"  > Age: {res['age']:.0f} | League: {'Top 5' if res['is_top_5_league'] else 'Other'}")
            print(f"  > Actual Listed Market Value:    {res['formatted_actual_value']}")
            print(f"  > Model Predicted Market Value:  {res['formatted_market_value']} (80% CI: {res['formatted_confidence_range']})")
            print(f"  > Classification Tier:           {res['market_tier']}")
            print(f"  > Model vs Listed Alignment:     {res['alignment']}")
            print(f"  > Key Drivers: {', '.join(res['valuation_drivers'][:3])}")

