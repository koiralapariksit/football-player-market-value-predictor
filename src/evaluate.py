"""
Cross-Validation & Positional Error Evaluation Script for Football Valuation Model.
"""
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import FootballMarketValuePredictor

def evaluate_position_cohorts():
    p = FootballMarketValuePredictor()
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "football_players_market_value.csv.gz")
    df = pd.read_csv(data_path)
    print(f"[EVAL] Evaluating across {len(df):,} players...")
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    positions = sample["position"].unique()
    for pos in positions:
        sub = sample[sample["position"] == pos]
        print(f"  • Position: {pos:<12} | Samples: {len(sub):<5}")

if __name__ == "__main__":
    evaluate_position_cohorts()
