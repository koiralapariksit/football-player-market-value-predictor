"""
Unit and Integration Tests for Football Player Market Value Pipeline.
Tests feature engineering robustness, model loading, inference stability,
and boundary conditions.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import extract_features, prepare_training_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.predict import FootballMarketValuePredictor, format_currency, categorize_tier


class TestFootballMarketValuePipeline(unittest.TestCase):

    def setUp(self):
        self.predictor = FootballMarketValuePredictor()

    def test_feature_extraction_single_profile(self):
        """Test feature extraction with typical player profile."""
        profile = {
            "name": "Test Striker",
            "age": 24,
            "position": "Attack",
            "sub_position": "Centre-Forward",
            "height_in_cm": 188,
            "foot": "right",
            "contract_years_left": 3.5,
            "career_minutes": 5000,
            "career_appearances": 60,
            "career_goals": 25,
            "career_assists": 10,
            "recent_minutes": 2500,
            "recent_goals": 15,
            "recent_assists": 5,
            "international_caps": 12,
            "continental_appearances": 8,
            "current_club_domestic_competition_id": "GB1"
        }
        df_in = pd.DataFrame([profile])
        X = extract_features(df_in, is_training=False)

        self.assertEqual(len(X), 1)
        self.assertFalse(X[NUMERIC_FEATURES].isnull().values.any())
        self.assertEqual(X["is_top_5_league"].iloc[0], 1.0)
        self.assertAlmostEqual(X["age"].iloc[0], 24.0)
        self.assertAlmostEqual(X["is_prime_age"].iloc[0], 1.0)

    def test_feature_extraction_empty_and_nulls(self):
        """Test that missing / completely null attributes default gracefully without NaN."""
        sparse_profile = {"name": "Unknown Player"}
        df_in = pd.DataFrame([sparse_profile])
        X = extract_features(df_in, is_training=False)

        self.assertEqual(len(X), 1)
        # Verify no NaN values in any numeric feature
        for col in NUMERIC_FEATURES:
            self.assertFalse(np.isnan(X[col].iloc[0]), f"NaN found in numeric column {col}")

    def test_prediction_output_structure(self):
        """Test prediction output keys and numerical boundaries."""
        sample_player = {
            "name": "Test Midfielder",
            "age": 22,
            "position": "Midfield",
            "sub_position": "Central Midfield",
            "current_club_domestic_competition_id": "ES1",
            "recent_minutes": 1800,
            "recent_goals": 4,
            "recent_assists": 7,
            "international_caps": 5,
            "continental_appearances": 4
        }
        res = self.predictor.predict_player_profile(sample_player)

        self.assertIn("predicted_market_value_eur", res)
        self.assertIn("confidence_80_lower_eur", res)
        self.assertIn("confidence_80_upper_eur", res)
        self.assertIn("market_tier", res)
        self.assertIn("valuation_drivers", res)

        # Logical boundaries: lower bound < predicted < upper bound
        self.assertGreater(res["predicted_market_value_eur"], 0)
        self.assertLess(res["confidence_80_lower_eur"], res["predicted_market_value_eur"])
        self.assertGreater(res["confidence_80_upper_eur"], res["predicted_market_value_eur"])

    def test_currency_formatting(self):
        """Test format_currency utility."""
        self.assertEqual(format_currency(120_000_000), "€120.00M")
        self.assertEqual(format_currency(500_000), "€500K")
        self.assertEqual(format_currency(850), "€850")

    def test_tier_classification(self):
        """Test categorize_tier thresholds."""
        self.assertEqual(categorize_tier(100_000_000), "Global Superstar / Elite Asset")
        self.assertEqual(categorize_tier(40_000_000), "Top European Club Starter")
        self.assertEqual(categorize_tier(20_000_000), "Established First-Tier Professional")
        self.assertEqual(categorize_tier(2_000_000), "Emerging Prospect / Second Tier Regular")

    def test_search_and_predict_known_star(self):
        """Test looking up a known player in the dataset."""
        res = self.predictor.search_and_predict("Erling Haaland")
        self.assertIsNotNone(res)
        self.assertEqual(res["player_name"], "Erling Haaland")
        self.assertGreater(res["predicted_market_value_eur"], 50_000_000)

    def test_extreme_ages_and_positions(self):
        """Test boundary conditions: teenage prodigy vs veteran goalkeeper."""
        prodigy = {
            "name": "Young Prodigy",
            "age": 16,
            "position": "Attack",
            "sub_position": "Left Winger",
            "recent_minutes": 900,
            "recent_goals": 6,
            "recent_assists": 4,
            "current_club_domestic_competition_id": "GB1"
        }
        res_prodigy = self.predictor.predict_player_profile(prodigy)
        self.assertGreater(res_prodigy["predicted_market_value_eur"], 200_000)

        veteran_gk = {
            "name": "Veteran Keeper",
            "age": 39,
            "position": "Goalkeeper",
            "sub_position": "Goalkeeper",
            "recent_minutes": 1500,
            "recent_goals": 0,
            "recent_assists": 0,
            "contract_years_left": 0.5,
            "current_club_domestic_competition_id": "IT1"
        }
        res_gk = self.predictor.predict_player_profile(veteran_gk)
        # Young attacker with high goal contribution should be valued significantly above veteran keeper
        self.assertGreater(res_prodigy["predicted_market_value_eur"], res_gk["predicted_market_value_eur"])



    def test_fifa_ovr_bounds(self):
        """Assert FIFA OVR proxy ratings stay within official EA FC scale (48-94)."""
        from src.predict import compute_fifa_ovr
        self.assertGreaterEqual(compute_fifa_ovr(200_000_000, 24, True), 88)
        self.assertLessEqual(compute_fifa_ovr(500_000, 32, False), 75)
        self.assertGreaterEqual(compute_fifa_ovr(100_000, 20, False), 48)

if __name__ == "__main__":
    unittest.main()
