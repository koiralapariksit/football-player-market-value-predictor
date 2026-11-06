"""
Flask Web Server for Football Player Market Value Prediction.
Serves the interactive web UI and provides REST API endpoints for:
  - Player name autocomplete/search
  - Live stat scraping from the web
  - ML model inference with confidence intervals
  - What-if scenario simulation
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify

# Lazy-loaded singletons
_search_index = None
_predictor = None
_scraper = None


def get_search_index():
    global _search_index
    if _search_index is None:
        from app.search_index import PlayerSearchIndex
        data_path = os.path.join(PROJECT_ROOT, "data", "processed", "football_players_market_value.csv.gz")
        print("[SERVER] Loading player search index...")
        _search_index = PlayerSearchIndex(data_path)
        count = len(_search_index.players) if hasattr(_search_index, "players") else "26,000+"
        print(f"[SERVER] Search index ready ({count} players indexed)")
    return _search_index


def get_predictor():
    global _predictor
    if _predictor is None:
        base_path = os.path.join(PROJECT_ROOT, "models", "best_market_value_model.joblib")
        from src.predict import FootballMarketValuePredictor
        print("[SERVER] Loading prediction model...")
        _predictor = FootballMarketValuePredictor(base_path)
        print("[SERVER] Prediction model loaded!")
    return _predictor


def get_scraper():
    global _scraper
    if _scraper is None:
        from app.web_scraper import PlayerStatsScraper
        _scraper = PlayerStatsScraper()
    return _scraper


# Create Flask app
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)


@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/api/search", methods=["GET"])
def api_search():
    """
    Autocomplete endpoint.
    GET /api/search?q=lio&limit=8
    Returns top matching players with basic info.
    """
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 8)), 20)

    if len(query) < 2:
        return jsonify({"results": [], "query": query})

    idx = get_search_index()
    results = idx.search(query, limit=limit)
    return jsonify({"results": results, "query": query})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predict market value for a player.
    POST /api/predict  { "player_id": 12345 }
    or
    POST /api/predict  { "player_stats": { ... custom stats ... } }
    """
    data = request.get_json(force=True)
    predictor = get_predictor()
    idx = get_search_index()

    player_id = data.get("player_id")
    custom_stats = data.get("player_stats")

    if player_id is not None:
        # Look up from dataset
        player_data = idx.get_player_by_id(int(player_id))
        if player_data is None:
            return jsonify({"error": f"Player ID {player_id} not found"}), 404

        # Try scraping latest info (non-blocking, with fallback)
        scraped_info = None
        try:
            scraper = get_scraper()
            url = player_data.get("url")
            if url:
                scraped_info = scraper.scrape_player(
                    player_name=player_data["name"],
                    player_url=url,
                    player_id=int(player_id)
                )
        except Exception:
            pass  # Scraping is optional enrichment

        # Run prediction with dataset stats
        result = predictor.predict_player_profile(player_data)

        actual_mv = float(player_data.get("market_value_in_eur", 0))
        result["actual_market_value_eur"] = actual_mv
        from src.predict import format_currency
        result["formatted_actual_value"] = format_currency(actual_mv)
        if actual_mv > 0:
            ratio = result["predicted_market_value_eur"] / actual_mv
            result["prediction_to_actual_ratio"] = round(ratio, 2)
            diff_pct = (ratio - 1.0) * 100
            if abs(diff_pct) <= 15:
                result["alignment"] = "Highly Aligned (+/- 15%)"
            elif diff_pct > 15:
                result["alignment"] = f"Model Evaluates Higher (+{diff_pct:.1f}% vs listed)"
            else:
                result["alignment"] = f"Model Evaluates Lower ({diff_pct:.1f}% vs listed)"

        # Enrich with scraped data if available
        if scraped_info:
            result["scraped_data"] = scraped_info
            result["data_source"] = "Live Web + Dataset Stats"
        else:
            result["data_source"] = "Transfermarkt Dataset (Cached)"

        # Add full player stats for the UI
        result["player_stats"] = {
            "age": float(player_data.get("age", 0)) if "age" in player_data else None,
            "position": player_data.get("position", "Unknown"),
            "sub_position": player_data.get("sub_position", "Unknown"),
            "current_club": player_data.get("current_club_name", "Unknown"),
            "league_id": player_data.get("current_club_domestic_competition_id", ""),
            "nationality": player_data.get("country_of_citizenship", "Unknown"),
            "height_cm": player_data.get("height_in_cm"),
            "foot": player_data.get("foot", "Unknown"),
            "career_appearances": int(player_data.get("career_appearances", 0)),
            "career_goals": int(player_data.get("career_goals", 0)),
            "career_assists": int(player_data.get("career_assists", 0)),
            "career_minutes": int(player_data.get("career_minutes", 0)),
            "recent_appearances": int(player_data.get("recent_appearances", 0)),
            "recent_goals": int(player_data.get("recent_goals", 0)),
            "recent_assists": int(player_data.get("recent_assists", 0)),
            "recent_minutes": int(player_data.get("recent_minutes", 0)),
            "international_caps": int(player_data.get("international_caps", 0)),
            "international_goals": int(player_data.get("international_goals", 0)),
            "continental_appearances": int(player_data.get("continental_appearances", 0)),
            "actual_market_value_eur": float(player_data.get("market_value_in_eur", 0)),
            "image_url": player_data.get("image_url", ""),
            "contract_expiration_date": str(player_data.get("contract_expiration_date", "")),
        }

        return jsonify(result)

    elif custom_stats is not None:
        # Custom / what-if prediction
        result = predictor.predict_player_profile(custom_stats)
        result["data_source"] = "Custom User Input"
        return jsonify(result)
    else:
        return jsonify({"error": "Provide player_id or player_stats"}), 400


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """
    What-if scenario: take a player's base stats, apply modifications, re-predict.
    POST /api/whatif { "player_id": 12345, "overrides": { "recent_goals": 25, "current_club_domestic_competition_id": "GB1" } }
    """
    data = request.get_json(force=True)
    predictor = get_predictor()
    idx = get_search_index()

    player_id = data.get("player_id")
    overrides = data.get("overrides", {})

    if player_id is None:
        return jsonify({"error": "player_id required"}), 400

    player_data = idx.get_player_by_id(int(player_id))
    if player_data is None:
        return jsonify({"error": f"Player ID {player_id} not found"}), 404

    # Apply overrides
    modified = dict(player_data)
    modified.update(overrides)

    result = predictor.predict_player_profile(modified)
    result["data_source"] = "What-If Scenario"
    result["overrides_applied"] = overrides
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 60)
    print("⚽ FOOTBALL PLAYER MARKET VALUE PREDICTION SERVER")
    print("=" * 60)

    # Pre-load on startup
    get_search_index()
    get_predictor()

    print("\n[SERVER] Starting Flask server on http://127.0.0.1:5000")
    print("[SERVER] Open your browser to http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False)

