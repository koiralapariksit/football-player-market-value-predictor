"""
Enhanced Feature Engineering for Football Player Market Value Prediction.
Integrates advanced domain features inspired by top sports analytics research:
- Club financial caliber & prestige index (Bayesian shrinkage)
- Wonderkid breakout interaction terms
- Position-specific production rates (striker goals/90, winger G+A/90, mid assists/90)
- National team star power & top-10 football nation multipliers
- Top agency representation premium
- Non-linear age curve dynamics and contract duration leverage
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

TOP_5_COMPETITIONS = {"GB1", "ES1", "IT1", "L1", "FR1"}
TIER_2_COMPETITIONS = {"PO1", "NL1", "BE1", "TR1", "GB2", "RU1", "BRA1", "ARG1"}
TOP_NATIONS = {"Brazil", "France", "England", "Spain", "Argentina", "Germany", "Portugal", "Netherlands", "Italy", "Belgium"}
ELITE_AGENTS = {"THE·TEAM", "CAA Stellar", "ProStar", "CAA Base Ltd", "Unique Sports Group", "SEG", "AS1", "Wasserman", "ROOF", "Gestifute", "Gersh"}

UEFA_LEAGUE_COEFFICIENTS = {
    "GB1": 100.0, "ES1": 88.5, "IT1": 85.0, "L1": 84.0, "FR1": 68.0,
    "NL1": 60.5, "PO1": 56.0, "BE1": 48.0, "TR1": 42.0, "RU1": 35.0,
    "BRA1": 52.0, "ARG1": 46.0, "GB2": 45.0
}

CALIBRATION_DATE = pd.Timestamp("2026-11-06")
CALIBRATION_YEAR = 2026
PRESTIGE_MAP_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "club_prestige_map.json")

NUMERIC_FEATURES = [
    # Age & Career Stage
    "age",
    "age_squared",
    "peak_age_diff",
    "is_young_talent",
    "is_prime_age",
    "is_veteran",
    "experience_years",
    # Physical
    "height_in_cm",
    "is_left_footed",
    # Contract
    "contract_years_left",
    "is_contract_expiring",
    # Career Performance & Productivity
    "career_appearances",
    "career_minutes",
    "career_goals",
    "career_assists",
    "career_yellow_cards",
    "career_red_cards",
    "career_goals_per_90",
    "career_assists_per_90",
    "career_ga_per_90",
    "career_cards_per_90",
    "minutes_per_match",
    # Recent Form (Last 2-3 Years)
    "recent_appearances",
    "recent_minutes",
    "recent_goals",
    "recent_assists",
    "recent_goals_per_90",
    "recent_assists_per_90",
    "recent_ga_per_90",
    "recent_activity_ratio",
    # International & Continental
    "continental_appearances",
    "has_continental_exp",
    "international_caps",
    "international_goals",
    "has_intl_caps",
    "intl_goals_per_cap",
    # Competition & Club Context
    "is_top_5_league",
    "is_tier_2_league",
    "competition_tier_level",
    "club_squad_size",
    "club_avg_age",
    "club_foreigners_pct",
    "club_national_players",
    # Advanced Institutional & Interaction Features
    "club_prestige_val",
    "log_career_minutes",
    "log_recent_minutes",
    "wonderkid_minutes",
    "wonderkid_ga",
    "wonderkid_top5",
    "is_top_nation",
    "intl_star_power",
    "has_elite_agent",
    "is_striker",
    "is_winger",
    "is_mid",
    "is_cb",
    "striker_goals_per_90",
    "winger_ga_per_90",
    "mid_assists_per_90",
    "cb_minutes",
    "uefa_league_coef",
    "wonderkid_ceiling_index",
    "contract_urgency_score",
    "free_agent_risk",
    "durability_factor",
    "cards_discipline_factor",
    "striker_clinical_ratio",
    "midfield_progression_index",
    "cb_clean_sheet_leverage"
]

CATEGORICAL_FEATURES = [
    "position",
    "sub_position"
]


def load_club_prestige_map() -> Dict[str, float]:
    """Loads saved club prestige mapping or returns empty dict."""
    if os.path.exists(PRESTIGE_MAP_FILE):
        try:
            with open(PRESTIGE_MAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def build_and_save_club_prestige_map(df: pd.DataFrame, default_val: float = 275000.0) -> Dict[str, float]:
    """
    Computes Bayesian shrunk median market value per club ID and club name.
    Saves to models/club_prestige_map.json.
    """
    valid = df[df["market_value_in_eur"].notnull() & (df["market_value_in_eur"] > 0)].copy()
    overall_median = valid["market_value_in_eur"].median() if not valid.empty else default_val

    # Shrunk median by club_id
    c_weight = 5.0
    club_map = {}

    if "current_club_id" in valid.columns:
        stats_id = valid.groupby("current_club_id")["market_value_in_eur"].agg(["count", "median"])
        shrunk_id = (stats_id["count"] * stats_id["median"] + c_weight * overall_median) / (stats_id["count"] + c_weight)
        for cid, val in shrunk_id.items():
            club_map[f"id_{cid}"] = float(np.log1p(val))

    if "current_club_name" in valid.columns:
        stats_name = valid.groupby("current_club_name")["market_value_in_eur"].agg(["count", "median"])
        shrunk_name = (stats_name["count"] * stats_name["median"] + c_weight * overall_median) / (stats_name["count"] + c_weight)
        for name, val in shrunk_name.items():
            club_map[f"name_{str(name).lower()}"] = float(np.log1p(val))

    club_map["__overall_default__"] = float(np.log1p(overall_median))

    os.makedirs(os.path.dirname(PRESTIGE_MAP_FILE), exist_ok=True)
    with open(PRESTIGE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(club_map, f, indent=2)

    return club_map


def extract_features(df: pd.DataFrame, is_training: bool = True, prestige_map: Dict[str, float] = None) -> pd.DataFrame:
    """
    Transforms raw player dataframe into engineered feature dataframe.
    Safe for both training batches and individual player inference.
    """
    if prestige_map is None:
        prestige_map = load_club_prestige_map()

    default_prestige = prestige_map.get("__overall_default__", float(np.log1p(275000.0)))

    X = pd.DataFrame(index=df.index)

    def _col(name: str, default_val: Any) -> pd.Series:
        if name in df.columns:
            s = df[name].copy()
            if isinstance(s, pd.Series):
                return s
        return pd.Series(default_val, index=df.index)

    # 1. Age & Career Trajectory (Calibrated to November 6, 2026)
    ref_year = _col("last_season", CALIBRATION_YEAR).fillna(CALIBRATION_YEAR)
    if "age_nov_2026" in df.columns and df["age_nov_2026"].notnull().any():
        age = _col("age_nov_2026", 25.0).fillna(25.0)
    elif "date_of_birth" in df.columns and df["date_of_birth"].notnull().any():
        dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
        age = ((CALIBRATION_DATE - dob).dt.days / 365.25).fillna(25.0)
    elif "age" in df.columns:
        age = _col("age", 25.0).fillna(25.0)
    else:
        age = pd.Series(25.0, index=df.index)

    age = age.clip(lower=15.0, upper=46.0)
    X["age"] = age
    X["age_squared"] = (age ** 2) / 100.0  # scaled
    X["peak_age_diff"] = (age - 26.5).abs()
    X["is_young_talent"] = (age <= 21.5).astype(float)
    X["is_prime_age"] = ((age >= 23.0) & (age <= 28.5)).astype(float)
    X["is_veteran"] = (age >= 32.0).astype(float)

    # Experience span
    first_yr = _col("first_match_year", ref_year).fillna(ref_year)
    last_yr = _col("last_match_year", CALIBRATION_YEAR).fillna(CALIBRATION_YEAR)
    exp = (last_yr - first_yr + 1).fillna(1.0).clip(lower=0, upper=25)
    X["experience_years"] = exp.astype(float)

    # 2. Physical & Foot
    height = _col("height_in_cm", 182.0).fillna(182.0)
    X["height_in_cm"] = height.clip(lower=160, upper=205)

    foot = _col("foot", "right").astype(str).str.lower()
    X["is_left_footed"] = foot.isin(["left", "both"]).astype(float)

    # 3. Contract Status (Anchored to November 6, 2026)
    if "contract_years_nov_2026" in df.columns and df["contract_years_nov_2026"].notnull().any():
        diff_years = _col("contract_years_nov_2026", 2.0).fillna(2.0).clip(lower=0.0, upper=8.0)
        X["contract_years_left"] = diff_years
        X["is_contract_expiring"] = (diff_years <= 1.0).astype(float)
    elif "contract_years_left" in df.columns:
        diff_years = _col("contract_years_left", 2.0).fillna(2.0).clip(lower=0.0, upper=8.0)
        X["contract_years_left"] = diff_years
        X["is_contract_expiring"] = (diff_years <= 1.0).astype(float)
    elif "contract_expiration_date" in df.columns:
        exp_date = pd.to_datetime(df["contract_expiration_date"], errors="coerce")
        diff_years = (exp_date - CALIBRATION_DATE).dt.days / 365.25
        diff_years = diff_years.fillna(1.5).clip(lower=0.0, upper=8.0)
        X["contract_years_left"] = diff_years
        X["is_contract_expiring"] = (diff_years <= 1.0).astype(float)
    else:
        X["contract_years_left"] = 2.0
        X["is_contract_expiring"] = 0.0

    # 4. Career Performance & Efficiency
    car_app = _col("career_appearances", 0).fillna(0).astype(float)
    car_min = _col("career_minutes", 0).fillna(0).astype(float)
    car_g = _col("career_goals", 0).fillna(0).astype(float)
    car_a = _col("career_assists", 0).fillna(0).astype(float)
    car_yc = _col("career_yellow_cards", 0).fillna(0).astype(float)
    car_rc = _col("career_red_cards", 0).fillna(0).astype(float)

    car_90s = car_min / 90.0
    safe_90s = np.maximum(car_90s, 1.0)

    X["career_appearances"] = car_app
    X["career_minutes"] = car_min
    X["career_goals"] = car_g
    X["career_assists"] = car_a
    X["career_yellow_cards"] = car_yc
    X["career_red_cards"] = car_rc
    X["career_goals_per_90"] = (car_g / safe_90s).clip(upper=2.0)
    X["career_assists_per_90"] = (car_a / safe_90s).clip(upper=2.0)
    X["career_ga_per_90"] = X["career_goals_per_90"] + X["career_assists_per_90"]
    X["career_cards_per_90"] = ((car_yc + 2 * car_rc) / safe_90s).clip(upper=1.5)
    X["minutes_per_match"] = (car_min / np.maximum(car_app, 1.0)).clip(upper=90.0)

    # 5. Recent Form (Last 2-3 Seasons)
    rec_app = _col("recent_appearances", 0).fillna(0).astype(float)
    rec_min = _col("recent_minutes", 0).fillna(0).astype(float)
    rec_g = _col("recent_goals", 0).fillna(0).astype(float)
    rec_a = _col("recent_assists", 0).fillna(0).astype(float)

    rec_90s = rec_min / 90.0
    safe_rec_90s = np.maximum(rec_90s, 1.0)

    X["recent_appearances"] = rec_app
    X["recent_minutes"] = rec_min
    X["recent_goals"] = rec_g
    X["recent_assists"] = rec_a
    X["recent_goals_per_90"] = (rec_g / safe_rec_90s).clip(upper=2.0)
    X["recent_assists_per_90"] = (rec_a / safe_rec_90s).clip(upper=2.0)
    X["recent_ga_per_90"] = X["recent_goals_per_90"] + X["recent_assists_per_90"]
    X["recent_activity_ratio"] = (rec_min / np.maximum(car_min, 90.0)).clip(upper=1.0)

    # 6. Continental & International
    cont_app = _col("continental_appearances", 0).fillna(0).astype(float)
    intl_caps = _col("international_caps", 0).fillna(0).astype(float)
    intl_goals = _col("international_goals", 0).fillna(0).astype(float)

    X["continental_appearances"] = cont_app
    X["has_continental_exp"] = (cont_app > 0).astype(float)
    X["international_caps"] = intl_caps
    X["international_goals"] = intl_goals
    X["has_intl_caps"] = (intl_caps > 0).astype(float)
    X["intl_goals_per_cap"] = (intl_goals / np.maximum(intl_caps, 1.0)).clip(upper=1.5)

    # 7. Competition & Club Context
    comp_id = _col("current_club_domestic_competition_id", "").fillna("").astype(str)
    is_top5 = comp_id.isin(TOP_5_COMPETITIONS).astype(float)
    is_tier2 = comp_id.isin(TIER_2_COMPETITIONS).astype(float)

    X["is_top_5_league"] = is_top5
    X["is_tier_2_league"] = is_tier2

    # Tier: 1 = Top 5, 2 = Tier 2, 3 = other domestic, 4 = lower/missing
    tier = pd.Series(3.0, index=df.index)
    tier[is_top5 == 1.0] = 1.0
    tier[is_tier2 == 1.0] = 2.0
    tier[comp_id == ""] = 4.0
    X["competition_tier_level"] = tier
    X["uefa_league_coef"] = comp_id.map(UEFA_LEAGUE_COEFFICIENTS).fillna(25.0).astype(float)

    # Club metrics
    X["club_squad_size"] = _col("club_squad_size", 26.0).fillna(26.0).astype(float)
    X["club_avg_age"] = _col("club_avg_age", 25.5).fillna(25.5).astype(float)
    X["club_foreigners_pct"] = _col("club_foreigners_pct", 40.0).fillna(40.0).astype(float)
    X["club_national_players"] = _col("club_national_players", 5.0).fillna(5.0).astype(float)

    # 8. Categoricals (Position & Sub-position)
    pos = _col("position", "Attack").fillna("Attack").astype(str)
    valid_positions = {"Attack", "Midfield", "Defender", "Goalkeeper"}
    pos = pos.apply(lambda p: p if p in valid_positions else "Attack")
    X["position"] = pos

    sub_pos = _col("sub_position", "Centre-Forward").fillna("Centre-Forward").astype(str)
    X["sub_position"] = sub_pos

    # 9. Advanced Domain & Interaction Features (from GitHub / Literature)
    # Club Prestige Index
    def _lookup_prestige(row):
        cid = row.get("current_club_id")
        if pd.notnull(cid) and f"id_{cid}" in prestige_map:
            return prestige_map[f"id_{cid}"]
        cname = str(row.get("current_club_name", "")).lower()
        if cname and f"name_{cname}" in prestige_map:
            return prestige_map[f"name_{cname}"]
        # If playing in Top 5 league, default higher prestige
        if row.get("current_club_domestic_competition_id") in TOP_5_COMPETITIONS:
            return float(np.log1p(15000000.0))
        return default_prestige

    X["club_prestige_val"] = df.apply(_lookup_prestige, axis=1)

    # Log transformed minutes
    X["log_career_minutes"] = np.log1p(X["career_minutes"])
    X["log_recent_minutes"] = np.log1p(X["recent_minutes"])

    # Wonderkid Breakout Interactions
    X["wonderkid_minutes"] = X["is_young_talent"] * X["log_recent_minutes"]
    X["wonderkid_ga"] = X["is_young_talent"] * X["recent_ga_per_90"]
    X["wonderkid_top5"] = X["is_young_talent"] * X["is_top_5_league"]

    # National Star Power
    nat = _col("country_of_citizenship", "").astype(str)
    is_top_nat = nat.isin(TOP_NATIONS).astype(float)
    X["is_top_nation"] = is_top_nat
    X["intl_star_power"] = X["international_caps"] * is_top_nat

    # Elite Agent
    agent = _col("agent_name", "").astype(str)
    X["has_elite_agent"] = agent.isin(ELITE_AGENTS).astype(float)

    # Positional Specialization
    is_str = sub_pos.isin(["Centre-Forward", "Second Striker"]).astype(float)
    is_wng = sub_pos.isin(["Left Winger", "Right Winger"]).astype(float)
    is_md = sub_pos.isin(["Central Midfield", "Attacking Midfield", "Defensive Midfield"]).astype(float)
    is_c_b = (sub_pos == "Centre-Back").astype(float)

    X["is_striker"] = is_str
    X["is_winger"] = is_wng
    X["is_mid"] = is_md
    X["is_cb"] = is_c_b

    X["striker_goals_per_90"] = is_str * X["recent_goals_per_90"]
    X["winger_ga_per_90"] = is_wng * X["recent_ga_per_90"]
    X["mid_assists_per_90"] = is_md * X["recent_assists_per_90"]
    X["cb_minutes"] = is_c_b * X["log_recent_minutes"]

    # Wonderkid Ceiling Index & Elasticity
    X["wonderkid_ceiling_index"] = (23.5 - X["age"]).clip(lower=0) * np.log1p(X["career_minutes"]) * (1.0 + 0.6 * X["is_top_5_league"])

    # Bosman Contract Urgency & Free Agent Risk
    X["contract_urgency_score"] = 1.0 / np.maximum(X["contract_years_left"], 0.25)
    X["free_agent_risk"] = X["is_contract_expiring"] * np.log1p(X["recent_minutes"])

    # Physical Durability & Disciplinary Squad Load
    X["durability_factor"] = (X["recent_minutes"] / np.maximum(X["recent_appearances"] * 90.0, 1.0)).clip(upper=1.0)
    X["cards_discipline_factor"] = (X["career_cards_per_90"]).clip(upper=1.5)

    # Position Mastery & Dominance
    X["striker_clinical_ratio"] = is_str * (X["recent_goals"] / np.maximum(X["recent_appearances"], 1.0))
    X["midfield_progression_index"] = is_md * (X["recent_assists_per_90"] + 0.5 * (X["recent_minutes"] / 90.0))
    X["cb_clean_sheet_leverage"] = is_c_b * (X["recent_minutes"] / 90.0) * X["is_top_5_league"]

    # Ensure all numeric columns are clean float without NaN
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    return X


def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Prepares training feature matrix X, original target y, and log target y_log.
    Builds the club prestige map on the input dataframe.
    """
    valid = df[df["market_value_in_eur"].notnull() & (df["market_value_in_eur"] > 0)].copy()
    y_raw = valid["market_value_in_eur"].values.astype(float)
    y_log = np.log1p(y_raw)

    prestige_map = build_and_save_club_prestige_map(valid)
    X = extract_features(valid, is_training=True, prestige_map=prestige_map)
    return X, y_log, y_raw

# UEFA 5-Year Country Coefficients (Active 2026/27)

# UEFA 5-Year Country Coefficients (Active 2026/27)
