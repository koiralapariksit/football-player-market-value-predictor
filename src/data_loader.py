"""
Data Loader & Ingestion Pipeline for Football Player Market Value Modeling.
Downloads, caches, and joins Transfermarkt players, match appearances,
clubs, and competition tiers.
"""

import os
import gzip
import io
import time
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
DEFAULT_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
DEFAULT_PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

FILES = {
    "players": "players.csv.gz",
    "appearances": "appearances.csv.gz",
    "clubs": "clubs.csv.gz",
    "competitions": "competitions.csv.gz"
}

TOP_5_COMPETITIONS = {"GB1", "ES1", "IT1", "L1", "FR1"}
TIER_2_COMPETITIONS = {"PO1", "NL1", "BE1", "TR1", "GB2", "RU1", "BRA1", "ARG1"}
EUROPEAN_COMPETITIONS = {"CL", "EL", "ECL", "CLQ", "ELQ", "ECLQ", "USC"}


def download_file_if_missing(filename: str, raw_dir: str = DEFAULT_RAW_DIR) -> str:
    """Download dataset file from Cloudflare R2 bucket if not cached locally."""
    os.makedirs(raw_dir, exist_ok=True)
    target_path = os.path.join(raw_dir, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[CACHE] Using cached {filename} ({os.path.getsize(target_path)/1024/1024:.2f} MB)")
        return target_path

    url = f"{BASE_URL}/{filename}"
    print(f"[DOWNLOAD] Downloading {filename} from {url}...")
    t0 = time.time()
    response = requests.get(url, headers=HEADERS, stream=True)
    response.raise_for_status()

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(target_path) / 1024 / 1024
    print(f"[DOWNLOAD] Saved {filename} ({size_mb:.2f} MB in {elapsed:.2f}s)")
    return target_path


def load_raw_tables(raw_dir: str = DEFAULT_RAW_DIR):
    """Ensure all required tables are downloaded and return file paths."""
    paths = {}
    for key, fname in FILES.items():
        paths[key] = download_file_if_missing(fname, raw_dir=raw_dir)
    return paths


def aggregate_appearances(appearances_path: str) -> pd.DataFrame:
    """
    Read appearances.csv.gz and aggregate player career, recent, and continental statistics.
    """
    print("[ETL] Reading and aggregating match appearances...")
    t0 = time.time()

    cols = [
        "player_id", "date", "competition_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played"
    ]

    with gzip.open(appearances_path, "rt", encoding="utf-8") as f:
        df_app = pd.read_csv(f, usecols=cols)

    df_app["date"] = pd.to_datetime(df_app["date"], errors="coerce")
    df_app["year"] = df_app["date"].dt.year
    df_app["is_continental"] = df_app["competition_id"].isin(EUROPEAN_COMPETITIONS).astype(int)

    # 1. Career totals
    career_agg = df_app.groupby("player_id").agg(
        career_appearances=("minutes_played", "count"),
        career_minutes=("minutes_played", "sum"),
        career_goals=("goals", "sum"),
        career_assists=("assists", "sum"),
        career_yellow_cards=("yellow_cards", "sum"),
        career_red_cards=("red_cards", "sum"),
        continental_appearances=("is_continental", "sum"),
        first_match_year=("year", "min"),
        last_match_year=("year", "max")
    ).reset_index()

    # 2. Recent form (matches in last 2-3 years: >= 2023)
    recent_mask = df_app["year"] >= 2023
    recent_agg = df_app[recent_mask].groupby("player_id").agg(
        recent_appearances=("minutes_played", "count"),
        recent_minutes=("minutes_played", "sum"),
        recent_goals=("goals", "sum"),
        recent_assists=("assists", "sum"),
        recent_yellow_cards=("yellow_cards", "sum"),
        recent_red_cards=("red_cards", "sum")
    ).reset_index()

    stats_merged = pd.merge(career_agg, recent_agg, on="player_id", how="left").fillna(0)
    print(f"[ETL] Appearance aggregation complete ({len(stats_merged)} players processed in {time.time()-t0:.2f}s)")
    return stats_merged


def build_football_dataset(
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    min_last_season: int = 2021
) -> pd.DataFrame:
    """
    Download, process, join, and save the curated football dataset.
    """
    os.makedirs(processed_dir, exist_ok=True)
    processed_file = os.path.join(processed_dir, "football_players_market_value.csv.gz")

    paths = load_raw_tables(raw_dir)

    # 1. Load players
    print("[ETL] Loading players table...")
    with gzip.open(paths["players"], "rt", encoding="utf-8") as f:
        df_players = pd.read_csv(f)

    # Keep players with valid market values and modern seasons
    valid_mask = (
        df_players["market_value_in_eur"].notnull() &
        (df_players["market_value_in_eur"] > 0) &
        (df_players["last_season"] >= min_last_season)
    )
    df_players = df_players[valid_mask].copy()
    print(f"[ETL] Filtered to {len(df_players)} active players with market valuations.")

    # 2. Aggregate appearances
    df_stats = aggregate_appearances(paths["appearances"])

    # 3. Load clubs
    print("[ETL] Loading clubs table...")
    with gzip.open(paths["clubs"], "rt", encoding="utf-8") as f:
        df_clubs = pd.read_csv(f, usecols=[
            "club_id", "squad_size", "average_age",
            "foreigners_percentage", "national_team_players"
        ])
    df_clubs.rename(columns={
        "squad_size": "club_squad_size",
        "average_age": "club_avg_age",
        "foreigners_percentage": "club_foreigners_pct",
        "national_team_players": "club_national_players"
    }, inplace=True)

    # 4. Load competitions
    with gzip.open(paths["competitions"], "rt", encoding="utf-8") as f:
        df_comp = pd.read_csv(f, usecols=["competition_id", "name", "country_name", "sub_type"])
    df_comp.rename(columns={
        "name": "competition_name",
        "country_name": "competition_country",
        "sub_type": "competition_tier_type"
    }, inplace=True)

    # Merge players with match stats
    merged = pd.merge(df_players, df_stats, on="player_id", how="left")
    stat_fill_zeros = [
        "career_appearances", "career_minutes", "career_goals", "career_assists",
        "career_yellow_cards", "career_red_cards", "continental_appearances",
        "recent_appearances", "recent_minutes", "recent_goals", "recent_assists",
        "recent_yellow_cards", "recent_red_cards"
    ]
    for col in stat_fill_zeros:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Merge clubs info
    merged = pd.merge(merged, df_clubs, left_on="current_club_id", right_on="club_id", how="left")

    # Merge competitions info
    merged = pd.merge(
        merged, df_comp,
        left_on="current_club_domestic_competition_id",
        right_on="competition_id",
        how="left"
    )

    # Save to compressed CSV
    print(f"[ETL] Saving processed dataset ({len(merged)} records) to {processed_file}...")
    merged.to_csv(processed_file, index=False, compression="gzip")
    print(f"[ETL] Processed dataset successfully created!")

    return merged


if __name__ == "__main__":
    df = build_football_dataset()
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist()[:15])

