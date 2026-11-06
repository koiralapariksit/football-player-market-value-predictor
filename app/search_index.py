"""
Fast in-memory player name autocomplete/search engine.

Builds a pre-computed, accent-insensitive substring index over player names
so that any query resolves in <15 ms.  Results are ranked by prefix-match
priority, then by descending market value so the most notable players
surface first.
"""

import os
import re
import unicodedata
from datetime import date
from typing import Dict, List, Optional

import pandas as pd


def _strip_accents(text: str) -> str:
    """Remove diacritical marks (accents) from *text* via NFD decomposition."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def _normalize(text: str) -> str:
    """Lower-case, accent-stripped version of *text* for index look-ups."""
    return _strip_accents(text).lower()


def _compute_age(dob: Optional[str], last_season: Optional[int]) -> Optional[int]:
    """
    Approximate age at end of *last_season*.

    Uses June 30 of ``last_season`` as the reference date (typical end of a
    European football season).  Falls back to today's date when
    *last_season* is missing.
    """
    if not dob or pd.isna(dob):
        return None
    try:
        birth = pd.Timestamp(dob)
    except Exception:
        return None

    if last_season and not pd.isna(last_season):
        ref = date(int(last_season), 6, 30)
    else:
        ref = date.today()

    age = ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
    return int(age) if age >= 0 else None


class PlayerSearchIndex:
    """
    In-memory, accent-insensitive autocomplete index over football player
    names.

    Parameters
    ----------
    data_path : str
        Path to a (possibly gzipped) CSV containing at least the columns
        ``player_id`` and ``name``.

    Attributes
    ----------
    players : list[dict]
        One dict per player row with the fields the API needs.
    _id_map : dict[int, dict]
        Quick look-up by ``player_id``.
    _norm_names : list[str]
        Parallel list of normalised (lower-case, accent-free) names used for
        substring matching.
    """

    # Fields surfaced in search results
    _RESULT_FIELDS = (
        "player_id",
        "name",
        "position",
        "sub_position",
        "current_club_name",
        "current_club_domestic_competition_id",
        "market_value_in_eur",
        "country_of_citizenship",
        "age",
        "image_url",
    )

    def __init__(self, data_path: str) -> None:
        self._load_and_index(data_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_index(self, data_path: str) -> None:
        """Read the CSV, compute derived fields, and build look-up structures."""
        df = pd.read_csv(data_path)

        # Compute age from date_of_birth and last_season
        df["age"] = df.apply(
            lambda r: _compute_age(r.get("date_of_birth"), r.get("last_season")),
            axis=1,
        )

        # Coerce market_value to float (NaN-safe) for sorting
        df["market_value_in_eur"] = pd.to_numeric(
            df["market_value_in_eur"], errors="coerce"
        )

        # Keep only needed columns (plus name for indexing)
        available = [c for c in self._RESULT_FIELDS if c in df.columns]
        df_slim = df[available].copy()

        # Build list-of-dicts for fast iteration
        self.players: List[Dict] = df_slim.to_dict(orient="records")

        # Fix NaN → None for JSON-friendliness
        for p in self.players:
            for k, v in p.items():
                if pd.isna(v) if isinstance(v, float) else False:
                    p[k] = None

        # Pre-compute normalised names (parallel list, same order)
        self._norm_names: List[str] = [
            _normalize(str(p.get("name", ""))) for p in self.players
        ]

        # Map player_id → dict for O(1) look-up
        self._id_map: Dict[int, Dict] = {}
        # Also store the full rows for get_player_by_id
        full_records = df.to_dict(orient="records")
        for rec in full_records:
            pid = rec.get("player_id")
            if pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
                # Clean NaN values
                clean = {
                    k: (None if isinstance(v, float) and pd.isna(v) else v)
                    for k, v in rec.items()
                }
                self._id_map[int(pid)] = clean

        print(
            f"[SearchIndex] Loaded {len(self.players):,} players, "
            f"index ready."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 8) -> List[Dict]:
        """
        Return up to *limit* players whose name contains *query*
        (accent-insensitive, case-insensitive).

        Results are ordered by:
          1. Exact prefix match first (normalised name starts with query)
          2. Descending ``market_value_in_eur``
        """
        if not query:
            return []

        q = _normalize(query)

        prefix_hits: List[Dict] = []
        substring_hits: List[Dict] = []

        for idx, norm_name in enumerate(self._norm_names):
            if q in norm_name:
                player = self.players[idx]
                # Check if any word in the name starts with the query,
                # or the full normalised name starts with it.
                if norm_name.startswith(q) or any(
                    part.startswith(q) for part in norm_name.split()
                ):
                    prefix_hits.append(player)
                else:
                    substring_hits.append(player)

        # Sort each bucket by market value descending (None → -1)
        def _mv(p: Dict) -> float:
            v = p.get("market_value_in_eur")
            return v if v is not None else -1

        prefix_hits.sort(key=_mv, reverse=True)
        substring_hits.sort(key=_mv, reverse=True)

        return (prefix_hits + substring_hits)[:limit]

    def get_player_by_id(self, player_id: int) -> Optional[Dict]:
        """Return the full player row as a dict, or ``None``."""
        return self._id_map.get(int(player_id))


# ----------------------------------------------------------------------
# Quick interactive smoke-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import time

    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "processed",
        "football_players_market_value.csv.gz",
    )

    idx = PlayerSearchIndex(DATA_PATH)

    test_queries = ["lio", "mbap", "haal", "pal", "bell", "ron", "mes", "ney"]

    for q in test_queries:
        t0 = time.perf_counter()
        results = idx.search(q)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"\n🔎 '{q}' ({elapsed_ms:.2f} ms) → {len(results)} results:")
        for r in results:
            mv = r.get("market_value_in_eur")
            mv_str = f"€{mv:,.0f}" if mv else "N/A"
            print(
                f"   {r.get('name', '?'):30s}  "
                f"{str(r.get('position', '')):15s}  "
                f"{str(r.get('current_club_name', '')):25s}  "
                f"{mv_str}"
            )
