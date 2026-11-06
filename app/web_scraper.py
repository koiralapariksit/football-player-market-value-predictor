"""
Live web stats scraper for football player data from Transfermarkt.

Used as an optional enrichment layer — if scraping fails for any reason,
returns None so the caller can fall back to local dataset stats.
"""

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = Path(r"d:\ml\data\cache")
CACHE_MAX_AGE = timedelta(hours=24)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
}

REQUEST_TIMEOUT = 15  # seconds
MIN_REQUEST_INTERVAL = 2.0  # rate-limit: max 1 request per 2 seconds


class PlayerStatsScraper:
    """Scrapes Transfermarkt player profile pages for live stats.

    Results are cached locally as JSON; stale caches (>24 h) are refreshed.
    Every public method is wrapped so it **never** raises — returns ``None``
    on failure instead.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        self._session.headers.update(REQUEST_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_player(
        self,
        player_name: str,
        player_url: str = None,
        player_id: int = None,
    ) -> Optional[Dict]:
        """Fetch the latest stats for a player.

        Parameters
        ----------
        player_name:
            Human-readable name (used for logging / cache key fallback).
        player_url:
            Full Transfermarkt profile URL.  If ``None`` scraping is skipped
            and the method returns ``None``.
        player_id:
            Numeric Transfermarkt player id.  Used as cache key when given;
            otherwise derived from *player_url* or *player_name*.

        Returns
        -------
        dict | None
            Dict with keys ``name, age, position, current_club, league,
            market_value_scraped, contract_expiry, scraped_at, source``
            — or ``None`` if scraping fails for any reason.
        """
        try:
            # Derive a stable cache key
            cache_key = self._resolve_cache_key(player_name, player_url, player_id)

            # Check cache first
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info("Cache hit for %s (key=%s)", player_name, cache_key)
                return cached

            if player_url is None:
                logger.warning(
                    "No URL provided for %s — cannot scrape.", player_name
                )
                return None

            # Scrape
            data = self._scrape_transfermarkt(player_url)
            if data is None:
                return None

            # Persist to cache
            self._save_cache(cache_key, data)
            return data

        except Exception:
            logger.exception("Unexpected error in scrape_player for %s", player_name)
            return None

    # ------------------------------------------------------------------
    # Scraping internals
    # ------------------------------------------------------------------

    def _scrape_transfermarkt(self, url: str) -> Optional[Dict]:
        """Parse a Transfermarkt player profile page.

        Returns a normalised dict or ``None`` on any failure.
        """
        try:
            self._rate_limit()

            # Add a small random jitter (1-3 s) on top of the rate limit
            time.sleep(random.uniform(1.0, 3.0))

            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            data: Dict = {
                "name": self._extract_name(soup),
                "age": self._extract_age(soup),
                "position": self._extract_position(soup),
                "current_club": self._extract_club(soup),
                "league": self._extract_league(soup),
                "market_value_scraped": self._extract_market_value(soup),
                "contract_expiry": self._extract_contract_expiry(soup),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "source": "transfermarkt",
            }

            logger.info("Successfully scraped %s from %s", data.get("name"), url)
            return data

        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s", url)
            return None
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP %s for %s", exc.response.status_code, url)
            return None
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error for %s: %s", url, exc)
            return None
        except Exception:
            logger.exception("Parse error for %s", url)
            return None

    # ------------------------------------------------------------------
    # HTML field extractors — each returns a value or None
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_name(soup: BeautifulSoup) -> Optional[str]:
        """Extract player name from the profile header."""
        try:
            # <h1 class="data-header__headline-wrapper"> … <strong>Name</strong></h1>
            tag = soup.select_one("h1.data-header__headline-wrapper")
            if tag:
                strong = tag.find("strong")
                if strong:
                    return strong.get_text(strip=True)
                return tag.get_text(" ", strip=True)

            # Fallback: <title> usually contains the player name
            title = soup.find("title")
            if title:
                return title.get_text(strip=True).split("-")[0].strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_age(soup: BeautifulSoup) -> Optional[int]:
        """Extract age (integer) from the info table."""
        try:
            # The info-table contains rows; age is usually shown as "(XX)"
            info = soup.select_one("span.info-table__content--bold")
            if info:
                text = info.get_text(strip=True)
                match = re.search(r"\((\d{1,2})\)", text)
                if match:
                    return int(match.group(1))

            # Broader search across the whole page for an age pattern
            for li in soup.select("li.data-header__detail, span.data-header__content"):
                text = li.get_text(strip=True)
                match = re.search(r"\((\d{1,2})\)", text)
                if match:
                    return int(match.group(1))

            # Try the info-table rows
            for row in soup.select("span.info-table__content"):
                text = row.get_text(strip=True)
                match = re.search(r"\((\d{1,2})\)", text)
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_position(soup: BeautifulSoup) -> Optional[str]:
        """Extract playing position."""
        try:
            # Look for the position detail in the data-header or info-table
            for li in soup.select("li.data-header__detail"):
                label = li.get_text(" ", strip=True)
                if "Position:" in label or "position" in label.lower():
                    # The value is typically after the label text
                    span = li.find("span", class_="data-header__content")
                    if span:
                        return span.get_text(strip=True)
                    parts = label.split(":")
                    if len(parts) > 1:
                        return parts[-1].strip()

            # info-table fallback
            for row in soup.select("tr, li"):
                text = row.get_text(" ", strip=True)
                if "Position:" in text:
                    parts = text.split("Position:")
                    if len(parts) > 1:
                        return parts[-1].strip().split("\n")[0].strip()
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_club(soup: BeautifulSoup) -> Optional[str]:
        """Extract current club name."""
        try:
            # data-header span with "Current club"
            for item in soup.select("span.data-header__club"):
                a_tag = item.find("a")
                if a_tag:
                    return a_tag.get_text(strip=True)
                return item.get_text(strip=True)

            # Fallback: info-table
            for row in soup.select("tr, li"):
                text = row.get_text(" ", strip=True)
                if "Current club" in text or "current club" in text.lower():
                    a_tag = row.find("a")
                    if a_tag:
                        return a_tag.get_text(strip=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_league(soup: BeautifulSoup) -> Optional[str]:
        """Extract the league of the current club."""
        try:
            # The league usually appears in data-header__league
            tag = soup.select_one("span.data-header__league a")
            if tag:
                return tag.get_text(strip=True)

            # Fallback: look for a league link near the club info
            for row in soup.select("tr, li"):
                text = row.get_text(" ", strip=True)
                if "League" in text or "league" in text.lower():
                    a_tag = row.find("a")
                    if a_tag:
                        return a_tag.get_text(strip=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_market_value(soup: BeautifulSoup) -> Optional[str]:
        """Extract current market value as a raw string (e.g. '€180.00m')."""
        try:
            # Primary: .tm-market-value or waarde element
            for selector in (
                "a.data-header__market-value-wrapper",
                ".tm-market-value",
                "[data-market-value]",
                ".waarde",
            ):
                tag = soup.select_one(selector)
                if tag:
                    text = tag.get_text(" ", strip=True)
                    # Clean up the text to just keep the value
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        return text

            # Fallback: search for €-pattern anywhere in data-header
            header = soup.select_one(".data-header")
            if header:
                text = header.get_text(" ", strip=True)
                match = re.search(r"€[\d.,]+[mkMK]?", text)
                if match:
                    return match.group(0)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_contract_expiry(soup: BeautifulSoup) -> Optional[str]:
        """Extract contract expiration date as a string."""
        try:
            # Look for "Contract expires" in data-header details
            for li in soup.select("li.data-header__detail, span.data-header__content"):
                label_text = ""
                parent = li.parent if li.name == "span" else li
                label_text = parent.get_text(" ", strip=True)
                if "contract" in label_text.lower() and (
                    "expires" in label_text.lower() or "expir" in label_text.lower()
                ):
                    span = li if li.name == "span" else li.find(
                        "span", class_="data-header__content"
                    )
                    if span:
                        return span.get_text(strip=True)

            # info-table fallback
            for row in soup.select("tr, li"):
                text = row.get_text(" ", strip=True)
                if "Contract expires" in text or "contract expires" in text.lower():
                    # Try to pull out a date pattern
                    match = re.search(
                        r"(\w{3}\s+\d{1,2},\s+\d{4}|\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{2}-\d{2})",
                        text,
                    )
                    if match:
                        return match.group(1)
                    # Return whatever comes after the label
                    parts = re.split(r"[Cc]ontract\s+expires:?\s*", text)
                    if len(parts) > 1:
                        return parts[-1].strip()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Rate-limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Block until at least MIN_REQUEST_INTERVAL seconds have passed
        since the last outgoing request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Disk cache helpers
    # ------------------------------------------------------------------

    def _resolve_cache_key(
        self,
        player_name: str,
        player_url: Optional[str],
        player_id: Optional[int],
    ) -> str:
        """Return a stable string key for the cache file."""
        if player_id is not None:
            return str(player_id)
        if player_url:
            # Try to extract the numeric id from the URL
            match = re.search(r"/spieler/(\d+)", player_url)
            if match:
                return match.group(1)
        # Fallback: slugified player name
        slug = re.sub(r"[^a-z0-9]+", "_", player_name.lower()).strip("_")
        return slug

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"player_{key}.json"

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load cached data if it exists and is younger than CACHE_MAX_AGE."""
        try:
            path = self._cache_path(key)
            if not path.exists():
                return None

            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if datetime.now(timezone.utc) - mtime > CACHE_MAX_AGE:
                logger.debug("Cache stale for key=%s, will refresh.", key)
                return None

            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logger.debug("Failed to load cache for key=%s", key, exc_info=True)
            return None

    def _save_cache(self, key: str, data: Dict) -> None:
        """Persist scraped data to disk as JSON."""
        try:
            path = self._cache_path(key)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            logger.debug("Cached data written to %s", path)
        except Exception:
            logger.debug("Failed to write cache for key=%s", key, exc_info=True)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    scraper = PlayerStatsScraper()

    test_url = (
        "https://www.transfermarkt.co.uk/erling-haaland/profil/spieler/418560"
    )
    print(f"\nScraping Erling Haaland from:\n  {test_url}\n")

    result = scraper.scrape_player(
        player_name="Erling Haaland",
        player_url=test_url,
        player_id=418560,
    )

    if result is None:
        print("Scraping returned None (expected if blocked / no network).")
        print("The main app will fall back to local dataset stats.")
    else:
        print("Scraped data:")
        for k, v in result.items():
            print(f"  {k:25s}: {v}")
