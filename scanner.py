"""
scanner.py — Fetches live market data from CoinGecko (free tier).
Paginates requests to scan up to COINS_PER_CYCLE coins per cycle.
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional

from config import (
    COINGECKO_BASE_URL, COINGECKO_API_KEY,
    COINS_PER_CYCLE, COINGECKO_PAGE_SIZE,
)
from signals import MarketSnapshot

logger = logging.getLogger(__name__)

_HEADERS: Dict[str, str] = {"accept": "application/json"}
if COINGECKO_API_KEY:
    _HEADERS["x-cg-demo-api-key"] = COINGECKO_API_KEY

REQUEST_DELAY = 2.5   # seconds — keeps us within free-tier rate limits


def _get(endpoint: str, params: dict, retries: int = 3) -> Optional[Any]:
    url = f"{COINGECKO_BASE_URL}{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                logger.warning("Rate-limited by CoinGecko. Sleeping 60s …")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("Request error attempt %d/%d: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


def fetch_market_data(coins_to_fetch: int = COINS_PER_CYCLE) -> List[MarketSnapshot]:
    """
    Return a list of MarketSnapshot for the top coins by market cap.
    Paginates automatically until we hit coins_to_fetch.
    """
    results: List[MarketSnapshot] = []
    page   = 1
    fetched = 0

    while fetched < coins_to_fetch:
        per_page = min(COINGECKO_PAGE_SIZE, coins_to_fetch - fetched)
        params = {
            "vs_currency":                        "usd",
            "order":                              "market_cap_desc",
            "per_page":                           per_page,
            "page":                               page,
            "sparkline":                          "false",
            "price_change_percentage":            "24h,7d",
        }
        data = _get("/coins/markets", params)
        if not data:
            logger.warning("Empty response on page %d — stopping pagination.", page)
            break

        for coin in data:
            try:
                snap = MarketSnapshot(
                    id            = coin.get("id", ""),
                    symbol        = coin.get("symbol", "").upper(),
                    name          = coin.get("name", ""),
                    current_price = float(coin.get("current_price") or 0),
                    market_cap    = float(coin.get("market_cap") or 0),
                    total_volume  = float(coin.get("total_volume") or 0),
                    price_change_24h = float(coin.get("price_change_percentage_24h") or 0),
                    price_change_7d  = float(
                        coin.get("price_change_percentage_7d_in_currency") or 0
                    ),
                    high_24h      = float(coin.get("high_24h") or 0),
                    low_24h       = float(coin.get("low_24h") or 0),
                )
                results.append(snap)
            except (TypeError, ValueError) as exc:
                logger.debug("Skipping malformed coin entry: %s", exc)

        fetched += len(data)
        logger.info("Page %d → fetched %d / %d coins", page, fetched, coins_to_fetch)

        if len(data) < per_page:
            break   # CoinGecko returned fewer rows than requested → last page

        page += 1
        time.sleep(REQUEST_DELAY)

    logger.info("Scan complete — %d coins fetched.", len(results))
    return results


def fetch_coin_history(coin_id: str, days: int = 30) -> List[Dict]:
    """
    Fetch daily OHLCV history for a single coin (used by backtester).
    Returns list of [timestamp_ms, open, high, low, close, volume] rows.
    """
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    data = _get(f"/coins/{coin_id}/ohlc", params)
    if not data:
        return []
    # CoinGecko OHLC returns: [timestamp, open, high, low, close]
    # Volume not included in free OHLC — we attach None
    return [
        {"timestamp": row[0], "open": row[1], "high": row[2],
         "low": row[3], "close": row[4], "volume": None}
        for row in data
    ]
