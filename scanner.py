"""
scanner.py — Fetches live market data from CoinGecko + CoinDCX (both free).
"""

import time
import logging
import requests
from typing import List, Dict, Any, Optional

from config import (
    COINGECKO_BASE_URL, COINGECKO_API_KEY,
    COINS_PER_CYCLE, COINGECKO_PAGE_SIZE, FUTURES_CACHE_TTL_SECONDS,
)
from signals import MarketSnapshot

logger = logging.getLogger(__name__)

_CG_HEADERS: Dict[str, str] = {"accept": "application/json"}
if COINGECKO_API_KEY:
    _CG_HEADERS["x-cg-demo-api-key"] = COINGECKO_API_KEY

COINDCX_BASE_URL = "https://api.coindcx.com"
REQUEST_DELAY    = 2.5
_MARKET_CHART_CACHE: Dict[str, Dict[str, Any]] = {}
_MARKET_CHART_TTL_SECONDS = 15 * 60
_DERIVATIVES_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "payload": []}


def _coerce_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cg_get(endpoint: str, params: dict, retries: int = 3) -> Optional[Any]:
    url = f"{COINGECKO_BASE_URL}{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=_CG_HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                logger.warning("Rate-limited by CoinGecko. Sleeping 60s ...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("CoinGecko error attempt %d/%d: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


def fetch_coingecko_data(coins_to_fetch: int = COINS_PER_CYCLE) -> List[MarketSnapshot]:
    results: List[MarketSnapshot] = []
    page    = 1
    fetched = 0

    while fetched < coins_to_fetch:
        per_page = min(COINGECKO_PAGE_SIZE, coins_to_fetch - fetched)
        params = {
            "vs_currency":             "usd",
            "order":                   "market_cap_desc",
            "per_page":                per_page,
            "page":                    page,
            "sparkline":               "false",
            "price_change_percentage": "24h,7d",
        }
        data = _cg_get("/coins/markets", params)
        if not data:
            break

        for coin in data:
            try:
                snap = MarketSnapshot(
                    id               = coin.get("id", ""),
                    symbol           = coin.get("symbol", "").upper(),
                    name             = coin.get("name", ""),
                    current_price    = float(coin.get("current_price") or 0),
                    market_cap       = float(coin.get("market_cap") or 0),
                    total_volume     = float(coin.get("total_volume") or 0),
                    price_change_24h = float(coin.get("price_change_percentage_24h") or 0),
                    price_change_7d  = float(coin.get("price_change_percentage_7d_in_currency") or 0),
                    high_24h         = float(coin.get("high_24h") or 0),
                    low_24h          = float(coin.get("low_24h") or 0),
                )
                results.append(snap)
            except (TypeError, ValueError):
                pass

        fetched += len(data)
        logger.info("[CoinGecko] Page %d fetched %d/%d coins", page, fetched, coins_to_fetch)

        if len(data) < per_page:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return results


def fetch_coindcx_data() -> List[MarketSnapshot]:
    results: List[MarketSnapshot] = []
    try:
        resp = requests.get(
            f"{COINDCX_BASE_URL}/exchange/ticker",
            timeout=20,
            headers={"accept": "application/json"},
        )
        resp.raise_for_status()
        tickers = resp.json()

        seen_symbols = set()
        for ticker in tickers:
            market = ticker.get("market", "")
            if not market.endswith("USDT"):
                continue

            symbol = market.replace("USDT", "").upper()
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)

            try:
                last_price = float(ticker.get("last_price") or 0)
                volume     = float(ticker.get("volume") or 0)
                change_24h = float(ticker.get("change_24_hour") or 0)
                high_24h   = float(ticker.get("high") or 0)
                low_24h    = float(ticker.get("low") or 0)

                if last_price <= 0:
                    continue

                snap = MarketSnapshot(
                    id               = f"coindcx_{symbol.lower()}",
                    symbol           = symbol,
                    name             = f"{symbol} (CoinDCX)",
                    current_price    = last_price,
                    market_cap       = 0.0,
                    total_volume     = volume * last_price,
                    price_change_24h = change_24h,
                    price_change_7d  = 0.0,
                    high_24h         = high_24h,
                    low_24h          = low_24h,
                )
                results.append(snap)
            except (TypeError, ValueError):
                pass

        logger.info("[CoinDCX] Fetched %d USDT pairs", len(results))

    except Exception as exc:
        logger.error("[CoinDCX] Failed to fetch: %s", exc)

    return results


def fetch_market_data(coins_to_fetch: int = COINS_PER_CYCLE) -> List[MarketSnapshot]:
    logger.info("Scanning CoinGecko (%d coins) ...", coins_to_fetch)
    cg_snaps   = fetch_coingecko_data(coins_to_fetch)
    cg_symbols = {s.symbol for s in cg_snaps}

    logger.info("Scanning CoinDCX ...")
    dcx_snaps = fetch_coindcx_data()

    added = 0
    for snap in dcx_snaps:
        if snap.symbol not in cg_symbols:
            cg_snaps.append(snap)
            cg_symbols.add(snap.symbol)
            added += 1

    logger.info(
        "Total: %d CoinGecko + %d CoinDCX = %d coins scanned",
        len(cg_symbols) - added, added, len(cg_snaps),
    )
    return cg_snaps


def fetch_coin_market_chart(coin_id: str, days: int = 7) -> Dict[str, List[float]]:
    """
    Fetch short historical market context for a coin.

    Returns prices, volumes, and market caps from CoinGecko's market_chart endpoint.
    CoinDCX-only synthetic ids do not have historical context here.
    """
    if not coin_id or coin_id.startswith("coindcx_"):
        return {}

    cache_key = f"{coin_id}:{days}"
    cached = _MARKET_CHART_CACHE.get(cache_key)
    now = time.time()
    if cached and now - cached["fetched_at"] < _MARKET_CHART_TTL_SECONDS:
        return cached["payload"]

    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    data = _cg_get(f"/coins/{coin_id}/market_chart", params)
    if not data:
        return {}

    payload = {
        "prices": [float(row[1]) for row in data.get("prices", []) if len(row) >= 2],
        "volumes": [float(row[1]) for row in data.get("total_volumes", []) if len(row) >= 2],
        "market_caps": [float(row[1]) for row in data.get("market_caps", []) if len(row) >= 2],
    }
    _MARKET_CHART_CACHE[cache_key] = {
        "fetched_at": now,
        "payload": payload,
    }
    return payload


def fetch_derivatives_tickers() -> List[Dict[str, Any]]:
    """
    Fetch live derivatives tickers from CoinGecko and cache them briefly.

    This powers the futures confirmation layer with funding, open interest,
    basis, spread, and 24h derivatives volume when available.
    """
    now = time.time()
    cached = _DERIVATIVES_CACHE.get("payload") or []
    if cached and now - float(_DERIVATIVES_CACHE.get("fetched_at") or 0.0) < FUTURES_CACHE_TTL_SECONDS:
        return cached

    data = _cg_get("/derivatives", {})
    if not isinstance(data, list):
        return cached

    tickers: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue

        tickers.append(
            {
                "symbol": symbol,
                "index_id": str(row.get("index_id") or row.get("index_name") or "").upper(),
                "market": str(row.get("market") or row.get("name") or ""),
                "contract_type": str(row.get("contract_type") or ""),
                "price": _coerce_float(row.get("price")),
                "price_change_24h": _coerce_float(row.get("price_percentage_change_24h")),
                "funding_rate": _coerce_float(row.get("funding_rate")),
                "open_interest": _coerce_float(row.get("open_interest")),
                "volume_24h": _coerce_float(row.get("volume_24h")),
                "basis": _coerce_float(row.get("basis")),
                "spread": _coerce_float(row.get("spread")),
                "expired_at": row.get("expired_at"),
            }
        )

    _DERIVATIVES_CACHE["fetched_at"] = now
    _DERIVATIVES_CACHE["payload"] = tickers
    logger.info("[CoinGecko] Fetched %d derivative tickers", len(tickers))
    return tickers


def fetch_coin_history(coin_id: str, days: int = 30) -> List[Dict]:
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    data   = _cg_get(f"/coins/{coin_id}/ohlc", params)
    if not data:
        return []
    return [
        {"timestamp": row[0], "open": row[1], "high": row[2],
         "low": row[3], "close": row[4], "volume": None}
        for row in data
    ]
