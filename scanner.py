"""
scanner.py — Fetches live market data from CoinGecko + CoinDCX (both free).
"""

import logging
import time
from typing import List, Dict, Any, Optional

import requests

from config import (
    COINGECKO_BASE_URL, COINGECKO_API_KEY,
    COINS_PER_CYCLE,
    COINGECKO_PAGE_SIZE,
    FUTURES_CACHE_TTL_SECONDS,
    COINDCX_DETAILS_CACHE_TTL_SECONDS,
    COINDCX_FUTURES_ONLY,
    COINDCX_PREFERRED_MARGIN_ASSET,
    COINDCX_PRICE_CACHE_TTL_SECONDS,
    COINDCX_UNIVERSE_CACHE_TTL_SECONDS,
)
from signals import MarketSnapshot

logger = logging.getLogger(__name__)

_CG_HEADERS: Dict[str, str] = {"accept": "application/json"}
if COINGECKO_API_KEY:
    _CG_HEADERS["x-cg-demo-api-key"] = COINGECKO_API_KEY

COINDCX_BASE_URL = "https://api.coindcx.com"
COINDCX_PUBLIC_BASE_URL = "https://public.coindcx.com"
REQUEST_DELAY    = 2.5
COINGECKO_SYMBOLS_CHUNK_SIZE = 50
_MARKET_CHART_CACHE: Dict[str, Dict[str, Any]] = {}
_MARKET_CHART_TTL_SECONDS = 15 * 60
_DERIVATIVES_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "payload": []}
_COINDCX_UNIVERSE_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "payload": {"spot": {}, "futures": []}}
_COINDCX_FUTURES_DETAILS_CACHE: Dict[str, Dict[str, Any]] = {}
_COINDCX_FUTURES_PRICES_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "payload": {}}


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


def _coindcx_get(endpoint: str, params: Optional[dict] = None, retries: int = 3) -> Optional[Any]:
    return _coindcx_json_get(COINDCX_BASE_URL, endpoint, params=params, retries=retries, label="CoinDCX")


def _coindcx_public_get(endpoint: str, params: Optional[dict] = None, retries: int = 3) -> Optional[Any]:
    return _coindcx_json_get(
        COINDCX_PUBLIC_BASE_URL,
        endpoint,
        params=params,
        retries=retries,
        label="CoinDCX public",
    )


def _coindcx_json_get(
    base_url: str,
    endpoint: str,
    params: Optional[dict] = None,
    retries: int = 3,
    label: str = "CoinDCX",
) -> Optional[Any]:
    url = f"{base_url}{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url,
                headers={"accept": "application/json"},
                params=params or {},
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("%s error attempt %d/%d for %s: %s", label, attempt, retries, endpoint, exc)
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


def _normalise_symbol(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def _chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _extract_payload_rows(payload: Any) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "result", "rows", "items", "active_instruments", "instruments", "pairs"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    return []


def _parse_instrument_name(instrument_name: str) -> Dict[str, str]:
    raw = str(instrument_name or "").strip().upper()
    if not raw:
        return {
            "instrument_name": "",
            "underlying_symbol": "",
            "margin_asset": "",
            "quote_asset": "",
            "contract_type": "",
        }

    canonical = raw.replace("/", "_")
    left = canonical
    margin_asset = ""
    if "_" in canonical:
        left, margin_part = canonical.rsplit("_", 1)
        margin_asset = _normalise_symbol(margin_part)

    left_parts = [part for part in left.split("-") if part]
    base_segment = left
    if left_parts:
        base_segment = left_parts[-1]
        if base_segment in {"PERP", "PERPETUAL"} and len(left_parts) >= 2:
            base_segment = left_parts[-2]

    base_symbol = _normalise_symbol(base_segment)
    underlying = base_symbol
    quote_asset = margin_asset
    for suffix in ("USDT", "USD", "INR", "BTC", "ETH"):
        if base_symbol.endswith(suffix) and len(base_symbol) > len(suffix) + 1:
            underlying = base_symbol[: -len(suffix)]
            quote_asset = quote_asset or suffix
            break

    if margin_asset and underlying.endswith(margin_asset) and len(underlying) > len(margin_asset) + 1:
        underlying = underlying[: -len(margin_asset)]

    contract_type = "PERPETUAL" if any(token in canonical for token in ("PERP", "PERPETUAL")) else ""
    return {
        "instrument_name": canonical,
        "underlying_symbol": _normalise_symbol(underlying),
        "margin_asset": margin_asset,
        "quote_asset": quote_asset,
        "contract_type": contract_type,
    }


def _extract_underlying_symbol(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("underlying_currency_short_name"),
        row.get("base_currency_short_name"),
        row.get("base_asset"),
        row.get("underlying_asset"),
        row.get("index_id"),
    ]
    for candidate in candidates:
        normalised = _normalise_symbol(str(candidate or ""))
        if normalised and normalised not in {"USDT", "USD", "INR"}:
            return normalised

    instrument_name = str(
        row.get("instrument_name")
        or row.get("pair")
        or row.get("instrument")
        or row.get("name")
        or row.get("symbol")
        or ""
    )
    return _parse_instrument_name(instrument_name).get("underlying_symbol", "")


def _extract_margin_asset(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("margin_currency_short_name"),
        row.get("settlement_currency_short_name"),
        row.get("quote_currency_short_name"),
        row.get("margin_asset"),
        row.get("quote_asset"),
    ]
    for candidate in candidates:
        normalised = _normalise_symbol(str(candidate or ""))
        if normalised:
            return normalised

    instrument_name = str(
        row.get("instrument_name")
        or row.get("pair")
        or row.get("instrument")
        or row.get("name")
        or row.get("symbol")
        or ""
    )
    return _parse_instrument_name(instrument_name).get("margin_asset", "")


def _extract_quote_asset(row: Dict[str, Any]) -> str:
    candidates = [
        row.get("quote_currency_short_name"),
        row.get("target_currency_short_name"),
        row.get("settlement_currency_short_name"),
        row.get("quote_asset"),
    ]
    for candidate in candidates:
        normalised = _normalise_symbol(str(candidate or ""))
        if normalised:
            return normalised

    instrument_name = str(
        row.get("instrument_name")
        or row.get("pair")
        or row.get("instrument")
        or row.get("name")
        or row.get("symbol")
        or ""
    )
    return _parse_instrument_name(instrument_name).get("quote_asset", "")


def _pick_best_futures_instrument(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def rank(row: Dict[str, Any]) -> tuple:
        margin_asset = str(row.get("margin_asset") or "").upper()
        instrument_name = str(row.get("instrument_name") or "")
        return (
            1 if margin_asset == COINDCX_PREFERRED_MARGIN_ASSET else 0,
            1 if "PERP" in instrument_name.upper() or "PERPETUAL" in instrument_name.upper() else 0,
            len(instrument_name),
        )

    return sorted(rows, key=rank, reverse=True)[0]


def _build_snapshot_from_coingecko_row(coin: Dict[str, Any]) -> Optional[MarketSnapshot]:
    try:
        return MarketSnapshot(
            id=coin.get("id", ""),
            symbol=coin.get("symbol", "").upper(),
            name=coin.get("name", ""),
            current_price=float(coin.get("current_price") or 0),
            market_cap=float(coin.get("market_cap") or 0),
            total_volume=float(coin.get("total_volume") or 0),
            price_change_24h=float(coin.get("price_change_percentage_24h") or 0),
            price_change_7d=float(coin.get("price_change_percentage_7d_in_currency") or 0),
            high_24h=float(coin.get("high_24h") or 0),
            low_24h=float(coin.get("low_24h") or 0),
        )
    except (TypeError, ValueError):
        return None


def _relative_price_gap(price_a: float, price_b: float) -> float:
    if price_a <= 0 or price_b <= 0:
        return float("inf")
    baseline = max(abs(price_a), abs(price_b), 1e-9)
    return abs(price_a - price_b) / baseline


def _pick_best_coingecko_candidate(
    symbol: str,
    rows: List[Dict[str, Any]],
    preferred_price: float = 0.0,
) -> Optional[Dict[str, Any]]:
    if not rows:
        return None

    symbol_key = _normalise_symbol(symbol)
    exact_rows = [
        row for row in rows
        if _normalise_symbol(str(row.get("symbol") or "")) == symbol_key
    ] or rows

    if preferred_price > 0:
        ranked = sorted(
            exact_rows,
            key=lambda row: (
                _relative_price_gap(_coerce_float(row.get("current_price")), preferred_price),
                -_coerce_float(row.get("market_cap")),
                -_coerce_float(row.get("total_volume")),
            ),
        )
        winner = ranked[0]
        if _relative_price_gap(_coerce_float(winner.get("current_price")), preferred_price) <= 0.35:
            return winner

    return sorted(
        exact_rows,
        key=lambda row: (
            -_coerce_float(row.get("market_cap")),
            -_coerce_float(row.get("total_volume")),
            str(row.get("id") or ""),
        ),
    )[0]


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
            snap = _build_snapshot_from_coingecko_row(coin)
            if snap:
                results.append(snap)

        fetched += len(data)
        logger.info("[CoinGecko] Page %d fetched %d/%d coins", page, fetched, coins_to_fetch)

        if len(data) < per_page:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    return results


def fetch_coingecko_data_for_symbols(
    symbols: List[str],
    preferred_prices: Optional[Dict[str, float]] = None,
) -> List[MarketSnapshot]:
    requested_symbols = []
    seen = set()
    for symbol in symbols:
        normalised = _normalise_symbol(symbol)
        if not normalised or normalised in seen:
            continue
        seen.add(normalised)
        requested_symbols.append(normalised)

    if not requested_symbols:
        return []

    preferred_prices = {(_normalise_symbol(key)): float(value or 0.0) for key, value in (preferred_prices or {}).items()}
    results: List[MarketSnapshot] = []
    chunk_count = 0
    for chunk in _chunked(requested_symbols, COINGECKO_SYMBOLS_CHUNK_SIZE):
        params = {
            "vs_currency": "usd",
            "symbols": ",".join(symbol.lower() for symbol in chunk),
            "include_tokens": "all",
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
        }
        data = _cg_get("/coins/markets", params)
        if not isinstance(data, list):
            continue

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for coin in data:
            coin_symbol = _normalise_symbol(str(coin.get("symbol") or ""))
            if coin_symbol:
                grouped.setdefault(coin_symbol, []).append(coin)

        for symbol in chunk:
            choice = _pick_best_coingecko_candidate(
                symbol,
                grouped.get(symbol, []),
                preferred_price=preferred_prices.get(symbol, 0.0),
            )
            if not choice:
                continue
            snap = _build_snapshot_from_coingecko_row(choice)
            if snap:
                results.append(snap)

        chunk_count += 1
        logger.info(
            "[CoinGecko] Symbol batch %d fetched %d/%d requested CoinDCX symbols",
            chunk_count,
            len(results),
            len(requested_symbols),
        )
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


def fetch_coindcx_spot_markets() -> Dict[str, str]:
    cached = _COINDCX_UNIVERSE_CACHE.get("payload") or {}
    now = time.time()
    if cached and now - float(_COINDCX_UNIVERSE_CACHE.get("fetched_at") or 0.0) < COINDCX_UNIVERSE_CACHE_TTL_SECONDS:
        return dict(cached.get("spot") or {})

    spot_map: Dict[str, str] = {}
    tickers = _coindcx_get("/exchange/ticker") or []
    for ticker in tickers:
        market = str(ticker.get("market") or "")
        if not market:
            continue
        symbol = market.replace("USDT", "").replace("INR", "").upper()
        if not symbol:
            continue
        if symbol not in spot_map:
            spot_map[symbol] = market
    _COINDCX_UNIVERSE_CACHE["payload"] = {
        **(_COINDCX_UNIVERSE_CACHE.get("payload") or {}),
        "spot": spot_map,
    }
    _COINDCX_UNIVERSE_CACHE["fetched_at"] = now
    return spot_map


def fetch_derivatives_tickers() -> List[Dict[str, Any]]:
    """
    Fetch active CoinDCX futures instruments.

    The rest of the agent uses this as its CoinDCX futures universe rather than
    generic derivatives listings from third-party exchanges.
    """
    now = time.time()
    cached = _COINDCX_UNIVERSE_CACHE.get("payload") or {}
    if cached and now - float(_COINDCX_UNIVERSE_CACHE.get("fetched_at") or 0.0) < COINDCX_UNIVERSE_CACHE_TTL_SECONDS:
        futures_rows = list(cached.get("futures") or [])
        if futures_rows:
            return futures_rows

    query_params = {"margin_currency_short_name[]": COINDCX_PREFERRED_MARGIN_ASSET} if COINDCX_PREFERRED_MARGIN_ASSET else None
    data = _coindcx_get("/exchange/v1/derivatives/futures/data/active_instruments", query_params)
    rows = _extract_payload_rows(data)
    if not rows and query_params:
        data = _coindcx_get("/exchange/v1/derivatives/futures/data/active_instruments")
        rows = _extract_payload_rows(data)
    if not rows:
        return list((_COINDCX_UNIVERSE_CACHE.get("payload") or {}).get("futures") or [])

    futures_rows: List[Dict[str, Any]] = []
    for raw_row in rows:
        if isinstance(raw_row, str):
            row = {"instrument_name": raw_row}
        elif isinstance(raw_row, dict):
            row = raw_row
        else:
            continue

        parsed = _parse_instrument_name(
            str(
                row.get("instrument_name")
                or row.get("pair")
                or row.get("instrument")
                or row.get("symbol")
                or row.get("name")
                or ""
            )
        )
        instrument_name = str(
            parsed.get("instrument_name")
            or row.get("instrument_name")
            or row.get("pair")
            or row.get("instrument")
            or row.get("symbol")
            or row.get("name")
            or ""
        )
        if not instrument_name:
            continue
        futures_rows.append(
            {
                "instrument_name": instrument_name,
                "underlying_symbol": _extract_underlying_symbol(row) or str(parsed.get("underlying_symbol") or ""),
                "margin_asset": _extract_margin_asset(row) or str(parsed.get("margin_asset") or ""),
                "quote_asset": _extract_quote_asset(row) or str(parsed.get("quote_asset") or ""),
                "contract_type": str(
                    row.get("contract_type")
                    or row.get("instrument_type")
                    or parsed.get("contract_type")
                    or ""
                ),
                "raw": row,
            }
        )

    _COINDCX_UNIVERSE_CACHE["payload"] = {
        "spot": dict((_COINDCX_UNIVERSE_CACHE.get("payload") or {}).get("spot") or {}),
        "futures": futures_rows,
    }
    _COINDCX_UNIVERSE_CACHE["fetched_at"] = now
    logger.info("[CoinDCX] Fetched %d active futures instruments", len(futures_rows))
    return futures_rows


def fetch_coindcx_instrument_details(instrument_name: str) -> Dict[str, Any]:
    if not instrument_name:
        return {}

    cached = _COINDCX_FUTURES_DETAILS_CACHE.get(instrument_name)
    now = time.time()
    if cached and now - float(cached.get("fetched_at") or 0.0) < COINDCX_DETAILS_CACHE_TTL_SECONDS:
        return dict(cached.get("payload") or {})

    parsed = _parse_instrument_name(instrument_name)
    payload = _coindcx_get(
        "/exchange/v1/derivatives/futures/data/instrument",
        {
            "pair": parsed.get("instrument_name") or instrument_name,
            "margin_currency_short_name": parsed.get("margin_asset") or COINDCX_PREFERRED_MARGIN_ASSET,
        },
    )
    if not payload:
        payload = _coindcx_get(
            "/exchange/v1/derivatives/futures/data/instrument_details",
            {"instrument_name": instrument_name},
        )
    if not isinstance(payload, dict):
        return {}

    raw = payload.get("instrument") if isinstance(payload.get("instrument"), dict) else payload.get("data")
    if not isinstance(raw, dict):
        raw = payload

    details = {
        "instrument_name": instrument_name,
        "max_leverage_long": _coerce_float(
            raw.get("max_leverage_long")
            or raw.get("maximum_leverage")
            or raw.get("max_leverage")
            or raw.get("leverage")
        ),
        "max_leverage_short": _coerce_float(
            raw.get("max_leverage_short")
            or raw.get("maximum_leverage")
            or raw.get("max_leverage")
            or raw.get("leverage")
        ),
        "dynamic_position_leverage_details": raw.get("dynamic_position_leverage_details") or [],
        "funding_rate": _coerce_float(raw.get("funding_rate") or raw.get("funding")),
        "open_interest": _coerce_float(raw.get("open_interest") or raw.get("oi")),
        "best_bid": _coerce_float(raw.get("best_bid") or raw.get("bid")),
        "best_ask": _coerce_float(raw.get("best_ask") or raw.get("ask")),
        "last_price": _coerce_float(raw.get("last_price") or raw.get("mark_price") or raw.get("price")),
        "raw": raw,
    }
    _COINDCX_FUTURES_DETAILS_CACHE[instrument_name] = {
        "fetched_at": now,
        "payload": details,
    }
    return details


def fetch_coindcx_futures_prices(instrument_names: List[str]) -> Dict[str, Dict[str, float]]:
    unique_names = sorted({name for name in instrument_names if name})
    if not unique_names:
        return {}

    now = time.time()
    cached_payload = _COINDCX_FUTURES_PRICES_CACHE.get("payload") or {}
    if cached_payload and now - float(_COINDCX_FUTURES_PRICES_CACHE.get("fetched_at") or 0.0) < COINDCX_PRICE_CACHE_TTL_SECONDS:
        cached_lookup = {_normalise_symbol(key): key for key in cached_payload}
        cached_matches = {}
        for name in unique_names:
            source_key = name if name in cached_payload else cached_lookup.get(_normalise_symbol(name))
            if source_key and source_key in cached_payload:
                cached_matches[name] = cached_payload[source_key]
        if len(cached_matches) == len(unique_names):
            return cached_matches

    prices: Dict[str, Dict[str, float]] = {}
    requested_lookup = {_normalise_symbol(name): name for name in unique_names}

    payload = _coindcx_public_get("/market_data/v3/current_prices/futures/rt")
    public_prices = payload.get("prices") if isinstance(payload, dict) else None
    if isinstance(public_prices, dict):
        for source_name, row in public_prices.items():
            last_price = _coerce_float(row.get("last_price") if isinstance(row, dict) else row)
            prices[source_name] = {
                "last_price": last_price,
                "best_bid": _coerce_float(row.get("best_bid") if isinstance(row, dict) else 0.0),
                "best_ask": _coerce_float(row.get("best_ask") if isinstance(row, dict) else 0.0),
                "spread": _coerce_float(row.get("spread") if isinstance(row, dict) else 0.0),
                "funding_rate": _coerce_float(
                    row.get("funding_rate") if isinstance(row, dict) else 0.0
                ),
                "open_interest": _coerce_float(
                    row.get("open_interest") if isinstance(row, dict) else 0.0
                ),
                "volume_24h": _coerce_float(row.get("volume_24h") if isinstance(row, dict) else 0.0),
            }

    if not any(_normalise_symbol(name) in requested_lookup for name in prices):
        payload = _coindcx_get(
            "/exchange/v1/derivatives/futures/data/current_prices",
            {"instrument_names": ",".join(unique_names)},
        )
        rows = _extract_payload_rows(payload)
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(
                row.get("instrument_name")
                or row.get("pair")
                or row.get("instrument")
                or row.get("symbol")
                or row.get("name")
                or ""
            )
            requested_name = requested_lookup.get(_normalise_symbol(name))
            if not requested_name:
                continue
            bid = _coerce_float(row.get("best_bid"))
            ask = _coerce_float(row.get("best_ask"))
            last_price = _coerce_float(row.get("last_price") or row.get("mark_price") or row.get("price"))
            spread = 0.0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                if mid > 0:
                    spread = ((ask - bid) / mid) * 100.0
            prices[requested_name] = {
                "last_price": last_price,
                "best_bid": bid,
                "best_ask": ask,
                "spread": spread,
                "funding_rate": _coerce_float(row.get("funding_rate")),
                "open_interest": _coerce_float(row.get("open_interest")),
                "volume_24h": _coerce_float(row.get("volume_24h")),
            }

    _COINDCX_FUTURES_PRICES_CACHE["fetched_at"] = now
    _COINDCX_FUTURES_PRICES_CACHE["payload"] = prices
    source_lookup = {_normalise_symbol(key): key for key in prices}
    selected: Dict[str, Dict[str, float]] = {}
    for name in unique_names:
        source_key = name if name in prices else source_lookup.get(_normalise_symbol(name))
        if source_key and source_key in prices:
            selected[name] = prices[source_key]
    return selected


def fetch_market_data(coins_to_fetch: int = COINS_PER_CYCLE) -> List[MarketSnapshot]:
    logger.info("Loading CoinDCX tradable universe ...")
    spot_markets = fetch_coindcx_spot_markets()
    futures_rows = fetch_derivatives_tickers()
    dcx_spot_snaps = fetch_coindcx_data()
    dcx_spot_by_symbol = {snap.symbol: snap for snap in dcx_spot_snaps}

    futures_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for row in futures_rows:
        symbol = str(row.get("underlying_symbol") or "").upper()
        if not symbol:
            continue
        futures_by_symbol.setdefault(symbol, []).append(row)

    eligible_symbols = set(futures_by_symbol) if COINDCX_FUTURES_ONLY else set(spot_markets) | set(futures_by_symbol)
    ranked_symbols = sorted(
        eligible_symbols,
        key=lambda symbol: (
            -(1 if symbol in futures_by_symbol else 0),
            -_coerce_float(dcx_spot_by_symbol.get(symbol).total_volume if symbol in dcx_spot_by_symbol else 0.0),
            -_coerce_float(dcx_spot_by_symbol.get(symbol).current_price if symbol in dcx_spot_by_symbol else 0.0),
            symbol,
        ),
    )
    selected_symbols = ranked_symbols[:coins_to_fetch] if coins_to_fetch > 0 else ranked_symbols
    logger.info(
        "Scanning CoinGecko for CoinDCX universe (%d selected / %d eligible symbols) ...",
        len(selected_symbols),
        len(eligible_symbols),
    )
    cg_snaps = fetch_coingecko_data_for_symbols(
        selected_symbols,
        preferred_prices={symbol: snap.current_price for symbol, snap in dcx_spot_by_symbol.items() if symbol in eligible_symbols},
    )
    cg_symbols = {s.symbol for s in cg_snaps}

    for snap in cg_snaps:
        snap.coindcx_spot_market = spot_markets.get(snap.symbol, "")
        symbol_futures = futures_by_symbol.get(snap.symbol, [])
        if symbol_futures:
            best = _pick_best_futures_instrument(symbol_futures)
            snap.coindcx_has_futures = True
            snap.coindcx_futures_instrument = str(best.get("instrument_name") or "")
            snap.coindcx_margin_asset = str(best.get("margin_asset") or "")

    added = 0
    for symbol in selected_symbols:
        if symbol in cg_symbols:
            continue
        snap = dcx_spot_by_symbol.get(symbol)
        if not snap:
            continue
        if snap.symbol in cg_symbols:
            continue
        if snap.symbol not in eligible_symbols:
            continue

        snap.coindcx_spot_market = spot_markets.get(snap.symbol, "")
        symbol_futures = futures_by_symbol.get(snap.symbol, [])
        if symbol_futures:
            best = _pick_best_futures_instrument(symbol_futures)
            snap.coindcx_has_futures = True
            snap.coindcx_futures_instrument = str(best.get("instrument_name") or "")
            snap.coindcx_margin_asset = str(best.get("margin_asset") or "")

        cg_snaps.append(snap)
        cg_symbols.add(snap.symbol)
        added += 1

    eligible_count = sum(
        1
        for snap in cg_snaps
        if (snap.coindcx_has_futures if COINDCX_FUTURES_ONLY else (snap.coindcx_spot_market or snap.coindcx_has_futures))
    )
    logger.info(
        "Total scanned=%d | CoinDCX eligible=%d | CoinDCX futures underlyings=%d | mode=%s | extras_added=%d",
        len(cg_snaps),
        eligible_count,
        len(futures_by_symbol),
        "futures_only" if COINDCX_FUTURES_ONLY else "spot_and_futures",
        added,
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
