import unittest
from unittest.mock import patch

import scanner


class ScannerCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        scanner._COINDCX_UNIVERSE_CACHE.clear()
        scanner._COINDCX_UNIVERSE_CACHE.update({"fetched_at": 0.0, "payload": {"spot": {}, "futures": []}})
        scanner._COINDCX_FUTURES_DETAILS_CACHE.clear()
        scanner._COINDCX_FUTURES_PRICES_CACHE.clear()
        scanner._COINDCX_FUTURES_PRICES_CACHE.update({"fetched_at": 0.0, "payload": {}})

    def test_extract_underlying_symbol_from_current_coindcx_pair_format(self) -> None:
        symbol = scanner._extract_underlying_symbol({"instrument_name": "B-BTC_USDT"})
        self.assertEqual(symbol, "BTC")

    @patch("scanner._coindcx_get")
    def test_fetch_derivatives_tickers_supports_string_payload(self, mock_get) -> None:
        mock_get.return_value = ["B-BTC_USDT", "B-ETH_USDT"]

        rows = scanner.fetch_derivatives_tickers()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["instrument_name"], "B-BTC_USDT")
        self.assertEqual(rows[0]["underlying_symbol"], "BTC")
        self.assertEqual(rows[0]["margin_asset"], "USDT")
        self.assertEqual(rows[1]["underlying_symbol"], "ETH")

    @patch("scanner._coindcx_get")
    def test_fetch_coindcx_instrument_details_supports_nested_instrument_payload(self, mock_get) -> None:
        mock_get.return_value = {
            "instrument": {
                "maximum_leverage": "25",
                "funding_rate": "0.0025",
                "open_interest": "15300000",
                "price": "92345.2",
            }
        }

        details = scanner.fetch_coindcx_instrument_details("B-BTC_USDT")

        self.assertEqual(details["max_leverage_long"], 25.0)
        self.assertEqual(details["max_leverage_short"], 25.0)
        self.assertEqual(details["funding_rate"], 0.0025)
        self.assertEqual(details["open_interest"], 15300000.0)
        self.assertEqual(details["last_price"], 92345.2)

    @patch("scanner._coindcx_public_get")
    def test_fetch_coindcx_futures_prices_supports_public_prices_payload(self, mock_public_get) -> None:
        mock_public_get.return_value = {
            "prices": {
                "B-BTC_USDT": "93000.5",
                "B-ETH_USDT": {
                    "last_price": "2500.25",
                    "volume_24h": "1234567",
                },
            }
        }

        prices = scanner.fetch_coindcx_futures_prices(["B-BTC_USDT", "B-ETH_USDT"])

        self.assertEqual(prices["B-BTC_USDT"]["last_price"], 93000.5)
        self.assertEqual(prices["B-ETH_USDT"]["last_price"], 2500.25)
        self.assertEqual(prices["B-ETH_USDT"]["volume_24h"], 1234567.0)

    def test_pick_best_coingecko_candidate_prefers_price_match_for_same_symbol(self) -> None:
        rows = [
            {"id": "wrong-aaa", "symbol": "aaa", "current_price": 2.0, "market_cap": 5_000_000},
            {"id": "right-aaa", "symbol": "aaa", "current_price": 98.0, "market_cap": 500_000},
        ]

        choice = scanner._pick_best_coingecko_candidate("AAA", rows, preferred_price=100.0)

        self.assertIsNotNone(choice)
        self.assertEqual(choice["id"], "right-aaa")

    @patch("scanner.fetch_coingecko_data_for_symbols")
    @patch("scanner.fetch_coindcx_data")
    @patch("scanner.fetch_derivatives_tickers")
    @patch("scanner.fetch_coindcx_spot_markets")
    def test_fetch_market_data_builds_coin_gecko_requests_from_coindcx_universe(
        self,
        mock_spot_markets,
        mock_futures,
        mock_coindcx_data,
        mock_coingecko_for_symbols,
    ) -> None:
        mock_spot_markets.return_value = {"AAA": "AAAUSDT", "BBB": "BBBUSDT"}
        mock_futures.return_value = []
        mock_coindcx_data.return_value = [
            scanner.MarketSnapshot(
                id="coindcx_aaa",
                symbol="AAA",
                name="AAA",
                current_price=10.0,
                market_cap=0.0,
                total_volume=1_000_000.0,
                price_change_24h=1.0,
            ),
            scanner.MarketSnapshot(
                id="coindcx_bbb",
                symbol="BBB",
                name="BBB",
                current_price=20.0,
                market_cap=0.0,
                total_volume=2_000_000.0,
                price_change_24h=2.0,
            ),
        ]
        mock_coingecko_for_symbols.return_value = [
            scanner.MarketSnapshot(
                id="bbb-id",
                symbol="BBB",
                name="BBB Coin",
                current_price=20.1,
                market_cap=10_000_000.0,
                total_volume=3_000_000.0,
                price_change_24h=2.0,
            )
        ]

        with patch.object(scanner, "COINDCX_FUTURES_ONLY", False):
            snapshots = scanner.fetch_market_data(coins_to_fetch=10)

        requested_symbols = mock_coingecko_for_symbols.call_args.args[0]
        self.assertCountEqual(requested_symbols, ["AAA", "BBB"])
        self.assertEqual(len(snapshots), 2)
        self.assertEqual({snap.symbol for snap in snapshots}, {"AAA", "BBB"})
        fallback = next(snap for snap in snapshots if snap.symbol == "AAA")
        self.assertEqual(fallback.id, "coindcx_aaa")


if __name__ == "__main__":
    unittest.main()
