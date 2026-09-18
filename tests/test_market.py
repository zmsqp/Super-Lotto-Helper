# -*- coding: utf-8 -*-
import json
import math
import tempfile
import unittest
from pathlib import Path

from services.fetcher import parse_sporttery_json
from services.market import combo_features, fit_crowd_model
from services.store import Store


class OfficialMarketParseTests(unittest.TestCase):
    def test_parse_sales_and_prize_counts(self):
        payload = {
            "value": {
                "list": [
                    {
                        "lotteryDrawNum": "26106",
                        "lotteryDrawTime": "2026-09-16",
                        "lotteryDrawResult": "14 17 21 25 28 04 06",
                        "totalSaleAmount": "290,236,360",
                        "poolBalanceAfterdraw": "826,970,066.74",
                        "prizeLevelList": [
                            {"group": "101", "stakeCount": "3", "stakeAmountFormat": "10000000"},
                            {"group": "301", "stakeCount": "71", "stakeAmountFormat": "200438"},
                        ],
                    }
                ]
            }
        }
        rows = parse_sporttery_json(json.dumps(payload))
        self.assertEqual(rows[0]["sales"], 290236360)
        self.assertEqual(rows[0]["prize_counts"], {"101": 3, "301": 71})
        self.assertEqual(rows[0]["pool_balance"], 826970066.74)


class CrowdFeatureTests(unittest.TestCase):
    def test_features_only_use_supplied_history(self):
        history = [{"front": [1, 2, 3, 4, 5], "back": [1, 2]}]
        features = combo_features([1, 6, 11, 16, 21], [2, 7], history)
        changed = combo_features([1, 6, 11, 16, 21], [2, 7], history + [{"front": [30] * 5, "back": [12, 12]}])
        self.assertNotEqual(features, changed)
        self.assertEqual(len(features), 16)

    def test_model_fits_official_density_target(self):
        draws = []
        market = []
        for index in range(100):
            front = sorted({(index + offset * 6) % 35 + 1 for offset in range(5)})
            while len(front) < 5:
                front = sorted(set(front + [front[-1] % 35 + 1]))
            back = sorted({index % 12 + 1, (index + 5) % 12 + 1})
            period = f"{index + 1:05d}"
            draws.append({"period": period, "date": "2026-01-01", "front": front, "back": back})
            market.append(
                {
                    "period": period,
                    "sales": 200_000_000 + index * 1000,
                    "prize_counts": {"101": index % 7},
                }
            )
        model = fit_crowd_model(draws, market)
        self.assertIsNotNone(model)
        self.assertEqual(model.samples, 100)
        self.assertTrue(math.isfinite(model.score(draws[-1]["front"], draws[-1]["back"], draws[:-1])))


class MarketStoreTests(unittest.TestCase):
    def test_market_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "test.db")
            store.upsert_draws(
                [
                    {
                        "period": "26106",
                        "date": "2026-09-16",
                        "front": [14, 17, 21, 25, 28],
                        "back": [4, 6],
                        "sales": 290236360,
                        "pool_balance": 826970066.74,
                        "prize_counts": {"101": 3, "301": 71},
                        "prize_amounts": {"101": 10000000},
                    }
                ]
            )
            rows = store.list_market_asc()
            store.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sales"], 290236360)
        self.assertEqual(rows[0]["prize_counts"]["101"], 3)


if __name__ == "__main__":
    unittest.main()
