"""Portfolio invariants that guard the research conclusions."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from microcap.portfolio import peer_panel, rank_universe, select_winner, simulate


def panel(months=("2018-01",), n=501):
    rows = []
    for month in months:
        period = pd.Period(month, freq="M")
        for i in range(1, n + 1):
            rows.append({"signal_date": str((period - 1).end_time.date()),
                         "horizon_end": str(period.end_time.date()),
                         "return_month": month, "stock_code": f"{i:06d}.SZ",
                         "market_cap": float(i), "eligible": True,
                         "return_1m": 0.0, "st": i % 3 == 0,
                         "net_income_ttm": np.nan})
    return pd.DataFrame(rows)


class RankingTests(unittest.TestCase):
    def test_boundaries_and_future_st_do_not_affect_ranking(self):
        source = panel()
        actual = rank_universe(source)
        lookup = actual.set_index("rank")["bucket"]
        self.assertEqual(lookup[100], "R001_100")
        self.assertEqual(lookup[101], "R101_200")
        self.assertEqual(lookup[200], "R101_200")
        self.assertEqual(lookup[201], "R201_300")
        self.assertEqual(lookup[500], "R401_500")
        self.assertTrue(pd.isna(lookup[501]))
        source["st"] = ~source["st"]
        source["future_st"] = True
        source["return_1m"] = np.nan
        source["net_income_ttm"] = -100.0
        changed = rank_universe(source)
        pd.testing.assert_frame_equal(actual[["stock_code", "rank", "bucket"]],
                                      changed[["stock_code", "rank", "bucket"]])

    def test_ties_and_invalid_market_cap(self):
        source = panel(n=5)
        source.loc[0:1, "market_cap"] = 1.0
        source.loc[2, "market_cap"] = np.inf
        source.loc[3, "market_cap"] = -1
        source.loc[4, "eligible"] = False
        actual = rank_universe(source.iloc[::-1])
        self.assertEqual(actual["stock_code"].tolist(), ["000001.SZ", "000002.SZ"])
        self.assertEqual(actual["rank"].tolist(), [1, 2])


class SimulationTests(unittest.TestCase):
    def test_initial_cash_zero_return_and_drift(self):
        source = panel(("2018-01", "2018-02"), n=2)
        source.loc[(source.return_month == "2018-01") & (source.stock_code == "000001.SZ"), "return_1m"] = 1.0
        monthly, holdings = simulate(rank_universe(source), 30, {"TWO": (1, 2)})
        self.assertAlmostEqual(monthly.iloc[0].gross_return, 0.5)
        self.assertAlmostEqual(monthly.iloc[0].turnover, 1.0)
        self.assertAlmostEqual(monthly.iloc[0].cost, 0.003)
        self.assertAlmostEqual(monthly.iloc[1].turnover, 1 / 3)
        self.assertAlmostEqual(monthly.iloc[1].net_return, -0.001)
        feb = holdings.loc[holdings.return_month == "2018-02"]
        self.assertAlmostEqual(feb.iloc[0].pretrade_weight, 2 / 3)
        self.assertAlmostEqual(feb.iloc[1].pretrade_weight, 1 / 3)

    def test_missing_return_is_not_zero_or_reweighted(self):
        source = panel(("2018-01", "2018-02", "2018-03"), n=501)
        source.loc[(source.return_month == "2018-01") & (source.stock_code == "000101.SZ"), "return_1m"] = np.nan
        monthly, holdings = simulate(rank_universe(source), 30)
        band = monthly.loc[monthly.strategy == "R101_200"].reset_index(drop=True)
        self.assertEqual(band.iloc[0].n_target, 100)
        self.assertEqual(band.iloc[0].n_returns, 99)
        self.assertTrue(np.isnan(band.iloc[0].gross_return))
        self.assertTrue(np.isnan(band.iloc[1].net_return))
        self.assertEqual(band.iloc[1].gross_return, 0)
        self.assertIn("unknown_pretrade", band.iloc[1].status)
        self.assertEqual(band.iloc[2].status, "ok")
        january = holdings.loc[(holdings.return_month == "2018-01")
                               & (holdings.strategy == "R101_200")]
        self.assertNotIn("000201.SZ", january.stock_code.tolist())
        self.assertTrue(january.target_weight.eq(0.01).all())

    def test_exit_weights_are_in_turnover(self):
        source = panel(("2018-01", "2018-02"), n=3)
        source.loc[(source.return_month == "2018-02") & (source.stock_code == "000001.SZ"), "market_cap"] = 4
        monthly, _ = simulate(rank_universe(source), 30, {"TWO": (1, 2)})
        self.assertAlmostEqual(monthly.iloc[1].turnover, 1.0)
        self.assertAlmostEqual(monthly.iloc[1].exit_turnover, 0.5)

    def test_incomplete_target_and_invalid_return(self):
        source = panel(n=99)
        monthly, _ = simulate(rank_universe(source), 30, {"R001_100": (1, 100)})
        self.assertEqual(monthly.iloc[0].n_target, 99)
        self.assertTrue(np.isnan(monthly.iloc[0].gross_return))
        self.assertIn("incomplete_target", monthly.iloc[0].status)
        source = panel(n=100)
        source.loc[0, "return_1m"] = -1.01
        monthly, _ = simulate(rank_universe(source), 30, {"R001_100": (1, 100)})
        self.assertIn("invalid_returns", monthly.iloc[0].status)
        source.loc[0, "return_1m"] = -1.0
        monthly, _ = simulate(rank_universe(source), 30, {"R001_100": (1, 100)})
        self.assertEqual(monthly.iloc[0].status, "ok")
        self.assertAlmostEqual(monthly.iloc[0].gross_return, -0.01)

    def test_peer_internal_migration_does_not_trade(self):
        source = panel(("2018-01", "2018-02"))
        source.loc[(source.return_month == "2018-02") & (source.stock_code == "000300.SZ"), "market_cap"] = 301
        source.loc[(source.return_month == "2018-02") & (source.stock_code == "000301.SZ"), "market_cap"] = 300
        ranked = rank_universe(source)
        bands, _ = simulate(ranked, 30)
        peers, holdings = simulate(peer_panel(ranked, "R101_200"), 30, {"PEERS": 300})
        self.assertEqual(peers.iloc[1].turnover, 0.0)
        self.assertEqual(peers.iloc[1].cost, 0.0)
        self.assertGreater(bands.loc[(bands.return_month == "2018-02")
                                     & (bands.strategy == "R201_300"), "cost"].iloc[0], 0)
        self.assertTrue(holdings.target_weight.eq(1 / 300).all())


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        months = pd.period_range(self.config["train_start"], self.config["test_end"], freq="M").astype(str)
        rows = []
        for i, month in enumerate(months):
            for j, candidate in enumerate(self.config["candidates"]):
                rows.append({"return_month": month, "strategy": candidate,
                             "net_return": 0.01 * (j + 1) + (0.02 if i % 2 else -0.02),
                             "turnover": 0.2, "cost_bps": 30, "status": "ok",
                             "n_target": 100, "n_expected": 100, "n_returns": 100})
        self.monthly = pd.DataFrame(rows)

    def test_oos_edits_cannot_change_winner_or_hash(self):
        before = select_winner(self.monthly, self.config)
        changed = self.monthly.copy()
        changed.loc[changed.return_month >= "2024-01", ["net_return", "turnover"]] = [999, 999]
        changed.loc[changed.return_month >= "2024-01", "status"] = "missing_returns"
        changed["future_st"] = True
        after = select_winner(changed.sample(frac=1, random_state=1), self.config)
        self.assertEqual(before, after)
        self.assertEqual(before["winner"], "R401_500")
        self.assertEqual(len(before["train_months"]), len(pd.period_range(self.config["train_start"], self.config["train_end"], freq="M")))
        self.assertEqual(len(before["training_hash"]), 64)

    def test_incomplete_training_refused(self):
        with self.assertRaisesRegex(ValueError, "incomplete training months"):
            select_winner(self.monthly.iloc[1:], self.config)
        changed = self.monthly.copy()
        changed.loc[0, "net_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "unresolved training"):
            select_winner(changed, self.config)
        changed = self.monthly.copy()
        changed.loc[0, "n_returns"] = 99
        with self.assertRaisesRegex(ValueError, "incomplete training constituents"):
            select_winner(changed, self.config)

    def test_zero_volatility_and_wrong_cost_refused(self):
        changed = self.monthly.copy()
        changed["net_return"] = 0.0
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            select_winner(changed, self.config)
        config = copy.deepcopy(self.config)
        config["main_cost_bps"] = 10
        with self.assertRaisesRegex(ValueError, "primary cost"):
            select_winner(self.monthly, config)

    def test_exact_ties_follow_turnover_then_larger_band(self):
        changed = self.monthly.copy()
        month_number = pd.factorize(changed.return_month)[0]
        changed["net_return"] = np.where(month_number % 2, 0.04, -0.02)
        self.assertEqual(select_winner(changed, self.config)["winner"], "R401_500")
        changed.loc[changed.strategy == "R201_300", "turnover"] = 0.1
        self.assertEqual(select_winner(changed, self.config)["winner"], "R201_300")

    def test_all_five_bands_compete_and_smallest_can_win(self):
        self.assertEqual(set(self.config["candidates"]), set(self.config["buckets"]))
        changed = self.monthly.copy()
        train = changed.return_month.le(self.config["train_end"])
        changed.loc[train & changed.strategy.eq("R001_100"), "net_return"] += 0.20
        changed.loc[~train & changed.strategy.eq("R401_500"), "net_return"] += 0.80
        selected = select_winner(changed, self.config)
        self.assertEqual(selected["winner"], "R001_100")
        self.assertEqual(len(selected["scores"]), 5)
        self.assertEqual(selected["protocol"]["main_cost_bps"], 30)
        self.assertIn("actual executed notional", selected["protocol"]["cost_convention"])


if __name__ == "__main__":
    unittest.main()
