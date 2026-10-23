import unittest

import numpy as np
import pandas as pd

from microcap.analysis import metrics, paired_tests, regressions, risk_statistics


def configuration():
    return {
        "train_start": "2020-01", "train_end": "2020-12",
        "test_start": "2021-01", "test_end": "2021-12",
        "buckets": {"R001_100": [1, 100], "R101_200": [101, 200], "R201_300": [201, 300], "R301_400": [301, 400], "R401_500": [401, 500]},
        "candidates": ["R101_200", "R201_300", "R301_400", "R401_500"],
        "reference": "R001_100", "main_cost_bps": 30, "hac_lags": 6,
    }


def monthly_data():
    rows = []
    for i, month in enumerate(pd.period_range("2020-01", "2021-12", freq="M")):
        for strategy, increment in (("R101_200", 0.003), ("R001_100", 0), ("PEERS", 0.001)):
            rows.append({"strategy": strategy, "return_month": str(month), "cost_bps": 30,
                         "net_return": 0.01 + np.sin(i) * 0.02 + increment,
                         "turnover": 0.3})
    return pd.DataFrame(rows)


def regression_data():
    rng = np.random.default_rng(2718)
    rows = []
    config = configuration()
    for month in pd.period_range("2020-01", "2021-12", freq="M"):
        bucket = np.repeat(list(config["buckets"]), 100)
        log_adv = rng.normal(12, 1, 500)
        prior = rng.uniform(-0.3, 0.3, 500)
        loss = rng.binomial(1, 0.4, 500)
        industry = rng.choice(["A", "B", "C"], 500)
        effects = np.repeat([0, 0.01, 0.02, 0.03, 0.04], 100)
        clipped_adv = np.clip(log_adv, *np.quantile(log_adv, [0.01, 0.99]))
        clipped_prior = np.clip(prior, *np.quantile(prior, [0.01, 0.99]))
        response = (0.02 + effects + 0.003 * clipped_adv + 0.05 * clipped_prior +
                    0.02 * loss + 0.01 * (industry == "B") - 0.01 * (industry == "C"))
        for i in range(500):
            rows.append({"stock_code": f"S{i:04}", "return_month": str(month), "bucket": bucket[i],
                         "return_1m": response[i], "industry": industry[i], "adv20": np.exp(log_adv[i]),
                         "return_prev1m": prior[i], "is_loss": loss[i]})
    return pd.DataFrame(rows)


class PerformanceTests(unittest.TestCase):
    def test_complete_metrics_match_direct_calculation(self):
        data = monthly_data()
        result = metrics(data, configuration())
        row = result.query("strategy == 'R101_200' and period == 'TRAIN'").iloc[0]
        values = data.query("strategy == 'R101_200'").iloc[:12]["net_return"]
        self.assertTrue(row["complete"])
        self.assertAlmostEqual(row["cagr"], (1 + values).prod() - 1)
        self.assertAlmostEqual(row["return_vol_ratio"], np.sqrt(12) * values.mean() / values.std(ddof=1))
        self.assertAlmostEqual(row["sharpe"], row["return_vol_ratio"])
        self.assertEqual(row["n_valid"], 12)

    def test_gap_or_missing_value_never_publishes_full_period_nav(self):
        data = monthly_data()
        data = data[~((data["strategy"] == "R101_200") & (data["return_month"] == "2020-06"))]
        data.loc[(data["strategy"] == "PEERS") & (data["return_month"] == "2020-04"), "net_return"] = np.nan
        result = metrics(data, configuration())
        for strategy in ("R101_200", "PEERS"):
            row = result.query("strategy == @strategy and period == 'TRAIN'").iloc[0]
            self.assertFalse(row["complete"])
            self.assertEqual(row["n_valid"], 11)
            for column in ("cagr", "ending_nav", "max_drawdown", "return_vol_ratio", "sharpe"):
                self.assertTrue(np.isnan(row[column]))
            self.assertTrue(np.isfinite(row["mean"]))

    def test_sharpe_subtracts_compounded_monthly_risk_free_return(self):
        data = monthly_data()
        config = configuration()
        config["risk_free_rate_annual"] = 0.036
        row = metrics(data, config).query("strategy == 'R101_200' and period == 'TRAIN'").iloc[0]
        values = data.query("strategy == 'R101_200'").iloc[:12]["net_return"]
        excess = values - (1.036 ** (1 / 12) - 1)
        self.assertAlmostEqual(row.sharpe, np.sqrt(12) * excess.mean() / excess.std(ddof=1))
        self.assertLess(row.sharpe, row.return_vol_ratio)

    def test_zero_volatility_has_undefined_sharpe(self):
        data = monthly_data()
        data.loc[data.strategy.eq('R101_200'), 'net_return'] = 0.0
        row = metrics(data, configuration()).query("strategy == 'R101_200' and period == 'TRAIN'").iloc[0]
        self.assertTrue(np.isnan(row.sharpe))

    def test_benchmark_comparisons_replace_peer_comparison(self):
        data = monthly_data().replace({'strategy': {'PEERS': 'CSI1000'}})
        config = configuration()
        config['benchmark'] = {'code': '000852.SH', 'name': '中证1000'}
        result = paired_tests(data, 'R101_200', config)
        self.assertFalse(any(r['comparator'] == 'PEERS' for r in result['comparisons']))
        benchmark_rows = [r for r in result['comparisons'] if r['comparator'] == 'CSI1000']
        self.assertEqual(len(benchmark_rows), 10)
        smallest = next(r for r in benchmark_rows if r['strategy'] == 'R001_100' and r['period'] == 'TRAIN')
        self.assertAlmostEqual(smallest['mean_monthly_difference'], -.001)
        self.assertEqual(smallest['inference_status'], 'ok')

    def test_smallest_winner_has_no_self_comparison(self):
        data = monthly_data().replace({'strategy': {'PEERS': 'CSI1000'}})
        config = configuration()
        config['candidates'] = list(config['buckets'])
        config['benchmark'] = {'code': '000852.SH', 'name': '中证1000'}
        result = paired_tests(data, 'R001_100', config)
        self.assertEqual(result['winner'], 'R001_100')
        self.assertEqual(len(result['comparisons']), 18)
        self.assertTrue(all(r['strategy'] != r['comparator'] for r in result['comparisons']))
        self.assertEqual(sum(r['strategy'] == 'R001_100' and r['comparator'] == 'CSI1000'
                             for r in result['comparisons']), 2)

    def test_initial_loss_is_included_in_drawdown(self):
        data = monthly_data()
        mask = data["strategy"] == "R101_200"
        data.loc[mask, "net_return"] = 0.0
        data.loc[mask & (data["return_month"] == "2020-01"), "net_return"] = -0.25
        row = metrics(data, configuration()).query("strategy == 'R101_200' and period == 'TRAIN'").iloc[0]
        self.assertAlmostEqual(row["max_drawdown"], -0.25)

    def test_paired_ci_is_for_monthly_difference_and_rejects_gaps(self):
        data = monthly_data()
        result = paired_tests(data, "R101_200", configuration())
        train = next(row for row in result["comparisons"] if row["period"] == "TRAIN" and row["comparator"] == "R001_100")
        self.assertAlmostEqual(train["mean"], 0.003)
        self.assertEqual(train["inference_status"], "ok")
        self.assertAlmostEqual(train["ci95_low"], 0.003)
        self.assertAlmostEqual(train["correlation"], 1)
        data = data[~((data["strategy"] == "R101_200") & (data["return_month"] == "2020-06"))]
        result = paired_tests(data, "R101_200", configuration())
        train = next(row for row in result["comparisons"] if row["period"] == "TRAIN" and row["comparator"] == "R001_100")
        self.assertEqual(train["inference_status"], "missing_months")
        self.assertTrue(np.isnan(train["hac_se"]))
        self.assertEqual(train["n_valid"], 11)

    def test_every_candidate_is_compared_to_reference_without_changing_winner(self):
        data = monthly_data()
        config = configuration()
        # Use the same common component with different known monthly spreads.
        reference = data.loc[data.strategy.eq("R001_100")].copy()
        additions = []
        for number, candidate in enumerate(config["candidates"][1:], start=2):
            rows = reference.copy()
            rows["strategy"] = candidate
            rows["net_return"] += number * 0.004
            additions.append(rows)
        result = paired_tests(pd.concat([data, *additions]), "R101_200", config)
        self.assertEqual(result["winner"], "R101_200")
        self.assertEqual(len(result["comparisons"]), 10)
        reference_rows = [r for r in result["comparisons"] if r["comparator"] == "R001_100"]
        self.assertEqual({(r["strategy"], r["period"]) for r in reference_rows},
                         {(c, p) for c in config["candidates"] for p in ("TRAIN", "TEST")})
        for row in reference_rows:
            self.assertEqual(row["inference_status"], "ok")
            if row["strategy"] == "R401_500":
                self.assertAlmostEqual(row["mean_monthly_difference"], 0.016)
        peer_rows = [r for r in result["comparisons"] if r["comparator"] == "PEERS"]
        self.assertEqual(len(peer_rows), 2)
        self.assertEqual({r["strategy"] for r in peer_rows}, {"R101_200"})

    def test_absent_candidate_does_not_disappear_from_comparisons(self):
        result = paired_tests(monthly_data(), "R101_200", configuration())
        absent = [r for r in result["comparisons"] if r["strategy"] == "R401_500"]
        self.assertEqual(len(absent), 2)
        self.assertTrue(all(r["n_valid"] == 0 and r["inference_status"] == "missing_months"
                            and np.isnan(r["hac_se"]) for r in absent))

    def test_duplicate_month_fails(self):
        data = monthly_data()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            metrics(pd.concat([data, data.iloc[:1]]), configuration())


class RiskTests(unittest.TestCase):
    def test_unknown_outcomes_keep_denominator_and_own_counts(self):
        data = pd.DataFrame({
            "stock_code": list("ABCDE"), "return_month": ["2020-01"] * 5,
            "bucket": ["R101_200"] * 5,
            "st_current": [0, 0, 0, 1, np.nan],
            "new_st_next": [1, 0, np.nan, np.nan, np.nan],
            "delisted_next": [1, 0, np.nan, 0, np.nan],
        })
        month, summary = risk_statistics(data, configuration())
        row = month.query("return_month == '2020-01' and bucket == 'R101_200'").iloc[0]
        self.assertEqual(row["n_current_nonst"], 3)
        self.assertEqual(row["new_st_followup_unknown"], 1)
        self.assertAlmostEqual(row["new_st_event_lower_bound_rate"], 1 / 3)
        self.assertAlmostEqual(row["new_st_known_followup_rate"], 1 / 2)
        self.assertAlmostEqual(row["new_st_followup_coverage"], 2 / 3)
        self.assertAlmostEqual(row["delist_event_lower_bound_rate"], 1 / 5)
        self.assertAlmostEqual(row["delist_known_followup_rate"], 1 / 3)
        self.assertEqual(row["n_current_unknown"], 1)
        self.assertEqual(row["n_missing_target"], 95)
        total = summary.query("period == 'TRAIN' and bucket == 'R101_200'").iloc[0]
        self.assertEqual(total["n_full_target"], 1200)
        self.assertFalse(total["complete_holdings"])

    def test_unknown_binary_value_is_rejected(self):
        data = pd.DataFrame({"stock_code": ["A"], "return_month": ["2020-01"], "bucket": ["R001_100"],
                             "st_current": [2], "new_st_next": [np.nan], "delisted_next": [0]})
        with self.assertRaisesRegex(ValueError, "only 0, 1"):
            risk_statistics(data, configuration())


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = regression_data()

    def test_recovers_known_bucket_effects(self):
        coeff, status, summary = regressions(self.data, configuration())
        self.assertEqual(set(status["full_status"]), {"ok"})
        for term, truth in (("R101_200", 0.01), ("R401_500", 0.04), ("is_loss", 0.02)):
            values = coeff.query("model == 'full' and term == @term")["coefficient"]
            np.testing.assert_allclose(values, truth, atol=1e-10)
        row = summary.query("period == 'TEST' and model == 'full' and term == 'R101_200'").iloc[0]
        self.assertEqual(row["inference_status"], "ok")

    def test_five_selection_candidates_do_not_change_regression_reference(self):
        config = configuration()
        original = regressions(self.data, config)
        config['candidates'] = list(config['buckets'])
        expanded = regressions(self.data, config)
        for before, after in zip(original, expanded):
            pd.testing.assert_frame_equal(before, after)
        self.assertEqual(set(expanded[1].full_status), {'ok'})
        self.assertNotIn('R001_100', set(expanded[0].term))

    def test_baseline_and_full_use_identical_complete_sample(self):
        data = self.data.copy()
        mask = (data["return_month"] == "2020-01") & data["stock_code"].isin(["S0001", "S0002", "S0250"])
        data.loc[mask, "industry"] = np.nan
        coeff, status, _ = regressions(data, configuration())
        january = coeff.query("return_month == '2020-01'")
        self.assertEqual(set(january["n_obs"]), {497})
        counts = status.query("return_month == '2020-01'").set_index("bucket")
        self.assertEqual(counts.loc["R001_100", "n_complete"], 98)
        self.assertEqual(counts.loc["R201_300", "n_complete"], 99)

    def test_predictor_bounds_do_not_depend_on_future_return_coverage(self):
        data = self.data.copy()
        original = regressions(data, configuration())[1]
        data.loc[(data["return_month"] == "2020-01") & (data["adv20"] > data["adv20"].quantile(0.8)), "return_1m"] = np.nan
        modified = regressions(data, configuration())[1]
        columns = ["log_adv20_p01", "log_adv20_p99", "return_prev1m_p01", "return_prev1m_p99"]
        pd.testing.assert_frame_equal(original[columns], modified[columns])

    def test_missing_month_blocks_hac(self):
        data = self.data[self.data["return_month"] != "2020-06"]
        _, status, summary = regressions(data, configuration())
        row = summary.query("period == 'TRAIN' and model == 'full' and term == 'R101_200'").iloc[0]
        self.assertEqual(row["n_valid"], 11)
        self.assertEqual(row["inference_status"], "missing_months")
        self.assertTrue(np.isnan(row["ci95_low"]))
        self.assertEqual(set(status.query("return_month == '2020-06'")["full_status"]), {"too_few_observations"})

    def test_rank_deficiency_is_reported_not_pseudoinverted(self):
        data = self.data.copy()
        data.loc[data["return_month"] == "2020-01", "is_loss"] = 0
        coeff, status, _ = regressions(data, configuration())
        self.assertTrue(coeff.query("return_month == '2020-01'").empty)
        self.assertEqual(set(status.query("return_month == '2020-01'")["full_status"]), {"rank_deficient"})


if __name__ == "__main__":
    unittest.main()
