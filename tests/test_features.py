"""Acceptance examples for point-in-time data and historical coverage."""

import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from microcap.features import historical_industry, profit_ttm, st_labels


def income(period, announced, value, **kwargs):
    return dict(ts_code="000001.SZ", end_date=period, ann_date=announced,
                f_ann_date=kwargs.get("f_ann_date"), report_type=kwargs.get("report_type", "1"),
                update_flag=kwargs.get("update_flag", "0"), n_income_attr_p=value)


class ProfitTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.bdate_range("2022-01-01", "2025-12-31")
        self.rows = pd.DataFrame([income("20230331", "20230428", 3),
                                  income("20231231", "20240329", 10),
                                  income("20240331", "20240430", -20)])

    def build(self, rows=None, dates=("2024-05-01",)):
        return profit_ttm(self.rows if rows is None else rows, dates, self.calendar)

    def test_announcement_day_and_negative_ttm(self):
        output, _ = self.build(dates=("2024-04-30", "2024-05-01"))
        self.assertEqual(output.iloc[0].net_profit_ttm, 10)
        self.assertEqual(output.iloc[1].net_profit_ttm, -13)
        self.assertEqual(output.iloc[1].is_loss, 1)
        self.assertEqual(output.iloc[1].available_date, pd.Timestamp("2024-05-01"))
        self.assertEqual(len(json.loads(output.iloc[1].input_trace)), 3)

    def test_max_announcement_then_next_session(self):
        self.rows.loc[2, "f_ann_date"] = "20240503"
        output, _ = self.build(dates=("2024-05-03", "2024-05-06"))
        self.assertEqual(output.iloc[0].net_profit_ttm, 10)
        self.assertEqual(output.iloc[1].net_profit_ttm, -13)

    def test_annual_does_not_need_prior_assets_or_profit(self):
        output, _ = self.build(self.rows.iloc[[1]], dates=("2024-04-01",))
        self.assertEqual(output.iloc[0].net_profit_ttm, 10)

    def test_future_revision_does_not_rewrite_past(self):
        revision = income("20240331", "20240603", 30, update_flag="1")
        extended = pd.concat([self.rows, pd.DataFrame([revision])], ignore_index=True)
        assert_frame_equal(self.build()[0], self.build(extended)[0])
        output, _ = self.build(extended, dates=("2024-06-04",))
        self.assertEqual(output.iloc[0].net_profit_ttm, 37)

    def test_conflict_is_quarantined_not_favorable_update_flag(self):
        conflict = income("20240331", "20240430", 30, update_flag="1")
        output, audit = self.build(pd.concat([self.rows, pd.DataFrame([conflict])], ignore_index=True))
        self.assertTrue(np.isnan(output.iloc[0].net_profit_ttm))
        self.assertEqual(output.iloc[0].ttm_status, "conflicting_version")
        self.assertIn("conflicting_same_date_versions", audit.issue.tolist())

    def test_identical_duplicate_and_report_type_filter(self):
        extra = income("20240331", "20240430", 999, report_type="4")
        output, audit = self.build(pd.concat([self.rows, self.rows.iloc[[2]], pd.DataFrame([extra])]))
        self.assertEqual(output.iloc[0].net_profit_ttm, -13)
        self.assertIn("excluded_nonconsolidated_cumulative_type", audit.issue.tolist())

    def test_missing_latest_comparison_does_not_fallback(self):
        output, _ = self.build(self.rows.iloc[[1, 2]])
        self.assertTrue(np.isnan(output.iloc[0].net_profit_ttm))
        self.assertEqual(output.iloc[0].ttm_status, "missing_comparison_period")

    def test_missing_revision_replaces_old_value(self):
        revision = income("20240331", "20240603", None)
        output, _ = self.build(pd.concat([self.rows, pd.DataFrame([revision])]), dates=("2024-06-04",))
        self.assertTrue(np.isnan(output.iloc[0].net_profit_ttm))
        self.assertEqual(output.iloc[0].ttm_status, "missing_profit")

    def test_stale_and_invalid_calendar(self):
        output, _ = self.build(self.rows.iloc[[1]], dates=("2024-12-02",))
        self.assertEqual(output.iloc[0].ttm_status, "stale_report")
        with self.assertRaises(ValueError):
            profit_ttm(self.rows, ["2024-05-04"], self.calendar)


class IndustryTests(unittest.TestCase):
    def test_half_open_boundaries_and_duplicate_labels(self):
        panel = pd.DataFrame(dict(stock_code=["A"] * 3, signal_date=["2024-01-30", "2024-01-31", "2024-02-01"]))
        intervals = pd.DataFrame([
            dict(ts_code="A", in_date="20200101", out_date="20240131", industry="银行"),
            dict(ts_code="A", in_date="20240131", out_date=None, industry="机械设备"),
            dict(ts_code="A", in_date="20240131", out_date=None, industry="机械设备")])
        result = historical_industry(panel, intervals)
        self.assertEqual(result.industry.tolist(), ["银行", "机械设备", "机械设备"])
        self.assertEqual(result.is_financial.tolist(), [1, 0, 0])

    def test_conflicting_or_missing_history_is_unknown(self):
        panel = pd.DataFrame(dict(stock_code=["A", "B"], signal_date=["2024-01-31"] * 2))
        intervals = pd.DataFrame([dict(ts_code="A", in_date="20200101", out_date=None, industry=name)
                                  for name in ["银行", "机械设备"]])
        result = historical_industry(panel, intervals)
        self.assertEqual(result.industry_status.tolist(), ["conflict", "unknown"])
        self.assertTrue(result.is_financial.isna().all())


class STTests(unittest.TestCase):
    def setUp(self):
        self.panel = pd.DataFrame([dict(stock_code="A", signal_date="2024-01-31", horizon_end="2024-02-29")])
        self.stocks = pd.DataFrame([dict(ts_code="A", list_date="20100101", delist_date=None, list_status="L")])
        self.names = pd.DataFrame([dict(ts_code="A", name="普通公司", start_date="20200101", end_date="20240209", ann_date="20191231")])

    def label(self, names=None, panel=None, stocks=None):
        return st_labels(self.panel if panel is None else panel,
                         self.names if names is None else names,
                         self.stocks if stocks is None else stocks).iloc[0]

    def test_future_st_does_not_change_past_current(self):
        future = pd.DataFrame([dict(ts_code="A", name="*ST公司", start_date="20240210", end_date=None, ann_date="20240209")])
        self.assertEqual(self.label().st_current, 0)
        result = self.label(pd.concat([self.names, future]))
        self.assertEqual(result.st_current, 0)
        self.assertEqual(result.new_st_next, 1)

    def test_intramonth_st_and_revocation_is_counted(self):
        future = pd.DataFrame([
            dict(ts_code="A", name="sst公司", start_date="20240210", end_date="20240215", ann_date="20240209"),
            dict(ts_code="A", name="普通新名", start_date="20240216", end_date=None, ann_date="20240215")])
        result = self.label(pd.concat([self.names, future]))
        self.assertEqual(result.new_st_next, 1)
        self.assertEqual(result.st_future_unknown, 0)

    def test_gaps_cannot_become_non_events(self):
        result = self.label()
        self.assertEqual(result.st_current, 0)
        self.assertTrue(np.isnan(result.new_st_next))
        self.assertEqual(result.st_future_unknown, 1)

    def test_missing_early_history_and_same_day_announcements(self):
        names = self.names.copy()
        names.loc[0, "start_date"] = "20240131"
        names.loc[0, "ann_date"] = "20240131"
        result = self.label(names)
        self.assertTrue(np.isnan(result.st_current))
        self.assertTrue(np.isnan(result.new_st_next))

    def test_event_known_despite_other_unknown_future_interval(self):
        future = pd.DataFrame([dict(ts_code="A", name="*ST公司", start_date="20240210", end_date="20240215", ann_date="20240209")])
        result = self.label(pd.concat([self.names, future]))
        self.assertEqual(result.new_st_next, 1)
        self.assertEqual(result.st_future_unknown, 1)

    def test_announced_in_future_record_is_not_past_evidence(self):
        future = pd.DataFrame([dict(ts_code="A", name="*ST公司", start_date="20200101", end_date=None, ann_date="20240301")])
        self.assertEqual(self.label(pd.concat([self.names, future])).st_current, 0)

    def test_conflicting_current_names_are_unknown(self):
        conflict = self.names.copy()
        conflict.loc[0, "name"] = "*ST公司"
        result = self.label(pd.concat([self.names, conflict]))
        self.assertTrue(np.isnan(result.st_current))
        self.assertEqual(result.st_status, "conflict")

    def test_delisting_is_event_label_not_selection_and_p_missing_not_zero(self):
        stocks = self.stocks.copy()
        stocks.loc[0, "delist_date"] = "20240220"
        stocks.loc[0, "list_status"] = "D"
        result = self.label(stocks=stocks)
        self.assertEqual(result.delisted_next, 1)
        self.assertEqual(result.st_current, 0)
        stocks.loc[0, "delist_date"] = None
        self.assertTrue(np.isnan(self.label(stocks=stocks).delisted_next))
        self.assertEqual(self.label().delisted_next, 0)


if __name__ == "__main__":
    unittest.main()
