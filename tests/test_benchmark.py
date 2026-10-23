import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from microcap.acquire import dump, sha
from microcap.benchmark import monthly_returns, prepare_benchmark


def fixture():
    config = {'train_start': '2015-01', 'test_end': '2015-02'}
    calendar = pd.bdate_range('2014-12-31', '2015-02-27')
    prices = np.linspace(100, 120, len(calendar))
    daily = pd.DataFrame({'trade_date': calendar, 'ts_code': '000852.SH', 'close': prices})
    return config, calendar, daily


class BenchmarkTests(unittest.TestCase):
    def test_monthly_returns_are_exact_endpoint_ratios(self):
        config, calendar, daily = fixture()
        result = monthly_returns(daily.iloc[::-1], calendar, config)
        january = daily.loc[daily.trade_date.eq(pd.Timestamp('2015-01-30')), 'close'].iloc[0]
        self.assertEqual(result.return_month.tolist(), ['2015-01', '2015-02'])
        self.assertAlmostEqual(result.net_return.iloc[0], january / 100 - 1)
        self.assertAlmostEqual(result.net_return.iloc[1], 120 / january - 1)
        self.assertTrue(result.strategy.eq('CSI1000').all())
        self.assertTrue(result.turnover.isna().all())
        self.assertTrue(result.cost.eq(0).all())
        np.testing.assert_array_equal(result.net_return, result.gross_return)

    def test_daily_gap_is_rejected_even_if_month_end_is_present(self):
        config, calendar, daily = fixture()
        with self.assertRaisesRegex(ValueError, 'calendar mismatch'):
            monthly_returns(daily.drop(index=5), calendar, config)

    def test_duplicate_date_is_rejected(self):
        config, calendar, daily = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            monthly_returns(pd.concat([daily, daily.iloc[:1]]), calendar, config)

    def test_wrong_code_and_invalid_close_are_rejected(self):
        config, calendar, daily = fixture()
        wrong = daily.copy()
        wrong.loc[0, 'ts_code'] = '000905.SH'
        with self.assertRaisesRegex(ValueError, 'wrong index'):
            monthly_returns(wrong, calendar, config)
        for invalid in (np.nan, np.inf, 0, -1):
            changed = daily.copy()
            changed.loc[0, 'close'] = invalid
            with self.assertRaisesRegex(ValueError, 'finite and strictly positive'):
                monthly_returns(changed, calendar, config)

    def test_frozen_reuse_is_offline_and_detects_raw_tampering(self):
        config, calendar, daily = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'data/prepared'
            prepared.mkdir(parents=True)
            dump(root / 'config.json', config)
            pd.DataFrame({'trade_date': calendar}).to_csv(prepared / 'calendar.csv', index=False)
            daily_path = prepared / 'benchmark_daily.csv'
            monthly_path = prepared / 'benchmark_monthly.csv'
            daily.to_csv(daily_path, index=False)
            monthly_returns(daily, calendar, config).to_csv(monthly_path, index=False)
            raw = root / 'frozen_raw.json'
            raw.write_text('{"verified_source":true}')
            specification = dict(code='000852.SH', name='中证1000', return_type='price_index',
                                 start_date='2014-12-31', end_date='2015-02-27',
                                 train_start='2015-01', test_end='2015-02')
            entry = lambda path: dict(path=str(path.relative_to(root)), sha256=sha(path))
            dump(prepared / 'benchmark_manifest.json', dict(
                specification=specification,
                artifacts=[entry(daily_path), entry(monthly_path)],
                raw_inputs=[entry(raw)], calendar=entry(prepared / 'calendar.csv')))
            with patch('microcap.benchmark.FrozenStore', side_effect=AssertionError('Network store used')):
                self.assertEqual(len(prepare_benchmark(root)), 2)
                raw.write_text('{"verified_source":false}')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    prepare_benchmark(root)


if __name__ == '__main__':
    unittest.main()
