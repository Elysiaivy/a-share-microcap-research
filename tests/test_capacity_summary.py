"""Public-data audit tests: detect omissions and materially wrong performance.

Two monthly observations have an intentionally known first-day loss. The fixture
has every declared protocol cell, without vendor data or a trading-engine replay.
"""
from pathlib import Path
import hashlib
import itertools
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.verify_capacity import verify_capacity


PROJECT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest(root):
    out = root / 'results/capacity'
    protocol = json.loads((out / 'protocol.json').read_text())
    names = ['config.json', 'results/selection.json', 'results/capacity/protocol.json']
    manifest = {
        'full_predeclared_grid': True,
        'n_scenarios': len(protocol['periods']) * len(protocol['scenarios']) * len(protocol['capitals_yuan']),
        'inputs': {name: file_hash(root / name) for name in names},
        'outputs': {p.name: file_hash(p) for p in out.iterdir() if p.suffix in ('.csv', '.gz')},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest))


def make_fixture(root):
    out = root / 'results/capacity'
    out.mkdir(parents=True)
    (root / 'results/supplement').mkdir()
    protocol = json.loads((PROJECT / 'results/capacity/protocol.json').read_text())
    protocol['scenarios'].setdefault('CAP_1P_SIGMA', {
        'execution': 'daily_vwap_proxy', 'participation': .01,
        'impact_eta': 0, 'require_known_sigma': True})
    protocol['periods'] = {'TRAIN': ['2015-01', '2015-02'], 'TEST': ['2024-01', '2024-02']}
    (out / 'protocol.json').write_text(json.dumps(protocol))
    (root / 'config.json').write_text(json.dumps({
        'main_cost_bps': 30, 'train_start': '2015-01', 'train_end': '2015-02',
        'test_start': '2024-01', 'test_end': '2024-02', 'buckets': {'R001_100': [1, 100]}}))
    (root / 'results/selection.json').write_text(json.dumps({'winner': 'R001_100'}))
    perf, monthly, daily, attempts, old, differences, ledgers = [], [], [], [], [], [], []
    for period, scenario, cap in itertools.product(protocol['periods'], protocol['scenarios'], protocol['capitals_yuan']):
        tag = {'period': period, 'scenario': scenario, 'initial_capital': cap}
        periods = pd.period_range(*protocol['periods'][period], freq='M')
        for month, value, ret in zip(periods, [.9, 1.08], [-.1, .2]):
            monthly.append({**tag, 'strategy': 'R001_100', 'cost_bps': 30,
                            'return_month': str(month), 'net_return': ret, 'nav': cap * value})
            daily.append({**tag, 'date': str(month.end_time.date()), 'nav': cap * value,
                          'cash': cap * value * .2, 'cash_weight': .2, 'hold_count': 100})
            if scenario == 'OPEN_BASE':
                old.append({'period': period, 'mode': 'LOT_MIN5', 'initial_capital': cap,
                            'return_month': str(month), 'net_return': ret})
        perf.append({**tag, 'n_months': 2, 'cagr': 1.08 ** 6 - 1, 'sharpe': np.sqrt(2 / 3),
                     'daily_max_drawdown': -.1, 'worst_month': -.1, 'mean_cash': .2,
                     'mean_holdings': 100., 'max_nav_yuan': cap * 1.08, 'ending_nav_yuan': cap * 1.08,
                     'min_cash_yuan': cap * .18, 'attempt_completion': .75, 'n_attempts': 4,
                     'partial_fill_attempts': 0, 'zero_fill_attempts': 1,
                     'impact_cost_yuan': 0., 'base_cost_yuan': .9, 'total_traded_yuan': 300.,
                     'impact_bps_per_traded_yuan': 0.})
        ledgers.append({**tag, 'checked_sessions': 2, 'checked_submitted_orders': 4,
                        'checked_fills': 3, 'checked_stock_day_capacity_budgets': 0 if scenario == 'OPEN_BASE' else 3,
                        'terminal_events': 0, 'base_fee_yuan': .9, 'incremental_impact_yuan': 0.,
                        'cash_positions_fees_reconciled': True,
                        'scope': 'Synthetic ledger verification receipt'})
        for side, reason, n, requested, filled, zero in [
            ('all', 'all', 4, 400., 300., 1), ('buy', 'all', 2, 200., 100., 1),
            ('sell', 'all', 2, 200., 200., 0), ('sell', 'ST', 0, 0., 0., 0),
            ('buy', 'ST_replacement', 0, 0., 0., 0),
        ]:
            attempts.append({**tag, 'side': side, 'order_reason': reason, 'attempts': n,
                             'attempt_requested_value': requested, 'attempt_filled_value': filled,
                             'attempt_completion': filled / requested if requested else np.nan,
                             'zero_fill_attempts': zero, 'partial_fill_attempts': 0})
    for period, cap in itertools.product(protocol['periods'], protocol['capitals_yuan']):
        for scenario, reference in [('VWAP_BASE', 'OPEN_BASE'), ('CAP_1P', 'VWAP_BASE'),
                                    ('CAP_1P_SIGMA', 'CAP_1P'), ('MAIN_1P', 'CAP_1P_SIGMA')]:
            differences.append({'period': period, 'initial_capital': cap, 'scenario': scenario,
                                'reference': reference, 'interpretation': 'synthetic unchanged path',
                                **{name + '_difference': 0. for name in ['cagr', 'sharpe', 'daily_max_drawdown', 'mean_cash', 'attempt_completion']}})
    for name, rows in [('performance.csv', perf), ('monthly.csv', monthly), ('daily.csv.gz', daily),
                       ('attempt_summary.csv', attempts), ('performance_differences.csv', differences),
                       ('ledger_checks.csv', ledgers)]:
        pd.DataFrame(rows).to_csv(out / name, index=False)
    pd.DataFrame(old).to_csv(root / 'results/supplement/capital_monthly.csv', index=False)
    update_manifest(root)


class CapacitySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / 'results/capacity'
        make_fixture(self.root)

    def rewrite(self, name, mutate):
        path = self.out / name
        frame = pd.read_csv(path)
        frame = mutate(frame)
        frame.to_csv(path, index=False)
        update_manifest(self.root)  # Exercise the semantic audit, not only checksum rejection.

    def test_valid_full_grid_and_known_first_loss(self):
        result = verify_capacity(self.root)
        protocol = json.loads((self.out / 'protocol.json').read_text())
        expected_runs = len(protocol['periods']) * len(protocol['capitals_yuan']) * len(protocol['scenarios'])
        self.assertEqual(result['capacity_scenario_runs'], expected_runs)
        self.assertEqual(result['capacity_verified_ledgers'], expected_runs)
        comparisons = pd.read_csv(self.out / 'performance_differences.csv')
        self.assertEqual(result['sequential_comparisons'], len(comparisons))

    def test_impact_comparison_must_use_matched_volatility_control(self):
        def use_unmatched_control(frame):
            frame.loc[frame.scenario.eq('MAIN_1P'), 'reference'] = 'CAP_1P'
            return frame
        self.rewrite('performance_differences.csv', use_unmatched_control)
        with self.assertRaisesRegex(ValueError, 'incomplete sequential comparison grid'):
            verify_capacity(self.root)

    def test_matched_control_requires_volatility_coverage(self):
        path = self.out / 'protocol.json'
        protocol = json.loads(path.read_text())
        protocol['scenarios']['CAP_1P_SIGMA']['require_known_sigma'] = False
        path.write_text(json.dumps(protocol))
        update_manifest(self.root)
        with self.assertRaisesRegex(ValueError, 'Matched volatility control must require known sigma'):
            verify_capacity(self.root)

    def test_partial_grid_cannot_be_published(self):
        self.rewrite('performance.csv', lambda x: x.iloc[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete or unexpected scenario grid'):
            verify_capacity(self.root)

    def test_each_scenario_requires_its_own_ledger_verification(self):
        self.rewrite('ledger_checks.csv', lambda x: x.iloc[:-1])
        with self.assertRaisesRegex(ValueError, 'Ledger checks: incomplete or unexpected scenario grid'):
            verify_capacity(self.root)

    def test_failed_ledger_check_cannot_be_published(self):
        def fail_check(frame):
            frame.loc[0, 'cash_positions_fees_reconciled'] = False
            return frame
        self.rewrite('ledger_checks.csv', fail_check)
        with self.assertRaisesRegex(ValueError, 'execution ledger was not verified'):
            verify_capacity(self.root)

    def test_ledger_counts_must_match_evaluated_sessions_and_attempts(self):
        for field in ['checked_sessions', 'checked_submitted_orders', 'checked_fills']:
            with self.subTest(field=field):
                path = self.out / 'ledger_checks.csv'
                original = path.read_bytes()
                self.rewrite('ledger_checks.csv', lambda x: x.assign(**{field: x[field] + 1}))
                with self.assertRaisesRegex(ValueError, f'ledger check {field} does not reconcile'):
                    verify_capacity(self.root)
                path.write_bytes(original)
                update_manifest(self.root)

    def test_ledger_fees_must_match_performance(self):
        self.rewrite('ledger_checks.csv', lambda x: x.assign(incremental_impact_yuan=123.))
        with self.assertRaisesRegex(ValueError, 'ledger check incremental_impact_yuan does not reconcile'):
            verify_capacity(self.root)

    def test_missing_ledger_receipts_are_a_core_input_error(self):
        (self.out / 'ledger_checks.csv').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing core capacity input: results/capacity/ledger_checks.csv'):
            verify_capacity(self.root)

    def test_duplicate_cells_do_not_substitute_for_missing_cells(self):
        self.rewrite('performance.csv', lambda x: pd.concat([x, x.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, 'duplicate scenario rows'):
            verify_capacity(self.root)

    def test_missing_return_is_not_cash(self):
        def damage(frame):
            frame.loc[0, 'net_return'] = np.nan
            return frame
        self.rewrite('monthly.csv', damage)
        with self.assertRaisesRegex(ValueError, 'Monthly: non-finite'):
            verify_capacity(self.root)

    def test_missing_month_is_not_ignored(self):
        self.rewrite('monthly.csv', lambda x: x.drop(index=0))
        with self.assertRaisesRegex(ValueError, 'missing or unexpected month'):
            verify_capacity(self.root)

    def test_drawdown_must_include_initial_cash(self):
        def omit_initial_loss(frame):
            frame.loc[0, 'daily_max_drawdown'] = 0.
            return frame
        self.rewrite('performance.csv', omit_initial_loss)
        with self.assertRaisesRegex(ValueError, 'daily_max_drawdown does not reconcile'):
            verify_capacity(self.root)

    def test_completion_is_a_ratio_not_a_claim(self):
        def overstate(frame):
            frame.loc[0, 'attempt_completion'] = .99
            return frame
        self.rewrite('attempt_summary.csv', overstate)
        with self.assertRaisesRegex(ValueError, 'completion ratio does not reconcile'):
            verify_capacity(self.root)

    def test_same_number_of_daily_rows_can_hide_missing_session(self):
        def shift(frame):
            frame.loc[0, 'date'] = '2015-01-30'
            return frame
        self.rewrite('daily.csv.gz', shift)
        with self.assertRaisesRegex(ValueError, 'daily trading-date set differs'):
            verify_capacity(self.root)

    def test_original_control_must_match_previous_experiment(self):
        path = self.root / 'results/supplement/capital_monthly.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'net_return'] += .001
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'OPEN_BASE differs from prior LOT_MIN5'):
            verify_capacity(self.root)

    def test_missing_core_file_fails_with_clear_message(self):
        (self.out / 'daily.csv.gz').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing core capacity input'):
            verify_capacity(self.root)

    def test_tampered_output_fails_before_recomputation(self):
        with (self.out / 'performance.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            verify_capacity(self.root)


if __name__ == '__main__':
    unittest.main()
