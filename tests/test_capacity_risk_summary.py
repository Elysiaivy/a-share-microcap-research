"""Public-only terminal-risk checks; no vendor input or private ledger cache."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import pandas as pd

from scripts.verify_capacity_risk import verify_capacity_risk


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CapacityRiskSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / 'results/capacity'
        self.out.mkdir(parents=True)
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/analyze_capacity_risk.py').write_text('# synthetic analysis source\n')
        (self.root / 'config.json').write_text(json.dumps({'terminal_fraction': 0.}))
        protocol = {'periods': {'TRAIN': ['2015-01', '2015-12'], 'TEST': ['2024-01', '2024-12']},
                    'scenarios': {'OPEN_BASE': {}, 'MAIN_1P': {}}, 'capitals_yuan': [1000000]}
        (self.out / 'protocol.json').write_text(json.dumps(protocol))
        rows = [{'period': period, 'scenario': scenario, 'initial_capital': 1000000,
                 'terminal_events': int(period == 'TEST'),
                 'terminal_marked_value_yuan': 100. if period == 'TEST' else 0.,
                 'terminal_residual_yuan': 0., 'terminal_writeoff_yuan': 100. if period == 'TEST' else 0.}
                for period in protocol['periods'] for scenario in protocol['scenarios']]
        frame = pd.DataFrame(rows)
        frame.to_csv(self.out / 'risk_diagnostics.csv', index=False)
        frame[['period', 'scenario', 'initial_capital', 'terminal_events']].assign(
            cash_positions_fees_reconciled=True).to_csv(self.out / 'ledger_checks.csv', index=False)
        case = frame[frame.period.eq('TEST') & frame.scenario.eq('MAIN_1P')].drop(columns='terminal_events')
        case.assign(date='2024-06-27', stock_code='600321.SH').to_csv(self.out / 'terminal_cases.csv', index=False)
        original_inputs = {name: sha(self.root / name) for name in ['config.json', 'results/capacity/protocol.json']}
        (self.out / 'manifest.json').write_text(json.dumps({'run_id': 'a' * 64, 'full_predeclared_grid': True,
            'inputs': original_inputs, 'outputs': {'ledger_checks.csv': sha(self.out / 'ledger_checks.csv')}}))
        inputs = {name: sha(self.root / name) for name in [*original_inputs, 'results/capacity/manifest.json',
                  'results/capacity/ledger_checks.csv', 'scripts/analyze_capacity_risk.py']}
        self.manifest = {'run_id': 'a' * 64, 'capacity_manifest_sha256': sha(self.out / 'manifest.json'),
                         'analysis_script_sha256': sha(self.root / 'scripts/analyze_capacity_risk.py'),
                         'inputs': inputs, 'scenario_runs': 4, 'scenario_terminal_events': 2,
                         'published_cases': 1, 'distinct_terminal_stock_dates': 1}
        self.refresh_outputs()

    def refresh_outputs(self):
        self.manifest['outputs'] = {name: sha(self.out / name) for name in ['risk_diagnostics.csv', 'terminal_cases.csv']}
        (self.out / 'risk_diagnostics_manifest.json').write_text(json.dumps(self.manifest))

    def modify(self, name, mutate):
        frame = mutate(pd.read_csv(self.out / name))
        frame.to_csv(self.out / name, index=False)
        self.refresh_outputs()  # Prove semantic checks, not just file-digest checks.

    def test_public_files_are_sufficient_without_private_cache(self):
        self.assertFalse((self.root / '.build').exists())
        result = verify_capacity_risk(self.root)
        self.assertEqual(result['capacity_risk_scenario_runs'], 4)
        self.assertEqual(result['scenario_terminal_events'], 2)
        self.assertEqual(result['distinct_published_terminal_stock_dates'], 1)

    def test_wrong_execution_run_is_rejected(self):
        self.manifest['run_id'] = 'b' * 64
        self.refresh_outputs()
        with self.assertRaisesRegex(ValueError, 'another execution run'):
            verify_capacity_risk(self.root)

    def test_changed_analysis_source_is_rejected(self):
        (self.root / 'scripts/analyze_capacity_risk.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'analysis script binding changed'):
            verify_capacity_risk(self.root)

    def test_output_hash_is_required(self):
        with (self.out / 'terminal_cases.csv').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'risk output checksum mismatch'):
            verify_capacity_risk(self.root)

    def test_missing_scenario_cannot_disappear(self):
        self.modify('risk_diagnostics.csv', lambda x: x.iloc[:-1])
        with self.assertRaisesRegex(ValueError, 'incomplete scenario grid'):
            verify_capacity_risk(self.root)

    def test_terminal_count_must_match_original_ledger(self):
        self.modify('risk_diagnostics.csv', lambda x: x.assign(terminal_events=x.terminal_events + 1))
        with self.assertRaisesRegex(ValueError, 'terminal counts differ'):
            verify_capacity_risk(self.root)

    def test_writeoff_is_an_accounting_identity(self):
        self.modify('risk_diagnostics.csv', lambda x: x.assign(terminal_writeoff_yuan=0.))
        with self.assertRaisesRegex(ValueError, 'carrying value does not equal'):
            verify_capacity_risk(self.root)

    def test_balanced_but_wrong_case_amount_is_rejected(self):
        self.modify('terminal_cases.csv', lambda x: x.assign(terminal_marked_value_yuan=50., terminal_writeoff_yuan=50.))
        with self.assertRaisesRegex(ValueError, 'complete case amounts differ'):
            verify_capacity_risk(self.root)


if __name__ == '__main__':
    unittest.main()
