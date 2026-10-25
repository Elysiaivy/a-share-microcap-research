"""Verify published terminal-risk diagnostics without private execution data."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ['period', 'scenario', 'initial_capital']
AMOUNTS = ['terminal_marked_value_yuan', 'terminal_residual_yuan', 'terminal_writeoff_yuan']


def require(condition, message):
    if not bool(condition):
        raise ValueError('Capacity risk verification: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def amounts(frame, label):
    require(set(AMOUNTS).issubset(frame), f'{label} lacks monetary fields')
    require(np.isfinite(frame[AMOUNTS].to_numpy(dtype=float)).all() and
            (frame[AMOUNTS] >= 0).all().all(), f'{label} has unknown or negative amounts')
    require(np.allclose(frame.terminal_marked_value_yuan,
                        frame.terminal_residual_yuan + frame.terminal_writeoff_yuan,
                        atol=1e-7, rtol=1e-10), f'{label} carrying value does not equal recovery plus writeoff')


def verify_capacity_risk(root=ROOT):
    root = Path(root).resolve()
    output = root / 'results/capacity'
    names = ['risk_diagnostics_manifest.json', 'manifest.json', 'protocol.json',
             'ledger_checks.csv', 'risk_diagnostics.csv', 'terminal_cases.csv']
    for path in [*(output / name for name in names), root / 'scripts/analyze_capacity_risk.py', root / 'config.json']:
        if not path.is_file():
            raise FileNotFoundError(f'Missing public capacity risk input: {path.relative_to(root)}')
    manifest = json.loads((output / 'risk_diagnostics_manifest.json').read_text())
    capacity = json.loads((output / 'manifest.json').read_text())
    require(capacity.get('full_predeclared_grid') is True, 'underlying capacity experiment is incomplete')
    require(manifest.get('run_id') == capacity['run_id'], 'risk analysis is bound to another execution run')
    require(manifest.get('capacity_manifest_sha256') == sha(output / 'manifest.json'), 'capacity manifest binding changed')
    require(manifest.get('analysis_script_sha256') == sha(root / 'scripts/analyze_capacity_risk.py'), 'analysis script binding changed')
    required_inputs = {'results/capacity/manifest.json', 'scripts/analyze_capacity_risk.py',
                       'results/capacity/protocol.json', 'results/capacity/ledger_checks.csv', 'config.json'}
    require(required_inputs.issubset(manifest.get('inputs', {})), 'risk manifest omits a public input')
    for name, digest in manifest['inputs'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'public input escapes repository')
        require(path.is_file() and sha(path) == digest, f'public input checksum mismatch: {name}')
    for name in ['config.json', 'results/capacity/protocol.json']:
        require(capacity.get('inputs', {}).get(name) == sha(root / name), f'original frozen input changed: {name}')
    require(capacity.get('outputs', {}).get('ledger_checks.csv') == sha(output / 'ledger_checks.csv'),
            'original ledger verification output changed')
    for name in ['risk_diagnostics.csv', 'terminal_cases.csv']:
        require(manifest.get('outputs', {}).get(name) == sha(output / name), f'risk output checksum mismatch: {name}')
    protocol = json.loads((output / 'protocol.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    expected = set(itertools.product(protocol['periods'], protocol['scenarios'], protocol['capitals_yuan']))
    diagnostics = pd.read_csv(output / 'risk_diagnostics.csv', float_precision='round_trip')
    checked = pd.read_csv(output / 'ledger_checks.csv', float_precision='round_trip')
    for label, frame in [('diagnostics', diagnostics), ('ledger checks', checked)]:
        require(set(KEYS + ['terminal_events']).issubset(frame), f'{label} lacks scenario keys or counts')
        require(not frame.duplicated(KEYS).any(), f'{label} has duplicate scenarios')
        require(set(frame[KEYS].itertuples(index=False, name=None)) == expected, f'{label} has an incomplete scenario grid')
        counts = frame.terminal_events.to_numpy(dtype=float)
        require(np.isfinite(counts).all() and (counts >= 0).all() and np.array_equal(counts, np.round(counts)),
                f'{label} has invalid event counts')
    require(checked.cash_positions_fees_reconciled.eq(True).all(), 'original ledger was not verified')
    d = diagnostics.set_index(KEYS).sort_index()
    checked = checked.set_index(KEYS).sort_index()
    require(d.terminal_events.equals(checked.terminal_events) or
            np.array_equal(d.terminal_events.to_numpy(), checked.terminal_events.to_numpy()),
            'terminal counts differ from original ledger verification')
    amounts(diagnostics, 'diagnostics')
    require((diagnostics.loc[diagnostics.terminal_events.eq(0), AMOUNTS] == 0).all().all(),
            'amounts are recorded without a terminal event')
    require(0 <= config['terminal_fraction'] <= 1 and np.allclose(
        diagnostics.terminal_residual_yuan,
        diagnostics.terminal_marked_value_yuan * config['terminal_fraction'], atol=1e-7, rtol=1e-10),
        'recovery differs from frozen terminal assumption')
    cases = pd.read_csv(output / 'terminal_cases.csv', float_precision='round_trip')
    case_keys = KEYS + ['date', 'stock_code']
    require(set(case_keys).issubset(cases), 'cases lack identifying fields')
    require(not cases[case_keys].isna().any().any() and not cases.duplicated(case_keys).any(),
            'cases have unidentified or duplicate records')
    require(set(cases[KEYS].itertuples(index=False, name=None)).issubset(expected), 'case belongs to an unknown scenario')
    amounts(cases, 'cases')
    for period, group in cases.groupby('period'):
        start, end = protocol['periods'][period]
        require(pd.to_datetime(group.date, errors='raise').between(
            pd.Period(start).start_time, pd.Period(end).end_time).all(), 'case date is outside its evaluated period')
    for key, group in cases.groupby(KEYS):
        saved = d.loc[key]
        require(len(group) <= saved.terminal_events, 'case count exceeds scenario terminal count')
        totals = group[AMOUNTS].sum().to_numpy(dtype=float)
        aggregate = saved[AMOUNTS].to_numpy(dtype=float)
        require((totals <= aggregate + np.maximum(1e-7, aggregate * 1e-10)).all(),
                'case amounts exceed their scenario totals')
        if len(group) == saved.terminal_events:
            require(np.allclose(totals, aggregate, atol=1e-7, rtol=1e-10),
                    'complete case amounts differ from their scenario totals')
    require(manifest.get('scenario_runs') == len(expected), 'risk manifest scenario count differs')
    require(manifest.get('scenario_terminal_events') == int(diagnostics.terminal_events.sum()), 'risk manifest event count differs')
    require(manifest.get('published_cases') == len(cases), 'risk manifest case count differs')
    distinct_cases = len(cases[['period', 'date', 'stock_code']].drop_duplicates())
    require(manifest.get('distinct_terminal_stock_dates') == distinct_cases, 'risk manifest distinct case count differs')
    return {'capacity_risk_scenario_runs': len(expected), 'scenario_terminal_events': int(diagnostics.terminal_events.sum()),
            'published_terminal_cases': len(cases), 'distinct_published_terminal_stock_dates': distinct_cases,
            'scope': 'Public risk-summary and provenance reconciliation; private event ledgers are not re-read.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_capacity_risk(args.root), ensure_ascii=False, indent=2))
