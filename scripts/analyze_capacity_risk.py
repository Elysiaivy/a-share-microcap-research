"""Summarize terminal accounting from verified, frozen capacity event ledgers.

This is a separate read-only analysis of the completed execution runs. It does
not rerun the strategy or modify the original capacity manifest. A terminal
writeoff is the last recorded carrying value less modeled cash recovery, not
the position's total investment loss or evidence of a successful market sale.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ['period', 'scenario', 'initial_capital']
AMOUNTS = ['terminal_marked_value_yuan', 'terminal_residual_yuan', 'terminal_writeoff_yuan']
CASE_COLUMNS = KEYS + ['date', 'stock_code', *AMOUNTS]


def require(condition, message):
    if not bool(condition):
        raise ValueError('Capacity risk analysis: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def analyze_capacity_risk(root=ROOT):
    root = Path(root).resolve()
    output = root / 'results/capacity'
    manifest_path = output / 'manifest.json'
    capacity_manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('full_predeclared_grid') is True, 'capacity grid is incomplete')
    run_id = manifest['run_id']
    require(isinstance(run_id, str) and len(run_id) == 64 and
            all(x in '0123456789abcdef' for x in run_id), 'invalid frozen run ID')
    inputs = {'results/capacity/manifest.json': capacity_manifest_hash,
              'scripts/analyze_capacity_risk.py': sha(Path(__file__))}
    for name in ['results/capacity/protocol.json', 'config.json']:
        require(name in manifest['inputs'], f'original manifest omits {name}')
        inputs[name] = sha(root / name)
        require(inputs[name] == manifest['inputs'][name], f'frozen input changed: {name}')
    ledger_name = 'ledger_checks.csv'
    require(ledger_name in manifest['outputs'], 'original manifest omits ledger verification')
    inputs['results/capacity/' + ledger_name] = sha(output / ledger_name)
    require(inputs['results/capacity/' + ledger_name] == manifest['outputs'][ledger_name],
            'ledger verification output changed')
    protocol = json.loads((output / 'protocol.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    terminal_fraction = config['terminal_fraction']
    require(0 <= terminal_fraction <= 1, 'invalid frozen recovery fraction')
    expected = set(itertools.product(protocol['periods'], protocol['scenarios'], protocol['capitals_yuan']))
    checks = pd.read_csv(output / ledger_name)
    require(not checks.duplicated(KEYS).any() and
            set(checks[KEYS].itertuples(index=False, name=None)) == expected,
            'ledger verification grid differs from frozen protocol')
    require(checks.cash_positions_fees_reconciled.eq(True).all(), 'an original ledger failed verification')
    checks = checks.set_index(KEYS)
    cache = root / '.build/capacity_runs' / run_id[:16]
    rows, all_cases, sources = [], [], {}
    for key in sorted(expected):
        period, scenario, capital = key
        name = f'{period}_{scenario}_{capital}'
        result_path, events_path = cache / name / 'result.json', cache / name / 'events.csv.gz'
        result_hash = sha(result_path)
        result = json.loads(result_path.read_text())
        require(result.get('run_id') == run_id and tuple(result[k] for k in KEYS) == key,
                f'wrong cached run: {name}')
        require(result.get('ledger_verification', {}).get('cash_positions_fees_reconciled') is True,
                f'missing cached ledger verification: {name}')
        events_hash = sha(events_path)
        require(result.get('ledger_hashes', {}).get('events.csv.gz') == events_hash,
                f'event ledger checksum mismatch: {name}')
        sources[str(result_path.relative_to(root))] = result_hash
        sources[str(events_path.relative_to(root))] = events_hash
        events = pd.read_csv(events_path, float_precision='round_trip')
        require('event' in events, f'missing event type: {name}')
        terminal = events[events.event.eq('terminal')].copy()
        count = len(terminal)
        require(count == checks.loc[key, 'terminal_events'] ==
                result['ledger_verification']['terminal_events'], f'terminal count differs from verified ledger: {name}')
        tag = dict(zip(KEYS, key))
        if count:
            require({'date', 'stock_code', 'strategy', 'marked_value', 'residual'}.issubset(terminal),
                    f'terminal carrying value or recovery absent: {name}')
            require(terminal.strategy.eq(protocol['winner']).all(), f'wrong strategy in terminal event: {name}')
            dates = pd.to_datetime(terminal.date, errors='raise')
            start, end = protocol['periods'][period]
            require(dates.between(pd.Period(start).start_time, pd.Period(end).end_time).all(),
                    f'terminal date outside evaluated period: {name}')
            require(terminal.stock_code.notna().all() and not terminal.duplicated(['date', 'stock_code']).any(),
                    f'duplicate or unidentified terminal stock: {name}')
            values = terminal[['marked_value', 'residual']].to_numpy(dtype=float)
            require(np.isfinite(values).all() and (values >= 0).all(), f'invalid terminal amounts: {name}')
            require(np.allclose(terminal.residual, terminal.marked_value * terminal_fraction,
                                atol=1e-7, rtol=1e-10), f'recovery differs from frozen assumption: {name}')
            terminal['terminal_marked_value_yuan'] = terminal.marked_value
            terminal['terminal_residual_yuan'] = terminal.residual
            terminal['terminal_writeoff_yuan'] = terminal.marked_value - terminal.residual
            require((terminal.terminal_writeoff_yuan >= -1e-7).all(), f'negative terminal writeoff: {name}')
            totals = terminal[AMOUNTS].sum().to_dict()
            all_cases.append(terminal.assign(**tag)[CASE_COLUMNS])
        else:
            totals = dict.fromkeys(AMOUNTS, 0.)
        rows.append({**tag, 'terminal_events': count, **totals})

    diagnostics = pd.DataFrame(rows)
    all_cases = pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame(columns=CASE_COLUMNS)
    selected = []
    # Fixed illustrative capitals, not cases selected for a favorable return.
    # Retain one deterministic row if an event exists only outside these cells.
    for _, group in all_cases.groupby(['period', 'date', 'stock_code'], sort=True):
        preferred = group[group.scenario.eq('MAIN_1P') & group.initial_capital.isin([1000000, 50000000])]
        selected.append(preferred if len(preferred) else group.sort_values(KEYS).head(1))
    cases = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=CASE_COLUMNS)
    require(not cases.duplicated(KEYS + ['date', 'stock_code']).any(), 'duplicate published case')
    # Do not publish a mixed snapshot if any original evidence changed in flight.
    for name, expected_hash in {**inputs, **sources}.items():
        require(sha(root / name) == expected_hash, f'input changed during analysis: {name}')
    provenance = {
        'analysis': 'terminal_accounting_from_frozen_capacity_ledgers',
        'run_id': run_id,
        'capacity_manifest_sha256': capacity_manifest_hash,
        'analysis_script_sha256': inputs['scripts/analyze_capacity_risk.py'],
        'inputs': inputs,
        'private_source_hashes': sources,
        'scenario_runs': len(diagnostics),
        'scenario_terminal_events': int(diagnostics.terminal_events.sum()),
        'distinct_terminal_stock_dates': len(all_cases[['period', 'date', 'stock_code']].drop_duplicates()),
        'published_cases': len(cases),
        'case_selection': 'MAIN_1P at CNY 1m and 50m for each distinct terminal stock/date; otherwise first sorted scenario/capital row.',
        'definition': 'Last carrying value minus modeled residual cash at terminal accounting; not total investment loss and not a market sale.',
        'counting': 'Scenario event counts repeat the same underlying event across independent runs; do not interpret their sum as unique company events.',
        'scope': 'Frozen event-ledger accounting only; no new recovery evidence or execution assumption.',
    }
    with tempfile.TemporaryDirectory(prefix='.risk-analysis-', dir=output) as temporary:
        stage = Path(temporary)
        diagnostics.to_csv(stage / 'risk_diagnostics.csv', index=False)
        cases.to_csv(stage / 'terminal_cases.csv', index=False)
        provenance['outputs'] = {name: sha(stage / name)
                                 for name in ['risk_diagnostics.csv', 'terminal_cases.csv']}
        (stage / 'risk_diagnostics_manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        for name in [*provenance['outputs'], 'risk_diagnostics_manifest.json']:
            (stage / name).replace(output / name)
    return {key: provenance[key] for key in ['run_id', 'scenario_runs', 'scenario_terminal_events',
                                             'distinct_terminal_stock_dates', 'published_cases']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze_capacity_risk(args.root), ensure_ascii=False, indent=2))
