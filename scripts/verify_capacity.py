"""Independently reconcile the public capacity experiment without raw market data.

This verifies published portfolio aggregates, not intraday fillability or the
private order ledger. Run ``python scripts/verify_capacity.py`` after a full run.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEYS = ['period', 'scenario', 'initial_capital']
ATTEMPT_CATEGORIES = {('all', 'all'), ('buy', 'all'), ('sell', 'all'),
                      ('sell', 'ST'), ('buy', 'ST_replacement')}


def difference_pairs(protocol):
    """Keep impact attribution matched on the required volatility coverage."""
    pairs = {('VWAP_BASE', 'OPEN_BASE'), ('CAP_1P', 'VWAP_BASE')}
    if 'CAP_1P_SIGMA' in protocol['scenarios']:
        pairs.update({('CAP_1P_SIGMA', 'CAP_1P'), ('MAIN_1P', 'CAP_1P_SIGMA')})
    else:
        pairs.add(('MAIN_1P', 'CAP_1P'))
    return pairs


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def close(actual, expected, message, *, atol=1e-10, rtol=1e-10):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def columns(frame, needed, label):
    require(set(needed).issubset(frame), f'{label}: missing columns {set(needed) - set(frame)}')


def finite(frame, fields, label):
    columns(frame, fields, label)
    require(np.isfinite(frame[fields].to_numpy(dtype=float)).all(), f'{label}: non-finite values in {fields}')


def keys(frame):
    columns(frame, KEYS, 'Scenario table')
    require(not frame[KEYS].isna().any().any(), 'Scenario table contains an unknown key')
    return set(frame[KEYS].itertuples(index=False, name=None))


def verify_capacity(root=ROOT):
    root = Path(root).resolve()
    output = root / 'results/capacity'
    core = ['protocol.json', 'performance.csv', 'monthly.csv', 'daily.csv.gz',
            'attempt_summary.csv', 'ledger_checks.csv', 'manifest.json']
    for path in [*(output / name for name in core), root / 'config.json',
                 root / 'results/selection.json', root / 'results/supplement/capital_monthly.csv']:
        if not path.is_file():
            raise FileNotFoundError(f'Missing core capacity input: {path.relative_to(root)}')
    protocol = json.loads((output / 'protocol.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    selection = json.loads((root / 'results/selection.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    require(manifest.get('full_predeclared_grid') is True, 'Capacity manifest is not a full predeclared grid')
    require(protocol['winner'] == selection['winner'], 'Capacity winner differs from frozen selection')
    require(protocol['winner'] in config['buckets'], 'Capacity winner is absent from frozen buckets')
    require(protocol['base_cost_bps'] == config['main_cost_bps'], 'Capacity base cost differs from frozen configuration')
    require(protocol['periods'] == {
        'TRAIN': [config['train_start'], config['train_end']],
        'TEST': [config['test_start'], config['test_end']]}, 'Capacity periods differ from frozen configuration')
    capitals = protocol['capitals_yuan']
    require(len(capitals) == len(set(capitals)) and all(x > 0 for x in capitals), 'Invalid or duplicate capital grid')
    expected = set(itertools.product(protocol['periods'], protocol['scenarios'], capitals))
    require({'OPEN_BASE', 'VWAP_BASE', 'CAP_1P', 'MAIN_1P'}.issubset(protocol['scenarios']), 'Required control scenarios absent')
    if 'CAP_1P_SIGMA' in protocol['scenarios']:
        matched = protocol['scenarios']['CAP_1P_SIGMA']
        capacity = protocol['scenarios']['CAP_1P']
        main = protocol['scenarios']['MAIN_1P']
        require(matched.get('require_known_sigma') is True and matched['impact_eta'] == 0,
                'Matched volatility control must require known sigma and charge zero impact')
        require(matched['participation'] == capacity['participation'] == main['participation'] and
                matched['execution'] == capacity['execution'] == main['execution'],
                'Matched volatility and impact controls differ in capacity or execution')
    require(manifest.get('n_scenarios') == len(expected), 'Manifest scenario count disagrees with protocol')
    for name in core[1:-1]:
        require(name in manifest.get('outputs', {}), f'Manifest omits core output: {name}')
    for name, expected_sha in manifest.get('outputs', {}).items():
        path = (output / name).resolve()
        require(path.is_relative_to(output), f'Manifest output escapes capacity directory: {name}')
        require(path.is_file() and digest(path) == expected_sha, f'Capacity output missing or checksum mismatch: {name}')
    for name in ['config.json', 'results/selection.json', 'results/capacity/protocol.json']:
        require(name in manifest.get('inputs', {}), f'Manifest omits frozen input: {name}')
    for name, expected_sha in manifest.get('inputs', {}).items():
        if name.startswith('external_full_project/'):
            continue  # Raw vendor inputs are deliberately not in the public repository.
        path = (root / name).resolve()
        require(path.is_relative_to(root), f'Manifest input escapes repository: {name}')
        require(path.is_file() and digest(path) == expected_sha, f'Published capacity input changed: {name}')

    performance = pd.read_csv(output / 'performance.csv', float_precision='round_trip')
    monthly = pd.read_csv(output / 'monthly.csv', float_precision='round_trip')
    daily = pd.read_csv(output / 'daily.csv.gz', float_precision='round_trip')
    attempts = pd.read_csv(output / 'attempt_summary.csv', float_precision='round_trip')
    ledgers = pd.read_csv(output / 'ledger_checks.csv', float_precision='round_trip')
    for label, frame in [('Performance', performance), ('Monthly', monthly), ('Daily', daily),
                         ('Attempts', attempts), ('Ledger checks', ledgers)]:
        require(keys(frame) == expected, f'{label}: incomplete or unexpected scenario grid')
    require(not performance.duplicated(KEYS).any(), 'Performance: duplicate scenario rows')
    require(not ledgers.duplicated(KEYS).any(), 'Ledger checks: duplicate scenario rows')
    columns(ledgers, ['cash_positions_fees_reconciled'], 'Ledger checks')
    require(ledgers.cash_positions_fees_reconciled.eq(True).all(), 'Ledger checks: an execution ledger was not verified')
    ledger_counts = ['checked_sessions', 'checked_submitted_orders', 'checked_fills',
                     'checked_stock_day_capacity_budgets', 'terminal_events']
    finite(ledgers, ledger_counts + ['base_fee_yuan', 'incremental_impact_yuan'], 'Ledger checks')
    require((ledgers[ledger_counts] >= 0).all().all(), 'Ledger checks: negative count')
    close(ledgers[ledger_counts], np.round(ledgers[ledger_counts]), 'Ledger checks: fractional count', atol=0, rtol=0)
    require((ledgers.checked_fills <= ledgers.checked_submitted_orders).all(), 'Ledger checks: fills exceed submitted orders')
    require((ledgers.checked_stock_day_capacity_budgets <= ledgers.checked_fills).all(), 'Ledger checks: capacity budgets exceed fills')
    ledger_rows = ledgers.set_index(KEYS)
    columns(monthly, ['return_month', 'net_return', 'nav'], 'Monthly')
    columns(daily, ['date', 'nav', 'cash', 'cash_weight', 'hold_count'], 'Daily')
    require(not monthly.duplicated(KEYS + ['return_month']).any(), 'Monthly: duplicate month')
    require(not daily.duplicated(KEYS + ['date']).any(), 'Daily: duplicate date')
    finite(monthly, ['net_return', 'nav'], 'Monthly')
    finite(daily, ['nav', 'cash', 'cash_weight', 'hold_count'], 'Daily')
    require((monthly.net_return > -1).all(), 'Monthly: non-positive ending wealth')
    require((daily.nav > 0).all() and (daily.hold_count >= 0).all(), 'Daily: non-positive NAV or negative holdings')
    if 'strategy' in monthly:
        require(monthly.strategy.eq(protocol['winner']).all(), 'Monthly table contains a different strategy')
    if 'cost_bps' in monthly:
        require(monthly.cost_bps.eq(protocol['base_cost_bps']).all(), 'Monthly table contains a different base cost')
    daily['_date'] = pd.to_datetime(daily.date, errors='raise')
    require(daily['_date'].notna().all(), 'Daily: missing date')
    require(not daily.duplicated(KEYS + ['_date']).any(), 'Daily: duplicate normalized date')
    daily['_month'] = daily['_date'].dt.to_period('M').astype(str)
    metrics = ['n_months', 'cagr', 'sharpe', 'daily_max_drawdown', 'worst_month', 'mean_cash',
               'mean_holdings', 'max_nav_yuan', 'ending_nav_yuan', 'min_cash_yuan',
               'n_attempts', 'partial_fill_attempts', 'zero_fill_attempts',
               'impact_cost_yuan', 'base_cost_yuan', 'total_traded_yuan']
    finite(performance, metrics, 'Performance')
    require((performance[['impact_cost_yuan', 'base_cost_yuan', 'total_traded_yuan']] >= -1e-7).all().all(), 'Performance: negative transaction cost or traded value')
    columns(performance, ['impact_bps_per_traded_yuan'], 'Performance')
    traded = performance.total_traded_yuan.gt(0)
    close(performance.loc[traded, 'impact_bps_per_traded_yuan'],
          performance.loc[traded, 'impact_cost_yuan'] / performance.loc[traded, 'total_traded_yuan'] * 10000,
          'Performance: impact basis points do not reconcile')
    p = performance.set_index(KEYS)
    month_groups = {key: g.sort_values('return_month') for key, g in monthly.groupby(KEYS)}
    day_groups = {key: g.sort_values('_date') for key, g in daily.groupby(KEYS)}
    period_dates = {}
    for key in sorted(expected):
        period, scenario, capital = key
        saved, m, d = p.loc[key], month_groups[key], day_groups[key]
        checked = ledger_rows.loc[key]
        for name, value in [('checked_sessions', len(d)), ('checked_submitted_orders', saved.n_attempts),
                            ('checked_fills', saved.n_attempts - saved.zero_fill_attempts),
                            ('base_fee_yuan', saved.base_cost_yuan),
                            ('incremental_impact_yuan', saved.impact_cost_yuan)]:
            close(checked[name], value, f'{key}: ledger check {name} does not reconcile')
        if protocol['scenarios'][scenario]['execution'] == 'original_next_open':
            require(checked.checked_stock_day_capacity_budgets == 0, f'{key}: opening control has capacity checks')
        elif checked.checked_fills > 0:
            require(checked.checked_stock_day_capacity_budgets > 0, f'{key}: fills have no checked capacity budgets')
        start, end = protocol['periods'][period]
        expected_months = pd.period_range(start, end, freq='M').astype(str).tolist()
        require(m.return_month.tolist() == expected_months, f'{key}: missing or unexpected month')
        dates = tuple(d['_date'])
        require(d['_date'].between(pd.Period(start).start_time, pd.Period(end).end_time).all(), f'{key}: date outside period')
        if period not in period_dates:
            period_dates[period] = dates
        require(dates == period_dates[period], f'{key}: daily trading-date set differs across scenarios')
        tolerance = max(1e-7, capital * 1e-12)
        require((d.cash >= -tolerance).all() and (d.cash <= d.nav + tolerance).all(), f'{key}: cash is negative or exceeds wealth')
        close(d.cash_weight, d.cash / d.nav, f'{key}: cash weights do not reconcile')
        ends = d.groupby('_month', sort=True).tail(1).set_index('_month')
        require(ends.index.tolist() == expected_months, f'{key}: daily table misses a month')
        close(m.nav, ends.nav, f'{key}: monthly and daily ending NAV differ')
        previous = np.r_[capital, ends.nav.to_numpy()[:-1]]
        close(m.net_return, ends.nav.to_numpy() / previous - 1, f'{key}: month returns do not start from period cash')
        returns = m.net_return.to_numpy()
        wealth = np.r_[capital, d.nav.to_numpy()]
        std = returns.std(ddof=1)
        require(std > 0, f'{key}: Sharpe undefined for a constant return sequence')
        computed = {
            'n_months': len(returns), 'cagr': (wealth[-1] / capital) ** (12 / len(returns)) - 1,
            'sharpe': np.sqrt(12) * returns.mean() / std, 'worst_month': returns.min(),
            'daily_max_drawdown': (wealth / np.maximum.accumulate(wealth) - 1).min(),
            'mean_cash': (d.cash / d.nav).mean(), 'mean_holdings': d.hold_count.mean(),
            'max_nav_yuan': wealth.max(), 'ending_nav_yuan': wealth[-1], 'min_cash_yuan': d.cash.min(),
        }
        if 'turnover' in m and 'annual_turnover' in saved:
            computed['annual_turnover'] = m.turnover.mean() * 12
        for name, value in computed.items():
            close(saved[name], value, f'{key}: {name} does not reconcile')
        if 'impact_cost' in d and d.impact_cost.notna().any():
            finite(d, ['impact_cost'], str(key))
            require((d.impact_cost >= -tolerance).all(), f'{key}: negative incremental impact')
            close(saved['impact_cost_yuan'], d.impact_cost.sum(), f'{key}: daily impact does not reconcile')
            if protocol['scenarios'][scenario]['impact_eta'] == 0:
                close(d.impact_cost.sum(), 0., f'{key}: zero-impact control charged an impact cost')
            if 'impact_cost_yuan' in m:
                by_month = d.groupby('_month').impact_cost.sum().reindex(expected_months)
                close(m.impact_cost_yuan, by_month, f'{key}: monthly impact does not reconcile')
                if 'impact_cost' in m:
                    close(m.impact_cost, by_month.to_numpy() / previous, f'{key}: monthly impact ratio does not reconcile')
        for name in ['target_shortfall_value', 'target_shortfall_weight', 'pending_sell_value', 'replacement_unfilled_budget']:
            if name in d and d[name].notna().any():
                finite(d, [name], str(key))
                require((d[name] >= -tolerance).all(), f'{key}: negative {name}')
                for label, value in [(f'month_end_mean_{name}', ends[name].mean()), (f'final_{name}', d[name].iloc[-1])]:
                    require(label in saved, f'{key}: missing {label}')
                    close(saved[label], value, f'{key}: {label} does not reconcile')

    columns(attempts, ['side', 'order_reason', 'attempts', 'attempt_requested_value',
                      'attempt_filled_value', 'attempt_completion', 'zero_fill_attempts',
                      'partial_fill_attempts'], 'Attempts')
    require(not attempts.duplicated(KEYS + ['side', 'order_reason']).any(), 'Attempts: duplicate category')
    finite(attempts, ['attempts', 'attempt_requested_value', 'attempt_filled_value',
                      'zero_fill_attempts', 'partial_fill_attempts'], 'Attempts')
    for key, group in attempts.groupby(KEYS):
        require(set(zip(group.side, group.order_reason)) == ATTEMPT_CATEGORIES, f'{key}: missing attempt category')
        g = group.set_index(['side', 'order_reason'])
        for category, row in g.iterrows():
            requested, filled = row.attempt_requested_value, row.attempt_filled_value
            require(requested >= 0 and filled >= 0 and filled <= requested + max(1e-7, requested * 1e-12), f'{key} {category}: filled value exceeds requested value')
            require(row.attempts >= 0 and row.zero_fill_attempts >= 0 and row.partial_fill_attempts >= 0 and
                    row.zero_fill_attempts + row.partial_fill_attempts <= row.attempts,
                    f'{key} {category}: invalid attempt counts')
            counts = row[['attempts', 'zero_fill_attempts', 'partial_fill_attempts']].to_numpy(dtype=float)
            close(counts, np.round(counts), f'{key} {category}: fractional attempt count', atol=0, rtol=0)
            if requested > 0:
                close(row.attempt_completion, filled / requested, f'{key} {category}: completion ratio does not reconcile')
                require(-1e-12 <= row.attempt_completion <= 1 + 1e-12, f'{key} {category}: completion ratio outside [0,1]')
            else:
                require(pd.isna(row.attempt_completion), f'{key} {category}: undefined completion must remain missing')
        additive = ['attempts', 'attempt_requested_value', 'attempt_filled_value', 'zero_fill_attempts', 'partial_fill_attempts']
        for name in additive:
            close(g.loc[('all', 'all'), name], g.loc[('buy', 'all'), name] + g.loc[('sell', 'all'), name], f'{key}: buy/sell attempt totals disagree')
        for category, parent in [(('sell', 'ST'), ('sell', 'all')), (('buy', 'ST_replacement'), ('buy', 'all'))]:
            require((g.loc[category, additive] <= g.loc[parent, additive] + 1e-7).all(), f'{key}: risk attempts exceed side totals')
        overall = g.loc[('all', 'all')]
        for name, attempt_name in [('attempt_completion', 'attempt_completion'), ('n_attempts', 'attempts'),
                                   ('partial_fill_attempts', 'partial_fill_attempts'), ('zero_fill_attempts', 'zero_fill_attempts')]:
            close(p.loc[key, name], overall[attempt_name], f'{key}: performance attempt summary disagrees')

    old = pd.read_csv(root / 'results/supplement/capital_monthly.csv', float_precision='round_trip')
    columns(old, ['period', 'mode', 'initial_capital', 'return_month', 'net_return'], 'Prior capital experiment')
    old = old[old['mode'].eq('LOT_MIN5') & old.period.isin(protocol['periods']) & old.initial_capital.isin(capitals)]
    overlap = set(old.initial_capital)
    require(bool(overlap), 'No overlapping LOT_MIN5 controls to reconcile')
    comparisons = 0
    for period, capital in itertools.product(protocol['periods'], overlap):
        previous = old[old.period.eq(period) & old.initial_capital.eq(capital)].sort_values('return_month')
        current = month_groups[(period, 'OPEN_BASE', capital)]
        require(previous.return_month.tolist() == current.return_month.tolist(), f'{period} {capital}: prior opening-control months differ')
        close(current.net_return, previous.net_return, f'{period} {capital}: OPEN_BASE differs from prior LOT_MIN5', atol=1e-10, rtol=0)
        comparisons += 1

    difference_path = output / 'performance_differences.csv'
    difference_count = 0
    if difference_path.is_file():
        require(difference_path.name in manifest['outputs'], 'Manifest omits scenario differences')
        diffs = pd.read_csv(difference_path, float_precision='round_trip')
        diff_keys = ['period', 'initial_capital', 'scenario', 'reference']
        columns(diffs, diff_keys, 'Differences')
        require(not diffs.duplicated(diff_keys).any(), 'Differences: duplicate comparisons')
        expected_diffs = {(per, cap, scenario, reference) for per in protocol['periods'] for cap in capitals for scenario, reference in difference_pairs(protocol)}
        require(set(diffs[diff_keys].itertuples(index=False, name=None)) == expected_diffs, 'Differences: incomplete sequential comparison grid')
        for row in diffs.itertuples(index=False):
            for metric in ['cagr', 'sharpe', 'daily_max_drawdown', 'mean_cash', 'attempt_completion']:
                value = getattr(row, metric + '_difference')
                actual = p.loc[(row.period, row.scenario, row.initial_capital), metric] - p.loc[(row.period, row.reference, row.initial_capital), metric]
                close(value, actual, f'Differences: {row.period} {row.initial_capital} {row.scenario} {metric} mismatch')
        difference_count = len(diffs)
    return {'capacity_scenario_runs': len(expected), 'capacity_monthly_rows': len(monthly),
            'capacity_verified_ledgers': len(ledgers),
            'opening_control_comparisons': comparisons, 'sequential_comparisons': difference_count,
            'scope': 'Public aggregate reconciliation; no raw-data or intraday fill certification'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_capacity(args.root), ensure_ascii=False, indent=2))
