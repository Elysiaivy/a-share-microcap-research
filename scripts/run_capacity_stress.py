"""Re-run the frozen winner under predeclared daily liquidity pressure scenarios.

Raw market data and full order ledgers remain outside the public release.
Usage: python scripts/run_capacity_stress.py --data-root /authorized/full/project
The result-recomputation notebook never calls this entry point.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from microcap.execution import run
from microcap.execution_constraints import OrderConstraints
from microcap.research import inputs
from microcap.preflight import require_full_inputs
from scripts.capacity_ledger_checks import verify_capacity_ledgers

PROTOCOL_PATH = ROOT / 'results/capacity/protocol.json'
MARKET = PANEL = CONFIG = PROTOCOL = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sources(sources, data_root, liquidity_file):
    for name, digest in sources.items():
        if name == 'external_full_project/data/capacity/capacity_market.npz':
            path = liquidity_file
        elif name.startswith('external_full_project/'):
            path = data_root/name.removeprefix('external_full_project/')
        else:
            path = ROOT/name
        if sha(path) != digest:
            raise ValueError(f'Capacity run input/code changed during evaluation: {name}')


def initialize(data_root, liquidity_file):
    global MARKET, PANEL, CONFIG, PROTOCOL
    CONFIG, PANEL, _, MARKET = inputs(data_root)
    with np.load(liquidity_file, allow_pickle=False) as z:
        if not np.array_equal(z['dates'], MARKET['dates']) or not np.array_equal(z['codes'], MARKET['codes']):
            raise ValueError('Capacity matrix axes differ from frozen market')
        for key in z.files:
            if key.startswith('capacity_'):
                MARKET[key] = z[key]
    PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
    expected = json.loads((ROOT / 'config.json').read_text())
    if CONFIG != expected:
        raise ValueError('External full project is not the frozen public protocol')


def order_attempts(details):
    """For the opening control derive the same attempt-based denominator."""
    if 'liquidity_orders' in details and len(details['liquidity_orders']):
        return details['liquidity_orders'].copy()
    orders = details['order_plans'].copy().rename(columns={'execute_date': 'date'})
    trades = details['trades']
    keys = ['strategy', 'date', 'stock_code', 'side']
    if orders.duplicated(keys).any():
        raise ValueError('Opening control needs disambiguated attempt IDs')
    filled = trades.groupby(keys, as_index=False).agg(filled_units=('filled_units', 'sum'), base_fee=('cost', 'sum'))
    orders = orders.merge(filled, on=keys, how='left', validate='one_to_one')
    orders[['filled_units', 'base_fee']] = orders[['filled_units', 'base_fee']].fillna(0.)
    orders['requested_reference_notional'] = orders.planned_units * orders.previous_close
    orders['filled_reference_notional'] = orders.filled_units * orders.previous_close
    orders['impact_cost'] = 0.
    orders['constraint_reason'] = np.where(orders.filled_units > 0, 'none', 'opening_control_unfilled')
    return orders


def summarize_attempts(orders, tag):
    out = []
    q = orders.copy()
    for side, reason in [('all', 'all'), ('buy', 'all'), ('sell', 'all'), ('sell', 'ST'), ('buy', 'ST_replacement')]:
        v = q.copy()
        if side != 'all':
            v = v[v.side.eq(side)]
        if reason != 'all':
            v = v[v.reason.eq(reason)]
        planned = float(v.requested_reference_notional.sum())
        filled = float(v.filled_reference_notional.sum())
        out.append({**tag, 'side': side, 'order_reason': reason, 'attempts': len(v),
                    'attempt_requested_value': planned, 'attempt_filled_value': filled,
                    'attempt_completion': filled / planned if planned > 0 else np.nan,
                    'zero_fill_attempts': int(v.filled_units.le(1e-12).sum()),
                    'partial_fill_attempts': int(((v.filled_units > 1e-12) &
                        (v.filled_units < v.planned_units - 1e-10)).sum())})
    return pd.DataFrame(out)


def execute_job(task):
    period, scenario, capital, cache_dir, run_id = task
    from microcap.liquidity import LiquidityConstraints
    cache_dir = Path(cache_dir)
    key = f'{period}_{scenario}_{capital}'
    path = cache_dir / key
    done = path / 'result.json'
    if done.exists():
        row = json.loads(done.read_text())
        if row.get('run_id') == run_id:
            for name, digest in row.get('ledger_hashes', {}).items():
                if not (path/name).is_file() or sha(path/name) != digest:
                    raise ValueError(f'Cached capacity ledger changed: {key}/{name}')
            if (len(row.get('ledger_hashes', {})) != 7
                    or not row.get('ledger_verification',{}).get('cash_positions_fees_reconciled')):
                raise ValueError(f'Incomplete cached capacity ledger: {key}')
            return row
    path.mkdir(parents=True, exist_ok=True)
    start, end = PROTOCOL['periods'][period]
    winner = PROTOCOL['winner']
    config = {**CONFIG, 'train_start': start, 'test_end': end,
              'buckets': {winner: CONFIG['buckets'][winner]}}
    scenario_config = PROTOCOL['scenarios'][scenario]
    liquidity = None if scenario == 'OPEN_BASE' else LiquidityConstraints(
        participation=scenario_config['participation'], impact_eta=scenario_config['impact_eta'],
        require_known_sigma=scenario_config.get('require_known_sigma', False))
    implementation = OrderConstraints(capital, PROTOCOL['buy_lot_shares'],
                                      PROTOCOL['minimum_commission_yuan'], PROTOCOL['commission_bps'])
    then = time.monotonic()
    monthly, d = run(MARKET, PANEL, config, PROTOCOL['base_cost_bps'], CONFIG['terminal_fraction'],
                     details=True, implementation=implementation, liquidity=liquidity)
    verification = verify_capacity_ledgers(monthly, d, MARKET, PROTOCOL, scenario, capital,
                                            terminal_fraction=CONFIG['terminal_fraction'])
    tag = {'period': period, 'scenario': scenario, 'initial_capital': capital}
    daily, trades, events = d['daily_nav'], d['trades'], d['events']
    expected_months = len(pd.period_range(start, end, freq='M'))
    if len(monthly) != expected_months or monthly.net_return.isna().any():
        raise ValueError(f'Incomplete period {key}')
    if len(events) and events.event.eq('unverified_stale_mark').any():
        raise ValueError(f'Unverified held mark in {key}')
    if (daily.cash < -capital * 1e-10).any():
        raise ValueError(f'Negative cash {key}')
    r = monthly.net_return.to_numpy()
    nav = np.r_[capital, daily.nav.to_numpy()]
    impact = float(trades.get('impact_cost', pd.Series(0., index=trades.index)).fillna(0.).sum())
    orders = order_attempts(d)
    summaries = summarize_attempts(orders, tag)
    overall = summaries.iloc[0]
    risk_exits = trades[trades.side.eq('sell') & trades.reason.eq('ST')]
    final_holdings = d['holdings'][d['holdings'].return_month.eq(end)]
    final_pending = final_holdings[final_holdings.pending_ST.astype(bool)]
    row = {**tag, 'run_id': run_id, 'ledger_verification': verification, 'n_months': len(r),
           'cagr': float(np.prod(1 + r) ** (12 / len(r)) - 1),
           'sharpe': float(np.sqrt(12) * r.mean() / r.std(ddof=1)),
           'daily_max_drawdown': float(np.min(nav / np.maximum.accumulate(nav) - 1)),
           'worst_month': float(r.min()), 'mean_cash': float(daily.cash_weight.mean()),
           'mean_holdings': float(daily.hold_count.mean()), 'max_nav_yuan': float(nav.max()),
           'ending_nav_yuan': float(nav[-1]), 'annual_turnover': float(monthly.turnover.mean() * 12),
           'total_traded_yuan': float(trades.notional.sum()), 'impact_cost_yuan': impact,
           'impact_bps_per_traded_yuan': impact / trades.notional.sum() * 10000,
           'base_cost_yuan': float(trades.cost.sum() - impact),
           'attempt_completion': float(overall.attempt_completion),
           'n_attempts': int(overall.attempts),
           'partial_fill_attempts': int(overall.partial_fill_attempts),
           'zero_fill_attempts': int(overall.zero_fill_attempts),
           'risk_sell_fills': int(len(risk_exits)),
           'max_risk_sell_delay': int(risk_exits.delay_sessions.max()) if len(risk_exits) else 0,
           'final_pending_risk_stocks': int(len(final_pending)),
           'final_pending_risk_weight': float(final_pending.weight.sum()),
           'min_cash_yuan': float(daily.cash.min()), 'runtime_seconds': time.monotonic() - then}
    # These fields are available only in the optional constrained execution layer.
    month_last = daily.assign(month=daily.date.str[:7]).groupby('month', as_index=False).tail(1)
    for field in ['target_shortfall_value', 'target_shortfall_weight', 'pending_sell_value',
                  'replacement_unfilled_budget', 'pending_ST_age_max_sessions',
                  'pending_ST_count', 'pending_ST_value']:
        if field in daily:
            row[f'month_end_mean_{field}'] = float(month_last[field].mean())
            row[f'final_{field}'] = float(daily[field].iloc[-1])
    # Retain full ledgers privately; publish only portfolio aggregates and bounded diagnostics.
    for name, frame in [('monthly', monthly), ('daily', daily), ('orders', orders),
                         ('trades', trades), ('events', events), ('attempt_summary', summaries),
                         ('order_plans', d['order_plans'])]:
        frame.to_csv(path / f'{name}.csv.gz', index=False, compression='gzip')
    row['ledger_hashes'] = {f'{name}.csv.gz':sha(path/f'{name}.csv.gz')
                           for name in ['monthly','daily','orders','trades','events','attempt_summary','order_plans']}
    done.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return row


def aggregate(cache_dir, run_id, jobs, output):
    rows, months, days, attempts, reasons, burdens, checks = [], [], [], [], [], [], []
    for period, scenario, capital, _, _ in jobs:
        path = cache_dir / f'{period}_{scenario}_{capital}'
        row = json.loads((path / 'result.json').read_text())
        assert row['run_id'] == run_id
        rows.append({k:v for k,v in row.items() if k not in ['run_id','runtime_seconds','ledger_hashes','ledger_verification']})
        tag = {k:row[k] for k in ['period','scenario','initial_capital']}
        checks.append({**tag, **row['ledger_verification']})
        months.append(pd.read_csv(path / 'monthly.csv.gz').assign(**tag))
        daily = pd.read_csv(path / 'daily.csv.gz')
        keep = [x for x in ['date', 'nav', 'cash', 'cash_weight', 'hold_count', 'impact_cost',
                            'target_shortfall_value', 'target_shortfall_weight', 'pending_sell_value',
                            'replacement_unfilled_budget', 'pending_ST_age_max_sessions',
                            'pending_ST_count', 'pending_ST_value'] if x in daily]
        days.append(daily[keep].assign(**tag))
        attempts.append(pd.read_csv(path / 'attempt_summary.csv.gz'))
        orders = pd.read_csv(path / 'orders.csv.gz')
        if scenario == 'MAIN_1P' and 'adv20' in orders:
            diagnostic = orders[orders.adv20.gt(0)].copy()
            diagnostic['requested_adv_ratio'] = diagnostic.requested_reference_notional / diagnostic.adv20
            diagnostic['filled_adv_ratio'] = diagnostic.filled_reference_notional / diagnostic.adv20
            diagnostic['attempt_completion'] = diagnostic.filled_units / diagnostic.planned_units
            cols = ['date','stock_code','side','reason','requested_adv_ratio','filled_adv_ratio','attempt_completion']
            burdens.append(diagnostic.nlargest(3, 'requested_adv_ratio')[cols].assign(**tag))
        reason_col = 'constraint_reason' if 'constraint_reason' in orders else 'constraint'
        if reason_col in orders:
            counts = orders.groupby(['side', 'reason', reason_col], dropna=False).agg(
                n_attempts=('planned_units','size'), requested_reference_notional=('requested_reference_notional','sum'),
                filled_reference_notional=('filled_reference_notional','sum')).reset_index()
            counts = counts.rename(columns={reason_col:'constraint_reason'})
            reasons.append(counts.assign(**tag))
    performance = pd.DataFrame(rows).sort_values(['period','initial_capital','scenario'])
    performance.to_csv(output / 'performance.csv',index=False)
    pd.DataFrame(checks).to_csv(output/'ledger_checks.csv',index=False)
    pd.concat(months,ignore_index=True).to_csv(output / 'monthly.csv',index=False)
    pd.concat(days,ignore_index=True).to_csv(output / 'daily.csv.gz',index=False,compression='gzip')
    pd.concat(attempts,ignore_index=True).to_csv(output / 'attempt_summary.csv',index=False)
    if reasons:
        pd.concat(reasons,ignore_index=True).to_csv(output / 'constraint_summary.csv',index=False)
    if burdens:
        pd.concat(burdens,ignore_index=True).to_csv(output / 'largest_order_burdens.csv',index=False)
    differences = []
    for (period, capital), g in performance.groupby(['period','initial_capital']):
        g = g.set_index('scenario')
        for scenario, reference, interpretation in [
            ('VWAP_BASE','OPEN_BASE','execution_reference_and_eligibility'),
            ('CAP_1P','VWAP_BASE','capacity_and_delay'),
            ('CAP_1P_SIGMA','CAP_1P','volatility_data_coverage'),
            ('MAIN_1P','CAP_1P_SIGMA','additional_impact_and_holdings_path'),
        ]:
            if scenario not in g.index or reference not in g.index:
                continue
            row = {'period':period,'initial_capital':capital,'scenario':scenario,'reference':reference,
                   'interpretation':interpretation}
            for metric in ['cagr','sharpe','daily_max_drawdown','mean_cash','attempt_completion']:
                row[metric+'_difference'] = g.loc[scenario,metric] - g.loc[reference,metric]
            differences.append(row)
    if differences:
        pd.DataFrame(differences).to_csv(output / 'performance_differences.csv',index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--liquidity-file', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--periods', nargs='+')
    parser.add_argument('--scenarios', nargs='+')
    parser.add_argument('--capitals', nargs='+', type=int)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    require_full_inputs(data_root, 'evaluate')
    liquidity_file = args.liquidity_file or data_root / 'data/capacity/capacity_market.npz'
    if not liquidity_file.is_file():
        raise FileNotFoundError('Run prepare_capacity_inputs.py first; unknown capacity inputs cannot be filled with zeros.')
    protocol = json.loads(PROTOCOL_PATH.read_text())
    frozen = json.loads((ROOT/'results/selection.json').read_text())
    config = json.loads((ROOT/'config.json').read_text())
    if protocol['winner'] != frozen['winner'] or protocol['base_cost_bps'] != config['main_cost_bps']:
        raise ValueError('Capacity protocol differs from frozen winner/cost')
    if protocol['periods'] != {'TRAIN':[config['train_start'],config['train_end']],
                               'TEST':[config['test_start'],config['test_end']]}:
        raise ValueError('Capacity protocol changes frozen evaluation periods')
    sources = {'config.json': sha(ROOT / 'config.json'), 'results/selection.json': sha(ROOT / 'results/selection.json'),
               'results/capacity/protocol.json': sha(PROTOCOL_PATH)}
    for name in ['data/prepared/daily_market.npz', 'data/prepared/panel.csv', 'data/prepared/selection_universe.csv']:
        sources['external_full_project/' + name] = sha(data_root / name)
    sources['external_full_project/data/capacity/capacity_market.npz'] = sha(liquidity_file)
    official_inputs = liquidity_file.resolve() == (data_root/'data/capacity/capacity_market.npz').resolve()
    if official_inputs:
        receipt = json.loads((data_root/'data/capacity/input_manifest.json').read_text())
        if (receipt.get('capacity_market_sha256') != sha(liquidity_file)
                or receipt.get('original_inputs_changed') is not False):
            raise ValueError('Capacity input receipt does not certify this matrix')
        for name in ['data/capacity/input_manifest.json', 'data/capacity/coverage.json']:
            sources['external_full_project/'+name] = sha(data_root/name)
    for name in ['microcap/execution.py', 'microcap/liquidity.py', 'microcap/execution_constraints.py',
                 'microcap/capacity_data.py', 'scripts/run_capacity_stress.py','scripts/capacity_ledger_checks.py']:
        sources[name] = sha(ROOT / name)
    run_id = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    cache_dir = ROOT / '.build/capacity_runs' / run_id[:16]
    cache_dir.mkdir(parents=True,exist_ok=True)
    chosen_periods = args.periods or list(protocol['periods'])
    chosen_scenarios = args.scenarios or list(protocol['scenarios'])
    chosen_capitals = args.capitals or protocol['capitals_yuan']
    if any(len(items) != len(set(items)) for items in [chosen_periods, chosen_scenarios, chosen_capitals]):
        raise ValueError('Duplicate scenario/period/capital arguments are not allowed')
    if set(chosen_periods)-set(protocol['periods']) or set(chosen_scenarios)-set(protocol['scenarios']) or set(chosen_capitals)-set(protocol['capitals_yuan']):
        raise ValueError('Requested job is outside predeclared protocol')
    jobs = [(p,s,c,str(cache_dir),run_id) for p in chosen_periods for c in chosen_capitals for s in chosen_scenarios]
    actual_grid = {(p,s,c) for p,s,c,_,_ in jobs}
    expected_grid = {(p,s,c) for p in protocol['periods'] for s in protocol['scenarios'] for c in protocol['capitals_yuan']}
    full_grid = actual_grid == expected_grid
    if full_grid and not official_inputs:
        raise ValueError('A full release requires the official complete input builder receipt; custom matrices are pilot-only')
    started = datetime.now(timezone.utc).isoformat()
    if args.workers == 1:
        initialize(data_root,liquidity_file)
        for i,job in enumerate(jobs,1):
            row=execute_job(job)
            print(i,len(jobs),job[:3],{k:round(row[k],4) for k in ['cagr','attempt_completion','mean_cash']},flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(data_root,liquidity_file)) as executor:
            futures={executor.submit(execute_job,job):job for job in jobs}
            for i,future in enumerate(as_completed(futures),1):
                row=future.result()
                print(i,len(jobs),futures[future][:3],{k:round(row[k],4) for k in ['cagr','attempt_completion','mean_cash']},flush=True)
    # A pilot may populate the private cache but never replace the published full grid.
    output = ROOT/'results/capacity' if full_grid else cache_dir/'pilot'
    # Do not publish a mixed-code or mixed-input run, even if each worker succeeded.
    verify_sources(sources, data_root, liquidity_file)
    output.mkdir(parents=True, exist_ok=True)
    aggregate(cache_dir,run_id,jobs,output)
    if full_grid:
        shutil.copyfile(data_root/'data/capacity/coverage.json', output/'coverage.json')
    manifest={'run_id':run_id,'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
              'full_predeclared_grid':full_grid,'n_scenarios':len(jobs),'inputs':sources,
              'outputs':{p.name:sha(p) for p in output.iterdir() if p.suffix in ['.csv','.gz'] or p.name == 'coverage.json'},
              'scope':'Historical daily execution pressure scenarios; raw inputs/full ledgers are not public; no live fill certification.'}
    verify_sources(sources, data_root, liquidity_file)
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Completed',len(jobs),'jobs; full grid:',full_grid,flush=True)


if __name__=='__main__':
    main()
