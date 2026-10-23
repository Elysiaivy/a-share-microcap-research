"""Frozen external price-index benchmark, aligned to the research calendar.

The benchmark is an index level comparison: no dividends, implementation fees,
or strategy turnover are invented for the published price index.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import FrozenStore, dump, sha

DEFAULT_BENCHMARK = {'code': '000852.SH', 'name': '中证1000', 'return_type': 'price_index'}
FIELDS = 'ts_code,trade_date,close'


def _window(config, calendar):
    first = pd.Period(config['train_start'], freq='M')
    last = pd.Period(config['test_end'], freq='M')
    baseline = first - 1
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values()
    if calendar.has_duplicates:
        raise ValueError('Benchmark calendar contains duplicate dates')
    earlier = calendar[calendar.to_period('M') == baseline]
    if earlier.empty:
        raise ValueError('Benchmark calendar has no baseline month end')
    end_days = calendar[calendar.to_period('M') == last]
    if end_days.empty:
        raise ValueError('Benchmark calendar has no final month end')
    dates = calendar[(calendar >= earlier[-1]) & (calendar <= end_days[-1])]
    months = pd.period_range(first, last, freq='M')
    expected_periods = pd.period_range(baseline, last, freq='M')
    if not pd.PeriodIndex(dates.to_period('M').unique()).equals(expected_periods):
        raise ValueError('Benchmark calendar has missing months')
    return dates, months


def monthly_returns(daily, calendar, config, code='000852.SH'):
    """Validate every expected session; then compute month-end close ratios."""
    dates, months = _window(config, calendar)
    required = {'trade_date', 'ts_code', 'close'}
    if not required.issubset(daily.columns):
        raise ValueError('Benchmark daily schema is incomplete')
    frame = daily[list(required)].copy()
    frame['trade_date'] = pd.to_datetime(frame.trade_date, errors='raise').dt.normalize()
    if frame.trade_date.duplicated().any():
        raise ValueError('Benchmark contains duplicate daily dates')
    if not frame.ts_code.eq(code).all():
        raise ValueError('Benchmark contains the wrong index code')
    frame['close'] = pd.to_numeric(frame.close, errors='raise')
    if not np.isfinite(frame.close).all() or not frame.close.gt(0).all():
        raise ValueError('Benchmark close must be finite and strictly positive')
    frame = frame.set_index('trade_date').sort_index()
    expected = dates
    if not frame.index.equals(expected):
        missing = expected.difference(frame.index)
        extra = frame.index.difference(expected)
        raise ValueError(f'Benchmark daily calendar mismatch: missing={len(missing)}, extra={len(extra)}')
    endpoints = frame.groupby(frame.index.to_period('M')).tail(1)
    values = endpoints.close.to_numpy()
    if len(values) != len(months) + 1:
        raise ValueError('Benchmark month-end series is incomplete')
    returns = values[1:] / values[:-1] - 1
    return pd.DataFrame({
        'return_month': months.astype(str), 'strategy': 'CSI1000',
        'gross_return': returns, 'net_return': returns, 'turnover': np.nan,
        'cost': 0.0, 'benchmark_close': values[1:],
        'benchmark_previous_close': values[:-1],
        'benchmark_date': endpoints.index[1:].strftime('%Y-%m-%d'),
    })


def prepare_benchmark(root):
    """Return complete monthly benchmark data, obtaining only uncached dates.

    Existing frozen files are verified and reused offline. An initial acquisition
    uses the project's existing Tushare environment/keychain credential, freezes
    provider responses with FrozenStore, and records exact source hashes.
    """
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    benchmark = {**DEFAULT_BENCHMARK, **config.get('benchmark', {})}
    if benchmark['return_type'] != 'price_index':
        raise ValueError('Only an explicitly labelled price-index benchmark is supported')
    code = benchmark['code']
    calendar = pd.to_datetime(pd.read_csv(root / 'data/prepared/calendar.csv').trade_date)
    dates, months = _window(config, calendar)
    prepared = root / 'data/prepared'
    daily_path = prepared / 'benchmark_daily.csv'
    monthly_path = prepared / 'benchmark_monthly.csv'
    manifest_path = prepared / 'benchmark_manifest.json'
    specification = dict(code=code, name=benchmark['name'], return_type='price_index',
                         start_date=dates[0].strftime('%Y-%m-%d'),
                         end_date=dates[-1].strftime('%Y-%m-%d'),
                         train_start=config['train_start'], test_end=config['test_end'])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('specification') != specification:
            raise ValueError('Frozen benchmark specification changed; acquire a separate certified input')
        for entry in [*manifest['artifacts'], *manifest['raw_inputs'], manifest['calendar']]:
            if sha(root / entry['path']) != entry['sha256']:
                raise ValueError(f'Frozen benchmark checksum mismatch: {entry["path"]}')
        daily = pd.read_csv(daily_path, dtype={'ts_code': str})
        monthly = monthly_returns(daily, calendar, config, code)
        persisted = pd.read_csv(monthly_path)
        pd.testing.assert_frame_equal(monthly, persisted, check_dtype=False, rtol=1e-12)
        return monthly

    store = FrozenStore(root, online=True)
    pieces = []
    # Reuse overlapping cached official daily series, accepting extra fields but
    # never substituting a stock, ETF, or total-return index for this price index.
    for entry in store.find('index_daily', ts_code=code):
        body = entry['request']['body']
        if not set(FIELDS.split(',')).issubset(body.get('fields', '').split(',')):
            continue
        params = body.get('params', {})
        if str(params.get('end_date', '99999999')) < dates[0].strftime('%Y%m%d'):
            continue
        if str(params.get('start_date', '00000000')) > dates[-1].strftime('%Y%m%d'):
            continue
        part = store.read(entry)[FIELDS.split(',')]
        part['trade_date'] = pd.to_datetime(part.trade_date, format='%Y%m%d', errors='raise')
        part = part[part.trade_date.isin(dates)]
        if len(part):
            pieces.append(part)
    covered = pd.DatetimeIndex(pd.concat(pieces).trade_date.unique()) if pieces else pd.DatetimeIndex([])
    missing = dates.difference(covered)
    if len(missing):
        # One eleven-year index series is below the documented response limit.
        # Query the enclosing missing interval, then reject truncation/gaps below.
        part = store.query('index_daily', dict(ts_code=code,
                           start_date=missing[0].strftime('%Y%m%d'),
                           end_date=missing[-1].strftime('%Y%m%d'), limit=6000, offset=0), FIELDS)
        if len(part) >= 6000:
            raise ValueError('Benchmark response reached pagination limit')
        part['trade_date'] = pd.to_datetime(part.trade_date, format='%Y%m%d', errors='raise')
        pieces.append(part[FIELDS.split(',')])
    if not pieces:
        raise ValueError('No benchmark observations returned')
    daily = pd.concat(pieces, ignore_index=True)
    # Overlap is permitted only when all official close values agree exactly.
    if daily.groupby('trade_date').close.nunique(dropna=False).gt(1).any():
        raise ValueError('Overlapping benchmark sources disagree')
    daily = daily.drop_duplicates('trade_date').sort_values('trade_date')
    monthly = monthly_returns(daily, calendar, config, code)
    daily.to_csv(daily_path, index=False, date_format='%Y-%m-%d')
    monthly.to_csv(monthly_path, index=False)
    store.save_manifest()
    manifest = dict(
        version=1, specification=specification, provider='tushare', api='index_daily',
        documentation='https://tushare.pro/document/2?doc_id=95',
        return_convention='Month-end closing price-index ratio; excludes dividends; no investment implementation fees or turnover',
        cash_return=0, benchmark_implementation_cost_applied=False,
        daily_rows=len(daily), monthly_rows=len(monthly), missing_sessions=0,
        artifacts=[dict(path=str(p.relative_to(root)), sha256=sha(p)) for p in (daily_path, monthly_path)],
        raw_inputs=sorted(store.used.values(), key=lambda item: item['path']),
        calendar=dict(path='data/prepared/calendar.csv', sha256=sha(root / 'data/prepared/calendar.csv')),
    )
    dump(manifest_path, manifest)
    return monthly
