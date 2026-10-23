"""Verify the compact public release without market data or network access."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from microcap.portfolio import select_winner
from microcap.analysis import metrics, _hac


def verify_results(root=ROOT):
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    frozen = json.loads((root/'results/selection.json').read_text())
    monthly = pd.read_csv(root/'results/monthly.csv', float_precision='round_trip')
    primary = monthly[monthly.cost_bps.eq(config['main_cost_bps'])]
    selected = select_winner(primary, config)
    assert selected['winner'] == frozen['winner']
    assert selected['training_hash'] == frozen['training_hash']
    scores = pd.read_csv(root/'results/selection_scores.csv').set_index('strategy')
    for band, value in selected['scores'].items():
        np.testing.assert_allclose(value['score'], scores.loc[band, 'score'], rtol=1e-11)
    recalculated = metrics(monthly, config)
    saved = pd.read_csv(root/'results/performance.csv')
    keys = ['strategy', 'cost_bps', 'period']
    both = recalculated.merge(saved, on=keys, suffixes=('_new','_saved'), validate='one_to_one')
    assert len(both) == len(saved)
    for metric in ['cagr','sharpe','worst_month']:
        np.testing.assert_allclose(both[metric+'_new'], both[metric+'_saved'], rtol=1e-10, atol=1e-12, equal_nan=True)
    wide = primary.pivot(index='return_month',columns='strategy',values='net_return')
    wide.index = pd.PeriodIndex(wide.index, freq='M')
    pairs = json.loads((root/'results/paired_tests.json').read_text())['comparisons']
    for row in pairs:
        period = row['period']
        bounds = {'TRAIN': (config['train_start'],config['train_end']),
                  'TEST': (config['test_start'],config['test_end']),
                  'ALL': (config['train_start'],config['test_end'])}
        if period not in bounds:
            continue
        expected = pd.period_range(*bounds[period], freq='M')
        result = _hac(wide[row['strategy']]-wide[row['comparator']], expected, config['hac_lags'])
        for key in ['mean','hac_se','ci95_low','ci95_high']:
            np.testing.assert_allclose(result[key],row[key],atol=1e-12,rtol=1e-10)
    # Recompute daily drawdown, including the wealth at the start of each period.
    daily = pd.read_csv(root/'results/daily_nav.csv')
    daily['date'] = pd.to_datetime(daily.date, format='mixed')
    for band,g in daily.groupby('strategy'):
        g = g.sort_values('date')
        for period,start,end in [('TRAIN','2015-01-01','2023-12-31'),('TEST','2024-01-01','2025-12-31')]:
            prior = g[g.date < start]
            initial = prior.nav.iloc[-1] if len(prior) else 1.0
            nav = np.r_[initial,g[g.date.between(start,end)].nav.to_numpy()]
            dd = (nav/np.maximum.accumulate(nav)-1).min()
            published = saved.query('strategy == @band and period == @period and cost_bps == 30').iloc[0]
            np.testing.assert_allclose(dd,published.daily_max_drawdown,atol=1e-10)
    annual = primary.assign(year=primary.return_month.str[:4].astype(int)).groupby(['year','strategy']).net_return.agg(lambda x: np.prod(1+x)-1).unstack()
    saved_annual = pd.read_csv(root/'results/applicability/annual_returns.csv').set_index('year')
    np.testing.assert_allclose(annual.loc[saved_annual.index,saved_annual.columns], saved_annual,atol=1e-12)
    return {'winner': selected['winner'], 'train_months': len(selected['train_months']),
            'performance_rows': len(both), 'paired_comparisons':len(pairs),
            'verification_scope':'published results; not a raw-data backtest'}


def verify_manifest(root=ROOT):
    root = Path(root)
    manifest = json.loads((root/'PUBLIC_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Path escapes root: {name}')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Public artifact missing or changed: {name}')
    return len(manifest['files'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',action='store_true')
    args = parser.parse_args()
    result = verify_results()
    if args.manifest:
        result['public_files_verified'] = verify_manifest()
    print(json.dumps(result,ensure_ascii=False,indent=2))
