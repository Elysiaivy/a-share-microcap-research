"""Extend the existing early suspension capital ledger without future data.

The last observed capitalization is a labelled estimate, not a contemporaneous
capitalization certification. It never uses today's share count or later prices.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from microcap.acquire import FrozenStore, dump, sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--online',action='store_true');args=parser.parse_args()
    store=FrozenStore(ROOT,online=args.online)
    universe=pd.read_csv(ROOT/'data/prepared/universe.csv',low_memory=False)
    gaps=universe[universe.is_financial.eq(0)&universe.market_cap.isna()].copy()
    if gaps.empty:
        print('No missing nonfinancial capitalization rows; existing frozen ledger retained')
        return
    market=np.load(ROOT/'data/prepared/daily_market.npz',allow_pickle=False)
    closes=market['close']
    dates=pd.DatetimeIndex(market['dates']); index={code:j for j,code in enumerate(market['codes'])}
    rows=[];queries={}
    try:
        for row in gaps.itertuples():
            day=pd.Timestamp(row.signal_date);j=index[row.stock_code]
            valid=(dates<=day)&np.isfinite(closes[:,j])&(closes[:,j]>0)
            if not valid.any():raise ValueError('No frozen prior quote for '+row.stock_code)
            observed=dates[valid][-1]
            if observed>day:raise AssertionError('Future capitalization source')
            ds=observed.strftime('%Y%m%d')
            if ds not in queries:
                queries[ds]=store.date_query('daily_basic',ds,'ts_code,trade_date,total_mv')
            source=queries[ds];hit=source[source.ts_code.eq(row.stock_code)]
            if len(hit)!=1 or not float(hit.iloc[0].total_mv)>0:
                raise ValueError('No uniquely verified source capitalization '+row.stock_code+' '+ds)
            rows.append(dict(ts_code=row.stock_code,signal_date=str(day.date()),
                market_cap=float(hit.iloc[0].total_mv)*10000,source_date=str(observed.date()),
                source_sha256=source.attrs['source_sha256'],
                source='last_observed_capitalization_during_suspension',stale_days=(day-observed).days))
        directory=ROOT/'data/repair_sources/early';target=directory/'market_cap.csv'
        prior=pd.read_csv(target)
        supplement=pd.DataFrame(rows)
        combined=pd.concat([prior,supplement],ignore_index=True).drop_duplicates(['ts_code','signal_date'],keep='last')
        combined.sort_values(['signal_date','ts_code']).to_csv(target,index=False)
        point=ROOT/'data/repair_sources/point_in_time';point.mkdir(parents=True,exist_ok=True)
        supplement.to_csv(point/'newly_eligible_capitalization_sources.csv',index=False)
        dump(directory/'manifest.json',dict(artifacts=[dict(path=str(target.relative_to(ROOT)),sha256=sha(target))],
            policy='Last observed historical daily_basic capitalization for frozen suspension gaps; no future observations or current shares; stale dates explicitly disclosed',
            limitation='An estimated ranking input; not certified contemporaneous capitalization and not a zero-width confidence interval'))
        print('Added',len(supplement),'rows;',len(combined),'total; maximum new staleness',int(supplement.stale_days.max()),'days')
    finally:
        store.save_manifest()


if __name__=='__main__':main()
