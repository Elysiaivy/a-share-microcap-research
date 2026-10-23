"""Small, isolated availability audit; never changes frozen research inputs."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microcap.acquire import FrozenStore, dump


def main():
    target = ROOT / 'data/probes/tushare_gap_probe'
    store = FrozenStore(target, sources=[], online=True)
    queries = [
        ('industry_same_day', 'bak_basic', {'trade_date': '20210129', 'limit': 7000}, 'ts_code,trade_date,name,industry'),
        ('industry_prior_day', 'bak_basic', {'trade_date': '20210128', 'limit': 7000}, 'ts_code,trade_date,name,industry'),
        ('backup_same_day', 'bak_daily', {'trade_date': '20210129', 'ts_code': '002289.SZ'}, 'ts_code,trade_date,close,total_mv,total_share,industry'),
        ('mv_same_day', 'daily_basic', {'trade_date': '20210129', 'ts_code': '002289.SZ'}, 'ts_code,trade_date,close,total_mv,total_share'),
        ('factor_short_suspend', 'stk_factor_pro', {'trade_date': '20210129', 'ts_code': '002289.SZ'}, 'ts_code,trade_date,close,close_hfq,adj_factor,total_mv,total_share'),
        ('factor_long_suspend', 'stk_factor_pro', {'trade_date': '20201231', 'ts_code': '000670.SZ'}, 'ts_code,trade_date,close,close_hfq,adj_factor,total_mv,total_share'),
        ('basic_long_suspend', 'bak_basic', {'trade_date': '20201231', 'ts_code': '000670.SZ'}, 'ts_code,trade_date,name,industry,total_share'),
        ('daily_long_suspend', 'daily_basic', {'trade_date': '20201231', 'ts_code': '000670.SZ'}, 'ts_code,trade_date,close,total_mv,total_share'),
        ('monthly_delisting', 'monthly', {'ts_code': '000511.SZ', 'start_date': '20180601', 'end_date': '20180731'}, 'ts_code,trade_date,close,pre_close,pct_chg'),
        ('daily_delisting', 'daily', {'ts_code': '000511.SZ', 'start_date': '20180701', 'end_date': '20180731'}, 'ts_code,trade_date,close,vol'),
        ('premarket_new_gap', 'stk_premarket', {'ts_code': '600647.SH', 'trade_date': '20240628'}, 'ts_code,trade_date,total_share,pre_close'),
    ]
    summary = []
    universe = pd.read_csv(ROOT/'results/unresolved_universe.csv')
    need_industry = set(universe.loc[universe.is_financial.isna(), 'ts_code'])
    for label, api, params, fields in queries:
        record = dict(label=label, api=api, params=params)
        try:
            frame = store.query(api, params, fields)
            frame.to_csv(target/f'{label}.csv', index=False)
            record.update(rows=len(frame), source_sha256=frame.attrs.get('source_sha256'))
            if label.startswith('industry'):
                subset = frame[frame.ts_code.isin(need_industry)] if 'ts_code' in frame else frame
                record['gap_codes_matched'] = len(subset)
                record['nonmissing_industry'] = int(subset.industry.notna().sum()) if 'industry' in subset else 0
            else:
                record['sample'] = json.loads(frame.head(3).to_json(orient='records'))
        except Exception as exc:
            # FrozenStore strips credential-bearing provider diagnostics.
            record['error'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
        summary.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        store.save_manifest()
        dump(target/'summary.json', summary)


if __name__ == '__main__':
    main()
