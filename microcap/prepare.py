"""Build the historical panel without letting outcome coverage alter rankings."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .acquire import FrozenStore, default_sources, dump, sha
from .industry_repair import repair_financial_classification
from .features import profit_ttm, st_labels
from .portfolio import rank_universe
from .valuation_repair import load_valuations
from .rank_uncertainty import rank_bounds

MAINBOARD = r'(?:(?:600|601|603|605)\d{3}\.SH|(?:000|001|002|003)\d{3}\.SZ)'
FINANCIAL = {'银行','非银金融','非银行金融','金融','银行业','保险','证券','多元金融'}


def aliases(frame, day, config):
    """Join verified legacy codes to a stable provider ID, without double counts."""
    frame = frame.copy()
    frame['source_ts_code'] = frame.ts_code
    for alias in config.get('security_aliases',[]):
        if pd.Timestamp(day) < pd.Timestamp(alias['effective_date']):
            frame.loc[frame.ts_code.eq(alias['old']),'ts_code'] = alias['canonical']
    if frame.ts_code.duplicated().any():
        compare=[c for c in ['trade_date','close','total_mv','total_share','adj_factor','amount'] if c in frame]
        for code,rows in frame[frame.ts_code.duplicated(False)].groupby('ts_code'):
            if not compare or len(rows[compare].drop_duplicates())!=1:
                raise ValueError(f'Conflicting alias observations for {code} on {day}')
        # Some vendor history explicitly contains both old and new codes with
        # identical numbers. Preserve only one issuer after checking equality.
        frame=frame.assign(_legacy=frame.source_ts_code.ne(frame.ts_code)).sort_values('_legacy').drop_duplicates('ts_code').drop(columns='_legacy')
    return frame


def dt(series):
    return pd.to_datetime(series.astype('string').str.replace(r'\.0$', '', regex=True),format='mixed',errors='coerce')


def catalog(store):
    securities = []
    for status in ['L','D','P']:
        entries = store.find('stock_basic',list_status=status,offset=0)
        if not entries:
            raise FileNotFoundError(f'No complete stock_basic {status} snapshot')
        e = entries[-1]
        rows = store.read(e)
        if len(rows) >= int(e['request']['body']['params'].get('limit',6000)):
            raise ValueError('Incomplete stock_basic pagination')
        securities.append(rows)
    securities = pd.concat(securities,ignore_index=True).drop_duplicates('ts_code')
    for c in ['list_date','delist_date']:
        securities[c] = dt(securities[c])
    calendar = pd.concat([store.read(e) for e in store.find('trade_cal')],ignore_index=True)
    calendar['cal_date'] = dt(calendar.cal_date)
    conflicting = calendar.groupby('cal_date').is_open.nunique().gt(1)
    if conflicting.any():
        raise ValueError('Conflicting calendars')
    calendar = pd.DatetimeIndex(calendar.loc[pd.to_numeric(calendar.is_open).eq(1),'cal_date'].drop_duplicates().sort_values())
    # All 31 SW industries, current and historical members, not today's label.
    industry = pd.concat([store.read(e) for e in store.find('index_member_all')],ignore_index=True).drop_duplicates()
    industry = industry.rename(columns={'l1_name':'industry'})
    for c in ['in_date','out_date']:
        industry[c] = dt(industry[c])
    # Use one latest, complete offset sequence; never mix pages of two vintages.
    name_entries = store.find('namechange',offset=0)
    first = name_entries[-1]
    parent = first.get('source_manifest')
    pages = [e for e in store.find('namechange') if e.get('source_manifest') == parent and e['request']['body']['params'].get('limit') == first['request']['body']['params'].get('limit')]
    pages = sorted(pages,key=lambda e:int(e['request']['body']['params'].get('offset',0)))
    offset, names = 0, []
    for entry in pages:
        p = entry['request']['body']['params']
        if int(p.get('offset',0)) != offset:
            raise ValueError('Non-contiguous name history pagination')
        rows = store.read(entry); names.append(rows)
        if len(rows) < int(p['limit']):
            break
        offset += int(p['limit'])
    else:
        raise ValueError('No terminating name history page')
    return securities, calendar, industry, pd.concat(names,ignore_index=True).drop_duplicates()


def universe(store, config, securities, calendar, industry):
    repairs = load_valuations(store.root)
    months = pd.period_range(pd.Period(config['train_start'])-2,config['test_end'],freq='M')
    ends = {str(m): calendar[calendar.to_period('M') == m][-1] for m in months}
    signals = pd.period_range(pd.Period(config['train_start'])-1,pd.Period(config['test_end'])-1,freq='M')
    master = securities[securities.ts_code.str.fullmatch(MAINBOARD)].copy()
    panels, coverage = [], []
    for m in signals:
        signal, horizon = ends[str(m)],ends[str(m+1)]
        code_date = signal.strftime('%Y%m%d')
        active = master[master.list_date.le(signal) & (master.delist_date.isna() | master.delist_date.gt(signal))].copy()
        active['listing_sessions'] = calendar.searchsorted(signal,side='right') - calendar.searchsorted(active.list_date,side='left')
        active = active[active.listing_sessions.ge(config['min_listing_sessions'])]
        # Retrospective SW membership is not evidence of the business known on
        # this date. Both exclusion and controls use dated historical sources.
        active['industry'] = pd.Series(None, index=active.index, dtype=object)
        active['is_financial'] = np.nan
        active['financial_source'] = 'unknown'
        market = store.date_query('daily_basic',code_date,'ts_code,trade_date,total_mv')
        market = aliases(market,signal,config)
        if market.ts_code.duplicated().any():
            raise ValueError(f'Duplicate market snapshot {signal}')
        active = active.merge(market[['ts_code','total_mv']],on='ts_code',how='left',validate='one_to_one')
        active['market_cap'] = pd.to_numeric(active.total_mv,errors='coerce')*10000  # 万元 -> 元
        active.loc[~np.isfinite(active.market_cap) | active.market_cap.le(0),'market_cap'] = np.nan
        active['market_cap_source'] = np.where(active.market_cap.notna(),'daily_basic','missing')
        quote_entries = store.find('bak_daily',trade_date=code_date,offset=0)
        backup_quotes = None
        if quote_entries:
            backup_quotes = aliases(store.read(quote_entries[-1]),signal,config).set_index('ts_code')
            # Backup units are 亿元, independently cross-checked against daily_basic.
            backup_mv = active.ts_code.map(backup_quotes.total_mv)*1e8
            use = active.market_cap.isna() & backup_mv.gt(0)
            active.loc[use,'market_cap'] = backup_mv[use]
            active.loc[use,'market_cap_source'] = 'bak_daily_rounded'
        active['market_cap_lower'] = active.market_cap
        active['market_cap_upper'] = active.market_cap
        if len(repairs):
            repair = repairs[repairs.trade_date.eq(signal.strftime('%Y-%m-%d'))].set_index('ts_code')
            estimates = active.ts_code.map(repair.market_cap)
            use = active.market_cap.isna() & np.isfinite(estimates) & estimates.gt(0)
            for column in ['market_cap','market_cap_lower','market_cap_upper','market_cap_source']:
                active.loc[use,column] = active.ts_code.map(repair[column])[use]
            active.loc[use,'market_cap_evidence'] = active.ts_code.map(repair.source_sha256)[use]
        active = repair_financial_classification(store, active, signal, calendar, FINANCIAL,
            config.get('repair_policy',{}).get('financial_snapshot_max_lag_sessions',1), config)
        early_dir=store.root/'data/repair_sources/early'
        if (early_dir/'market_cap.csv').exists():
            early=pd.read_csv(early_dir/'market_cap.csv')
            early=early[early.signal_date.eq(str(signal.date()))].set_index('ts_code')
            use=active.market_cap.isna() & active.ts_code.isin(early.index)
            active.loc[use,'market_cap']=active.ts_code.map(early.market_cap)[use]
            active.loc[use,'market_cap_source']='asof_last_observed_capitalization'
            active.loc[use,'capital_source_date']=active.ts_code.map(early.source_date)[use]
            active.loc[use,'market_cap_lower']=active.market_cap[use]
            active.loc[use,'market_cap_upper']=active.market_cap[use]
        active['signal_date'],active['horizon_end'] = signal,horizon
        active['return_month'] = str(m+1)
        active['stock_code'] = active.ts_code
        active['historical_code'] = active.stock_code
        for alias in config.get('security_aliases',[]):
            if signal < pd.Timestamp(alias['effective_date']):
                active.loc[active.stock_code.eq(alias['canonical']),'historical_code'] = alias['old']
        active['eligible'] = active.is_financial.eq(0)
        bounds=rank_bounds(active)
        active=active.merge(bounds,on='stock_code',how='left',validate='one_to_one')
        # Missing caps are retained in universe.csv and block certified rankings.
        coverage.append(dict(signal_date=signal,return_month=str(m+1),n_age_eligible=len(active),
                             n_finance=int(active.is_financial.eq(1).sum()),
                             n_finance_unknown=int(active.is_financial.isna().sum()),
                             n_repair_boundary_uncertain=int(active.band_uncertain.eq(True).sum()),
                             n_backup_market_cap=int(active.market_cap_source.eq('bak_daily_rounded').sum()),
                             n_unmapped_historical_codes=0 if backup_quotes is None else int((backup_quotes.total_mv.gt(0)&backup_quotes.index.str.fullmatch(MAINBOARD)&~backup_quotes.index.isin(master.ts_code)).sum()),
                             n_nonfinance_mv_missing=int((active.is_financial.eq(0)&(~np.isfinite(active.market_cap)|active.market_cap.le(0))).sum())))
        panels.append(active)
    combined = pd.concat(panels,ignore_index=True)
    evidence_dir = store.root/'data/repair_sources/point_in_time'
    evidence_dir.mkdir(parents=True,exist_ok=True)
    industry_coverage=combined.groupby(['return_month','financial_source'],dropna=False).size().rename('n_company_months').reset_index()
    industry_coverage.to_csv(evidence_dir/'industry_coverage.csv',index=False)
    prepared_dir=store.root/'data/prepared';prepared_dir.mkdir(parents=True,exist_ok=True)
    industry_coverage.to_csv(prepared_dir/'historical_industry_coverage.csv',index=False)
    case_codes = ['000617.SZ','002423.SZ','000627.SZ','600093.SH']
    columns = ['return_month','signal_date','ts_code','industry','is_financial','financial_source',
               'financial_source_date','financial_source_sha256','industry_available_date']
    combined.loc[combined.ts_code.isin(case_codes),[c for c in columns if c in combined]].to_csv(evidence_dir/'industry_repair_cases.csv',index=False)
    return combined, pd.DataFrame(coverage), ends


def price_features(store, ranked, calendar, ends, config):
    """Raw endpoint quotes and a fixed 20-exchange-session ADV window.

    A separately audited suspension mark can supply an explicitly labelled
    research valuation. A suspension cannot certify delisting recovery proceeds.
    """
    endpoints = {}
    repairs = load_valuations(store.root)
    endpoint_dates = sorted(set(ranked.signal_date)|set(ranked.horizon_end)|{ends[str(pd.Period(m)-2)] for m in ranked.return_month.unique()})
    for day in endpoint_dates:
        q = store.date_query('daily',day.strftime('%Y%m%d'),'ts_code,trade_date,close,amount')
        a = store.date_query('adj_factor',day.strftime('%Y%m%d'),'ts_code,trade_date,adj_factor')
        if q.ts_code.duplicated().any() or a.ts_code.duplicated().any():
            raise ValueError(f'Duplicate quotes/factors at {day}')
        z = q.merge(a[['ts_code','adj_factor']],on='ts_code',how='outer',validate='one_to_one').set_index('ts_code')
        z['valuation_source'] = np.where(z.close.notna(),'daily_close','unresolved')
        backup_entries = store.find('bak_daily',trade_date=day.strftime('%Y%m%d'),offset=0)
        susp_entries = store.find('suspend_d',trade_date=day.strftime('%Y%m%d'),offset=0)
        if backup_entries and susp_entries:
            backup = aliases(store.read(backup_entries[-1]),day,config).set_index('ts_code')
            susp = store.read(susp_entries[-1])
            whole = susp.suspend_type.eq('S') & (susp.suspend_timing.isna()|susp.suspend_timing.eq(''))
            quote = z.index.map(backup.close)
            use = z.close.isna() & pd.Series(quote,index=z.index).gt(0) & z.index.isin(susp.loc[whole,'ts_code'])
            z.loc[use,'close'] = quote[use]
            z.loc[use,'valuation_source'] = 'backup_close_confirmed_suspension'
        if len(repairs):
            repair = repairs[repairs.trade_date.eq(day.strftime('%Y-%m-%d'))].set_index('ts_code')
            z = z.reindex(z.index.union(repair.index))
            mark = z.index.to_series().map(repair.close)
            use = z.close.isna() & np.isfinite(mark) & mark.gt(0)
            factor = z.index.to_series().map(repair.adj_factor)
            comparable = use & z.adj_factor.notna() & factor.notna()
            if not np.isclose(z.loc[comparable,'adj_factor'],factor[comparable],rtol=1e-10,atol=1e-10).all():
                raise ValueError(f'Suspension repair conflicts with frozen adjustment factor at {day}')
            z.loc[use,'close'] = mark[use]
            z.loc[use,'adj_factor'] = factor[use]
            z.loc[use,'valuation_source'] = z.index.to_series().map(repair.valuation_source)[use]
        z['adjusted_price'] = pd.to_numeric(z.close)*pd.to_numeric(z.adj_factor)
        endpoints[day] = z
    frames, gap = [], []
    for number,(signal, stocks) in enumerate(ranked.groupby('signal_date',sort=True)):
        stocks = stocks.copy()
        horizon = stocks.horizon_end.iloc[0]
        prev = ends[str(pd.Period(stocks.return_month.iloc[0])-2)]
        for name,day in [('start',signal),('end',horizon),('previous',prev)]:
            for col in ['close','adj_factor','adjusted_price','valuation_source']:
                stocks[f'{name}_{col}'] = stocks.stock_code.map(endpoints[day][col])
        stocks['return_1m'] = stocks.end_adjusted_price/stocks.start_adjusted_price-1
        stocks['return_prev1m'] = stocks.start_adjusted_price/stocks.previous_adjusted_price-1
        stocks['return_status'] = np.where(stocks.return_1m.notna(),'quoted_endpoints','unresolved_quote_or_terminal_value')
        marked = stocks.return_1m.notna() & (stocks.start_valuation_source.ne('daily_close')|stocks.end_valuation_source.ne('daily_close'))
        stocks.loc[marked,'return_status'] = 'research_suspension_valuation'
        window = calendar[calendar<=signal][-20:]
        amounts = pd.DataFrame(index=stocks.stock_code,columns=window,dtype=float)
        for day in window:
            q = store.date_query('daily',day.strftime('%Y%m%d'),'ts_code,trade_date,close,amount')
            amounts[day] = amounts.index.map(q.set_index('ts_code').amount)*1000  # 千元 -> 元
            if amounts[day].isna().any():
                entries = store.find('suspend_d',trade_date=day.strftime('%Y%m%d'),offset=0)
                if entries:
                    susp = store.read(entries[-1])
                    whole = susp.suspend_type.eq('S') & (susp.suspend_timing.isna()|susp.suspend_timing.eq(''))
                    mask = amounts[day].isna() & amounts.index.isin(susp.loc[whole,'ts_code'])
                    amounts.loc[mask,day] = 0.0  # known no trade, not a return imputation
        stocks['adv20_n_known'] = stocks.stock_code.map(amounts.notna().sum(axis=1))
        stocks['adv20'] = stocks.stock_code.map(amounts.sum(axis=1,min_count=20)/20)
        stocks['adv20_status'] = np.where(stocks.adv20_n_known.eq(20),'complete','missing_volume_sessions')
        frames.append(stocks)
        if number%12==0:
            print(f'price panel: {number+1}/{ranked.return_month.nunique()} months',flush=True)
    return pd.concat(frames,ignore_index=True)


def financials(store, stocks):
    # Preserve one consistent endpoint query per report period, including all pages.
    entries = store.find('income_vip',report_type='1')
    periods = sorted({e['request']['body']['params'].get('period') for e in entries if e['request']['body']['params'].get('period') and '20121231'<=str(e['request']['body']['params']['period'])<='20250930'})
    chunks = []
    for period in periods:
        candidates = [e for e in entries if e['request']['body']['params'].get('period')==period]
        # Prefer the recent full-panel acquisition, not universe-filtered old fetches.
        candidates = [e for e in candidates if 'full-panel-v1' in str(e.get('source_manifest',''))]
        by_offset = {}
        for e in candidates:
            by_offset[int(e['request']['body']['params'].get('offset',0))] = e
        offset = 0
        while offset in by_offset:
            e = by_offset[offset]; chunk = store.read(e)
            chunks.append(chunk[chunk.ts_code.isin(stocks)])
            limit = int(e['request']['body']['params']['limit'])
            if len(chunk)<limit:
                break
            offset += limit
        else:
            raise ValueError(f'Missing financial page {period} offset {offset}')
    if not chunks:
        raise ValueError('No historical income statements available')
    return pd.concat(chunks,ignore_index=True).drop_duplicates()


def prepare(root, online=False):
    root = Path(root); out=root/'data/prepared'; out.mkdir(parents=True,exist_ok=True)
    config = json.loads((root/'config.json').read_text())
    supplements = sorted((root/'data/probes').glob('*/data/raw/manifest.json')) + sorted((root/'data/repair_sources').glob('*/data/raw/manifest.json'))
    store = FrozenStore(root,sources=[*default_sources(),*supplements],online=online)
    try:
        securities,calendar,industry,names = catalog(store)
        securities.to_csv(out/'securities.csv',index=False)
        names.to_csv(out/'names.csv',index=False)
        industry.to_csv(out/'industry_intervals.csv',index=False)
        pd.DataFrame({'trade_date':calendar}).to_csv(out/'calendar.csv',index=False)
        panel,coverage,ends = universe(store,config,securities,calendar,industry)
        panel.to_csv(out/'universe.csv',index=False); coverage.to_csv(out/'universe_coverage.csv',index=False)
        panel[panel.financial_source.eq('prior_bak_basic_bounded_asof')].to_csv(out/'financial_repairs.csv',index=False)
        print('universe built:',len(panel),'company-months; cap gaps:',int(coverage.n_nonfinance_mv_missing.sum()),'finance unknown:',int(coverage.n_finance_unknown.sum()),flush=True)
        ranked = rank_universe(panel)
        if config.get('selection_policy')=='exclude_observable_risk_then_rank':
            from .selection import eligible_ranks
            from . import execution,historical_status
            market_manifest=out/'daily_market_manifest.json'
            old_market=json.loads(market_manifest.read_text()) if market_manifest.exists() else {}
            stale=(not (out/'daily_market.npz').exists()
                   or old_market.get('config_sha256')!=sha(root/'config.json')
                   or old_market.get('builder_sha256')!=execution.market_builder_sha256()
                   or old_market.get('status_builder_sha256')!=sha(historical_status.__file__))
            if stale:
                execution.build_market(root,panel,config)
            with np.load(out/'daily_market.npz',allow_pickle=False) as market:
                # Report coverage before removing unknown/risk names. The
                # filtered pool's zero unknown rate is a construction result.
                dates=pd.DatetimeIndex(market['dates']); codes={x:j for j,x in enumerate(market['codes'])}
                status = market['signal_state'] if 'signal_state' in market else market['state']
                known = market['signal_known_ST'] if 'signal_known_ST' in market else market['known_ST']
                # Missing industry/capitalization of a stock already excluded by
                # the observable-risk rule cannot affect this filtered ranking.
                # Preserve the raw gaps and explicitly identify why they do not
                # enter the decision; never use a future ST/delisting outcome.
                pt=dates.get_indexer(pd.DatetimeIndex(panel.signal_date));pj=panel.stock_code.map(codes).to_numpy(int)
                ordinary=(status[pt,pj]==0)&~known[pt,pj]
                risk_now=known[pt,pj]
                cm=panel.is_financial.eq(0)&(~np.isfinite(panel.market_cap)|panel.market_cap.le(0))
                fm=panel.is_financial.isna()
                gaps=panel[['return_month']].copy()
                gaps['n_finance_unknown_raw']=fm.astype(int)
                gaps['n_nonfinance_mv_missing_raw']=cm.astype(int)
                gaps['n_finance_unknown']= (fm&ordinary).astype(int)
                gaps['n_nonfinance_mv_missing']= (cm&ordinary).astype(int)
                gaps['n_finance_unknown_excluded_known_risk']= (fm&risk_now).astype(int)
                gaps['n_cap_missing_excluded_known_risk']= (cm&risk_now).astype(int)
                gaps=gaps.groupby('return_month',as_index=False).sum()
                coverage=coverage.drop(columns=['n_finance_unknown','n_nonfinance_mv_missing']).merge(gaps,on='return_month',validate='one_to_one')
                coverage.to_csv(out/'universe_coverage.csv',index=False)
                t=dates.get_indexer(pd.DatetimeIndex(ranked.signal_date)); j=ranked.stock_code.map(codes).to_numpy(int)
                rows=ranked[['return_month','rank']].copy()
                rows['unknown']=status[t,j]!=0
                rows['known_risk']=known[t,j]
                rows['unknown'] &= ~rows.known_risk
                monthly=[]
                for month,g in rows.groupby('return_month'):
                    tail=g[g['rank']<=500]
                    monthly.append(dict(return_month=month,n_rankable=len(g),n_unknown=int(g.unknown.sum()),n_known_risk=int(g.known_risk.sum()),
                        n_raw_top500=len(tail),n_raw_top500_unknown=int(tail.unknown.sum()),n_raw_top500_known_risk=int(tail.known_risk.sum())))
                pd.DataFrame(monthly).to_csv(root/'data/repair_sources/point_in_time/status_universe_monthly_coverage.csv',index=False)
                pd.DataFrame(monthly).to_csv(out/'status_coverage.csv',index=False)
                ranked=eligible_ranks(ranked,market)
        ranked[[col for col in ['return_month','signal_date','stock_code','rank','original_rank','bucket','market_cap'] if col in ranked]].to_csv(out/'selection_universe.csv',index=False)
        ranked = ranked[ranked['rank']<=500].copy()
        ranked.to_csv(out/'ranks.csv',index=False)
        ranked = price_features(store,ranked,calendar,ends,config)
        ranked = st_labels(ranked,names,securities)
        if config.get('selection_policy')=='exclude_observable_risk_then_rank':
            # Explicitly observable non-risk at the monthly signal implies
            # non-ST. Namechange-only follow-up gaps still stay unknown.
            ranked['st_current']=0.0
            ranked['st_status']='known_nonrisk_asof_interval_or_observed_name'
            # A known future ST interval is retained even when the old
            # namechange-only function could not establish the initial state.
            future_names=names.copy()
            for column in ['start_date','end_date','ann_date']:
                future_names[column]=dt(future_names[column])
            future_names=future_names[future_names.name.astype(str).str.upper().str.contains('ST|PT',regex=True)]
            for row in future_names.itertuples():
                if pd.isna(row.start_date) or pd.isna(row.ann_date): continue
                available=max(row.start_date,row.ann_date+pd.Timedelta(days=1))
                use=ranked.stock_code.eq(row.ts_code)&ranked.signal_date.lt(available)&ranked.horizon_end.ge(available)
                ranked.loc[use,'new_st_next']=1.0
        reports = financials(store,set(ranked.stock_code))
        reports.to_csv(out/'income_source.csv',index=False)
        profits,audit = profit_ttm(reports,sorted(ranked.signal_date.unique()),calendar,config['max_report_age_days'])
        audit.to_csv(out/'financial_version_audit.csv',index=False)
        ranked = ranked.merge(profits,on=['stock_code','signal_date'],how='left',validate='one_to_one')
        ranked.to_csv(out/'panel.csv',index=False)
        ranked.loc[ranked.return_1m.isna()].to_csv(out/'unresolved_returns.csv',index=False)
        # Delisting ledgers and terminal scenarios are downstream of this panel;
        # their own evidence manifest validates them without a circular input lock.
        repair_files = [p for p in (root/'data/repair_sources/suspension').glob('*')
                        if p.is_file() and p.suffix in ['.csv','.json']]
        point_in_time_files=[p for p in (root/'data/repair_sources/point_in_time').rglob('*') if p.is_file()]
        dump(out/'manifest.json',dict(config_sha256=sha(root/'config.json'),artifacts=[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in [*sorted(out.glob('*.csv')),*sorted(repair_files),*sorted(point_in_time_files),root/'data/repair_sources/early/market_cap.csv'] if p.exists()],limitations=[
            'Historical vendor snapshots; not a certified original financial filing vintage archive',
            'Industry: same-date or <=1 prior-session historical backup snapshots; early published CSRC quarterly tables lag real business changes and use a different taxonomy',
            'Observed names are carried forward for at most 185 calendar days to fill interval gaps; unknown future ST follow-up remains unknown',
            'Missing market caps can affect ranks; coverage gate must pass before selecting winner',
            'Unresolved terminal quotes or delisting proceeds are not filled with zero']))
        return ranked,coverage
    finally:
        store.save_manifest()
