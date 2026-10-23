"""Reproducible case sheets and explicit terminal-value sensitivity.

A generated sheet is not a human sign-off. Record the actual reviewer and notes
in audit_reviews.csv; unresolved cases remain unresolved after review.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .acquire import FrozenStore, dump, sha
from .workflow import load,write_json,clean,ranking_gate
from .valuation_repair import load_valuations


def source_evidence(store,cases,panel,names,securities):
    """Independently expose raw endpoint records for a reviewer to inspect."""
    evidence=[]
    valuations=load_valuations(store.root)
    for _,case in cases.iterrows():
        row=panel[(panel.stock_code==case.stock_code)&(panel.return_month==case.return_month)].iloc[0]
        record={'case_id':case.case_id,'stock_code':case.stock_code,'month':case.return_month,'endpoints':[]}
        for day in [row.signal_date,row.horizon_end]:
            stamp=day.strftime('%Y%m%d')
            quotes=store.date_query('daily',stamp,'ts_code,trade_date,close,amount');q=quotes[quotes.ts_code.eq(case.stock_code)]
            factors=store.date_query('adj_factor',stamp,'ts_code,trade_date,adj_factor');f=factors[factors.ts_code.eq(case.stock_code)]
            point=dict(day=str(day.date()),daily_close=None if q.empty else q.iloc[0]['close'],factor=None if f.empty else f.iloc[0].adj_factor,quote_source=quotes.attrs['source_sha256'],factor_source=factors.attrs['source_sha256'])
            if q.empty:
                backup=store.date_query('bak_daily',stamp,'ts_code,trade_date,close,total_mv,total_share,industry',7000);b=backup[backup.ts_code.eq(case.stock_code)]
                point.update(backup_close=None if b.empty else b.iloc[0]['close'],backup_source=backup.attrs['source_sha256'])
                entries=store.find('suspend_d',trade_date=stamp,offset=0)
                if entries:
                    susp=store.read(entries[-1]);point.update(suspension_records=susp[susp.ts_code.eq(case.stock_code)].to_dict('records'),suspension_source=susp.attrs['source_sha256'])
                if len(valuations):
                    repairs=valuations[valuations.ts_code.eq(case.stock_code)&valuations.trade_date.eq(day.strftime('%Y-%m-%d'))]
                    point['audited_research_valuation']=repairs.to_dict('records')
            record['endpoints'].append(point)
        history=names[names.ts_code.eq(case.stock_code)].copy()
        for key in ['start_date','end_date','ann_date']:
            history[key]=pd.to_datetime(history[key].astype('string').str.replace(r'\.0$','',regex=True),format='mixed',errors='coerce')
        visible=history.start_date.le(row.signal_date)&history.ann_date.lt(row.signal_date)&(history.end_date.isna()|history.end_date.ge(row.signal_date))
        record['active_name_records']=history.loc[visible,['name','start_date','end_date','ann_date']].astype(str).to_dict('records')
        record['security_master']=securities[securities.ts_code.eq(case.stock_code)][['ts_code','list_date','delist_date','list_status']].to_dict('records')
        record['ttm_trace']=json.loads(case.ttm_trace) if isinstance(case.ttm_trace,str) else []
        evidence.append(record)
    return evidence


def audit(root):
    root=Path(root);config,panel,coverage=load(root);out=root/'results'
    categories={
        'ST':panel.st_current.eq(1),
        'loss':panel.is_loss.eq(1)&panel.net_profit_ttm.notna(),
        'corporate_action':panel.start_adj_factor.ne(panel.end_adj_factor)&panel.return_1m.notna(),
        'suspension':panel.return_status.eq('research_suspension_valuation'),
        'delisting':panel.delisted_next.eq(1),
        'ordinary':panel.return_1m.notna()&panel.st_current.eq(0)&panel.is_loss.eq(0)}
    selected=[];used=set()
    # First and last observations provide both early and late-period checks.
    for category,mask in categories.items():
        frame=panel[mask].sort_values(['return_month','stock_code'])
        candidates=pd.concat([frame.head(15),frame.tail(15)])
        count=0
        for _,row in candidates.iterrows():
            key=(row.stock_code,row.return_month)
            if key in used:continue
            selected.append((category,row));used.add(key);count+=1
            if count==4:break
    rows=[]
    for number,(category,r) in enumerate(selected,1):
        trace=json.loads(r.input_trace) if isinstance(r.input_trace,str) else []
        complete=bool(trace) and all(t['value'] is not None for t in trace)
        recalculated_ttm=sum(t['coefficient']*t['value'] for t in trace) if complete else np.nan
        recomputed=r.end_close*r.end_adj_factor/(r.start_close*r.start_adj_factor)-1
        trace_available=all(pd.Timestamp(t['available_date'])<=r.signal_date for t in trace if t['available_date'])
        rows.append(dict(case_id=f'C{number:02}',category=category,stock_code=r.stock_code,
                         signal_date=r.signal_date,return_month=r.return_month,bucket=r.bucket,rank=r['rank'],
                         market_cap=r.market_cap,market_cap_source=r.market_cap_source,
                         start_close=r.start_close,end_close=r.end_close,
                         start_factor=r.start_adj_factor,end_factor=r.end_adj_factor,
                         start_valuation_source=r.start_valuation_source,end_valuation_source=r.end_valuation_source,
                         reported_return=r.return_1m,recomputed_return=recomputed,
                         return_check='unresolved' if pd.isna(r.return_1m) else ('pass' if np.isclose(recomputed,r.return_1m,atol=1e-12) else 'fail'),
                         net_profit_ttm=r.net_profit_ttm,recomputed_ttm=recalculated_ttm,
                         ttm_check='unresolved' if pd.isna(r.net_profit_ttm) else ('pass' if np.isclose(recalculated_ttm,r.net_profit_ttm,rtol=1e-10) else 'fail'),
                         trace_available_before_signal=trace_available,st_current=r.st_current,new_st_next=r.new_st_next,
                         delisted_next=r.delisted_next,ttm_trace=r.input_trace,
                         panel_sha256=sha(root/'data/prepared/panel.csv')))
    cases=pd.DataFrame(rows);cases.to_csv(out/'audit_cases.csv',index=False)
    # Missing market-cap rows are visible, never silently discarded.
    universe=pd.read_csv(root/'data/prepared/universe.csv')
    gaps=universe[(universe.is_financial.eq(0)&universe.market_cap.isna())|universe.is_financial.isna()]
    gaps.to_csv(out/'unresolved_universe.csv',index=False)
    bounds=panel[panel['rank'].isin([1,99,100,101,199,200,201,299,300,301,399,400,401,499,500])]
    bounds[['return_month','stock_code','market_cap','rank','bucket']].to_csv(out/'rank_boundaries.csv',index=False)
    # Terminal recovery scenarios are hypothetical, never observations or fills.
    exits=panel[panel.delisted_next.eq(1)].copy();store=FrozenStore(root)
    last_quotes=[]
    for stock,g in exits.groupby('stock_code'):
        begin=g.signal_date.min();end=g.horizon_end.max()
        candidates=[]
        for e in store.find('daily'):
            day=e['request']['body'].get('params',{}).get('trade_date','')
            if not day or not (begin.strftime('%Y%m%d')<=day<=end.strftime('%Y%m%d')):continue
            frame=store.read(e);hit=frame[frame.ts_code.eq(stock)]
            if len(hit):candidates.append(hit)
        if candidates:
            q=pd.concat(candidates).drop_duplicates(['ts_code','trade_date']).sort_values('trade_date')
            for _,r in g.iterrows():
                known=q[(pd.to_datetime(q.trade_date,format='%Y%m%d')>r.signal_date)&(pd.to_datetime(q.trade_date,format='%Y%m%d')<=r.horizon_end)]
                last=known.iloc[-1] if len(known) else None
                factor=np.nan
                if last is not None:
                    entries=store.find('adj_factor',trade_date=str(last.trade_date),offset=0)
                    if entries:
                        a=store.read(entries[-1]);hit=a[a.ts_code.eq(stock)]
                        if len(hit):factor=float(hit.iloc[0].adj_factor)
                last_quotes.append(dict(stock_code=stock,return_month=r.return_month,
                     last_trade_date=None if last is None else str(last.trade_date),
                     last_trade_close=np.nan if last is None else float(last.close),last_trade_factor=factor,
                     start_adjusted_price=r.start_adjusted_price))
        else:
            for _,r in g.iterrows():last_quotes.append(dict(stock_code=stock,return_month=r.return_month,last_trade_date=None,last_trade_close=np.nan,last_trade_factor=np.nan,start_adjusted_price=r.start_adjusted_price))
    terminal=pd.DataFrame(last_quotes)
    if len(terminal):
        terminal['observed_last_trade_return']=terminal.last_trade_close*terminal.last_trade_factor/terminal.start_adjusted_price-1
        terminal['last_trade_is_realizable_proceeds']=False
        for fraction in [0,.5,1]:
            terminal[f'hypothetical_terminal_return_{fraction:g}x_last_quote']=fraction*terminal.last_trade_close*terminal.last_trade_factor/terminal.start_adjusted_price-1
        terminal.loc[terminal.start_adjusted_price.gt(0),'hypothetical_terminal_return_0x_last_quote']=-1.
        terminal['terminal_proceeds_certified']=False
    terminal.to_csv(out/'delisting_sensitivity.csv',index=False)
    evidence=source_evidence(store,cases,panel,pd.read_csv(root/'data/prepared/names.csv'),pd.read_csv(root/'data/prepared/securities.csv'))
    write_json(out/'audit_source_evidence.json',evidence)
    store.save_manifest()
    summary=dict(cases_generated=len(cases),categories=cases.category.value_counts().to_dict(),
                 arithmetic_failures=int((cases.return_check.eq('fail')|cases.ttm_check.eq('fail')).sum()),
                 unresolved_case_returns=int(cases.return_check.eq('unresolved').sum()),
                 manual_signoff_file='audit_reviews.csv',
                 instruction='Generated checks do not constitute filing-version certification; complete a case-by-case review and preserve unresolved findings.')
    write_json(out/'audit_summary.json',summary)
    return cases
