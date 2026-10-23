"""Inspect independent raw records for representative company-month cases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .acquire import FrozenStore,dump,sha
from .prepare import catalog,dt
from .research import inputs
from .workflow import clean


def audit(root):
 root=Path(root);out=root/'results';c,p,coverage,m=inputs(root);s=FrozenStore(root)
 _,_,_,names=catalog(s)
 for col in ['start_date','end_date','ann_date']:names[col]=dt(names[col])
 dated_names=pd.read_csv(root/'data/repair_sources/point_in_time/status_evidence.csv',parse_dates=['observation_date','availability_date'])
 dates=pd.DatetimeIndex(m['dates']);idx={code:i for i,code in enumerate(m['codes'])}
 trades=pd.read_csv(out/'trades.csv');blocked=pd.read_csv(out/'blocked_orders.csv')
 cases=[];seen=set()
 categories={'new_ST':p.new_st_next.eq(1),'loss':p.is_loss.eq(1),'corporate_action':p.start_adj_factor.ne(p.end_adj_factor)&p.start_adj_factor.notna()&p.end_adj_factor.notna(),'suspension':p.return_status.eq('research_suspension_valuation'),'delisting':p.delisted_next.eq(1),'ordinary':p.st_current.eq(0)&p.is_loss.eq(0)&p.return_1m.notna()}
 evidence=[]
 for cat,mask in categories.items():
  choices=p[mask].sort_values(['return_month','stock_code']); count=0
  for r in choices.itertuples():
   key=(r.stock_code,r.return_month)
   if key in seen:continue
   seen.add(key);count+=1
   trace=json.loads(r.input_trace) if isinstance(r.input_trace,str) else []
   complete=bool(trace) and all(t['value'] is not None for t in trace)
   recomputed=sum(t['value']*t['coefficient'] for t in trace) if complete else np.nan
   available=all(pd.Timestamp(t['available_date'])<=r.signal_date for t in trace if t['available_date'])
   sources=[]
   for date in [r.signal_date,r.horizon_end]:
    day=date.strftime('%Y%m%d');q=s.date_query('daily',day,'ts_code,trade_date,close');a=s.date_query('adj_factor',day,'ts_code,trade_date,adj_factor');su=s.date_query('suspend_d',day,'ts_code,trade_date,suspend_timing,suspend_type')
    sources.append(dict(date=str(date.date()),daily=q[q.ts_code.eq(r.stock_code)].to_dict('records'),factor=a[a.ts_code.eq(r.stock_code)].to_dict('records'),suspension=su[su.ts_code.eq(r.stock_code)].to_dict('records'),hashes=[q.attrs['source_sha256'],a.attrs['source_sha256'],su.attrs['source_sha256']]))
   h=names[names.ts_code.eq(r.stock_code)]
   h=h[h.start_date.le(r.signal_date)&h.ann_date.lt(r.signal_date)&(h.end_date.isna()|h.end_date.ge(r.signal_date))]
   cases.append(dict(category=cat,stock_code=r.stock_code,return_month=r.return_month,signal_date=str(r.signal_date.date()),rank=r.rank,market_cap=r.market_cap,st_current=r.st_current,net_profit_ttm=r.net_profit_ttm,recomputed_ttm=recomputed,ttm_visible=available,ttm_check=bool(np.isclose(recomputed,r.net_profit_ttm)) if complete and pd.notna(r.net_profit_ttm) else None,return_status=r.return_status,delisted_next=r.delisted_next))
   observed=dated_names[dated_names.ts_code.eq(r.stock_code)&dated_names.availability_date.le(r.signal_date)&dated_names.observation_date.le(r.signal_date)]
   if len(observed):observed=observed[observed.observation_date.eq(observed.observation_date.max())]
   evidence.append(dict(category=cat,stock_code=r.stock_code,return_month=r.return_month,endpoints=sources,visible_names=h.astype(str).to_dict('records'),dated_name_evidence=observed.astype(str).to_dict('records'),ttm_trace=trace))
   if count==4:break
 case=pd.DataFrame(cases);case.to_csv(out/'audit_cases.csv',index=False);dump(out/'audit_source_evidence.json',clean(evidence))
 execution=[]
 # Execution cases are separate from original rank-pool company-month audits.
 for r in trades[trades.reason.eq('ST')].sort_values('delay_sessions',ascending=False).drop_duplicates(['stock_code','date']).head(12).itertuples():
  t=dates.get_loc(pd.Timestamp(r.date));j=idx[r.stock_code]
  execution.append(dict(stock_code=r.stock_code,date=r.date,category='ST_sell',open=m['open'][t,j],down_limit=m['down'][t,j],suspended=bool(m['suspended'][t,j]),delay_sessions=r.delay_sessions,cost=r.cost,expected_cost=r.notional*.003,verified=bool(m['open'][t,j]>m['down'][t,j] and not m['suspended'][t,j] and np.isclose(r.cost,r.notional*.003))))
 for r in blocked[blocked.ST_order.eq(True)].drop_duplicates(['stock_code','date']).head(8).itertuples():
  t=dates.get_loc(pd.Timestamp(r.date));j=idx[r.stock_code]
  op=m['open'][t,j];down=m['down'][t,j];sus=m['suspended'][t,j]
  execution.append(dict(stock_code=r.stock_code,date=r.date,category='ST_blocked',open=op,down_limit=down,suspended=bool(sus),verified=bool(sus or not np.isfinite(op) or not np.isfinite(down) or op<=down+1e-7)))
 for r in trades[trades.reason.eq('ST_replacement')].head(8).itertuples():
  t=dates.get_loc(pd.Timestamp(r.date));j=idx[r.stock_code]
  sales=trades[trades.strategy.eq(r.strategy)&trades.stock_code.eq(r.replaces)&trades.reason.eq('ST')&trades.date.le(r.date)]
  sale=sales.iloc[-1]
  verified=bool(m['state'][t,j]==0 and not m['known_ST'][t,j] and not m['suspended'][t,j] and m['open'][t,j]<m['up'][t,j] and np.isclose(r.cost,r.notional*.003) and r.notional+r.cost<=r.funding_budget+1e-10 and pd.Timestamp(r.date)>pd.Timestamp(sale.date) and pd.Timestamp(r.plan_date)<pd.Timestamp(r.date))
  execution.append(dict(stock_code=r.stock_code,date=r.date,category='ST_replacement',open=m['open'][t,j],up_limit=m['up'][t,j],replaces=r.replaces,verified=verified))
 for code,day in [('002208.SZ','2015-03-02'),('002169.SZ','2015-03-02'),('002110.SZ','2015-04-01')]:
  t=dates.get_loc(pd.Timestamp(day));j=idx[code]
  execution.append(dict(stock_code=code,date=day,category='afternoon_halt_open',open=m['open'][t,j],down_limit=m['down'][t,j],verified=bool(m['suspended_any'][t,j] and not m['suspended'][t,j] and m['open'][t,j]>m['down'][t,j])))
 pd.DataFrame(execution).to_csv(out/'execution_audit_cases.csv',index=False)
 dump(out/'audit_summary.json',dict(company_month_cases=len(case),categories=case.category.value_counts().to_dict(),execution_cases=len(execution),execution_checks_pass=all(r['verified'] for r in execution),human_signoff=False,review_status='source_records_exposed_for_case_review'))
 s.save_manifest();return case
