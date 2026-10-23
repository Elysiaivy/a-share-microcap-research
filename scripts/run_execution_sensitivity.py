"""Reproducible supplemental experiments; never modifies the frozen selection.

Run from project: python3 scripts/run_execution_sensitivity.py
"""
from pathlib import Path
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from microcap.research import inputs
from microcap.execution import run
from microcap.execution_constraints import OrderConstraints
from microcap.analysis import metrics
from microcap.diagnostics import daily_risk
ROOT=Path(__file__).resolve().parents[1]
from microcap.preflight import require_full_inputs
require_full_inputs(ROOT, 'evaluate')
R=ROOT/'results';OUT=R/'supplement';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarize(monthly,daily,initial):
 r=monthly.net_return.to_numpy();v=np.r_[initial,daily.nav.to_numpy()]
 return dict(n_months=len(r),cagr=np.prod(1+r)**(12/len(r))-1,
             sharpe=np.sqrt(12)*r.mean()/r.std(ddof=1),
             daily_max_drawdown=np.min(v/np.maximum.accumulate(v)-1),
             mean_cash=daily.cash_weight.mean(),mean_holdings=daily.hold_count.mean(),
             min_holdings=daily.hold_count.min(),annual_turnover=monthly.turnover.mean()*12,
             annual_cost=monthly.cost.mean()*12,ending_capital=float(v[-1]),
             worst_month=r.min())

def main():
 c,p,coverage,m=inputs(ROOT);selection_hash=sha(R/'selection.json');config_hash=sha(ROOT/'config.json')
 winner=json.loads((R/'selection.json').read_text())['winner']
 caps=[100000,200000,300000,500000,1000000,2000000,5000000]
 protocol=dict(frozen_winner=winner,capitals=caps,cost_bps=30,commission_bps=2.5,minimum_commission=5,
               buy_lot=100,capital_reset='each period starts in cash; no transfers or withdrawals afterwards',
               accounting='adjusted units with automatic distribution reinvestment; full exit may include synthetic fractional claims',
               extension='501-1000 exploratory only; excluded from original winner selection',
               periods={'TRAIN':['2015-01','2023-12'],'TEST':['2024-01','2025-12']})
 (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
 rows=[];all_months=[];all_daily=[];checks=[]
 for period,(start,stop) in ([] if '--extension-only' in sys.argv else protocol['periods'].items()):
  cc={**c,'train_start':start,'test_end':stop,'buckets':{winner:c['buckets'][winner]}}
  for mode,capital in [('CONTINUOUS',1)]+[(mode,cap) for cap in caps for mode in ['LOT','LOT_MIN5']]:
   constr=None if mode=='CONTINUOUS' else OrderConstraints(capital,100,5 if mode=='LOT_MIN5' else 0,2.5)
   monthly,d=run(m,p,cc,30,c['terminal_fraction'],details=True,implementation=constr)
   tag=dict(period=period,mode=mode,initial_capital=capital)
   stats=summarize(monthly,d['daily_nav'],capital)
   tr=d['trades'];ev=d['events'];stale=int(ev.event.eq('unverified_stale_mark').sum()) if len(ev) else 0
   assert stale==0,(tag,ev[ev.event.eq('unverified_stale_mark')].head())
   if mode=='CONTINUOUS' and period=='TRAIN':
    old=pd.read_csv(R/'selection_training_monthly.csv');old=old[old.strategy.eq(winner)].sort_values('return_month')
    np.testing.assert_allclose(monthly.sort_values('return_month').net_return,old.net_return,rtol=0,atol=1e-12)
   buy=tr[tr.side.eq('buy')].copy()
   ti=pd.Index(pd.DatetimeIndex(m['dates']).strftime('%Y-%m-%d')).get_indexer(buy.date)
   ji=pd.Index(m['codes']).get_indexer(buy.stock_code)
   assert (ti>=0).all() and (ji>=0).all()
   rawshares=buy.filled_units.to_numpy()*m['factor'][ti,ji]
   lot_error=float(np.max(abs(rawshares/100-np.round(rawshares/100)))) if len(buy) else 0.
   if mode!='CONTINUOUS':assert lot_error<1e-7,lot_error
   extra=np.maximum(tr.cost-tr.notional*.003,0).sum()
   stats.update(n_trades=len(tr),minimum_fee_extra_yuan=float(extra),extra_fee_over_initial=float(extra/capital))
   rows.append({**tag,**stats})
   all_months.append(monthly.assign(**tag));all_daily.append(d['daily_nav'].assign(**tag))
   checks.append({**tag,'unverified_stale_marks':stale,'buy_lot_max_error':lot_error if mode!='CONTINUOUS' else None,
                  'min_cash_yuan':float(d['daily_nav'].cash.min()),'terminal_events':int(ev.event.eq('terminal').sum()) if len(ev) else 0})
   if mode!='CONTINUOUS':
    tr.to_csv(OUT/f'trades_{period}_{mode}_{capital}.csv.gz',index=False,compression='gzip')
   print(period,mode,capital,{k:round(stats[k],4) for k in ['cagr','sharpe','mean_cash','mean_holdings']},flush=True)
   del d
 if rows:
  pd.DataFrame(rows).to_csv(OUT/'capital_performance.csv',index=False)
  pd.concat(all_months).to_csv(OUT/'capital_monthly.csv',index=False)
  pd.concat(all_daily).to_csv(OUT/'capital_daily.csv.gz',index=False,compression='gzip')
  pd.DataFrame(checks).to_csv(OUT/'capital_checks.csv',index=False)
 # Same historical eligible universe; keep the reserve beyond rank 1000 too.
 u=pd.read_csv(ROOT/'data/prepared/selection_universe.csv');u.attrs.update(p.attrs)
 ec={**c,'buckets':{f'R{lo:03d}_{lo+99}':[lo,lo+99] for lo in range(501,1000,100)}}
 assert set(u.stock_code)<=set(m['codes'])
 count=u[u['rank'].between(501,1000)].groupby('return_month').size()
 assert len(count)==132 and count.eq(500).all()
 patch_path=ROOT/'data/repair_sources/extension/suspension_patch.json'
 patch=json.loads(patch_path.read_text())
 for source in patch['sources']:
  assert sha(ROOT/source['path'])==source['sha256']
 dates=pd.DatetimeIndex(m['dates']);j=list(m['codes']).index(patch['stock_code'])
 mask=(dates>=patch['start'])&(dates<=patch['end'])
 assert mask.sum()==209 and np.isnan(m['open'][mask,j]).all() and np.isnan(m['close'][mask,j]).all()
 # Isolated, evidenced suspension annotation; never invent a quote or change rankings.
 em=dict(m)
 for key in ['suspended','suspended_any']:
  em[key]=m[key].copy();em[key][mask,j]=True
 ext,d=run(em,u,ec,30,c['terminal_fraction'],details=True)
 ev=d['events'];ev.to_csv(OUT/'extended_events_diagnostic.csv',index=False);assert not len(ev[ev.event.eq('unverified_stale_mark')]) if len(ev) else True
 ep=metrics(ext,c).merge(daily_risk(d['daily_nav'].assign(cost_bps=30),c),on=['strategy','cost_bps','period'],validate='one_to_one')
 ext.to_csv(OUT/'extended_monthly.csv',index=False);ep.to_csv(OUT/'extended_performance.csv',index=False)
 for name in ['events','holdings','daily_nav']:
  d[name].to_csv(OUT/f'extended_{name}.csv.gz',index=False,compression='gzip')
 assert sha(R/'selection.json')==selection_hash and sha(ROOT/'config.json')==config_hash
 sources=['config.json','results/selection.json','data/prepared/daily_market.npz','data/prepared/selection_universe.csv','data/prepared/panel.csv','microcap/execution.py','microcap/execution_constraints.py','scripts/run_execution_sensitivity.py','data/repair_sources/extension/suspension_patch.json']
 manifest=dict(protocol=protocol,sources={f:sha(ROOT/f) for f in sources},
               outputs={str(f.relative_to(ROOT)):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='manifest.json'},
               checks=dict(frozen_selection_unchanged=True,config_unchanged=True,baseline_training_reproduced=True,
                           missing_holding_marks=0,extended_months=132,extended_target_stocks_per_month=500))
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print(ep[ep.period.isin(['TRAIN','TEST'])][['strategy','period','cagr','sharpe','daily_max_drawdown']].to_string(index=False),flush=True)
if __name__=='__main__':main()
