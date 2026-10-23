"""Descriptive applicability analysis only; no strategy, timing or parameter changes."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/applicability';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=pd.read_csv(ROOT/'results/monthly.csv');m=m[m.cost_bps.eq(30)].pivot(index='return_month',columns='strategy',values='net_return').sort_index()
 r=m.R001_100;rows=[]
 for name,a,z in [('TRAIN','2015-01','2023-12'),('TEST','2024-01','2025-12'),('ALL','2015-01','2025-12')]:
  q=r.loc[a:z];logs=np.log1p(q);top=logs.nlargest(5)
  rows.append(dict(period=name,n=len(q),cagr=np.expm1(logs.mean()*12),total_log_return=logs.sum(),best5_log_share=top.sum()/logs.sum(),best5_months=','.join(top.index),cagr_best5_replaced_cash=np.expm1((logs.sum()-top.sum())/len(q)*12),positive_months=int(q.gt(0).sum()),tail10=int(q.le(-.1).sum())))
 pd.DataFrame(rows).to_csv(OUT/'concentration.csv',index=False)
 annual=m.groupby(m.index.str[:4]).agg(lambda s:np.prod(1+s)-1);annual.to_csv(OUT/'annual_returns.csv',index_label='year')
 ex=r.loc['2016-01':'2023-12'];summary={'train_without_2015_cagr':float(np.expm1(np.log1p(ex).mean()*12)),'all_without_2015_cagr':float(np.expm1(np.log1p(r.loc['2016-01':]).mean()*12)),'year2015_log_share_train':float(np.log1p(r.loc['2015-01':'2015-12']).sum()/np.log1p(r.loc[:'2023-12']).sum())}
 # Freeze ONLY the already available local CSI300 cache, no network.
 cached=OUT/'csi300_monthly.csv'
 if not cached.exists():
  raise FileNotFoundError('Missing frozen results/applicability/csi300_monthly.csv; no automatic network or private-cache fallback.')
 c=pd.read_csv(cached).set_index('return_month').csi300_return
 panel=m[['R001_100','CSI1000']].join(c);panel['style_spread']=panel.CSI1000-panel.csi300_return
 env=[]
 for dim,series in [('market',panel.CSI1000),('style',panel.style_spread)]:
  for lag in [0,1]:
   x=series.shift(lag)
   for period,a,z in [('TRAIN','2015-01','2023-12'),('TEST','2024-01','2025-12'),('ALL','2015-01','2025-12')]:
    for positive in [True,False]:
     valid=x.notna()&panel.index.to_series().between(a,z);mask=valid&((x>0) if positive else (x<=0));q=panel.loc[mask]
     if not len(q):continue
     rr=q.R001_100
     env.append(dict(dimension=dim,lag=lag,period=period,state='positive' if positive else 'nonpositive',n=len(q),first=q.index.min(),last=q.index.max(),mean=rr.mean(),median=rr.median(),win_rate=rr.gt(0).mean(),tail10_rate=rr.le(-.1).mean(),worst=rr.min(),benchmark_mean=q.CSI1000.mean(),mean_spread=(rr-q.CSI1000).mean()))
 pd.DataFrame(env).to_csv(OUT/'environment.csv',index=False);panel.to_csv(OUT/'environment_panel.csv',index_label='return_month')
 # Do not annualize or splice non-contiguous state months into a tradable curve.
 p=pd.read_csv(ROOT/'results/performance.csv');ext=pd.read_csv(ROOT/'results/supplement/extended_performance.csv')
 grad=pd.concat([p[p.cost_bps.eq(30)&p.strategy.ne('CSI1000')],ext[ext.cost_bps.eq(30)]])
 grad.sort_values(['period','strategy']).to_csv(OUT/'size_groups.csv',index=False)
 costs=pd.read_csv(ROOT/'results/cost_sensitivity.csv');costs[costs.strategy.eq('R001_100')].to_csv(OUT/'execution_costs.csv',index=False)
 capital=pd.read_csv(ROOT/'results/supplement/capital_performance.csv');capital.to_csv(OUT/'capital_constraints.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 # Independent annual compounding consistency and exhaustive regime partition.
 for year,v in annual.R001_100.items():assert np.isclose(v,np.expm1(np.log1p(r[r.index.str.startswith(year)]).sum()))
 e=pd.DataFrame(env)
 for (dim,lag,period),g in e.groupby(['dimension','lag','period']):
  a,z={'TRAIN':('2015-01','2023-12'),'TEST':('2024-01','2025-12'),'ALL':('2015-01','2025-12')}[period]
  series=(panel.CSI1000 if dim=='market' else panel.style_spread).shift(lag);idx=series.loc[a:z].dropna().index
  assert g.n.sum()==len(idx)
  assert np.isclose((g['mean']*g.n).sum()/g.n.sum(),r.loc[idx].mean())
 files=['results/monthly.csv','results/performance.csv','results/supplement/extended_performance.csv','results/cost_sensitivity.csv','results/supplement/capital_performance.csv','config.json','results/selection.json','results/applicability/csi300_monthly.csv','results/applicability/csi300_source.json']
 manifest={'network_requests':0,'strategy_modified':False,'parameter_search':False,'style_source_period':'CSI300 cached monthly returns from 2020-01; lagged states from 2020-02','concentration_definition':'best 5 observed log returns replaced with zero, fixed original N; ex post diagnostic, not deployable strategy','checks':'annual compounding and exhaustive environment weighted means passed','inputs':{f:sha(ROOT/f) for f in files},'script_sha256':sha(__file__)}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print(pd.DataFrame(rows).to_string(index=False));print(summary);print(e[e.period.eq('TEST')].to_string(index=False))
if __name__=='__main__':main()
