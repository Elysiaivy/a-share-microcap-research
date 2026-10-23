"""Verify frozen inputs, final selection, execution and deliverable checksums."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from microcap.acquire import sha,canonical,dump
from microcap.research import inputs,fingerprint
from microcap.portfolio import select_winner
c,p,coverage,m=inputs(ROOT);out=ROOT/'results';errors=[]
sel=json.loads((out/'selection.json').read_text())
if sel['training_input_sha256']!=fingerprint(m,p,c):errors.append('training_hash')
if set(c['candidates'])!=set(c['buckets']) or len(c['candidates'])!=5:errors.append('all_five_candidates_required')
if set(sel['scores'])!=set(c['candidates']):errors.append('frozen_candidate_scores')
if c['main_cost_bps']!=30 or sel['config_sha256']!=sha(ROOT/'config.json'):errors.append('selection_protocol')
evaluation=json.loads((out/'evaluation_manifest.json').read_text())
if evaluation['config_sha256']!=sha(ROOT/'config.json') or evaluation['selection_sha256']!=sha(out/'selection.json'):errors.append('evaluation_selection_lock')
raw=json.loads((ROOT/'data/raw/manifest.json').read_text())
for e in raw['inputs']:
 path=ROOT/e['path']
 if sha(path)!=e['sha256']:errors.append('raw_hash:'+e['path'])
 value=json.loads(path.read_text())
 if hashlib.sha256(canonical(value['response'])).hexdigest()!=e['response_sha256']:errors.append('response_hash:'+e['path'])
 if 'token' in value['request']['body']:errors.append('credential_in_request')
for name in ['evaluation_manifest.json','report_manifest.json']:
 manifest=json.loads((out/name).read_text())
 for e in [*manifest['artifacts'],*manifest.get('source_artifacts',[])]:
  if sha(ROOT/e['path'])!=e['sha256']:errors.append('artifact_hash:'+e['path'])
dates=pd.DatetimeIndex(m['dates']);si=p.stock_code.map({x:i for i,x in enumerate(m['codes'])}).to_numpy();di=dates.get_indexer(pd.to_datetime(p.signal_date))
if not (m['signal_state'][di,si]==0).all() or m['signal_known_ST'][di,si].any():errors.append('risk_stock_selected')
if not p.original_rank.ge(p['rank']).all():errors.append('invalid_reranking')
counts=p.groupby(['return_month','bucket']).size()
if len(counts)!=660 or not counts.eq(100).all():errors.append('rank_pool_shape')
if p.duplicated(['return_month','stock_code']).any():errors.append('overlapping_ranks')
if coverage[['n_nonfinance_mv_missing','n_finance_unknown']].to_numpy().sum():errors.append('decision_input_gaps')
if not coverage.n_finance_unknown_raw.eq(coverage.n_finance_unknown+coverage.n_finance_unknown_excluded_known_risk).all():errors.append('unexplained_raw_industry_gaps')
monthly=pd.read_csv(out/'monthly.csv',float_precision='round_trip')
if len(monthly)!=132*6*3 or monthly.net_return.isna().any():errors.append('monthly_shape')
if monthly.duplicated(['return_month','strategy','cost_bps']).any():errors.append('duplicate_month')
if set(monthly.strategy)!=set(c['buckets'])|{'CSI1000'}:errors.append('comparison_strategies')
bm_manifest=json.loads((ROOT/'data/prepared/benchmark_manifest.json').read_text())
for item in [*bm_manifest.get('artifacts',[]),*bm_manifest.get('raw_inputs',[]),bm_manifest['calendar']]:
 if sha(ROOT/item['path'])!=item['sha256']:errors.append('benchmark_hash:'+item['path'])
benchmark=pd.read_csv(ROOT/'data/prepared/benchmark_monthly.csv')
if len(benchmark)!=132 or benchmark.net_return.isna().any():errors.append('benchmark_months')
for cost in c['cost_bps']:
 observed=monthly[monthly.strategy.eq('CSI1000')&monthly.cost_bps.eq(cost)].sort_values('return_month')
 if not np.allclose(observed.net_return,benchmark.sort_values('return_month').net_return):errors.append('benchmark_cost_dependence')
performance=pd.read_csv(out/'performance.csv')
main_months=monthly[monthly.cost_bps.eq(c['main_cost_bps'])]
for strategy in [*c['buckets'],'CSI1000']:
 for period,start,stop in [('TRAIN',c['train_start'],c['train_end']),('TEST',c['test_start'],c['test_end'])]:
  values=main_months.loc[main_months.strategy.eq(strategy)&main_months.return_month.between(start,stop),'net_return']
  rf=(1+c['risk_free_rate_annual'])**(1/12)-1
  expected=np.sqrt(12)*(values-rf).mean()/values.std(ddof=1)
  row=performance[performance.strategy.eq(strategy)&performance.cost_bps.eq(c['main_cost_bps'])&performance.period.eq(period)].iloc[0]
  if not np.isclose(row.sharpe,expected):errors.append('sharpe:'+strategy+':'+period)
main=monthly[monthly.cost_bps.eq(30)]
replayed=select_winner(main,c)
if replayed['winner']!=sel['winner'] or replayed['training_hash']!=sel['training_hash']:errors.append('selection_replay')
for strategy,score in sel['scores'].items():
 row=performance[performance.strategy.eq(strategy)&performance.cost_bps.eq(30)&performance.period.eq('TRAIN')].iloc[0]
 if score['n_months']!=108 or not np.isclose(score['score'],row.sharpe):errors.append('training_score:'+strategy)
changed=main.copy();changed.loc[changed.return_month.ge(c['test_start']),'net_return']=999
if select_winner(changed,c)['training_hash']!=select_winner(main,c)['training_hash']:errors.append('future_returns_change_selection')
trades=pd.read_csv(out/'trades.csv')
if not np.allclose(trades.cost,trades.notional*.003):errors.append('trade_cost')
if not (pd.to_datetime(trades.plan_date)<pd.to_datetime(trades.date)).all():errors.append('same_open_order_plan')
if not (pd.to_datetime(trades.funding_asof)<pd.to_datetime(trades.date)).all():errors.append('same_open_funding')
if not np.allclose(trades.planned_units,trades.filled_units):errors.append('open_resized_quantity')
for row in trades[trades.reason.eq('ST_replacement')].itertuples():
 sales=trades[trades.strategy.eq(row.strategy)&trades.stock_code.eq(row.replaces)&trades.reason.eq('ST')&trades.date.lt(row.date)]
 if sales.empty:errors.append('replacement_without_prior_sale')
 if row.notional+row.cost>row.funding_budget+1e-10:errors.append('replacement_budget')
daily=pd.read_csv(out/'daily_nav.csv')
if daily.cash_weight.lt(-1e-10).any():errors.append('cash_borrowing')
plans=pd.read_csv(out/'order_plans.csv')
for strategy,g in daily[daily.strategy.isin(c['buckets'])].groupby('strategy'):
 g=g.sort_values('date');prior=dict(zip(g.date,g.cash.shift().fillna(1.)))
 budgets=plans[plans.strategy.eq(strategy)&plans.side.eq('buy')].groupby('execute_date').reserved_budget.sum()
 if any(budget>prior[date]+1e-9 for date,budget in budgets.items()):errors.append('preopen_budget:'+strategy)
ablation=pd.read_csv(out/'st_ablation_monthly.csv')
if len(ablation)!=132*5*3 or ablation.net_return.isna().any():errors.append('st_ablation_shape')
base=ablation[ablation.rule.eq('EXIT_REPLACE')].sort_values(['strategy','return_month'])
original=main[main.strategy.isin(c['buckets'])].sort_values(['strategy','return_month'])
if not np.allclose(base.net_return,original.net_return):errors.append('ablation_baseline_mismatch')
dr=pd.read_csv(out/'daily_risk.csv')
for row in performance[performance.period.isin(['TRAIN','TEST'])&performance.cost_bps.eq(30)].itertuples():
 if row.daily_max_drawdown>row.max_drawdown+1e-10:errors.append('daily_drawdown_less_severe:'+row.strategy)
audit=json.loads((out/'audit_summary.json').read_text())
if audit['company_month_cases']<20 or not audit['execution_checks_pass']:errors.append('audit_cases')
result=dict(raw_inputs_verified=len(raw['inputs']),prepared_rows=len(p),signal_months=p.return_month.nunique(),train_months=108,test_months=24,winner=sel['winner'],errors=errors,notebook_execution=json.loads((out/'report_manifest.json').read_text()).get('notebook_execution'),human_signoff=False,code_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'microcap').glob('*.py'))})
dump(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='code_sha256'},ensure_ascii=False,indent=2))
if errors:sys.exit(1)
