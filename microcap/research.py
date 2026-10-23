"""Final research entry points: daily execution and training-only selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from .acquire import sha,dump
from .execution import build_market,run
from .portfolio import select_winner
from .analysis import metrics,paired_tests,regressions,risk_statistics,_hac
from .workflow import clean,load
from .benchmark import prepare_benchmark
from .diagnostics import daily_risk,ablation_tests


def inputs(root):
    root=Path(root);config,panel,coverage=load(root)
    path=root/'data/prepared/daily_market.npz'
    if not path.exists(): build_market(root,pd.read_csv(root/'data/prepared/universe.csv',usecols=['stock_code']),config)
    manifest=json.loads((root/'data/prepared/daily_market_manifest.json').read_text())
    if sha(path)!=manifest['sha256']: raise ValueError('Daily input checksum failed')
    if config.get('selection_policy')=='exclude_observable_risk_then_rank':
        from . import execution,historical_status
        for key,value in [('config_sha256',sha(root/'config.json')),('builder_sha256',execution.market_builder_sha256()),('status_builder_sha256',sha(historical_status.__file__))]:
            if manifest.get(key)!=value:raise ValueError('Daily builder/protocol changed; run prepare before research')
    with np.load(path,allow_pickle=False) as z: market={k:z[k] for k in z.files}
    if not set(panel.stock_code)<=set(market['codes']): raise ValueError('Daily universe incomplete')
    panel.attrs['selection_universe_path']=str(root/'data/prepared/selection_universe.csv')
    return config,panel,coverage,market


def fingerprint(m,p,c):
    mask=pd.DatetimeIndex(m['dates'])<=pd.Period(c['train_end']).end_time
    h=hashlib.sha256()
    # Terminal dates after training are censored; future events cannot enter selection.
    for key in sorted(m):
        a=m[key]
        if key=='delist':a=np.where(a<=np.datetime64(pd.Period(c['train_end']).end_time,'D'),a,np.datetime64('NaT'))
        elif a.shape[0]==len(mask):a=a[mask]
        h.update(key.encode());h.update(np.ascontiguousarray(a).tobytes())
    q=p[p.return_month.le(c['train_end'])][['return_month','signal_date','stock_code','rank']].sort_values(['return_month','rank'])
    h.update(q.to_csv(index=False).encode())
    reserve=pd.read_csv(p.attrs['selection_universe_path'],parse_dates=['signal_date']) if 'selection_universe_path' in p.attrs else p.attrs.get('selection_universe')
    if reserve is not None:h.update(reserve[reserve.return_month.le(c['train_end'])].to_csv(index=False).encode())
    return h.hexdigest()


def select(root):
    root=Path(root);out=root/'results';out.mkdir(exist_ok=True)
    c,p,cov,m=inputs(root)
    expected=pd.period_range(c['train_start'],c['test_end'],freq='M').astype(str)
    if set(p.return_month)!=set(expected): raise ValueError('Expected 132 signal months')
    missing=cov[['n_nonfinance_mv_missing','n_finance_unknown']].sum()
    if missing.sum(): raise ValueError('Unresolved rank inputs: '+str(missing.to_dict()))
    train,_=run(m,p,c,c['main_cost_bps'],c['terminal_fraction'],end=c['train_end'])
    chosen=select_winner(train,c)
    chosen.update(status='frozen',config=c,config_sha256=sha(root/'config.json'),training_input_sha256=fingerprint(m,p,c))
    path=out/'selection.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('config_sha256')==chosen['config_sha256'] and old.get('status')=='frozen':
            if old.get('training_input_sha256')!=chosen['training_input_sha256'] or old['winner']!=chosen['winner']:
                raise ValueError('Frozen training input changed')
    dump(path,clean(chosen));train.to_csv(out/'selection_training_monthly.csv',index=False)
    print('Frozen training winner',chosen['winner'],flush=True);return chosen


def evaluate(root):
    root=Path(root);out=root/'results';c,p,cov,m=inputs(root)
    selected=json.loads((out/'selection.json').read_text())
    if selected['config_sha256']!=sha(root/'config.json') or selected['training_input_sha256']!=fingerprint(m,p,c):raise ValueError('Selection input mismatch')
    winner=selected['winner'];frames=[];logs={};scenarios=[];daily_summaries=[]
    baseline_details=None;baseline_monthly=None
    benchmark=prepare_benchmark(root)
    bd=pd.read_csv(root/'data/prepared/benchmark_daily.csv',parse_dates=['trade_date'])
    base_close=float(bd.close.iloc[0]);bd=bd[bd.trade_date.ge(pd.Period(c['train_start']).start_time)]
    benchmark_daily=pd.DataFrame(dict(strategy='CSI1000',date=bd.trade_date,nav=bd.close/base_close,cash_weight=np.nan))
    for cost in c['cost_bps']:
        monthly,details=run(m,p,c,cost,c['terminal_fraction'],details=True)
        ds=daily_risk(pd.concat([details['daily_nav'],benchmark_daily],ignore_index=True).assign(cost_bps=cost),c)
        daily_summaries.append(ds)
        # cost_bps labels the comparison scenario; the price index itself has no
        # simulated transaction cost and is not an investable ETF return.
        frames.extend([monthly,benchmark.assign(cost_bps=cost)])
        if cost==c['main_cost_bps']:
            baseline_details=details;baseline_monthly=monthly
            for name,f in details.items():logs.setdefault(name,[]).append(f)
            logs['daily_nav'].append(benchmark_daily)
        print('Executed cost',cost,flush=True)
    monthly=pd.concat(frames,ignore_index=True);monthly.to_csv(out/'monthly.csv',index=False)
    daily_summary=pd.concat(daily_summaries,ignore_index=True);daily_summary.to_csv(out/'daily_risk.csv',index=False)
    performance=metrics(monthly,c).merge(daily_summary,on=['strategy','cost_bps','period'],how='left',validate='one_to_one')
    performance.to_csv(out/'performance.csv',index=False)
    performance[performance.strategy.isin(c['buckets'])&performance.period.isin(['TRAIN','TEST','ALL'])].to_csv(out/'cost_sensitivity.csv',index=False)
    gradient=[];adjacent=[]
    for period,start,stop in [('TRAIN',c['train_start'],c['train_end']),('TEST',c['test_start'],c['test_end']),('ALL',c['train_start'],c['test_end'])]:
        values=performance[performance.period.eq(period)&performance.cost_bps.eq(c['main_cost_bps'])].set_index('strategy').reindex(list(c['buckets'])).cagr
        gradient.append(dict(period=period,smallest_best=bool(values.iloc[0]==values.max()),strictly_decreasing=bool(np.all(np.diff(values.to_numpy())<0)),best_band=values.idxmax(),rank_cagr_spearman=pd.Series(range(len(values))).corr(pd.Series(values.to_numpy()),method='spearman')))
        wide=monthly[monthly.cost_bps.eq(c['main_cost_bps'])].pivot(index='return_month',columns='strategy',values='net_return');wide.index=pd.PeriodIndex(wide.index,freq='M')
        bands=list(c['buckets'])
        for smaller,larger in zip(bands,bands[1:]):
            row=_hac(wide[smaller]-wide[larger],pd.period_range(start,stop,freq='M'),c['hac_lags'])
            row.update(period=period,smaller=smaller,larger=larger);adjacent.append(row)
    pd.DataFrame(gradient).to_csv(out/'size_gradient.csv',index=False)
    pd.DataFrame(adjacent).to_csv(out/'adjacent_band_tests.csv',index=False)
    for name,fs in logs.items():pd.concat(fs,ignore_index=True).to_csv(out/(name+'.csv'),index=False)
    dump(out/'paired_tests.json',clean(paired_tests(monthly,winner,c)))
    quotes=pd.DataFrame(m['close']*m['factor']).ffill().to_numpy();dates=pd.DatetimeIndex(m['dates']);code_index={code:i for i,code in enumerate(m['codes'])}
    exit_rows=[]
    all_trades=pd.concat(logs['trades'],ignore_index=True)
    for row in all_trades[all_trades.reason.eq('ST')].itertuples():
        t=dates.get_loc(pd.Timestamp(row.date));j=code_index[row.stock_code];before=t-int(row.delay_sessions)-1
        previous=quotes[before,j] if before>=0 else np.nan
        impact=m['open'][t,j]*m['factor'][t,j]/previous-1 if previous>0 else np.nan
        exit_rows.append(dict(strategy=row.strategy,stock_code=row.stock_code,exit_date=row.date,delay_sessions=row.delay_sessions,pretrigger_to_exit_gross_return=impact))
    pd.DataFrame(exit_rows).to_csv(out/'risk_exit_price_changes.csv',index=False)
    for fraction in c['terminal_sensitivities']:
        if fraction==c['terminal_fraction']:
            v=performance[performance.cost_bps.eq(c['main_cost_bps'])].copy()
        else:
            s,d=run(m,p,c,c['main_cost_bps'],fraction,details=True)
            ds=daily_risk(pd.concat([d['daily_nav'],benchmark_daily],ignore_index=True).assign(cost_bps=c['main_cost_bps']),c)
            s=pd.concat([s,benchmark.assign(cost_bps=c['main_cost_bps'])],ignore_index=True)
            v=metrics(s,c).merge(ds,on=['strategy','cost_bps','period'],validate='one_to_one')
        v['terminal_fraction']=fraction;scenarios.append(v)
    pd.concat(scenarios,ignore_index=True).to_csv(out/'terminal_sensitivity.csv',index=False)
    ablation_months=[];ablation_stats=[];ablation_execution=[]
    for rule,settings in c['st_ablation_rules'].items():
        rc={**c,**settings}
        if rule=='EXIT_REPLACE':a,d=baseline_monthly,baseline_details
        else:a,d=run(m,p,rc,c['main_cost_bps'],c['terminal_fraction'],details=True)
        ar=daily_risk(d['daily_nav'].assign(cost_bps=c['main_cost_bps']),c)
        ablation_months.append(a.assign(rule=rule))
        ablation_stats.append(metrics(a,c).merge(ar,on=['strategy','cost_bps','period'],validate='one_to_one').assign(rule=rule))
        for strategy in c['buckets']:
            tr=d['trades'];st=tr[tr.strategy.eq(strategy)] if len(tr) else tr
            ablation_execution.append(dict(rule=rule,strategy=strategy,risk_sells=int(st.reason.eq('ST').sum()) if len(st) else 0,replacement_buys=int(st.reason.eq('ST_replacement').sum()) if len(st) else 0))
        print('Executed ST rule',rule,flush=True)
    am=pd.concat(ablation_months,ignore_index=True);am.to_csv(out/'st_ablation_monthly.csv',index=False)
    pd.concat(ablation_stats,ignore_index=True).to_csv(out/'st_ablation_performance.csv',index=False)
    pd.DataFrame(ablation_execution).to_csv(out/'st_ablation_execution.csv',index=False)
    ablation_tests(am,c).to_csv(out/'st_ablation_tests.csv',index=False)
    risks,summary=risk_statistics(p,c);risks.to_csv(out/'rank_pool_risk_monthly.csv',index=False);summary.to_csv(out/'rank_pool_risk_summary.csv',index=False)
    events=pd.concat(logs['events'],ignore_index=True)
    unknown=events[events.event.eq('unverified_stale_mark')] if len(events) else events
    state_coverage_path=root/'data/prepared/status_coverage.csv'
    state_coverage=pd.read_csv(state_coverage_path) if state_coverage_path.exists() else pd.DataFrame()
    dump(out/'quality.json',clean(dict(months=len(expected_months(c)),train_months=108,test_months=24,
        complete_strategy_months=bool(monthly.net_return.notna().all()),unverified_stale_holding_days=len(unknown),
        ranking_missing=cov[['n_nonfinance_mv_missing','n_finance_unknown']].sum().to_dict(),
        ranking_missing_raw=cov[[x for x in ['n_nonfinance_mv_missing_raw','n_finance_unknown_raw'] if x in cov]].sum().to_dict(),
        missing_inputs_excluded_by_signal_risk=cov[[x for x in ['n_cap_missing_excluded_known_risk','n_finance_unknown_excluded_known_risk'] if x in cov]].sum().to_dict(),
        status_raw_top500_unknown_company_months=int(state_coverage.n_raw_top500_unknown.sum()) if len(state_coverage) else None,
        estimated_capitalization_rows=int(pd.read_csv(root/'data/repair_sources/early/market_cap.csv').shape[0]),
        publication_status='conditional_research_estimates' if unknown.empty else 'unresolved_marks')))
    dump(out/'evaluation_manifest.json',dict(config_sha256=sha(root/'config.json'),selection_sha256=sha(out/'selection.json'),daily_input_sha256=sha(root/'data/prepared/daily_market.npz'),
        artifacts=[dict(path=str(f.relative_to(root)),sha256=sha(f)) for f in [out/name for name in ['monthly.csv','performance.csv','trades.csv','order_plans.csv','holdings.csv','daily_nav.csv','daily_risk.csv','events.csv','blocked_orders.csv','paired_tests.json','cost_sensitivity.csv','terminal_sensitivity.csv','st_ablation_monthly.csv','st_ablation_performance.csv','st_ablation_tests.csv','st_ablation_execution.csv']]+[root/'data/prepared/benchmark_monthly.csv',root/'data/prepared/benchmark_manifest.json']]))
    return monthly


def expected_months(c):return pd.period_range(c['train_start'],c['test_end'],freq='M')


def attribute(root):
    root=Path(root);out=root/'results';c,p,cov,m=inputs(root)
    dates=pd.DatetimeIndex(m['dates']);ix={s:i for i,s in enumerate(m['codes'])}
    marks=pd.DataFrame(m['close']*m['factor']).ffill().to_numpy()
    values=[];prior_returns=[]
    suspended=m.get('suspended_any',m['suspended'])
    for row in p.itertuples():
        j=ix[row.stock_code];a=dates.searchsorted(row.signal_date,side='right')-1;b=dates.searchsorted(row.horizon_end,side='right')-1
        entry=marks[a,j];exit_value=marks[b,j]
        if m['delist'][j]<=np.datetime64(row.horizon_end,'D'):exit_value=0.
        # Endpoints require a quote, a documented suspension, or terminal policy.
        valid_start=np.isfinite(m['close'][a,j]) or suspended[a,j]
        valid_end=np.isfinite(m['close'][b,j]) or suspended[b,j] or exit_value==0
        values.append(exit_value/entry-1 if valid_start and valid_end and entry>0 else np.nan)
        before=dates.searchsorted((pd.Period(row.return_month)-2).end_time,side='right')-1
        valid_before=before>=0 and (np.isfinite(m['close'][before,j]) or suspended[before,j])
        previous=marks[before,j] if before>=0 else np.nan
        prior_returns.append(entry/previous-1 if valid_start and valid_before and previous>0 else np.nan)
    p['return_1m']=values;p['return_prev1m']=prior_returns;p.to_csv(out/'attribution_panel.csv',index=False)
    coeff,status,summary=regressions(p,c)
    coeff.to_csv(out/'regression_coefficients.csv',index=False);status.to_csv(out/'regression_coverage.csv',index=False);summary.to_csv(out/'regression_summary.csv',index=False)
    return summary
