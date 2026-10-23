"""Daily risk summaries and explicit experiment comparisons."""
import numpy as np
import pandas as pd
from .analysis import _periods, _hac


def daily_risk(daily, config):
    rows=[]
    keys=['strategy']+(['cost_bps'] if 'cost_bps' in daily else [])
    for key, frame in daily.groupby(keys,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        frame=frame.copy();frame['date']=pd.to_datetime(frame.date)
        frame=frame.sort_values('date').set_index('date')
        if frame.index.has_duplicates:raise ValueError('Duplicate daily strategy date')
        for period, months in _periods(config,annual=True).items():
            g=frame.loc[(frame.index>=months[0].start_time)&(frame.index<=months[-1].end_time)]
            if g.empty:continue
            prior=frame.loc[frame.index<g.index[0],'nav']
            base=float(prior.iloc[-1]) if len(prior) else 1.0
            values=np.r_[base,g.nav.to_numpy(float)]
            if not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Invalid daily NAV')
            peaks=np.maximum.accumulate(values)
            drawdown=values/peaks-1
            trough=int(np.argmin(drawdown));peak=int(np.argmax(values[:trough+1]))
            baseline_date=prior.index[-1] if len(prior) else pd.Timestamp(config.get('initial_signal',g.index[0]-pd.offsets.BDay()))
            dates=[baseline_date,*g.index]
            recovery=np.flatnonzero(values[trough+1:]>=values[peak])
            recovered=trough+1+int(recovery[0]) if len(recovery) else None
            ret=values[1:]/values[:-1]-1
            row=dict(zip(keys,key))
            row.update(period=period,daily_max_drawdown=float(drawdown[trough]),
                       worst_day=float(ret.min()),drawdown_peak=str(pd.Timestamp(dates[peak]).date()),
                       drawdown_trough=str(pd.Timestamp(dates[trough]).date()),
                       recovery_date=str(pd.Timestamp(dates[recovered]).date()) if recovered is not None else None,
                       recovery_sessions=(recovered-peak) if recovered is not None else None,
                       drawdown_unrecovered=recovered is None,
                       n_daily=len(g),mean_cash=float(g.cash_weight.mean()) if 'cash_weight' in g else np.nan,
                       max_cash=float(g.cash_weight.max()) if 'cash_weight' in g else np.nan,
                       mean_holdings=float(g.hold_count.mean()) if 'hold_count' in g else np.nan,
                       min_holdings=float(g.hold_count.min()) if 'hold_count' in g else np.nan,
                       max_holdings=float(g.hold_count.max()) if 'hold_count' in g else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def ablation_tests(monthly, config):
    rows=[]
    for strategy in config['buckets']:
        wide=monthly[monthly.strategy.eq(strategy)].pivot(index='return_month',columns='rule',values='net_return')
        wide.index=pd.PeriodIndex(wide.index,freq='M')
        for treatment,reference in [('EXIT_CASH','FILTER_ONLY'),('EXIT_REPLACE','EXIT_CASH')]:
            for period,months in _periods(config).items():
                row=_hac(wide[treatment]-wide[reference],months,config['hac_lags'])
                row.update(strategy=strategy,rule=treatment,reference_rule=reference,period=period)
                rows.append(row)
    return pd.DataFrame(rows)
