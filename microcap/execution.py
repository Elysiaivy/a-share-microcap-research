"""Daily-bar execution with orders fixed before the opening auction.

Close prices determine order quantities.  A sale's cash is available for planning
from the following session, never for another fill at that same opening price.
"""
from pathlib import Path
import inspect,hashlib
import re
import numpy as np
import pandas as pd
from .acquire import FrozenStore, sha, dump
from .prepare import catalog, aliases


def risk_name(name):
    name = str(name).upper()
    return 'ST' in name or 'PT' in name or '退' in name


def suspended_at_open(timing):
    """Whether a suspend_d record makes the daily open unsuitable for a fill.

    Missing/ambiguous timing is conservative.  Explicit afternoon or later
    intraday intervals do not erase the observed 09:30 opening auction.  A halt
    covering 09:30 is blocked: the daily `open` might instead be the restart.
    """
    if pd.isna(timing) or not str(timing).strip():
        return True
    text = str(timing).strip().replace('：', ':').replace('—', '-').replace('－', '-')
    intervals = re.findall(r'(\d{1,2}):(\d{2})\s*[-~～至]\s*(\d{1,2}):(\d{2})', text)
    if not intervals:
        return True
    # Reject partial parses rather than assume an unparsed interval is harmless.
    remainder = re.sub(r'\d{1,2}:\d{2}\s*[-~～至]\s*\d{1,2}:\d{2}', '', text)
    if remainder.strip(' ,，;；/、'):
        return True
    for h1, m1, h2, m2 in intervals:
        start, end = int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)
        if not (0 <= start < end <= 24 * 60):
            return True
        if start <= 9 * 60 + 30 < end:
            return True
    return False


def market_builder_sha256():
    """Hash the data transformation, independently of trading-order logic."""
    source = inspect.getsource(build_market)
    marker = "    target = root / 'data/prepared/daily_market.npz'"
    if marker not in source:
        raise ValueError('Unrecognized market builder layout')
    parts = [source.split(marker)[0], inspect.getsource(suspended_at_open),
             inspect.getsource(aliases), inspect.getsource(catalog)]
    return hashlib.sha256('\n'.join(parts).encode()).hexdigest()


def build_market(root, panel, config):
    root = Path(root); store = FrozenStore(root)
    securities, calendar, _, names = catalog(store)
    dates = calendar[(calendar >= '2014-09-01') & (calendar <= '2025-12-31')]
    codes = sorted(panel.stock_code.unique()); index = {s: i for i, s in enumerate(codes)}
    shape = (len(dates), len(codes))
    arrays = {k: np.full(shape, np.nan) for k in ['open', 'close', 'factor', 'up', 'down']}
    suspended = np.zeros(shape, bool); suspended_any = np.zeros(shape, bool)
    for t, day in enumerate(dates):
        ds = day.strftime('%Y%m%d')
        for api, fields, mapping in [
            ('daily', 'ts_code,trade_date,open,close', {'open': 'open', 'close': 'close'}),
            ('adj_factor', 'ts_code,trade_date,adj_factor', {'adj_factor': 'factor'}),
            ('stk_limit', 'ts_code,trade_date,up_limit,down_limit', {'up_limit': 'up', 'down_limit': 'down'})]:
            f = store.date_query(api, ds, fields)
            if api == 'adj_factor':
                for a in config.get('security_aliases', []):
                    if a['canonical'] in set(f.ts_code):
                        f = f[~f.ts_code.eq(a['old'])]
            f = aliases(f, day, config); f = f[f.ts_code.isin(index)]
            ix = f.ts_code.map(index).to_numpy(int)
            for source, target in mapping.items():
                arrays[target][t, ix] = pd.to_numeric(f[source], errors='coerce')
        f = store.date_query('suspend_d', ds, 'ts_code,trade_date,suspend_timing,suspend_type')
        if len(f):
            f = aliases(f[f.suspend_type.eq('S')], day, config)
            f = f[f.ts_code.isin(index)]
            for row in f.itertuples():
                j = index[row.ts_code]
                suspended_any[t, j] = True
                suspended[t, j] |= suspended_at_open(row.suspend_timing)
        if t % 250 == 0:
            print('daily inputs', ds, flush=True)
    from . import historical_status
    state, known_ST, coverage = historical_status.build_states(store, codes, dates, config, names)
    for field in ['signal_state', 'signal_known_ST']:
        if field in coverage:
            arrays[field] = coverage.pop(field)
    delist = securities.set_index('ts_code').delist_date.reindex(codes)
    arrays.update(suspended=suspended, suspended_any=suspended_any, state=state,
                  known_ST=known_ST, dates=dates.to_numpy(dtype='datetime64[D]'), codes=np.array(codes),
                  delist=delist.to_numpy(dtype='datetime64[D]'))
    target = root / 'data/prepared/daily_market.npz'
    np.savez_compressed(target, **arrays); store.save_manifest()
    dump(root / 'data/prepared/daily_market_manifest.json',
         dict(sha256=sha(target), config_sha256=sha(root / 'config.json'),
              builder_sha256=market_builder_sha256(),
              status_builder_sha256=sha(Path(historical_status.__file__)),
              n_dates=len(dates), n_stocks=len(codes),
              suspension_policy='block only intervals covering open; missing timing conservative',
              status_coverage=coverage.get('summary', {})))
    return arrays


def run(market, panel, config, cost_bps=30, terminal_fraction=0., peer=None, end=None, details=False,
        implementation=None, exposure_schedule=None):
    """Execute daily orders without observing an opening price before planning.

    Monthly target *share* counts are fixed using previous-close NAV/prices.
    Purchases are funded only by cash present at the preceding close.  Unfilled
    monthly target deficits retry on later sessions.  Risk proceeds enter a
    separate replacement budget eligible from t + replacement_delay_sessions.
    A failed replacement order cannot jump to another stock at that same open.

    ``exposure_schedule`` is an optional DataFrame with ``return_month``,
    ``signal_date`` and ``exposure`` columns.  Each execution month must have
    exactly one 0--1 exposure, observed no later than its preceding session's
    close.  Exposure sets monthly target shares and a cash reserve using that
    preceding NAV; it does not scale already-realized returns.  The reserve is
    protected from ordinary and risk-replacement purchases throughout the month.
    Existing positions may remain above target when their sales cannot execute.
    """
    dates = pd.DatetimeIndex(market['dates']); codes = market['codes']; n = len(codes)
    index = {s: i for i, s in enumerate(codes)}
    known_ST = market.get('known_ST', market['state'] == 1)
    signal_state = market.get('signal_state', market['state'])
    signal_risk = market.get('signal_known_ST', known_ST)
    reserve_path = panel.attrs.get('selection_universe_path')
    reserve_panel = pd.read_csv(reserve_path) if reserve_path else panel.attrs.get('selection_universe', panel)
    replenish = bool(config.get('intramonth_replacement', False))
    exit_enabled = bool(config.get('risk_exit_enabled', True))
    delay = int(config.get('replacement_delay_sessions', 1))
    if delay < 1:
        raise ValueError('Daily replacement requires at least one session after its funding sale')
    defs = dict(config['buckets']) if peer is None else {'PEERS': None}
    reserves = {m: np.array([index[x] for x in g.sort_values('rank').stock_code], int)
                for m, g in reserve_panel.groupby('return_month')}
    reserve_ranks = {m: g.sort_values('rank')['rank'].to_numpy() for m, g in reserve_panel.groupby('return_month')}
    frames = []; trades = []; holds = []; blocks = []; events = []; daily = []; plans = []
    start = pd.Period(config['train_start']).start_time
    stop = pd.Period(end or config['test_end']).end_time
    exposures = {}
    if exposure_schedule is not None:
        required = {'return_month', 'signal_date', 'exposure'}
        if not isinstance(exposure_schedule, pd.DataFrame) or not required.issubset(exposure_schedule.columns):
            raise ValueError('Exposure schedule requires return_month, signal_date and exposure columns')
        schedule = exposure_schedule.copy()
        schedule['return_month'] = schedule['return_month'].map(lambda value: str(pd.Period(value, freq='M')))
        if schedule.return_month.duplicated().any():
            raise ValueError('Exposure schedule contains duplicate execution months')
        schedule['signal_date'] = pd.to_datetime(schedule.signal_date, errors='raise')
        schedule['exposure'] = pd.to_numeric(schedule.exposure, errors='raise')
        if (schedule.signal_date.isna().any() or not np.isfinite(schedule.exposure).all()
                or not schedule.exposure.between(0., 1.).all()):
            raise ValueError('Exposure must be finite and between zero and one, with a known signal date')
        observations = schedule.set_index('return_month')
        execution_dates = dates[(dates >= start) & (dates <= stop)]
        first_sessions = pd.Series(execution_dates, index=execution_dates.to_period('M')).groupby(level=0).first()
        for period, first_day in first_sessions.items():
            month = str(period)
            if month not in observations.index:
                raise ValueError(f'Missing exposure for execution month {month}')
            position = dates.get_loc(first_day)
            if position == 0 or observations.loc[month, 'signal_date'] > dates[position - 1]:
                raise ValueError(f'Exposure signal is not known before execution month {month}')
            if 'signal_date' in panel:
                selection_asof = pd.to_datetime(panel.loc[panel.return_month.eq(month), 'signal_date'])
                if (selection_asof.empty or selection_asof.nunique() != 1
                        or selection_asof.iloc[0] > dates[position - 1]
                        or observations.loc[month, 'signal_date'] > selection_asof.iloc[0]):
                    raise ValueError(f'Exposure and selection signal dates disagree for execution month {month}')
            exposures[month] = float(observations.loc[month, 'exposure'])
    rate = cost_bps / 10000
    if implementation is not None:
        implementation.validate(rate)
    initial = implementation.capital if implementation is not None else 1.
    cash_tolerance = max(1e-10, initial * 1e-12) if implementation is not None else 1e-10
    fee_for = (lambda amount: implementation.fee(amount, rate)) if implementation is not None else (lambda amount: amount * rate)
    for strategy, bounds in defs.items():
        selected = (panel[panel.bucket.isin([c for c in config['candidates'] if c != peer])] if peer
                    else panel[panel['rank'].between(*bounds)])
        size = 300 if peer else bounds[1] - bounds[0] + 1
        groups = {m: g for m, g in selected.groupby('return_month')}
        units = np.zeros(n); marks = np.full(n, np.nan); cash = initial; previous_nav = initial; month_nav = initial
        forced = np.zeros(n, bool); forced_since = np.full(n, -1)
        # A persistent ordinary sell is a fixed share quantity, never recomputed at the open.
        sell_left = np.zeros(n); desired_units = np.zeros(n); target = np.zeros(n, bool)
        deferred_budget = np.zeros(n)
        oldmonth = None; turnover = 0.; costs = 0.; idle_risk_cash = 0.
        active_exposure = 1.; protected_cash = 0.
        banned = np.zeros(n, bool); replacement_queue = []; spare = np.array([], int)
        for t, day in enumerate(dates):
            factor = market['factor'][t]; close = market['close'][t] * factor; op = market['open'][t] * factor
            valid = np.isfinite(close) & (close > 0)
            if day < start:
                marks[valid] = close[valid]
                continue
            if day > stop:
                break
            if t == 0:
                raise ValueError('Execution needs a preceding close for order planning')
            plan_date = str(dates[t - 1].date()); date = str(day.date())
            month = str(day.to_period('M')); newmonth = month != oldmonth
            state = market['state'][t]
            # State[t] contains only public information usable before this open.
            trigger = ((units > 0) & known_ST[t] & ~forced) if exit_enabled else np.zeros(n, bool)
            forced[trigger] = True; forced_since[trigger] = t; banned[trigger] = True
            desired_units[trigger] = 0.; target[trigger] = False
            if details:
                for j in np.flatnonzero(trigger):
                    events.append(dict(strategy=strategy, date=date, stock_code=codes[j], event='ST_trigger'))
            # Freeze buying power before any same-session liquidation or terminal proceeds.
            cash_before = cash
            # The terminal date values an already-held asset. It never screens a candidate.
            dead = (market['delist'] <= np.datetime64(day, 'D')) & (units > 0)
            for j in np.flatnonzero(dead):
                value = units[j] * marks[j]
                if not np.isfinite(value):
                    raise ValueError('Unmarked terminal holding')
                cash += value * terminal_fraction; units[j] = 0; forced[j] = False; sell_left[j] = 0
                desired_units[j] = 0; target[j] = False
                if details:
                    events.append(dict(strategy=strategy, date=date, stock_code=codes[j], event='terminal',
                                       marked_value=value, residual=value * terminal_fraction))
            if newmonth:
                # EXIT_CASH keeps each risk-sale receipt idle only until this
                # scheduled rebalance. It remains part of NAV throughout.
                idle_risk_cash = 0.
                replacement_queue = []; banned = forced.copy()
                turnover = 0.; costs = 0.; month_nav = previous_nav
                active_exposure = exposures.get(month, 1.)
                protected_cash = month_nav * (1. - active_exposure)
                g = groups.get(month)
                if g is None or len(g) != size:
                    raise ValueError(f'Incomplete rank pool: {strategy} {month}')
                target[:] = False
                ix = np.array([index[s] for s in g.stock_code]); target[ix] = True
                spare = reserves[month][reserve_ranks[month] > g['rank'].max()]
                # Known first-session risk may disqualify a signal-day target, but
                # current quotes/limits are not examined when choosing its successor.
                target &= (state == 0) & ~known_ST[t] & ~forced
                for j in spare:
                    if target.sum() >= size:
                        break
                    if state[j] == 0 and not known_ST[t, j] and not forced[j]:
                        target[j] = True
                priced = target & np.isfinite(marks) & (marks > 0)
                unpriced = target & ~priced
                desired_units[:] = 0.
                desired_units[priced] = month_nav * active_exposure / size / marks[priced]
                # Keep the unpriced name's slot and budget. It is neither deleted
                # nor funded using its eventual reopening price retrospectively.
                deferred_budget[:] = 0.
                deferred_budget[unpriced] = month_nav * active_exposure / size
                if details:
                    for j in np.flatnonzero(unpriced):
                        events.append(dict(strategy=strategy, date=date, stock_code=codes[j],
                                           event='pending_unpriced_target', slot_budget=deferred_budget[j]))
                sell_left = np.maximum(units - desired_units, 0.)
            # Marks are still yesterday's observations here. A first reopening
            # close becomes usable only at the following session's planning.
            activate = ((deferred_budget > 0) & target & np.isfinite(marks) & (marks > 0)
                        & (state == 0) & ~known_ST[t] & ~forced & ~banned)
            for j in np.flatnonzero(activate):
                desired_units[j] = deferred_budget[j] / marks[j]
                if details:
                    events.append(dict(strategy=strategy, date=date, stock_code=codes[j],
                                       event='unpriced_target_priced', slot_budget=deferred_budget[j],
                                       observed_close=marks[j], observed_asof=plan_date))
                deferred_budget[j] = 0.
            desired_units[forced] = 0.
            sell_qty = np.where(forced, units, np.minimum(sell_left, units))
            if implementation is not None and implementation.lot:
                partial = (sell_qty > 0) & (sell_qty < units - 1e-12) & np.isfinite(factor)
                sell_qty[partial] = implementation.round_units(sell_qty[partial], factor[partial])
            # Preselect replacements from yesterday's public state and the frozen
            # monthly reserve. Today's opening price/limit cannot change this name.
            buy_orders = []
            # Daily exchange limits are public before the opening auction. Reserve
            # cash at this upper price, so an opening gap cannot create an
            # unaffordable order or free funding for a different same-open order.
            buy_limit = market['up'][t] * factor
            reserved = sum(q['budget'] for q in replacement_queue)
            reserved_names = set()
            replacement_capacity = max(0., cash_before - protected_cash - idle_risk_cash - deferred_budget.sum())
            if replenish:
                for q in replacement_queue:
                    if t < q['eligible_t']:
                        continue
                    candidates = [j for j in spare
                                  if signal_state[t - 1, j] == 0 and not signal_risk[t - 1, j]
                                  and not banned[j] and not target[j] and units[j] == 0
                                  and j not in reserved_names and np.isfinite(marks[j]) and marks[j] > 0]
                    if not candidates:
                        continue
                    # A rejected candidate is skipped only on a later session.
                    tried = q.get('tried', set())
                    fresh = [j for j in candidates if j not in tried]
                    j = fresh[0] if fresh else candidates[0]
                    reserved_names.add(j)
                    if not np.isfinite(buy_limit[j]) or buy_limit[j] <= 0:
                        continue
                    budget = (min(q['budget'], replacement_capacity) if active_exposure < 1. else q['budget'])
                    quantity = budget / (buy_limit[j] * (1 + rate))
                    if implementation is not None:
                        quantity = implementation.affordable(quantity, factor[j], buy_limit[j], budget, rate)
                    if quantity <= 0:
                        continue
                    buy_orders.append(dict(j=j, quantity=quantity, reason='ST_replacement', queue=q,
                                           limit_price=buy_limit[j], funding_budget=budget,
                                           funding_asof=q['funding_date']))
                    replacement_capacity = max(0., replacement_capacity - budget)
            # Ordinary monthly target deficits use only unreserved prior-close cash.
            available = max(0., cash_before - reserved - idle_risk_cash - deferred_budget.sum() - protected_cash)
            needed = np.maximum(desired_units - units, 0.)
            needed[~target | forced | (state != 0) | known_ST[t]
                   | ~np.isfinite(buy_limit) | (buy_limit <= 0)] = 0.
            notional = needed * np.nan_to_num(buy_limit)
            full_budget = notional + fee_for(notional)
            total = full_budget.sum() if implementation is not None else notional.sum() * (1 + rate)
            scale = min(1., available / total) if total > 0 else 0.
            for j in np.flatnonzero(needed * scale > 1e-14):
                quantity = needed[j] * scale
                budget = notional[j] * scale * (1 + rate)
                if implementation is not None:
                    budget = full_budget[j] * scale
                    quantity = implementation.affordable(quantity, factor[j], buy_limit[j], budget, rate)
                if quantity <= 0:
                    continue
                buy_orders.append(dict(j=j, quantity=quantity, reason='rebalance',
                                       limit_price=buy_limit[j],
                                       funding_budget=budget,
                                       funding_asof=plan_date))
            if details:
                for order in buy_orders:
                    plans.append(dict(strategy=strategy, plan_date=plan_date, execute_date=date,
                                      stock_code=codes[order['j']], side='buy', planned_units=order['quantity'],
                                      previous_close=marks[order['j']], reason=order['reason'],
                                      limit_price=order['limit_price'], reserved_budget=order['funding_budget'],
                                      funding_budget=order['funding_budget'], funding_asof=order['funding_asof']))
            quote = np.isfinite(op) & (op > 0) & ~market['suspended'][t]
            can_sell = quote & np.isfinite(market['down'][t]) & (market['open'][t] > market['down'][t] + 1e-7)
            can_buy = quote & np.isfinite(market['up'][t]) & (market['open'][t] < market['up'][t] - 1e-7)
            # Turnover uses previous close NAV rather than an unobserved opening NAV.
            nav_base = previous_nav
            for j in np.flatnonzero(sell_qty > 1e-14):
                reason = 'ST' if forced[j] else 'rebalance'
                if details:
                    plans.append(dict(strategy=strategy, plan_date=plan_date, execute_date=date,
                                      stock_code=codes[j], side='sell', planned_units=sell_qty[j],
                                      previous_close=marks[j], reason=reason, funding_asof=plan_date))
                if not can_sell[j]:
                    if details:
                        blocks.append(dict(strategy=strategy, date=date, stock_code=codes[j], side='sell',
                                           ST_order=bool(forced[j]), reason='open_untradeable', planned_units=sell_qty[j],
                                           plan_date=plan_date))
                    continue
                quantity = min(sell_qty[j], units[j]); amount = quantity * op[j]; fee = float(fee_for(amount))
                if forced[j] and not newmonth and not replenish:
                    idle_risk_cash += amount - fee
                if replenish and forced[j] and not newmonth:
                    replacement_queue.append(dict(budget=amount - fee, origin=codes[j], exit_t=t,
                                                  eligible_t=t + delay, funding_date=date, tried=set()))
                units[j] -= quantity; cash += amount - fee; sell_left[j] = max(0., sell_left[j] - quantity)
                turnover += amount / nav_base; costs += fee
                if details:
                    trades.append(dict(strategy=strategy, date=date, stock_code=codes[j], side='sell',
                                       notional=amount, cost=fee, reason=reason,
                                       delay_sessions=t - forced_since[j] if forced[j] else 0,
                                       plan_date=plan_date, execute_date=date, planned_units=quantity,
                                       filled_units=quantity, previous_close=marks[j], cash_before_open=cash_before,
                                       funding_asof=plan_date))
                if units[j] < 1e-12:
                    units[j] = 0.; forced[j] = False; sell_left[j] = 0.
            # Reserved budgets and names are fixed before the auction. A cheaper
            # fill or a rejected order does not enlarge another order at this open.
            if sum(o['funding_budget'] for o in buy_orders) > cash_before + cash_tolerance:
                raise ValueError('Pre-open order budgets exceed prior-close cash')
            if (active_exposure < 1. and sum(o['funding_budget'] for o in buy_orders)
                    > max(0., cash_before - protected_cash) + cash_tolerance):
                raise ValueError('Pre-open order budgets spend protected exposure cash')
            for order in buy_orders:
                j = order['j']; quantity = order['quantity']; q = order.get('queue')
                amount = quantity * op[j]; fee = float(fee_for(amount))
                rejection = None
                if not can_buy[j]:
                    rejection = 'open_untradeable'
                elif state[j] != 0 or known_ST[t, j] or banned[j]:
                    rejection = 'risk_state'
                elif op[j] > order['limit_price'] + 1e-10:
                    rejection = 'above_planned_limit'
                if rejection:
                    if q is not None:
                        q['tried'].add(j)
                    if details:
                        blocks.append(dict(strategy=strategy, date=date, stock_code=codes[j], side='buy',
                                           ST_order=False, reason=rejection, order_reason=order['reason'],
                                           planned_units=quantity, plan_date=plan_date))
                    continue
                if amount + fee > order['funding_budget'] + cash_tolerance:
                    raise ValueError('Fill exceeded its pre-open reserved budget')
                units[j] += quantity; cash -= amount + fee
                turnover += amount / nav_base; costs += fee
                if q is not None:
                    target[j] = True; desired_units[j] = units[j]
                    # Unspent gap-down cash returns to the general cash balance.
                    replacement_queue.remove(q)
                if details:
                    trades.append(dict(strategy=strategy, date=date, stock_code=codes[j], side='buy',
                                       notional=amount, cost=fee, reason=order['reason'],
                                       delay_sessions=t - q['exit_t'] if q is not None else 0,
                                       replaces=q['origin'] if q is not None else None,
                                       plan_date=plan_date, execute_date=date, planned_units=quantity,
                                       filled_units=quantity, previous_close=marks[j],
                                       limit_price=order['limit_price'], reserved_budget=order['funding_budget'],
                                       funding_budget=order['funding_budget'], funding_asof=order['funding_asof'],
                                       cash_before_open=cash_before))
            if np.any((units > 0) & np.isfinite(market['close'][t]) & ~np.isfinite(factor)):
                raise ValueError(f'Missing factor on quoted holding {day}')
            if details:
                all_suspended = market.get('suspended_any', market['suspended'])
                unexplained = (units > 0) & ~valid & ~all_suspended[t]
                for j in np.flatnonzero(unexplained):
                    events.append(dict(strategy=strategy, date=date, stock_code=codes[j], event='unverified_stale_mark'))
            marks[valid] = close[valid]
            nav = cash + np.nansum(units * marks)
            if idle_risk_cash > cash + max(1e-9, cash_tolerance):
                raise ValueError('EXIT_CASH risk-sale reserve was spent before monthly rebalance')
            if cash < -max(1e-9, cash_tolerance) or not np.isfinite(nav) or nav <= 0:
                raise ValueError('Cash/NAV invariant')
            if details:
                daily.append(dict(strategy=strategy, date=date, nav=nav, cash=cash,
                                  cash_weight=cash / nav, invested_weight=1 - cash / nav,
                                  target_exposure=active_exposure, protected_cash=protected_cash,
                                  idle_risk_cash=idle_risk_cash, idle_risk_cash_weight=idle_risk_cash / nav,
                                  hold_count=int((units > 0).sum()), target_count=int(target.sum()),
                                  pending_sell_count=int(((sell_left > 0) | forced).sum()),
                                  pending_unpriced_target=int((deferred_budget > 0).sum()),
                                  unpriced_target_budget=float(deferred_budget.sum()),
                                  replacement_budget=sum(q['budget'] for q in replacement_queue)))
            last = (t == len(dates) - 1 or dates[t + 1].to_period('M') != day.to_period('M')
                    or dates[t + 1] > stop)
            if last:
                frames.append(dict(strategy=strategy, return_month=month, net_return=nav / month_nav - 1,
                                   turnover=turnover, cost_bps=cost_bps, cost=costs / month_nav,
                                   nav=nav, cash_weight=cash / nav, status='ok'))
                if details:
                    for j in np.flatnonzero(units > 0):
                        holds.append(dict(strategy=strategy, return_month=month, stock_code=codes[j],
                                          weight=units[j] * marks[j] / nav, ST_state=int(state[j]),
                                          pending_ST=bool(forced[j])))
            previous_nav = nav; oldmonth = month
    return pd.DataFrame(frames), dict(trades=pd.DataFrame(trades), holdings=pd.DataFrame(holds),
                                     blocked_orders=pd.DataFrame(blocks), events=pd.DataFrame(events),
                                     daily_nav=pd.DataFrame(daily), order_plans=pd.DataFrame(plans))
