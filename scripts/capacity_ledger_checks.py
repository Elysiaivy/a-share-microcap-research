"""Reconcile private execution ledgers independently of order-generation code.

The checks establish accounting and declared daily-model consistency. They do
not establish that an order could obtain the daily VWAP in an actual queue.
Only the compact returned check report should enter the public release.
"""
import numpy as np
import pandas as pd


def require(condition, message):
    if not bool(condition):
        raise ValueError('Capacity ledger: ' + message)


def close(actual, expected, message, *, atol=1e-7, rtol=1e-10):
    require(np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                        atol=atol, rtol=rtol, equal_nan=False), message)


def table(details, name, required):
    require(name in details, f'missing {name}')
    result = details[name].copy().reset_index(drop=True)
    if result.empty:
        return result.reindex(columns=list(dict.fromkeys([*result.columns, *required])))
    require(set(required).issubset(result), f'{name} missing columns {set(required) - set(result)}')
    return result


def verify_capacity_ledgers(monthly, details, market, protocol, scenario, initial_capital,
                            *, terminal_fraction=0.):
    """Validate one period/scenario/capital run before any aggregate is accepted.

    Call immediately after ``run(..., details=True)``. ``details`` must retain
    ``order_plans``; the compact published attempt summary cannot replace it.
    ``market`` is the already-loaded frozen matrix, not fetched by this helper.
    """
    require(initial_capital > 0, 'initial capital must be positive')
    rule = protocol['scenarios'][scenario]
    constrained = rule['execution'] != 'original_next_open'
    eta = float(rule['impact_eta'])
    participation = rule['participation']
    rate = protocol['base_cost_bps'] / 10000
    commission = protocol['commission_bps'] / 10000
    minimum = protocol['minimum_commission_yuan']
    lot = protocol['buy_lot_shares']
    require(0 <= commission <= rate and minimum >= 0 and lot >= 0, 'invalid fee/lot protocol')
    require(0 <= terminal_fraction <= 1, 'invalid terminal valuation fraction')
    fee = lambda amount: amount * (rate - commission) + np.maximum(amount * commission, minimum)
    money_tolerance = max(1e-7, initial_capital * 1e-11)
    dates = pd.DatetimeIndex(pd.to_datetime(market['dates'])).normalize()
    codes = pd.Index(market['codes'])
    require(dates.is_unique and dates.is_monotonic_increasing and codes.is_unique, 'ambiguous market axes')
    daily = table(details, 'daily_nav', ['strategy', 'date', 'nav', 'cash', 'cash_weight', 'hold_count'])
    trades = table(details, 'trades', ['strategy', 'date', 'stock_code', 'side', 'reason',
                                     'planned_units', 'filled_units', 'notional', 'cost',
                                     'previous_close', 'plan_date', 'funding_asof', 'cash_before_open'])
    plans = table(details, 'order_plans', ['strategy', 'execute_date', 'stock_code', 'side',
                                         'reason', 'planned_units', 'previous_close', 'plan_date', 'funding_asof'])
    events = table(details, 'events', ['strategy', 'date', 'event'])
    require(not daily.empty and daily.strategy.nunique() == 1, 'requires one nonempty strategy ledger')
    strategy = daily.strategy.iloc[0]
    require((trades.strategy == strategy).all() and (plans.strategy == strategy).all(), 'mixed strategies')
    daily['date'] = pd.to_datetime(daily.date, errors='raise').dt.normalize()
    require(daily.date.notna().all() and daily.date.is_unique, 'missing or duplicate daily date')
    daily = daily.sort_values('date').reset_index(drop=True)
    ix_daily = dates.get_indexer(daily.date)
    require((ix_daily > 0).all() and np.array_equal(ix_daily, np.arange(ix_daily[0], ix_daily[-1] + 1)),
            'daily ledger omits a frozen market session or has no preceding close')
    for frame, date_field in [(plans, 'execute_date'), (trades, 'date'), (events, 'date')]:
        frame[date_field] = pd.to_datetime(frame[date_field], errors='raise').dt.normalize()
        require(frame[date_field].notna().all() and frame[date_field].isin(daily.date).all(), 'ledger date outside evaluated sessions')
    plans = plans.rename(columns={'execute_date': 'date'})
    keys = ['strategy', 'date', 'stock_code', 'side']
    # The current engine emits at most one attempt per stock/side/session. If a
    # later engine emits child orders, it must supply attempt IDs to disambiguate.
    require(not plans.duplicated(keys).any(), 'ambiguous plan records need unique attempt IDs')
    require(not trades.duplicated(keys).any(), 'ambiguous fill records need unique attempt IDs')
    require(plans.side.isin(['buy', 'sell']).all() and trades.side.isin(['buy', 'sell']).all(), 'unknown trade side')
    for frame, values in [(plans, ['planned_units', 'previous_close']),
                          (trades, ['planned_units', 'filled_units', 'notional', 'cost']),
                          (daily, ['nav', 'cash', 'cash_weight', 'hold_count'])]:
        require(np.isfinite(frame[values].to_numpy(dtype=float)).all(), f'non-finite {values}')
    require((plans.planned_units > 0).all() and (trades.filled_units > 0).all(), 'non-positive submitted order or fill')
    require((trades.notional > 0).all() and (trades.cost >= 0).all(), 'negative amount or cost')
    require((daily.nav > 0).all() and (daily.cash >= -money_tolerance).all(), 'non-positive wealth or negative cash')
    close(daily.cash_weight, daily.cash / daily.nav, 'cash weight disagrees with ledger cash')
    for frame in [plans, trades]:
        frame['_t'] = dates.get_indexer(frame.date)
        frame['_j'] = codes.get_indexer(frame.stock_code)
        require((frame['_t'] > 0).all() and (frame['_j'] >= 0).all(), 'unknown stock or execution date')
        frame['plan_date'] = pd.to_datetime(frame.plan_date, errors='raise').dt.normalize()
        frame['funding_asof'] = pd.to_datetime(frame.funding_asof, errors='raise').dt.normalize()
        require(frame.plan_date.notna().all() and frame.funding_asof.notna().all(), 'missing information-as-of date')
        require(np.array_equal(frame.plan_date.to_numpy(), dates.take(frame['_t'] - 1).to_numpy()),
                'order was not planned at the preceding market close')
        require((frame.funding_asof <= frame.plan_date).all(), 'current-session or future proceeds fund a trade')

    plans['_plan_number'] = np.arange(len(plans))
    paired = trades.merge(plans[keys + ['planned_units', 'previous_close', '_plan_number']],
                          on=keys, how='left', validate='one_to_one', suffixes=('', '_intent'))
    require(paired['_plan_number'].notna().all(), 'fill has no submitted order')
    quantity_tolerance = 1e-9 * np.maximum(1., paired.planned_units_intent.to_numpy(dtype=float))
    require((paired.filled_units.to_numpy(dtype=float) <= paired.planned_units_intent + quantity_tolerance).all(),
            'fill exceeds its pre-session planned quantity')
    close(paired.planned_units, paired.planned_units_intent, 'trade planned quantity differs from order plan')
    if 'previous_close' in trades:
        close(paired.previous_close, paired.previous_close_intent, 'trade previous mark differs from order plan')
    trades['_factor'] = market['factor'][trades['_t'].to_numpy(int), trades['_j'].to_numpy(int)]
    require(np.isfinite(trades['_factor']).all() and (trades['_factor'] > 0).all(), 'invalid factor on a fill')
    raw_shares = trades.filled_units.to_numpy(dtype=float) * trades['_factor'].to_numpy()
    buy_mask = trades.side.eq('buy').to_numpy()
    if lot:
        close(raw_shares[buy_mask] / lot, np.round(raw_shares[buy_mask] / lot),
              'buy fill is not an integer lot', atol=1e-7, rtol=0)
    trades['_base_fee'] = fee(trades.notional.to_numpy(dtype=float))
    trades['_impact'] = 0.
    expected_impact = np.zeros(len(trades))
    totals_by_stock_day = []
    if constrained:
        for key in ['amount_yuan', 'volume_shares', 'adv20', 'sigma20', 'low', 'high']:
            require('capacity_' + key in market, f'missing capacity matrix {key}')
        require({'impact_cost', 'base_cost', 'reference_price'}.issubset(trades) or trades.empty,
                'constrained trades omit cost/reference-price components')
        if not trades.empty:
            trades['_impact'] = trades.impact_cost
            close(trades.base_cost, trades['_base_fee'], 'base fee omits minimum commission or double-counts it')
        for (date, stock), group in trades.groupby(['date', 'stock_code'], sort=False):
            t, j = int(group['_t'].iloc[0]), int(group['_j'].iloc[0])
            amount, volume = float(market['capacity_amount_yuan'][t, j]), float(market['capacity_volume_shares'][t, j])
            adv, sigma = float(market['capacity_adv20'][t, j]), float(market['capacity_sigma20'][t, j])
            require(np.isfinite([amount, volume]).all() and amount > 0 and volume > 0, 'fill uses unknown daily amount or volume')
            raw_vwap = amount / volume
            low, high = market['capacity_low'][t, j], market['capacity_high'][t, j]
            price_tolerance = max(.011, abs(raw_vwap) * 1e-5)
            require(np.isfinite([low, high]).all() and low > 0 and high >= low and
                    low - price_tolerance <= raw_vwap <= high + price_tolerance, 'VWAP reference is outside observed price range')
            require(not market.get('suspended_any', market['suspended'])[t, j], 'fill on an intraday-suspended day')
            if participation is not None or eta > 0:
                require(np.isfinite(adv) and adv > 0, 'fill uses unknown or zero lagged turnover')
            if eta > 0 or rule.get('require_known_sigma', False):
                require(np.isfinite(sigma) and sigma >= 0, 'fill uses unknown lagged volatility')
            total = float(group.notional.sum())
            cap = np.inf if participation is None else participation * min(adv, amount)
            require(total <= cap + max(money_tolerance, total * 1e-11), 'buy and sell fills exceed their shared daily capacity')
            close(group.reference_price, raw_vwap * group['_factor'], 'fill reference differs from daily VWAP')
            close(group.notional, group.filled_units * raw_vwap * group['_factor'], 'fill notional differs from quantity times VWAP')
            cumulative = group.notional.cumsum().to_numpy()
            prior = np.r_[0., cumulative[:-1]]
            impacts = np.zeros(len(group)) if eta == 0 else eta * sigma / np.sqrt(adv) * (cumulative ** 1.5 - prior ** 1.5)
            expected_impact[group.index.to_numpy()] = impacts
            totals_by_stock_day.append((date, stock, total, cap))
        close(trades['_impact'], expected_impact, 'marginal impact cost differs from shared cumulative notional')
    elif not trades.empty:
        expected_price = market['open'][trades['_t'].to_numpy(int), trades['_j'].to_numpy(int)] * trades['_factor']
        close(trades.notional, trades.filled_units * expected_price, 'opening control notional differs from opening price')
    close(trades.cost, trades['_base_fee'] + expected_impact, 'total trade fee must include base and incremental impact exactly once')

    if constrained:
        orders = table(details, 'liquidity_orders', [*keys, 'planned_units', 'filled_units', 'previous_close',
                         'requested_reference_notional', 'filled_reference_notional', 'notional', 'base_fee', 'impact_cost'])
        orders['date'] = pd.to_datetime(orders.date, errors='raise').dt.normalize()
        require(not orders.duplicated(keys).any(), 'ambiguous audited attempts')
        all_attempts = orders.merge(plans[keys + ['planned_units']], on=keys, how='outer', indicator=True,
                                    validate='one_to_one', suffixes=('', '_intent'))
        require(all_attempts['_merge'].eq('both').all(), 'audited attempts do not cover every submitted order')
        close(all_attempts.planned_units, all_attempts.planned_units_intent, 'audited planned quantity differs')
        finite_columns = ['planned_units', 'filled_units', 'previous_close', 'requested_reference_notional',
                          'filled_reference_notional', 'notional', 'base_fee', 'impact_cost']
        require(np.isfinite(orders[finite_columns].to_numpy(dtype=float)).all(), 'non-finite order audit amounts')
        require((orders.filled_units >= 0).all() and
                (orders.filled_units <= orders.planned_units + 1e-9 * np.maximum(1, orders.planned_units)).all(),
                'audited fill exceeds its planned quantity')
        close(orders.requested_reference_notional, orders.planned_units * orders.previous_close, 'attempt denominator changes reference price')
        close(orders.filled_reference_notional, orders.filled_units * orders.previous_close, 'attempt numerator changes reference price')
        filled = orders[orders.filled_units > 0]
        reconciliation = filled.merge(trades[keys + ['filled_units', 'notional', 'cost', '_base_fee', '_impact']],
                                      on=keys, how='outer', indicator=True, validate='one_to_one', suffixes=('_audit', '_trade'))
        require(reconciliation['_merge'].eq('both').all(), 'positive attempt has no fill or fill has no audited attempt')
        for field in ['filled_units', 'notional']:
            close(reconciliation[field + '_audit'], reconciliation[field + '_trade'], f'audit and trades disagree on {field}')
        close(reconciliation.base_fee, reconciliation['_base_fee'], 'audit and trades disagree on base fees')
        close(reconciliation.impact_cost, reconciliation['_impact'], 'audit and trades disagree on impact')
        zero = orders[orders.filled_units == 0]
        close(zero[['notional', 'base_fee', 'impact_cost']].to_numpy(), 0., 'zero fill charged fees or turnover')

    # Cash and adjusted positions are reconstructed from executed transactions,
    # independently of the engine's stored closing cash and NAV.
    units = np.zeros(len(codes))
    marks = np.full(len(codes), np.nan)
    close_prices = market['close'] * market['factor']
    for t in range(ix_daily[0]):
        valid = np.isfinite(close_prices[t]) & (close_prices[t] > 0)
        marks[valid] = close_prices[t, valid]
    require(not events.event.eq('unverified_stale_mark').any(), 'unverified valuation in event ledger')
    terminal = events[events.event.eq('terminal')]
    if len(terminal):
        require({'stock_code', 'marked_value', 'residual'}.issubset(terminal), 'terminal event lacks value or recovery')
        require(not terminal.duplicated(['date', 'stock_code']).any(), 'duplicate terminal event')
    by_trades = {day: group for day, group in trades.groupby('date', sort=False)}
    by_plans = {day: group for day, group in plans.groupby('date', sort=False)}
    by_terminal = {day: group for day, group in terminal.groupby('date', sort=False)}
    cash = float(initial_capital)
    for position, saved in daily.iterrows():
        t, day = ix_daily[position], saved.date
        previous_cash = cash
        tplans = by_plans.get(day, plans.iloc[:0])
        day_trades = by_trades.get(day, trades.iloc[:0])
        if len(tplans):
            close(tplans.previous_close, marks[tplans['_j'].to_numpy(int)], 'plan uses a mark not yet observed')
        buy_plans = tplans[tplans.side.eq('buy')]
        if len(buy_plans):
            require({'funding_budget', 'limit_price'}.issubset(buy_plans), 'buy order omits pre-session funding or limit')
            require(np.isfinite(buy_plans[['funding_budget', 'limit_price']]).all().all(), 'unknown pre-session budget or price limit')
            require(buy_plans.funding_budget.sum() <= previous_cash + money_tolerance, 'buy budgets borrow current-session sale proceeds')
            planned_prior = {}
            for sell in tplans[tplans.side.eq('sell')].itertuples(index=False):
                j = codes.get_loc(sell.stock_code)
                planned_prior[j] = sell.planned_units * market['up'][t, j] * market['factor'][t, j]
            for order in buy_plans.itertuples(index=False):
                j = codes.get_loc(order.stock_code)
                expected_limit = market['up'][t, j] * market['factor'][t, j]
                close(order.limit_price, expected_limit, 'buy limit is not the preannounced upper price')
                qvalue = order.planned_units * expected_limit
                impact_reserve = 0.
                if constrained and eta:
                    adv, sigma = market['capacity_adv20'][t, j], market['capacity_sigma20'][t, j]
                    if np.isfinite([adv, sigma]).all() and adv > 0 and sigma >= 0:
                        before = planned_prior.get(j, 0.)
                        impact_reserve = eta * sigma / np.sqrt(adv) * ((before + qvalue) ** 1.5 - before ** 1.5)
                require(qvalue + fee(qvalue) + impact_reserve <= order.funding_budget + max(money_tolerance, order.funding_budget * 1e-11),
                        'pre-session buy quantity lacks worst-price fee/impact funding')
                planned_prior[j] = planned_prior.get(j, 0.) + qvalue
        if len(day_trades):
            close(day_trades.cash_before_open, previous_cash, 'cash-before-session differs from prior closing cash')
            buys = day_trades[day_trades.side.eq('buy')]
            require((buys.notional + buys.cost).sum() <= previous_cash + money_tolerance,
                    'executed buys borrow current-session sale proceeds')
            for fill in buys.itertuples(index=False):
                match = buy_plans[buy_plans.stock_code.eq(fill.stock_code)].iloc[0]
                require(fill.notional + fill.cost <= match.funding_budget + money_tolerance, 'fill spends more than its reserved budget')
                require(fill.notional / fill.filled_units <= match.limit_price + 1e-8, 'buy fill exceeds planned price limit')
        for event in by_terminal.get(day, terminal.iloc[:0]).itertuples(index=False):
            j = codes.get_loc(event.stock_code)
            close(event.marked_value, units[j] * marks[j], 'terminal writeoff differs from held marked value')
            close(event.residual, event.marked_value * terminal_fraction, 'terminal cash differs from declared recovery fraction')
            cash += event.residual
            units[j] = 0.
        # The recorded engine fills sells before buys; preserve that order.
        for fill in day_trades.itertuples(index=False):
            j = codes.get_loc(fill.stock_code)
            require(np.isfinite(market['open'][t, j]) and market['open'][t, j] > 0 and not market['suspended'][t, j],
                    'trade filled on an unusable opening observation')
            if fill.side == 'sell':
                require(market['open'][t, j] > market['down'][t, j] + 1e-7, 'sell filled at blocked opening lower limit')
                require(fill.filled_units <= units[j] + max(1e-8, units[j] * 1e-10), 'sell exceeds actual adjusted holdings')
                if lot and fill.filled_units < units[j] - 1e-8:
                    raw = fill.filled_units * market['factor'][t, j] / lot
                    close(raw, round(raw), 'partial sell is not an integer lot', atol=1e-7, rtol=0)
                units[j] -= fill.filled_units
                if abs(units[j]) < 1e-8: units[j] = 0.
                cash += fill.notional - fill.cost
            else:
                require(market['open'][t, j] < market['up'][t, j] - 1e-7, 'buy filled at blocked opening upper limit')
                units[j] += fill.filled_units
                cash -= fill.notional + fill.cost
        valid = np.isfinite(close_prices[t]) & (close_prices[t] > 0)
        marks[valid] = close_prices[t, valid]
        require(np.isfinite(marks[units > 0]).all(), 'held position has no known closing mark')
        close(saved.cash, cash, 'daily cash does not reconcile to fills and terminal receipts', atol=money_tolerance)
        close(saved.nav, cash + np.nansum(units * marks), 'daily NAV does not reconcile to cash and positions', atol=money_tolerance)
        require(saved.hold_count == int((units > 0).sum()), 'daily holding count disagrees with reconstructed positions')
        if constrained:
            close(saved.impact_cost, day_trades['_impact'].sum(), 'daily impact differs from trade ledger')

    monthly = monthly.copy()
    require({'return_month', 'nav', 'net_return', 'cost', 'turnover'}.issubset(monthly), 'monthly costs or returns absent')
    require(monthly.return_month.is_unique, 'duplicate monthly records')
    daily['_month'] = daily.date.dt.to_period('M').astype(str)
    trades['_month'] = trades.date.dt.to_period('M').astype(str)
    endings = daily.groupby('_month').tail(1).set_index('_month')
    require(set(monthly.return_month) == set(endings.index), 'monthly and daily coverage differ')
    preceding_nav = initial_capital
    daily_prior_nav = pd.Series(np.r_[initial_capital, daily.nav.to_numpy()[:-1]], index=daily.date)
    for row in monthly.sort_values('return_month').itertuples(index=False):
        group = trades[trades['_month'].eq(row.return_month)]
        expected_nav = endings.loc[row.return_month, 'nav']
        close(row.nav, expected_nav, 'monthly NAV differs from closing ledger')
        close(row.net_return, expected_nav / preceding_nav - 1, 'monthly return differs from closing ledger')
        close(row.cost, group.cost.sum() / preceding_nav, 'monthly cost excludes or repeats trade fees')
        turnover = (group.notional.to_numpy() / daily_prior_nav.reindex(group.date).to_numpy()).sum()
        close(row.turnover, turnover, 'monthly turnover differs from previous-close NAV convention')
        if constrained:
            close(row.impact_cost_yuan, group['_impact'].sum(), 'monthly impact cash differs from fills')
            close(row.impact_cost, group['_impact'].sum() / preceding_nav, 'monthly impact ratio differs from fills')
        preceding_nav = expected_nav
    return {'checked_sessions': len(daily), 'checked_submitted_orders': len(plans),
            'checked_fills': len(trades), 'checked_stock_day_capacity_budgets': len(totals_by_stock_day),
            'terminal_events': len(terminal), 'base_fee_yuan': float(trades['_base_fee'].sum()),
            'incremental_impact_yuan': float(trades['_impact'].sum()),
            'cash_positions_fees_reconciled': True,
            'scope': 'Declared daily-model ledger consistency, not real intraday execution certification'}
