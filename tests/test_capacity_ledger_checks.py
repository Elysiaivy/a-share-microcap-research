"""Adversarial accounting tests on a small manually specified trade ledger."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from scripts.capacity_ledger_checks import verify_capacity_ledgers


def ledger_fixture(*, roundtrip=False, sell_units=100., odd_exit=False, terminal=False, no_orders=False):
    dates = pd.to_datetime(['2014-12-31', '2015-01-05', '2015-01-06', '2015-01-30'])
    price = np.tile([10., 20.], (4, 1))
    factor = np.ones_like(price)
    if odd_exit:
        factor[2:, 0] = .95
        price[2:, 0] /= .95
    market = {
        'dates': dates.to_numpy(), 'codes': np.array(['A', 'B']), 'factor': factor,
        'open': price.copy(), 'close': price.copy(), 'up': price * 1.1, 'down': price * .9,
        'suspended': np.zeros_like(price, dtype=bool),
        'capacity_amount_yuan': np.full_like(price, 100000.),
        'capacity_volume_shares': 100000. / price,
        'capacity_adv20': np.full_like(price, 100000.),
        'capacity_sigma20': np.full_like(price, .02),
        'capacity_high': price * 1.1, 'capacity_low': price * .9,
    }
    protocol = {'winner': 'S', 'base_cost_bps': 30, 'commission_bps': 2.5,
                'minimum_commission_yuan': 5, 'buy_lot_shares': 100,
                'scenarios': {'MAIN': {'execution': 'daily_vwap_proxy', 'participation': .03, 'impact_eta': .5}}}
    # A buy on the first day; liquidate A and buy B (or round-trip A) on day two.
    instructions = [] if no_orders else [
        (1, 'A', 'buy', 100., 1200.), (2, 'A', 'sell', sell_units, None),
        (2, 'A' if roundtrip else 'B', 'buy', 100., 1200. if roundtrip else 2300.),
    ]
    plans, trades, orders, daily, events = [], [], [], [], []
    cash = 10000.
    positions = {'A': 0., 'B': 0.}
    previous_nav = 10000.
    turnover, total_cost, total_impact = 0., 0., 0.
    for t in range(1, len(dates)):
        previous_cash = cash
        used = {'A': 0., 'B': 0.}
        day_impact = 0.
        for row_t, stock, side, quantity, budget in instructions:
            if row_t != t:
                continue
            j = 0 if stock == 'A' else 1
            mark = 10. if stock == 'A' else 20.
            notional = quantity * mark
            base_fee = .00275 * notional + max(.00025 * notional, 5.)
            before = used[stock]
            # This example has sigma=.02, eta=.5 and ADV=100000 in every cell.
            impact = .01 / np.sqrt(100000.) * ((before + notional) ** 1.5 - before ** 1.5)
            total_fee = base_fee + impact
            date, plan_date = str(dates[t].date()), str(dates[t - 1].date())
            plan = dict(strategy='S', execute_date=date, stock_code=stock, side=side, reason='rebalance',
                        planned_units=quantity, previous_close=mark, plan_date=plan_date, funding_asof=plan_date)
            if side == 'buy':
                plan.update(funding_budget=budget, limit_price=mark * 1.1)
            plans.append(plan)
            trade = dict(strategy='S', date=date, stock_code=stock, side=side, reason='rebalance',
                         planned_units=quantity, filled_units=quantity, notional=notional, cost=total_fee,
                         previous_close=mark, plan_date=plan_date, funding_asof=plan_date,
                         cash_before_open=previous_cash, base_cost=base_fee, impact_cost=impact,
                         reference_price=mark)
            if side == 'buy':
                trade.update(funding_budget=budget, limit_price=mark * 1.1)
            trades.append(trade)
            orders.append(dict(strategy='S', date=date, stock_code=stock, side=side, reason='rebalance',
                               planned_units=quantity, filled_units=quantity, previous_close=mark,
                               requested_reference_notional=notional, filled_reference_notional=notional,
                               notional=notional, base_fee=base_fee, impact_cost=impact,
                               capacity_total=3000., capacity_before=3000.-before,
                               capacity_after=3000.-before-notional))
            used[stock] += notional
            if side == 'buy':
                cash -= notional + total_fee
                positions[stock] += quantity
            else:
                cash += notional - total_fee
                positions[stock] -= quantity
            turnover += notional / previous_nav
            total_cost += total_fee
            day_impact += impact
        total_impact += day_impact
        if t == 3 and terminal:
            value = positions['B'] * 20.
            events.append(dict(strategy='S', date=str(dates[t].date()), event='terminal',
                               stock_code='B', marked_value=value, residual=value * .5))
            cash += value * .5
            positions['B'] = 0.
        nav = cash + positions['A'] * 10. + positions['B'] * 20.
        daily.append(dict(strategy='S', date=str(dates[t].date()), nav=nav, cash=cash,
                          cash_weight=cash/nav, hold_count=sum(v > 0 for v in positions.values()),
                          impact_cost=day_impact))
        previous_nav = nav
    monthly = pd.DataFrame([dict(strategy='S', return_month='2015-01', nav=previous_nav,
                                 net_return=previous_nav / 10000. - 1,
                                 cost=total_cost / 10000., turnover=turnover,
                                 impact_cost_yuan=total_impact, impact_cost=total_impact / 10000.)])
    details = dict(order_plans=pd.DataFrame(plans), trades=pd.DataFrame(trades),
                   liquidity_orders=pd.DataFrame(orders), daily_nav=pd.DataFrame(daily), events=pd.DataFrame(events))
    return monthly, details, market, protocol


class CapacityLedgerTests(unittest.TestCase):
    def check(self, fixture, **kwargs):
        monthly, details, market, protocol = fixture
        return verify_capacity_ledgers(monthly, details, market, protocol, 'MAIN', 10000., **kwargs)

    def test_manually_balanced_cash_and_positions_pass(self):
        fixture = ledger_fixture()
        # Known cash computation: buy A; sell A; buy B, including three minimum fees.
        cash = 10000 - 1008.75 + 991.25 - (2000 + 10.5 + 2.8284271247461903)
        self.assertAlmostEqual(fixture[1]['daily_nav'].iloc[-1].cash, cash)
        result = self.check(fixture)
        self.assertEqual(result['checked_fills'], 3)
        self.assertTrue(result['cash_positions_fees_reconciled'])

    def test_shared_stock_day_capacity_cannot_reset_for_sell_and_buy(self):
        fixture = ledger_fixture(roundtrip=True)
        self.check(fixture)  # Total 2000 is within its original shared 3000 budget.
        fixture[3]['scenarios']['MAIN']['participation'] = .015
        with self.assertRaisesRegex(ValueError, 'shared daily capacity'):
            self.check(fixture)  # Each side is only 1000, but together they exceed 1500.

    def test_fill_cannot_exceed_submitted_quantity(self):
        fixture = ledger_fixture()
        fixture[1]['trades'].loc[0, 'filled_units'] = 200.
        with self.assertRaisesRegex(ValueError, 'pre-session planned quantity'):
            self.check(fixture)

    def test_minimum_commission_is_required_even_for_small_notional(self):
        fixture = ledger_fixture()
        fixture[1]['trades'].loc[0, 'base_cost'] = 3.
        with self.assertRaisesRegex(ValueError, 'base fee omits minimum'):
            self.check(fixture)

    def test_impact_cannot_be_charged_twice(self):
        fixture = ledger_fixture()
        frame = fixture[1]['trades']
        frame['cost'] += frame.impact_cost
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            self.check(fixture)

    def test_splitting_same_stock_notional_cannot_reduce_impact(self):
        fixture = ledger_fixture(roundtrip=True)
        # Incorrectly charge the second 1000 as a fresh trade instead of a marginal one.
        frame = fixture[1]['trades']
        frame.loc[2, 'impact_cost'] = 1.
        frame.loc[2, 'cost'] = frame.loc[2, 'base_cost'] + 1.
        with self.assertRaisesRegex(ValueError, 'shared cumulative notional'):
            self.check(fixture)

    def test_buy_budget_cannot_include_today_sale_proceeds(self):
        fixture = ledger_fixture()
        plans = fixture[1]['order_plans']
        plans.loc[2, 'funding_budget'] = 9000.
        with self.assertRaisesRegex(ValueError, 'budgets borrow current-session'):
            self.check(fixture)

    def test_cheap_actual_fill_does_not_excuse_underfunded_upper_price_plan(self):
        fixture = ledger_fixture()
        fixture[1]['order_plans'].loc[0, 'funding_budget'] = 1050.
        with self.assertRaisesRegex(ValueError, 'worst-price fee/impact funding'):
            self.check(fixture)

    def test_future_funding_timestamp_is_rejected(self):
        fixture = ledger_fixture()
        fixture[1]['trades'].loc[2, 'funding_asof'] = '2015-01-06'
        with self.assertRaisesRegex(ValueError, 'current-session or future proceeds'):
            self.check(fixture)

    def test_partial_sell_requires_integer_lots(self):
        fixture = ledger_fixture(sell_units=50.)
        with self.assertRaisesRegex(ValueError, 'partial sell is not an integer lot'):
            self.check(fixture)

    def test_complete_odd_lot_exit_is_allowed(self):
        # The earlier 100 adjusted units correspond to 95 raw shares after a factor change.
        self.check(ledger_fixture(odd_exit=True))

    def test_fictitious_closing_profit_is_detected(self):
        fixture = ledger_fixture()
        frame = fixture[1]['daily_nav']
        frame.loc[2, 'nav'] += 100.
        frame.loc[2, 'cash_weight'] = frame.loc[2, 'cash'] / frame.loc[2, 'nav']
        with self.assertRaisesRegex(ValueError, 'NAV does not reconcile'):
            self.check(fixture)

    def test_positive_audit_attempt_cannot_disappear_from_trade_ledger(self):
        fixture = ledger_fixture()
        fixture[1]['trades'] = fixture[1]['trades'].iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, 'positive attempt has no fill'):
            self.check(fixture)

    def test_terminal_receipt_is_separate_from_trade_cash(self):
        fixture = ledger_fixture(terminal=True)
        result = self.check(fixture, terminal_fraction=.5)
        self.assertEqual(result['terminal_events'], 1)
        fixture[1]['events'].loc[0, 'residual'] = 999.
        with self.assertRaisesRegex(ValueError, 'declared recovery fraction'):
            self.check(fixture, terminal_fraction=.5)

    def test_unfilled_cash_portfolio_remains_a_valid_ledger(self):
        result = self.check(ledger_fixture(no_orders=True))
        self.assertEqual(result['checked_fills'], 0)


if __name__ == '__main__':
    unittest.main()
