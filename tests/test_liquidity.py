import unittest
import numpy as np
import pandas as pd
import test_execution
from microcap.execution import run
from microcap.execution_constraints import OrderConstraints
from microcap.liquidity import LiquidityConstraints


def add_liquidity(m, amount=1_000_000.):
    shape = m['open'].shape
    m['capacity_amount_yuan'] = np.full(shape, amount)
    m['capacity_volume_shares'] = m['capacity_amount_yuan'] / m['close']
    m['capacity_high'] = m['up'].copy()
    m['capacity_low'] = m['down'].copy()
    m['capacity_adv20'] = np.full(shape, amount)
    m['capacity_sigma20'] = np.full(shape, .02)
    return m


class LiquidityTests(unittest.TestCase):
    def fixture(self):
        m, p, c = test_execution.DailyExecution().fixture()
        return add_liquidity(m), p, c

    def test_unlimited_zero_impact_at_open_equal_vwap_reproduces_baseline(self):
        m, p, c = self.fixture(); rules = OrderConstraints(10000, 100, 5)
        a, da = run(m, p, c, implementation=rules, details=True)
        b, db = run(m, p, c, implementation=rules, liquidity=LiquidityConstraints(None, 0), details=True)
        np.testing.assert_allclose(a.net_return, b.net_return, atol=1e-12, rtol=0)
        np.testing.assert_allclose(da['daily_nav'].nav, db['daily_nav'].nav, atol=1e-10, rtol=0)
        self.assertTrue(db['trades'].impact_cost.eq(0).all())

    def test_impact_is_marginal_and_splitting_does_not_reduce_it(self):
        q = LiquidityConstraints(.01, .5)
        full = q.charge(10000, 1000000, .02)
        self.assertAlmostEqual(full, 10.)
        self.assertAlmostEqual(q.charge(4000, 1000000, .02) + q.charge(6000, 1000000, .02, 4000), full)
        self.assertGreater(q.charge(20000, 1000000, .02), full)
        self.assertEqual(LiquidityConstraints(.01, 0).charge(10000, 1000000, .02), 0)

    def test_worst_price_impact_budget_and_lot_search(self):
        q = LiquidityConstraints(.01, 1.)
        imp = OrderConstraints(10000, 100, 5)
        fee = lambda x: imp.fee(x, .003)
        qty = q.affordable(1000, 1., 11., 10000., fee, 100000., .1, imp, already=5000.)
        cost = qty * 11 + fee(qty * 11) + q.charge(qty * 11, 100000, .1, 5000)
        next_cost = (qty+100) * 11 + fee((qty+100) * 11) + q.charge((qty+100) * 11, 100000, .1, 5000)
        self.assertLessEqual(cost, 10000 + 1e-8)
        self.assertGreater(next_cost, 10000)
        self.assertEqual(qty % 100, 0)

    def test_realized_daily_amount_cannot_change_prior_plan(self):
        m, p, c = self.fixture(); rules = OrderConstraints(10000, 100, 5)
        _, a = run(m, p, c, implementation=rules, liquidity=LiquidityConstraints(.01, .5), details=True)
        m['capacity_amount_yuan'][1] = 100000.
        m['capacity_volume_shares'][1] = 10000.
        _, b = run(m, p, c, implementation=rules, liquidity=LiquidityConstraints(.01, .5), details=True)
        aa = a['order_plans'].query("execute_date == '2015-01-05'").reset_index(drop=True)
        bb = b['order_plans'].query("execute_date == '2015-01-05'").reset_index(drop=True)
        pd.testing.assert_frame_equal(aa, bb)
        self.assertGreater(a['trades'].iloc[0].notional, b['trades'].iloc[0].notional)

    def test_daily_capacity_uses_lower_of_lag_and_realized_amount_and_lots(self):
        m, p, c = self.fixture(); m['capacity_amount_yuan'][:] = 100000.; m['capacity_volume_shares'][:] = 10000.
        _, d = run(m, p, c, implementation=OrderConstraints(10000, 100, 5), liquidity=LiquidityConstraints(.01, 0), details=True)
        self.assertEqual(d['trades'].iloc[0].filled_units, 100)
        self.assertTrue((d['trades'].groupby(['date','stock_code']).notional.sum() <= 1000.000001).all())
        self.assertTrue(np.allclose(d['trades'].filled_units / 100, np.round(d['trades'].filled_units / 100)))
        a = d['liquidity_orders']
        self.assertTrue(a.constraint_reason.str.contains('partial').any())
        self.assertTrue((a.filled_reference_notional <= a.requested_reference_notional + 1e-8).all())
        self.assertTrue((d['daily_nav'].cash >= 0).all())

    def test_capacity_is_shared_between_sides_and_orders(self):
        m, _, _ = self.fixture(); day = LiquidityConstraints(.01, .5).session(m, 1)
        first = day.fill_quantity(0, 800, 1., OrderConstraints(10000, 100))
        self.assertEqual(first, 800)
        day.consume(0, first * 10)
        self.assertEqual(day.fill_quantity(0, 800, 1., OrderConstraints(10000, 100)), 200)
        day.consume(0, 2000.)
        self.assertEqual(day.fill_quantity(0, 100, 1., None), 0.)

    def test_zero_or_missing_lag_data_blocks_and_retries_not_zero_filled(self):
        for value in [0., np.nan]:
            m, p, c = self.fixture(); m['capacity_adv20'][1] = value
            _, d = run(m, p, c, implementation=OrderConstraints(10000, 100, 5), liquidity=LiquidityConstraints(.01, 0), details=True)
            a = d['liquidity_orders'].iloc[0]
            self.assertEqual(a.constraint_reason, 'missing_or_zero_lag_liquidity')
            self.assertGreater(a.planned_units, 0); self.assertEqual(a.filled_units, 0)
            self.assertEqual(d['trades'].iloc[0].date, '2015-01-06')

    def test_intraday_suspension_and_open_limit_are_conservative_blocks(self):
        m, p, c = self.fixture(); m['suspended_any'] = m['suspended'].copy(); m['suspended_any'][1] = True
        _, d = run(m, p, c, implementation=OrderConstraints(10000, 100, 5), liquidity=LiquidityConstraints(None, 0), details=True)
        self.assertEqual(d['liquidity_orders'].iloc[0].constraint_reason, 'intraday_suspension')
        m, p, c = self.fixture(); m['open'][1] = m['up'][1]
        _, d = run(m, p, c, implementation=OrderConstraints(10000, 100, 5), liquidity=LiquidityConstraints(None, 0), details=True)
        self.assertEqual(d['liquidity_orders'].iloc[0].constraint_reason, 'open_untradeable')

    def test_vwap_does_not_override_preannounced_buy_limit(self):
        m, p, c = self.fixture(); m['capacity_volume_shares'][1] = m['capacity_amount_yuan'][1] / 11.1
        m['capacity_high'][1] = 12.
        _, d = run(m, p, c, implementation=OrderConstraints(10000, 100, 5), liquidity=LiquidityConstraints(None, 0), details=True)
        self.assertEqual(d['liquidity_orders'].iloc[0].constraint_reason, 'above_planned_limit')
        self.assertEqual(d['liquidity_orders'].iloc[0].filled_units, 0.)

    def test_total_fee_contains_incremental_impact_once(self):
        m, p, c = self.fixture()
        r, d = run(m, p, c, implementation=OrderConstraints(10000, 100, 5), liquidity=LiquidityConstraints(.01, .5), details=True)
        tr = d['trades']
        self.assertGreater(tr.impact_cost.sum(), 0.)
        np.testing.assert_allclose(tr.cost, tr.base_cost + tr.impact_cost)
        self.assertAlmostEqual(r.iloc[-1].nav, 10000 - tr.cost.sum())
        self.assertAlmostEqual(d['daily_nav'].impact_cost.sum(), tr.impact_cost.sum())
        self.assertAlmostEqual(r.impact_cost_yuan.sum(), tr.impact_cost.sum())
        for field in ['target_shortfall_value','target_shortfall_weight','pending_sell_value','replacement_unfilled_budget']:
            self.assertIn(field, d['daily_nav'])

    def replacement_fixture(self):
        dates = pd.to_datetime(['2014-12-31', '2015-01-05', '2015-01-06', '2015-01-07', '2015-01-08', '2015-01-09', '2015-01-12', '2015-01-13', '2015-01-14', '2015-01-15', '2015-01-16', '2015-01-30'])
        price = np.tile([10.,1.,1.], (len(dates),1))
        m = dict(open=price.copy(),close=price.copy(),factor=np.ones_like(price),up=price*1.1,down=price*.9,
                 dates=dates.to_numpy(),codes=np.array(['A','B','C']),state=np.zeros_like(price,dtype=int),
                 suspended=np.zeros_like(price,dtype=bool),delist=np.full(3,np.datetime64('NaT'),dtype='datetime64[D]'))
        m = add_liquidity(m)
        m['capacity_adv20'][2:,0] = 100000.
        m['capacity_amount_yuan'][2:,0] = 100000.;m['capacity_volume_shares'][2:,0] = 10000.
        m['capacity_adv20'][:,1:] = 50000.
        m['capacity_amount_yuan'][:,1:] = 50000.;m['capacity_volume_shares'][:,1:] = 50000.
        m['state'][2:,0] = 1
        p = pd.DataFrame([dict(stock_code=code,rank=i+1,bucket='S',return_month='2015-01',signal_date='2014-12-31') for i,code in enumerate(['A','B','C'])])
        c = dict(train_start='2015-01',test_end='2015-01',buckets={'S':[1,1]},intramonth_replacement=True,replacement_delay_sessions=1)
        return m,p,c

    def test_split_ST_exit_uses_one_locked_replacement_and_receipt_delays(self):
        m,p,c = self.replacement_fixture()
        _,d=run(m,p,c,cost_bps=0,implementation=OrderConstraints(3300,100,0,0),liquidity=LiquidityConstraints(.01,0),details=True)
        tr=d['trades'];sells=tr.query("reason == 'ST'");buys=tr.query("reason == 'ST_replacement'")
        self.assertEqual(len(sells),3)
        self.assertGreater(len(buys),3)
        self.assertEqual(set(buys.stock_code),{'B'})
        self.assertEqual(buys.iloc[0].date,'2015-01-07')
        self.assertTrue((pd.to_datetime(buys.funding_asof)<pd.to_datetime(buys.date)).all())
        for date in buys.date.unique():
            spent=buys[buys.date.le(date)].eval('notional + cost').sum()
            funded=sells[sells.date.lt(date)].eval('notional - cost').sum()
            self.assertLessEqual(spent,funded+1e-8)
        self.assertTrue((d['daily_nav'].hold_count<=2).all())
        self.assertTrue((d['daily_nav'].cash>=0).all())
        self.assertEqual(d['daily_nav'].iloc[-1].replacement_unfilled_budget,0.)
        np.testing.assert_allclose(d['daily_nav'].nav,3300.,atol=1e-8)

    def test_partial_replacement_residual_is_fixed_before_next_open(self):
        m,p,c = self.replacement_fixture()
        # One complete risk exit funds a large fixed replacement order; its
        # candidate is retained over subsequent constrained partial fills.
        m['capacity_adv20'][2:,0]=1000000.
        m['capacity_amount_yuan'][2:,0]=1000000.;m['capacity_volume_shares'][2:,0]=100000.
        _,d=run(m,p,c,cost_bps=0,implementation=OrderConstraints(3300,100,0,0),liquidity=LiquidityConstraints(.01,0),details=True)
        b=d['trades'].query("reason == 'ST_replacement'")
        self.assertGreater(len(b),1)
        self.assertEqual(set(b.stock_code),{'B'})
        self.assertAlmostEqual(b.iloc[1].planned_units,b.iloc[0].planned_units-b.iloc[0].filled_units)
        self.assertAlmostEqual(b.filled_units.sum(),b.iloc[0].planned_units)

    def test_vwap_only_control_does_not_add_liquidity_eligibility_filter(self):
        m,p,c=self.fixture();m['capacity_adv20'][:]=0.;m['capacity_sigma20'][:]=np.nan
        a,da=run(m,p,c,implementation=OrderConstraints(10000,100,5),details=True)
        b,db=run(m,p,c,implementation=OrderConstraints(10000,100,5),liquidity=LiquidityConstraints(None,0),details=True)
        np.testing.assert_allclose(a.net_return,b.net_return,atol=1e-12,rtol=0)
        self.assertFalse(db['trades'].empty)
        m['capacity_adv20'][:]=1000000.
        _,dc=run(m,p,c,implementation=OrderConstraints(10000,100,5),liquidity=LiquidityConstraints(.01,0),details=True)
        self.assertFalse(dc['trades'].empty)
        _,di=run(m,p,c,implementation=OrderConstraints(10000,100,5),liquidity=LiquidityConstraints(.01,.5),details=True)
        self.assertTrue(di['trades'].empty)

    def test_unfilled_ST_age_is_reported_after_last_successful_partial_exit(self):
        m,p,c=self.replacement_fixture();m['capacity_amount_yuan'][3:,0]=0.;m['capacity_volume_shares'][3:,0]=0.
        _,d=run(m,p,c,cost_bps=0,implementation=OrderConstraints(3300,100,0,0),liquidity=LiquidityConstraints(.01,0),details=True)
        self.assertEqual(d['daily_nav'].iloc[-1].pending_ST_count,1)
        self.assertEqual(d['daily_nav'].iloc[-1].pending_ST_age_max_sessions,len(m['dates'])-1-2)
        self.assertGreater(d['daily_nav'].iloc[-1].pending_ST_value,0.)
        self.assertEqual(d['trades'].query("reason=='ST'").delay_sessions.max(),0.)

    def test_zero_impact_can_match_known_sigma_eligibility(self):
        m,p,c=self.fixture();m['capacity_sigma20'][:]=np.nan
        imp=OrderConstraints(10000,100,5)
        _,default=run(m,p,c,implementation=imp,liquidity=LiquidityConstraints(.01,0),details=True)
        _,matched=run(m,p,c,implementation=imp,
                      liquidity=LiquidityConstraints(.01,0,require_known_sigma=True),details=True)
        self.assertFalse(default['trades'].empty)
        self.assertTrue(matched['trades'].empty)
        self.assertTrue(matched['liquidity_orders'].constraint_reason.eq('missing_or_zero_lag_liquidity').all())
        self.assertTrue((matched['liquidity_orders'].planned_units>0).all())
        self.assertTrue(matched['daily_nav'].nav.eq(10000).all())

    def test_risky_replacement_inherits_original_slot_instead_of_fanning_out(self):
        m,p,c=self.replacement_fixture()
        for key,value in list(m.items()):
            if isinstance(value,np.ndarray) and value.ndim==2:
                m[key]=np.column_stack([value,value[:,-1]])
        m['codes']=np.array(['A','B','C','D'])
        m['delist']=np.full(4,np.datetime64('NaT'),dtype='datetime64[D]')
        m['state'][6:,1]=1
        p=pd.concat([p,p.iloc[[-1]].assign(stock_code='D',rank=4)],ignore_index=True)
        _,d=run(m,p,c,cost_bps=0,implementation=OrderConstraints(3300,100,0,0),liquidity=LiquidityConstraints(.01,0),details=True)
        buys=d['trades'].query("reason=='ST_replacement'")
        self.assertEqual(set(buys.stock_code),{'B','C'})
        self.assertEqual(set(buys[buys.date.ge('2015-01-12')].stock_code),{'C'})
        np.testing.assert_allclose(d['daily_nav'].nav,3300.,atol=1e-8)

    def test_invalid_shapes_and_parameters_fail(self):
        m,p,c=self.fixture()
        with self.assertRaises(ValueError):run(m,p,c,liquidity=LiquidityConstraints(0,.5))
        del m['capacity_high']
        with self.assertRaises(ValueError):run(m,p,c,liquidity=LiquidityConstraints(.01,.5))


if __name__ == '__main__': unittest.main()
