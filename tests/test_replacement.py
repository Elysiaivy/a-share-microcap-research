import unittest
import numpy as np
import pandas as pd
import test_execution as fixtures
from microcap.execution import run
from microcap.selection import eligible_ranks


class ReplacementTests(unittest.TestCase):
 def fixture(self):
  m,p,c=fixtures.DailyExecution().fixture()
  for k in ['open','close','factor','up','down','state','suspended']:
   m[k]=np.repeat(m[k],3,axis=1)
  m['codes']=np.array(['A','B','C']);m['delist']=np.repeat(m['delist'],3)
  p=pd.concat([p,p.assign(stock_code='B',rank=2,bucket='C'),p.assign(stock_code='C',rank=3,bucket='D')],ignore_index=True)
  p.attrs['selection_universe']=p.copy()
  c.update(intramonth_replacement=True,replacement_delay_sessions=1)
  return m,p,c

 def test_sale_proceeds_only_fund_following_session(self):
  m,p,c=self.fixture();m['state'][2:,0]=1;m['open'][2,0]=9
  _,d=run(m,p,c,cost_bps=30,details=True)
  sell=d['trades'].query("reason=='ST'").iloc[0]
  buy=d['trades'].query("reason=='ST_replacement'").iloc[0]
  self.assertEqual(buy.stock_code,'B');self.assertGreater(buy.date,sell.date)
  self.assertEqual(buy.date,'2015-01-30');self.assertEqual(buy.replaces,'A')
  self.assertEqual(buy.funding_asof,sell.date)
  self.assertEqual(buy.plan_date,sell.date)
  self.assertEqual(buy.delay_sessions,1)
  self.assertAlmostEqual(buy.reserved_budget,sell.notional-sell.cost)
  self.assertLessEqual(buy.notional+buy.cost,buy.reserved_budget+1e-12)
  self.assertAlmostEqual(buy.planned_units,buy.filled_units)

 def test_known_risk_skipped_before_order(self):
  m,p,c=self.fixture();m['state'][2:,0]=1;m['state'][2:,1]=1
  _,d=run(m,p,c,details=True)
  self.assertEqual(d['trades'].query("reason=='ST_replacement'").iloc[0].stock_code,'C')

 def test_limit_failure_cannot_switch_name_at_same_open(self):
  m,p,c=self.fixture();m['state'][2:,0]=1;m['open'][3,1]=11
  _,d=run(m,p,c,details=True)
  blocked=d['blocked_orders'].query("side=='buy' and order_reason=='ST_replacement'")
  self.assertEqual(blocked.iloc[0].stock_code,'B')
  self.assertEqual(blocked.iloc[0].date,'2015-01-07')
  buy=d['trades'].query("reason=='ST_replacement'").iloc[0]
  self.assertEqual(buy.stock_code,'C');self.assertEqual(buy.date,'2015-01-30')
  self.assertFalse(((d['trades'].date=='2015-01-07')&(d['trades'].side=='buy')).any())

 def test_failed_fill_retries_later_session(self):
  m,p,c=self.fixture();m['state'][2:,0]=1;m['suspended'][3,1]=True
  _,d=run(m,p,c,details=True)
  buy=d['trades'].query("reason=='ST_replacement'").iloc[0]
  self.assertEqual(buy.date,'2015-01-30');self.assertEqual(buy.delay_sessions,2)

 def test_same_open_replacement_configuration_rejected(self):
  m,p,c=self.fixture();c['replacement_delay_sessions']=0
  with self.assertRaisesRegex(ValueError,'at least one session'):
   run(m,p,c)

 def test_risk_exit_disabled_is_monthly_filter_only(self):
  m,p,c=self.fixture();m['state'][2:,0]=1;c['risk_exit_enabled']=False
  _,d=run(m,p,c,details=True)
  self.assertFalse(d['trades'].reason.eq('ST').any())
  self.assertFalse(d['trades'].reason.eq('ST_replacement').any())
  sells=d['trades'].query("side=='sell'")
  self.assertEqual(sells.iloc[0].date,'2015-02-02')

 def test_reranking_future_independent(self):
  p=pd.DataFrame({'stock_code':['A','B','C'],'signal_date':pd.Timestamp('2014-12-31'),'return_month':'2015-01','rank':[1,2,3]})
  m={'codes':np.array(['A','B','C']),'dates':np.array(['2014-12-31','2015-01-05'],dtype='datetime64[D]'),'state':np.array([[1,0,0],[1,0,1]]),'known_ST':np.array([[True,False,False],[True,False,True]])}
  a=eligible_ranks(p,m)
  self.assertEqual(a.stock_code.tolist(),['B','C']);self.assertEqual(a['rank'].tolist(),[1,2])
  self.assertEqual(a.original_rank.tolist(),[2,3])
  m['state'][1,:]=1;m['known_ST'][1,:]=True
  pd.testing.assert_frame_equal(a,eligible_ranks(p,m))

 def test_exit_cash_cannot_fund_blocked_monthly_target_until_next_month(self):
  m,p,c=self.fixture();c['buckets']={'B':[1,2]}
  # B is a monthly target, but two consecutive limit-up opens prevent entry.
  # Its third open is feasible and now costs more than the original cash slot.
  m['open'][1,1]=11.;m['up'][1,1]=11.;m['close'][1,1]=11.
  m['open'][2,1]=12.1;m['up'][2,1]=12.1;m['close'][2,1]=12.1
  m['open'][3,1]=13.2;m['up'][3,1]=13.31;m['close'][3,1]=13.2
  m['state'][2:,0]=1
  cash_rule={**c,'intramonth_replacement':False}
  _,cash_details=run(m,p,cash_rule,cost_bps=0,details=True)
  sale=cash_details['trades'].query("reason=='ST'").iloc[0]
  bought=cash_details['trades'].query("side=='buy' and date=='2015-01-07'")
  prior=cash_details['daily_nav'].query("date=='2015-01-06'").iloc[0]
  self.assertAlmostEqual(prior.idle_risk_cash,sale.notional-sale.cost)
  self.assertLessEqual((bought.notional+bought.cost).sum(),prior.cash-prior.idle_risk_cash+1e-12)
  # Without a separate idle reserve, B would receive the full 0.05-share target
  # at 13.2 (0.66 notional), implicitly spending part of A's risk-sale receipt.
  self.assertLess(bought.filled_units.sum(),.05)
  february=cash_details['daily_nav'].query("date=='2015-02-02'").iloc[0]
  self.assertEqual(february.idle_risk_cash,0.)
  # The baseline instead retains the dedicated risk budget and buys reserve C.
  _,replace_details=run(m,p,c,cost_bps=0,details=True)
  replacement=replace_details['trades'].query("reason=='ST_replacement'").iloc[0]
  self.assertEqual(replacement.stock_code,'C')
  self.assertEqual(replacement.date,'2015-01-07')
  self.assertAlmostEqual(replacement.reserved_budget,sale.notional-sale.cost)
  self.assertTrue(replace_details['daily_nav'].idle_risk_cash.eq(0).all())
