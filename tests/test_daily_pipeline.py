"""Timing invariants of the formal daily execution path, not the old workflow."""
import unittest
import numpy as np
import pandas as pd
import test_execution as fixtures
from microcap.execution import run, suspended_at_open


class OpeningTimeline(unittest.TestCase):
 def test_afternoon_suspension_does_not_erase_morning_auction(self):
  for text in ['13:00-15:00','13:00至15:00','10:00-10:30']:
   self.assertFalse(suspended_at_open(text))
  for text in ['',None,'09:30-10:30','09:00-15:00','盘中停牌','09:30-10:00,13:00-15:00']:
   self.assertTrue(suspended_at_open(text))

 def test_confirmed_afternoon_halt_cases_have_valid_opens(self):
  # These are the three frozen audit cases previously blocked as full-day halts.
  cases=[('002208.SZ','2015-03-02','13:00-15:00'),
         ('002169.SZ','2015-03-02','13:00-15:00'),
         ('002110.SZ','2015-04-01','13:00-15:00')]
  for code,date,timing in cases:
   with self.subTest(code=code,date=date):
    m,p,c=fixtures.DailyExecution().fixture();m['state'][2:]=1
    m['suspended'][2,0]=suspended_at_open(timing)
    _,d=run(m,p,c,cost_bps=0,details=True)
    self.assertEqual(d['trades'].query("reason=='ST'").iloc[0].date,'2015-01-06')

 def test_opening_price_does_not_resize_preplanned_shares(self):
  records=[]
  for opening in [8.,9.]:
   m,p,c=fixtures.DailyExecution().fixture();m['open'][1]=opening;m['down'][1]=7.
   _,d=run(m,p,c,cost_bps=0,details=True)
   records.append(d['trades'].query("side=='buy'").iloc[0])
  self.assertAlmostEqual(records[0].filled_units,1/11)
  self.assertAlmostEqual(records[0].filled_units,records[1].filled_units)
  self.assertNotEqual(records[0].notional,records[1].notional)
  self.assertEqual(records[0].plan_date,'2014-12-31')

 def test_upper_limit_budget_funds_gap_without_open_based_resizing(self):
  m,p,c=fixtures.DailyExecution().fixture();m['open'][1]=10.5
  _,d=run(m,p,c,cost_bps=0,details=True)
  first=d['trades'].query("side=='buy'").iloc[0]
  self.assertEqual(first.date,'2015-01-05')
  self.assertAlmostEqual(first.planned_units,1/11)
  self.assertAlmostEqual(first.reserved_budget,1.)
  self.assertLess(first.notional,first.reserved_budget)
  self.assertEqual(first.limit_price,11.)

 def test_gap_down_cannot_fund_another_order_at_same_auction(self):
  records=[]
  for first_open in [8.,10.]:
   m,p,c=fixtures.DailyExecution().fixture()
   for key in ['open','close','factor','up','down','state','suspended']:
    m[key]=np.repeat(m[key],2,axis=1)
   m['codes']=np.array(['A','B']);m['delist']=np.repeat(m['delist'],2)
   m['open'][1,0]=first_open;m['down'][1,0]=7.
   p=pd.concat([p,p.assign(stock_code='B',rank=2)],ignore_index=True)
   c['buckets']={'B':[1,2]}
   _,d=run(m,p,c,cost_bps=0,details=True)
   orders=d['order_plans'].query("execute_date=='2015-01-05' and side=='buy'")
   self.assertAlmostEqual(orders.reserved_budget.sum(),1.)
   records.append(d['trades'].query("date=='2015-01-05' and stock_code=='B'").iloc[0])
  self.assertEqual(records[0].planned_units,records[1].planned_units)
  self.assertEqual(records[0].notional,records[1].notional)

 def test_monthly_sale_funds_regular_buy_only_next_session(self):
  m,p,c=fixtures.DailyExecution().fixture()
  for key in ['open','close','factor','up','down','state','suspended']:
   m[key]=np.repeat(m[key],2,axis=1)
  m['codes']=np.array(['A','B']);m['delist']=np.repeat(m['delist'],2)
  p.loc[p.return_month.eq('2015-02'),'stock_code']='B'
  _,d=run(m,p,c,cost_bps=0,details=True)
  sale=d['trades'].query("side=='sell'").iloc[0]
  purchases=d['trades'].query("side=='buy' and stock_code=='B'")
  initial=purchases[purchases.date.eq('2015-02-02')]
  if len(initial):
   self.assertLessEqual((initial.notional+initial.cost).sum(),initial.cash_before_open.iloc[0]+1e-12)
  purchase=purchases[purchases.date.gt(sale.date)].iloc[0]
  self.assertEqual(sale.date,'2015-02-02');self.assertEqual(purchase.date,'2015-02-27')
  self.assertEqual(purchase.funding_asof,sale.date)
  self.assertEqual(purchase.planned_units,purchase.filled_units)
  self.assertTrue((d['daily_nav'].cash_weight>=-1e-12).all())
  self.assertTrue({'hold_count','invested_weight','pending_sell_count'}.issubset(d['daily_nav'].columns))


class FormalResearchPipeline(unittest.TestCase):
 def test_formal_select_evaluate_replays_and_freezes_training(self):
  """Exercise real prepared-input validation, daily run and all research outputs."""
  import json
  import tempfile
  from pathlib import Path
  from unittest.mock import patch
  from microcap import research
  from microcap.acquire import dump,sha
  from microcap.benchmark import monthly_returns
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);prepared=root/'data/prepared';prepared.mkdir(parents=True)
   (root/'results').mkdir()
   early=root/'data/repair_sources/early';early.mkdir(parents=True)
   pd.DataFrame(columns=['stock_code']).to_csv(early/'market_cap.csv',index=False)
   names=['R001_100','R101_200','R201_300','R301_400','R401_500']
   buckets={name:[i+1,i+1] for i,name in enumerate(names)}
   c=dict(train_start='2015-01',train_end='2015-02',test_start='2015-03',test_end='2015-03',
          initial_signal='2014-12-31',buckets=buckets,candidates=names,reference=names[0],
          main_cost_bps=30,cost_bps=[10,30,50],terminal_fraction=0.,terminal_sensitivities=[0.],
          execution='prior_close_orders_next_open_funded',
          hac_lags=1,intramonth_replacement=True,risk_exit_enabled=True,replacement_delay_sessions=1,
          benchmark={'code':'000852.SH','name':'中证1000','return_type':'price_index'},
          st_ablation_rules={'FILTER_ONLY':{'risk_exit_enabled':False,'intramonth_replacement':False},
                             'EXIT_CASH':{'risk_exit_enabled':True,'intramonth_replacement':False},
                             'EXIT_REPLACE':{'risk_exit_enabled':True,'intramonth_replacement':True}})
   dump(root/'config.json',c)
   dates=pd.to_datetime(['2014-12-31','2015-01-05','2015-01-30','2015-02-02','2015-02-27','2015-03-02','2015-03-31'])
   close=np.array([[10]*5,[10.5+i*.1 for i in range(5)],[11+i*.3 for i in range(5)],
                   [9.5+i*.2 for i in range(5)],[10+i*.15 for i in range(5)],
                   [10.1+i*.2 for i in range(5)],[12+i*.1 for i in range(5)]])
   opening=np.vstack([close[0],close[:-1]])
   m=dict(open=opening,close=close,factor=np.ones_like(close),up=opening*1.1,down=opening*.9,
          state=np.zeros_like(close,dtype=np.int8),known_ST=np.zeros_like(close,dtype=bool),
          suspended=np.zeros_like(close,dtype=bool),dates=dates.to_numpy(),codes=np.array(list('ABCDE')),
          delist=np.full(5,np.datetime64('NaT'),dtype='datetime64[D]'))
   np.savez_compressed(prepared/'daily_market.npz',**m)
   dump(prepared/'daily_market_manifest.json',{'sha256':sha(prepared/'daily_market.npz')})
   rows=[]
   for month,signal,horizon in [('2015-01','2014-12-31','2015-01-30'),('2015-02','2015-01-30','2015-02-27'),('2015-03','2015-02-27','2015-03-31')]:
    for j,name in enumerate(names):
     rows.append(dict(return_month=month,signal_date=signal,horizon_end=horizon,stock_code=chr(65+j),
                      rank=j+1,bucket=name,market_cap=(j+1)*100.,st_current=0,new_st_next=0,delisted_next=0))
   p=pd.DataFrame(rows);p.to_csv(prepared/'panel.csv',index=False)
   p[['return_month','signal_date','stock_code','rank','bucket','market_cap']].to_csv(prepared/'selection_universe.csv',index=False)
   cov=pd.DataFrame({'return_month':['2015-01','2015-02','2015-03'],'n_nonfinance_mv_missing':0,'n_finance_unknown':0})
   cov.to_csv(prepared/'universe_coverage.csv',index=False)
   dump(prepared/'manifest.json',{'config_sha256':sha(root/'config.json'),
        'artifacts':[{'path':str(f.relative_to(root)),'sha256':sha(f)} for f in
                     [prepared/'panel.csv',prepared/'selection_universe.csv',prepared/'universe_coverage.csv']]})
   bd=pd.DataFrame({'trade_date':dates,'ts_code':'000852.SH','close':[100,101,102,100,103,104,105]})
   bd.to_csv(prepared/'benchmark_daily.csv',index=False)
   bm=monthly_returns(bd,dates,c);bm.to_csv(prepared/'benchmark_monthly.csv',index=False)
   dump(prepared/'benchmark_manifest.json',{'source':'synthetic independent index fixture'})
   first=research.select(root)
   self.assertEqual(set(first['scores']),set(names))
   # Future quotes change materially; the real training fingerprint and winner cannot.
   m['close'][5:]*=2
   np.savez_compressed(prepared/'daily_market.npz',**m)
   dump(prepared/'daily_market_manifest.json',{'sha256':sha(prepared/'daily_market.npz')})
   replay=research.select(root)
   self.assertEqual(first['winner'],replay['winner'])
   self.assertEqual(first['training_input_sha256'],replay['training_input_sha256'])
   with patch('microcap.research.prepare_benchmark',return_value=bm):
    result=research.evaluate(root)
   from microcap.portfolio import select_winner
   persisted=pd.read_csv(root/'results/monthly.csv',float_precision='round_trip')
   self.assertEqual(select_winner(persisted[persisted.cost_bps.eq(30)],c)['training_hash'],first['training_hash'])
   self.assertEqual(len(result),3*6*3)
   self.assertTrue(result.net_return.notna().all())
   for file in ['daily_risk.csv','cost_sensitivity.csv','st_ablation_performance.csv',
                'st_ablation_tests.csv','st_ablation_execution.csv','order_plans.csv']:
    self.assertTrue((root/'results'/file).is_file())
   trades=pd.read_csv(root/'results/trades.csv')
   self.assertTrue((trades.plan_date<trades.date).all())
   self.assertTrue((trades.funding_asof<trades.date).all())
   np.testing.assert_allclose(trades.planned_units,trades.filled_units)
   plan=pd.read_csv(root/'results/order_plans.csv')
   buys=plan[plan.side.eq('buy')]
   for (strategy,date),g in buys.groupby(['strategy','execute_date']):
    executed=trades[trades.strategy.eq(strategy)&trades.date.eq(date)]
    if len(executed):
     self.assertLessEqual(g.reserved_budget.sum(),executed.cash_before_open.iloc[0]+1e-10)
   sensitivity=pd.read_csv(root/'results/cost_sensitivity.csv')
   self.assertEqual(set(sensitivity.strategy),set(names))
   self.assertEqual(set(sensitivity.cost_bps),{10,30,50})
   ablation=pd.read_csv(root/'results/st_ablation_performance.csv')
   self.assertEqual(set(ablation.rule),set(c['st_ablation_rules']))
   manifest=json.loads((root/'results/evaluation_manifest.json').read_text())
   for item in manifest['artifacts']:
    self.assertEqual(sha(root/item['path']),item['sha256'])


class UnpricedMonthlyTarget(unittest.TestCase):
 def test_first_reopening_close_only_enables_following_open_order(self):
  m,p,c=fixtures.DailyExecution().fixture()
  m['open'][:2]=np.nan;m['close'][:2]=np.nan;m['suspended'][:2]=True
  _,d=run(m,p,c,cost_bps=0,details=True)
  buys=d['trades'].query("side=='buy'")
  self.assertEqual(buys.iloc[0].date,'2015-01-07')
  self.assertEqual(buys.iloc[0].plan_date,'2015-01-06')
  self.assertAlmostEqual(buys.iloc[0].previous_close,10.)
  self.assertFalse(buys.date.eq('2015-01-06').any())
  before=d['daily_nav'].query("date=='2015-01-05'").iloc[0]
  self.assertEqual(before.pending_unpriced_target,1)
  self.assertAlmostEqual(before.cash,1.)
  activated=d['events'].query("event=='unpriced_target_priced'").iloc[0]
  self.assertEqual(activated.date,'2015-01-07')
  self.assertEqual(activated.observed_asof,'2015-01-06')

 def test_never_quoted_target_keeps_its_cash_slot_without_future_backfill(self):
  m,p,c=fixtures.DailyExecution().fixture()
  m['open'][:]=np.nan;m['close'][:]=np.nan;m['suspended'][:]=True
  result,d=run(m,p,c,cost_bps=0,details=True)
  self.assertTrue(d['trades'].empty)
  self.assertTrue(result.nav.eq(1.).all())
  self.assertTrue(d['daily_nav'].pending_unpriced_target.eq(1).all())
  self.assertTrue(d['daily_nav'].unpriced_target_budget.eq(1.).all())
  self.assertEqual(len(d['events'].query("event=='pending_unpriced_target'")),2)
