import unittest
import numpy as np
import pandas as pd
from microcap.execution import run, risk_name

class DailyExecution(unittest.TestCase):
 def test_delisting_label_is_not_ST_withdrawal(self):
  self.assertTrue(risk_name('新都退'));self.assertTrue(risk_name('*ST新都'))
  self.assertFalse(risk_name('新都酒店'))

 def fixture(self):
  dates=pd.to_datetime(['2014-12-31','2015-01-05','2015-01-06','2015-01-07','2015-01-30','2015-02-02','2015-02-27'])
  shape=(len(dates),1)
  m={k:np.full(shape,v,dtype=float) for k,v in [('open',10),('close',10),('factor',1),('up',11),('down',9)]}
  m.update(dates=dates.to_numpy(),codes=np.array(['A']),state=np.zeros(shape,int),suspended=np.zeros(shape,bool),delist=np.array(['NaT'],dtype='datetime64[D]'))
  p=pd.DataFrame([dict(stock_code='A',rank=1,bucket='B',return_month='2015-01',signal_date='2014-12-31'),dict(stock_code='A',rank=1,bucket='B',return_month='2015-02',signal_date='2015-01-30')])
  c=dict(train_start='2015-01',test_end='2015-02',buckets={'B':[1,1]})
  return m,p,c
 def test_cost_cash_conservation(self):
  m,p,c=self.fixture();r,d=run(m,p,c,cost_bps=30,details=True)
  # Some cash is deliberately reserved at the pre-open upper limit. At flat
  # marks, the only NAV loss is the sum of actual fees.
  self.assertAlmostEqual(r.iloc[0].nav,1-d['trades'].query("date<='2015-01-30'").cost.sum())
  self.assertTrue((d['daily_nav'].cash_weight>=-1e-12).all())
 def test_ST_block_and_retry(self):
  m,p,c=self.fixture();m['state'][2:]=1;m['open'][2]=9;m['close'][2]=9
  r,d=run(m,p,c,cost_bps=0,details=True)
  sells=d['trades'].query("side=='sell'")
  self.assertEqual(sells.iloc[0].date,'2015-01-07');self.assertEqual(sells.iloc[0].delay_sessions,1)
  self.assertEqual(len(d['blocked_orders']),1)
  self.assertEqual(r.iloc[-1].cash_weight,1)
 def test_unknown_no_purchase(self):
  m,p,c=self.fixture();m['state'][:]=-1
  r,d=run(m,p,c,details=True);self.assertTrue(d['trades'].empty);self.assertTrue(r.nav.eq(1).all())
 def test_ST_no_reentry_same_month(self):
  m,p,c=self.fixture();m['state'][2]=1
  r,d=run(m,p,c,details=True)
  buys=d['trades'].query("side=='buy'").date.tolist()
  self.assertEqual([day for day in buys if day<'2015-02-01'],['2015-01-05'])
  self.assertIn('2015-02-02',buys)
 def test_suspension_carry_and_exit(self):
  m,p,c=self.fixture();m['state'][2:]=1;m['suspended'][2:4]=True
  m['open'][2:4]=np.nan;m['close'][2:4]=np.nan
  r,d=run(m,p,c,cost_bps=0,details=True)
  self.assertEqual(d['trades'].query("side=='sell'").iloc[0].date,'2015-01-30')
  self.assertTrue(r.nav.eq(1).all())
 def test_future_ST_cannot_change_past(self):
  m,p,c=self.fixture();a,_=run(m,p,c,end='2015-01')
  m['state'][5:]=1;b,_=run(m,p,c,end='2015-01')
  pd.testing.assert_frame_equal(a,b)

 def test_terminal_residual_on_delist_day(self):
  m,p,c=self.fixture();m['delist'][0]=np.datetime64('2015-01-07')
  m['open'][3:]=np.nan;m['close'][3:]=np.nan;m['state'][3:]=1
  r,d=run(m,p,c,cost_bps=0,terminal_fraction=.5,details=True)
  event=d['events'].query("event=='terminal'").iloc[0]
  self.assertAlmostEqual(r.iloc[0].nav,1-event.marked_value+event.residual)
  self.assertEqual(event.date,'2015-01-07')
  self.assertAlmostEqual(event.residual,event.marked_value*.5)

if __name__=='__main__':unittest.main()
