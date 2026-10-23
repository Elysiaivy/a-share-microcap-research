import unittest
import numpy as np
import test_execution
from microcap.execution import run
from microcap.execution_constraints import OrderConstraints

class ConstraintTests(unittest.TestCase):
 def test_fee_component_not_double_counted(self):
  c=OrderConstraints(minimum_commission=5)
  self.assertAlmostEqual(c.fee(1000,.003),7.75)
  self.assertAlmostEqual(c.fee(20000,.003),60)
  self.assertEqual(c.fee(0,.003),0)
 def test_reserve_includes_minimum_and_lot(self):
  c=OrderConstraints(minimum_commission=5)
  self.assertEqual(c.affordable(1000,1,10,1004,.003),0)
  self.assertEqual(c.affordable(1000,1,10,1007.75,.003),100)
 def test_factor_mapping(self):
  c=OrderConstraints();self.assertEqual(c.round_units(90,2),50)
  self.assertEqual(c.round_units(100,2),100)
 def test_flat_nav_equals_initial_less_all_fees(self):
  m,p,c=test_execution.DailyExecution().fixture();rules=OrderConstraints(10000,100,5)
  r,d=run(m,p,c,details=True,implementation=rules)
  self.assertAlmostEqual(r.iloc[-1].nav,10000-d['trades'].cost.sum())
  buys=d['trades'].query("side=='buy'")
  self.assertTrue(np.allclose(buys.filled_units/100,np.round(buys.filled_units/100)))
  self.assertTrue((d['daily_nav'].cash>=0).all())
 def test_insufficient_capital_does_not_drop_name_or_borrow(self):
  m,p,c=test_execution.DailyExecution().fixture()
  r,d=run(m,p,c,details=True,implementation=OrderConstraints(900,100,5))
  self.assertTrue(d['trades'].empty);self.assertTrue(r.nav.eq(900).all())
  self.assertTrue(d['daily_nav'].target_count.eq(1).all())
 def test_odd_balance_full_ST_exit(self):
  m,p,c=test_execution.DailyExecution().fixture();m['factor'][2:]=1.1
  m['open'][2:]/=1.1;m['close'][2:]/=1.1;m['up'][2:]/=1.1;m['down'][2:]/=1.1
  m['state'][3:]=1
  r,d=run(m,p,c,details=True,implementation=OrderConstraints(1050,100,5))
  sells=d['trades'].query("side=='sell'")
  self.assertEqual(len(sells),1);self.assertEqual(sells.iloc[0].reason,'ST')
  self.assertEqual(d['daily_nav'].iloc[-1].hold_count,0)
 def test_no_rounding_matches_baseline_scaled(self):
  m,p,c=test_execution.DailyExecution().fixture();a,_=run(m,p,c)
  b,_=run(m,p,c,implementation=OrderConstraints(100000,0,0))
  np.testing.assert_allclose(a.net_return,b.net_return,atol=1e-12)

if __name__=='__main__':unittest.main()
