import unittest
import pandas as pd
from microcap.diagnostics import daily_risk


class DailyRiskTests(unittest.TestCase):
    def test_monthly_recovery_does_not_hide_intramonth_drawdown(self):
        c=dict(train_start='2020-01',train_end='2020-01',test_start='2020-02',test_end='2020-02')
        d=pd.DataFrame(dict(strategy=['A']*4,date=['2020-01-02','2020-01-15','2020-01-31','2020-02-03'],nav=[1.,.5,1.,1.2],cash_weight=0.))
        rows=daily_risk(d,c).set_index('period')
        self.assertEqual(rows.loc['TRAIN','daily_max_drawdown'],-.5)
        self.assertEqual(rows.loc['TRAIN','recovery_date'],'2020-01-31')
        self.assertEqual(rows.loc['TEST','daily_max_drawdown'],0.)

    def test_holdout_uses_previous_close_and_initial_loss(self):
        c=dict(train_start='2020-01',train_end='2020-01',test_start='2020-02',test_end='2020-02')
        d=pd.DataFrame(dict(strategy=['A']*3,date=['2020-01-31','2020-02-03','2020-02-28'],nav=[2.,1.5,1.6]))
        row=daily_risk(d,c).query("period=='TEST'").iloc[0]
        self.assertEqual(row.daily_max_drawdown,-.25)
        self.assertTrue(row.drawdown_unrecovered)
