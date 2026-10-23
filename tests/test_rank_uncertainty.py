import unittest
import pandas as pd
from microcap.rank_uncertainty import rank_bounds


class RankBoundsTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'stock_code':[f'{x:04}' for x in range(1,502)],
                             'eligible':True,'market_cap':[float(x) for x in range(1,502)],
                             'market_cap_lower':[float(x) for x in range(1,502)],'market_cap_upper':[float(x) for x in range(1,502)]})
    def test_uncertainty_across_100_boundary_is_reported(self):
        f=self.frame();f.loc[99,'market_cap_upper']=101.5
        r=rank_bounds(f)
        self.assertEqual(set(r.loc[r.band_uncertain,'stock_code']),{'0100','0101'})
    def test_internal_order_changes_do_not_change_band(self):
        f=self.frame();f.loc[9,'market_cap_upper']=12
        self.assertFalse(rank_bounds(f).band_uncertain.any())
    def test_deterministic_equal_cap_tie_break(self):
        f=self.frame();f.loc[100,'market_cap']=100;f.loc[100,['market_cap_lower','market_cap_upper']]=100
        r=rank_bounds(f).set_index('stock_code')
        self.assertEqual(r.loc['0100','min_possible_rank'],100)
        self.assertEqual(r.loc['0101','max_possible_rank'],101)
    def test_uncertainty_beyond_top_500_does_not_change_selected_bands(self):
        f=self.frame();f.loc[500,'market_cap_upper']=600
        self.assertFalse(rank_bounds(f).band_uncertain.any())
