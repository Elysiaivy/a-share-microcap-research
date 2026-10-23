import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from microcap.historical_status import build_states


class FakeStore:
    def __init__(self, root, snapshots):
        self.root = Path(root)
        self.snapshots = snapshots

    def find(self, api):
        return [dict(sha256=key, request=dict(body=dict(api_name=api,
            fields='ts_code,trade_date,name', params=dict(trade_date=day, limit=7000))))
            for key, (kind, day, frame) in self.snapshots.items() if kind == api]

    def read(self, entry):
        return self.snapshots[entry['sha256']][2].copy()


class HistoricalStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.codes = ['A', 'B']
        self.days = pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06','2025-01-07','2025-01-08'])
        self.names = pd.DataFrame(columns=['ts_code','name','start_date','end_date','ann_date'])
        frame = pd.DataFrame(dict(ts_code=['A','B'], trade_date=['20250102']*2, name=['普通公司','普通乙']))
        self.store = FakeStore(self.tmp.name, {'hash': ('bak_basic','20250102',frame)})

    def build(self, names=None):
        return build_states(self.store, self.codes, self.days, {}, self.names if names is None else names)

    def test_snapshot_used_at_close_then_next_open_and_never_backfilled(self):
        state, risk, coverage = self.build()
        self.assertEqual(state[:,0].tolist(),[-1,0,0,0,0])
        self.assertEqual(coverage['signal_state'][:,0].tolist(),[0,0,0,0,0])
        self.assertFalse(risk.any())
        self.assertEqual(coverage['summary']['snapshot_filled_stock_days'],8)

    def test_new_risk_event_supersedes_older_ordinary_snapshot(self):
        names = pd.DataFrame([dict(ts_code='A', name='*ST甲', start_date='20250106', end_date=None, ann_date='20250106')])
        state, risk, coverage = self.build(names)
        self.assertEqual(state[:,0].tolist(),[-1,0,0,1,1])
        self.assertEqual(coverage['signal_state'][:,0].tolist(),[0,0,1,1,1])
        self.assertTrue(risk[3,0])

    def test_future_events_do_not_change_past_states(self):
        before = self.build()
        future = pd.DataFrame([dict(ts_code='A', name='ST未来', start_date='20260201', end_date=None, ann_date='20260131')])
        after = self.build(future)
        np.testing.assert_array_equal(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])

    def test_conflicting_interval_and_snapshot_preserve_risk_blocker(self):
        names = pd.DataFrame([dict(ts_code='A', name='ST旧名', start_date='20240101', end_date=None, ann_date='20231229')])
        state, risk, coverage = self.build(names)
        self.assertEqual(state[1,0],-1)
        self.assertTrue(risk[1,0])
        self.assertGreater(coverage['summary']['conflict_stock_days'],0)

    def test_conflicting_snapshots_are_unknown_with_known_risk(self):
        frame=pd.DataFrame(dict(ts_code=['A'],trade_date=['20250102'],name=['ST甲']))
        self.store.snapshots['other']=('bak_daily','20250102',frame)
        state,risk,coverage=self.build()
        self.assertEqual(state[1,0],-1)
        self.assertTrue(risk[1,0])

    def test_current_security_name_is_not_an_input_and_stale_anchor_expires(self):
        days=pd.to_datetime(['2025-01-02','2025-01-03','2025-09-01'])
        state,_,_=build_states(self.store,self.codes,days,{},self.names)
        self.assertEqual(state[-1,0],-1)


if __name__=='__main__':unittest.main()
