import unittest
import numpy as np
import pandas as pd
import tempfile
import json
import hashlib
from pathlib import Path
from microcap.valuation_repair import research_mark, rounded_cap, known_notice, merge_gap_seeds, verified_share_inputs


class ValuationRepairTests(unittest.TestCase):
    def test_future_precise_share_disclosure_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'precision_share_announcements.json'
            path.write_text(json.dumps([dict(ts_code='X',trade_date='2020-10-30',shares_date='2020-09-30',ann_date='2020-10-31',total_share=100,accepted=True)]))
            with self.assertRaisesRegex(ValueError,'Future'):
                verified_share_inputs(tmp,tmp)

    def test_conflicting_precise_share_disclosures_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'proof.pdf').write_bytes(b'original proof')
            row=dict(ts_code='X',trade_date='2020-10-30',shares_date='2020-09-30',ann_date='2020-10-29',total_share=100,accepted=True,path='proof.pdf',sha256=hashlib.sha256(b'original proof').hexdigest())
            (Path(tmp)/'first_share_announcements.json').write_text(json.dumps([row]))
            (Path(tmp)/'second_share_announcements.json').write_text(json.dumps([dict(row,total_share=101)]))
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                verified_share_inputs(tmp,tmp)

    def test_precise_share_count_requires_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=dict(ts_code='X',trade_date='2020-10-30',shares_date='2020-09-30',ann_date='2020-10-29',total_share=100,accepted=True)
            (Path(tmp)/'first_share_announcements.json').write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError,'requires an original'):
                verified_share_inputs(tmp,tmp)

    def test_repaired_gap_seeds_survive_replay(self):
        old=pd.DataFrame([dict(ts_code='X',signal_date='2020-01-31')])
        new=pd.DataFrame(columns=old.columns)
        pd.testing.assert_frame_equal(merge_gap_seeds(old,new),old)
        self.assertEqual(len(merge_gap_seeds(old,old)),1)

    def test_blank_is_not_a_suspension(self):
        self.assertTrue(np.isnan(research_mark(2,3,3,confirmed=False)))
        self.assertTrue(np.isnan(research_mark(2,3,np.nan,confirmed=True)))
        self.assertTrue(np.isnan(research_mark(2,3,3,confirmed=True,delisted=True)))

    def test_corporate_action_mark_preserves_adjusted_value(self):
        self.assertAlmostEqual(research_mark(10,2,4,confirmed=True),5)

    def test_rounded_shares_bounds(self):
        cap,lo,hi=rounded_cap(2,8.17)
        self.assertAlmostEqual(cap,1.634e9)
        self.assertLessEqual(lo,2*816627360)
        self.assertGreaterEqual(hi,2*816627360)

    def test_notice_must_already_be_published_and_not_superseded(self):
        n=[dict(ts_code='X',ann_date='2020-02-02',effective_date='2020-01-20')]
        self.assertIsNone(known_notice('X','2020-01-31','2020-01-17',n))
        self.assertIsNotNone(known_notice('X','2020-02-28','2020-01-17',n))
        self.assertIsNone(known_notice('X','2020-02-28','2020-02-10',n))


if __name__=='__main__': unittest.main()
