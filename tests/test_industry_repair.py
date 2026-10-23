import unittest
import tempfile
import json
from pathlib import Path
import pandas as pd
from microcap.industry_repair import apply_prior_financial_snapshot, repair_financial_classification
from microcap.acquire import sha


class IndustryRepairTests(unittest.TestCase):
    def setUp(self):
        self.active = pd.DataFrame({'ts_code':['A','B','C'],'is_financial':[None,None,0.],
                                    'industry':[None,None,'制造'],'financial_source':['unknown','unknown','SW']})
        self.snapshot = pd.DataFrame({'ts_code':['A','B','C'],'trade_date':['20210128']*3,
                                      'industry':['证券','机械','银行']})
        self.snapshot.attrs['source_sha256'] = 'evidence'
        self.calendar = pd.to_datetime(['2021-01-27','2021-01-28','2021-01-29','2021-02-01'])

    def apply(self, day='2021-01-28', snapshot=None):
        return apply_prior_financial_snapshot(self.active, self.snapshot if snapshot is None else snapshot,
            '2021-01-29', day, self.calendar, {'证券','银行'})

    def test_prior_snapshot_fills_only_exclusion_and_keeps_sw_and_known_labels(self):
        result = self.apply()
        self.assertEqual(result.is_financial.tolist(), [1.,0.,0.])
        self.assertTrue(result.loc[:1,'industry'].isna().all())
        self.assertEqual(result.loc[0,'financial_source_date'],'2021-01-28')
        self.assertEqual(result.loc[0,'financial_source_sha256'],'evidence')

    def test_future_stale_and_wrong_response_dates_rejected(self):
        for day in ['2021-02-01','2021-01-27']:
            with self.assertRaises(ValueError): self.apply(day)
        wrong = self.snapshot.copy();wrong['trade_date'] = '20210129'
        with self.assertRaises(ValueError): self.apply(snapshot=wrong)

    def test_unknown_not_classified_nonfinancial_and_duplicates_rejected(self):
        missing=self.snapshot.copy();missing.loc[0,'industry']=None
        self.assertTrue(pd.isna(self.apply(snapshot=missing).loc[0,'is_financial']))
        with self.assertRaises(ValueError): self.apply(snapshot=pd.concat([missing,missing]))

    def test_dated_snapshot_updates_control_industry_and_exclusion_together(self):
        result=apply_prior_financial_snapshot(self.active,self.snapshot,'2021-01-29',
            '2021-01-28',self.calendar,{'证券','银行'},update_industry=True)
        self.assertEqual(result.loc[0,'industry'],'证券')
        self.assertEqual(result.loc[1,'industry'],'机械')
        self.assertEqual(result.loc[2,'industry'],'制造')

    def test_official_quarter_is_unavailable_until_publication_and_uses_sector_j(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); directory=root/'data/repair_sources/point_in_time/official_industry';directory.mkdir(parents=True)
            pdf=directory/'source.pdf';pdf.write_bytes(b'frozen-original')
            parsed=directory/'source.csv'
            pd.DataFrame([dict(ts_code='A',name='金融甲',industry='资本市场服务',industry_code='67',industry_sector='J',
                is_financial=1,observation_date='2020-12-31',publication_date='2021-01-22',source_sha256=sha(pdf))]).to_csv(parsed,index=False)
            (directory/'manifest.json').write_text(json.dumps(dict(sources=[dict(path=str(pdf.relative_to(root)),sha256=sha(pdf),
                parsed_path=str(parsed.relative_to(root)),parsed_sha256=sha(parsed))])))
            class EmptyStore:
                def __init__(self):self.root=root
                def find(self,*args,**kwargs):return []
            store=EmptyStore()
            early=repair_financial_classification(store,self.active,'2021-01-21',self.calendar,{'银行'})
            later=repair_financial_classification(store,self.active,'2021-01-29',self.calendar,{'银行'})
            self.assertTrue(pd.isna(early.loc[0,'is_financial']))
            self.assertEqual(later.loc[0,'is_financial'],1)
            self.assertEqual(later.loc[0,'industry'],'资本市场服务')


if __name__ == '__main__': unittest.main()
