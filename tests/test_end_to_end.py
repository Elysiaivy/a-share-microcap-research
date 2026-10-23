"""A small deterministic synthetic dataset tests orchestration, not evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from microcap.acquire import dump,sha
from microcap.portfolio import DEFAULT_BUCKETS
from microcap import workflow

class EndToEndTests(unittest.TestCase):
    def test_complete_pipeline_freezes_and_evaluates_without_test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prep=root/'data/prepared';prep.mkdir(parents=True)
            config=dict(train_start='2018-01',train_end='2018-08',test_start='2018-09',test_end='2019-04',
                        main_cost_bps=30,cost_bps=[10,30,50],buckets=DEFAULT_BUCKETS,
                        candidates=list(DEFAULT_BUCKETS)[1:],reference='R001_100',hac_lags=6)
            dump(root/'config.json',config)
            rng=np.random.default_rng(120);rows=[]
            for m,month in enumerate(pd.period_range('2018-01','2019-04',freq='M')):
                for i in range(500):
                    group=i//100;bucket=list(DEFAULT_BUCKETS)[group]
                    rows.append(dict(stock_code=f'{i:06}.SH',return_month=str(month),signal_date=(month-1).end_time.normalize(),
                        horizon_end=month.end_time.normalize(),rank=i+1,bucket=bucket,market_cap=float(i+1),
                        return_1m=.02*np.sin(m)+(.02 if group==1 and m<8 else .0)+rng.normal(0,.015),
                        return_status='quoted_endpoints',market_cap_source='daily_basic',
                        industry=['A','B','C'][i%3],adv20=float(rng.uniform(1e6,1e8)),return_prev1m=rng.normal(0,.06),
                        is_loss=float(i%2),st_current=float(i%7==0),new_st_next=float(i%19==0),delisted_next=0.))
            panel=pd.DataFrame(rows)
            coverage=pd.DataFrame({'return_month':sorted(panel.return_month.unique()),'n_nonfinance_mv_missing':0,'n_finance_unknown':0,'n_unmapped_historical_codes':0,'n_age_eligible':500})
            def save():
                panel.to_csv(prep/'panel.csv',index=False);coverage.to_csv(prep/'universe_coverage.csv',index=False)
                dump(prep/'manifest.json',{'config_sha256':sha(root/'config.json'),'artifacts':[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in prep.glob('*.csv')]})
            save();first=workflow.select(root)
            self.assertEqual(first['status'],'frozen');self.assertEqual(first['winner'],'R101_200')
            result=workflow.evaluate(root)
            self.assertEqual(len(result),16*6*3)
            peers=result[(result.strategy=='PEERS')&(result.cost_bps==30)]
            self.assertTrue(peers.n_target.eq(300).all());self.assertTrue(peers.status.eq('ok').all())
            self.assertTrue(workflow.quality(root)['full_performance_release'])
            risk=pd.read_csv(root/'results/risk_summary_provisional.csv')
            self.assertTrue(risk.n_uncertified_ranking_months.eq(0).all())
            self.assertTrue(risk.composition_status.eq('coverage_passed').all())
            status=workflow.attribute(root)
            self.assertTrue(status.baseline_status.eq('ok').all());self.assertTrue(status.full_status.eq('ok').all())
            pair=json.loads((root/'results/paired_tests.json').read_text())
            self.assertEqual(len(pair['comparisons']),2*(len(config['candidates'])+1))
            self.assertTrue(all(c['inference_status']=='ok' for c in pair['comparisons']))
            # Entire holdout can be rewritten without altering frozen choice/hash.
            panel.loc[panel.return_month.ge('2018-09'),'return_1m']=-.5
            save();again=workflow.select(root)
            self.assertEqual(first,again)
            panel.loc[0,'return_1m']+=.1;save()
            with self.assertRaisesRegex(ValueError,'immutable'):workflow.select(root)

    def test_reference_return_gap_keeps_selection_frozen_but_blocks_full_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prep=root/'data/prepared';prep.mkdir(parents=True)
            config=dict(train_start='2018-01',train_end='2018-02',test_start='2018-03',test_end='2018-04',
                        main_cost_bps=30,cost_bps=[30],buckets=DEFAULT_BUCKETS,
                        candidates=list(DEFAULT_BUCKETS)[1:],reference='R001_100',hac_lags=6)
            dump(root/'config.json',config);rows=[]
            for m,month in enumerate(pd.period_range('2018-01','2018-04',freq='M')):
                for i in range(500):
                    rows.append(dict(stock_code=f'{i:06}.SH',return_month=str(month),
                        signal_date=(month-1).end_time.normalize(),horizon_end=month.end_time.normalize(),
                        rank=i+1,bucket=list(DEFAULT_BUCKETS)[i//100],market_cap=float(i+1),
                        return_1m=[.01,-.01,.02,.03][m],return_status='quoted_endpoints',
                        market_cap_source='daily_basic',st_current=0.,new_st_next=0.,delisted_next=0.,is_loss=0.))
            panel=pd.DataFrame(rows);panel.loc[0,'return_1m']=np.nan
            coverage=pd.DataFrame(dict(return_month=sorted(panel.return_month.unique()),
                n_nonfinance_mv_missing=0,n_finance_unknown=0,n_unmapped_historical_codes=0,n_age_eligible=500))
            panel.to_csv(prep/'panel.csv',index=False);coverage.to_csv(prep/'universe_coverage.csv',index=False)
            dump(prep/'manifest.json',dict(config_sha256=sha(root/'config.json'),
                artifacts=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in prep.glob('*.csv')]))
            selection=workflow.select(root)
            self.assertEqual(selection['status'],'frozen')
            self.assertEqual(selection['winner'],'R401_500')
            monthly=workflow.evaluate(root)
            self.assertTrue(monthly[monthly.strategy.isin(config['candidates'])].status.eq('ok').all())
            quality=workflow.quality(root)
            self.assertFalse(quality['full_performance_release'])
            self.assertEqual(quality['return_gaps'],1)
            self.assertFalse(quality['main_cost_path_complete'])
            performance=pd.read_csv(root/'results/performance.csv')
            reference=performance[performance.strategy.eq('R001_100')&performance.period.eq('TRAIN')].iloc[0]
            self.assertFalse(reference.complete)
            risk=pd.read_csv(root/'results/risk_summary_provisional.csv')
            self.assertTrue(risk.composition_status.eq('coverage_passed').all())
            self.assertTrue(risk.n_uncertified_ranking_months.eq(0).all())

if __name__=='__main__':unittest.main()
