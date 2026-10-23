import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from microcap.acquire import FrozenStore, canonical, dump, sha
from microcap.prepare import aliases
from microcap.workflow import model_panel,training_fingerprint,ranking_gate
from microcap.portfolio import DEFAULT_BUCKETS
from microcap import workflow

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config={'train_start':'2018-01','train_end':'2023-12'}
        self.panel=pd.DataFrame(dict(return_month=['2018-01','2024-01'],signal_date=['2017-12-29','2023-12-29'],stock_code=['A','A'],rank=[1,1],bucket=['R001_100']*2,market_cap=[100.,200.],return_1m=[.1,.2],st_current=[0.,0.]))
        self.cov=pd.DataFrame(dict(return_month=['2018-01','2024-01'],n_nonfinance_mv_missing=[0,0],n_finance_unknown=[0,0],n_unmapped_historical_codes=[0,0]))
    def test_training_fingerprint_ignores_oos_and_future_labels(self):
        before=training_fingerprint(self.panel,self.cov,self.config)
        changed=self.panel.copy();changed.loc[1,'return_1m']=-.9;changed['st_current']=1
        self.assertEqual(before,training_fingerprint(changed,self.cov,self.config))
        changed.loc[0,'return_1m']+=.01
        self.assertNotEqual(before,training_fingerprint(changed,self.cov,self.config))
    def test_universe_gap_invalidates_entire_month_not_only_missing_stock(self):
        self.cov.loc[0,'n_nonfinance_mv_missing']=1
        p=model_panel(self.panel,self.cov)
        self.assertTrue(pd.isna(p.loc[0,'return_1m']))
        self.assertEqual(p.loc[1,'return_1m'],.2)
    def test_unmapped_positive_legacy_code_blocks_rank(self):
        self.cov.loc[1,'n_unmapped_historical_codes']=1
        self.assertEqual(ranking_gate(self.cov).tolist(),[True,False])
    def test_alias_uses_effective_date_and_refuses_double_count(self):
        config={'security_aliases':[{'old':'000022.SZ','canonical':'001872.SZ','effective_date':'2018-12-26'}]}
        data=pd.DataFrame({'ts_code':['000022.SZ'],'close':[10.]})
        self.assertEqual(aliases(data,'2018-01-01',config).ts_code.iloc[0],'001872.SZ')
        self.assertEqual(aliases(data,'2019-01-01',config).ts_code.iloc[0],'000022.SZ')
        same=pd.concat([data,pd.DataFrame({'ts_code':['001872.SZ'],'close':[10.]})])
        self.assertEqual(len(aliases(same,'2018-01-01',config)),1)
        double=pd.concat([data,pd.DataFrame({'ts_code':['001872.SZ'],'close':[11.]})])
        with self.assertRaises(ValueError):aliases(double,'2018-01-01',config)
    def test_raw_response_copy_is_verified_and_portable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source.json'
            response={'code':0,'data':{'fields':['ts_code','trade_date','total_mv'],'items':[['A','20180131',100.]]}}
            req={'provider':'tushare','body':{'api_name':'daily_basic','params':{'trade_date':'20180131','offset':0,'limit':6000},'fields':'ts_code,trade_date,total_mv'}}
            dump(source,dict(request=req,response=response,response_sha256=hashlib.sha256(canonical(response)).hexdigest(),acquired_at='2020-01-01'))
            store=FrozenStore(root/'project',sources=[])
            entry={'source_path':str(source),'sha256':sha(source),'request':req}
            self.assertEqual(store.read(entry).total_mv.iloc[0],100.)
            store.save_manifest();source.unlink()
            replay=FrozenStore(root/'project',sources=[])
            self.assertEqual(replay.date_query('daily_basic','20180131','ts_code,trade_date,total_mv').total_mv.iloc[0],100.)
    def test_tampered_raw_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad.json';dump(p,{'response':{'code':0,'data':{'fields':[],'items':[]}},'response_sha256':'wrong'})
            with self.assertRaisesRegex(ValueError,'Response hash mismatch'):FrozenStore(Path(temp)/'p',sources=[]).read({'source_path':str(p)})

    def test_external_project_manifest_resolves_project_relative_raw_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'original_response.json'
            response={'code':0,'data':{'fields':['ts_code','trade_date','total_mv'],
                                      'items':[['A','20180131',100.]]}}
            req={'provider':'tushare','body':{'api_name':'daily_basic',
                  'params':{'trade_date':'20180131','offset':0,'limit':6000},
                  'fields':'ts_code,trade_date,total_mv'}}
            dump(source,dict(request=req,response=response,
                            response_sha256=hashlib.sha256(canonical(response)).hexdigest()))
            first=FrozenStore(root/'first_project',sources=[])
            first.read({'source_path':str(source),'sha256':sha(source),'request':req})
            first.save_manifest();source.unlink()
            external_manifest=first.raw/'manifest.json'
            second=FrozenStore(root/'second_project',sources=[external_manifest])
            self.assertEqual(second.date_query('daily_basic','20180131',
                                             'ts_code,trade_date,total_mv').total_mv.iloc[0],100.)
            second.save_manifest()
            # The imported copy can subsequently replay without the external project.
            for path in first.raw.iterdir():path.unlink()
            replay=FrozenStore(root/'second_project',sources=[])
            self.assertEqual(replay.date_query('daily_basic','20180131',
                                             'ts_code,trade_date,total_mv').total_mv.iloc[0],100.)


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.prep=self.root/'data/prepared';self.prep.mkdir(parents=True)
        self.config=dict(train_start='2018-01',train_end='2018-02',test_start='2018-03',test_end='2018-04',
                         buckets=DEFAULT_BUCKETS,candidates=list(DEFAULT_BUCKETS)[1:],
                         reference='R001_100',main_cost_bps=30,cost_bps=[30],hac_lags=6)
        dump(self.root/'config.json',self.config)
        rows=[]
        for m,month in enumerate(pd.period_range('2018-01','2018-04',freq='M')):
            for i in range(500):
                rows.append(dict(stock_code=f'{i:06}.SH',return_month=str(month),
                    signal_date=(month-1).end_time.normalize(),horizon_end=month.end_time.normalize(),
                    rank=i+1,bucket=list(DEFAULT_BUCKETS)[i//100],market_cap=float(i+1),
                    return_1m=[.01,-.02,.03,-.01][m],return_status='quoted_endpoints',
                    market_cap_source='daily_basic',st_current=0.,new_st_next=0.,delisted_next=0.,is_loss=0.))
        self.panel=pd.DataFrame(rows)
        self.coverage=pd.DataFrame(dict(return_month=sorted(self.panel.return_month.unique()),
            n_nonfinance_mv_missing=0,n_finance_unknown=0,n_unmapped_historical_codes=0,n_age_eligible=500))
        self.save();workflow.select(self.root);workflow.evaluate(self.root)

    def save(self):
        self.panel.to_csv(self.prep/'panel.csv',index=False)
        self.coverage.to_csv(self.prep/'universe_coverage.csv',index=False)
        dump(self.prep/'manifest.json',dict(config_sha256=sha(self.root/'config.json'),
             artifacts=[dict(path=str(p.relative_to(self.root)),sha256=sha(p)) for p in self.prep.glob('*.csv')]))

    def test_complete_evaluated_project_passes_release_gate(self):
        result=workflow.quality(self.root)
        self.assertTrue(result['full_performance_release'])
        self.assertTrue(result['portfolio_structure_complete'])
        self.assertTrue(result['main_cost_path_complete'])
        self.assertTrue(result['evaluation_artifacts_verified'])

    def test_nonfinite_or_below_total_loss_return_blocks_release(self):
        index=self.panel.index[self.panel.return_month.eq('2018-03')][0]
        for invalid in [np.inf,-np.inf,-1.001,np.nan]:
            with self.subTest(value=invalid):
                self.panel.loc[index,'return_1m']=invalid;self.save();workflow.evaluate(self.root)
                result=workflow.quality(self.root)
                self.assertEqual(result['return_gaps'],1)
                self.assertFalse(result['full_performance_release'])

    def test_missing_month_or_group_blocks_release(self):
        original=self.panel.copy()
        for kind in ['month','group']:
            with self.subTest(missing=kind):
                remove=original.return_month.eq('2018-03')
                if kind=='group':remove &= original.bucket.eq('R101_200')
                self.panel=original.loc[~remove].copy();self.save();workflow.evaluate(self.root)
                result=workflow.quality(self.root)
                self.assertFalse(result['portfolio_structure_complete'])
                self.assertFalse(result['full_performance_release'])

    def test_incomplete_cost_status_or_nonfinite_net_blocks_release(self):
        original_simulate=workflow.simulate
        for kind in ['status','net','missing_row']:
            with self.subTest(problem=kind):
                def incomplete_simulation(*args,**kwargs):
                    monthly,holdings=original_simulate(*args,**kwargs)
                    mask=monthly.strategy.eq('R101_200')&monthly.return_month.eq('2018-03')
                    if kind=='status':monthly.loc[mask,'status']='unknown_pretrade'
                    elif kind=='net':monthly.loc[mask,'net_return']=np.inf
                    else:monthly=monthly.loc[~mask]
                    return monthly,holdings
                with patch('microcap.workflow.simulate',side_effect=incomplete_simulation):
                    workflow.evaluate(self.root)
                result=workflow.quality(self.root)
                self.assertFalse(result['main_cost_path_complete'])
                self.assertFalse(result['full_performance_release'])

    def test_missing_evaluation_manifest_blocks_release_without_error(self):
        (self.root/'results/evaluation_manifest.json').unlink()
        result=workflow.quality(self.root)
        self.assertTrue(result['main_cost_path_complete'])
        self.assertFalse(result['evaluation_artifacts_verified'])
        self.assertFalse(result['full_performance_release'])

    def test_finite_holdout_update_requires_fresh_evaluation_without_reselecting(self):
        original_selection=(self.root/'results/selection.json').read_bytes()
        self.panel.loc[self.panel.return_month.eq('2018-03'),'return_1m']=.25
        self.save()
        with self.assertRaisesRegex(ValueError,'Stale evaluation inputs'):
            workflow.quality(self.root)
        workflow.evaluate(self.root)
        self.assertEqual((self.root/'results/selection.json').read_bytes(),original_selection)
        self.assertTrue(workflow.quality(self.root)['full_performance_release'])

    def test_tampered_or_missing_evaluation_outputs_are_rejected(self):
        for name in workflow.EVALUATION_ARTIFACTS:
            with self.subTest(artifact=name):
                path=self.root/'results'/name;original=path.read_bytes()
                path.write_bytes(original+b'\n')
                with self.assertRaisesRegex(ValueError,'Evaluation artifact checksum failed'):
                    workflow.quality(self.root)
                path.unlink()
                with self.assertRaisesRegex(ValueError,'Evaluation artifact checksum failed'):
                    workflow.quality(self.root)
                path.write_bytes(original)

    def test_manifest_cannot_omit_required_artifact(self):
        path=self.root/'results/evaluation_manifest.json'
        manifest=json.loads(path.read_text());manifest['artifacts'].pop();dump(path,manifest)
        with self.assertRaisesRegex(ValueError,'Incomplete evaluation artifact manifest'):
            workflow.quality(self.root)

    def test_risk_summary_identifies_untrusted_ranking_by_period(self):
        self.coverage.loc[self.coverage.return_month.eq('2018-03'),'n_nonfinance_mv_missing']=1
        self.save();workflow.evaluate(self.root)
        summary=pd.read_csv(self.root/'results/risk_summary_provisional.csv')
        for period in ['TEST','ALL','2018']:
            rows=summary[summary.period.eq(period)]
            self.assertTrue(rows.n_uncertified_ranking_months.eq(1).all())
            self.assertTrue(rows.composition_status.eq('provisional_missing_universe').all())
            # A full provisional 100-name group does not certify the universe.
            self.assertTrue(rows.complete_holdings.all())
        self.assertTrue(summary.loc[summary.period.eq('TRAIN'),'n_uncertified_ranking_months'].eq(0).all())
        self.assertTrue(summary.loc[summary.period.eq('TRAIN'),'composition_status'].eq('coverage_passed').all())

if __name__=='__main__':unittest.main()
