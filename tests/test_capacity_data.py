import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import pandas as pd
from microcap.capacity_data import lagged_statistics,validate_new_daily,normalize_codes,StartLimiter,vwap_range_mask,fetch_days


class CapacityDataTests(unittest.TestCase):
    def fixture(self,n=50):
        amount=np.arange(1,n+1,dtype=float)[:,None]*1000
        price=np.exp(np.arange(n,dtype=float)[:,None]*.01)
        return amount,price,np.ones_like(price),np.zeros_like(price,dtype=bool)

    def test_statistics_observe_only_previous_session(self):
        a,p,f,s=self.fixture();_,adv,sigma,_=lagged_statistics(a,p,f,s)
        a2=a.copy();p2=p.copy();a2[30:]*=100;p2[30:]*=2
        _,adv2,sigma2,_=lagged_statistics(a2,p2,f,s)
        np.testing.assert_array_equal(adv[:31],adv2[:31]);np.testing.assert_array_equal(sigma[:31],sigma2[:31])
        self.assertAlmostEqual(adv[30,0],a[10:30,0].mean())
        self.assertTrue(np.isnan(adv[19,0]));self.assertTrue(np.isfinite(adv[20,0]))

    def test_unknown_gap_is_not_zero_or_forward_filled(self):
        a,p,f,s=self.fixture();a[25]=np.nan;p[25]=np.nan
        amount,adv,sigma,zero=lagged_statistics(a,p,f,s)
        self.assertTrue(np.isnan(amount[25,0]));self.assertFalse(zero[25,0])
        self.assertTrue(np.isnan(adv[26,0]));self.assertTrue(np.isnan(sigma[26,0]))
        self.assertTrue(np.isnan(sigma[46,0]));self.assertTrue(np.isfinite(sigma[47,0]))

    def test_only_known_suspension_gets_zero_amount_and_stale_mark(self):
        a,p,f,s=self.fixture();a[25]=np.nan;p[25]=np.nan;s[25]=True
        amount,adv,sigma,zero=lagged_statistics(a,p,f,s)
        self.assertEqual(amount[25,0],0);self.assertTrue(zero[25,0]);self.assertTrue(np.isfinite(sigma[26,0]))
        self.assertAlmostEqual(adv[26,0],amount[6:26,0].mean())

    def test_vwap_units_and_range_guard(self):
        d=pd.DataFrame([dict(ts_code='000001.SZ',trade_date='20240102',open=10,high=11,low=9,close=10,pre_close=10,vol=1000,amount=1000)])
        self.assertEqual(validate_new_daily(d,'20240102')[0],10)
        d.loc[0,'amount']=2000
        with self.assertRaisesRegex(ValueError,'VWAP outside'):validate_new_daily(d,'20240102')

    def test_pagination_and_account_rate_are_explicit(self):
        d=pd.DataFrame([dict(ts_code='000001.SZ',trade_date='20240102',open=10,high=11,low=9,close=10,pre_close=10,vol=1000,amount=1000)])
        with self.assertRaisesRegex(ValueError,'Pagination'):validate_new_daily(d,'20240102',limit=1)
        with self.assertRaises(ValueError):StartLimiter(.6)
        self.assertEqual(StartLimiter(.35,standard_access_verified=True).seconds,.35)
        with self.assertRaises(ValueError):StartLimiter(.2,standard_access_verified=True)

    def test_unused_provider_preclose_can_be_missing_without_imputation(self):
        d=pd.DataFrame([dict(ts_code='920152.BJ',trade_date='20141021',open=2,high=2,low=2,
                            close=2,pre_close=np.nan,vol=1380,amount=276)])
        self.assertEqual(validate_new_daily(d,'20141021')[0],2)
        self.assertTrue(np.isnan(d.pre_close.iloc[0]))

    def test_audit_mode_preserves_outlier_instead_of_clipping_or_stopping_download(self):
        d=pd.DataFrame([dict(ts_code='920405.BJ',trade_date='20231226',open=11.47,high=11.56,low=11.20,close=11.53,pre_close=11.41,vol=13671.56,amount=15158.372)])
        original=d.copy(deep=True)
        result=validate_new_daily(d,'20231226',strict_vwap=False)
        self.assertLess(result[0],d.low.iloc[0])
        self.assertTrue(vwap_range_mask(d)[2][0])
        pd.testing.assert_frame_equal(d,original)

    def test_negative_turnover_is_never_treated_as_missing_or_zero(self):
        a,p,f,s=self.fixture();a[25]=-1
        with self.assertRaisesRegex(ValueError,'Negative turnover'):
            lagged_statistics(a,p,f,s)

    def test_anomalous_response_is_frozen_and_reused_without_credential(self):
        fields=['ts_code','trade_date','open','high','low','close','pre_close','vol','amount']
        payload={'code':0,'data':{'fields':fields,'items':[['920405.BJ','20231226',11.47,11.56,11.2,11.53,11.41,13671.56,15158.372]]}}
        class Response:
            status_code=200
            def json(self):return payload
        with TemporaryDirectory() as tmp, patch.dict('os.environ',{'TUSHARE_TOKEN':'unit-test-only'}):
            with patch('microcap.capacity_data.requests.post',return_value=Response()) as post:
                fetch_days(tmp,['20231226'],workers=1)
                self.assertEqual(post.call_count,1)
            path=Path(tmp)/'data/capacity/raw/20231226.json'
            self.assertNotIn('unit-test-only',path.read_text())
            self.assertEqual(json.loads(path.read_text())['response'],payload)
            with patch('microcap.capacity_data.requests.post',side_effect=AssertionError('must reuse frozen evidence')):
                fetch_days(tmp,['20231226'],workers=1)

    def test_alias_mapping_is_time_bounded(self):
        d=pd.DataFrame({'ts_code':['000022.SZ'],'amount':[10.]});a=[{'old':'000022.SZ','canonical':'001872.SZ','effective_date':'2018-12-26'}]
        self.assertEqual(normalize_codes(d,'2018-12-25',a).ts_code.iloc[0],'001872.SZ')
        self.assertEqual(normalize_codes(d,'2018-12-26',a).ts_code.iloc[0],'000022.SZ')

    def test_faster_fetch_requires_current_standard_permission(self):
        class Denied:
            status_code=200
            def json(self):return {'code':-1,'data':None}
        with TemporaryDirectory() as tmp, patch.dict('os.environ',{'TUSHARE_TOKEN':'unit-test-only'}):
            with patch('microcap.capacity_data.requests.post',return_value=Denied()) as post:
                with self.assertRaisesRegex(Exception,'access not confirmed'):
                    fetch_days(tmp,['20240102'],workers=1,interval=.35)
                self.assertEqual(post.call_count,1)
                self.assertEqual(post.call_args.kwargs['json']['api_name'],'daily_basic')

    def test_current_standard_permission_is_recorded_without_secret(self):
        class Response:
            status_code=200
            def __init__(self,payload):self.payload=payload
            def json(self):return self.payload
        probe={'code':0,'data':{'fields':['ts_code','trade_date','total_mv'],
                               'items':[['000001.SZ','20251231',100000]]}}
        daily={'code':0,'data':{'fields':['ts_code','trade_date','open','high','low','close','pre_close','vol','amount'],
                               'items':[['000001.SZ','20240102',10,10,10,10,10,1000,1000]]}}
        with TemporaryDirectory() as tmp, patch.dict('os.environ',{'TUSHARE_TOKEN':'unit-test-only'}):
            with patch('microcap.capacity_data.requests.post',side_effect=[Response(probe),Response(daily)]):
                fetch_days(tmp,['20240102'],workers=1,interval=.35)
            receipt=Path(tmp)/'data/capacity/frequency_evidence.json'
            self.assertNotIn('unit-test-only',receipt.read_text())
            self.assertEqual(json.loads(receipt.read_text())['request_interval_seconds'],.35)

    def test_multiple_suspension_intervals_are_not_conflicting_quotes(self):
        frame=pd.DataFrame({'ts_code':['836961.BJ','836961.BJ'],'suspend_type':['S','S'],
                            'suspend_timing':['09:30-09:40','10:00-10:10']})
        self.assertEqual(len(normalize_codes(frame,'20231221',[])),2)


if __name__=='__main__':unittest.main()
