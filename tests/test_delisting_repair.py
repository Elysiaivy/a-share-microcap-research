import numpy as np
import pandas as pd
import unittest
import tempfile
import json
from pathlib import Path
from microcap.delisting_repair import classify_announcement, eligible_cash_return, terminal_scenarios, verify_evidence
from microcap.acquire import sha


def test_terminal_scenarios_require_an_explicit_fraction_and_do_not_fill_panel():
    panel = pd.DataFrame([dict(stock_code='A',delisted_next=1,start_adjusted_price=10.,horizon_end='2024-03-29',return_1m=np.nan)])
    ledger = pd.DataFrame([dict(stock_code='A',last_trade_date='20240312',last_trade_close=4.,last_trade_factor=2.)])
    x = terminal_scenarios(panel,ledger)
    assert x.iloc[0]['research_terminal_return_0x_last_quote'] == -1
    assert np.isclose(x.iloc[0]['research_terminal_return_0.5x_last_quote'],-.6)
    assert np.isclose(x.iloc[0]['research_terminal_return_1x_last_quote'],-.2)
    assert not x.iloc[0].last_quote_is_cash_proceeds
    assert panel.return_1m.isna().all()


def test_cash_election_cannot_be_assigned_after_its_record_date():
    event = dict(record_date='2024-09-05',settlement_by='2024-09-27',cash_per_share=6.42,
                 cash_evidence_certified=True,event_adj_factor=2.)
    late = dict(signal_date='2024-09-30',horizon_end='2024-10-31',start_adjusted_price=12.)
    assert np.isnan(eligible_cash_return(late,event))
    valid = dict(signal_date='2024-08-30',horizon_end='2024-09-30',start_adjusted_price=12.)
    assert np.isclose(eligible_cash_return(valid,event),.07)
    event['cash_evidence_certified']=False
    assert np.isnan(eligible_cash_return(valid,event))


def test_future_terminal_quote_is_rejected():
    panel = pd.DataFrame([dict(stock_code='A',delisted_next=1,start_adjusted_price=10.,horizon_end='2024-03-29')])
    ledger = pd.DataFrame([dict(stock_code='A',last_trade_date='20240402',last_trade_close=4.,last_trade_factor=2.)])
    with unittest.TestCase().assertRaisesRegex(ValueError, 'future'):
        terminal_scenarios(panel,ledger)
    ledger['last_trade_date']=20240402
    with unittest.TestCase().assertRaisesRegex(ValueError, 'future'):
        terminal_scenarios(panel,ledger)


def test_legal_category_requires_explicit_notice_text():
    assert classify_announcement('公司股票终止上市，不进入退市整理期') == 'mandatory_without_final_trading_period'
    assert classify_announcement('主动撤回股票上市，提供现金选择权') == 'voluntary_cash_election'
    assert classify_announcement('公司股票在退市整理期交易后摘牌') == 'mandatory_with_final_trading_period'
    assert classify_announcement('最近价格0.01元') == 'unclassified_requires_notice_review'
    assert classify_announcement('股票终止上市，不设退市整理期') == 'mandatory_without_final_trading_period'


def test_changed_original_notice_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        source=root/'original.txt'; source.write_text('退市公告原文')
        (root/'evidence_manifest.json').write_text(json.dumps(dict(sources=[dict(path=source.name,sha256=sha(source))])))
        verify_evidence(root)
        source.write_text('changed')
        with unittest.TestCase().assertRaisesRegex(ValueError,'hash mismatch'):
            verify_evidence(root)

class TestDelistingRepair(unittest.TestCase):
    test_terminal_scenarios = staticmethod(test_terminal_scenarios_require_an_explicit_fraction_and_do_not_fill_panel)
    test_cash_election = staticmethod(test_cash_election_cannot_be_assigned_after_its_record_date)
    test_future_quote = staticmethod(test_future_terminal_quote_is_rejected)
    test_classification = staticmethod(test_legal_category_requires_explicit_notice_text)
    test_changed_original = staticmethod(test_changed_original_notice_is_rejected)
