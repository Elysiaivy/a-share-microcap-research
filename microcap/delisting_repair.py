"""Event-level delisting evidence and explicit terminal-value scenarios.

Last exchange quotes are marks, not certified liquidation receipts.  This module
never makes that substitution implicitly.  Cash elections have record dates;
an election that closed before a signal cannot repair a newly formed holding.
"""
from __future__ import annotations

import hashlib
import importlib.util
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import requests

from .acquire import FrozenStore, canonical, dump, sha, default_sources

CNINFO_QUERY = 'https://www.cninfo.com.cn/new/hisAnnouncement/query'
PDF_PYTHON = sys.executable


def _dates(values):
    # CSV reloads YYYYMMDD as integers; do not parse those as nanoseconds.
    return pd.to_datetime(values.astype('string').str.replace(r'\.0$','',regex=True),format='mixed',errors='coerce')


def classify_announcement(text):
    """Classify only explicit wording; a price history alone is not legal proof."""
    compact = re.sub(r'\s+', '', text)
    if any(s in compact for s in ['主动终止', '主动撤回', '主动退市']):
        return 'voluntary_cash_election' if '现金选择权' in compact else 'voluntary_other'
    if any(s in compact for s in ['换股吸收合并', '被吸收合并']):
        return 'share_exchange_merger'
    if re.search(r'(不进入|不设置|不设|不安排|无)[^。；]{0,12}退市整理期', compact):
        return 'mandatory_without_final_trading_period'
    if '退市整理期' in compact and any(s in compact for s in ['终止上市', '摘牌']):
        return 'mandatory_with_final_trading_period'
    if any(s in compact for s in ['强制退市', '交易类退市']):
        return 'mandatory_period_unconfirmed'
    return 'unclassified_requires_notice_review'


def classification_evidence(text):
    compact = re.sub(r'\s+', '', text)
    clauses = re.split(r'[。；]', compact)
    patterns = ('退市整理期', '主动终止', '主动撤回', '主动退市', '换股吸收合并', '被吸收合并')
    return '。'.join(s[-260:] for s in clauses if any(p in s for p in patterns))[:1000]


def _pdf_text(path):
    script = 'import sys; from pypdf import PdfReader; print("\\n".join(p.extract_text() or "" for p in PdfReader(sys.argv[1]).pages))'
    runtime = sys.executable if importlib.util.find_spec('pypdf') else os.environ.get('MICROCAP_PDF_PYTHON',PDF_PYTHON)
    result = subprocess.run([runtime, '-c', script, str(path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('PDF extraction failed; install pypdf or set MICROCAP_PDF_PYTHON')
    return result.stdout


def verify_evidence(directory):
    directory = Path(directory)
    manifest = directory/'evidence_manifest.json'
    if not manifest.exists():
        return
    for item in json.loads(manifest.read_text()).get('sources',[]):
        path = directory/item['path']
        if not path.exists() or sha(path) != item['sha256']:
            raise ValueError(f'Delisting source hash mismatch: {path}')


def freeze_evidence(directory, panel_path):
    directory=Path(directory)
    paths=[]
    for folder in ['queries','pdf','cash_elections']:
        paths.extend(p for p in (directory/folder).rglob('*') if p.is_file())
    if (directory/'notice_overrides.json').exists():
        paths.append(directory/'notice_overrides.json')
    dump(directory/'evidence_manifest.json',dict(
        sources=[dict(path=str(p.relative_to(directory)),sha256=sha(p)) for p in sorted(paths)],
        panel_sha256=sha(panel_path),ledger_sha256=sha(directory/'delisting_ledger.csv'),
        purpose='Sources are immutable; active ledger is regenerated against the current panel and ranking.'))


def _announcement_query(session, directory, params, online):
    key = hashlib.sha256(canonical(params)).hexdigest()
    path = directory/'queries'/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())['response']
    if not online:
        raise FileNotFoundError(f'Uncached CNINFO announcement query {params}')
    time.sleep(.3)
    response = session.post(CNINFO_QUERY, data=params, timeout=35)
    response.raise_for_status()
    data = response.json()
    dump(path, dict(url=CNINFO_QUERY, params=params, response=data,
                    retrieved_at=pd.Timestamp.now(tz='UTC').isoformat()))
    return data


def collect_notices(events, directory, online=False):
    """Download the actual issuer notices from CNINFO, with frozen provenance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    verify_evidence(directory)
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.cninfo.com.cn/'})
    rows = []
    overrides_path=directory/'notice_overrides.json'
    overrides={x['stock_code']:x for x in json.loads(overrides_path.read_text())} if overrides_path.exists() else {}
    for i, event in events.iterrows():
        code = event.stock_code.split('.')[0]
        end = pd.Timestamp(event.delist_date)
        prefix = 'gssh0' if event.stock_code.endswith('.SH') else 'gssz0'
        params = dict(pageNum=1, pageSize=50, column='sse' if event.stock_code.endswith('.SH') else 'szse',
                      tabName='fulltext', stock=f'{code},{prefix}{code}', searchkey='终止上市',
                      secid='', plate='', category='', trade='',
                      seDate=f'{(end-pd.Timedelta(days=60)).date()}~{(end+pd.Timedelta(days=5)).date()}',
                      sortName='time', sortType='desc', isHLtitle='false')
        try:
            data = _announcement_query(session, directory, params, online)
            candidates = [x for x in (data.get('announcements') or []) if x.get('secCode') == code]
            if not candidates:
                # Newer issuers use different org IDs. A date-limited public
                # search is filtered by exact security code before using it.
                params.update(stock='', seDate=f'{(end-pd.Timedelta(days=7)).date()}~{(end+pd.Timedelta(days=2)).date()}')
                data = _announcement_query(session, directory, params, online)
                candidates = [x for x in (data.get('announcements') or []) if x.get('secCode') == code]
                pages = int(np.ceil((data.get('totalAnnouncement') or 0)/params['pageSize']))
                for page in range(2,pages+1):
                    more = _announcement_query(session, directory, dict(params,pageNum=page),online)
                    candidates.extend(x for x in (more.get('announcements') or []) if x.get('secCode') == code)
            if not candidates:
                params.update(stock='',searchkey='摘牌',pageNum=1)
                data=_announcement_query(session,directory,params,online)
                candidates=[x for x in (data.get('announcements') or []) if x.get('secCode')==code]
                pages=int(np.ceil((data.get('totalAnnouncement') or 0)/params['pageSize']))
                for page in range(2,pages+1):
                    more=_announcement_query(session,directory,dict(params,pageNum=page),online)
                    candidates.extend(x for x in (more.get('announcements') or []) if x.get('secCode')==code)
            if not candidates and event.stock_code in overrides:
                item=overrides[event.stock_code]
                candidates=[dict(announcementTitle=item['title'],announcementTime=int(pd.Timestamp(item['date']).timestamp()*1000),
                                 announcementId='verified_direct_source',direct_url=item['url'])]
            candidates.sort(key=lambda x: (int('摘牌' in x['announcementTitle']),
                                           int('决定' in x['announcementTitle']), x['announcementTime']), reverse=True)
            if not candidates:
                rows.append(dict(stock_code=event.stock_code, notice_status='no_matching_notice', event_type='unclassified_requires_notice_review'))
                continue
            notice = candidates[0]
            url = notice.get('direct_url') or 'https://static.cninfo.com.cn/'+notice['adjunctUrl']
            suffix='.pdf' if url.lower().endswith('.pdf') else '.html'
            pdf = directory/'pdf'/f'{event.stock_code}_{notice["announcementId"]}{suffix}'
            pdf.parent.mkdir(parents=True, exist_ok=True)
            if not pdf.exists():
                if not online:
                    raise FileNotFoundError(str(pdf))
                response = session.get(url, timeout=35)
                response.raise_for_status()
                if suffix=='.pdf' and not response.content.startswith(b'%PDF'):
                    raise ValueError('CNINFO response was not a PDF')
                pdf.write_bytes(response.content)
            text_path = pdf.with_suffix('.txt')
            if not text_path.exists():
                text_path.write_text(_pdf_text(pdf) if suffix=='.pdf' else html.unescape(re.sub(r'<[^>]+>',' ',pdf.read_text())))
            text = text_path.read_text()
            rows.append(dict(stock_code=event.stock_code, notice_status='original_pdf_read' if suffix=='.pdf' else 'original_exchange_html_read',
                             event_type=classify_announcement(text), notice_title=notice['announcementTitle'],
                             classification_evidence=classification_evidence(text),
                             notice_date=pd.Timestamp(notice['announcementTime'],unit='ms',tz='UTC').tz_convert('Asia/Shanghai').strftime('%Y-%m-%d'),
                             notice_url=url, notice_sha256=sha(pdf), notice_text_sha256=sha(text_path),
                             notice_path=str(pdf), note='Classification is based on explicit announcement wording; cash recoveries require separate evidence.'))
        except (requests.RequestException, ValueError, FileNotFoundError, RuntimeError) as error:
            rows.append(dict(stock_code=event.stock_code, notice_status='retrieval_failed',
                             event_type='unclassified_requires_notice_review', note=f'{type(error).__name__}: {str(error)[:150]}'))
        if (len(rows) % 20) == 0:
            print(f'delisting notices reviewed: {len(rows)}/{len(events)}', flush=True)
            pd.DataFrame(rows).to_csv(directory/'notice_reviews.partial.csv', index=False)
    result = pd.DataFrame(rows)
    result.to_csv(directory/'notice_reviews.csv', index=False)
    return result


def last_quotes(root, events, directory):
    """Find the actual last daily transaction, rather than a labelled monthly bar."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    store = FrozenStore(directory/'tushare', sources=[*default_sources(), root/'data/raw/manifest.json'])
    limits = {r.stock_code: pd.Timestamp(r.delist_date).strftime('%Y%m%d') for r in events.itertuples()}
    pending = set(limits)
    matches = {}
    entries = store.find('daily')
    entries.sort(key=lambda e: str(e['request']['body'].get('params', {}).get('trade_date', '')), reverse=True)
    for entry in entries:
        day = str(entry['request']['body'].get('params', {}).get('trade_date', ''))
        if not day or not pending or day < '20150101':
            continue
        relevant = {c for c in pending if day <= limits[c]}
        if not relevant:
            continue
        path = Path(entry['source_path'])
        blob = path.read_bytes()
        if entry.get('sha256') and hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError(f'Raw hash mismatch: {path}')
        obj = json.loads(blob)
        if hashlib.sha256(canonical(obj['response'])).hexdigest() != obj['response_sha256']:
            raise ValueError(f'Response hash mismatch: {path}')
        fields = obj['response']['data']['fields']
        if not {'ts_code','trade_date','close'}.issubset(fields):
            continue
        ci, di, pi = fields.index('ts_code'), fields.index('trade_date'), fields.index('close')
        for values in obj['response']['data']['items']:
            code = values[ci]
            if code not in relevant or str(values[di]) > limits[code] or values[pi] is None:
                continue
            q = store.read(entry)
            factor_entries = store.find('adj_factor', trade_date=str(values[di]), offset=0)
            factor, factor_sha = np.nan, None
            if factor_entries:
                f = store.read(factor_entries[-1])
                hit = f[f.ts_code.eq(code)]
                if len(hit):
                    factor, factor_sha = float(hit.iloc[-1].adj_factor), f.attrs['source_sha256']
            matches[code] = dict(stock_code=code, last_trade_date=str(values[di]), last_trade_close=float(values[pi]),
                                 last_trade_factor=factor, quote_source_sha256=q.attrs['source_sha256'], factor_source_sha256=factor_sha)
            pending.remove(code)
    store.save_manifest()
    result = pd.DataFrame([matches.get(c, dict(stock_code=c)) for c in sorted(limits)])
    result.to_csv(directory/'last_quotes.csv', index=False)
    return result


def security_share_evidence(root, events, directory):
    """Attach historical backup share counts and the delisted master provenance."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    store = FrozenStore(directory/'tushare', sources=[*default_sources(), root/'data/raw/manifest.json'])
    master_sources = {}
    for entry in store.find('stock_basic', list_status='D'):
        frame = store.read(entry)
        for code in frame.ts_code:
            master_sources[code] = frame.attrs['source_sha256']
    rows = []
    for day, group in events.groupby('signal_date'):
        stamp = pd.Timestamp(day).strftime('%Y%m%d')
        entries = store.find('bak_daily',trade_date=stamp,offset=0)
        backup = store.read(entries[-1]) if entries else pd.DataFrame()
        basic_entries=store.find('bak_basic',trade_date=stamp,offset=0)
        basic=store.read(basic_entries[-1]) if basic_entries else pd.DataFrame()
        for event in group.itertuples():
            hit = backup[backup.ts_code.eq(event.stock_code)] if 'ts_code' in backup else pd.DataFrame()
            shares = float(hit.iloc[0].total_share) if len(hit) and 'total_share' in hit else np.nan
            if not shares > 0:
                shares = np.nan
            source_sha=backup.attrs.get('source_sha256') if len(hit) else None
            source='historical_bak_daily' if pd.notna(shares) else 'unresolved'
            if pd.isna(shares) and 'ts_code' in basic:
                b=basic[basic.ts_code.eq(event.stock_code)]
                if len(b) and 'total_share' in b and float(b.iloc[0].total_share or 0)>0:
                    shares=float(b.iloc[0].total_share)
                    source_sha=basic.attrs.get('source_sha256')
                    source='historical_bak_basic'
            rows.append(dict(stock_code=event.stock_code,signal_total_shares_100m=shares,
                             shares_source_sha256=source_sha,shares_status=source,
                             delisting_master_source_sha256=master_sources.get(event.stock_code)))
    store.save_manifest()
    return pd.DataFrame(rows)


def terminal_scenarios(panel, ledger, fractions=(0., .5, 1.)):
    """Return clearly labelled marks; these are not observed recovery returns."""
    p = panel[panel.delisted_next.eq(1)].copy()
    fields = ['stock_code', 'last_trade_date', 'last_trade_close', 'last_trade_factor']
    out = p.merge(ledger[fields], on='stock_code', how='left', validate='many_to_one')
    if (_dates(out.last_trade_date) > _dates(out.horizon_end)).any():
        raise ValueError('Terminal quote is in the future')
    mark = out.last_trade_close * out.last_trade_factor
    out['last_quote_is_cash_proceeds'] = False
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError('A terminal valuation fraction must be within [0, 1]')
        value = fraction * mark / out.start_adjusted_price - 1
        if fraction == 0:
            value = pd.Series(np.where(out.start_adjusted_price.gt(0), -1., np.nan), index=out.index)
        out[f'research_terminal_return_{fraction:g}x_last_quote'] = value
    return out


def eligible_cash_return(row, event):
    """Only an eligible holding with settlement evidence can use actual proceeds.

    Explicit eligibility does not make the historical strategy executable during
    suspension. That issue is retained separately by the portfolio implementation.
    """
    required = ['record_date', 'settlement_by', 'cash_per_share', 'cash_evidence_certified']
    if any(k not in event or pd.isna(event[k]) for k in required) or not event['cash_evidence_certified']:
        return np.nan
    if pd.Timestamp(row['signal_date']) > pd.Timestamp(event['record_date']):
        return np.nan
    if pd.Timestamp(row['horizon_end']) < pd.Timestamp(event['settlement_by']):
        return np.nan
    if not (row['start_adjusted_price'] > 0 and event.get('event_adj_factor', np.nan) > 0):
        return np.nan
    return event['cash_per_share'] * event['event_adj_factor'] / row['start_adjusted_price'] - 1


def build_ledger(root, online=False):
    root = Path(root).resolve()
    directory = root/'data/repair_sources/delisting'
    directory.mkdir(parents=True, exist_ok=True)
    verify_evidence(directory)
    panel = pd.read_csv(root/'data/prepared/panel.csv')
    events = panel[panel.delisted_next.eq(1)].copy().sort_values('stock_code')
    seed_path=directory/'event_seed.csv'
    seed_fields=['stock_code','delist_date','name']
    seeds=events[seed_fields]
    if seed_path.exists():
        seeds=pd.concat([pd.read_csv(seed_path),seeds]).drop_duplicates(['stock_code','delist_date'],keep='last')
    seeds.sort_values(['stock_code','delist_date']).to_csv(seed_path,index=False)
    # All event exposures, including those with a month-end quote, are audited.
    quotes = last_quotes(root, events, directory)
    share_evidence = security_share_evidence(root,events,directory)
    notices = collect_notices(events, directory, online=online)
    fields = ['stock_code','name','signal_date','horizon_end','return_month','delist_date','bucket','rank',
              'start_close','start_adj_factor','start_adjusted_price','market_cap','market_cap_source','return_status']
    ledger = events[fields].merge(quotes, on='stock_code', how='left', validate='one_to_one').merge(notices,on='stock_code',how='left',validate='one_to_one')
    ledger = ledger.merge(share_evidence,on='stock_code',how='left',validate='one_to_one')
    ledger['last_quote_age_at_delisting_days'] = (_dates(ledger.delist_date)-_dates(ledger.last_trade_date)).dt.days
    ledger['last_quote_is_realizable_proceeds'] = False
    ledger['cash_proceeds_certified'] = False
    ledger['cash_election_price'] = np.nan
    ledger['cash_election_record_date'] = None
    ledger['cash_election_deadline'] = None
    ledger['cash_election_terms_known'] = False
    ledger['signal_qualifies_for_election'] = False
    cash_events_path = directory/'cash_elections'/'events.json'
    if cash_events_path.exists():
        for event in json.loads(cash_events_path.read_text()):
            mask = ledger.stock_code.eq(event['stock_code'])
            ledger.loc[mask, 'cash_election_price'] = event['cash_per_share']
            ledger.loc[mask, 'cash_election_record_date'] = event['record_date']
            ledger.loc[mask, 'cash_election_deadline'] = event['election_deadline']
            ledger.loc[mask, 'cash_election_terms_known'] = True
            ledger.loc[mask, 'event_type'] = 'voluntary_cash_election'
            ledger.loc[mask, 'signal_qualifies_for_election'] = pd.to_datetime(ledger.loc[mask,'signal_date']).le(pd.Timestamp(event['record_date']))
    ledger['reviewer'] = 'Codex source and arithmetic review; not human sign-off'
    ledger.to_csv(directory/'delisting_ledger.csv',index=False)
    terminal_scenarios(events, ledger).to_csv(directory/'terminal_scenarios.csv',index=False)
    dump(directory/'summary.json', dict(events=len(ledger), distinct_stocks=ledger.stock_code.nunique(),
         original_notices=int(ledger.notice_status.isin(['original_pdf_read','original_exchange_html_read']).sum()),
         last_quotes=int(ledger.last_trade_close.notna().sum()), last_factors=int(ledger.last_trade_factor.notna().sum()),
         historical_share_snapshots=int(ledger.signal_total_shares_100m.notna().sum()),
         missing_share_snapshots=int(ledger.signal_total_shares_100m.isna().sum()),
         delisting_master_sources=int(ledger.delisting_master_source_sha256.notna().sum()),
         cash_elections_with_current_signal_eligibility=int(ledger.signal_qualifies_for_election.sum()),
         certified_cash_return_repairs=0,input_panel_sha256=sha(root/'data/prepared/panel.csv'),
         event_types=ledger.event_type.value_counts().to_dict(),
         limitation='Terminal quote fractions are valuation assumptions, not observed proceeds; no implicit formal-panel fill.'))
    freeze_evidence(directory,root/'data/prepared/panel.csv')
    return ledger


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='.')
    parser.add_argument('--online',action='store_true')
    args=parser.parse_args()
    result=build_ledger(args.root,online=args.online)
    print(result.event_type.value_counts().to_string())
