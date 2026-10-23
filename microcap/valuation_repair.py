"""Audited suspension valuations, kept separate from observable execution prices.

A missing trade is not evidence of a suspension. A repair requires either a
full-session Tushare suspend_d record or an already published suspension notice.
The fixed research mark carries the last *adjusted* traded quote; it never claims
that the stock was tradable or that a delisting recovery was received.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .acquire import FrozenStore, canonical, default_sources, dump, sha


# These are original company disclosures, not later descriptions backdated into
# the information set. New verified records may also be supplied in notices.json.
NOTICES = [
    dict(ts_code='000670.SZ', effective_date='2020-04-07', ann_date='2020-04-30',
         url='https://static.cninfo.com.cn/finalpage/2020-04-30/1207679635.PDF',
         title='盈方微2020年第一季度报告：自2020年4月7日起暂停上市'),
    dict(ts_code='000939.SZ', effective_date='2019-05-13', ann_date='2019-05-13',
         url='https://disc.static.szse.cn/download/disc/disk01/finalpage/2019-05-13/76619a1d-4064-45c7-bf36-cdb74ffa7942.PDF',
         title='凯迪生态关于股票暂停上市的公告'),
    dict(ts_code='002260.SZ', effective_date='2019-05-15', ann_date='2019-11-07',
         url='https://disc.static.szse.cn/disc/disk02/finalpage/2019-11-07/61fa2ae2-dc52-4b6a-aa1f-02bb7427cb55.PDF',
         title='德奥通航关于重整及恢复上市进展情况的公告'),
]


def date_key(value):
    return pd.Timestamp(str(value)).strftime('%Y-%m-%d')


def research_mark(last_close, last_factor, factor, *, confirmed, delisted=False):
    """Return a documented stale mark, never infer a zero return from a blank."""
    vals = np.asarray([last_close, last_factor, factor], dtype=float)
    if delisted or not confirmed or not np.isfinite(vals).all() or (vals <= 0).any():
        return np.nan
    return float(last_close * last_factor / factor)


def rounded_cap(close, shares_yi):
    """bak_basic shares are rounded to .01 hundred-million shares.

    This +/- .005 bound applies to ordinary rounding. The source's rounding
    convention is not certified, so the exported conservative envelope uses
    +/- .01 (one last-displayed unit), also covering truncation.
    """
    if not np.isfinite(close) or not np.isfinite(shares_yi) or close <= 0 or shares_yi <= 0:
        return np.nan, np.nan, np.nan
    return (close * shares_yi * 1e8, close * max(0, shares_yi - .01) * 1e8,
            close * (shares_yi + .01) * 1e8)


def known_notice(code, day, last_trade_date, notices):
    day, last_trade_date = pd.Timestamp(day), pd.Timestamp(last_trade_date)
    valid = [n for n in notices if n['ts_code'] == code and
             pd.Timestamp(n['ann_date']) <= day and
             last_trade_date < pd.Timestamp(n['effective_date']) <= day]
    # A later real trade invalidates an older open-ended suspension notice.
    return max(valid, key=lambda n:n['ann_date']) if valid else None


def _inspect(entry):
    """Read/verify a cache while searching; freeze only selected source files."""
    path = Path(entry['source_path']); raw = path.read_bytes()
    if entry.get('sha256') and hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError(f'Raw hash mismatch: {path}')
    obj = json.loads(raw)
    if obj['response'].get('code') != 0 or hashlib.sha256(canonical(obj['response'])).hexdigest() != obj['response_sha256']:
        raise ValueError(f'Response verification failed: {path}')
    data = obj['response']['data']
    return pd.DataFrame(data.get('items', []), columns=data.get('fields', []))


REQUIRED = {
    'daily': {'ts_code','trade_date','close'},
    'adj_factor': {'ts_code','trade_date','adj_factor'},
    'daily_basic': {'ts_code','trade_date','total_mv'},
    'bak_basic': {'ts_code','trade_date','total_share'},
    'suspend_d': {'ts_code','trade_date','suspend_type','suspend_timing'},
}


def _validate_response_date(frame, day):
    if not frame.empty and not frame.trade_date.astype(str).str.replace(r'\.0$', '',regex=True).eq(pd.Timestamp(day).strftime('%Y%m%d')).all():
        raise ValueError('Historical response date differs from requested date')


def verified_raw_source(store, digest):
    entries=[e for e in store.index.values() if e.get('sha256')==digest]
    if not entries:
        raise ValueError('Referenced raw source hash is not in the frozen manifest')
    return store.read(entries[-1])


def verified_candidate_shares(store, candidate):
    source=verified_raw_source(store,candidate.source_sha256)
    rows=source[source.ts_code.eq(candidate.ts_code) & source.trade_date.astype(str).str.replace(r'\.0$', '',regex=True).eq(pd.Timestamp(candidate.shares_date).strftime('%Y%m%d'))]
    if len(rows)!=1:
        raise ValueError('Share candidate has no unique raw observation')
    units={'last_known_bak_basic_one_session_stale':1e8,'stock_specific_daily_basic':10000}
    if candidate.source not in units or not np.isclose(float(rows.iloc[0].total_share)*units[candidate.source],float(candidate.total_share),rtol=0,atol=.001):
        raise ValueError('Share candidate differs from its frozen raw observation')
    return source


def _date_entries(store, api):
    candidates={}
    for e in store.find(api):
        body=e['request']['body'];p=body['params']
        if p.get('trade_date') and int(p.get('offset',0))==0 and not p.get('ts_code') and REQUIRED[api].issubset(set(body.get('fields','').split(','))):
            candidates.setdefault(str(p['trade_date']),[]).append(e)
    out={}
    for day,entries in candidates.items():
        if len(entries)==1:
            out[day]=entries[0];continue
        viable=[]
        for e in entries:
            f=_inspect(e);_validate_response_date(f,day)
            if len(f)>=int(e['request']['body']['params'].get('limit',1000000)):
                raise ValueError('Incomplete historical snapshot pagination')
            if not f.empty:viable.append((len(f.columns),e))
        out[day]=max(viable,key=lambda x:x[0])[1] if viable else entries[-1]
    return out


class History:
    def __init__(self, store, codes):
        self.store = store
        self.entries = {api:_date_entries(store, api) for api in ['daily','adj_factor','bak_basic','daily_basic','suspend_d']}
        self.frames = {}
        pieces = []
        for day, e in sorted(self.entries['daily'].items()):
            if not '20140101' <= day <= '20251231':
                continue
            f = _inspect(e)
            _validate_response_date(f,day)
            f = f[f.ts_code.isin(codes)].copy()
            if len(f):
                f['source_day'] = day
                pieces.append(f[['ts_code','trade_date','close','source_day']])
        self.daily = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=['ts_code','trade_date','close','source_day'])
        self.daily['trade_date'] = pd.to_datetime(self.daily.trade_date.astype(str))
        self.daily = self.daily.sort_values(['ts_code','trade_date'])

    def frame(self, api, day):
        key = (api, pd.Timestamp(day).strftime('%Y%m%d'))
        if key not in self.frames:
            entry = self.entries.get(api, {}).get(key[1])
            self.frames[key] = self.store.read(entry) if entry else pd.DataFrame()
            _validate_response_date(self.frames[key],day)
        return self.frames[key]

    def row(self, api, day, code):
        f = self.frame(api, day)
        f1 = f[f.ts_code.eq(code)] if not f.empty and 'ts_code' in f else pd.DataFrame()
        if f1.empty:
            matches=self.store.find(api,trade_date=pd.Timestamp(day).strftime('%Y%m%d'),ts_code=code)
            for entry in reversed(matches):
                if not REQUIRED[api].issubset(set(entry['request']['body'].get('fields','').split(','))):continue
                f=self.store.read(entry);_validate_response_date(f,day);f1=f[f.ts_code.eq(code)]
                if not f1.empty:break
        if len(f1) > 1:
            raise ValueError(f'Duplicate {api} {day} {code}')
        return (f1.iloc[0], f.attrs.get('source_sha256')) if len(f1) else (None, f.attrs.get('source_sha256'))

    def last(self, code, day):
        f = self.daily[self.daily.ts_code.eq(code) & self.daily.trade_date.le(pd.Timestamp(day))]
        if f.empty:
            return None, None
        row = f.iloc[-1]
        entry = self.entries['daily'][row.source_day]
        source = self.store.read(entry).attrs['source_sha256']
        return row, source


def merge_gap_seeds(previous, current):
    """Keep repaired historical cases in all subsequent repair rebuilds."""
    if previous.empty:
        return current.copy()
    return pd.concat([previous,current],ignore_index=True).drop_duplicates(['ts_code','signal_date'],keep='first')


def verified_share_inputs(root, target):
    root,target=Path(root),Path(target)
    precise_shares={}; precise_source_files={}
    for path in sorted(target.glob('*_share_announcements.json')):
        precise_source_files[str(path.relative_to(root))]=sha(path)
        for record in json.loads(path.read_text()):
            if record.get('accepted') is not True:continue
            key=(record['ts_code'],record['trade_date'])
            if pd.Timestamp(record['ann_date'])>pd.Timestamp(key[1]) or pd.Timestamp(record['shares_date'])>pd.Timestamp(key[1]):
                raise ValueError('Future share announcement cannot value a historical date')
            if not record.get('path') or not record.get('sha256'):
                raise ValueError('Verified share announcement requires an original file and hash')
            if not np.isfinite(float(record['total_share'])) or float(record['total_share'])<=0:
                raise ValueError('Verified total share count must be finite and positive')
            if sha(root/record['path'])!=record['sha256']:
                raise ValueError('Share announcement PDF hash mismatch')
            if key in precise_shares and precise_shares[key]['total_share']!=record['total_share']:
                raise ValueError('Conflicting verified share counts')
            precise_shares[key]=record
    return precise_shares, precise_source_files


def build_repairs(root, online=False):
    """Create an exhaustive missing-date ledger plus independently usable lookup.

    Uses all month endpoints for every company with a capitalization gap, because
    repaired ranks may put formerly omitted issuers into the 1--500 universe.
    Strictly no values on/after a known delisting date are created by this module.
    """
    root = Path(root); target = root/'data/repair_sources/suspension'
    target.mkdir(parents=True, exist_ok=True)
    store = FrozenStore(target, sources=[*default_sources(), root/'data/raw/manifest.json'], online=online)
    universe = pd.read_csv(root/'data/prepared/universe.csv')
    missing = universe[universe.is_financial.eq(0) & universe.market_cap.isna()].copy()
    seed_path=target/'market_cap_gap_seeds.csv'
    previous=pd.read_csv(seed_path) if seed_path.exists() else pd.DataFrame()
    missing=merge_gap_seeds(previous,missing)
    missing.to_csv(seed_path,index=False)
    returns = pd.read_csv(root/'data/prepared/unresolved_returns.csv')
    codes = set(missing.ts_code) | set(returns.loc[returns.delisted_next.eq(0), 'stock_code'])
    if (target/'valuation_ledger.csv').exists():
        codes |= set(pd.read_csv(target/'valuation_ledger.csv').ts_code)
    securities = pd.read_csv(root/'data/prepared/securities.csv').set_index('ts_code')
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(root/'data/prepared/calendar.csv').trade_date))
    ends = pd.Series(calendar, index=calendar).groupby(calendar.to_period('M')).last()
    days = list(ends[(ends >= '2017-11-01') & (ends <= '2025-12-31')])
    notices = list(NOTICES)
    if (target/'notices.json').exists():
        external=json.loads((target/'notices.json').read_text())
        for notice in external:
            if notice.get('path') and sha(root/notice['path'])!=notice['sha256']:
                raise ValueError('Suspension notice PDF hash mismatch')
        notices=[n for n in notices if n['ts_code'] not in {x['ts_code'] for x in external}]+external
    share_candidates=pd.read_csv(target/'share_candidates.csv') if (target/'share_candidates.csv').exists() else pd.DataFrame()
    share_reviews=json.loads((target/'share_action_reviews.json').read_text()) if (target/'share_action_reviews.json').exists() else []
    share_reviews={(r['ts_code'],r['trade_date']):r for r in share_reviews if r.get('accepted') is True}
    factor_reviews=json.loads((target/'factor_action_reviews.json').read_text()) if (target/'factor_action_reviews.json').exists() else []
    factor_reviews={(r['ts_code'],r['trade_date']):r for r in factor_reviews if r.get('accepted') is True}
    precise_shares,precise_source_files=verified_share_inputs(root,target)
    try:
        history = History(store, codes)
        ledger = []
        for code in sorted(codes):
            sec = securities.loc[code]
            delist = pd.to_datetime(sec.delist_date, errors='coerce')
            for day in days:
                if day < pd.Timestamp(sec.list_date) or (pd.notna(delist) and day >= delist):
                    continue
                direct, direct_sha = history.row('daily', day, code)
                caprow, cap_sha = history.row('daily_basic', day, code)
                cap = np.nan if caprow is None else pd.to_numeric(caprow.get('total_mv'), errors='coerce') * 10000
                if not np.isfinite(cap) or cap<=0:cap=np.nan
                if direct is not None and np.isfinite(cap):
                    continue
                record = dict(ts_code=code, stock_code=code, trade_date=date_key(day),
                              status='unresolved', reason='', close=np.nan, adj_factor=np.nan, adj_factor_source='same_day_provider',
                              adjusted_price=np.nan, market_cap=np.nan, market_cap_lower=np.nan,
                              market_cap_upper=np.nan, market_cap_source='', valuation_source='',
                              last_trade_date='', last_close=np.nan, last_adj_factor=np.nan,
                              shares_date='', total_share_yi=np.nan, share_rounding_yi=.01, share_staleness_sessions=0, share_certification='', share_review_notes='',
                              suspension_source='', notice_url='', notice_ann_date='', source_sha256='')
                hashes = []
                last, last_sha = history.last(code, day)
                if last is None:
                    record['reason'] = 'no_prior_observed_trade'; ledger.append(record); continue
                hashes.append(last_sha)
                record.update(last_trade_date=date_key(last.trade_date), last_close=float(last.close))
                factorrow, factor_sha = history.row('adj_factor', day, code)
                prevfactor, prev_sha = history.row('adj_factor', last.trade_date, code)
                factor = np.nan if factorrow is None else pd.to_numeric(factorrow.get('adj_factor'), errors='coerce')
                last_factor = np.nan if prevfactor is None else pd.to_numeric(prevfactor.get('adj_factor'), errors='coerce')
                action_review=factor_reviews.get((code,date_key(day)))
                if not np.isfinite(factor) and action_review:
                    if pd.Timestamp(action_review['ann_date'])>day or date_key(last.trade_date)!=action_review['last_trade_date'] or not np.isclose(last_factor,action_review['last_adj_factor']):
                        raise ValueError('Factor continuation evidence does not match historical observation')
                    if sha(root/action_review['path'])!=action_review['sha256']:
                        raise ValueError('Factor action disclosure hash mismatch')
                    dividends=verified_raw_source(store,action_review['dividend_source_sha256'])
                    ex=pd.to_datetime(dividends.ex_date.astype('string').str.replace(r'\.0$', '',regex=True),format='%Y%m%d',errors='coerce')
                    changes=pd.to_numeric(dividends.stk_div,errors='coerce').fillna(0).ne(0)|pd.to_numeric(dividends.cash_div_tax,errors='coerce').fillna(0).ne(0)
                    if not dividends.ts_code.eq(code).all() or (ex.gt(last.trade_date)&ex.le(day)&changes).any():
                        raise ValueError('Factor continuation contradicted by corporate-action evidence')
                    factor=last_factor
                    record['adj_factor_source']='last_known_factor_verified_no_action_reconstruction'
                    hashes.extend([action_review['sha256'],action_review['dividend_source_sha256']])
                hashes.extend([factor_sha, prev_sha]); record.update(adj_factor=factor,last_adj_factor=last_factor)
                suspend, suspend_sha = history.row('suspend_d', day, code)
                whole = suspend is not None and suspend.get('suspend_type') == 'S' and (pd.isna(suspend.get('suspend_timing')) or suspend.get('suspend_timing') == '')
                notice = known_notice(code, day, last.trade_date, notices)
                if direct is not None:
                    mark = float(direct.close); source = 'daily_close'; hashes.append(direct_sha)
                elif whole:
                    mark = research_mark(last.close,last_factor,factor,confirmed=True)
                    source = 'last_adjusted_quote_confirmed_suspension'
                    record['suspension_source'] = 'suspend_d_full_session'; hashes.append(suspend_sha)
                elif notice:
                    mark = research_mark(last.close,last_factor,factor,confirmed=True)
                    source = 'last_adjusted_quote_published_suspension_notice'
                    record.update(suspension_source='published_notice',notice_url=notice['url'],notice_ann_date=notice['ann_date'])
                else:
                    mark = np.nan; source = ''
                reasons = []
                if not np.isfinite(mark):
                    reasons.append('suspension_not_confirmed' if direct is None and not (whole or notice) else 'adjustment_factor_missing')
                record.update(close=mark, adjusted_price=mark*factor, valuation_source=source)
                share, share_sha = history.row('bak_basic', day, code)
                if share is not None and pd.notna(share.get('total_share')) and float(share.total_share) > 0:
                    shares = float(share.total_share)
                    record.update(shares_date=date_key(day),total_share_yi=shares,share_certification='same_day_historical_snapshot')
                    hashes.append(share_sha)
                    approx, lo, hi = rounded_cap(mark, shares)
                    record.update(market_cap=approx,market_cap_lower=lo,market_cap_upper=hi,
                                  market_cap_source='historical_bak_basic_shares_x_research_mark')
                else:
                    key=(code,date_key(day)); review=share_reviews.get(key)
                    candidates=share_candidates[(share_candidates.ts_code==code)&(share_candidates.trade_date==key[1])] if not share_candidates.empty else pd.DataFrame()
                    if review and len(candidates)==1:
                        candidate=candidates.iloc[0]
                        verified_candidate_shares(store,candidate)
                        share_day=pd.Timestamp(candidate.shares_date)
                        previous_session=calendar[calendar<day][-1]
                        if candidate.staleness_sessions>1 or share_day not in {day,previous_session}:
                            raise ValueError('Share repair outside allowed prior-session window')
                        shares=float(candidate.total_share)/1e8
                        radius=float(candidate.rounding_share_units)
                        approx=mark*float(candidate.total_share)
                        lo=mark*max(0,float(candidate.total_share)-radius)
                        hi=mark*(float(candidate.total_share)+radius)
                        record.update(shares_date=candidate.shares_date,total_share_yi=shares,
                                      share_staleness_sessions=int(candidate.staleness_sessions),
                                      share_certification='last_known_observation_one_session_stale',
                                      share_review_notes=review.get('notes',''),market_cap=approx,
                                      market_cap_lower=lo,market_cap_upper=hi,
                                      market_cap_source='last_known_one_session_shares_x_research_mark')
                        hashes.append(candidate.source_sha256)
                    else:
                        reasons.append('same_day_historical_shares_missing')
                precise=precise_shares.get((code,date_key(day)))
                if precise:
                    count=float(precise['total_share']); radius=1.
                    record.update(shares_date=precise['shares_date'],total_share_yi=count/1e8,
                                  share_rounding_yi=radius/1e8,share_staleness_sessions=np.nan,
                                  share_certification='published_share_count_verified_pre_signal',
                                  share_review_notes=precise.get('notes',''),
                                  market_cap=mark*count,market_cap_lower=mark*max(0,count-radius),
                                  market_cap_upper=mark*(count+radius),
                                  market_cap_source='published_precise_share_count_x_research_mark')
                    hashes.append(precise.get('sha256')); reasons=[r for r in reasons if r!='same_day_historical_shares_missing']
                if np.isfinite(cap):
                    record.update(market_cap=float(cap),market_cap_lower=float(cap),market_cap_upper=float(cap),market_cap_source='daily_basic')
                    hashes.append(cap_sha)
                record['source_sha256'] = '|'.join(sorted(set(x for x in hashes if x)))
                record['status'] = 'repaired' if np.isfinite(record['market_cap']) and np.isfinite(record['adjusted_price']) else ('partial' if np.isfinite(record['adjusted_price']) else 'unresolved')
                record['reason'] = '|'.join(reasons)
                ledger.append(record)
            print('suspension audit:',code,flush=True)
        frame = pd.DataFrame(ledger)
        frame.to_csv(target/'valuation_ledger.csv',index=False)
        frame[frame.adjusted_price.notna() | frame.market_cap.notna()].to_csv(target/'valuation_lookup.csv',index=False)
        mv = missing.assign(trade_date=pd.to_datetime(missing.signal_date).dt.strftime('%Y-%m-%d')).merge(
            frame[['ts_code','trade_date','status','market_cap','reason']],on=['ts_code','trade_date'],how='left',suffixes=('_original','_repair'))
        mv.to_csv(target/'market_cap_gap_audit.csv',index=False)
        summary = dict(codes=len(codes),endpoint_audit_rows=len(frame),statuses=frame.status.value_counts().to_dict(),
                       original_mv_gaps=len(missing),mv_gaps_with_estimate=int(mv.market_cap_repair.notna().sum()),
                       remaining_mv_reasons=mv.loc[mv.market_cap_repair.isna(),'reason'].fillna('not_covered').value_counts().to_dict(),
                       valuation_policy='Carry last observed adjusted traded quote only with dated suspension evidence; not an execution or recovery price',
                       shares_policy='Verified pre-signal precise shares take priority; otherwise same-day historical bak_basic or reviewed one-session-old snapshot; source-specific precision bounds are retained',
                       share_certification_counts=frame.share_certification.fillna('').value_counts().to_dict(),
                       factor_source_counts=frame.adj_factor_source.fillna('').value_counts().to_dict(),
                       ledger_sha256=sha(target/'valuation_ledger.csv'),lookup_sha256=sha(target/'valuation_lookup.csv'),
                       gap_seeds_sha256=sha(seed_path),
                       notices_sha256=hashlib.sha256(canonical(notices)).hexdigest(),
                       share_candidates_sha256=sha(target/'share_candidates.csv') if (target/'share_candidates.csv').exists() else None,
                       share_reviews_sha256=sha(target/'share_action_reviews.json') if (target/'share_action_reviews.json').exists() else None,
                       precise_share_source_files=precise_source_files,
                       factor_action_reviews_sha256=sha(target/'factor_action_reviews.json') if (target/'factor_action_reviews.json').exists() else None)
        dump(target/'summary.json',summary)
        return frame, summary
    finally:
        store.save_manifest()


def load_valuations(root):
    path = Path(root)/'data/repair_sources/suspension/valuation_lookup.csv'
    if not path.exists():
        return pd.DataFrame()
    summary=json.loads((path.parent/'summary.json').read_text())
    if sha(path)!=summary['lookup_sha256']:
        raise ValueError('Suspension lookup hash mismatch')
    frame = pd.read_csv(path)
    if frame.duplicated(['ts_code','trade_date']).any():
        raise ValueError('Duplicate suspension repair keys')
    return frame


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--online',action='store_true')
    args = parser.parse_args()
    _, result = build_repairs(args.root,args.online)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def collect_notice_candidates(root):
    """Cache public CNINFO original notices for manual effective-date review.

    Search results are only candidates: they never enter the valuation rule until
    their actual text is checked and an explicit notices.json record is written.
    """
    import re
    import time
    import requests
    root=Path(root); target=root/'data/repair_sources/suspension'
    pub=target/'public_announcements'; pub.mkdir(parents=True,exist_ok=True)
    ledger=pd.read_csv(target/'market_cap_gap_audit.csv')
    counts=ledger.groupby('ts_code').size()
    need=ledger[ledger.reason.fillna('').str.contains('suspension_not_confirmed') | ledger.ts_code.isin(counts[counts>1].index)].groupby('ts_code').trade_date.min()
    session=requests.Session(); session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/'})
    records=[]
    def query(path,params):
        key=hashlib.sha256(canonical(dict(path=path,params=params))).hexdigest(); dest=pub/(key+'.json')
        if dest.exists():
            return json.loads(dest.read_text())['response']
        time.sleep(.4)
        response=session.post('https://www.cninfo.com.cn'+path,data=params,timeout=30)
        response.raise_for_status(); data=response.json()
        dump(dest,dict(url='https://www.cninfo.com.cn'+path,params=params,response=data))
        return data
    for code,end in need.items():
        listings=query('/new/information/topSearch/query',dict(keyWord=code[:6],maxSecNum=10,maxListNum=10))
        matches=[x for x in listings if x.get('code')==code[:6] and x.get('orgId')]
        if not matches:
            records.append(dict(ts_code=code,status='no_orgid')); continue
        org=matches[0]['orgId']
        data=query('/new/hisAnnouncement/query',dict(pageNum=1,pageSize=30,column='szse' if code.endswith('SZ') else 'sse',tabName='fulltext',stock=code[:6]+','+org,searchkey='暂停上市',seDate='2015-01-01~'+str(end),isHLtitle='false'))
        anns=data.get('announcements') or []
        def priority(a):
            title=re.sub('<[^>]+>','',a['announcementTitle'])
            actual=('暂停上市' in title and not any(x in title for x in ['债券','风险','可能','终止','提示','年报','年度','议案','决议','名单']))
            return (0 if actual and '进展' not in title else (1 if '进展' in title else 2),-a['announcementTime'])
        for ann in sorted(anns,key=priority)[:3]:
            url='https://static.cninfo.com.cn/'+ann['adjunctUrl']
            filename=code+'_'+Path(ann['adjunctUrl']).name
            path=pub/filename
            if not path.exists():
                time.sleep(.4); result=session.get(url,timeout=30); result.raise_for_status()
                if not result.content.startswith(b'%PDF'):
                    raise ValueError('Announcement is not a PDF')
                path.write_bytes(result.content)
            records.append(dict(ts_code=code,status='candidate_requires_text_review',title=re.sub('<[^>]+>','',ann['announcementTitle']),ann_date=ann['adjunctUrl'].split('/')[1],url=url,path=str(path.relative_to(root)),sha256=sha(path)))
        print('public notice candidates',code,len(anns),flush=True)
        dump(pub/'candidates.json',records)
    dump(pub/'candidates.json',records)
    return records


def acquire_share_sources(root):
    """Try stock-specific official basic records and one-session earlier shares.

    Prior-day records remain clearly labelled estimates; this acquisition does
    not assert a same-day share count and never carries data for longer.
    """
    root=Path(root); target=root/'data/repair_sources/suspension'
    store=FrozenStore(target,sources=[*default_sources(),root/'data/raw/manifest.json'],online=True)
    ledger=pd.read_csv(target/'valuation_ledger.csv')
    seed=pd.read_csv(target/'market_cap_gap_seeds.csv') if (target/'market_cap_gap_seeds.csv').exists() else pd.read_csv(root/'results/unresolved_universe.csv')
    cases=ledger[ledger.reason.fillna('').str.contains('shares_missing')]
    # There is one known all-market outage. Fetch its previous session once.
    records=[]; audit=[]
    try:
        prior=store.query('bak_basic',dict(trade_date='20210128',limit=7000,offset=0),'ts_code,trade_date,total_share')
        prior_sha=prior.attrs['source_sha256']
        if len(prior)>=7000: raise ValueError('Share snapshot pagination needed')
        for row in cases.itertuples(index=False):
            query=store.query('daily_basic',dict(ts_code=row.ts_code,trade_date=row.trade_date.replace('-','')),'ts_code,trade_date,close,total_mv,total_share')
            audit.append(dict(ts_code=row.ts_code,trade_date=row.trade_date,rows=len(query),source_sha256=query.attrs['source_sha256']))
            if len(query)>1: raise ValueError('Non-unique stock daily basic response')
            if len(query) and pd.notna(query.iloc[0].total_share) and query.iloc[0].total_share>0:
                q=query.iloc[0]
                records.append(dict(ts_code=row.ts_code,trade_date=row.trade_date,shares_date=row.trade_date,
                                    total_share=float(q.total_share)*10000,source='stock_specific_daily_basic',
                                    staleness_sessions=0,rounding_share_units=1.,source_sha256=query.attrs['source_sha256']))
            elif row.trade_date=='2021-01-29':
                p=prior[prior.ts_code.eq(row.ts_code)]
                if len(p)==1 and pd.notna(p.iloc[0].total_share) and p.iloc[0].total_share>0:
                    records.append(dict(ts_code=row.ts_code,trade_date=row.trade_date,shares_date='2021-01-28',
                                        total_share=float(p.iloc[0].total_share)*1e8,
                                        source='last_known_bak_basic_one_session_stale',staleness_sessions=1,
                                        rounding_share_units=1e6,source_sha256=prior_sha))
            print('share data query',row.ts_code,row.trade_date,len(query),flush=True)
            pd.DataFrame(records).to_csv(target/'share_candidates.csv',index=False)
            pd.DataFrame(audit).to_csv(target/'share_query_audit.csv',index=False)
            store.save_manifest()
    finally:
        store.save_manifest()
    return pd.DataFrame(records)


def collect_share_action_candidates(root):
    """Freeze pre-signal public announcements for one-day share carry review."""
    import re
    import time
    import requests
    root=Path(root); target=root/'data/repair_sources/suspension';pub=target/'public_share_actions';pub.mkdir(parents=True,exist_ok=True)
    shares=pd.read_csv(target/'share_candidates.csv');records=[]
    session=requests.Session();session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/'})
    def query(path,params):
        key=hashlib.sha256(canonical(dict(path=path,params=params))).hexdigest();dest=pub/(key+'.json')
        if dest.exists():return json.loads(dest.read_text())['response'],str(dest.relative_to(root)),sha(dest)
        time.sleep(.4);r=session.post('https://www.cninfo.com.cn'+path,data=params,timeout=30);r.raise_for_status();data=r.json()
        dump(dest,dict(url='https://www.cninfo.com.cn'+path,params=params,response=data));return data,str(dest.relative_to(root)),sha(dest)
    for row in shares.itertuples(index=False):
        listing,_,_=query('/new/information/topSearch/query',dict(keyWord=row.ts_code[:6],maxSecNum=10,maxListNum=10))
        match=[x for x in listing if x.get('code')==row.ts_code[:6] and x.get('orgId')]
        if not match:raise ValueError('Missing historical company disclosure identifier')
        hashes=[];files=[]
        def fetch_range(start,end):
            data,path,digest=query('/new/hisAnnouncement/query',dict(pageNum=1,pageSize=30,column='szse' if row.ts_code.endswith('SZ') else 'sse',tabName='fulltext',stock=row.ts_code[:6]+','+match[0]['orgId'],searchkey='',seDate=start+'~'+end,isHLtitle='false'))
            hashes.append(digest);files.append(path);ann=data.get('announcements') or []
            count=int(data.get('totalAnnouncement',0))
            if len({a['announcementId'] for a in ann})==count:return ann
            # This public endpoint repeated page 1 for later pageNum requests.
            # Narrow dates instead; validate counts for each terminal partition.
            if start==end:raise ValueError('One-day announcement pagination unresolved')
            lo,hi=pd.Timestamp(start),pd.Timestamp(end);mid=lo+(hi-lo)//2
            return fetch_range(start,mid.strftime('%Y-%m-%d'))+fetch_range((mid+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),end)
        anns=fetch_range('2020-10-01','2021-01-29')
        anns=list({a['announcementId']:a for a in anns}.values())
        selected=[a for a in anns if re.search('总股本|资本公积|转增|权益分派|发行股份|非公开发行|公开增发|配股|可转换|可转债|回购注销|注销部分|转股|配售',a['announcementTitle'])]
        metas=[]
        for ann in selected:
            url='https://static.cninfo.com.cn/'+ann['adjunctUrl'];dest=pub/(row.ts_code+'_'+Path(ann['adjunctUrl']).name)
            if not dest.exists():
                time.sleep(.4);r=session.get(url,timeout=30);r.raise_for_status();dest.write_bytes(r.content)
            metas.append(dict(title=re.sub('<[^>]+>','',ann['announcementTitle']),ann_date=ann['adjunctUrl'].split('/')[1],url=url,path=str(dest.relative_to(root)),sha256=sha(dest)))
        record=dict(ts_code=row.ts_code,trade_date=row.trade_date,shares_date=row.shares_date,query_start='2020-10-01',query_end='2021-01-29',announcements=len(anns),query_files=files,query_sha256=hashes,potential_share_actions=metas)
        records.append(record);dump(pub/'candidates.json',records)
        print('share action review candidates',row.ts_code,len(anns),[a['title'] for a in metas],flush=True)
    return records


def collect_factor_action_evidence(root):
    """Collect original no-action evidence for the three Shanghai Putian gaps."""
    import re
    import time
    import requests
    root=Path(root);target=root/'data/repair_sources/suspension';pub=target/'factor_action_evidence';pub.mkdir(parents=True,exist_ok=True)
    store=FrozenStore(target,sources=[*default_sources(),root/'data/raw/manifest.json'],online=True)
    try:
        f=store.query('dividend',dict(ts_code='600680.SH',limit=6000,offset=0),'ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,ex_date,imp_ann_date')
        f.to_csv(pub/'dividend.csv',index=False)
        if len(f)>=6000:raise ValueError('Dividend pagination required')
        dump(pub/'dividend_source.json',dict(source_sha256=f.attrs['source_sha256'],rows=len(f)))
    finally:store.save_manifest()
    session=requests.Session();session.headers.update({'User-Agent':'Mozilla/5.0','Referer':'https://www.cninfo.com.cn/'})
    def query(path,params):
        dest=pub/(hashlib.sha256(canonical(dict(path=path,params=params))).hexdigest()+'.json')
        if dest.exists():return json.loads(dest.read_text())['response']
        time.sleep(.4);r=session.post('https://www.cninfo.com.cn'+path,data=params,timeout=30);r.raise_for_status();data=r.json();dump(dest,dict(url='https://www.cninfo.com.cn'+path,params=params,response=data));return data
    listings=query('/new/information/topSearch/query',dict(keyWord='600680',maxSecNum=10,maxListNum=10))
    org=next(x['orgId'] for x in listings if x.get('code')=='600680')
    def gather(start,end):
        data=query('/new/hisAnnouncement/query',dict(pageNum=1,pageSize=30,column='sse',tabName='fulltext',stock='600680,'+org,searchkey='董事会',seDate=start+'~'+end,isHLtitle='false'))
        anns=data.get('announcements') or []
        if len({a['announcementId'] for a in anns})==int(data.get('totalAnnouncement',0)):return anns
        if start==end:raise ValueError('One-day announcement pagination unresolved')
        lo,hi=pd.Timestamp(start),pd.Timestamp(end);mid=lo+(hi-lo)//2
        return gather(start,mid.strftime('%Y-%m-%d'))+gather((mid+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),end)
    records=[];anns=gather('2018-04-27','2018-07-31')
    for ann in anns:
        title=re.sub('<[^>]+>','',ann['announcementTitle'])
        if not re.search('总股本|董事会|年度股东大会|资本公积|转增|权益分派|利润分配|发行股份|非公开发行|公开增发|配股|可转换|可转债|回购注销|注销部分|转股|配售',title):continue
        url='https://static.cninfo.com.cn/'+ann['adjunctUrl'];dest=pub/Path(ann['adjunctUrl']).name
        if not dest.exists():
            time.sleep(.4);r=session.get(url,timeout=30);r.raise_for_status();dest.write_bytes(r.content)
        records.append(dict(title=title,ann_date=ann['adjunctUrl'].split('/')[1],url=url,path=str(dest.relative_to(root)),sha256=sha(dest)))
    dump(pub/'candidates.json',dict(announcements=len(anns),selected=records))
    return records
