"""Frozen daily inputs for capacity scenarios, separate from the original data.

Tushare daily amount is thousand CNY and vol is 100-share lots.  Only lagged
statistics may inform orders.  Same-day volume/VWAP are execution outcomes,
not information available when those orders are planned.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import threading
import time

import numpy as np
import pandas as pd
import requests

FIELDS = 'ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
LIMIT = 6000
DOCUMENTATION = 'https://tushare.pro/document/2?doc_id=27'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value)); temporary.replace(path)


def normalize_codes(frame, day, aliases):
    out = frame.copy()
    for alias in aliases:
        if pd.Timestamp(day) < pd.Timestamp(alias['effective_date']):
            out.loc[out.ts_code.eq(alias['old']), 'ts_code'] = alias['canonical']
    if 'suspend_type' in out:
        # Multiple intraday suspension intervals for one security are valid.
        return out
    duplicate = out[out.ts_code.duplicated(False)]
    if len(duplicate):
        for code, group in duplicate.groupby('ts_code'):
            if len(group.drop_duplicates()) > 1:
                raise ValueError(f'Conflicting alias rows: {day} {code}')
        out = out.drop_duplicates('ts_code')
    return out


def read_response(path, expected_sha=None):
    data = Path(path).read_bytes()
    if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Frozen input file hash mismatch')
    obj = json.loads(data)
    response = obj['response']
    if response.get('code') != 0:
        raise ValueError('Frozen provider response is unsuccessful')
    if hashlib.sha256(canonical(response)).hexdigest() != obj['response_sha256']:
        raise ValueError('Frozen response hash mismatch')
    return obj, pd.DataFrame(response['data']['items'], columns=response['data']['fields'])


def lagged_statistics(amount_yuan, close, factor, whole_day_suspended, window=20):
    """Market-session windows ending at t-1; unknown observations remain unknown."""
    amount = np.asarray(amount_yuan, dtype=float).copy()
    close = np.asarray(close, dtype=float)
    factor = np.asarray(factor, dtype=float)
    whole = np.asarray(whole_day_suspended, dtype=bool)
    if amount.shape != close.shape or close.shape != factor.shape or whole.shape != close.shape:
        raise ValueError('Capacity input shapes differ')
    if np.any(np.isfinite(amount) & (amount < 0)):
        raise ValueError('Negative turnover amount')
    known_zero = ~np.isfinite(amount) & whole & ~np.isfinite(close)
    amount[known_zero] = 0.0
    price = close * factor
    price[~np.isfinite(price) | (price <= 0)] = np.nan
    for t in range(1, len(price)):
        # Carry only across a documented full suspension, never across unknown gaps.
        carry = whole[t] & ~np.isfinite(close[t]) & np.isfinite(price[t-1])
        price[t, carry] = price[t-1, carry]
    returns = np.full_like(price, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        returns[1:] = price[1:] / price[:-1] - 1
    adv = pd.DataFrame(amount).rolling(window, min_periods=window).mean().shift(1).to_numpy()
    sigma = pd.DataFrame(returns).rolling(window, min_periods=window).std(ddof=1).shift(1).to_numpy()
    return amount, adv, sigma, known_zero


def vwap_range_mask(frame):
    vwap = frame.amount.to_numpy(float) * 10 / frame.vol.to_numpy(float)
    tolerance = np.maximum(.011, np.abs(vwap) * 1e-5)
    bad = (vwap < frame.low.to_numpy() - tolerance) | (vwap > frame.high.to_numpy() + tolerance)
    return vwap, tolerance, bad


def validate_new_daily(frame, day, limit=LIMIT, strict_vwap=True):
    required = set(FIELDS.split(','))
    if not required <= set(frame):
        raise ValueError('Supplemental daily response lacks required fields')
    if len(frame) >= limit:
        raise ValueError(f'Pagination required for daily {day}; response reached {limit}')
    if frame.empty or frame.ts_code.duplicated().any():
        raise ValueError('Empty or duplicated daily response')
    if not frame.trade_date.astype(str).eq(str(day)).all():
        raise ValueError('Supplemental daily response has wrong dates')
    # pre_close is retained as provider evidence but is not an input here:
    # lagged returns use the frozen adjusted close matrix, never this field.
    # Some historical first-quote rows have no pre_close.
    for col in ['open', 'high', 'low', 'close', 'vol', 'amount']:
        if not np.isfinite(pd.to_numeric(frame[col], errors='coerce')).all():
            raise ValueError(f'Non-finite supplemental daily {col}')
    if frame[['open','high','low','close','vol']].le(0).any().any() or frame.amount.lt(0).any():
        raise ValueError('Non-positive price/volume or negative amount')
    if (frame.low > frame.high).any():
        raise ValueError('Daily low exceeds high')
    vwap, tolerance, bad = vwap_range_mask(frame)
    if strict_vwap and bad.any():
        raise ValueError(f'VWAP outside high/low for {day}: {int(bad.sum())} rows')
    return vwap


class ProviderError(RuntimeError):
    """Safe exception: never includes provider messages or credentials."""


class StartLimiter:
    def __init__(self, seconds=1.3, standard_access_verified=False):
        minimum = .35 if standard_access_verified else 1.25
        if seconds < minimum:
            raise ValueError('Request interval exceeds the verified provider frequency tier')
        self.seconds = seconds; self.lock = threading.Lock(); self.next_start = 0.

    def wait(self):
        with self.lock:
            delay = max(0., self.next_start - time.monotonic())
            if delay:
                time.sleep(delay)
            self.next_start = time.monotonic() + self.seconds


def fetch_days(data_root, days, workers=4, interval=1.3):
    """Resume immutable per-day files; globally throttle all worker starts."""
    if not 1 <= workers <= 4:
        raise ValueError('Use between 1 and 4 workers')
    target = Path(data_root) / 'data/capacity'
    raw = target / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    manifest_path = target / 'acquisition_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'provider': 'tushare', 'api': 'daily', 'source_documentation': DOCUMENTATION,
        'fields': FIELDS, 'units': {'amount':'thousand CNY','vol':'100-share lots','prices':'unadjusted CNY/share'},
        'request_limit': LIMIT, 'inputs': {},
    }
    existing = manifest['inputs']; todo = []
    for day in days:
        day = str(day); path = raw / f'{day}.json'
        if path.exists():
            obj, frame = read_response(path, existing.get(day, {}).get('sha256'))
            validate_new_daily(frame, day, strict_vwap=False)
            if obj['request']['body'] != {'api_name':'daily','params':{'trade_date':day,'limit':LIMIT,'offset':0},'fields':FIELDS}:
                raise ValueError('Cached supplemental query differs from protocol')
            existing[day] = {'path':str(path.relative_to(target)), 'sha256':sha(path),
                             'response_sha256':obj['response_sha256'], 'n_rows':len(frame),
                             'acquired_at':obj['acquired_at']}
        else:
            todo.append(day)
    atomic_json(manifest_path, manifest)
    print(f'Capacity daily cache: {len(days)-len(todo)}/{len(days)} ready; {len(todo)} requests remaining', flush=True)
    if not todo:
        return manifest
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        raise ProviderError('TUSHARE_TOKEN is unavailable')
    # Official daily_basic access requires >=2,000 points, whose standard
    # interface tier permits 200/min. Confirm current access before allowing
    # faster than the default <=48/min; keep headroom at <=172/min.
    # A denied probe or quota response stops rather than evading the limit.
    standard_access_verified = False
    if interval < 1.25:
        if interval < .35:
            raise ValueError('Even verified standard access is capped at <=172 starts/min here')
        probe_body = {'api_name':'daily_basic',
                      'params':{'ts_code':'000001.SZ','trade_date':'20251231','limit':1},
                      'fields':'ts_code,trade_date,total_mv'}
        try:
            probe_response = requests.post('https://api.tushare.pro',json={**probe_body,'token':token},timeout=35)
            probe = probe_response.json()
        except (requests.RequestException,ValueError):
            raise ProviderError('Frequency-tier permission check failed') from None
        if probe_response.status_code != 200 or probe.get('code') != 0 or not probe.get('data',{}).get('items'):
            raise ProviderError('Current daily_basic access not confirmed; do not increase request frequency')
        standard_access_verified = True
        atomic_json(target/'frequency_evidence.json',{
            'acquired_at':datetime.now(timezone.utc).isoformat(),'request':probe_body,'response':probe,
            'scope':'Permission check only; not a strategy input',
            'official_docs':['https://tushare.pro/document/1?doc_id=290','https://tushare.pro/document/2?doc_id=32'],
            'request_interval_seconds':interval})
        print('Current daily_basic access verified; using documented standard tier with request headroom',flush=True)
    limiter = StartLimiter(interval,standard_access_verified=standard_access_verified)
    stop = threading.Event()

    def acquire(day):
        body = {'api_name':'daily','params':{'trade_date':day,'limit':LIMIT,'offset':0},'fields':FIELDS}
        for attempt in range(3):
            if stop.is_set():
                raise ProviderError('Acquisition stopped after an earlier failure')
            limiter.wait()
            try:
                response = requests.post('https://api.tushare.pro', json={**body,'token':token}, timeout=35)
                if response.status_code != 200:
                    if response.status_code in [429,500,502,503,504] and attempt < 2:
                        time.sleep(2 ** (attempt + 1)); continue
                    raise ProviderError(f'Provider HTTP status {response.status_code}')
                result = response.json()
            except (requests.RequestException, ValueError) as error:
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1)); continue
                raise ProviderError(f'Network/JSON failure {type(error).__name__}') from None
            if result.get('code') != 0:
                # Permissions and provider limits do not trigger evasive retries.
                raise ProviderError(f'Provider rejected daily request with code {result.get("code")}')
            frame = pd.DataFrame(result['data']['items'], columns=result['data']['fields'])
            obj = {'acquired_at':datetime.now(timezone.utc).isoformat(),
                   'request':{'provider':'tushare','method':'POST','url':'https://api.tushare.pro','body':body},
                   'response':result,'response_sha256':hashlib.sha256(canonical(result)).hexdigest()}
            path = raw / f'{day}.json'
            if path.exists():
                raise ValueError('Refusing to overwrite a frozen capacity response')
            atomic_json(path, obj)
            # Freeze successful provider evidence before business validation.
            validate_new_daily(frame, day, strict_vwap=False)
            bad_count = int(vwap_range_mask(frame)[2].sum())
            if bad_count:
                print(f'VWAP range audit flags {day}: {bad_count} rows; raw evidence preserved', flush=True)
            return day, {'path':str(path.relative_to(target)), 'sha256':sha(path),
                         'response_sha256':obj['response_sha256'], 'n_rows':len(frame),'acquired_at':obj['acquired_at']}
        raise ProviderError('Retry budget exhausted')

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(acquire, day):day for day in todo}
        try:
            for future in as_completed(futures):
                day, entry = future.result(); existing[day] = entry; completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    atomic_json(manifest_path, manifest)
                    print(f'Capacity daily completed {len(days)-len(todo)+completed}/{len(days)}; latest {day}', flush=True)
        except Exception:
            stop.set()
            for future in futures: future.cancel()
            atomic_json(manifest_path, manifest)
            raise
    return manifest


def build_inputs(data_root):
    root = Path(data_root); target = root/'data/capacity'; target.mkdir(parents=True, exist_ok=True)
    with np.load(root/'data/prepared/daily_market.npz', allow_pickle=False) as z:
        market = {k:z[k] for k in z.files}
    config = json.loads((root/'config.json').read_text())
    aliases = config.get('security_aliases', [])
    dates = pd.DatetimeIndex(market['dates']); codes = list(market['codes'])
    ri = {day.strftime('%Y%m%d'):i for i,day in enumerate(dates)}
    ci = {code:i for i,code in enumerate(codes)}
    shape = market['open'].shape
    amount = np.full(shape, np.nan); whole = np.zeros(shape, bool)
    old_entries = json.loads((root/'data/raw/manifest.json').read_text())['inputs']
    chosen = {}
    for entry in old_entries:
        b = entry['request']['body']; api = b['api_name']; day = b.get('params',{}).get('trade_date')
        if day not in ri or api not in ('daily','suspend_d'): continue
        if int(b.get('params',{}).get('offset',0)) != 0: continue
        if api == 'daily' and 'amount' not in b.get('fields','').split(','): continue
        key = (api,day)
        if key not in chosen or (entry.get('acquired_at',''),entry['path']) > (chosen[key].get('acquired_at',''),chosen[key]['path']):chosen[key] = entry
    missing = sorted(set(ri)-{day for api,day in chosen if api=='daily'})
    if missing: raise ValueError(f'Frozen daily turnover dates missing: {len(missing)}')
    for (api,day),entry in chosen.items():
        _, f = read_response(root/entry['path'], entry['sha256'])
        if len(f) >= int(entry['request']['body'].get('params',{}).get('limit',LIMIT)):
            raise ValueError('Frozen source requires pagination')
        f = normalize_codes(f,day,aliases); f=f[f.ts_code.isin(ci)]
        ix = f.ts_code.map(ci).to_numpy(int); t=ri[day]
        if api == 'daily':
            amount[t,ix] = pd.to_numeric(f.amount,errors='coerce').to_numpy()*1000
        else:
            full = f.suspend_type.eq('S') & (f.suspend_timing.isna()|f.suspend_timing.eq(''))
            whole[t, f.loc[full,'ts_code'].map(ci).to_numpy(int)] = True
    amount, adv, sigma, known_zero = lagged_statistics(amount,market['close'],market['factor'],whole)
    arrays = {'dates':market['dates'],'codes':market['codes'],'capacity_amount_yuan':amount,
              'capacity_adv20':adv,'capacity_sigma20':sigma,'capacity_full_suspension':whole}
    supplementary = json.loads((target/'acquisition_manifest.json').read_text())
    if set(ri)-set(supplementary['inputs']):
        raise ValueError('Supplemental daily acquisition is incomplete; refuse a final NPZ')
    for key in ['capacity_volume_shares','capacity_high','capacity_low']:
        arrays[key] = np.full(shape,np.nan)
    changes=[];new_rows=0;vwap_anomalies=[];unused_preclose_missing=0
    for day,t in ri.items():
        entry=supplementary['inputs'][day]; _,f=read_response(target/entry['path'],entry['sha256'])
        validate_new_daily(f,day,strict_vwap=False);f=normalize_codes(f,day,aliases)
        unused_preclose_missing += int((~np.isfinite(pd.to_numeric(f.pre_close,errors='coerce'))).sum())
        vwap,tolerance,bad=vwap_range_mask(f)
        for k in np.flatnonzero(bad):
            row=f.iloc[k]
            vwap_anomalies.append({'date':day,'stock_code':row.ts_code,'in_market':row.ts_code in ci,
                                  'open':row.open,'high':row.high,'low':row.low,'close':row.close,
                                  'vol_lots':row.vol,'amount_thousand_yuan':row.amount,'vwap':vwap[k],
                                  'tolerance':tolerance[k],
                                  'outside_range':max(row.low-vwap[k],vwap[k]-row.high,0.)})
        invalid_codes=set(f.loc[bad,'ts_code'])
        f=f[f.ts_code.isin(ci)]
        ix=f.ts_code.map(ci).to_numpy(int);new_rows+=len(f)
        arrays['capacity_volume_shares'][t,ix]=f.vol.to_numpy(float)*100
        arrays['capacity_high'][t,ix]=f.high.to_numpy(float)
        arrays['capacity_low'][t,ix]=f.low.to_numpy(float)
        # Preserve raw response and legacy marks. Missing execution volume makes
        # the scenario explicitly block this day, never clip the bad VWAP.
        for code in invalid_codes & set(ci):
            arrays['capacity_volume_shares'][t,ci[code]]=np.nan
        for field,old,new in [('open',market['open'][t,ix],f.open.to_numpy(float)),
                              ('close',market['close'][t,ix],f.close.to_numpy(float)),
                              ('amount_yuan',amount[t,ix],f.amount.to_numpy(float)*1000)]:
            changed=~np.isclose(old,new,rtol=1e-10,atol=1e-6,equal_nan=True)
            for k in np.flatnonzero(changed):
                changes.append({'date':day,'stock_code':f.ts_code.iloc[k],'field':field,
                                'frozen_value':old[k],'new_value':new[k],
                                'difference':new[k]-old[k] if np.isfinite(old[k]) else None})
    # Retain frozen amount for ADV and execution VWAP. Revisions are logged and
    # block certification; without revisions it is coherent with the new volume.
    pd.DataFrame(changes,columns=['date','stock_code','field','frozen_value','new_value','difference']).to_csv(target/'source_differences.csv',index=False)
    pd.DataFrame(vwap_anomalies,columns=['date','stock_code','in_market','open','high','low','close','vol_lots','amount_thousand_yuan','vwap','tolerance','outside_range']).to_csv(target/'vwap_anomalies.csv',index=False)
    coverage={'vwap_anomalies_all':len(vwap_anomalies),
              'unused_provider_preclose_missing_all':unused_preclose_missing,
              'provider_preclose_policy':'raw pre_close retained but unused; all lagged returns use the original adjusted close matrix',
              'vwap_anomalies_market':sum(x['in_market'] for x in vwap_anomalies),
              'vwap_anomaly_policy':'preserve raw evidence and legacy amount/prices; relevant anomalous execution volume is NA and must block fills; no clipping',
              'missing_volume_on_quoted_market_rows':int((np.isfinite(market['open']) & ~np.isfinite(arrays['capacity_volume_shares'])).sum()),
              'n_dates':len(dates),'n_stocks':len(codes),'new_daily_rows':new_rows,
              'old_new_differences':len(changes),'known_zero_amount_sessions':int(known_zero.sum()),
              'adv_unknown_all':int((~np.isfinite(adv)).sum()),'adv_zero_all':int((adv==0).sum()),
              'sigma_unknown_all':int((~np.isfinite(sigma)).sum()),
              'sigma_policy':'20 market-session adjusted simple returns, sample standard deviation, shifted one session; only evidenced full suspensions carry prior adjusted mark',
              'suspension_policy':'frozen suspend_d type S and blank timing, absent quote; unknown gaps remain missing',
              'source_difference_policy':'all differences are logged; final build fails on any overlapping revision',
              'vwap_validation':'amount*10/vol within low/high allowing max(0.011 CNY, 1e-5*VWAP) rounding tolerance'}
    for filename in ['order_plans.csv','trades.csv']:
        q=pd.read_csv(root/'results'/filename,usecols=['stock_code','execute_date','strategy'])
        q=q[q.strategy.eq(json.loads((root/'results/selection.json').read_text())['winner'])]
        ti=dates.get_indexer(pd.to_datetime(q.execute_date));ji=pd.Index(codes).get_indexer(q.stock_code)
        if min(ti)<0 or min(ji)<0:raise ValueError('Recorded order outside input panel')
        av=adv[ti,ji];sv=sigma[ti,ji]
        bad_pairs={(x['date'],x['stock_code']) for x in vwap_anomalies if x['in_market']}
        affected=sum((day,code) in bad_pairs for day,code in zip(pd.to_datetime(q.execute_date).dt.strftime('%Y%m%d'),q.stock_code))
        coverage[filename]={'n':len(q),'adv_unknown':int((~np.isfinite(av)).sum()),
                            'adv_zero':int((av==0).sum()),'sigma_unknown':int((~np.isfinite(sv)).sum()),
                            'sigma_zero':int((sv==0).sum()),'original_rows_on_vwap_anomaly':affected,
                            'scope':'original frozen winner records; scenario-specific attempted/blocked orders require replay diagnostics'}
    atomic_json(target/'coverage.json',coverage)
    if changes:raise ValueError(f'{len(changes)} old/new daily differences require review before combining fields')
    out=target/'capacity_market.npz';tmp=target/'capacity_market.tmp.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(out)
    sources={str(Path('data/raw/manifest.json')):sha(root/'data/raw/manifest.json'),
             'data/prepared/daily_market.npz':sha(root/'data/prepared/daily_market.npz'),
             'config.json':sha(root/'config.json'),'results/selection.json':sha(root/'results/selection.json'),
             'data/capacity/acquisition_manifest.json':sha(target/'acquisition_manifest.json')}
    atomic_json(target/'input_manifest.json',{'sources':sources,'capacity_market_sha256':sha(out),
                'original_inputs_changed':False,'coverage':coverage})
    print(json.dumps(coverage,ensure_ascii=False),flush=True)
    return coverage
