"""Causal risk-name evidence from name intervals and observed historical names.

The opening matrix uses evidence available strictly before the session. The
signal matrix includes information available by that day's close. Historical
snapshots fill namechange gaps only forward; current security names are unused.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .acquire import dump, sha
from .industry_repair import canonical_code, official_classifications


def risk_name(name):
    if pd.isna(name):
        return None
    text = str(name).strip().upper()
    return None if not text else int('ST' in text or 'PT' in text or '退' in text)


def dates_of(values):
    return pd.to_datetime(values.astype('string').str.replace(r'\.0$', '', regex=True),
                          format='mixed', errors='coerce')


def snapshot_evidence(store, codes, config):
    """Return only dated names; source date is distinct from availability date."""
    codes = set(codes)
    frames = []
    for api in ['bak_basic', 'bak_daily']:
        by_day = {}
        for entry in store.find(api):
            body = entry['request']['body']; params = body.get('params', {})
            if params.get('ts_code') or not params.get('trade_date'):
                continue
            if not {'ts_code', 'trade_date', 'name'}.issubset(body.get('fields', '').split(',')):
                continue
            by_day[str(params['trade_date'])] = entry
        for requested, entry in sorted(by_day.items()):
            frame = store.read(entry)
            limit = int(entry['request']['body']['params'].get('limit', 7000))
            if len(frame) >= limit:
                raise ValueError(f'Historical name snapshot may be truncated: {api} {requested}')
            if frame.empty:
                continue
            frame = frame[['ts_code', 'trade_date', 'name']].copy()
            frame['observation_date'] = dates_of(frame.trade_date)
            if not frame.observation_date.eq(pd.Timestamp(requested)).all():
                raise ValueError('Historical name response date does not match request')
            frame['ts_code'] = frame.ts_code.map(lambda x: canonical_code(x, config))
            frame = frame[frame.ts_code.isin(codes)]
            frame['availability_date'] = frame.observation_date
            frame['source'] = api
            frame['source_sha256'] = entry['sha256']
            frames.append(frame.drop(columns='trade_date'))
    official = official_classifications(store.root)
    if len(official):
        frame = official[['ts_code', 'name', 'observation_date', 'publication_date', 'source_sha256']].copy()
        frame['ts_code'] = frame.ts_code.map(lambda x: canonical_code(x, config))
        frame = frame[frame.ts_code.isin(codes)]
        frame = frame.rename(columns={'publication_date': 'availability_date'})
        frame['source'] = 'published_CSRC_quarterly'
        frames.append(frame)
    columns = ['ts_code', 'name', 'observation_date', 'availability_date', 'source', 'source_sha256', 'risk']
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    frame['risk'] = frame.name.map(risk_name)
    frame = frame[frame.risk.notna()].copy()
    frame['risk'] = frame.risk.astype(int)
    return frame[columns].drop_duplicates()


def _interval_states(codes, dates, names, config, close=False):
    index = {code: j for j, code in enumerate(codes)}
    shape = len(dates), len(codes)
    state = np.full(shape, -1, dtype=np.int8)
    risk = np.zeros(shape, dtype=bool)
    count = np.zeros(shape, dtype=np.int16)
    effective = np.full(shape, np.datetime64('NaT', 'D'), dtype='datetime64[D]')
    for row in names.itertuples():
        code = canonical_code(row.ts_code, config)
        if code not in index or pd.isna(row.start_date) or pd.isna(row.ann_date):
            continue
        value = risk_name(row.name)
        if value is None:
            continue
        mask = (dates >= row.start_date) & ((dates >= row.ann_date) if close else (dates > row.ann_date))
        if pd.notna(row.end_date):
            mask &= dates <= row.end_date
        j = index[code]
        prior = state[mask, j].copy(); n = count[mask, j]
        state[mask, j] = np.where(n == 0, value, np.where(prior == value, value, -1))
        risk[mask, j] |= bool(value)
        effective[mask, j] = np.datetime64(row.start_date, 'D')
        count[mask, j] += 1
    return state, risk, effective, count > 1


def _merge_snapshots(base, risk, effective, dates, codes, evidence, config, close=False):
    state = base.copy(); known_risk = risk.copy()
    n = len(codes); index = {code: j for j, code in enumerate(codes)}
    observed = np.full(n, np.datetime64('NaT', 'D'), dtype='datetime64[D]')
    anchor = np.full(n, -1, dtype=np.int8)
    anchor_conflict = np.zeros(n, dtype=bool)
    age = np.full(state.shape, -1, dtype=np.int16)
    filled = np.zeros(state.shape, bool); conflict = np.zeros(state.shape, bool)
    max_age = int(config.get('repair_policy', {}).get('status_snapshot_max_age_days', 185))
    groups = evidence.groupby('availability_date', sort=True) if len(evidence) else []
    events = iter(groups); pending = next(events, None)
    for t, day in enumerate(dates):
        while pending is not None and (pending[0] <= day if close else pending[0] < day):
            _, batch = pending
            # Combine competing sources on one observed date. A risk observation
            # remains a buy blocker even if another source calls it ordinary.
            combined = batch.groupby(['ts_code', 'observation_date'], sort=True).risk.agg(['min', 'max']).reset_index()
            # A source can publish several observation dates on one day. Apply
            # them in observation order without a costly per-stock DataFrame.
            for source_day, rows in combined.groupby('observation_date', sort=True):
                source_day = np.datetime64(source_day, 'D')
                j = rows.ts_code.map(index).to_numpy(int)
                high = rows['max'].to_numpy(np.int8)
                different = rows['min'].to_numpy() != high
                newer = np.isnat(observed[j]) | (source_day > observed[j])
                fresh = j[newer]
                observed[fresh] = source_day; anchor[fresh] = high[newer]
                anchor_conflict[fresh] = different[newer]
                equal = ~newer & (source_day == observed[j])
                same = j[equal]
                anchor_conflict[same] |= different[equal] | (anchor[same] != high[equal])
                anchor[same] = np.maximum(anchor[same], high[equal])
            pending = next(events, None)
        elapsed = (np.datetime64(day, 'D') - observed).astype('timedelta64[D]').astype(np.int64)
        valid = (anchor >= 0) & ~np.isnat(observed) & (elapsed >= 0) & (elapsed <= max_age)
        age[t, valid] = elapsed[valid]
        gap = (base[t] == -1) & ~risk[t] & valid
        state[t, gap] = anchor[gap]; filled[t, gap] = True
        known_risk[t, gap] = anchor[gap] == 1
        contradictory_anchor = gap & anchor_conflict
        state[t, contradictory_anchor] = -1
        known_risk[t, contradictory_anchor] = True
        conflict[t, contradictory_anchor] = True
        # A snapshot older than a new known name event cannot override it.
        competing = valid & ~np.isnat(effective[t]) & (observed >= effective[t]) & (base[t] >= 0)
        disagree = competing & (anchor != base[t])
        state[t, disagree] = -1
        known_risk[t, disagree] |= (anchor[disagree] == 1) | risk[t, disagree]
        conflict[t, disagree] = True
        # Existing conflicting intervals retain the known-risk blocker.
        state[t, risk[t] & (base[t] == -1)] = -1
    return state, known_risk, filled, conflict, age


def build_states(store, codes, dates, config, names):
    """Return opening states, known-risk bits and coverage/signal matrices.

    -1 is unknown/conflicting, 0 is ordinary, 1 is risk. Snapshot carry is an
    explicitly aged last-observed-name assumption (185 calendar day maximum),
    not independent proof that every subsequent day has a complete event feed.
    """
    codes = list(codes); dates = pd.DatetimeIndex(dates)
    names = names.copy()
    for column in ['start_date', 'end_date', 'ann_date']:
        names[column] = dates_of(names[column])
    evidence = snapshot_evidence(store, codes, config)
    base, base_risk, effective, interval_overlap = _interval_states(codes, dates, names, config)
    state, known, filled, conflict, ages = _merge_snapshots(base, base_risk, effective,
                                                          dates, codes, evidence, config)
    signal_base, signal_risk, signal_effective, _ = _interval_states(codes, dates, names, config, close=True)
    signal, signal_known, signal_filled, signal_conflict, _ = _merge_snapshots(
        signal_base, signal_risk, signal_effective, dates, codes, evidence, config, close=True)
    daily = pd.DataFrame(dict(date=dates.strftime('%Y-%m-%d'), n_codes=len(codes),
        n_namechange_unknown=(base == -1).sum(axis=1), n_unknown=(state == -1).sum(axis=1),
        n_risk=known.sum(axis=1), n_snapshot_filled=filled.sum(axis=1),
        n_conflict=conflict.sum(axis=1), n_signal_unknown=(signal == -1).sum(axis=1),
        n_signal_snapshot_filled=signal_filled.sum(axis=1), n_signal_conflict=signal_conflict.sum(axis=1),
        n_snapshot_age_over_31=((ages > 31) & filled).sum(axis=1)))
    directory = Path(store.root) / 'data/repair_sources/point_in_time'
    directory.mkdir(parents=True, exist_ok=True)
    daily.to_csv(directory/'status_daily_coverage.csv', index=False)
    evidence.to_csv(directory/'status_evidence.csv', index=False)
    month_ends = daily.assign(month=pd.to_datetime(daily.date).dt.to_period('M')).groupby('month').tail(1)
    month_ends.to_csv(directory/'status_monthly_coverage.csv', index=False)
    summary = dict(n_dates=len(dates), n_codes=len(codes), snapshot_evidence_rows=len(evidence),
        namechange_unknown_stock_days=int((base == -1).sum()), remaining_unknown_stock_days=int((state == -1).sum()),
        snapshot_filled_stock_days=int(filled.sum()), conflict_stock_days=int(conflict.sum()),
        snapshot_age_limit_days=int(config.get('repair_policy', {}).get('status_snapshot_max_age_days', 185)),
        opening_evidence='ann_date < session; snapshot/publication date < session',
        signal_evidence='ann_date <= signal close; snapshot/publication date <= signal close',
        limitations=['Unknown coverage includes pre-listing/post-delisting rows in the rectangular matrix; use universe stock-month coverage for eligibility.',
            'Snapshots are carried forward as last observed names for at most 185 days, not proof of a complete intervening event feed.',
            'Published quarterly names can lag the true effective name; newer known name events take priority.'],
        evidence_sha256=sha(directory/'status_evidence.csv'), daily_coverage_sha256=sha(directory/'status_daily_coverage.csv'))
    dump(directory/'status_summary.json', summary)
    return state, known, dict(summary=summary, daily=daily,
        signal_state=signal, signal_known_ST=signal_known)
