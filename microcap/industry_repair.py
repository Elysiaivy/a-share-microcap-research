"""Point-in-time industry evidence for eligibility and monthly controls.

Use dated market snapshots bounded to one prior session, then an already
published official CSRC classification. Never use retrospective SW identities
or report-end company types to certify a historical company's business.
"""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from .acquire import sha


def canonical_code(code, config):
    for alias in config.get('security_aliases', []):
        if code == alias['old']:
            return alias['canonical']
    return code


@lru_cache(maxsize=4)
def official_classifications(root):
    """Read individually hashed official publications and parsed rows."""
    directory = Path(root) / 'data/repair_sources/point_in_time/official_industry'
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        return pd.DataFrame()
    manifest = json.loads(manifest_path.read_text())
    frames = []
    for entry in manifest['sources']:
        pdf, parsed = Path(root) / entry['path'], Path(root) / entry['parsed_path']
        if sha(pdf) != entry['sha256'] or sha(parsed) != entry['parsed_sha256']:
            raise ValueError('Official historical industry source hash mismatch')
        frame = pd.read_csv(parsed, dtype={'industry_code': str})
        if not frame.source_sha256.eq(entry['sha256']).all():
            raise ValueError('Parsed classifications have the wrong source hash')
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    for column in ['observation_date', 'publication_date']:
        result[column] = pd.to_datetime(result[column])
    return result


def apply_prior_financial_snapshot(active, snapshot, signal, source_day, calendar,
                                   financial_industries, max_lag=1,
                                   update_industry=False, source='prior_bak_basic_bounded_asof'):
    """Apply dated evidence, rejecting future or stale snapshots."""
    result = active.copy()
    signal, source_day = pd.Timestamp(signal), pd.Timestamp(source_day)
    calendar = pd.DatetimeIndex(calendar)
    if source_day > signal:
        raise ValueError('Future industry snapshot cannot repair a historical signal')
    lag = int(((calendar > source_day) & (calendar <= signal)).sum())
    if source_day not in calendar or lag > max_lag:
        raise ValueError('Industry snapshot outside the allowed trading-session lag')
    if snapshot.ts_code.duplicated().any():
        raise ValueError('Duplicate issuer in historical industry snapshot')
    if 'trade_date' in snapshot:
        dates = pd.to_datetime(snapshot.trade_date.astype(str), format='mixed')
        if not dates.eq(source_day).all():
            raise ValueError('Snapshot response date does not match its requested date')
    labels = result.ts_code.map(snapshot.set_index('ts_code').industry).replace('', np.nan)
    use = result.is_financial.isna() & labels.notna()
    result.loc[use, 'is_financial'] = labels[use].isin(financial_industries).astype(float)
    result.loc[use, 'financial_source'] = source
    result.loc[use, 'financial_source_date'] = source_day.strftime('%Y-%m-%d')
    result.loc[use, 'financial_source_sha256'] = snapshot.attrs.get('source_sha256', '')
    result.loc[use, 'financial_snapshot_lag_sessions'] = lag
    result.loc[use, 'financial_snapshot_industry'] = labels[use]
    if update_industry:
        result.loc[use, 'industry'] = labels[use]
        result.loc[use, 'industry_source'] = source
        result.loc[use, 'industry_observation_date'] = source_day.strftime('%Y-%m-%d')
        result.loc[use, 'industry_available_date'] = source_day.strftime('%Y-%m-%d')
    return result


def repair_financial_classification(store, active, signal, calendar,
                                    financial_industries, max_lag=1, config=None):
    """Use close-date/prior-session snapshots, then published CSRC evidence.

    Same-date backups are admissible after the signal close. Official quarterly
    tables are admitted only after their publication date and expire after 185
    calendar days. Quarterly publication can lag the actual business change.
    """
    config = config or {}
    result = active.copy()
    signal = pd.Timestamp(signal)
    calendar = pd.DatetimeIndex(calendar)
    prior = calendar[calendar < signal]
    for lag in range(0, min(int(max_lag), 1) + 1):
        day = signal if lag == 0 else prior[-1] if len(prior) else None
        if day is None:
            continue
        for api in ['bak_basic', 'bak_daily']:
            entries = [e for e in store.find(api, trade_date=day.strftime('%Y%m%d'))
                if not e['request']['body']['params'].get('ts_code')
                and {'ts_code', 'trade_date', 'industry'}.issubset(
                    e['request']['body'].get('fields', '').split(','))]
            if not entries:
                continue
            snapshot = store.read(entries[-1])
            limit = int(entries[-1]['request']['body']['params'].get('limit', 7000))
            if len(snapshot) >= limit:
                raise ValueError('Historical industry snapshot may be truncated')
            if snapshot.empty:
                continue
            snapshot = snapshot.copy()
            snapshot['ts_code'] = snapshot.ts_code.map(lambda x: canonical_code(x, config))
            conflict = snapshot.groupby('ts_code').industry.nunique().gt(1)
            snapshot = snapshot[~snapshot.ts_code.isin(conflict[conflict].index)].drop_duplicates('ts_code')
            result = apply_prior_financial_snapshot(result, snapshot, signal, day, calendar,
                financial_industries, max_lag=1, update_industry=True,
                source=f'historical_{api}' if lag == 0 else f'prior_{api}_bounded_asof')
    official = official_classifications(store.root)
    if len(official) and result.is_financial.isna().any():
        available = official[(official.publication_date < signal) &
            ((signal - official.publication_date).dt.days <= 185)].copy()
        if len(available):
            available['ts_code'] = available.ts_code.map(lambda x: canonical_code(x, config))
            latest = available.sort_values('publication_date').drop_duplicates('ts_code', keep='last').set_index('ts_code')
            use = result.is_financial.isna() & result.ts_code.isin(latest.index)
            for target, column in [('is_financial', 'is_financial'), ('industry', 'industry'),
                ('financial_snapshot_industry', 'industry'), ('financial_source_sha256', 'source_sha256'),
                ('industry_sector', 'industry_sector')]:
                result.loc[use, target] = result.ts_code.map(latest[column])[use]
            result.loc[use, 'financial_source'] = 'published_CSRC_quarterly_asof'
            result.loc[use, 'industry_source'] = 'published_CSRC_quarterly_asof'
            for target, column in [('financial_source_date', 'observation_date'),
                ('industry_observation_date', 'observation_date'),
                ('industry_available_date', 'publication_date')]:
                result.loc[use, target] = result.ts_code.map(latest[column].dt.strftime('%Y-%m-%d'))[use]
    return result
