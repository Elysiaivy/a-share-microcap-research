"""Separate selection, evaluation and attribution with verifiable input locks."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .acquire import canonical, dump, sha
from .portfolio import simulate, select_winner, peer_panel
from .analysis import metrics, paired_tests, risk_statistics, regressions


EVALUATION_ARTIFACTS = (
    'monthly.csv', 'performance.csv', 'holdings.csv', 'risk_monthly.csv',
    'risk_summary_provisional.csv', 'paired_tests.json',
)


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value): return None
    return value


def write_json(path,value):
    dump(path,clean(value))


def load(root):
    root=Path(root)
    config=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'data/prepared/manifest.json').read_text())
    if manifest['config_sha256'] != sha(root/'config.json'):
        raise ValueError('Protocol changed after preparation. Reprepare in a separate research version.')
    for item in manifest['artifacts']:
        if sha(root/item['path']) != item['sha256']:
            raise ValueError(f'Prepared input checksum failed: {item["path"]}')
    panel=pd.read_csv(root/'data/prepared/panel.csv',parse_dates=['signal_date','horizon_end'])
    coverage=pd.read_csv(root/'data/prepared/universe_coverage.csv')
    return config,panel,coverage


def ranking_gate(coverage):
    good = coverage.n_nonfinance_mv_missing.eq(0)&coverage.n_finance_unknown.eq(0)
    if 'n_unmapped_historical_codes' in coverage:
        good &= coverage.n_unmapped_historical_codes.eq(0)
    if 'n_repair_boundary_uncertain' in coverage:
        good &= coverage.n_repair_boundary_uncertain.eq(0)
    return good


def model_panel(panel,coverage):
    """Invalidate bad-ranking months before cost calculation, also their drift."""
    valid=set(coverage.loc[ranking_gate(coverage),'return_month'])
    result=panel.copy()
    result['ranking_certified']=result.return_month.isin(valid)
    result.loc[~result.ranking_certified,'return_1m']=np.nan
    return result


def training_fingerprint(panel,coverage,config):
    # Outcome labels ST/delist and all test rows intentionally absent.
    p=panel[panel.return_month.between(config['train_start'],config['train_end'])]
    cols=['return_month','signal_date','stock_code','rank','bucket','market_cap','return_1m']
    text=p[cols].sort_values(['return_month','stock_code']).to_csv(index=False,float_format='%.17g',lineterminator='\n')
    c=coverage[coverage.return_month.between(config['train_start'],config['train_end'])]
    text+=c.sort_values('return_month').to_csv(index=False,lineterminator='\n')
    return hashlib.sha256(text.encode()).hexdigest()


def select(root):
    root=Path(root); out=root/'results';out.mkdir(exist_ok=True)
    config,panel,coverage=load(root)
    train=panel[panel.return_month.between(config['train_start'],config['train_end'])].copy()
    cov=coverage[coverage.return_month.between(config['train_start'],config['train_end'])].copy()
    monthly,holdings=simulate(model_panel(train,cov),config['main_cost_bps'])
    monthly.to_csv(out/'selection_training_monthly.csv',index=False)
    frozen=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),config=config,
                config_sha256=sha(root/'config.json'),training_input_sha256=training_fingerprint(panel,coverage,config),
                status='blocked',winner=None,scores=None)
    if not ranking_gate(cov).all():
        frozen['reason']='Historical ranking universe has unresolved market-cap or financial-industry coverage.'
        frozen['blocked_train_months']=cov.loc[~ranking_gate(cov),'return_month'].tolist()
    else:
        try:
            frozen.update(select_winner(monthly,config));frozen['status']='frozen'
        except ValueError as exc:
            frozen['reason']=str(exc)
    destination=out/'selection.json'
    if destination.exists():
        prior=json.loads(destination.read_text())
        if prior['status']=='frozen':
            if prior['training_input_sha256']!=frozen['training_input_sha256'] or prior['config_sha256']!=frozen['config_sha256']:
                raise ValueError('Existing winner is immutable. Use a new project version for changed training inputs.')
            if prior['winner']!=frozen['winner']:
                raise ValueError('Selection replay disagrees with frozen winner')
            return prior
    write_json(destination,frozen)
    return frozen


def checked_selection(root,config,panel,coverage):
    frozen=json.loads((Path(root)/'results/selection.json').read_text())
    if frozen['config_sha256']!=sha(Path(root)/'config.json') or frozen['training_input_sha256']!=training_fingerprint(panel,coverage,config):
        raise ValueError('Frozen selection no longer matches training inputs/protocol')
    return frozen


def evaluate(root):
    root=Path(root);out=root/'results';config,panel,coverage=load(root)
    frozen=checked_selection(root,config,panel,coverage)
    p=model_panel(panel,coverage)
    frames=[];holds=[]
    for cost in config['cost_bps']:
        monthly,h=simulate(p,cost);frames.append(monthly)
        if cost==config['main_cost_bps']: holds.append(h)
        if frozen['status']=='frozen':
            peers,ph=simulate(peer_panel(p,frozen['winner']),cost,buckets={'PEERS':300});frames.append(peers)
            if cost==config['main_cost_bps']: holds.append(ph)
    monthly=pd.concat(frames,ignore_index=True);monthly.to_csv(out/'monthly.csv',index=False)
    pd.concat(holds,ignore_index=True).to_csv(out/'holdings.csv',index=False)
    metrics(monthly,config).to_csv(out/'performance.csv',index=False)
    if frozen['status']=='frozen':
        write_json(out/'paired_tests.json',paired_tests(monthly,frozen['winner'],config))
    else:
        write_json(out/'paired_tests.json',dict(status='blocked',reason='No frozen in-sample winner',comparisons=[]))
    risks,summary=risk_statistics(panel,config)
    # Provisional composition is explicit; no claim about the intended full pool.
    certified_months=set(coverage.loc[ranking_gate(coverage),'return_month'])
    risks['ranking_certified']=risks.return_month.isin(certified_months)
    risks['composition_status']=np.where(risks.ranking_certified,'coverage_passed','provisional_missing_universe')
    # Summaries retain all originally specified months, including missing ones.
    # Complete target counts alone cannot certify an incomplete ranking pool.
    def period_months(period):
        if period=='TRAIN':
            return pd.period_range(config['train_start'],config['train_end'],freq='M').astype(str)
        if period=='TEST':
            return pd.period_range(config['test_start'],config['test_end'],freq='M').astype(str)
        all_months=pd.period_range(config['train_start'],config['test_end'],freq='M')
        return (all_months if period=='ALL' else all_months[all_months.year==int(period)]).astype(str)
    summary['n_uncertified_ranking_months']=[
        int((~risks.loc[risks.bucket.eq(row.bucket)&risks.return_month.isin(period_months(row.period)),
                        'ranking_certified']).sum())
        for row in summary.itertuples(index=False)
    ]
    summary['composition_status']=np.where(summary.n_uncertified_ranking_months.eq(0),
                                            'coverage_passed','provisional_missing_universe')
    risks.to_csv(out/'risk_monthly.csv',index=False);summary.to_csv(out/'risk_summary_provisional.csv',index=False)
    # Write last: a partial evaluation must never certify a mixed result set.
    write_json(out/'evaluation_manifest.json',dict(
        version=1,config_sha256=sha(root/'config.json'),
        prepared_manifest_sha256=sha(root/'data/prepared/manifest.json'),
        selection_sha256=sha(out/'selection.json'),
        artifacts=[dict(path=f'results/{name}',sha256=sha(out/name))
                   for name in EVALUATION_ARTIFACTS]))
    return monthly


def attribute(root):
    root=Path(root);config,panel,coverage=load(root)
    # Also save coverage diagnostics on the provisional sample. Inference is
    # invalidated at the same ranking gate used by the strategy.
    coefficients,status,summary=regressions(model_panel(panel,coverage),config)
    coefficients.to_csv(root/'results/regression_coefficients.csv',index=False)
    status.to_csv(root/'results/regression_coverage.csv',index=False)
    summary.to_csv(root/'results/regression_summary.csv',index=False)
    return status


def quality(root):
    root=Path(root);config,panel,coverage=load(root)
    unresolved=~np.isfinite(panel.return_1m)|panel.return_1m.lt(-1)
    expected_months = set(pd.period_range(config['train_start'],config['test_end'],freq='M').astype(str))
    expected_groups = set(config['buckets'])
    shape = panel.groupby(['return_month','bucket'],observed=True).size()
    expected_pairs = {(m,b) for m in expected_months for b in expected_groups}
    structure_complete = set(shape.index)==expected_pairs and shape.eq(100).all()
    selection_path = root/'results/selection.json'
    selection = checked_selection(root,config,panel,coverage) if selection_path.exists() else {}
    evaluation_path=root/'results/evaluation_manifest.json'
    evaluation_verified=False
    if evaluation_path.exists():
        evaluation=json.loads(evaluation_path.read_text())
        if evaluation.get('version')!=1:
            raise ValueError('Unsupported evaluation manifest version')
        for key,path in [('config_sha256',root/'config.json'),
                         ('prepared_manifest_sha256',root/'data/prepared/manifest.json'),
                         ('selection_sha256',selection_path)]:
            if not path.exists() or evaluation.get(key)!=sha(path):
                raise ValueError(f'Stale evaluation inputs: {key}; rerun evaluate')
        artifacts=evaluation.get('artifacts',[])
        expected_paths={f'results/{name}' for name in EVALUATION_ARTIFACTS}
        if len(artifacts)!=len(expected_paths) or {item.get('path') for item in artifacts}!=expected_paths:
            raise ValueError('Incomplete evaluation artifact manifest; rerun evaluate')
        for item in artifacts:
            path=root/item['path']
            if not path.exists() or item.get('sha256')!=sha(path):
                raise ValueError(f'Evaluation artifact checksum failed: {item["path"]}')
        evaluation_verified=True
    monthly_path = root/'results/monthly.csv'
    cost_complete = False
    if monthly_path.exists():
        monthly=pd.read_csv(monthly_path)
        main=monthly[monthly.cost_bps.eq(config['main_cost_bps'])&monthly.strategy.isin(expected_groups)]
        cost_complete = (set(zip(main.return_month,main.strategy))==expected_pairs and
                         len(main)==len(expected_pairs) and main.status.eq('ok').all() and np.isfinite(main.net_return).all())
    data=dict(company_months=len(panel),signal_months=int(panel.return_month.nunique()),
              historical_universe_company_months=int(coverage.n_age_eligible.sum()),
              market_cap_gaps=int(coverage.n_nonfinance_mv_missing.sum()),
              financial_classification_gaps=int(coverage.n_finance_unknown.sum()),
              repair_boundary_uncertain=int(coverage.get('n_repair_boundary_uncertain',pd.Series(dtype=float)).sum()),
              ranking_months_passed=int(ranking_gate(coverage).sum()),
              return_gaps=int(unresolved.sum()),
              suspension_marks=int(panel.return_status.eq('research_suspension_valuation').sum()),
              backup_market_caps=int(panel.market_cap_source.eq('bak_daily_rounded').sum()),
              st_unknown=int(panel.st_current.isna().sum()),
              loss_unknown=int(panel.is_loss.isna().sum()),
              delisting_exposures=int(panel.delisted_next.eq(1).sum()),
              portfolio_structure_complete=bool(structure_complete),
              main_cost_path_complete=bool(cost_complete),
              evaluation_artifacts_verified=bool(evaluation_verified),
              full_performance_release=bool(evaluation_verified and ranking_gate(coverage).all() and not unresolved.any() and structure_complete and cost_complete and selection.get('status')=='frozen'),
              financial_original_version_certified=False,
              historical_security_aliases_certified=False,
              rationale='No full-period conclusions until ranking, return and identity gaps are resolved. Financial attribution remains exploratory pending original-version certification.')
    write_json(root/'results/quality.json',data)
    return data
