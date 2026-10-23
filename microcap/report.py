"""Latest research report data, reproducible notebook and presentation handoff."""
from pathlib import Path
import html,json,os,tempfile
import numpy as np
import pandas as pd
import nbformat
from .acquire import dump,sha
from .workflow import clean
from .portfolio import select_winner

LABELS={'R001_100':'1—100','R101_200':'101—200','R201_300':'201—300','R301_400':'301—400','R401_500':'401—500','CSI1000':'中证1000'}
RULE_LABELS={'FILTER_ONLY':'仅月初过滤','EXIT_CASH':'过滤＋风险退出，留现金','EXIT_REPLACE':'过滤＋风险退出，下一交易日补仓'}
def pct(x):return f'{x:.2%}' if pd.notna(x) else '—'
def records(f):return clean(f.to_dict('records'))

def selection_evidence(monthly,performance,config,selection):
 """Recompute all-five training scores and expose the frozen validation design."""
 if len(config['candidates'])!=5 or set(config['candidates'])!=set(config['buckets']):
  raise ValueError('All five market-cap bands must participate in training selection')
 if config['main_cost_bps']!=30:
  raise ValueError('The frozen selection cost must be one-way 30bp')
 main=monthly[monthly.cost_bps.eq(config['main_cost_bps'])]
 replay=select_winner(main,config)
 if replay['winner']!=selection['winner'] or replay['training_hash']!=selection['training_hash']:
  raise ValueError('Frozen selection does not match the current training input')
 altered=main.copy()
 altered.loc[altered.return_month.between(config['test_start'],config['test_end']),'net_return']=.99
 invariant=select_winner(altered,config)
 if invariant['winner']!=replay['winner'] or invariant['training_hash']!=replay['training_hash']:
  raise ValueError('Test returns changed the training selection')
 train=main[main.return_month.between(config['train_start'],config['train_end'])]
 rows=[]
 for strategy in config['candidates']:
  values=train[train.strategy.eq(strategy)].net_return
  score=float(np.sqrt(12)*values.mean()/values.std(ddof=1))
  if not np.isclose(score,selection['scores'][strategy]['score']):
   raise ValueError('Independent training score mismatch: '+strategy)
  metric=performance[performance.strategy.eq(strategy)&performance.period.eq('TRAIN')&performance.cost_bps.eq(30)].iloc[0]
  rows.append(dict(strategy=strategy,score=score,sharpe=score,cagr=float(metric.cagr),turnover=float(metric.turnover),n_train=len(values),rank_lower=config['buckets'][strategy][0],rank_upper=config['buckets'][strategy][1],selected=strategy==selection['winner']))
 scores=pd.DataFrame(rows).sort_values(['score','turnover','rank_lower','strategy'],ascending=[False,True,False,True]).reset_index(drop=True)
 scores['rank']=np.arange(1,len(scores)+1)
 winner=selection['winner']
 pair=performance[performance.strategy.eq(winner)&performance.period.isin(['TRAIN','TEST'])&performance.cost_bps.eq(30)].copy()
 pair['period']=pd.Categorical(pair.period,categories=['TRAIN','TEST'],ordered=True);pair=pair.sort_values('period')
 benchmark=performance[performance.strategy.eq('CSI1000')&performance.period.isin(['TRAIN','TEST'])&performance.cost_bps.eq(30)].set_index('period')
 for key in ['cagr','sharpe','max_drawdown','daily_max_drawdown']:
  pair['index_'+key]=pair.period.astype(str).map(benchmark[key])
 pair['cagr_difference_vs_index']=pair.cagr-pair.index_cagr
 test=performance[performance.strategy.isin(config['buckets'])&performance.period.eq('TEST')&performance.cost_bps.eq(30)].set_index('strategy')
 ranks={key:int(test[key].rank(ascending=False,method='min').loc[winner]) for key in ['cagr','sharpe']}
 best_return=str(test.cagr.idxmax());best_sharpe=str(test.sharpe.idxmax())
 protocol=dict(candidates=list(config['candidates']),train_start=config['train_start'],train_end=config['train_end'],test_start=config['test_start'],test_end=config['test_end'],main_cost_bps=30,metric='sqrt(12)*monthly_net_mean/monthly_net_sample_std(ddof=1)',risk_free_rate_annual=config['risk_free_rate_annual'],tie_break=selection['protocol']['tie_break'],selection_only_train=True,winner_frozen_for_test=True,other_test_bands_role='descriptive comparisons; never replace the training winner')
 evidence=dict(protocol=protocol,scores=records(scores),winner=winner,winner_label=LABELS[winner],training_hash=selection['training_hash'],train_test_performance=records(pair),winner_rank_in_test=ranks,test_best_cagr=best_return,test_best_sharpe=best_sharpe,test_return_invariance=dict(passed=True,original_training_hash=replay['training_hash'],altered_test_training_hash=invariant['training_hash']),interpretation=f"样本内按30bp成本后夏普从五档中选出{LABELS[winner]}组并冻结；其样本外收益排名第{ranks['cagr']}，夏普排名第{ranks['sharpe']}。其余四档只作描述性对照，不按样本外表现换组。",disclosure='2024—2025为样本外（历史留出）；结果在研究过程中已查看，不宣称未窥视或完全独立验证。')
 return evidence,scores,pair

def figures(out,monthly,daily,config):
 os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'microcap-matplotlib'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 directory=out/'figures';directory.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
 for strategy,g in monthly[monthly.cost_bps.eq(config['main_cost_bps'])].groupby('strategy'):
  g=g.sort_values('return_month')
  ax.plot(pd.to_datetime(g.return_month),(1+g.net_return).cumprod(),label=strategy,ls='--' if strategy=='CSI1000' else '-',lw=2 if strategy in ['CSI1000','R001_100'] else 1)
 ax.axvline(pd.Timestamp('2024-01-01'),color='#9baab2',ls=':');ax.set_yscale('log');ax.set_ylabel('NAV (log scale)');ax.legend(ncol=3,frameon=False,fontsize=8)
 fig.savefig(directory/'nav.png',dpi=170);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
 for strategy,g in daily.groupby('strategy'):
  g=g.sort_values('date');test=g[g.date.ge('2024-01-01')];prior=g[g.date.lt('2024-01-01')]
  base=float(prior.nav.iloc[-1]) if len(prior) else 1.
  values=np.r_[base,test.nav];dd=values/np.maximum.accumulate(values)-1
  ax.plot(pd.to_datetime(test.date),dd[1:],label=strategy,ls='--' if strategy=='CSI1000' else '-')
 ax.set_ylabel('Daily drawdown, 2024-2025');ax.legend(ncol=3,frameon=False,fontsize=8)
 fig.savefig(directory/'drawdown_test.png',dpi=170);plt.close(fig)
 return [directory/'nav.png',directory/'drawdown_test.png']

def report(root):
 root=Path(root);out=root/'results';c=json.loads((root/'config.json').read_text());selection=json.loads((out/'selection.json').read_text())
 names={'performance':'performance.csv','daily_risk':'daily_risk.csv','cost_sensitivity':'cost_sensitivity.csv','terminal_sensitivity':'terminal_sensitivity.csv','st_ablation':'st_ablation_performance.csv','st_ablation_tests':'st_ablation_tests.csv','regression_summary':'regression_summary.csv','regression_coverage':'regression_coverage.csv','monthly':'monthly.csv','coverage':'../data/prepared/universe_coverage.csv'}
 # Preserve the original float values when replaying the frozen training hash.
 tables={name:pd.read_csv((out/file).resolve(),float_precision='round_trip') for name,file in names.items()}
 tables['relative_tests']=pd.DataFrame(json.loads((out/'paired_tests.json').read_text())['comparisons'])
 perf=tables['performance'];main=perf[perf.cost_bps.eq(c['main_cost_bps'])];test=main[main.period.eq('TEST')];train=main[main.period.eq('TRAIN')]
 validation,tables['selection_scores'],tables['winner_train_test']=selection_evidence(tables['monthly'],perf,c,selection)
 tables['selection_scores'].to_csv(out/'selection_scores.csv',index=False)
 tables['winner_train_test'].to_csv(out/'winner_validation.csv',index=False)
 winner=selection['winner'];winner_train=train[train.strategy.eq(winner)].iloc[0];winner_test=test[test.strategy.eq(winner)].iloc[0]
 small=test[test.strategy.eq(c['reference'])].iloc[0];index=test[test.strategy.eq('CSI1000')].iloc[0]
 grad=pd.read_csv(out/'size_gradient.csv').set_index('period')
 pair=tables['relative_tests'];relative=pair[pair.strategy.eq(winner)&pair.comparator.eq('CSI1000')&pair.period.eq('TEST')].iloc[0]
 trades=pd.read_csv(out/'trades.csv');blocked=pd.read_csv(out/'blocked_orders.csv');exit_changes=pd.read_csv(out/'risk_exit_price_changes.csv')
 chart_files=figures(out,tables['monthly'],pd.read_csv(out/'daily_nav.csv'),c)
 risk_rows=[]
 for strategy in c['buckets']:
  st=trades[trades.strategy.eq(strategy)];sell=st[st.reason.eq('ST')];change=exit_changes[exit_changes.strategy.eq(strategy)]
  risk_rows.append(dict(strategy=strategy,risk_sells=len(sell),delayed_sells=int(sell.delay_sessions.gt(0).sum()),max_delay_sessions=int(sell.delay_sessions.max()) if len(sell) else 0,replacement_buys=int(st.reason.eq('ST_replacement').sum()),risk_exit_median_gross_change=change.pretrigger_to_exit_gross_return.median(),risk_exit_worst_gross_change=change.pretrigger_to_exit_gross_return.min(),blocked_sell_days=int((blocked.strategy.eq(strategy)&blocked.side.eq('sell')).sum())))
 tables['risk_exits']=pd.DataFrame(risk_rows)
 tables['event_summary']=trades.groupby(['strategy','side','reason']).agg(trades=('notional','size'),notional=('notional','sum'),cost=('cost','sum')).reset_index()
 for filename,key in [('status_coverage.csv','status_coverage'),('historical_industry_coverage.csv','industry_coverage'),('historical_status_coverage.csv','historical_status_coverage'),('state_repair_coverage.csv','state_repair_coverage')]:
  path=root/'data/prepared'/filename
  if path.exists():tables[key]=pd.read_csv(path)
 best=str(test[test.strategy.isin(c['buckets'])].sort_values('cagr',ascending=False).iloc[0].strategy)
 best_sharpe=str(test[test.strategy.isin(c['buckets'])].sort_values('sharpe',ascending=False).iloc[0].strategy)
 selection_text=f"2015—2023样本内，五档均参与单边30bp成本后夏普选优；{LABELS[winner]}组以夏普{winner_train.sharpe:.2f}胜出，年化{pct(winner_train.cagr)}，随后冻结。"
 validation_text=f"冻结赢家在2024—2025样本外年化{pct(winner_test.cagr)}，夏普{winner_test.sharpe:.2f}，日度最大回撤{pct(winner_test.daily_max_drawdown)}；收益排名第{validation['winner_rank_in_test']['cagr']}，夏普排名第{validation['winner_rank_in_test']['sharpe']}。"
 ranking_text=f"样本外年化收益最高为{LABELS[best]}组，夏普最高为{LABELS[best_sharpe]}组；最小组的收益领先点估计延续，但风险调整后领先未延续，不能据此认定市值越小越好。"
 relative_text=f"冻结赢家相对中证1000价格指数月均收益差{pct(relative.mean_monthly_difference)}，HAC(6)95%区间[{pct(relative.ci95_low)}, {pct(relative.ci95_high)}]；指数年化{pct(index.cagr)}。"
 conclusions=[selection_text,validation_text,ranking_text,relative_text,validation['disclosure'],'成本、终值及ST规则对照均覆盖五档，不据此更换训练赢家；规则对照为解释实验。','月度排名固定；受阻交易、跨日资金和现金导致实际持仓并非始终100只等权。','本研究尚未识别壳价值因果效应，也没有验证资金容量或真实盘中即时补仓。']
 st=tables['st_ablation'];st=st[st.strategy.eq(c['reference'])&st.period.eq('TEST')].set_index('rule')
 st_text=f"最小组样本外：仅月初过滤年化{pct(st.loc['FILTER_ONLY','cagr'])}，增加风险退出留现金{pct(st.loc['EXIT_CASH','cagr'])}，再加次日补仓{pct(st.loc['EXIT_REPLACE','cagr'])}；该期间退出补仓收益未高于仅过滤。"
 conclusions.insert(4,st_text)
 limitations=['股票复权收益与中证1000价格指数存在分红口径差异；指数没有ETF实施费用。','日线开盘成交是近似，不含队列、成交量容量与冲击成本；买单为前一收盘计划，卖出回款下一交易日才可用于买入。','早期停牌市值仍含最后可观察值估计，历史财报版本未逐条认证。','历史分类按证据可得日期使用；季度分类存在公开发布滞后和分类体系切换。','ST未知状态仍限制买入；来源冲突和未解决覆盖必须单列，不能将过滤后未知占比为零解释为全池无缺口。']
 repairs=[dict(title='按执行时点解释停牌',problem='下午暂停曾阻挡上午开盘卖出。',fix='区分开盘暂停与日内暂停，使用suspend_timing，不用下午事件阻止上午订单。',evidence='2015-03-02：002208、002169；2015-04-01：002110。',limit='上午暂停后的日线open可能是复牌首价，不能一概视为开盘集合竞价。'),dict(title='行业按历史证据生效',problem='SW回溯分类曾决定早期金融剔除。',fix='优先当时快照；早期使用已发布证监会分类，按可得日推进。',evidence='000617重组前后的行业资格需按实际日期变化。',limit='季度公开分类并非逐日业务变更数据库。'),dict(title='补齐历史风险状态证据',problem='namechange缺区间导致普通股票被排为未知。',fix='合并历史简称快照与公开证据，保留来源、可得日和冲突记录；不以当前简称回填。',evidence='标准股份、金鹰股份、粤宏远A、民丰特纸。',limit='仍无法确认的股票明确未知并披露过滤前覆盖。'),dict(title='成交确认后跨日补仓',problem='实际卖出所得曾用于同一开盘价补仓。',fix='前收盘固定股数计划，旧现金才能资助当日买单；卖款最早下一交易日补仓。',evidence='交易账本保存plan_date、funding_asof及执行日。',limit='当天成交后立即补仓需分钟数据，本版不伪装为盘中立即。')]
 quality=json.loads((out/'quality.json').read_text())
 payload=dict(ready=True,title='A股小市值策略：样本内选组与样本外验证',generated_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),config=c,quality=quality,selection_validation=validation,repairs=repairs,conclusions=conclusions,limitations=limitations,tables={k:records(v) for k,v in tables.items()},metadata=dict(winner=winner,labels=LABELS,rule_labels=RULE_LABELS,train_period='2015—2023样本内',test_period='2024—2025样本外（历史留出）',main_cost_bps=c['main_cost_bps'],risk_free_rate_annual=c['risk_free_rate_annual']),source_notes=['Tushare冻结原始响应；证券交易所交易日历与公司历史记录。','历史行业补证使用官方发布的证监会分类与当日/此前快照；逐项来源在数据清单。','中证1000（000852.SH）价格指数月末收盘收益；不含分红、不模拟ETF费用。','全量结果与计划/成交账本保存在同目录CSV，Notebook独立复算。'])
 dump(out/'slide_payload.json',payload)
 conclusion_rows=[dict(question='样本内选组',result=selection_text,boundary='仅用2015—2023、30bp成本后月收益夏普；非按样本外挑赢家'),dict(question='样本外验证',result=validation_text,boundary=ranking_text),dict(question='相对市场',result=relative_text,boundary='未控制广泛风险暴露与股息口径'),dict(question='ST规则',result=st_text,boundary='描述性规则对照，不重新选组'),dict(question='独立性边界',result=validation['disclosure'],boundary='不能宣称未窥视或因果证明'),dict(question='交易可行性',result=conclusions[-2],boundary='日线条件成交仍不等于实盘保证')]
 pd.DataFrame(conclusion_rows).to_csv(out/'conclusions.csv',index=False)
 css='body{font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#183040;max-width:1250px;margin:36px auto;padding:0 26px}h1,h2{color:#17495c}table{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}th,td{padding:8px;border-bottom:1px solid #dde6e9;text-align:right}th:first-child,td:first-child{text-align:left}aside{padding:18px;background:#eef5f5}a{color:#287b79}'
 def tab(f,cols):
  f=f[cols].copy();f['strategy']=f.strategy.map(LABELS) if 'strategy' in f else None
  return f.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}',na_rep='—')
 body='<h1>'+payload['title']+'</h1><aside><a href="research_presentation.pptx">下载完整PowerPoint研究报告</a><p>本页为最新结果索引，完整方法、修复证据、敏感性和对照实验见演示文稿。</p></aside>'
 body+='<h2>核心结论</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in conclusions)+'</ul>'
 cols=['strategy','cagr','sharpe','max_drawdown','daily_max_drawdown','worst_month','turnover']
 body+='<h2>五档样本内选优评分</h2>'+tab(tables['selection_scores'],['strategy','rank','selected','n_train','score','cagr','turnover'])
 body+='<h2>冻结赢家：样本内与样本外对照</h2>'+tab(tables['winner_train_test'],['strategy','period','cagr','sharpe','max_drawdown','daily_max_drawdown','worst_month','turnover'])
 body+='<h2>样本内五档绩效</h2>'+tab(train,cols)+'<h2>样本外五档绩效（其余四档为描述性对照）</h2>'+tab(test,cols)
 body+='<p>cagr为年化复合收益；sharpe为月收益年化夏普，无风险率0；max_drawdown为月度回撤，daily_max_drawdown为日度回撤。百分比字段以小数保存。</p>'
 body+='<h2>五档成本、终值与ST规则对照</h2><p>完整结果：cost_sensitivity.csv、terminal_sensitivity.csv、st_ablation_performance.csv、st_ablation_tests.csv。</p><h2>数据与解释边界</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in limitations)+'</ul>'
 (out/'research_report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>小市值研究 · 最新结果</title><style>'+css+'</style>'+body+'</html>')
 (out/'conclusion_one_page.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>'+css+'</style><h1>研究结论</h1>'+pd.DataFrame(conclusion_rows).to_html(index=False,border=0)+'</html>')
 (out/'interview_script.md').write_text('五分钟面试讲稿\n\n'+'\n\n'.join([
  '第一分钟：研究问题与选组设计。每月按当时可见状态过滤后，以总市值划成五个100股目标组。五档全部参与2015—2023样本内选择，固定单边30bp，以成本后月收益年化夏普最高者胜出；完全并列先选低换手，再选较大市值区间。'+selection_text,
  '第二分钟：技术难点与修复。解释下午暂停、回溯行业、名称缺口与同一开盘先卖后买的时序问题；修复后使用冻结输入重跑。卖出回款最早下一交易日补仓。',
  '第三分钟：样本外验证。冻结样本内赢家后应用于2024—2025，交易、ST和成本规则不变；不按样本外表现换组。'+validation_text+ranking_text+relative_text,
  '第四分钟：解释实验。所有五档展示10/30/50bp成本和退市0/50/100%终值；三种ST规则使用相同数据与交易时序，只作敏感性与描述性对照。'+st_text,
  '第五分钟：证据边界。'+validation['disclosure']+conclusions[-1]+'收益领先不是风险调整后领先，也不是市值越小越好的普遍定理。'])+'\n')
 cells=[
  nbformat.v4.new_markdown_cell('# A股小市值研究：样本内选组与样本外验证\n五档全部参与2015—2023样本内选优：单边30bp成本后月收益年化夏普最高者胜出，随后冻结市值区间及交易规则，用于2024—2025样本外（历史留出）验证；仍每月按当时资料重选股票，不是持续持有2023年的名单。其余四档样本外结果仅作描述性对照，不更换赢家。\n\n2024—2025结果在研究过程中已查看，不宣称未窥视或完全独立验证。日线计划在前收盘形成，卖款从下一交易日使用。'),
  nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd, numpy as np\nfrom microcap.analysis import metrics, paired_tests\nfrom microcap.portfolio import select_winner\nfrom microcap.diagnostics import daily_risk, ablation_tests\nroot=Path.cwd()\nc=json.loads((root/'config.json').read_text())\nm=pd.read_csv(root/'results/monthly.csv',float_precision='round_trip')\ns=json.loads((root/'results/selection.json').read_text())\nassert set(m.strategy)==set(c['buckets'])|{'CSI1000'}\nassert set(c['candidates'])==set(c['buckets'])==set(s['scores'])\nassert len(c['candidates'])==5 and c['main_cost_bps']==30\nassert m.return_month.nunique()==132\nmain=m[m.cost_bps.eq(30)]\ns['winner']"),
  nbformat.v4.new_markdown_cell('## 仅用样本内独立复算五档评分\n评分为√12×月净收益均值÷样本标准差，无风险率0。108个月完整样本；完全并列时，按低换手、较大市值区间排序。年化复合收益不参与选优。'),
  nbformat.v4.new_code_cell("training=main[main.return_month.between(c['train_start'],c['train_end'])]\nscore_rows=[]\nfor strategy in c['candidates']:\n    x=training[training.strategy.eq(strategy)]\n    assert len(x)==108 and x.net_return.notna().all()\n    score_rows.append(dict(strategy=strategy,score=np.sqrt(12)*x.net_return.mean()/x.net_return.std(ddof=1),turnover=x.turnover.mean(),n_train=len(x),rank_lower=c['buckets'][strategy][0]))\nscores=pd.DataFrame(score_rows).sort_values(['score','turnover','rank_lower','strategy'],ascending=[False,True,False,True]).reset_index(drop=True)\nassert scores.iloc[0].strategy==s['winner']\npublished=pd.read_csv(root/'results/selection_scores.csv').set_index('strategy')\nfor row in scores.itertuples():\n    assert np.isclose(row.score,s['scores'][row.strategy]['score'])\n    assert np.isclose(row.score,published.loc[row.strategy,'score'])\nscores"),
  nbformat.v4.new_code_cell("replay=select_winner(main,c)\nassert replay['winner']==s['winner'] and replay['training_hash']==s['training_hash']\nchanged=main.copy()\ntest_mask=changed.return_month.between(c['test_start'],c['test_end'])\nchanged.loc[test_mask,'net_return']=.99\nchanged.loc[test_mask,'turnover']=999.\naltered=select_winner(changed,c)\nassert altered['winner']==s['winner']\nassert altered['training_hash']==replay['training_hash']\nprint('修改全部样本外收益与换手后，五档样本内评分、训练输入哈希和赢家不变')"),
  nbformat.v4.new_code_cell("r=pd.read_csv(root/'data/prepared/ranks.csv')\nassert r.groupby('return_month').size().eq(500).all()\nassert not r.duplicated(['return_month','stock_code']).any()\nwith np.load(root/'data/prepared/daily_market.npz') as z:\n    di=pd.DatetimeIndex(z['dates']).get_indexer(pd.to_datetime(r.signal_date))\n    si=r.stock_code.map({x:i for i,x in enumerate(z['codes'])}).to_numpy()\n    assert (z['signal_state'][di,si]==0).all()\n    assert not z['signal_known_ST'][di,si].any()\nr.head()"),
  nbformat.v4.new_code_cell("stats=metrics(m,c)\nstats[stats.cost_bps.eq(30)&stats.period.isin(['TRAIN','TEST'])][['strategy','period','cagr','sharpe','max_drawdown']]"),
  nbformat.v4.new_code_cell("for strategy,g in main.groupby('strategy'):\n    x=g[g.return_month.between(c['test_start'],c['test_end'])].net_return\n    expected=np.sqrt(12)*x.mean()/x.std(ddof=1)\n    row=stats[stats.strategy.eq(strategy)&stats.period.eq('TEST')&stats.cost_bps.eq(30)].iloc[0]\n    assert np.isclose(row.sharpe,expected)\nprint('Independent Sharpe checks passed; annual risk-free return=0')"),
  nbformat.v4.new_markdown_cell('## 冻结赢家的样本外验证\n比较同一赢家的样本内与样本外表现；其余四档只作描述性对照。收益与夏普分别判断，不把收益排序直接解释成风险调整后优势。'),
  nbformat.v4.new_code_cell("validation=pd.read_csv(root/'results/winner_validation.csv')\nassert len(validation)==2 and validation.strategy.eq(s['winner']).all()\nassert list(validation.period)==['TRAIN','TEST']\nfor row in validation.itertuples():\n    expected=stats[stats.strategy.eq(s['winner'])&stats.period.eq(row.period)&stats.cost_bps.eq(30)].iloc[0]\n    assert np.isclose(row.cagr,expected.cagr) and np.isclose(row.sharpe,expected.sharpe)\ntest_stats=stats[stats.strategy.isin(c['buckets'])&stats.period.eq('TEST')&stats.cost_bps.eq(30)].set_index('strategy')\nprint('赢家样本外收益排名：',int(test_stats.cagr.rank(ascending=False,method='min').loc[s['winner']]))\nprint('赢家样本外夏普排名：',int(test_stats.sharpe.rank(ascending=False,method='min').loc[s['winner']]))\nvalidation[['strategy','period','cagr','sharpe','daily_max_drawdown','index_cagr','index_sharpe']]"),
  nbformat.v4.new_code_cell("daily=pd.read_csv(root/'results/daily_nav.csv')\ndr=daily_risk(daily,c)\ndr[dr.period.eq('TEST')][['strategy','daily_max_drawdown','drawdown_peak','drawdown_trough','recovery_date','mean_cash','min_holdings','max_holdings']]"),
  nbformat.v4.new_code_cell("tr=pd.read_csv(root/'results/trades.csv')\nassert np.allclose(tr.cost,tr.notional*.003)\nassert (pd.to_datetime(tr.plan_date)<pd.to_datetime(tr.date)).all()\nassert (pd.to_datetime(tr.funding_asof)<pd.to_datetime(tr.date)).all()\nrep=tr[tr.reason.eq('ST_replacement')]\nfor x in rep.itertuples():\n    sales=tr[tr.strategy.eq(x.strategy)&tr.stock_code.eq(x.replaces)&tr.reason.eq('ST')&tr.date.lt(x.date)]\n    assert len(sales)>0\nrep.head()"),
  nbformat.v4.new_code_cell("pairs=pd.DataFrame(paired_tests(m,s['winner'],c)['comparisons'])\npairs[pairs.period.eq('TEST')&pairs.comparator.eq('CSI1000')]"),
  nbformat.v4.new_markdown_cell('## 五档敏感性与ST规则对照\n以下实验用于解释规则和假设的影响，不用于按样本外结果更换冻结赢家。'),
  nbformat.v4.new_code_cell("pd.read_csv(root/'results/cost_sensitivity.csv').query(\"period=='TEST'\").pivot(index='strategy',columns='cost_bps',values='cagr')"),
  nbformat.v4.new_code_cell("pd.read_csv(root/'results/terminal_sensitivity.csv').query(\"period=='TEST' and strategy!='CSI1000'\").pivot(index='strategy',columns='terminal_fraction',values='cagr')"),
  nbformat.v4.new_code_cell("a=pd.read_csv(root/'results/st_ablation_monthly.csv')\nbase=a[a.rule.eq('EXIT_REPLACE')].sort_values(['strategy','return_month'])\nexpected=main[main.strategy.isin(c['buckets'])].sort_values(['strategy','return_month'])\nassert np.allclose(base.net_return,expected.net_return)\npd.read_csv(root/'results/st_ablation_performance.csv').query(\"period=='TEST'\")[[ 'strategy','rule','cagr','sharpe','daily_max_drawdown','mean_cash']]"),
  nbformat.v4.new_code_cell("ablation_tests(a,c).query(\"period=='TEST'\")"),
  nbformat.v4.new_code_cell("coverage=pd.read_csv(root/'results/regression_coverage.csv')\ncoverage.head()"),
  nbformat.v4.new_code_cell("pd.read_csv(root/'results/regression_summary.csv').query(\"period=='TEST'\")"),
  nbformat.v4.new_code_cell("json.loads((root/'results/quality.json').read_text())"),
  nbformat.v4.new_markdown_cell('解释边界：2024—2025可用于冻结赢家的样本外历史评价，但结果在研究过程中已查看，不能宣称未窥视或完全独立验证。价格指数与含分红复权策略口径不同；ST对照为观察性规则实验；无容量与盘中立即成交认证；剩余收益不证明壳价值。')]
 nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}});nbformat.write(nb,root/'full_reproduction.ipynb')
 outputs=[out/'research_report.html',out/'conclusion_one_page.html',out/'interview_script.md',out/'slide_payload.json',out/'conclusions.csv',out/'selection_scores.csv',out/'winner_validation.csv',root/'full_reproduction.ipynb',*chart_files]
 source_files=[out/name for name in ['regression_summary.csv','regression_coverage.csv','risk_exit_price_changes.csv','audit_cases.csv','audit_source_evidence.json','execution_audit_cases.csv']]
 dump(out/'report_manifest.json',dict(artifacts=[dict(path=str(f.relative_to(root)),sha256=sha(f)) for f in outputs],source_artifacts=[dict(path=str(f.relative_to(root)),sha256=sha(f)) for f in source_files]))
 return payload
