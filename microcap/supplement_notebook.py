"""Attach reproducible supplement checks without changing selection cells."""
import nbformat

def add_supplement(notebook, root):
    notebook.cells=[cell for cell in notebook.cells if 'execution_supplement' not in cell.metadata.get('tags',[])]
    if not (root/'results/supplement/manifest.json').exists():
        return
    cells=[nbformat.v4.new_markdown_cell('## 扩展区间与资金执行敏感性\n新增501—1000名只作描述性扩展，不进入原五档选优。资金实验只使用冻结赢家，两段分别从现金开始；整手与最低5元佣金不改变选股规则。佣金2.5bp包含在单边30bp中，额外费用仅为不足5元的差额。复权收益记账仍假设分配收益再投资，不是完整券商结算。'),
    nbformat.v4.new_code_cell('''import hashlib
sup=root/'results/supplement'
manifest=json.loads((sup/'manifest.json').read_text())
assert manifest['protocol']['frozen_winner']==s['winner']
for name,digest in manifest['sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name,digest in manifest['outputs'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
capital=pd.read_csv(sup/'capital_performance.csv')
cm=pd.read_csv(sup/'capital_monthly.csv')
for key,g in cm.groupby(['period','mode','initial_capital']):
    row=capital[(capital.period==key[0])&(capital['mode']==key[1])&(capital.initial_capital==key[2])].iloc[0]
    assert len(g)==(108 if key[0]=='TRAIN' else 24)
    assert np.isclose(row.cagr,(1+g.net_return).prod()**(12/len(g))-1)
    assert np.isclose(row.sharpe,np.sqrt(12)*g.net_return.mean()/g.net_return.std(ddof=1))
checks=pd.read_csv(sup/'capital_checks.csv')
assert checks.unverified_stale_marks.eq(0).all()
assert checks.buy_lot_max_error.dropna().lt(1e-7).all()
assert checks.min_cash_yuan.ge(-1e-5).all()
capital.query("period=='TEST'")[['mode','initial_capital','cagr','sharpe','daily_max_drawdown','mean_cash','mean_holdings']]'''),
    nbformat.v4.new_code_cell('''extended=pd.read_csv(sup/'extended_monthly.csv')
assert len(extended)==5*132 and extended.net_return.notna().all()
assert not extended.duplicated(['strategy','return_month']).any()
assert not set(extended.strategy)&set(c['candidates'])
em=metrics(extended,c)
published=pd.read_csv(sup/'extended_performance.csv')
check=em.merge(published,on=['strategy','cost_bps','period'],suffixes=('_recalc','_saved'),validate='one_to_one')
assert np.allclose(check.cagr_recalc,check.cagr_saved)
assert np.allclose(check.sharpe_recalc,check.sharpe_saved)
published.query("period in ['TRAIN','TEST']")[['strategy','period','cagr','sharpe','daily_max_drawdown']]'''),
    nbformat.v4.new_code_cell('''from IPython.display import Image, display
for name in ['extended_bands','capital_sensitivity']:
    display(Image(filename=str(root/'results/word_figures'/f'{name}.png')))
print('扩展只用于描述；资金模型不构成真实容量上限或独立样本外证明。')''')]
    for cell in cells:cell.metadata['tags']=['execution_supplement']
    notebook.cells.extend(cells)
