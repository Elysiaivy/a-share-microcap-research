"""Build the static research presentation from frozen public result tables."""
from pathlib import Path
import json
import re
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
ASSETS = OUT / 'assets'
ASSETS.mkdir(exist_ok=True)
P = pd.read_csv(ROOT / 'results/performance.csv')
C = json.loads((ROOT / 'config.json').read_text())
S = json.loads((ROOT / 'results/selection.json').read_text())
W = S['winner']
B = C['candidates']
CAP = pd.read_csv(ROOT / 'results/supplement/capital_performance.csv')
ANNUAL = pd.read_csv(ROOT / 'results/applicability/annual_returns.csv').set_index('year')
CONC = pd.read_csv(ROOT / 'results/applicability/concentration.csv').set_index('period')
EXITS = pd.read_csv(ROOT / 'results/risk_exit_price_changes.csv')
EXITS = EXITS[EXITS.strategy.eq(W)]
PAIR = next(x for x in json.loads((ROOT / 'results/paired_tests.json').read_text())['comparisons']
            if x['strategy'] == W and x['comparator'] == 'CSI1000' and x['period'] == 'TEST')
REPO = 'https://github.com/Elysiaivy/a-share-microcap-research'

for name in ['nav_periods', 'test_drawdown', 'paired_intervals', 'extended_bands', 'capital_sensitivity']:
    shutil.copy2(ROOT / 'results/word_figures' / f'{name}.png', ASSETS / f'{name}.png')

def performance(strategy, period='TEST'):
    return P[P.strategy.eq(strategy) & P.period.eq(period) & P.cost_bps.eq(30)].iloc[0]

def capital(amount, mode='LOT_MIN5'):
    return CAP[CAP.period.eq('TEST') & CAP['mode'].eq(mode) & CAP.initial_capital.eq(amount)].iloc[0]

def rows(period):
    out = []
    for band in B + ['CSI1000']:
        q = performance(band, period)
        label = '中证1000价格指数' if band == 'CSI1000' else f"{C['buckets'][band][0]}—{C['buckets'][band][1]}名"
        tag = ' <span class="tag">冻结赢家</span>' if band == W else ''
        out.append(f'<tr class="{"selected" if band == W else ""}"><th scope="row">{label}{tag}</th>'
                   f'<td>{q.cagr:.2%}</td><td>{q.sharpe:.3f}</td><td>{q.daily_max_drawdown:.2%}</td></tr>')
    return ''.join(out)

def capital_rows():
    out = []
    for amount in [100000, 300000, 1000000, 2000000]:
        q = capital(amount)
        out.append(f'<tr><th scope="row">{amount / 10000:.0f}万元</th><td>{q.cagr:.2%}</td>'
                   f'<td>{q.daily_max_drawdown:.2%}</td><td>{q.mean_cash:.2%}</td><td>{q.mean_holdings:.1f}</td></tr>')
    q = capital(1, 'CONTINUOUS')
    out.append(f'<tr class="reference"><th scope="row">连续份额对照</th><td>{q.cagr:.2%}</td>'
               f'<td>{q.daily_max_drawdown:.2%}</td><td>{q.mean_cash:.2%}</td><td>{q.mean_holdings:.1f}</td></tr>')
    return ''.join(out)

def annual_rows():
    return ''.join(f'<tr><th scope="row">{year}</th><td>{q[W]:.2%}</td><td>{q.CSI1000:.2%}</td></tr>'
                   for year, q in ANNUAL.iterrows())

q, benchmark, train = performance(W), performance('CSI1000'), performance(W, 'TRAIN')
values = {
    'REPO': REPO, 'TEST': rows('TEST'), 'TRAIN': rows('TRAIN'), 'CAPITAL_ROWS': capital_rows(),
    'ANNUAL_ROWS': annual_rows(), 'CAGR': f'{q.cagr:.2%}', 'SHARPE': f'{q.sharpe:.3f}',
    'DRAWDOWN': f'{q.daily_max_drawdown:.2%}', 'BENCH_CAGR': f'{benchmark.cagr:.2%}',
    'BENCH_SHARPE': f'{benchmark.sharpe:.3f}', 'BENCH_DD': f'{benchmark.daily_max_drawdown:.2%}',
    'TRAIN_SHARPE': f'{train.sharpe:.3f}', 'TRAIN_CAGR': f'{train.cagr:.2%}',
    'YEAR24': f'{ANNUAL.loc[2024, W]:.2%}', 'YEAR25': f'{ANNUAL.loc[2025, W]:.2%}',
    'BENCH24': f'{ANNUAL.loc[2024, "CSI1000"]:.2%}', 'BENCH25': f'{ANNUAL.loc[2025, "CSI1000"]:.2%}',
    'PAIR_MEAN': f'{PAIR["mean"] * 100:.3f}', 'PAIR_LOW': f'{PAIR["ci95_low"] * 100:.3f}',
    'PAIR_HIGH': f'{PAIR["ci95_high"] * 100:.3f}',
    'CONCENTRATION': f'{CONC.loc["TEST", "cagr_best5_replaced_cash"]:.2%}',
    'CAP100': f'{capital(1000000).cagr:.2%}', 'CASH100': f'{capital(1000000).mean_cash:.2%}',
    'CAPBASE': f'{capital(1, "CONTINUOUS").cagr:.2%}',
    'EXITS': str(len(EXITS)), 'DELAYED': str(int(EXITS.delay_sessions.gt(0).sum())),
    'MAX_DELAY': str(int(EXITS.delay_sessions.max())),
}
page = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A股小市值策略研究：2024—2025历史留出年化23.42%、日度最大回撤45.16%。比较中证1000，检验ST退出、整手与最低佣金的执行代价。报告、Notebook与代码可查。">
<meta name="theme-color" content="#102b50">
<title>A股小市值策略 · 收益、风险与执行 | Elysiaivy</title>
<link rel="stylesheet" href="assets/style.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='8' fill='%23102b50'/%3E%3Cpath d='M17 46V18H47M17 31H40M17 46H47' fill='none' stroke='white' stroke-width='5'/%3E%3C/svg%3E">
</head>
<body>
<a class="skip" href="#main">跳至正文</a>
<header class="site-header"><div class="header-inner">
<a class="brand" href="#"><span class="brand-mark" aria-hidden="true">E</span><span>Elysiaivy <small>QUANTITATIVE RESEARCH</small></span></a>
<nav aria-label="主导航"><a href="#results">历史业绩</a><a href="#execution">现实执行</a><a href="#method">研究方法</a><a href="#read">报告与代码</a></nav>
<a class="github-link" href="@@REPO@@">GitHub <span aria-hidden="true">↗</span></a>
</div></header>
<main id="main">
<section class="hero" aria-labelledby="project-title">
<div class="hero-inner">
<div class="hero-copy"><p class="eyebrow">A-SHARE EQUITY RESEARCH <span>/</span> 中国A股 · 2015—2025</p>
<h1 id="project-title">A股小市值策略<br><span>收益延续与执行约束</span></h1>
<p class="lead">按历史总市值筛选小市值股票，用固定规则检验收益，再把ST退出、整手与最低佣金纳入执行评估。</p>
<div class="actions"><a class="button" href="#results">查看收益与风险 <span aria-hidden="true">↓</span></a><a class="outline-button" href="research_report.pdf">阅读完整报告 <span aria-hidden="true">↗</span></a></div>
</div>
<aside class="verdict" aria-label="研究结论摘要"><p class="eyebrow">研究结论 / AT A GLANCE</p>
<h2>历史有盈利，<br>实盘可行性仍待验证。</h2>
<p>最小市值组在样本内胜出，历史留出期收益高于中证1000；但回撤更大，相对收益尚缺乏统计显著证据。</p>
<div class="verdict-rule"><span>固定策略</span><strong>过滤后总市值第1—100名</strong></div>
<div class="verdict-rule"><span>当前证据</span><strong>历史回测 + 执行情景模拟</strong></div>
</aside>
</div>
<div class="metric-panel"><div class="metric-caption"><strong>2024—2025 历史留出表现</strong><span>冻结赢家 · 主回测扣单边30bp成本</span></div>
<div class="metrics">
<div class="metric"><span class="metric-label">复合年化收益</span><strong>@@CAGR@@</strong><span class="metric-compare">中证1000 <b>@@BENCH_CAGR@@</b></span></div>
<div class="metric"><span class="metric-label">年化夏普比率</span><strong>@@SHARPE@@</strong><span class="metric-compare">中证1000 <b>@@BENCH_SHARPE@@</b></span></div>
<div class="metric"><span class="metric-label">日度最大回撤</span><strong>@@DRAWDOWN@@</strong><span class="metric-compare">中证1000 <b>@@BENCH_DD@@</b></span></div>
</div></div>
<p class="hero-footnote">夏普使用月收益、无风险率0。中证1000为价格指数。2024—2025曾在研究迭代中查看，属于历史留出评价，并非未窥视验证；以上为回测，非实盘业绩。</p>
</section>

<section class="section" id="results">
<div class="section-heading"><div><p class="eyebrow">01 / 历史盈利能力</p><h2>赚到了收益，也承受了深度回撤</h2></div><p>先看冻结赢家与基准，<br>再看收益是否持续、是否集中。</p></div>
<div class="performance-layout">
<figure class="chart-main"><div class="figure-title"><h3>冻结赢家 vs 中证1000</h3><span>2024.01—2025.12</span></div>
<img src="assets/overview_nav.svg" alt="2024—2025日度净值和回撤：冻结赢家年化23.42%、最大回撤45.16%；中证1000年化13.58%、最大回撤27.08%。" width="880" height="550">
<figcaption>期初净值归一为1，策略承接2023年末持仓。复权股票收益含分红；价格指数不含分红，亦未扣ETF费用。点击图表可查看原尺寸。</figcaption></figure>
<aside class="year-panel"><p class="eyebrow">收益的时间分布</p><h3>两年的表现并不均匀</h3>
<div class="year-block"><span>2024</span><strong>@@YEAR24@@</strong><small>中证1000 @@BENCH24@@</small></div>
<div class="year-block"><span>2025</span><strong>@@YEAR25@@</strong><small>中证1000 @@BENCH25@@</small></div>
<p class="note">最强5个月收益置零后，留出期年化为<strong>@@CONCENTRATION@@</strong>。这是事后集中度诊断，说明收益依赖少数月份，不能据此择时。</p></aside>
</div>
<div class="statistical-note"><span class="note-label">相对收益证据</span><p>相对中证1000的平均月收益差为 <strong>@@PAIR_MEAN@@ 个百分点</strong>，HAC(6) 95%区间为 <strong>[@@PAIR_LOW@@, @@PAIR_HIGH@@]</strong>，包含0。当前样本不足以确认稳定的正超额收益。</p><small>区间衡量平均月收益差，不是年化收益差；亦不等于已经过风险调整的Alpha。</small></div>
<details class="detail-block"><summary>查看2015—2025逐年收益与相对收益区间</summary><div class="detail-content detail-grid">
<div class="table-wrap"><table><caption>各年回测复合收益 · 主回测30bp</caption><thead><tr><th scope="col">年份</th><th scope="col">冻结赢家</th><th scope="col">中证1000</th></tr></thead><tbody>@@ANNUAL_ROWS@@</tbody></table></div>
<figure><img loading="lazy" src="assets/paired_intervals.png" alt="最小组相对其余四档及中证1000的平均月收益差及HAC六阶95%置信区间，区间均包含零"><figcaption>24个月的留出期较短；点估计领先不等于已经确认独立Alpha。</figcaption></figure>
</div></details>
</section>

<section class="section" id="execution">
<div class="section-heading"><div><p class="eyebrow">02 / 现实执行条件</p><h2>从理论持仓，到资金实际能买到什么</h2></div><p>已评估交易约束的代价。<br>仍需验证容量、冲击与真实账户账本。</p></div>
<div class="execution-intro"><p>100万元的整手＋最低佣金情景，年化收益为<strong>@@CAP100@@</strong>，平均现金占比为<strong>@@CASH100@@</strong>。小资金更难接近100只股票的分散持仓，现金闲置会改变收益与回撤。</p><span class="status-label">执行模拟 · 非最低入场资金建议</span></div>
<div class="table-wrap"><table><caption>2024—2025资金敏感性 · 各情景均从现金建仓</caption><thead><tr><th scope="col">期初资金 / 情景</th><th scope="col">年化收益</th><th scope="col">日度最大回撤</th><th scope="col">平均现金</th><th scope="col">平均持股数</th></tr></thead><tbody>@@CAPITAL_ROWS@@</tbody></table></div>
<p class="note">100股买入整手；单笔综合费用＝27.5bp成交额＋max(2.5bp成交额，5元)，是研究假设。平均持股数可能因待售旧仓与补仓重叠而略超100只；连续份额为无整手、无最低佣金的同起点对照。</p>
<div class="disclosure"><strong>比较口径：</strong>@@CAP100@@应与同现金起点的@@CAPBASE@@比较；首页@@CAGR@@承接旧持仓。小资金回撤较低同时伴随更高现金占比，不能直接解释成策略风控改善。本实验未测出资金容量上限。</div>
<div class="execution-evidence"><article><p class="eyebrow">ST退出的实际障碍</p><h3>知道风险，不等于当日能卖出</h3><div class="exit-number"><strong>@@DELAYED@@ <span>/ @@EXITS@@</span></strong><small>全样本最小组风险退出成交中出现延迟</small></div><p>最长列示案例延迟@@MAX_DELAY@@个交易日。三种ST规则在留出期的最大回撤均约45.16%，退出规则没有消除组合的大幅下跌。</p></article>
<div class="evidence-list"><h3>可行性验证到哪一步？</h3><div><span class="evidence-state">已模拟</span><p>次日开盘、停牌与限价约束、现金预算、风险退出与递延补仓、整手和最低佣金。</p></div><div><span class="evidence-state pending">待验证</span><p>订单参与率、集合竞价容量、成交队列、动态冲击成本，以及真实股数和分红到账的账户账本。</p></div><div><span class="evidence-state pending">尚无记录</span><p>冻结规则后逐日留痕的前向模拟或实盘业绩。当前不能据回测认定可复制执行。</p></div></div></div>
<details class="detail-block"><summary>查看全部资金档位与执行口径</summary><div class="detail-content"><figure><img loading="lazy" src="assets/capital_sensitivity.png" alt="10万至500万元各资金档位，比较仅整手、整手加最低佣金和连续份额的年化收益"><figcaption>两段各自从现金开始；当前以复权权益模拟财富变化，尚非按未复权整数股数、现金分红到账逐笔核对的券商账户。</figcaption></figure><p class="note"><a href="@@REPO@@/blob/main/results/supplement/capital_performance.csv">下载资金实验完整结果</a> · <a href="@@REPO@@/blob/main/DATA_CONTRACT.md">交易与数据口径</a></p></div></details>
</section>

<section class="section" id="method">
<div class="section-heading"><div><p class="eyebrow">03 / 方法与稳健性</p><h2>先在样本内选组，再固定赢家验证</h2></div><p>五档均完整报告。<br>不根据历史留出期表现更换赢家。</p></div>
<ol class="method-grid"><li><span class="step">01 / 排名与建仓</span><h3>历史总市值，从小到大</h3><p>沪深主板，含原中小板和后来退市股票。排除金融、CDR、上市不足252交易日及当时已知ST、退市整理和状态未知股票，再排名分组。每组目标100只、月度等权。</p></li><li><span class="step">02 / 2015—2023 选优</span><h3>五档按成本后夏普竞争</h3><p>1—100至401—500名均参加。单边30bp下，1—100名以夏普<strong>@@TRAIN_SHARPE@@</strong>胜出，样本内年化<strong>@@TRAIN_CAGR@@</strong>。完全并列先低换手，再较大市值区间。</p></li><li><span class="step">03 / 2024—2025 评价</span><h3>冻结赢家，保留不利证据</h3><p>不换组、不重新选优。持仓出现已知风险后，在可成交时退出；卖款最早下一交易日按月初冻结候选顺延补仓。该时段已在迭代中查看，属于历史留出。</p></li></ol>
<div class="table-top"><h3>五档完整绩效</h3><div class="tabs" role="group" aria-label="选择绩效期间"><button class="active" aria-pressed="true" data-period="TEST">2024—2025 历史留出</button><button aria-pressed="false" data-period="TRAIN">2015—2023 样本内</button></div></div>
<div class="table-wrap"><table id="performance-table"><caption class="sr-only">五档组合与中证1000绩效：2024—2025历史留出</caption><thead><tr><th scope="col">过滤后总市值名次</th><th scope="col">年化收益</th><th scope="col">夏普</th><th scope="col">日度最大回撤</th></tr></thead><tbody id="TEST">@@TEST@@</tbody><tbody id="TRAIN" hidden>@@TRAIN@@</tbody></table></div>
<p class="note">留出期夏普最高的是401—500名。总体小市值梯度不等于“市值越小，在所有时期和所有指标下都越好”。窄屏可横向滚动表格。</p>
<details class="detail-block"><summary>查看扩展至1000名的结果及已完成实验</summary><div class="detail-content"><figure><img loading="lazy" src="assets/extended_bands.png" alt="十档市值区间年化收益及夏普：样本内总体随市值增大而降低，留出期并非严格单调"><figcaption>501—1000名是事后扩展，未参与原五档选优。</figcaption></figure><p class="note">已完成：10/30/50bp成本、三种ST规则、退市残值情景、同样本行业／流动性／动量／亏损控制、年度及收益集中度、市场环境分析、整手与最低佣金。完整定义和数值见报告及Notebook。</p></div></details>
<div class="data-boundary" id="limits"><h3>证据边界</h3><p>早期停牌市值含估计补全；财报历史版本未全部认证；退市残值属于情景假设。控制变量后的剩余收益、亏损或ST身份，均不能单独证明壳价值或因果关系。公开Notebook支持结果复算，完整历史回测仍需授权输入。</p></div>
</section>

<section class="section roadmap" id="next">
<div class="section-heading"><div><p class="eyebrow">04 / 下一步研究</p><h2>优先补足“能否成交”，再积累新证据</h2></div><a class="textlink" href="@@REPO@@/blob/main/RESEARCH_ROADMAP.md">查看实验方案 ↗</a></div>
<div class="roadmap-grid"><article><span class="step">P1 / 成交容量</span><h3>订单会不会太大？</h3><p>按历史成交额衡量参与率，加入成交上限、延迟与冲击情景，画出资金规模—完成率—净收益关系。</p><small>最直接补足实盘可行性的缺口。</small></article><article><span class="step">P2 / 账户账本</span><h3>现金和股数能否逐笔对上？</h3><p>用未复权整数股数记录交易、公司行动、现金分红和实际费率，核对当前复权份额近似的误差。</p><small>从执行敏感性走向账户级验证。</small></article><article><span class="step">并行启动 / 前向模拟</span><h3>固定规则后还能延续吗？</h3><p>从新的冻结日期起，逐日保存候选、订单、未成交原因和模拟净值，按预先约定周期评价。</p><small>以上为后续计划，尚未完成这些验证。</small></article></div>
<p class="note">此外，可补全收益口径基准和组合层面的因子归因；现有横截面控制回归不能替代它们。当前无需继续堆叠排名分组。</p>
</section>

<section class="section readmore" id="read">
<div><p class="eyebrow">05 / 报告与复现</p><h2>每个结论，都能追溯到结果表</h2><p>先读报告把握研究逻辑，再用Notebook复算选优、绩效与HAC区间，最后检查执行引擎和数据契约。</p><p class="note">无需API可复算公开结果；从原始行情重跑需要另行准备已授权数据，缺失输入会被检查拦截。</p></div>
<div class="resources"><a href="research_report.pdf"><div><strong>完整研究报告</strong><span>PDF · 14页 · 研究细节与反面证据</span></div><b aria-hidden="true">↗</b></a><a href="research_report.docx" download><div><strong>Word报告</strong><span>下载研究正文及完整图表</span></div><b aria-hidden="true">↓</b></a><a href="@@REPO@@/blob/main/research.ipynb"><div><strong>Notebook与结果复算</strong><span>从公开表重算核心指标</span></div><b aria-hidden="true">↗</b></a><a href="@@REPO@@"><div><strong>Python源码与测试</strong><span>执行引擎、数据契约及运行说明</span></div><b aria-hidden="true">↗</b></a></div>
</section>
</main>
<footer><a class="brand" href="#">Elysiaivy <span>/ Research</span></a><span>个人量化研究项目 · 数据截至2025年末<br>页面更新 2026.10 · 非实盘业绩</span></footer>
<script src="assets/site.js"></script>
</body></html>'''
for key, value in values.items():
    page = page.replace(f'@@{key}@@', value)
if re.search(r'@@[A-Z_0-9]+@@', page):
    raise ValueError('Unresolved template value')
page = re.sub(r'(<img[^>]*src="(assets/[^"]+)"[^>]*>)', r'<a href="\2" aria-label="打开原尺寸图表">\1</a>', page)
(OUT / 'index.html').write_text(page, encoding='utf-8')
(OUT / '.nojekyll').touch()
print('Built docs/index.html from frozen result tables')
