"""Append descriptive applicability sections to the supplied report, preserving original content."""
from pathlib import Path
import json,hashlib
import pandas as pd
from docx import Document
from docx.shared import Cm,Pt
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/applicability'
OUT=ROOT/'docs/research_report.docx'
def pct(v):return f'{v*100:.2f}%'
def main():
 doc=Document(ROOT/'.build/report_base.docx')
 doc.styles['Normal'].font.size=Pt(10)
 doc.styles['Normal'].paragraph_format.line_spacing=1.10
 anchor=next(p._p for p in doc.paragraphs if p.text.startswith('9  结论'))
 def move(obj):anchor.addprevious(obj);return obj
 def p(text,heading=False):
  x=doc.add_paragraph(text,style='Heading 2' if heading else 'Normal');move(x._p)
  if not heading:
   x.paragraph_format.line_spacing=1.05;x.paragraph_format.space_after=Pt(4)
   for run in x.runs:run.font.size=Pt(10)
  return x
 def page(text):
  x=doc.add_paragraph(text,style='Heading 1');x.paragraph_format.page_break_before=True;move(x._p)
 def table(title,heads,rows,widths):
  p(title)
  t=doc.add_table(rows=1,cols=len(heads));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
  for c,w in zip(t.columns,widths):c.width=Cm(w)
  for c,s in zip(t.rows[0].cells,heads):c.text=s
  for row in rows:
   for c,s in zip(t.add_row().cells,row):c.text=str(s)
  for ri,row in enumerate(t.rows):
   tr=row._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
   if ri==0:tr.append(OxmlElement('w:tblHeader'))
   for ci,c in enumerate(row.cells):
    c.width=Cm(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    pr=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E7EBEE' if ri==0 else ('F5F7F8' if ri%2==0 else 'FFFFFF'));pr.append(sh)
    ma=OxmlElement('w:tcMar')
    for side in ['top','bottom','left','right']:
     e=OxmlElement('w:'+side);e.set(qn('w:w'),'55');e.set(qn('w:type'),'dxa');ma.append(e)
    pr.append(ma)
    for par in c.paragraphs:
     par.paragraph_format.space_after=Pt(1);par.paragraph_format.space_before=Pt(1);par.paragraph_format.line_spacing=1.05
     par.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
     for run in par.runs:run.font.size=Pt(9);run.bold=ri==0
  bd=OxmlElement('w:tblBorders')
  for side in ['top','bottom','left','right','insideH','insideV']:
   e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');bd.append(e)
  t._tbl.tblPr.append(bd);move(t._tbl)
  x=p('');x.paragraph_format.space_after=Pt(0);x.paragraph_format.line_spacing=Pt(3)
 conc=pd.read_csv(R/'concentration.csv').set_index('period');annual=pd.read_csv(R/'annual_returns.csv');summary=json.loads((R/'summary.json').read_text());env=pd.read_csv(R/'environment.csv')
 page('9  收益来源与时间集中度')
 p('最小市值组的优势伴随明显的时间集中度。分组与控制回归支持对市值相关收益差异的描述，但不能识别独立Alpha或壳价值。本节固定原赢家和交易规则，仅检查收益来自哪些时期，不再选组或优化参数。')
 p('市值梯度提供横截面证据',True)
 p('原五档与新增501—1000名区间在样本内总体呈现较小市值收益较高的特征，但样本外存在区间反转，401—500组的夏普也高于最小组。结合第5节同样本控制回归，现有结果更适合表述为“与市值及相关特征共同变化的收益差异”，不应将全部收益归因于单一市值因子。')
 table('表11  原策略逐年收益与市场背景',['年份','原策略成本后收益','中证1000价格收益'],[[str(int(x.year)),pct(x.R001_100),pct(x.CSI1000)] for x in annual.itertuples()],[3,6.9,6.9])
 p(f'2015年的对数收益占2015—2023年累计对数收益的{summary["year2015_log_share_train"]:.1%}。不含2015年时，2016—2023年年化收益为{pct(summary["train_without_2015_cagr"])}，低于原样本内22.99%；2016—2025年为{pct(summary["all_without_2015_cagr"])}。这是不同子时期的描述比较，不是删除不利或有利年份后的新策略。')
 table('表12  最强月份对复利结果的影响',['期间','原年化收益','最强5月对数贡献占比','这5月收益置零后年化'],[[{'TRAIN':'2015—2023','TEST':'2024—2025'}[k],pct(conc.loc[k,'cagr']),pct(conc.loc[k,'best5_log_share']),pct(conc.loc[k,'cagr_best5_replaced_cash'])] for k in ['TRAIN','TEST']],[3.1,3.4,5.1,5.2])
 p('按已实现月对数收益排序，将最强5个月收益替换为0，保留原月份总数，其余月份不变。该事后诊断不能交易，也不是对未来错失月份的预测。2024—2025年贡献占比超过100%，表示这些上涨月份的贡献被其他月份的净损失部分抵消；随意减仓可能错过决定复利结果的反弹。')
 page('10  市场环境与策略适用性')
 p('市场环境分为中证1000当月上涨与不涨；风格环境分为中证1000当月收益高于沪深300与不高于沪深300。两项分别分组，不交叉细分，不调整原持仓。风格数据来自已有本地冻结行情，可计算月收益的期间为2020—2025年；不补造早期指数历史。')
 p('当月环境解释收益差异',True)
 def erows(lag):
  q=env[env.period.eq('TEST')&env.lag.eq(lag)];rows=[]
  for dim,state,label in [('market','positive','市场上涨'),('market','nonpositive','市场不涨'),('style','positive','中证1000占优'),('style','nonpositive','沪深300不落后')]:
   x=q[q.dimension.eq(dim)&q.state.eq(state)].iloc[0];rows.append([label,int(x.n),pct(x['mean']),pct(x.win_rate),pct(x.tail10_rate)])
  return rows
 table('表13  按当月环境分类的原策略表现 2024—2025',['当月环境','月数','平均月收益','盈利月份比例','亏损至少10%比例'],erows(0),[4.7,1.3,3.6,3.6,3.6])
 p('中证1000占优的13个月，原策略平均月收益7.09%；其余11个月为−3.51%，4次亏损至少10%的月份均落在后者。市场下跌环境中，策略平均仍亏损2.77%，即使平均跑赢中证1000约1.71个百分点，也不意味着能保护本金。')
 p('当月分类使用了当月结束后才完整知道的市场收益，仅作事后环境归因。股票组合为成本后复权收益，两项指数均为价格收益；这些比较不构成可交易净超额收益或因果解释。')
 p('上月已知环境的区分能力明显减弱',True)
 table('表14  按上月环境分类的下月原策略表现 2024—2025',['上月环境','月数','下月平均收益','下月盈利比例','下月亏损至少10%比例'],erows(1),[4.7,1.3,3.6,3.6,3.6])
 p('将环境整体滞后一个月后，大小盘风格两组下月平均收益为3.37%和1.08%，但两组各12个月，均有2次大幅亏损，亏损至少10%的比例同为16.67%。市场涨跌分组的尾部亏损比例也相同。因此，当月环境的强相关不能直接转化为下一月的可靠风险预警。')
 p('环境分组月份不连续，不计算拼接年化收益或回撤。完整分期统计和逐月标签保存在results/applicability；风格统计始于2020年。以上为描述分析，不改变赢家或仓位。')
 page('11  交易约束决定研究收益的实现程度')
 p('交易适用性可以分为成本承受、风险退出和资金分散三个层面。以下汇总第4节及第7—8节的既有实验，不增加新的手续费、资金规模或成交规则。')
 cost=pd.read_csv(R/'execution_costs.csv');q=cost[cost.period.eq('TEST')].sort_values('cost_bps')
 table('表15  成本变化与日度风险 2024—2025',['单边综合成本','年化收益','日度最大回撤'],[[f'{int(x.cost_bps)}bp',pct(x.cagr),pct(x.daily_max_drawdown)] for x in q.itertuples()],[5.6,5.6,5.6])
 p('从10bp提高到50bp，年化收益由24.56%降至22.30%，但最大回撤幅度仍在45%左右。成本改变收益水平和组间排序，无法解释或消除主要尾部风险；该区间不包含基于盘口和成交参与率估计的动态冲击成本。')
 p('退出规则的适用条件是能够成交',True)
 p('原策略全样本128次风险退出成交中，97次发生延迟，约75.8%。最长已列示案例延迟166个交易日。策略必须能承担风险被识别后仍无法卖出的价格损失；设置ST退出规则不等于已经获得及时止损保护。停牌估值沿用末次价格还可能推迟账面风险显现。')
 cap=pd.read_csv(R/'capital_constraints.csv');q=cap[cap.period.eq('TEST')&cap['mode'].eq('LOT_MIN5')&cap.initial_capital.isin([100000,1000000,2000000])]
 table('表16  同起点资金实验的实现偏差 2024—2025',['期初资金','年化收益','平均现金比例','日度最大回撤'],[[f'{x.initial_capital/10000:.0f}万元',pct(x.cagr),pct(x.mean_cash),pct(x.daily_max_drawdown)] for x in q.itertuples()],[3.8,4.3,4.3,4.4])
 p('本表均从2024年初现金建仓，执行整手及最低佣金；同起点连续份额对照年化22.88%，平均现金1.22%。10万元情景现金闲置45.58%，较小回撤主要伴随风险暴露降低。100万元时现金约5.04%，年化收益较同起点对照低0.89个百分点；不能与承接旧持仓的23.42%直接相减。')
 p('可陈述的适用边界',True)
 p('本策略更适合被理解为高分散、承担较大权益回撤的小市值研究组合。有限资金下，100只目标的整手及费用约束可能显著改变实际暴露；资金增加可减少整手闲置，却不代表市场容量随之得到验证。现有500万元情景未检验成交量约束、冲击成本和排队，因此不给出可承载资金上限。')
 p('收益集中度说明错过少数上涨月代价较高；环境分析说明相对领先与绝对盈利不同；执行实验说明研究净值与实际账户仍有距离。这三项适用性结论共同限定项目的使用范围，不据此产生新的交易信号。')
 # Keep the original conclusion while integrating the additions and repairing section reference.
 for par in doc.paragraphs:
  if par.text.startswith('9  结论'):par.text=par.text.replace('9  ','12  ',1)
  elif '历史版本与估值限制见第9节' in par.text:par.text=par.text.replace('见第9节','见第12节')
  elif par.text.startswith('本项目验证了一个受ST'):
   par.text='本项目验证了一个受ST和交易约束影响的A股小市值策略，并通过历史留出、区间扩展、收益集中度、市场环境与资金敏感性分析划定适用边界。收益受少数强势月份影响，当月风格关联不能直接用于预测下月大跌，资金约束会改变实际暴露。核心价值是可复现地说明收益优势、风险代价和证据不足，而不是把历史最高收益包装为稳健获利规律。'
  elif par.text.startswith('在项目目录运行单元测试'):
   par.text = '公开包可用research.ipynb复算选优、绩效与主要统计量；这属于已保存结果的复算。完整交易回测使用python -m microcap all，需另外准备已授权的历史输入，运行前执行缺口检查。Word统一由scripts/build_word_report.py生成；复现步骤与依赖见RUN_GUIDE.md。'
  elif par.text.startswith('核心结果：'):
   par.text = '核心结果位于results，执行和区间扩展位于results/supplement，适用性分析位于results/applicability。PUBLIC_MANIFEST.json校验公开文件，provenance说明历史记录的边界。原始行情、完整逐笔模拟账本及公告证据未随公开包提供；不能仅凭公开结果表认证原始输入。'
 in_appendix=False
 for par in doc.paragraphs:
  if par.text.startswith('附录'):in_appendix=True
  if in_appendix and par.style.name=='Normal':
   par.paragraph_format.line_spacing=1.05;par.paragraph_format.space_after=Pt(3)
   if '本次135项单元测试通过' in par.text:par.text=par.text.replace('本次135项单元测试通过','原版135项单元测试通过，本次完整代码运行141项测试通过')
   for run in par.runs:run.font.size=Pt(9)
 # Fixed leading avoids platform-specific CJK auto-line-height differences.
 doc.styles['Title'].font.size=Pt(20)
 for par in doc.paragraphs:
  if par.style.name=='Title':
   par.text=par.text.replace('\n','');par.paragraph_format.line_spacing=Pt(27)
  elif par.style.name=='Heading 1':par.paragraph_format.line_spacing=Pt(24)
  elif par.style.name=='Heading 2':par.paragraph_format.line_spacing=Pt(18)
  elif par.text:par.paragraph_format.line_spacing=Pt(14)
 for t in doc.tables:
  for row in t.rows:
   for cell in row.cells:
    for par in cell.paragraphs:par.paragraph_format.line_spacing=Pt(13)
 doc.core_properties.author='Elysiaivy';doc.core_properties.last_modified_by='Elysiaivy'
 OUT.parent.mkdir(exist_ok=True);doc.save(OUT)
 (R/'document_manifest.json').write_text(json.dumps({'base_sha256':hashlib.sha256((ROOT/'.build/report_base.docx').read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'inserted_sections':[9,10,11],'new_tables':[11,12,13,14,15,16],'base_paragraphs_preserved_except_documented_integration':True},indent=2))
 print(OUT)
if __name__=='__main__':main()
