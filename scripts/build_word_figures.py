"""Draw publication figures directly from the frozen research result tables."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.doc_build/mpl_cache'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ticker import PercentFormatter
from matplotlib import dates as mdates
# Resolve an installed CJK font, or set MICROCAP_FONT to an authorized local font.
font_path = os.environ.get('MICROCAP_FONT')
if not font_path:
 for family in ['Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC']:
  try:
   font_path = findfont(FontProperties(family=family), fallback_to_default=False)
   break
  except ValueError:
   pass
if not font_path:
 raise RuntimeError('Install Noto Sans CJK SC or set MICROCAP_FONT to a CJK font file before building Chinese figures.')
FONT=FontProperties(fname=font_path)
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B3BDC5','axes.labelcolor':'#303840','xtick.color':'#475560','ytick.color':'#475560','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
R=ROOT/'results';F=R/'word_figures';F.mkdir(exist_ok=True)
B=['R001_100','R101_200','R201_300','R301_400','R401_500'];LABEL={s:f'{1+i*100}—{100+i*100}' for i,s in enumerate(B)};LABEL['CSI1000']='中证1000'
COL=dict(zip(B+['CSI1000'],['#135A78','#3A9B99','#8FAD55','#BF8C4A','#8C6DA6','#555D65']))
P=pd.read_csv(R/'performance.csv');M=pd.read_csv(R/'monthly.csv',float_precision='round_trip');D=pd.read_csv(R/'daily_nav.csv');D['date']=pd.to_datetime(D.date,format="mixed")

def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=230,bbox_inches='tight');plt.close(fig)
def grid(ax):ax.grid(axis='y',color='#E4E9EC',lw=.7);ax.set_axisbelow(True)
def textfont(fig):
 for ax in fig.axes:
  for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:t.set_fontproperties(FONT)
  lg=ax.get_legend()
  if lg:
   for t in lg.get_texts():t.set_fontproperties(FONT)
# 1 training
v=P[(P.cost_bps==30)&(P.period=='TRAIN')].set_index('strategy').loc[B]
fig,ax=plt.subplots(figsize=(8.5,3.1));bars=ax.bar([LABEL[s] for s in B],v.sharpe,color=[COL[s] for s in B],width=.55)
for bar,val in zip(bars,v.sharpe):ax.text(bar.get_x()+bar.get_width()/2,val+.025,f'{val:.3f}',ha='center')
ax.set_ylim(0,1.02);ax.set_ylabel('月收益年化夏普（rf=0）');ax.set_xlabel('过滤后总市值名次');grid(ax);textfont(fig);save(fig,'train_sharpe')
# 2 normalized month-end nav each period
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,start,end,title in zip(axs,['2015-01','2024-01'],['2023-12','2025-12'],['样本内：2015—2023','样本外：2024—2025']):
 for s in B+['CSI1000']:
  q=M[(M.cost_bps==30)&(M.strategy==s)&M.return_month.between(start,end)].sort_values('return_month');x=pd.PeriodIndex(q.return_month,freq='M').to_timestamp('M');x=pd.DatetimeIndex([pd.Period(start,freq='M').start_time-pd.Timedelta(days=1),*x]);y=np.r_[1,np.cumprod(1+q.net_return)]
  ax.plot(x,y,label=LABEL[s],c=COL[s],lw=2 if s in [B[0],'CSI1000'] else 1.1,ls='--' if s=='CSI1000' else '-')
 ax.set_title(title);ax.set_ylabel('期初归一化净值');grid(ax);ax.xaxis.set_major_locator(mdates.YearLocator(2 if start=='2015-01' else 1));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,prop=FONT,frameon=False,bbox_to_anchor=(.5,-.08));fig.tight_layout();textfont(fig);save(fig,'nav_periods')
# 3 test drawdown includes initial NAV
fig,ax=plt.subplots(figsize=(8.8,3.6))
for s in B+['CSI1000']:
 allq=D[D.strategy==s].sort_values('date');q=allq[allq.date>='2024-01-01'];base=allq[allq.date<'2024-01-01'].nav.iloc[-1];nav=np.r_[base,q.nav];dd=nav/np.maximum.accumulate(nav)-1;dates=[pd.Timestamp('2023-12-29'),*q.date]
 ax.plot(dates,dd,label=LABEL[s],c=COL[s],lw=2 if s in [B[0],'CSI1000'] else 1.1,ls='--' if s=='CSI1000' else '-')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('自期内历史峰值的回撤');ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.legend(ncol=3,prop=FONT,frameon=False,loc='lower right');grid(ax);textfont(fig);save(fig,'test_drawdown')
# 4 cash/holdings
q=D[(D.strategy==B[0])&(D.date>='2024-01-01')]
fig,axs=plt.subplots(2,1,figsize=(8.8,3.6),sharex=True)
axs[0].plot(q.date,q.cash_weight,c=COL[B[0]]);axs[0].set_ylabel('现金占比');axs[0].yaxis.set_major_formatter(PercentFormatter(1));axs[1].plot(q.date,q.hold_count,c='#3A9B99',lw=1.2);axs[1].axhline(100,c='#666666',ls='--',lw=.8);axs[1].set_ylabel('实际持股数');axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
for a in axs:grid(a)
fig.tight_layout();textfont(fig);save(fig,'cash_holdings')
# 5 winner-relative CI (reverse stored differences when needed)
pairs=json.load(open(R/'paired_tests.json'))['comparisons'];rel=[]
for other in B[1:]+['CSI1000']:
 if other=='CSI1000':row=next(x for x in pairs if x['strategy']==B[0] and x['comparator']==other and x['period']=='TEST');vals=(row['mean'],row['ci95_low'],row['ci95_high'])
 else:row=next(x for x in pairs if x['strategy']==other and x['comparator']==B[0] and x['period']=='TEST');vals=(-row['mean'],-row['ci95_high'],-row['ci95_low'])
 rel.append((other,*vals))
fig,ax=plt.subplots(figsize=(8.8,3.3))
for i,(s,mu,lo,hi) in enumerate(rel):ax.errorbar(mu*100,i,xerr=[[100*(mu-lo)],[100*(hi-mu)]],fmt='o',c=COL[s],capsize=4)
ax.set_yticks(range(5),['最小组 − '+LABEL[s] for s,*_ in rel]);ax.invert_yaxis();ax.axvline(0,c='#89949E',ls='--');ax.set_xlabel('配对月收益差（百分点）及HAC(6) 95%区间');ax.grid(axis='x',color='#E4E9EC');fig.tight_layout();textfont(fig);save(fig,'paired_intervals')
# 6 cost
C=pd.read_csv(R/'cost_sensitivity.csv');fig,axs=plt.subplots(1,2,figsize=(8.8,3.4))
for ax,period,title in zip(axs,['TRAIN','TEST'],['样本内年化收益','样本外年化收益']):
 for s in B:
  q=C[(C.strategy==s)&(C.period==period)].sort_values('cost_bps');ax.plot(q.cost_bps,q.cagr,marker='o',label=LABEL[s],c=COL[s])
 ax.set_xticks([10,30,50]);ax.set_xlabel('单边成本（bp）');ax.set_title(title);ax.yaxis.set_major_formatter(PercentFormatter(1));grid(ax)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,prop=FONT,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.08));fig.tight_layout();textfont(fig);save(fig,'cost_sensitivity')
# 7 regression
G=pd.read_csv(R/'regression_summary.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.7),sharey=True)
for ax,per,title in zip(axs,['TRAIN','TEST'],['样本内：108个月','样本外：24个月']):
 for model,off,col,lab in [('baseline',-.13,'#8F9FA9','仅区间'),('full',.13,'#135A78','加入控制')]:
  q=G[(G.period==per)&(G.model==model)].set_index('term').loc[B[1:]]
  ax.errorbar(q['mean']*100,np.arange(4)+off,xerr=[(q['mean']-q.ci95_low)*100,(q.ci95_high-q['mean'])*100],fmt='o',color=col,capsize=3,label=lab)
 ax.axvline(0,c='#999',ls='--');ax.set_yticks(range(4),[LABEL[s] for s in B[1:]]);ax.set_title(title);ax.set_xlabel('相对1—100组毛收益差（百分点/月）');ax.grid(axis='x',color='#E4E9EC')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=2,prop=FONT,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.08));axs[0].invert_yaxis();fig.tight_layout();textfont(fig);save(fig,'regression_intervals')
# 8 ST bars
S=pd.read_csv(R/'st_ablation_performance.csv');fig,ax=plt.subplots(figsize=(8.8,3.3));x=np.arange(5)
for i,(rule,label,col) in enumerate([('FILTER_ONLY','仅月度过滤','#9EABB5'),('EXIT_CASH','退出后留现金','#3A9B99'),('EXIT_REPLACE','退出并次日补仓','#135A78')]):
 q=S[(S.period=='TEST')&(S.rule==rule)].set_index('strategy').loc[B];ax.bar(x+(i-1)*.24,q.cagr,width=.23,label=label,color=col)
ax.set_xticks(x,[LABEL[s] for s in B]);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('样本外年化收益');ax.legend(prop=FONT,ncol=3,frameon=False,loc='upper left');ax.set_ylim(0,.285);grid(ax);textfont(fig);save(fig,'st_rules')
print('Figures:',len(list(F.glob('*.png'))))
# Supplemental experiments are descriptive and do not enter frozen selection.
E=pd.read_csv(R/'supplement/extended_performance.csv');K=pd.read_csv(R/'supplement/capital_performance.csv')
allb=B+[f'R{lo:03d}_{lo+99}' for lo in range(501,1000,100)]
labels=[f'{lo}–{lo+99}' for lo in range(1,1000,100)]
q=pd.concat([P[P.cost_bps.eq(30)],E],ignore_index=True)
fig,axs=plt.subplots(1,2,figsize=(9,3.5))
for ax,metric,title in zip(axs,['cagr','sharpe'],['复合年化收益','月收益年化夏普']):
 for per,lab,col in [('TRAIN','2015—2023','#135A78'),('TEST','2024—2025','#BF8C4A')]:
  g=q[q.period.eq(per)].set_index('strategy').loc[allb]
  ax.plot(np.arange(10),g[metric],marker='o',label=lab,c=col,ms=4)
 ax.axvline(4.5,c='#777',ls=':',lw=1);ax.axhline(0,c='#AAA',lw=.7)
 ax.set_xticks(np.arange(10),labels,rotation=55,ha='right',fontsize=8);ax.set_title(title)
 if metric=='cagr':ax.yaxis.set_major_formatter(PercentFormatter(1))
 grid(ax);ax.legend(prop=FONT,frameon=False,fontsize=9)
fig.tight_layout();textfont(fig);save(fig,'extended_bands')
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
for ax,per,title in zip(axs,['TRAIN','TEST'],['2015—2023 期初投入','2024—2025 期初投入']):
 g=K[K.period.eq(per)];base=g[g['mode'].eq('CONTINUOUS')].iloc[0]
 ax.axhline(base.cagr,c='#777',ls='--',lw=1,label='连续份额对照')
 for mode,label,col in [('LOT','100股整手','#3A9B99'),('LOT_MIN5','整手＋最低佣金','#135A78')]:
  v=g[g['mode'].eq(mode)].sort_values('initial_capital');ax.plot(np.arange(len(v)),v.cagr,marker='o',c=col,label=label,ms=4)
 ax.set_xticks(np.arange(7),['10','20','30','50','100','200','500']);ax.set_xlabel('期初资金（万元）');ax.set_title(title)
 ax.set_ylim(.08,.25);ax.yaxis.set_major_formatter(PercentFormatter(1));grid(ax)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,prop=FONT,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.09))
fig.tight_layout();textfont(fig);save(fig,'capital_sensitivity')
