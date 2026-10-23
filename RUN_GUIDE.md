# 运行与复现说明

## 1 公开包能完成什么

公开文件包括研究代码、测试、冻结配置和选择结果、汇总收益、必要持仓与风险记录、补充实验结果、图表及最新报告。不需要API或账号即可运行公开Notebook。

以下步骤**重算已有结果的统计量**，没有从行情重新撮合交易，也不能认证原始数据的时点真实性：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_public.py --manifest
python -m unittest discover -s tests
python scripts/execute_notebook.py
```

Windows激活命令：`.venv\Scripts\activate`。使用Python 3.13，主要依赖版本与原研究一致，见requirements.txt。Notebook执行器使用运行它的Python解释器。`verify_public`检查冻结赢家、训练哈希、绩效、配对HAC区间、日度回撤和年度复利；缺失、不一致即报错。

`PUBLIC_MANIFEST.json`校验发布文件。重建Notebook或Word可能因元数据不同产生新哈希，因此先校验原下载文件，再重建；维护者在复核实际改动后运行`python scripts/build_public_manifest.py`更新清单，不能用更新清单掩盖异常。

## 2 图表 适用性和报告生成

```bash
python scripts/analyze_applicability.py
python scripts/build_word_figures.py
python scripts/build_word_report.py
python scripts/build_site.py
```

中文图表需要已安装Noto Sans CJK SC等中文字体，或用环境变量`MICROCAP_FONT`指定字体文件。适用性分析直接使用已提供的`results/applicability/csi300_monthly.csv`，其来源记录与哈希列入分析清单；没有回退到作者电脑的路径。

唯一最终Word为`docs/research_report.docx`；基于它导出`docs/research_report.pdf`。Word生成过程的临时基础文档在被忽略的`.build/`中，不作为第二个研究版本发布。

PDF导出需要LibreOffice，可使用：

```bash
soffice --headless --convert-to pdf --outdir docs docs/research_report.docx
```

转换后必须检查分页、中文字体和图表。仓库已包含校验后的PDF，普通阅读和统计复算不需要安装办公软件。`build_site.py`读取结果表生成网页表格并复制所需图表，不重跑策略。

## 3 完整交易回测需要额外数据

请在独立工作副本重跑，避免覆盖已发布结果。未包含原始行情、完整`trades.csv`/`order_plans.csv`、资金实验逐笔明细和全部公告证据。

需要另行准备、取得授权并固定：

- `data/raw/manifest.json`及其引用的全部冻结API响应，包含2015年前的初始化行情和财报；重新下载的数据可能因供应商修订与原版本不同。
- `data/repair_sources/`下的历史行业公开日、简称与风险状态、暂停/终止上市公告、早期总市值估计及相应证据清单。
- `data/prepared/`中的panel、universe、ranks、selection_universe、coverage、daily_market及各自清单；原始输入处理后生成这些文件。
- 完整核验Notebook还需主回测成交/委托账本及`results/supplement/manifest.json`引用的所有实验明细。

外部冻结清单通过`MICROCAP_SOURCE_MANIFESTS`指定，多路径使用操作系统路径分隔符。可选在线补缺通过本地环境变量`TUSHARE_TOKEN`和`--online`启用；不要将Token写入文件或Git。网络下载不自动解决历史授权、状态缺口和数据版本认证。

准备数据后：

```bash
python -m microcap prepare
python -m microcap market
python -m microcap select
python -m microcap evaluate
python -m microcap attribute
python -m microcap audit
python -m microcap report
python scripts/run_execution_sensitivity.py
```

`python -m microcap all`亦可执行主流程。入口先验证必要输入及冻结原始文件哈希，缺失时失败，绝不补零或删除消失股票；随后各阶段继续执行自己的覆盖率与版本检查。基础输入存在不代表所有经济假设已获认证。

`full_reproduction.ipynb`用于完整数据环境的细节核验，输出已清除，公开包无法直接运行它。完整流程生成的研究诊断与公开展示Notebook分别保存，避免互相覆盖。

## 4 版本与结论边界

- `config.json`、冻结赢家和原始收益表保持原研究内容；发布修改集中在路径、复现和展示。
- `provenance/`及结果目录中的历史运行清单是原完整数据环境的证据，不是公开包完整性的校验入口。脱敏路径会改变相应清单文件自身的哈希；原始数据内容哈希仍保留。
- 执行与样本外假设见DATA_CONTRACT.md。2024—2025曾被查看，不能称为全新独立验证。
- 测试与结果核对通过，不等于从原始数据重跑通过。本次发布未重跑完整行情回测。

## 5 发布维护

GitHub Pages从`main`分支的`docs/`发布。提交前运行验证、测试、Notebook；报告改动须重新导出PDF并检查。重建网页与公开清单后提交。GitHub Actions重复运行测试和公开结果复算，下载后的原始Notebook也接受文件校验；运行时产生的缓存和数据不入库。
