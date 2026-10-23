import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='2015–2025 microcap research with executable ST exits')
    p.add_argument('stage',choices=['prepare','market','select','evaluate','attribute','audit','report','all'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--online',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    from .preflight import require_full_inputs
    require_full_inputs(root, a.stage, a.online)
    from . import research
    if a.stage in ['prepare','all']:
        from .prepare import prepare
        prepare(root,a.online)
    if a.stage=='market':
        from .execution import build_market
        import json,pandas as pd
        c=json.loads((root/'config.json').read_text());panel=pd.read_csv(root/'data/prepared/universe.csv',usecols=['stock_code']);build_market(root,panel,c)
    for stage in ['select','evaluate','attribute']:
        if a.stage in [stage,'all']:getattr(research,stage)(root)
    if a.stage in ['audit','all']:
        from .execution_audit import audit
        audit(root)
    if a.stage in ['report','all']:
        from .report import report
        report(root)

if __name__=='__main__':main()
