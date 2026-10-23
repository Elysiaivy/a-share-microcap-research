from pathlib import Path
import sys,json
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from microcap.execution import build_market
root=Path(__file__).resolve().parents[1]
p=pd.read_csv(root/'data/prepared/universe.csv',usecols=['stock_code'])
build_market(root,p,json.loads((root/'config.json').read_text()))
