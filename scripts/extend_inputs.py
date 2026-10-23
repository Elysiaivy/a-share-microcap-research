from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from microcap.acquire import FrozenStore
s=FrozenStore(Path(__file__).resolve().parents[1],online=True)
try:
 for start in range(1990,2026,5):
  end=min(start+4,2025)
  f=s.query('trade_cal',dict(exchange='SSE',start_date=f'{start}0101',end_date=f'{end}1231',limit=6000,offset=0),'exchange,cal_date,is_open,pretrade_date')
  assert len(f)<6000
  s.save_manifest();print('calendar',start,end,len(f),flush=True)
finally:s.save_manifest()
