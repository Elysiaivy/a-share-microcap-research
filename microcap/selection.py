"""Observable risk exclusion followed by contiguous, disjoint monthly ranks."""
import numpy as np
import pandas as pd
from .portfolio import DEFAULT_BUCKETS

def eligible_ranks(ranked,market):
    dates=pd.DatetimeIndex(market['dates']);index={c:i for i,c in enumerate(market['codes'])}
    result=ranked.copy();result['original_rank']=result['rank']
    day=dates.get_indexer(pd.to_datetime(result.signal_date));stock=result.stock_code.map(index)
    if (day<0).any() or stock.isna().any():raise ValueError('Missing dated risk-state coverage')
    # A month-end signal is formed after that close. Execution states remain
    # opening states; same-day snapshots may only inform the closing signal.
    state=market.get('signal_state',market['state'])[day,stock.to_numpy(int)]
    known=market.get('signal_known_ST',market['known_ST'])[day,stock.to_numpy(int)]
    result=result.loc[(state==0)&~known].sort_values(['return_month','original_rank']).copy()
    result['rank']=result.groupby('return_month').cumcount()+1
    result['bucket']=pd.Series(pd.NA,index=result.index,dtype='string')
    for name,(lo,hi) in DEFAULT_BUCKETS.items():result.loc[result['rank'].between(lo,hi),'bucket']=name
    return result.reset_index(drop=True)
