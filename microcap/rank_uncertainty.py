"""Check whether documented capitalization precision could change a 100-stock band."""
import numpy as np
import pandas as pd


def rank_bounds(frame):
    f=frame.copy()
    caps=pd.to_numeric(f.market_cap,errors='coerce')
    f=f.loc[f.eligible.eq(True)&np.isfinite(caps)&caps.gt(0)].copy()
    if f.empty:
        return pd.DataFrame(columns=['stock_code','min_possible_rank','max_possible_rank','band_uncertain'])
    lower=f.market_cap_lower.fillna(f.market_cap).to_numpy(float)
    upper=f.market_cap_upper.fillna(f.market_cap).to_numpy(float)
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and (lower>0).all() and (lower<=upper).all()):
        raise ValueError('Invalid capitalization uncertainty bounds')
    if not ((lower<=f.market_cap.to_numpy())&(upper>=f.market_cap.to_numpy())).all():
        raise ValueError('Market-cap estimate outside documented bounds')
    codes=f.get('historical_code',f.stock_code).astype(str).to_numpy()
    minimum=1+np.searchsorted(np.sort(upper),lower,side='left')
    maximum=1+np.searchsorted(np.sort(lower),upper,side='left')-(lower<upper)
    # Only pairwise order ranges are needed; no assumption about independence or
    # a probability distribution of rounding error is made.
    upper_ties={v:sorted(g.code) for v,g in pd.DataFrame({'bound':upper,'code':codes}).groupby('bound')}
    lower_ties={v:sorted(g.code) for v,g in pd.DataFrame({'bound':lower,'code':codes}).groupby('bound')}
    import bisect
    for i,code in enumerate(codes):
        minimum[i]+=bisect.bisect_left(upper_ties.get(lower[i],[]),code)
        maximum[i]+=bisect.bisect_left(lower_ties.get(upper[i],[]),code)
    result=pd.DataFrame({'stock_code':f.stock_code.to_numpy(),
                         'min_possible_rank':minimum,'max_possible_rank':maximum})
    def band(values): return np.minimum((np.asarray(values)-1)//100,5)
    result['band_uncertain']=band(minimum)!=band(maximum)
    return result
