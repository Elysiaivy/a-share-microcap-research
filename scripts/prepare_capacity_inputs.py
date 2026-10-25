"""Fetch separate frozen daily capacity fields and build strictly lagged inputs.

Public authentication uses TUSHARE_TOKEN only. Original raw inputs and research
results are read-only; all new market data stay under DATA_ROOT/data/capacity.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from microcap.capacity_data import fetch_days,build_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--probe',help='Fetch and validate one trading day, YYYYMMDD; do not build')
    p.add_argument('--offline',action='store_true',help='Only build from complete frozen inputs')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--interval',type=float,default=1.3,help='Global interval; <1.25s requires current standard-access probe; never <0.35s')
    args=p.parse_args()
    with np.load(args.data_root/'data/prepared/daily_market.npz',allow_pickle=False) as z:
        dates=pd.DatetimeIndex(z['dates'])
    days=dates.strftime('%Y%m%d').tolist()
    if args.probe:
        if args.probe not in days:raise ValueError('Probe date is outside the frozen calendar')
        fetch_days(args.data_root,[args.probe],workers=1,interval=args.interval);return
    if not args.offline:
        # Make the holdout plus warmup available first; it does not change outputs.
        days=sorted(days,key=lambda day:(day<'20231101',day))
        fetch_days(args.data_root,days,workers=args.workers,interval=args.interval)
    build_inputs(args.data_root)


if __name__=='__main__':
    main()
