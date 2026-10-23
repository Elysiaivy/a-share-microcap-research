"""Acquire a small, explicit supplement from the official backup quote API."""
from pathlib import Path
import pandas as pd
from .acquire import FrozenStore, dump


def repair(root):
    root = Path(root)
    store = FrozenStore(root,online=True)
    panel = pd.read_csv(root/'data/prepared/panel.csv')
    dates = sorted(set(panel.signal_date)|set(panel.horizon_end))
    errors = []
    try:
        for number,day in enumerate(dates):
            try:
                store.date_query('bak_daily',day.replace('-',''),'ts_code,trade_date,close,total_mv,total_share,industry',7000)
            except Exception as exc:
                errors.append({'api':'bak_daily','date':day,'error':str(exc)})
                if len(errors)>3:
                    break
            if number%12==0:
                print(f'official backup snapshots: {number+1}/{len(dates)}',flush=True)
            store.save_manifest()
    finally:
        store.save_manifest()
        dump(root/'data/raw/repair_status.json',dict(errors=errors,network_requests=store.network_requests))
