"""Point-in-time profits, historical industry and ST/exit research labels.

Input data remain provider evidence rather than a certified vintage archive.
Missing histories and conflicting versions are deliberately not imputed. Name
intervals follow the existing project's inclusive Tushare end-date convention;
industry membership uses the requested left-closed/right-open convention.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd


PROFIT_COLUMNS = ["signal_date", "stock_code", "net_profit_ttm", "is_loss",
                  "report_date", "available_date", "input_trace", "ttm_status"]
AUDIT_COLUMNS = ["stock_code", "report_date", "announcement_date", "issue", "source_ids"]


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def _dates(values) -> pd.Series:
    """Parse ISO or YYYYMMDD dates without interpreting integers as epochs."""
    raw = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(raw):
        out = pd.to_datetime(raw, errors="coerce")
    else:
        out = pd.to_datetime(raw.astype("string").str.replace(r"\.0$", "", regex=True),
                             errors="coerce", format="mixed")
    if out.dt.tz is not None:
        raise ValueError("Use timezone-naive exchange dates")
    return out.dt.normalize()


def _source_id(row: dict) -> str:
    material = {key: None if pd.isna(value) else str(value) for key, value in row.items()}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:20]


def profit_ttm(reports: pd.DataFrame, signal_dates, calendar,
               max_age_days: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (as-of profit panel, source audit), preserving missing values.

    Only report_type=1 consolidated cumulative statements are accepted. A row
    becomes usable on the first trading date strictly after max(ann_date,
    f_ann_date). Equal-date conflicting versions invalidate that period until
    an unambiguous subsequent version is disclosed. The latest known quarter
    must be complete: this function never searches older quarters for a nicer
    or complete result. Year-end TTM is simply that annual income figure.
    """
    _require(reports, ["ts_code", "end_date", "ann_date", "report_type", "n_income_attr_p"])
    sessions = pd.DatetimeIndex(_dates(calendar))
    signals = pd.DatetimeIndex(_dates(signal_dates)).unique().sort_values()
    if sessions.hasnans or not sessions.is_unique or not sessions.is_monotonic_increasing:
        raise ValueError("calendar must contain unique increasing valid dates")
    if signals.hasnans or not signals.isin(sessions).all():
        raise ValueError("signal_dates must be valid sessions in calendar")
    if max_age_days < 0:
        raise ValueError("max_age_days cannot be negative")
    if reports.empty or not len(signals):
        return pd.DataFrame(columns=PROFIT_COLUMNS), pd.DataFrame(columns=AUDIT_COLUMNS)
    frame = reports.reset_index(drop=True).copy()
    if frame.ts_code.isna().any() or frame.ts_code.astype(str).str.strip().eq("").any():
        raise ValueError("ts_code cannot be missing")
    stocks = sorted(frame.ts_code.astype(str).unique())
    frame["source_id"] = [_source_id(row) for row in frame.to_dict("records")]
    frame["stock_code"] = frame.ts_code.astype(str)
    frame["report_date"] = _dates(frame.end_date)
    frame["ann"] = _dates(frame.ann_date)
    frame["f_ann"] = _dates(frame.get("f_ann_date", pd.Series(pd.NaT, index=frame.index)))
    frame["announcement_date"] = frame[["ann", "f_ann"]].max(axis=1)
    frame["value"] = pd.to_numeric(frame.n_income_attr_p, errors="coerce")
    frame.loc[~np.isfinite(frame.value), "value"] = np.nan
    frame["report_kind"] = pd.to_numeric(frame.report_type, errors="coerce")
    audit: list[dict] = []
    records = []
    for row in frame.to_dict("records"):
        issue = None
        if row["report_kind"] != 1:
            issue = "excluded_nonconsolidated_cumulative_type"
        elif pd.isna(row["report_date"]) or not row["report_date"].is_quarter_end:
            issue = "invalid_report_period"
        elif pd.isna(row["announcement_date"]) or row["announcement_date"] < row["report_date"]:
            issue = "invalid_announcement_date"
        else:
            position = sessions.searchsorted(row["announcement_date"], side="right")
            if position == len(sessions):
                issue = "announcement_beyond_calendar"
            else:
                row["available_date"] = sessions[position]
        if issue:
            audit.append({"stock_code": row["stock_code"], "report_date": row["report_date"],
                          "announcement_date": row["announcement_date"], "issue": issue,
                          "source_ids": json.dumps([row["source_id"]])})
        else:
            records.append(row)

    grouped: dict[tuple, list] = defaultdict(list)
    for row in records:
        grouped[(row["stock_code"], row["report_date"], row["announcement_date"])].append(row)
    by_stock: dict[str, list] = defaultdict(list)
    for key, versions in grouped.items():
        values = {"missing" if pd.isna(row["value"]) else float(row["value"]) for row in versions}
        conflict = len(values) > 1
        row = min(versions, key=lambda item: item["source_id"]).copy()
        row["source_ids"] = sorted({item["source_id"] for item in versions})
        row["conflict"] = conflict
        if conflict:
            row["value"] = np.nan
            audit.append({"stock_code": key[0], "report_date": key[1], "announcement_date": key[2],
                          "issue": "conflicting_same_date_versions", "source_ids": json.dumps(row["source_ids"])})
        elif pd.isna(row["value"]):
            audit.append({"stock_code": key[0], "report_date": key[1], "announcement_date": key[2],
                          "issue": "missing_profit", "source_ids": json.dumps(row["source_ids"])})
        by_stock[key[0]].append(row)

    result = []
    for stock in stocks:
        versions = sorted(by_stock[stock], key=lambda row: (row["available_date"], row["announcement_date"], row["report_date"]))
        cursor, known = 0, {}
        for signal in signals:
            while cursor < len(versions) and versions[cursor]["available_date"] <= signal:
                row = versions[cursor]
                known[row["report_date"]] = row
                cursor += 1
            output = dict(signal_date=signal, stock_code=stock, net_profit_ttm=np.nan, is_loss=np.nan,
                          report_date=pd.NaT, available_date=pd.NaT, input_trace="[]", ttm_status="no_known_report")
            if known:
                quarter = max(known)
                required = [quarter] if quarter.month == 12 else [quarter, pd.Timestamp(quarter.year - 1, 12, 31), quarter - pd.DateOffset(years=1)]
                output["report_date"] = quarter
                available = [known[period]["available_date"] for period in required if period in known]
                output["available_date"] = max(available)
                trace = []
                for position, period in enumerate(required):
                    source = known.get(period)
                    trace.append({"report_date": period.strftime("%Y-%m-%d"),
                                  "coefficient": -1 if position == 2 else 1,
                                  "available_date": source["available_date"].strftime("%Y-%m-%d") if source else None,
                                  "announcement_date": source["announcement_date"].strftime("%Y-%m-%d") if source else None,
                                  "value": float(source["value"]) if source and pd.notna(source["value"]) else None,
                                  "conflicting_versions": bool(source["conflict"]) if source else False,
                                  "source_ids": source["source_ids"] if source else []})
                output["input_trace"] = json.dumps(trace, ensure_ascii=False, sort_keys=True)
                if (signal - quarter).days > max_age_days:
                    output["ttm_status"] = "stale_report"
                elif any(period not in known for period in required):
                    output["ttm_status"] = "missing_comparison_period"
                elif any(known[period]["conflict"] for period in required):
                    output["ttm_status"] = "conflicting_version"
                elif any(pd.isna(known[period]["value"]) for period in required):
                    output["ttm_status"] = "missing_profit"
                else:
                    value = known[quarter]["value"]
                    if len(required) == 3:
                        value += known[required[1]]["value"] - known[required[2]]["value"]
                    output.update(net_profit_ttm=float(value), is_loss=float(value < 0), ttm_status="ok")
            result.append(output)
    return (pd.DataFrame(result, columns=PROFIT_COLUMNS).sort_values(["signal_date", "stock_code"], ignore_index=True),
            pd.DataFrame(audit, columns=AUDIT_COLUMNS))


def historical_industry(panel: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Attach historical industry and nullable financial flag; never current data.

    Membership dates are [in_date, out_date). Identical overlapping labels are
    harmless; conflicting labels are unknown. If the source includes ann_date,
    a membership record also has to have been announced before the signal.
    """
    _require(panel, ["signal_date", "stock_code"])
    _require(intervals, ["ts_code", "in_date", "out_date", "industry"])
    output = panel.reset_index(drop=True).copy()
    dates = _dates(output.signal_date)
    if dates.isna().any():
        raise ValueError("Invalid signal_date")
    source = intervals.reset_index(drop=True).copy()
    for column in ["in_date", "out_date"]:
        source[column] = _dates(source[column])
    has_ann = "ann_date" in source
    if has_ann:
        source["ann_date"] = _dates(source.ann_date)
    by_stock = {str(stock): rows for stock, rows in source.groupby("ts_code", sort=False)}
    labels, states, financial = [], [], []
    for index, row in output.iterrows():
        members = by_stock.get(str(row.stock_code))
        names = []
        if members is not None:
            active = (members.in_date.le(dates.iloc[index]) &
                      (members.out_date.isna() | members.out_date.gt(dates.iloc[index])))
            if has_ann:
                active &= members.ann_date.lt(dates.iloc[index])
            names = sorted(set(members.loc[active, "industry"].dropna().astype(str).str.strip()) - {""})
        label = names[0] if len(names) == 1 else None
        labels.append(label)
        states.append("ok" if label is not None else ("conflict" if len(names) > 1 else "unknown"))
        financial.append(np.nan if label is None else float(label in {"银行", "非银金融", "非银行金融", "金融", "银行业", "保险", "证券"}))
    output["industry"] = labels
    output["industry_status"] = states
    output["is_financial"] = financial
    return output


def _name_state(rows: list[dict], date: pd.Timestamp) -> tuple[float, str]:
    names = {row["name"] for row in rows
             if pd.notna(row["start_date"]) and pd.notna(row["ann_date"])
             and row["start_date"] <= date and row["ann_date"] < date
             and (pd.isna(row["end_date"]) or date <= row["end_date"])}
    if len(names) != 1:
        return np.nan, "conflict" if len(names) > 1 else "unknown"
    name = next(iter(names))
    # PT is a historical special-treatment label; included for completeness.
    return float("ST" in name or "PT" in name), "ok"


def st_labels(panel: pd.DataFrame, names: pd.DataFrame, securities: pd.DataFrame) -> pd.DataFrame:
    """Attach as-of ST status and ex-post next-month ST/delisting labels.

    All name intervals are inclusive at their published end. Announcement-date
    records are visible from the following day, hence not on the announcement
    trading session. Unknown intervals stay unknown, including before the first
    name observation. New ST is observed anywhere in (signal, horizon], even if
    subsequently revoked; a known event is retained despite other coverage gaps.
    No future label is intended for use in historical selection.
    """
    _require(panel, ["stock_code", "signal_date", "horizon_end"])
    _require(names, ["ts_code", "name", "start_date", "end_date", "ann_date"])
    _require(securities, ["ts_code", "list_date", "delist_date"])
    output = panel.reset_index(drop=True).copy()
    signals, horizons = _dates(output.signal_date), _dates(output.horizon_end)
    if signals.isna().any() or horizons.isna().any() or horizons.le(signals).any():
        raise ValueError("Each horizon_end must be after a valid signal_date")
    source = names.reset_index(drop=True).copy()
    for column in ["start_date", "end_date", "ann_date"]:
        source[column] = _dates(source[column])
    source = source[source.name.notna()].copy()
    source["name"] = source.name.astype(str).str.strip().str.upper()
    source = source[source.name.ne("")].drop_duplicates()
    by_stock = {str(stock): rows.to_dict("records") for stock, rows in source.groupby("ts_code", sort=False)}
    stocks = securities.reset_index(drop=True).copy()
    stocks["list_date"], stocks["delist_date"] = _dates(stocks.list_date), _dates(stocks.delist_date)
    if "list_status" not in stocks:
        stocks["list_status"] = None
    stock_rows = {}
    for stock, rows in stocks.groupby("ts_code", sort=False):
        unique = rows[["list_date", "delist_date", "list_status"]].drop_duplicates()
        stock_rows[str(stock)] = unique.iloc[0].to_dict() if len(unique) == 1 else None

    labels = []
    for index, row in output.iterrows():
        signal, horizon = signals.iloc[index], horizons.iloc[index]
        intervals = by_stock.get(str(row.stock_code), [])
        current, status = _name_state(intervals, signal)
        info = stock_rows.get(str(row.stock_code))
        delisted = np.nan
        end = horizon
        if info is not None and pd.notna(info["list_date"]) and info["list_date"] <= signal:
            exit_date = info["delist_date"]
            if pd.notna(exit_date):
                delisted = float(signal < exit_date <= horizon)
                if exit_date <= signal:
                    current, status = np.nan, "not_listed"
                else:
                    end = min(horizon, exit_date)
            elif str(info["list_status"]).upper() in {"L", "P"}:
                delisted = 0.0
        # Only boundaries can change state; scanning them catches intramonth
        # events while avoiding a stock x calendar-day expansion.
        points = {signal + pd.Timedelta(days=1), end}
        for interval in intervals:
            for key, offset in [("start_date", 0), ("ann_date", 1), ("end_date", 1)]:
                date = interval[key]
                if pd.notna(date):
                    point = date + pd.Timedelta(days=offset)
                    if signal < point <= end:
                        points.add(point)
        future = [_name_state(intervals, point)[0] for point in sorted(points) if signal < point <= end]
        future_unknown = not future or any(pd.isna(value) for value in future)
        new_st = np.nan
        if current == 0:
            if any(value == 1 for value in future):
                new_st = 1.0
            elif not future_unknown:
                new_st = 0.0
        labels.append(dict(st_current=current, st_status=status, new_st_next=new_st,
                           st_future_unknown=float(future_unknown), delisted_next=delisted))
    for column in ["st_current", "st_status", "new_st_next", "st_future_unknown", "delisted_next"]:
        output[column] = [row[column] for row in labels]
    return output
