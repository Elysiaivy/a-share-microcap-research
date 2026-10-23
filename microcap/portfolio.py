"""Deterministic market-cap portfolios and a strictly in-sample selector.

The portfolios are research portfolios. Costs use the approximation
``net_return = gross_return - cost`` and do not change the pretrade weights.
No execution, limit-down fill, or financing assumptions are hidden here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BUCKETS = {
    "R001_100": (1, 100),
    "R101_200": (101, 200),
    "R201_300": (201, 300),
    "R301_400": (301, 400),
    "R401_500": (401, 500),
}


def _require(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def rank_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank each historical eligible universe; ST and outcomes are unused.

    All source columns are retained for eligible, positive finite market-cap
    observations, including ranks beyond 500. The latter have a null bucket.
    ``eligible`` must contain actual booleans (missing means not eligible).
    """
    _require(panel, ["signal_date", "horizon_end", "return_month", "stock_code",
                     "market_cap", "eligible"])
    if panel[["signal_date", "stock_code", "return_month"]].isna().any().any():
        raise ValueError("Signal date, return month and stock code must be known")
    if panel.duplicated(["signal_date", "stock_code"]).any():
        raise ValueError("Duplicate stock within signal date")
    eligible_values = panel["eligible"].dropna()
    if not eligible_values.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("eligible must contain booleans, not strings or integers")
    caps = pd.to_numeric(panel["market_cap"], errors="coerce")
    keep = panel["eligible"].eq(True).fillna(False) & np.isfinite(caps) & caps.gt(0)
    ranked = panel.loc[keep].copy()
    ranked["market_cap"] = caps.loc[keep]
    ranked["stock_code"] = ranked["stock_code"].astype(str)
    tie_code = "historical_code" if "historical_code" in ranked else "stock_code"
    ranked = ranked.sort_values(["signal_date", "market_cap", tie_code], kind="stable")
    ranked["rank"] = ranked.groupby("signal_date", sort=False).cumcount() + 1
    ranked["bucket"] = pd.Series(pd.NA, index=ranked.index, dtype="string")
    for name, (lo, hi) in DEFAULT_BUCKETS.items():
        ranked.loc[ranked["rank"].between(lo, hi), "bucket"] = name
    return ranked.reset_index(drop=True)


def peer_panel(ranked: pd.DataFrame, winner: str) -> pd.DataFrame:
    """Return the three other candidate bands as one 300-stock PEERS pool.

    Combining holdings before costs avoids charging trades for stocks that
    move between the three constituent bands but remain in this portfolio.
    """
    _require(ranked, ["bucket", "rank"])
    candidates = list(DEFAULT_BUCKETS)[1:]
    if winner not in candidates:
        raise ValueError(f"Unknown candidate winner: {winner}")
    peers = ranked.loc[ranked["bucket"].isin([x for x in candidates if x != winner])].copy()
    peers["source_bucket"] = peers["bucket"]
    peers["bucket"] = "PEERS"
    return peers.reset_index(drop=True)


def simulate(
    ranked: pd.DataFrame,
    cost_bps: float,
    buckets: Mapping[str, tuple[int, int] | list[int] | int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate monthly equal weights without imputing unavailable returns.

    Range-valued buckets select original ranks and infer the target size;
    integer-valued buckets select existing bucket labels and specify the
    target size. Use ``{"PEERS": 300}`` with :func:`peer_panel`.

    Turnover is the *sum* of absolute stock-weight changes, including exits;
    it is not divided by two. Initial wealth is cash, so initial turnover is
    1. A missing/invalid return or missing constituent makes gross return NA,
    never a reweighted average. The next pretrade weight vector is unknown.
    A subsequent full target portfolio can define a new end-of-month weight
    vector, but that month's net return remains NA if its entry cost is
    unknown. Do not chain NAV across such gaps.
    """
    _require(ranked, ["signal_date", "horizon_end", "return_month", "stock_code",
                      "rank", "bucket", "return_1m"])
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and nonnegative")
    definitions = dict(DEFAULT_BUCKETS if buckets is None else buckets)
    if not definitions:
        raise ValueError("At least one portfolio is required")
    sizes: dict[str, int] = {}
    for strategy, definition in definitions.items():
        if isinstance(definition, (tuple, list)) and len(definition) == 2:
            lo, hi = definition
            if int(lo) != lo or int(hi) != hi or lo < 1 or hi < lo:
                raise ValueError(f"Invalid rank interval for {strategy}")
            sizes[strategy] = int(hi - lo + 1)
        elif isinstance(definition, (int, np.integer)) and definition > 0:
            sizes[strategy] = int(definition)
        else:
            raise ValueError(f"Invalid portfolio definition for {strategy}")
    data = ranked.copy()
    data["return_month"] = data["return_month"].astype(str)
    if data.duplicated(["return_month", "stock_code"]).any():
        raise ValueError("Duplicate stock within return month")
    for key in ("signal_date", "horizon_end"):
        if data.groupby("return_month")[key].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent {key} within return month")
    previous: dict[str, dict[str, float] | None] = {name: {} for name in definitions}
    previous_month: pd.Period | None = None
    monthly: list[dict[str, Any]] = []
    held_frames: list[pd.DataFrame] = []
    for month, universe in data.groupby("return_month", sort=True):
        period = pd.Period(month, freq="M")
        if previous_month is not None and period != previous_month + 1:
            previous = {name: None for name in definitions}
        for strategy, definition in definitions.items():
            if isinstance(definition, (tuple, list)):
                target = universe.loc[universe["rank"].between(*definition)].copy()
            else:
                target = universe.loc[universe["bucket"].eq(strategy)].copy()
            target = target.sort_values("stock_code", kind="stable")
            expected = sizes[strategy]
            n_target = len(target)
            if n_target > expected:
                raise ValueError(f"{strategy} has more stocks than its target size")
            target_weights = dict.fromkeys(target["stock_code"].astype(str), 1.0 / expected)
            returns = pd.to_numeric(target["return_1m"], errors="coerce")
            finite = np.isfinite(returns)
            valid = finite & returns.ge(-1)
            n_returns = int(valid.sum())
            flags = []
            if n_target != expected:
                flags.append("incomplete_target")
            if (~finite).any():
                flags.append("missing_returns")
            if (finite & returns.lt(-1)).any():
                flags.append("invalid_returns")
            complete = n_target == expected and n_returns == expected
            gross = float(returns.sum() / expected) if complete else float("nan")
            old_weights = previous[strategy]
            if old_weights is None:
                turnover = exit_turnover = float("nan")
                flags.append("unknown_pretrade")
            else:
                all_codes = set(old_weights) | set(target_weights)
                turnover = sum(abs(target_weights.get(code, 0.0) - old_weights.get(code, 0.0))
                               for code in all_codes)
                exit_turnover = sum(weight for code, weight in old_weights.items()
                                    if code not in target_weights)
            cost = float(cost_bps / 10_000 * turnover)
            net = float(gross - cost)
            if complete and gross > -1:
                previous[strategy] = {
                    str(code): float((1.0 / expected) * (1.0 + value) / (1.0 + gross))
                    for code, value in zip(target["stock_code"], returns, strict=True)
                }
            else:
                previous[strategy] = None
                if complete:
                    flags.append("zero_terminal_wealth")
            target["strategy"] = strategy
            target["target_weight"] = 1.0 / expected
            target["pretrade_weight"] = (
                target["stock_code"].astype(str).map(lambda x: old_weights.get(x, 0.0))
                if old_weights is not None else np.nan
            )
            target["weight_change"] = target["target_weight"] - target["pretrade_weight"]
            target["valid_return"] = valid
            target["return_contribution"] = returns.where(valid) / expected
            held_frames.append(target)
            monthly.append({
                "return_month": month,
                "signal_date": universe["signal_date"].iloc[0],
                "horizon_end": universe["horizon_end"].iloc[0],
                "strategy": strategy,
                "n_target": n_target,
                "n_expected": expected,
                "n_returns": n_returns,
                "gross_return": gross,
                "turnover": float(turnover),
                "exit_turnover": float(exit_turnover),
                "cost_bps": float(cost_bps),
                "cost": cost,
                "net_return": net,
                "status": "|".join(flags) if flags else "ok",
            })
        previous_month = period
    monthly_columns = ["return_month", "signal_date", "horizon_end", "strategy", "n_target",
                       "n_expected", "n_returns", "gross_return", "turnover", "exit_turnover",
                       "cost_bps", "cost", "net_return", "status"]
    holding_columns = list(ranked.columns) + ["strategy", "target_weight", "pretrade_weight",
                                             "weight_change", "valid_return", "return_contribution"]
    holdings = (pd.concat(held_frames, ignore_index=True) if held_frames
                else pd.DataFrame(columns=holding_columns))
    return pd.DataFrame(monthly, columns=monthly_columns), holdings


def select_winner(monthly: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Freeze the winning band using only complete configured training months.

    No test rows, ST labels, or whole-file checksum enter the training hash.
    Configuration and outcome fields on which the decision depends do enter
    it. Exactly tied scores use lower mean turnover, then the larger band.
    """
    _require(monthly, ["return_month", "strategy", "net_return", "turnover", "status"])
    start = pd.Period(config["train_start"], freq="M")
    end = pd.Period(config["train_end"], freq="M")
    months = pd.period_range(start, end, freq="M").astype(str).tolist()
    if len(months) < 2:
        raise ValueError("Training window needs at least two months")
    candidates = list(config["candidates"])
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Candidates must be a nonempty unique list")
    train = monthly.loc[monthly["return_month"].astype(str).isin(months)
                        & monthly["strategy"].isin(candidates)].copy()
    train["return_month"] = train["return_month"].astype(str)
    train = train.sort_values(["return_month", "strategy"], kind="stable")
    if train.duplicated(["return_month", "strategy"]).any():
        raise ValueError("Duplicate candidate-month observations")
    scores = {}
    for candidate in candidates:
        rows = train.loc[train["strategy"].eq(candidate)]
        if sorted(rows["return_month"].tolist()) != months:
            raise ValueError(f"{candidate}: incomplete training months (need {len(months)})")
        numeric = rows[["net_return", "turnover"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all() or not rows["status"].eq("ok").all():
            raise ValueError(f"{candidate}: unresolved training returns or weights")
        if numeric["turnover"].lt(0).any():
            raise ValueError(f"{candidate}: negative turnover")
        if "cost_bps" in rows and not rows["cost_bps"].eq(config["main_cost_bps"]).all():
            raise ValueError(f"{candidate}: selection must use the configured primary cost")
        if {"n_target", "n_expected", "n_returns"}.issubset(rows.columns):
            if not (rows["n_target"].eq(rows["n_expected"])
                    & rows["n_returns"].eq(rows["n_expected"])).all():
                raise ValueError(f"{candidate}: incomplete training constituents")
        std = float(numeric["net_return"].std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"{candidate}: sample standard deviation must be positive")
        lo, hi = config["buckets"][candidate]
        mean = float(numeric["net_return"].mean())
        scores[candidate] = {
            "score": float(np.sqrt(12) * mean / std),
            "mean_return": mean,
            "sample_std": std,
            "mean_turnover": float(numeric["turnover"].mean()),
            "n_months": len(rows),
            "rank_lower": int(lo),
            "rank_upper": int(hi),
        }
    winner = min(candidates, key=lambda key: (
        -scores[key]["score"], scores[key]["mean_turnover"], -scores[key]["rank_lower"], key
    ))
    protocol = {
        "train_start": str(start), "train_end": str(end),
        "candidates": candidates,
        "buckets": {key: list(config["buckets"][key]) for key in candidates},
        "main_cost_bps": config["main_cost_bps"],
        "metric": "sqrt(12)*monthly_net_mean/monthly_net_sample_std(ddof=1)",
        "tie_break": ["lower_mean_turnover", "larger_rank_lower"],
        "cost_convention": (
            "actual executed notional * one-way bps, funded from cash, daily NAV"
            if config.get("execution") == "prior_close_orders_next_open_funded"
            else "bps/10000*sum_abs_weight_changes; net=gross-cost"
        ),
    }
    hash_columns = ["return_month", "strategy", "net_return", "turnover", "status"]
    hash_columns += [key for key in ("cost_bps", "n_target", "n_expected", "n_returns")
                     if key in train.columns]
    dependency_rows = train.sort_values(["return_month", "strategy"])[hash_columns]
    canonical = {"protocol": protocol,
                 "rows": json.loads(dependency_rows.to_json(orient="records", double_precision=15))}
    digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"winner": winner, "scores": scores, "training_hash": digest,
            "train_months": months, "protocol": protocol}
