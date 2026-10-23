"""Descriptive performance, event exposure and pre-specified mechanism checks.

A descriptive mean may use available observations, but full-period performance
and HAC inference require all expected consecutive months. Missing risk follow-up
is always reported separately from an observed absence of an event.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _months(values: pd.Series) -> pd.Series:
    return values.map(lambda value: pd.Period(value, freq="M"))


def _periods(config: dict, annual: bool = False) -> dict[str, pd.PeriodIndex]:
    train = pd.period_range(config["train_start"], config["train_end"], freq="M")
    test = pd.period_range(config["test_start"], config["test_end"], freq="M")
    all_months = pd.period_range(min(train.min(), test.min()), max(train.max(), test.max()), freq="M")
    result = {"TRAIN": train, "TEST": test, "ALL": all_months}
    if annual:
        for year in sorted(set(all_months.year)):
            result[str(year)] = all_months[all_months.year == year]
    return result


def _monthly_input(monthly: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = monthly.copy()
    required = {"strategy", "return_month", "net_return", "turnover"}
    absent = required - set(data)
    if absent:
        raise ValueError(f"Monthly data missing columns: {sorted(absent)}")
    data["return_month"] = _months(data["return_month"])
    if "cost_bps" not in data:
        data["cost_bps"] = config["main_cost_bps"]
    keys = ["strategy", "cost_bps", "return_month"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate strategy/cost/month observations")
    for column in ("net_return", "turnover"):
        data[column] = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return data


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _hac(values: pd.Series, expected: pd.PeriodIndex, lags: int) -> dict[str, Any]:
    """Intercept-only HAC with a fixed lag and no deletion of calendar gaps."""
    if values.index.duplicated().any():
        raise ValueError("Duplicate month in inference input")
    aligned = pd.to_numeric(values.reindex(expected), errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = aligned.dropna()
    complete = len(observed) == len(expected)
    result = {
        "n_expected": len(expected), "n_valid": len(observed), "complete": complete,
        "mean": float(observed.mean()) if len(observed) else np.nan,
        "hac_lags": lags, "hac_se": np.nan, "ci95_low": np.nan,
        "ci95_high": np.nan, "t_stat": np.nan,
        "inference_status": "missing_months" if not complete else "too_few_months",
    }
    if not complete or len(observed) <= max(1, lags):
        return result
    fit = sm.OLS(observed.to_numpy(dtype=float), np.ones((len(observed), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True}
    )
    mean, se = float(fit.params[0]), float(fit.bse[0])
    result.update({
        "hac_se": se, "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
        "t_stat": mean / se if se > 0 else np.nan, "inference_status": "ok",
    })
    return result


def metrics(monthly: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Metrics by strategy, cost and pre-specified period, including each year."""
    data = _monthly_input(monthly, config)
    risk_free = float(config.get("risk_free_rate_annual", 0.0))
    if not np.isfinite(risk_free) or risk_free <= -1:
        raise ValueError("Annual risk-free return must be finite and greater than -1")
    risk_free_monthly = (1 + risk_free) ** (1 / 12) - 1
    rows = []
    for (strategy, cost_bps), frame in data.groupby(["strategy", "cost_bps"], sort=True):
        frame = frame.set_index("return_month").sort_index()
        for period, expected in _periods(config, annual=True).items():
            aligned = frame.reindex(expected)
            returns = aligned["net_return"]
            valid = returns.notna() & returns.ge(-1)
            observed = returns[valid]
            complete = bool(valid.all())
            turnover = aligned["turnover"].dropna()
            std = float(observed.std(ddof=1)) if len(observed) > 1 else np.nan
            mean = float(observed.mean()) if len(observed) else np.nan
            row = {
                "strategy": strategy, "cost_bps": cost_bps, "period": period,
                "n_expected": len(expected), "n_valid": int(valid.sum()),
                "complete": complete, "mean": mean, "monthly_std": std,
                "cagr": np.nan, "annualized_mean": np.nan,
                "return_vol_ratio": np.nan, "max_drawdown": np.nan,
                "sharpe": np.nan, "risk_free_rate_annual": risk_free,
                "worst_month": float(observed.min()) if len(observed) else np.nan,
                "turnover": float(turnover.mean()) if len(turnover) else np.nan,
                "n_turnover_valid": len(turnover), "ending_nav": np.nan,
                "status": "ok" if complete else "incomplete_returns",
            }
            if complete and len(expected):
                nav = np.r_[1.0, np.cumprod(1 + returns.to_numpy(dtype=float))]
                peaks = np.maximum.accumulate(nav)
                row.update({
                    "ending_nav": float(nav[-1]),
                    "cagr": float(nav[-1] ** (12 / len(expected)) - 1),
                    "annualized_mean": mean * 12,
                    "return_vol_ratio": math.sqrt(12) * mean / std if std > 0 else np.nan,
                    "sharpe": math.sqrt(12) * (mean - risk_free_monthly) / std if std > 0 else np.nan,
                    "max_drawdown": float(np.min(nav / peaks - 1)),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(monthly: pd.DataFrame, winner: str, config: dict) -> dict:
    """Candidates versus the smallest 100 and all bands versus the benchmark.

    These descriptive comparisons do not select or replace the winner. CIs
    concern mean monthly differences only, not differences between CAGRs.
    """
    data = _monthly_input(monthly, config)
    data = data[data["cost_bps"] == config["main_cost_bps"]]
    wide = data.pivot(index="return_month", columns="strategy", values="net_return")
    lags = int(config.get("hac_lags", 6))
    result = {"winner": winner, "cost_bps": config["main_cost_bps"], "comparisons": []}
    candidates = list(config["candidates"])
    if winner not in candidates:
        raise ValueError("Frozen winner must be one of the configured candidates")
    reference = config.get("reference", "R001_100")
    comparisons = [(candidate, reference)
                   for candidate in candidates if candidate != reference]
    if config.get("benchmark"):
        comparisons.extend((band, "CSI1000") for band in config["buckets"])
    else:
        comparisons.append((winner, "PEERS"))
    for strategy, comparator in comparisons:
        for period in ("TRAIN", "TEST"):
            expected = _periods(config)[period]
            pair = wide.reindex(index=expected, columns=[strategy, comparator])
            pair = pair.where(pair.ge(-1))
            delta = pair[strategy] - pair[comparator]
            inference = _hac(delta, expected, lags)
            valid = pair.dropna()
            corr = np.nan
            if len(valid) > 1 and valid[strategy].std() > 0 and valid[comparator].std() > 0:
                corr = float(valid[strategy].corr(valid[comparator]))
            result["comparisons"].append({
                "strategy": strategy, "comparator": comparator, "period": period,
                **inference, "correlation": corr,
                "mean_monthly_difference": inference["mean"],
                "ci_estimand": "mean monthly net return difference, not CAGR difference",
            })
    return result


_RISK_COUNTS = [
    "n_observed", "n_full_target", "n_missing_target", "n_current_st",
    "n_current_known", "n_current_unknown", "n_current_nonst", "new_st_events",
    "new_st_followup_known", "new_st_followup_unknown", "delist_events",
    "delist_followup_known", "delist_followup_unknown",
]


def _risk_rates(row: dict) -> dict:
    row.update({
        "current_st_exposure_rate": _ratio(row["n_current_st"], row["n_full_target"]),
        "current_st_known_rate": _ratio(row["n_current_st"], row["n_current_known"]),
        "current_unknown_exposure_rate": _ratio(row["n_current_unknown"], row["n_full_target"]),
        "missing_target_rate": _ratio(row["n_missing_target"], row["n_full_target"]),
        "new_st_event_lower_bound_rate": _ratio(row["new_st_events"], row["n_current_nonst"]),
        "new_st_known_followup_rate": _ratio(row["new_st_events"], row["new_st_followup_known"]),
        "new_st_followup_coverage": _ratio(row["new_st_followup_known"], row["n_current_nonst"]),
        "delist_event_lower_bound_rate": _ratio(row["delist_events"], row["n_observed"]),
        "delist_known_followup_rate": _ratio(row["delist_events"], row["delist_followup_known"]),
        "delist_followup_coverage": _ratio(row["delist_followup_known"], row["n_observed"]),
    })
    return row


def risk_statistics(ranked: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event counts and company-month denominators, without future selection."""
    required = {"return_month", "stock_code", "bucket", "st_current", "new_st_next", "delisted_next"}
    missing = required - set(ranked)
    if missing:
        raise ValueError(f"Risk data missing columns: {sorted(missing)}")
    data = ranked.copy()
    data["return_month"] = _months(data["return_month"])
    if data.duplicated(["return_month", "stock_code"]).any():
        raise ValueError("Duplicate stock/month risk observations")
    for column in ("st_current", "new_st_next", "delisted_next"):
        invalid = data[column].notna() & ~data[column].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{column} must contain only 0, 1, or missing")
    groups = {key: frame for key, frame in data.groupby(["return_month", "bucket"], observed=True)}
    empty = data.iloc[:0]
    monthly = []
    for month in _periods(config)["ALL"]:
        for bucket, (low, high) in config["buckets"].items():
            frame = groups.get((month, bucket), empty)
            target = high - low + 1
            if len(frame) > target:
                raise ValueError(f"{month} {bucket} has more holdings than target")
            current = frame["st_current"]
            nonst = current.eq(0)
            new = frame.loc[nonst, "new_st_next"]
            delist = frame["delisted_next"]
            row = {
                "return_month": str(month), "bucket": bucket,
                "n_observed": len(frame), "n_full_target": target,
                "n_missing_target": target - len(frame),
                "n_current_st": int(current.eq(1).sum()),
                "n_current_known": int(current.notna().sum()),
                "n_current_unknown": int(current.isna().sum()),
                "n_current_nonst": int(nonst.sum()),
                "new_st_events": int(new.eq(1).sum()),
                "new_st_followup_known": int(new.notna().sum()),
                "new_st_followup_unknown": int(new.isna().sum()),
                "delist_events": int(delist.eq(1).sum()),
                "delist_followup_known": int(delist.notna().sum()),
                "delist_followup_unknown": int(delist.isna().sum()),
            }
            monthly.append(_risk_rates(row))
    monthly_df = pd.DataFrame(monthly)
    monthly_df["_month"] = _months(monthly_df["return_month"])
    summary = []
    for period, expected in _periods(config, annual=True).items():
        for bucket, frame in monthly_df[monthly_df["_month"].isin(expected)].groupby("bucket", sort=True):
            row = {column: int(frame[column].sum()) for column in _RISK_COUNTS}
            row.update({"bucket": bucket, "period": period, "n_months": len(expected)})
            row["complete_holdings"] = row["n_missing_target"] == 0
            summary.append(_risk_rates(row))
    return monthly_df.drop(columns="_month"), pd.DataFrame(summary)


def regressions(ranked: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly cross-sectional OLS on a shared complete-case sample.

    Predictor winsorization is fitted to the entire ranked cross-section before
    future-return availability is examined. No response winsorization is used.
    Rank-deficient designs do not silently obtain pseudo-inverse estimates.
    """
    required = {"return_month", "stock_code", "bucket", "return_1m", "industry", "adv20", "return_prev1m", "is_loss"}
    missing = required - set(ranked)
    if missing:
        raise ValueError(f"Regression data missing columns: {sorted(missing)}")
    data = ranked.copy()
    data["return_month"] = _months(data["return_month"])
    if data.duplicated(["return_month", "stock_code"]).any():
        raise ValueError("Duplicate stock/month regression observations")
    data = data[data["bucket"].isin(config["buckets"])].copy()
    # Selection eligibility and regression coding are independent: the omitted
    # reference remains the smallest band even when it competes for selection.
    reference = config.get("reference", "R001_100")
    regression_bands = [band for band in config["buckets"] if band != reference]
    coeff_rows, status_rows = [], []
    groups = {key: frame for key, frame in data.groupby("return_month")}
    min_obs = int(config.get("regression_min_obs", 30))
    for month in _periods(config)["ALL"]:
        cross = groups.get(month, data.iloc[:0]).copy()
        adv = pd.to_numeric(cross["adv20"], errors="coerce")
        cross["log_adv20"] = np.log(adv.where(adv.gt(0))).replace([np.inf, -np.inf], np.nan)
        cross["return_prev1m"] = pd.to_numeric(cross["return_prev1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        bounds = {}
        for term in ("log_adv20", "return_prev1m"):
            values = cross[term]
            lower, upper = values.quantile([0.01, 0.99])
            cross[term] = values.clip(lower=lower, upper=upper)
            bounds[f"{term}_p01"], bounds[f"{term}_p99"] = lower, upper
        cross["return_1m"] = pd.to_numeric(cross["return_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        cross["is_loss"] = cross["is_loss"].where(cross["is_loss"].isin([0, 1]))
        cross["industry"] = cross["industry"].replace(r"^\s*$", np.nan, regex=True)
        needed = ["return_1m", "log_adv20", "return_prev1m", "is_loss", "industry"]
        sample = cross.dropna(subset=needed).copy()
        sample = sample[sample["return_1m"] >= -1]
        n = len(sample)
        basic = pd.DataFrame({"const": np.ones(n)}, index=sample.index)
        for bucket in regression_bands:
            basic[bucket] = sample["bucket"].eq(bucket).astype(float)
        full = basic.copy()
        for term in ("log_adv20", "return_prev1m", "is_loss"):
            full[term] = sample[term].astype(float)
        industry = pd.get_dummies(sample["industry"].astype(str), prefix="industry", drop_first=True, dtype=float)
        full = pd.concat([full, industry], axis=1)
        model_status = {}
        for name, design in (("baseline", basic), ("full", full)):
            if n < max(min_obs, design.shape[1] + 1):
                model_status[name] = "too_few_observations"
            elif np.linalg.matrix_rank(design.to_numpy(dtype=float)) < design.shape[1]:
                model_status[name] = "rank_deficient"
            elif np.linalg.cond(design.to_numpy(dtype=float)) > 1e12:
                model_status[name] = "ill_conditioned"
            else:
                model_status[name] = "ok"
        # Paired models are published only for jointly identifiable months.
        if all(value == "ok" for value in model_status.values()):
            for name, design in (("baseline", basic), ("full", full)):
                fit = sm.OLS(sample["return_1m"].to_numpy(dtype=float), design).fit(method="qr")
                for term, value in fit.params.items():
                    coeff_rows.append({
                        "return_month": str(month), "model": name, "term": term,
                        "coefficient": float(value), "n_obs": n, "r_squared": float(fit.rsquared),
                    })
        for bucket in config["buckets"]:
            original = int(cross["bucket"].eq(bucket).sum())
            complete = int(sample["bucket"].eq(bucket).sum())
            status_rows.append({
                "return_month": str(month), "bucket": bucket,
                "n_ranked": original, "n_complete": complete,
                "coverage": _ratio(complete, original), "n_model_obs": n,
                "baseline_status": model_status["baseline"], "full_status": model_status["full"],
                "n_full_parameters": full.shape[1], **bounds,
            })
    coefficients = pd.DataFrame(coeff_rows, columns=["return_month", "model", "term", "coefficient", "n_obs", "r_squared"])
    status = pd.DataFrame(status_rows)
    summary_rows = []
    coefficients["_month"] = _months(coefficients["return_month"])
    for period in ("TRAIN", "TEST"):
        expected = _periods(config)[period]
        for model in ("baseline", "full"):
            terms = regression_bands + (["log_adv20", "return_prev1m", "is_loss"] if model == "full" else [])
            for term in terms:
                frame = coefficients[(coefficients["model"] == model) & (coefficients["term"] == term)]
                values = frame.set_index("_month")["coefficient"]
                summary_rows.append({
                    "period": period, "model": model, "term": term,
                    **_hac(values, expected, int(config.get("hac_lags", 6))),
                })
    return coefficients.drop(columns="_month"), status, pd.DataFrame(summary_rows)
