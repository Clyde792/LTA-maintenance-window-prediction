"""
Headway - per-cycle to per-asset-day features.

The unit matters, and getting it wrong once already inverted a result. An
engineer does not receive 200 alerts for one door; they receive one line per
door per day. So the asset-day is the row the whole downstream system reasons
about, and the alert budget is denominated in asset-days.

Three families of feature come out of a day's cycles:

  LEVEL       median of each signal, and of the condition-normalised residual.
              Where the degradation eventually shows up.
  DISPERSION  inter-quartile range. The hypothesis was that a worn mechanism is
              not just higher but more VARIABLE - some cycles bind, some do not -
              and that dispersion moves before level does.
              MEASURED, AND IT DOES NOT, HERE: scripts/diagnose.py finds the IQR
              features detect 3-4 of 9 episodes at 2-4% precision. The reason is
              a limitation of our synthetic data, not a fact about doors - the
              generator scales the MEAN of each signal with wear and leaves the
              variance alone, so there is no dispersion signal to find. The rail
              literature says real worn mechanisms do show one. Until the
              generator models it we are blind to this whole feature family, so
              the columns are carried but must not be claimed as a contribution.
  RATE        obstruction and retry rates. Note these are driven overwhelmingly
              by crowding, not wear: a model that chases them learns the
              peak-hour timetable. Carried so the tournament can demonstrate
              that trap rather than fall into it.

Then trend features on the daily series - the smoothed index, its slope in
sigma/day, and its volatility. The slope is what RUL divides into a threshold to
get a margin, so it is the single most consequential number in the file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import contract

# Columns aggregated as a rate (mean) rather than a level (median + spread).
# Suffix-based so a new subsystem needs no edit here.
RATE_SUFFIXES = ("_flag", "_count")

# The daily quality contract, in one place. Downstream code that needs to reason
# about a SINGLE channel must not re-derive these: `data_quality_ok` folds the
# cross-channel completeness minimum together with the channel-agnostic terms,
# so one degraded channel voids the whole day. `quality_ready` carries exactly
# the channel-agnostic part, and the invariant downstream may rely on is
#
#     data_quality_ok == quality_ready & (min over ALL channels' completeness >= MIN_COMPLETENESS)
#
# Anything that widens `data_quality_ok` must widen `quality_ready` in step, or
# the invariant breaks and consumers fall back to the global flag.
MIN_COMPLETENESS = 0.9
MIN_CYCLES = 5
READINESS_FLAG = "quality_ready"


def _is_rate(col: str) -> bool:
    return col.endswith(RATE_SUFFIXES)


def _iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


def to_daily(
    cycles: pd.DataFrame,
    subsystem: str,
    extra: tuple[str, ...] = ("residual",),
) -> pd.DataFrame:
    """Collapse cycle-level telemetry into one row per asset per day.

    Args:
        cycles: a contract-shaped frame, ideally already carrying `residual`
            from ConditionNormaliser.
        subsystem: "door" or "bogie".
        extra: additional numeric columns to aggregate (residual, by default).

    Returns:
        One row per (asset_id, day), with level/dispersion/rate features, the
        day's mean operating conditions, and the fault label carried through.
    """
    sub = contract.get(subsystem)
    df = cycles.copy()
    df["day"] = df["ts"].dt.floor("D")

    levels = [c for c in sub.signals if not _is_rate(c)] + [
        c for c in extra if c in df.columns
    ]
    rates = [c for c in sub.signals if _is_rate(c)]

    agg: dict[str, tuple[str, object]] = {"n_cycles": ("ts", "size"),
        "last_observed_at": ("ts", "max")}
    # Batch aggregation avoids a Python callback for every signal/asset/day.
    # Infinite observations are missing, not extreme but valid medians.
    df[levels] = df[levels].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "context_supported" in df:
        agg["context_supported"] = ("context_supported", "all")
    for c in levels:
        agg[c] = (c, "median")
    for c in rates:
        agg[f"{c}_rate"] = (c, "mean")
    for c in contract.CONTEXT:
        if c in df.columns:
            agg[f"{c}_mean"] = (c, "mean")
    if "fault_confirmed" in df.columns:
        agg["fault_confirmed"] = ("fault_confirmed", "max")

    daily = (
        df.groupby(["asset_id", "day"], as_index=False)
          .agg(**agg)
          .sort_values(["asset_id", "day"], ignore_index=True)
    )
    grouped = df.groupby(["asset_id", "day"])[levels]
    spread = (grouped.quantile(.75) - grouped.quantile(.25)).add_suffix("_iqr")
    complete = df[levels].notna().groupby([df.asset_id, df.day]).mean().add_suffix("_completeness")
    daily = daily.merge(spread, on=["asset_id", "day"], validate="one_to_one")
    daily = daily.merge(complete, on=["asset_id", "day"], validate="one_to_one")

    # train_id is constant within an asset; carry it for fleet-level grouping.
    if "train_id" in df.columns:
        daily["train_id"] = daily["asset_id"].map(
            df.groupby("asset_id")["train_id"].first()
        )
    daily["available_at"] = daily.day + pd.Timedelta(days=1)
    completeness = [f"{c}_completeness" for c in levels]
    daily[READINESS_FLAG] = daily.n_cycles >= MIN_CYCLES
    daily["data_quality_ok"] = (daily[completeness].min(axis=1) >= MIN_COMPLETENESS) & daily.quality_ready
    return daily


def _slope_per_day(s: pd.Series) -> float:
    """Least-squares slope in elapsed days, including gaps."""
    y = s.to_numpy(float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if isinstance(s.index, pd.DatetimeIndex):
        x = np.asarray((s.index-s.index[0]).total_seconds()/86400, float)[ok]
    else:
        raise ValueError("slope requires a DatetimeIndex")
    return float(np.polyfit(x,y[ok],1)[0]) if np.ptp(x)>0 else np.nan


def add_trend(daily, col="health_index", smooth_days=3, slope_days=14, by="asset_id"):
    """Trailing calendar windows. A missing current observation remains missing."""
    if smooth_days < 1 or slope_days < 1:
        raise ValueError("windows must be positive")
    if daily.duplicated([by,"day"]).any() or not daily.index.is_unique:
        raise ValueError("unique asset-days and row indices are required")
    out = daily.copy()
    for suffix in ("smooth","slope","vol"):
        out[f"{col}_{suffix}"] = np.nan
    for _,g in daily.sort_values([by,"day"]).groupby(by,sort=False):
        s = pd.Series(g[col].to_numpy(float), index=pd.DatetimeIndex(g.day))
        s = s.where(np.isfinite(s))
        smooth = s.rolling(f"{smooth_days}D",min_periods=min(3,smooth_days)).median().where(s.notna())
        slope = smooth.rolling(f"{slope_days}D",min_periods=min(5,slope_days)).apply(_slope_per_day,raw=False).where(s.notna())
        vol = s.rolling(f"{slope_days}D",min_periods=min(5,slope_days)).quantile(.75)-s.rolling(f"{slope_days}D",min_periods=min(5,slope_days)).quantile(.25)
        for suffix,value in (("smooth",smooth),("slope",slope),("vol",vol)):
            out.loc[g.index,f"{col}_{suffix}"] = value.to_numpy()
    return out
