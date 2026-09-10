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

    agg: dict[str, tuple[str, object]] = {"n_cycles": ("ts", "size")}
    for c in levels:
        agg[c] = (c, "median")
        agg[f"{c}_iqr"] = (c, _iqr)
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

    # train_id is constant within an asset; carry it for fleet-level grouping.
    if "train_id" in df.columns:
        daily["train_id"] = daily["asset_id"].map(
            df.groupby("asset_id")["train_id"].first()
        )
    return daily


def _slope_per_day(s: pd.Series) -> float:
    """Least-squares slope of a window, in units per day.

    The x-axis is position in the window rather than calendar date, so a gap in
    the data compresses rather than distorts the fit. Windows are short enough
    (2 weeks) that the difference is immaterial, and this cannot divide by zero.
    """
    y = s.to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)[ok]
    return float(np.polyfit(x, y[ok], 1)[0])


def add_trend(
    daily: pd.DataFrame,
    col: str = "health_index",
    # 3, not 5: a trailing median lags a rising signal by about half its width,
    # so smoothing is bought with warning time. Swept on four independently
    # seeded fleets (scripts/holdout_check.py): 3 gives +0.2 d of worst-case
    # lead AND +1.2 pp precision over 5 at the 250-alert budget; 1 buys a
    # further day of lead but costs ~4 pp precision. 3 is the balanced pick,
    # chosen on holdout evidence rather than on the fleet we tuned against.
    smooth_days: int = 3,
    slope_days: int = 14,
    by: str = "asset_id",
) -> pd.DataFrame:
    """Add smoothed level, slope and volatility of `col`, per asset.

    All windows are trailing, so every value uses only data available on that
    day. Nothing here can leak the future into a score - which matters, because
    the lead-time metric is meaningless if the detector has seen ahead.

    The returned frame preserves the CALLER'S row order and index. An earlier
    version sorted with ignore_index=True and returned a re-ordered frame; that
    works only while the input happens to be pre-sorted, and silently misaligns
    every value the moment it is not. Results are computed on a sorted view and
    reindexed back.
    """
    work = daily.sort_values([by, "day"])   # keeps original index labels
    g = work.groupby(by, sort=False)[col]

    # min_periods must never exceed its window - pandas raises rather than
    # clamping, so a caller sweeping smooth_days down to 1 (a legitimate
    # "no smoothing" baseline) would crash instead of running.
    mp_s = min(3, smooth_days)
    mp_l = min(5, slope_days)

    smooth = g.transform(lambda s: s.rolling(smooth_days, min_periods=mp_s).median())
    slope = smooth.groupby(work[by], sort=False).transform(
        lambda s: s.rolling(slope_days, min_periods=mp_l).apply(_slope_per_day, raw=False)
    )
    vol = g.transform(lambda s: s.rolling(slope_days, min_periods=mp_l).apply(_iqr, raw=False))

    out = daily.copy()
    out[f"{col}_smooth"] = smooth.reindex(daily.index)
    out[f"{col}_slope"] = slope.reindex(daily.index)
    out[f"{col}_vol"] = vol.reindex(daily.index)
    return out
