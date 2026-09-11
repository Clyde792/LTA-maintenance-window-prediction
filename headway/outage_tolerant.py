"""EXPERIMENTAL outage-tolerant warning score. Not the deployment detector.

Why this exists
---------------
`robustness.score_frozen` smooths over a trailing three-CALENDAR-DAY window and
requires three observations inside it (`min_periods=smooth_days`). Under the
every-third-day outage in ROBUSTNESS_RESULTS.md that condition can never be met,
so monitoring stopped entirely: three consecutive daily observations never
accumulate. The failure is in the admission rule, not in the underlying score.

This challenger keeps the same per-day scores and the same abstention discipline,
and changes only which observations may support an assessment:

  * observations may be IRREGULARLY SPACED — what matters is elapsed time, not
    row adjacency;
  * `max_age_days` bounds how old a supporting observation may be;
  * `max_gap_days` bounds the elapsed gap between consecutive supporting
    observations, so a score is never assembled across a hole bigger than the
    operator is willing to reason about;
  * `min_observations` sets how much evidence is required before any score.

What it deliberately does NOT do
--------------------------------
* No interpolation, imputation or resampling. A missing reading stays missing;
  it is never turned into invented evidence.
* No carrying an old score forward. Every assessment is anchored on a valid
  observation FOR THAT DAY; without one the output is an explicit unavailable
  status and a NaN score, never yesterday's number relabelled as today's.
* No unsupported operating conditions. Rows failing `data_quality_ok` or
  `context_supported` are excluded from the window as well as from the current
  assessment, so an out-of-support reading cannot enter through history.
* No future data. Only observations at or before the assessment day are visible.

Availability is reported separately from health: a `*_status` column says why an
assessment is unavailable, so a silent detector during an outage is never
readable as a healthy asset.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Assessment statuses. Only SUPPORTED carries a score.
SUPPORTED = "supported"
UNSUPPORTED = "unsupported_or_incomplete"   # row present, telemetry not usable
INCOMPLETE = "incomplete_channels"          # supported row, required channel missing
NO_SCORE = "model_score_unavailable"        # usable inputs, unusable model output
TOO_FEW = "insufficient_observations"       # not enough valid evidence in window
GAP = "gap_too_large"                       # evidence separated by an unacceptable hole
STATUSES = (SUPPORTED, UNSUPPORTED, INCOMPLETE, NO_SCORE, TOO_FEW, GAP)


@dataclass(frozen=True)
class WindowPolicy:
    """Admission rule for supporting observations, all in ELAPSED days."""
    min_observations: int = 3
    max_age_days: float = 7.0
    max_gap_days: float = 3.0

    def __post_init__(self):
        if type(self.min_observations) is not int or self.min_observations < 2:
            raise ValueError("min_observations must be an integer of at least 2")
        for name in ("max_age_days", "max_gap_days"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a positive finite number of days")
        if self.max_gap_days > self.max_age_days:
            raise ValueError("max_gap_days cannot exceed max_age_days")

    def as_dict(self):
        return asdict(self)


def _columns(name):
    return (name, f"{name}_status", f"{name}_obs", f"{name}_span_days", f"{name}_max_gap_days")


def _run_back(i, days, valid, policy):
    """Most recent contiguous-enough run of valid observations ending at `i`.

    Walks backwards in time, admitting a valid observation while it is within
    `max_age_days` of the assessment day AND within `max_gap_days` of the last
    admitted (more recent) one. An invalid day between two valid ones does not
    itself end the run: the gap that matters is between the observations that
    are actually used.
    """
    chosen = [i]
    last = days[i]
    reason = TOO_FEW           # ran out of window or history
    for j in range(i - 1, -1, -1):
        if not valid[j]:
            continue
        if days[i] - days[j] > policy.max_age_days:
            break
        if last - days[j] > policy.max_gap_days:
            reason = GAP       # a hole bigger than the policy allows stopped us
            break
        chosen.append(j)
        last = days[j]
    return np.array(chosen[::-1]), reason


def score_outage_tolerant(detectors, daily, policy=None, *, day_col="day"):
    """Score fitted models over irregularly spaced valid observations.

    Mirrors `robustness.score_frozen`: models are already fitted, nothing is
    refitted here, and no missing channel is filled. Returns a copy of `daily`
    with, per detector, the score plus the columns naming its evidence.
    """
    policy = policy or WindowPolicy()
    out = daily.sort_values(["asset_id", day_col]).copy()
    if not out.index.is_unique or out.duplicated(["asset_id", day_col]).any():
        raise ValueError("unique daily rows required")

    supported_all = (out.data_quality_ok.fillna(False) & out.context_supported.fillna(False)).to_numpy(bool)
    for name, model in detectors.items():
        needs = model.needs if hasattr(model, "needs") else [model.column]
        finite = np.isfinite(out[needs].to_numpy(float)).all(axis=1)
        complete = finite & supported_all
        raw = pd.Series(np.nan, index=out.index)
        if complete.any():
            raw.loc[complete] = model.score(out.loc[complete])

        score_c, status_c, obs_c, span_c, gap_c = _columns(name)
        out[score_c] = np.nan
        out[status_c] = UNSUPPORTED
        out[obs_c] = 0
        out[span_c] = np.nan
        out[gap_c] = np.nan

        for _, g in out.groupby("asset_id", sort=False):
            pos = out.index.get_indexer(g.index)
            days = (pd.DatetimeIndex(g[day_col]) - pd.Timestamp(0)).total_seconds().to_numpy() / 86400.0
            vals = raw.loc[g.index].to_numpy(float)
            valid = np.isfinite(vals)
            sup, fin = supported_all[pos], finite[pos]

            status = np.full(len(g), UNSUPPORTED, dtype=object)
            score = np.full(len(g), np.nan)
            nobs = np.zeros(len(g), dtype=int)
            span = np.full(len(g), np.nan)
            maxgap = np.full(len(g), np.nan)

            for i in range(len(g)):
                # An assessment must be anchored on TODAY's valid observation.
                if not sup[i]:
                    status[i] = UNSUPPORTED
                    continue
                if not fin[i]:
                    status[i] = INCOMPLETE
                    continue
                # Finite INPUTS do not guarantee a finite model OUTPUT. Without
                # this guard a model returning NaN/inf for today would be folded
                # into the median and published as `supported` with a NaN score.
                if not valid[i]:
                    status[i] = NO_SCORE
                    continue
                idx, reason = _run_back(i, days, valid, policy)
                used = days[idx]
                nobs[i] = len(idx)
                span[i] = float(days[i] - used[0])
                maxgap[i] = float(np.diff(used).max()) if len(used) > 1 else 0.0
                if len(idx) < policy.min_observations:
                    status[i] = reason
                    continue
                status[i] = SUPPORTED
                score[i] = float(np.median(vals[idx]))

            out.loc[g.index, score_c] = score
            out.loc[g.index, status_c] = status
            out.loc[g.index, obs_c] = nobs
            out.loc[g.index, span_c] = span
            out.loc[g.index, gap_c] = maxgap
    return out


def availability(scored, score_name, expected_rows):
    """Supported assessments as a fraction of EXPECTED asset-days.

    `expected_rows` is the count before any outage removed rows, so days that
    vanished stay in the denominator and an outage cannot flatter availability.
    """
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    return float(np.isfinite(pd.to_numeric(scored[score_name], errors="coerce")).sum()) / expected_rows


def recovery_days(scored, score_name, *, resumed_at, assets, available_col="available_at"):
    """Elapsed days from telemetry resuming until an engineer can RECEIVE a score.

    Measured against `available_at`, not the observation date: a day's aggregate
    only lands the following midnight, so observation dates understate how long
    monitoring is actually dark. Fractional days are preserved.

    EVERY requested asset appears in the result. An asset that never produced a
    score again maps to None rather than vanishing, so a method cannot look fast
    by recovering only its easiest assets.
    """
    out = {a: None for a in assets}
    if available_col not in scored.columns:
        raise ValueError(f"{available_col} is required to measure operational recovery")
    resumed = pd.Timestamp(resumed_at)
    sub = scored[scored.asset_id.isin(assets)]
    for asset, g in sub.groupby("asset_id", sort=True):
        ok = g[np.isfinite(pd.to_numeric(g[score_name], errors="coerce"))]
        ok = ok[pd.to_datetime(ok[available_col]) >= resumed].sort_values(available_col)
        if len(ok):
            delta = pd.Timestamp(ok[available_col].iloc[0]) - resumed
            out[asset] = float(delta.total_seconds() / 86400.0)
    return out


def recovery_summary(per_asset):
    """Censored summary: unrecovered assets count as infinitely slow.

    Taking the median only over assets that recovered would reward a method for
    abandoning the hard ones. Here every requested asset enters the median, with
    non-recovery as +inf, so a method that recovers fewer assets cannot win by
    recovering those few faster. If more than half never recover the median is
    infinite and is reported as unresolved rather than as a number.
    """
    total = len(per_asset)
    recovered = [v for v in per_asset.values() if v is not None]
    if not total:
        return {"median_days": None, "median_is_censored": False,
                "assets_recovered": 0, "assets_total": 0, "per_asset": {}}
    censored = [v if v is not None else np.inf for v in per_asset.values()]
    med = float(np.median(censored))
    return {"median_days": med if np.isfinite(med) else None,
            "median_is_censored": not np.isfinite(med),
            "assets_recovered": len(recovered), "assets_total": total,
            "per_asset": dict(per_asset)}
