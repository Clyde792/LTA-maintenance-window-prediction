"""Fit-for-duty: is this asset fit for peak service, off-peak only, or neither?

The fleet normaliser already removes the load effect a HEALTHY asset shows.
What it cannot remove is an asset that has become MORE load-sensitive than the
fleet - a worn mechanism that looks tolerable across a day's average and fails
under the 08:15 crowd. This module estimates that excess sensitivity per asset
from within-day contrast (crowded cycles vs quiet cycles), projects the health
index forward to peak conditions with the SAME normaliser, index and RUL model
the card uses, and applies the SAME aspect policy to the result.

Outputs are duty labels and evidence, not probabilities. The peak-conditioned
margin reuses the fleet calibration on a shifted projection; it is not
separately calibrated and is labelled as such.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .aspect import Aspect, AspectPolicy
from .rul import HI, ConformalRUL

# Sufficient statistics of (load, cycle index) per asset-day. Trailing sums of
# these give the pooled within-asset regression exactly, without keeping cycles.
STATS = ("duty_n", "duty_sum_l", "duty_sum_ll", "duty_sum_y", "duty_sum_ly", "duty_sum_yy")

DUTY_LABELS = {
    "full_service": "No peak-specific restriction",
    "off_peak_only": "Off-peak service only",
    "withdraw": "Withdraw from service",
    "not_assessed": "Duty not assessed",
}
# Duty is orthogonal to the aspect. The aspect says how URGENT the maintenance
# decision is (a lower bound on days). Duty says what operating restriction the
# CURRENT LEVEL supports while that decision is made.
DUTY_MEANING = {
    "full_service": "Peak load does not change the picture; act on the aspect.",
    "off_peak_only": "At peak load this door already presents at or beyond the level at which doors failed; on the day average it does not. Keep it out of the peaks until maintained.",
    "withdraw": "On the day average the index is at or beyond the failure level. A duty restriction is not enough.",
    "not_assessed": "Insufficient contrast between crowded and quiet cycles, or the base estimate is unavailable.",
}


def cycle_stats(cycles, index="cycle_index", load="load_proxy"):
    """Per asset-day sufficient statistics of (load, index) over complete cycles."""
    c = cycles[["asset_id", "ts", load, index]].copy()
    c["day"] = c.ts.dt.floor("D")
    l = pd.to_numeric(c[load], errors="coerce").to_numpy(float)
    y = pd.to_numeric(c[index], errors="coerce").to_numpy(float)
    ok = np.isfinite(l) & np.isfinite(y)
    c = c[ok].assign(l=l[ok], y=y[ok])
    c["ll"], c["ly"], c["yy"] = c.l * c.l, c.l * c.y, c.y * c.y
    g = c.groupby(["asset_id", "day"], as_index=False).agg(
        duty_n=("y", "size"), duty_sum_l=("l", "sum"), duty_sum_ll=("ll", "sum"),
        duty_sum_y=("y", "sum"), duty_sum_ly=("ly", "sum"), duty_sum_yy=("yy", "sum"))
    return g


def peak_bands(reference, hour="hour_of_day", load="load_proxy", peak_quantile=0.9):
    """Contiguous hour ranges whose reference median load sits in the upper half
    of the diurnal range. Derived from the data, so a real timetable defines the
    peaks, not a hard-coded 07:00-09:00."""
    h = pd.to_numeric(reference[hour], errors="coerce")
    l = pd.to_numeric(reference[load], errors="coerce")
    ok = h.notna() & l.notna()
    if ok.sum() < 24:
        return [], np.nan, np.nan
    bins = np.floor(h[ok]).astype(int).clip(0, 23)
    prof = l[ok].groupby(bins).median().reindex(range(24))
    valid = prof.dropna()
    if valid.empty or (valid.max() - valid.min()) < 1e-6:
        return [], float(l[ok].mean()), float(l[ok].mean())
    cut = valid.min() + 0.5 * (valid.max() - valid.min())
    is_peak = (prof >= cut).fillna(False).to_numpy()
    bands, start = [], None
    for hr in range(25):
        on = hr < 24 and is_peak[hr]
        if on and start is None:
            start = hr
        if not on and start is not None:
            bands.append((start, hr))
            start = None
    peak_hours = bins.isin([hr for a, b in bands for hr in range(a, b)])
    # "Peak conditions" means the crowded cycles a door meets in the band, not
    # the band's average: the restriction is about the worst cycles, and the
    # day-average index is already centred on the typical ones.
    load_peak = float(l[ok][peak_hours].quantile(peak_quantile)) if peak_hours.any() else float(l[ok].mean())
    load_off = float(l[ok][~peak_hours].median()) if (~peak_hours).any() else float(l[ok].mean())
    return bands, load_peak, load_off


def format_bands(bands):
    return ", ".join(f"{a:02d}:00–{b:02d}:00" for a, b in bands) if bands else "no peak identified"


def load_sensitivity(daily, window_days=3, min_cycles=30, min_load_sd=0.08, min_t=2.5, exit_t=1.5, by="asset_id"):
    """Trailing pooled regression of cycle index on load, per asset.

    The window matches the index smoothing window on purpose: the shift is
    estimated over the same days as the level it is applied to, so a repair
    does not leave a stale sensitivity attached to a healthy level.

    Returns the slope in index units per unit load, its t-statistic, the mean
    load of the window, and `load_sensitive`: slope positive and significant.
    Significance is a Schmitt trigger - enter at `min_t`, stay while the
    t-statistic holds above `exit_t` on consecutive days - so a slope hovering
    at the entry line cannot switch a restriction on and off daily.
    """
    if not set(STATS) <= set(daily):
        raise ValueError("duty statistics missing; run HealthPipeline.transform first")
    out = daily.copy()
    for c in ("load_sensitivity", "load_sensitivity_t", "window_mean_load"):
        out[c] = np.nan
    out["load_sensitive"] = False
    for _, g in daily.sort_values([by, "day"]).groupby(by, sort=False):
        s = g.set_index(pd.DatetimeIndex(g.day))[list(STATS)].astype(float)
        w = s.rolling(f"{window_days}D", min_periods=1).sum()
        n, sl, sll, sy, sly, syy = (w[c].to_numpy() for c in STATS)
        with np.errstate(invalid="ignore", divide="ignore"):
            sxx = sll - sl * sl / n
            sxy = sly - sl * sy / n
            syy_c = syy - sy * sy / n
            b = sxy / sxx
            resid_var = np.maximum(syy_c - b * sxy, 0.) / np.maximum(n - 2, 1)
            t = b / np.sqrt(resid_var / sxx)
            load_sd = np.sqrt(sxx / n)
            mean_load = sl / n
        usable = np.isfinite(b) & (n >= min_cycles) & (load_sd >= min_load_sd)
        enter = usable & (b > 0) & (t >= min_t)
        stay = usable & (b > 0) & (t >= exit_t)
        days = g.day.to_numpy()
        sensitive = np.zeros(len(g), bool)
        for i in range(len(g)):
            consecutive = i > 0 and (days[i] - days[i - 1]) <= np.timedelta64(1, "D")
            sensitive[i] = enter[i] or (consecutive and sensitive[i - 1] and stay[i])
        out.loc[g.index, "load_sensitivity"] = np.where(usable, b, np.nan)
        out.loc[g.index, "load_sensitivity_t"] = np.where(usable, t, np.nan)
        out.loc[g.index, "window_mean_load"] = np.where(np.isfinite(mean_load), mean_load, np.nan)
        out.loc[g.index, "load_sensitive"] = sensitive
    return out


@dataclass
class DutyAssessor:
    """Re-run the card's RUL model and policy at peak conditions."""
    rul: ConformalRUL
    policy: AspectPolicy = field(default_factory=AspectPolicy)
    load_peak: float = np.nan
    load_offpeak: float = np.nan
    bands: list = field(default_factory=list)
    window_days: int = 3

    def assess(self, daily):
        if not (np.isfinite(self.load_peak) and np.isfinite(self.load_offpeak)):
            raise ValueError("peak and off-peak reference loads are required")
        if "aspect" not in daily or "rul_lower" not in daily:
            raise ValueError("assess() consumes a frame that already carries the card's aspect and margin")
        d = load_sensitivity(daily, window_days=self.window_days)
        b = np.where(d.load_sensitive, d.load_sensitivity, 0.)
        anchor = d.window_mean_load.to_numpy(float)
        d["peak_shift"] = b * (self.load_peak - anchor)
        d["offpeak_shift"] = b * (self.load_offpeak - anchor)
        # Same model, same policy, shifted index. Carry the base states through:
        # a day the card cannot assess is not assessable at peak either.
        for tag in ("peak", "offpeak"):
            shifted = daily.assign(**{HI: d[HI] + d[f"{tag}_shift"].fillna(0.)})
            p = self.rul.predict(shifted)
            d[f"{tag}_index"] = shifted[HI]
            d[f"{tag}_rul_point"], d[f"{tag}_rul_lower"], d[f"{tag}_state"] = p.rul_point, p.rul_lower, p.prediction_state
            temp = d.assign(rul_point=p.rul_point, rul_lower=p.rul_lower, prediction_state=p.prediction_state)
            applied = self.policy.apply(temp)
            d[f"{tag}_aspect"], d[f"{tag}_raw_aspect"] = applied.aspect, applied.raw_aspect
        d["threshold"] = self.rul.threshold_
        d["duty"], d["duty_reason"] = zip(*[
            _duty(r, self.rul.threshold_) for r in d.itertuples()])
        d["duty_bands"] = format_bands(self.bands)
        return d


def _duty(r, threshold):
    base_state, peak_state = r.prediction_state, r.peak_state
    if base_state == "threshold_exceeded":
        return "withdraw", "day-average index at or beyond the failure level"
    if r.aspect == Aspect.UNKNOWN or not np.isfinite(r.load_sensitivity):
        return "not_assessed", ("base estimate unavailable" if r.aspect == Aspect.UNKNOWN
                                else "insufficient crowded/quiet contrast in the trailing window")
    if not r.load_sensitive:
        return "full_service", "no excess load sensitivity beyond the fleet model"
    # Level rule, and only the level rule. A margin rule (peak-conditioned
    # lower bound crossing a policy line before the day-average one) was tried
    # and removed: near failure the two bounds differ by under a day, so it
    # fired on noise and flickered. The peak signal lives in the level.
    if peak_state == "threshold_exceeded":
        return "off_peak_only", f"index at peak load {r.peak_index:.1f} is at or beyond the failure level {threshold:.1f}; day average {getattr(r, HI):.1f} is not"
    return "full_service", "load-sensitive, but peak load has not reached the failure level"


def duty_summary(assessed, episodes):
    """Synthetic diagnostic: does the restriction precede withdrawal, and how
    often does it fire on a healthy asset? Not an operational rate."""
    from .rul import prediction_time
    when = prediction_time(assessed)
    ep_assets = set(episodes.asset_id)
    sens_lead, restrict_lead, restricted, sensitive = [], [], 0, 0
    for e in episodes.itertuples():
        w = assessed[(assessed.asset_id == e.asset_id) & (when >= e.onset_ts) & (when <= e.fault_ts)]
        t = prediction_time(w)
        fault = pd.Timestamp(e.fault_ts)
        s = w[w.load_sensitive]
        if len(s):
            sensitive += 1
            sens_lead.append((fault - t[s.index].min()).total_seconds() / 86400)
        r = w[w.duty == "off_peak_only"]
        if len(r):
            restricted += 1
            restrict_lead.append((fault - t[r.index].min()).total_seconds() / 86400)
    healthy = assessed[~assessed.asset_id.isin(ep_assets) & (assessed.duty != "not_assessed")]
    return {
        "episodes": int(len(episodes)),
        "episodes_load_sensitive_before_fault": int(sensitive),
        "median_sensitivity_lead_days": float(np.median(sens_lead)) if sens_lead else None,
        "episodes_restricted_before_fault": int(restricted),
        "median_restriction_lead_days": float(np.median(restrict_lead)) if restrict_lead else None,
        "healthy_asset_days_assessed": int(len(healthy)),
        "healthy_asset_days_flagged_sensitive": int(healthy.load_sensitive.sum()),
        "healthy_asset_days_restricted": int((healthy.duty == "off_peak_only").sum()),
        "healthy_false_restriction_rate": float((healthy.duty == "off_peak_only").mean()) if len(healthy) else None,
    }


def render(row):
    lines = [f"  {row['asset_id']}  fit-for-duty: {DUTY_LABELS.get(row['duty'], row['duty'])}"]
    if row["duty"] == "off_peak_only":
        lines.append(f"    avoid peak service {row.get('duty_bands', '')}")
    if np.isfinite(row.get("load_sensitivity", np.nan)):
        lines.append(f"    load sensitivity {row['load_sensitivity']:+.2f} index per unit load (t={row['load_sensitivity_t']:.1f})")
    if np.isfinite(row.get("peak_index", np.nan)):
        lines.append(f"    index at peak {row['peak_index']:.1f} / day-average {row.get(HI, np.nan):.1f} / "
                     f"off-peak {row.get('offpeak_index', np.nan):.1f}; threshold {row.get('threshold', np.nan):.1f}")
    lines.append(f"    {row.get('duty_reason', '')}")
    lines.append("  Peak projection reuses the fleet calibration on a shifted index; not separately calibrated.")
    return "\n".join(lines)
