"""Maintenance-window comparison using the exact margin on the aspect card.

An empirical lower RUL bound is not a survival distribution. This module does
not emit failure probabilities or a 'safe until' date.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .rul import ConformalRUL, prediction_time

DEFAULT_HORIZONS = (0., 3., 7., 14., 28.)
WINDOW_LABELS = {
    "within_margin": "Within estimated margin",
    "exceeds_margin": "Exceeds estimated margin",
    "no_positive_margin": "No positive margin established",
    "threshold_exceeded": "Threshold exceeded",
    "unknown": "Cannot assess",
}

def horizon_key(h):
    if not np.isfinite(h) or h < 0:
        raise ValueError("horizon must be finite and non-negative")
    return f"window_{h:g}d"

def compare_window(margin, state, horizon):
    if not np.isfinite(horizon) or horizon < 0:
        return "unknown"
    if state == "threshold_exceeded":
        return "threshold_exceeded"
    if state != "valid" or not np.isfinite(margin):
        return "unknown"
    if margin <= 0:
        return "no_positive_margin"
    return "within_margin" if horizon < margin else "exceeds_margin"

@dataclass
class DeferralLedger:
    horizons: tuple[float, ...] = DEFAULT_HORIZONS
    base: ConformalRUL = field(default_factory=ConformalRUL)

    def fit(self, daily, episodes):
        self.base.fit(daily, episodes)
        return self

    def ledger(self, daily, horizons=None, *, window_at=None):
        # Consume already predicted columns, so separately selected models cannot
        # silently disagree between the card and ledger.
        out = daily.copy() if {"rul_lower", "prediction_state"} <= set(daily) else self.base.predict(daily)
        # Strip deprecated quantities when upgrading an older frame.
        out = out.drop(columns=[c for c in out if c.startswith("risk_") or c == "safe_days_10pct"])
        for h in self.horizons if horizons is None else horizons:
            key = horizon_key(h)
            out[key] = [compare_window(m, s, h) for m, s in zip(out.rul_lower, out.prediction_state)]
        if window_at is not None:
            asof = prediction_time(out)
            end = pd.to_datetime(window_at)
            if np.isscalar(end):
                end = pd.Series(end, index=out.index)
            else:
                end = pd.Series(end, index=out.index)
            hours = (end - asof).dt.total_seconds() / 3600
            if (hours < 0).any():
                raise ValueError("maintenance window precedes prediction availability")
            out["window_at"] = end
            out["hours_to_window"] = hours
            out["scheduled_window"] = [compare_window(m, s, h / 24)
                for m, s, h in zip(out.rul_lower, out.prediction_state, hours)]
        return out

    def risk(self, *args, **kwargs):
        raise NotImplementedError("Individual failure probabilities are not calibrated. Use ledger().")

    def latest_date_under(self, *args, **kwargs):
        raise NotImplementedError("A lower RUL bound does not establish a safe date at a risk budget.")

def render(row, horizons=DEFAULT_HORIZONS):
    lines = [f"  {row['asset_id']}  maintenance-window comparison"]
    for h in horizons:
        key = horizon_key(h)
        if key in row:
            label = "act now" if h == 0 else f"in {h:g} days"
            lines.append(f"    {label:<16} {WINDOW_LABELS[row[key]]}")
    if "scheduled_window" in row:
        lines.append(f"    booked window: {WINDOW_LABELS[row.scheduled_window]}")
    lines.append("  Estimated margin only; operator review required. No failure probability is established.")
    return "\n".join(lines)
