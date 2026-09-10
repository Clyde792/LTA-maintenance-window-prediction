"""Illustrative maintenance policy, separate from estimation.

UNKNOWN is not a severity level. Missing evidence never clears an earlier
escalation. Day thresholds are configurable demo policy, not operator rules.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import pandas as pd

class Aspect(IntEnum):
    UNKNOWN = -1
    GREEN = 0
    DOUBLE_AMBER = 1
    AMBER = 2
    RED = 3

RECOMMENDATION = {-1: "REVIEW", 0: "MONITOR", 1: "PLAN", 2: "TONIGHT", 3: "WITHDRAW"}
DISPLAY = {-1: "UNKNOWN", 0: "GREEN", 1: "DOUBLE AMBER", 2: "AMBER", 3: "RED"}
MEANING = {-1: "Review data or inspect; no reliable countdown is available.",
    0: "No current escalation. Continue monitoring.",
    1: "Review an upcoming maintenance window.",
    2: "Prioritise the next suitable maintenance window.",
    3: "Urgent operator review for withdrawal or isolation."}

@dataclass
class AspectPolicy:
    withdraw_days: float = 1.
    tonight_days: float = 3.
    plan_days: float = 21.
    dwell_days: int = 3
    margin_col: str = "rul_lower"

    def raw(self, margin):
        m = np.asarray(margin, float)
        out = np.full(m.shape, int(Aspect.UNKNOWN), dtype=int)
        finite = np.isfinite(m)
        out[finite] = Aspect.GREEN
        out[finite & (m < self.plan_days)] = Aspect.DOUBLE_AMBER
        out[finite & (m < self.tonight_days)] = Aspect.AMBER
        out[finite & (m < self.withdraw_days)] = Aspect.RED
        return out

    def _damp(self, raw, dates=None):
        out = np.empty_like(raw)
        state, below, previous = int(Aspect.UNKNOWN), 0, None
        for i, r in enumerate(raw):
            now = pd.Timestamp(dates[i]) if dates is not None else None
            if previous is not None and (now - previous) > pd.Timedelta(days=1):
                below = 0  # separated readings are not consecutive clear days
            previous = now
            if r == Aspect.UNKNOWN:
                below = 0
                out[i] = state if state > Aspect.GREEN else Aspect.UNKNOWN
                continue
            if state == Aspect.UNKNOWN or r >= state:
                state, below = int(r), 0
            else:
                below += 1
                if below >= max(1, self.dwell_days):
                    state, below = int(r), 0
            out[i] = state
        return out

    def apply(self, df, by="asset_id", order="day"):
        work = df.sort_values([by, order]).copy()
        raw = self.raw(work[self.margin_col].to_numpy())
        if "prediction_state" in work:
            states = work.prediction_state.to_numpy()
            raw[states == "no_worsening_trend"] = Aspect.GREEN
            raw[~np.isin(states, ["valid", "threshold_exceeded", "no_worsening_trend"])] = Aspect.UNKNOWN
        work["raw_aspect"] = raw
        work["aspect"] = int(Aspect.UNKNOWN)
        for _, g in work.groupby(by, sort=False):
            work.loc[g.index, "aspect"] = self._damp(g.raw_aspect.to_numpy(), g[order].to_numpy())
        out = work.reindex(df.index)
        out["aspect_name"] = out.aspect.map(DISPLAY)
        out["recommendation"] = out.aspect.map(RECOMMENDATION)
        out["decision_stability"] = [confidence(self, p, l) for p, l in zip(out.rul_point, out.rul_lower)]
        return out

def confidence(policy, point, lower):
    """Backward-compatible function name: returns decision stability, not probability."""
    if not np.isfinite(point) or not np.isfinite(lower):
        return "n/a"
    gap = abs(int(policy.raw([point])[0]) - int(policy.raw([lower])[0]))
    return "HIGH" if gap == 0 else "MEDIUM" if gap == 1 else "LOW"

def why(row, top=3):
    out = []
    state = row.get("prediction_state", "valid")
    if not isinstance(state,str):
        state = "insufficient_history"
    if state != "valid":
        out.append("Prediction state: " + state.replace("_", " "))
    hi, slope = row.get("health_index_smooth", np.nan), row.get("health_index_slope", np.nan)
    if np.isfinite(hi):
        out.append(f"{hi:.1f} standardised health index (not a failure probability)")
    if np.isfinite(slope):
        out.append(f"trend {slope:+.2f} index units per elapsed day")
    return out[:top]

@dataclass
class AspectCard:
    asset_id: str
    train_id: str
    day: pd.Timestamp
    aspect: Aspect
    recommendation: str
    meaning: str
    margin_days: float
    projected_days: float
    confidence: str  # legacy storage field; rendered as decision stability
    evidence: list[str]
    margin_aspect: int = -1
    prediction_state: str = "unknown"

    @classmethod
    def from_row(cls, row, policy=None):
        policy = policy or AspectPolicy()
        a = Aspect(int(row["aspect"]))
        return cls(row.asset_id, row.get("train_id", ""), row.day, a,
            RECOMMENDATION[a], MEANING[a], float(row.rul_lower), float(row.rul_point),
            confidence(policy, row.rul_point, row.rul_lower), why(row),
            int(row.get("raw_aspect", policy.raw([row.rul_lower])[0])),
            row.get("prediction_state", "unknown"))

    @property
    def held(self):
        return self.aspect > Aspect.GREEN and self.aspect > self.margin_aspect

    def render(self):
        margin = f"{self.margin_days:.1f} days" if np.isfinite(self.margin_days) else "unavailable"
        text = [f"  {self.asset_id}  {DISPLAY[self.aspect]} / {self.recommendation}",
            f"  {self.meaning}", f"  estimated margin {margin}; decision stability {self.confidence}"]
        if self.held:
            text.append("  HELD from an earlier escalation; missing data cannot clear the order.")
            text.append(f"  Current evidence alone: {DISPLAY[self.margin_aspect]}")
        text.extend("    - " + e for e in self.evidence)
        return "\n".join(text)
