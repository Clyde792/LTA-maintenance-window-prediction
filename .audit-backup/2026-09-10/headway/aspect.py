"""
Headway - the aspect decision policy.

This is where estimation stops and a decision starts, and the two are kept in
separate files on purpose: you can change what Headway RECOMMENDS without
retraining anything. The policy is a handful of thresholds an engineer can read,
argue with, and overrule. A model that quietly encodes the operating policy in
its weights cannot be argued with, which is a bad property for something telling
people when to go on track.

FOUR-ASPECT SIGNALLING
----------------------
Railways already have a language for "what do I do next", and it has four
aspects. We borrow it exactly:

    GREEN         MONITOR    healthy, keep watching
    DOUBLE AMBER  PLAN       book into an upcoming engineering window
    AMBER         TONIGHT    this window - it cannot wait for the next one
    RED           WITHDRAW   take it out of service now

THE POLICY CONSUMES THE LOWER BOUND, NEVER THE POINT ESTIMATE
--------------------------------------------------------------
`rul_lower`, not `rul_point`. This is the entire reason the RUL module produces
a calibrated bound. Two doors with the same projected margin but different
uncertainty get different aspects, because the one whose projection is less
trustworthy has a lower bound and therefore escalates sooner.

WHY THESE THRESHOLDS
--------------------
They are operational, not statistical:

    1 day    a train that cannot be trusted to survive until the next nightly
             engineering window has to come out of service now.
    3 days   roughly the horizon within which the work must be placed in
             tonight's window rather than negotiated into a later one - the
             window is ~2 usable hours and is already contended.
    21 days  the maintenance planning horizon. Beyond it, "book it" is the
             right answer rather than "do it".

HYSTERESIS
----------
Aspects escalate immediately and de-escalate slowly. Without that, a single
quiet day drops a degrading door from AMBER back to GREEN and the alert list
flaps night to night - which is how operators learn to ignore a system. Going up
is urgent; coming down should require evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class Aspect(IntEnum):
    """Ordered by severity, so escalation is just `max`."""

    GREEN = 0
    DOUBLE_AMBER = 1
    AMBER = 2
    RED = 3


RECOMMENDATION: dict[int, str] = {
    Aspect.GREEN: "MONITOR",
    Aspect.DOUBLE_AMBER: "PLAN",
    Aspect.AMBER: "TONIGHT",
    Aspect.RED: "WITHDRAW",
}

MEANING: dict[int, str] = {
    Aspect.GREEN: "Healthy. Keep watching.",
    Aspect.DOUBLE_AMBER: "Book into an upcoming engineering window.",
    Aspect.AMBER: "This window. It cannot wait for the next one.",
    Aspect.RED: "Withdraw from service now.",
}

DISPLAY: dict[int, str] = {
    Aspect.GREEN: "GREEN",
    Aspect.DOUBLE_AMBER: "DOUBLE AMBER",
    Aspect.AMBER: "AMBER",
    Aspect.RED: "RED",
}


@dataclass
class AspectPolicy:
    """Maps a calibrated margin onto a signal aspect.

    Args:
        withdraw_days / tonight_days / plan_days: the operational thresholds,
            applied to `rul_lower`.
        dwell_days: consecutive days at a lower severity required before the
            aspect is allowed to fall. Escalation is always immediate.
    """

    withdraw_days: float = 1.0
    tonight_days: float = 3.0
    plan_days: float = 21.0
    dwell_days: int = 3
    margin_col: str = "rul_lower"

    # ------------------------------------------------------------------ core
    def raw(self, margin: np.ndarray) -> np.ndarray:
        """Aspect from margin alone, before hysteresis.

        A non-finite margin means the asset is not on a failure path at all -
        no meaningful slope - which is GREEN, not "unknown". Handing an engineer
        a countdown for a healthy door is a false alarm.
        """
        m = np.asarray(margin, dtype=float)
        out = np.full(m.shape, int(Aspect.GREEN), dtype=int)
        finite = np.isfinite(m)
        out = np.where(finite & (m < self.plan_days), int(Aspect.DOUBLE_AMBER), out)
        out = np.where(finite & (m < self.tonight_days), int(Aspect.AMBER), out)
        out = np.where(finite & (m < self.withdraw_days), int(Aspect.RED), out)
        return out

    def _damp(self, raw: np.ndarray) -> np.ndarray:
        """Escalate immediately, de-escalate only after `dwell_days` below."""
        if self.dwell_days <= 1:
            return raw
        out = np.empty_like(raw)
        state = int(Aspect.GREEN)
        below = 0
        for i, r in enumerate(raw):
            if r > state:
                state, below = int(r), 0
            elif r < state:
                below += 1
                if below >= self.dwell_days:
                    state, below = int(r), 0
            else:
                below = 0
            out[i] = state
        return out

    def apply(self, df: pd.DataFrame, by: str = "asset_id",
              order: str = "day") -> pd.DataFrame:
        """Add `aspect`, `aspect_name` and `recommendation`.

        Hysteresis is applied per asset in time order, then reindexed back to
        the caller's row order - so this is safe on an unsorted frame.
        """
        work = df.sort_values([by, order])
        raw = self.raw(work[self.margin_col].to_numpy())
        damped = (pd.Series(raw, index=work.index)
                  .groupby(work[by], sort=False)
                  .transform(lambda s: self._damp(s.to_numpy())))

        aspect = damped.reindex(df.index)
        return df.assign(
            aspect=aspect.astype(int),
            aspect_name=aspect.map(DISPLAY),
            recommendation=aspect.map(RECOMMENDATION),
        )


# --------------------------------------------------------------------------
# Aspect Card
# --------------------------------------------------------------------------

def confidence(policy: AspectPolicy, point: float, lower: float) -> str:
    """Is this recommendation robust to how we treat uncertainty?

    HIGH   the raw projection and the calibrated lower bound give the SAME
           aspect. The call does not depend on the uncertainty model at all.
    MEDIUM they differ by one aspect - the bound is doing the work, and a more
           optimistic reading would defer this by one step.
    LOW    they differ by two or more. Treat the recommendation as a prompt to
           go and look, not as a settled answer.

    Two earlier definitions were discarded, both for producing nonsense:

    - Relative interval width. Near failure the margin is ~1 day, so a +/-1 day
      interval reads as 100% error and a RED card came out labelled LOW
      confidence, despite the decision not being in any doubt.
    - Comparing the aspect at each END of the interval. That relies on the upper
      bound, and the upper bound is not trustworthy: the ratio conformity score
      is `true / projection`, which explodes as the projection approaches zero
      late in a degradation. At the time this was measured the fitted upper
      multiplier reached x4.7 - a door supposedly having nearly five times its
      projected life. The LOWER bound is
      the calibrated, well-behaved end, and it is the only one the product acts
      on - so it is the only one confidence should be built from.
    """
    if not np.isfinite(point) or not np.isfinite(lower):
        return "n/a"
    a_point = int(policy.raw(np.array([point]))[0])
    a_bound = int(policy.raw(np.array([lower]))[0])
    gap = abs(a_bound - a_point)
    return "HIGH" if gap == 0 else "MEDIUM" if gap == 1 else "LOW"


def why(row: pd.Series, top: int = 3) -> list[str]:
    """Plain-English evidence for the card.

    Structured strings rather than prose: an LLM narration layer can turn these
    into a shift-handover note later, but the reasons themselves are computed
    here so the card stands on its own if that layer is absent or wrong.
    """
    out: list[str] = []
    hi = row.get("health_index_smooth", np.nan)
    slope = row.get("health_index_slope", np.nan)

    if np.isfinite(hi):
        out.append(f"{hi:.1f} sigma above this door's own baseline")
    if np.isfinite(slope) and slope > 0:
        out.append(f"rising {slope:.2f} sigma/day over the last 14 days")

    temp = row.get("ambient_temp_c_mean", np.nan)
    load = row.get("load_proxy_mean", np.nan)
    if np.isfinite(temp) and np.isfinite(load):
        out.append(f"already normalised for {temp:.0f} C and {load:.0%} loading")

    obs = row.get("obstruction_flag_rate", np.nan)
    if np.isfinite(obs) and obs > 0.15:
        out.append(f"obstruction rate {obs:.0%} - check crowding before wear")
    return out[:top]


@dataclass
class AspectCard:
    """One decision, for one asset, on one day. The unit of the product."""

    asset_id: str
    train_id: str
    day: pd.Timestamp
    aspect: Aspect
    recommendation: str
    meaning: str
    margin_days: float          # the lower bound - what the decision used
    projected_days: float       # the point estimate - shown, never acted on
    confidence: str
    evidence: list[str]
    # The aspect today's margin would justify on its own. When the displayed
    # aspect is more severe than this, hysteresis is carrying it.
    margin_aspect: int = int(Aspect.GREEN)

    @classmethod
    def from_row(cls, row: pd.Series, policy: AspectPolicy | None = None) -> AspectCard:
        a = Aspect(int(row["aspect"]))
        policy = policy or AspectPolicy()
        return cls(
            asset_id=row["asset_id"],
            train_id=row.get("train_id", ""),
            day=row["day"],
            aspect=a,
            recommendation=RECOMMENDATION[a],
            meaning=MEANING[a],
            margin_days=float(row["rul_lower"]),
            projected_days=float(row["rul_point"]),
            confidence=confidence(policy, row["rul_point"], row["rul_lower"]),
            evidence=why(row),
            margin_aspect=int(policy.raw(np.array([row["rul_lower"]]))[0]),
        )

    @property
    def held(self) -> bool:
        """Elevated by hysteresis beyond what today's margin justifies.

        An earlier version tested only for an infinite margin, which missed the
        case that matters most. When a door is repaired the health index falls
        from ~40 sigma to ~0 overnight; the margin becomes large but FINITE, and
        the card rendered a WITHDRAW order beside evidence saying the door was
        healthier than its own baseline. That contradiction is worse than no
        card at all - an engineer who sees it once stops believing the next one.

        The right test is whether the displayed aspect outranks the aspect
        today's margin would earn on its own.
        """
        return self.aspect > self.margin_aspect

    def render(self) -> str:
        margin = "no failure path" if not np.isfinite(self.margin_days) \
            else f"{self.margin_days:.1f} days"
        conf = "held" if self.held else self.confidence
        lines = [
            f"  {self.asset_id}   {DISPLAY[self.aspect]} / {self.recommendation}",
            f"  {self.meaning}",
            f"  margin {margin}   confidence {conf}",
        ]
        if self.held:
            lines.append(
                f"    - HELD from an earlier escalation. Today's reading alone would"
                f" be {DISPLAY[Aspect(self.margin_aspect)]};"
            )
            lines.append(
                "      the order stands until the signal clears its dwell, or an"
                " engineer confirms the repair."
            )
        lines += [f"    - {e}" for e in self.evidence]
        return "\n".join(lines)
