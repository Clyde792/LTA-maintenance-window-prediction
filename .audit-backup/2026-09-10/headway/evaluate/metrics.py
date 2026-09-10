"""
Headway - operational evaluation metrics. The core of the Proving Ground.

WHAT THIS FILE REFUSES TO COMPUTE
---------------------------------
Accuracy. At a confirmed-fault base rate of 2e-05, a detector that answers
"healthy" to everything scores 99.998%. Any metric that rewards that is not
measuring a detector, it is measuring the base rate. The same objection applies
to ROC-AUC here: with tens of positives against hundreds of thousands of rows,
the false-positive-rate axis is compressed into invisibility and a useless
detector looks excellent.

WHAT IT COMPUTES INSTEAD
------------------------
Detectors are compared at an EQUAL ALERT BUDGET, denominated in ASSET-DAYS,
because that is the resource actually being spent: an engineer's attention, and
ultimately minutes inside a ~2-hour engineering window. Fixing the budget and
asking "what did you find with it?" is the only comparison that transfers to
the depot.

  detected        episodes caught at all. Coverage first - a detector that
                  misses an episode entirely cannot be rescued by a good median.
  median_lead_d   typical warning, in days before the confirmed fault.
  worst_lead_d    the shortest warning across episodes. THIS IS THE HEADLINE.
                  A detector whose worst warning is 2 days cannot be planned
                  around; one whose worst warning is 9 days can. Medians flatter
                  every detector equally; the worst case decides whether
                  deferring to an engineering window is a real option.
  precision       share of spent alerts that landed inside a genuine
                  degradation window - the false-alarm cost, at the true base
                  rate rather than a rebalanced one.
  capture         lead achieved / lead physically DETECTABLE, per episode.
                  No detector can warn before degradation begins, so raw lead
                  time silently rewards a detector for being handed long
                  degradation windows. Capture divides that out: 1.0 means the
                  detector flagged the asset from the first day it was possible
                  to. It is the metric that says whether the remaining headroom
                  is in the model or in the physics.

                  "Possible" is stricter than "after onset". At onset the wear
                  term is exactly zero, so the opening stretch of every window
                  sits below the noise floor of a daily aggregate no matter how
                  good the detector - on this data about 26% of each window
                  (scripts/diagnose.py, section [1]). When the episodes frame
                  carries a `detectable_days` column (the synthetic generator
                  declares one from its own physics), capture is measured
                  against that; otherwise it falls back to the full window and
                  should be read against a ceiling below 1.0.

                  Two earlier versions of this metric were wrong in ways worth
                  remembering: one divided the worst lead by the shortest
                  window - two different episodes - and claimed 86%; the other
                  measured against the full window and understated the model by
                  the undetectable ~26%. Per-episode, against the detectable
                  window, neither mistake is available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class DetectorResult:
    name: str
    budget: int
    alarms: int
    n_episodes: int
    detected: int
    median_lead_d: float
    worst_lead_d: float
    mean_lead_d: float
    precision: float
    median_capture: float = float("nan")
    worst_capture: float = float("nan")

    @property
    def missed(self) -> int:
        return self.n_episodes - self.detected

    def as_row(self) -> dict:
        d = asdict(self)
        d["missed"] = self.missed
        return d


def evaluate(
    daily: pd.DataFrame,
    score: str,
    episodes: pd.DataFrame,
    budget: int,
    *,
    name: str | None = None,
    day_col: str = "day",
    asset_col: str = "asset_id",
) -> DetectorResult:
    """Spend `budget` asset-day alerts on the highest scores; measure warning time.

    Args:
        daily: one row per asset per day, carrying `score`.
        score: column to rank by. Higher must mean worse.
        episodes: ground truth, with asset_id, onset_ts, fault_ts.
        budget: number of asset-days the detector is allowed to alert on.

    An alert counts toward an episode only if it falls between onset and the
    confirmed fault. Alerts fired after a fault are not credited - the asset has
    already failed, and telling someone about it is not a warning.
    """
    s = daily[score]
    valid = s.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return DetectorResult(name or score, budget, 0, len(episodes), 0,
                              np.nan, np.nan, np.nan, 0.0)

    # Take EXACTLY k rows. Selecting with `s >= threshold` would admit ties and
    # hand one detector a larger budget than another (observed: 152 alarms vs
    # 150), which quietly breaks the equal-budget comparison this whole module
    # exists to make. Ties are broken deterministically by day then asset so the
    # result does not depend on input row order.
    k = min(budget, n_valid)
    fired = (daily[valid]
             .sort_values([score, day_col, asset_col], ascending=[False, True, True])
             .head(k))

    leads: list[float] = []
    captures: list[float] = []
    useful = 0
    for _, e in episodes.iterrows():
        window = fired[
            (fired[asset_col] == e.asset_id)
            & (fired[day_col] >= pd.Timestamp(e.onset_ts).normalize())
            & (fired[day_col] <= pd.Timestamp(e.fault_ts).normalize())
        ]
        useful += len(window)
        if len(window):
            first = window[day_col].min()
            lead = (pd.Timestamp(e.fault_ts) - first).total_seconds() / 86400.0
            leads.append(lead)
            # The ceiling any detector could have reached on this episode:
            # the physically detectable window when the ground truth declares
            # one, the full onset-to-fault window otherwise.
            available = (pd.Timestamp(e.fault_ts)
                         - pd.Timestamp(e.onset_ts)).total_seconds() / 86400.0
            detectable = float(getattr(e, "detectable_days", np.nan))
            if np.isfinite(detectable) and 0 < detectable <= available:
                available = detectable
            if available > 0:
                captures.append(min(lead / available, 1.0))

    return DetectorResult(
        name=name or score,
        budget=budget,
        alarms=len(fired),
        n_episodes=len(episodes),
        detected=len(leads),
        median_lead_d=float(np.median(leads)) if leads else np.nan,
        worst_lead_d=float(np.min(leads)) if leads else np.nan,
        mean_lead_d=float(np.mean(leads)) if leads else np.nan,
        precision=useful / max(len(fired), 1),
        median_capture=float(np.median(captures)) if captures else np.nan,
        worst_capture=float(np.min(captures)) if captures else np.nan,
    )


def compare(
    daily: pd.DataFrame,
    scores: dict[str, str],
    episodes: pd.DataFrame,
    budget: int,
    **kw,
) -> pd.DataFrame:
    """Evaluate several detectors at one budget. `scores` maps label -> column."""
    return pd.DataFrame(
        [evaluate(daily, col, episodes, budget, name=label, **kw).as_row()
         for label, col in scores.items()]
    )


def budget_curve(
    daily: pd.DataFrame,
    score: str,
    episodes: pd.DataFrame,
    budgets: list[int],
    **kw,
) -> pd.DataFrame:
    """Sweep the alert budget - the operational trade-off curve.

    This replaces the ROC curve. The x-axis is a resource a depot manager
    actually controls (how many alerts per day the team can absorb) rather than
    a false-positive rate nobody can interpret at this base rate.
    """
    return pd.DataFrame(
        [evaluate(daily, score, episodes, b, **kw).as_row() for b in budgets]
    )


def format_table(df: pd.DataFrame) -> str:
    """Fixed-width rendering for the terminal and the README."""
    head = (f"    {'detector':<34} {'found':>7} {'median':>9} {'worst':>9} "
            f"{'prec':>7} {'capture':>9}")
    lines = [head, "    " + "-" * (len(head) - 4)]
    for _, r in df.iterrows():
        lines.append(
            f"    {r['name']:<34} {r['detected']:>3}/{r['n_episodes']:<3} "
            f"{r['median_lead_d']:>7.1f} d {r['worst_lead_d']:>7.1f} d "
            f"{r['precision']:>6.1%} {r['worst_capture']:>8.0%}"
        )
    return "\n".join(lines)
