"""
Headway - the Deferral Ledger. What does it cost to wait?

THE QUESTION THIS ANSWERS
-------------------------
An Aspect Card gives one answer: AMBER, tonight, margin 2 days. But the planner
at 01:00 has constraints we know nothing about - crew, parts, three other jobs,
a possession already booked on the wrong sector. When our single answer is
impossible, their only options are to ignore it or override it, and both erode
trust in everything else we say.

So instead of one answer, price the alternatives:

    act tonight ................  0%
    wait to Thursday ..........  14%
    wait to next Tuesday ......  51%
    wait a fortnight ..........  89%

Each figure is the probability the asset fails IN SERVICE before that date. The
system stops being an oracle and becomes a pricing engine: it does not decide,
it hands the decision back with the cost of each option attached.

HOW THE NUMBERS ARE PRODUCED
----------------------------
ConformalRUL already yields a calibrated lower bound L(alpha) with the property

    P(true RUL >= L(alpha)) ~= 1 - alpha        so    P(RUL < L(alpha)) ~= alpha

Fit the SAME machinery at several alphas and L becomes a curve rather than a
point - a discrete survival function over remaining life:

    alpha 0.50 -> at least 12 days      alpha 0.10 -> at least 5 days
    alpha 0.25 -> at least  8 days      alpha 0.05 -> at least 3 days

Then read it backwards. Rather than "how many days do I get at 90% confidence?",
ask "if I take 12 days, what confidence is that?" - and interpolate alpha at the
horizon. No new model, no distributional assumption: every point on the curve is
a conformal bound whose coverage we already measure out-of-fold.

A pleasing consequence: `latest_date_under(r)` - the last date whose risk stays
under r - is exactly L(r). The `rul_lower` the Aspect Card already prints IS
"the latest date at 10% risk". The ledger does not bolt a new idea onto the
product; it exposes one that was there all along.

That identity is load-bearing, not decorative: it is what stops the card and the
ledger giving an engineer two different answers to the same question. It holds
only while both use the same conformal mode, so `base` defaults to the same
ConformalRUL the Aspect Card uses and a test asserts they agree exactly.

WHAT WE WILL NOT QUOTE, AND WHY THAT IS THE FEATURE WORKING
------------------------------------------------------------
Measured against outcomes, the ledger is accurate where it is used and vague
where it is not:

    stated risk band     what actually failed     n
    under 5%             0-5%                     350+     trustworthy
    5-15%                0-9%                      45      trustworthy
    15-30%               15-22%                    30      trustworthy
    30% and above        28-71% against ~45%      270      NOT trustworthy

The low bands are the ones a deferral decision is actually made in - "can this
wait a week?" is only ever asked of assets that plausibly can. Above 30% the
answer is "do not defer" whether the true figure is 45% or 70%, so precision
there buys nothing and claiming it would be false.

So `RISK_CEILING` caps what we print. Above it the ledger reports HIGH rather
than a number. That is not a hedge - quoting 53% when we have measured
ourselves accurate to about 20 points in that region would be inventing
precision, and one engineer checking one number against reality would cost us
the whole product's credibility.

With 9 confirmed episodes the deep tails are not earned either: the reportable
range is capped at the fitted alphas, and beyond them we say "< 5%" rather than
extrapolate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .rul import ConformalRUL

# Confidence levels to calibrate. Spaced to give resolution where decisions
# actually change hands - the 5-25% band - without pretending to a precision
# that 9 episodes cannot support.
DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)

# Horizons an engineer actually chooses between. Engineering windows are
# nightly, so "tonight" is 0 and the rest are the shapes of a maintenance plan:
# a few days, a week, a fortnight, a month.
DEFAULT_HORIZONS: tuple[float, ...] = (0.0, 3.0, 7.0, 14.0, 28.0)

# Above this, we report HIGH instead of a percentage. Set from the measured
# calibration, not taste: below it we track outcomes to within a few points,
# above it we are out by ~20 and the decision does not depend on the number.
RISK_CEILING: float = 0.30


@dataclass
class DeferralLedger:
    """A calibrated survival curve per asset-day, and the risk of waiting.

    Args:
        alphas: confidence levels to calibrate. Each becomes one point on the
            curve, and each carries its own measured out-of-fold coverage.
        horizons: candidate wait-times, in days, to price by default.
        base: template ConformalRUL. Cloned per alpha so the projection form,
            mode and thresholds stay identical across the curve - only the
            confidence level moves.

            The mode MUST match the one the Aspect Card uses. It is not a free
            choice, and getting it wrong put a visible contradiction on screen.

            This was first built on `ratio`, reasoning that an additive shift
            clamps 38.8% of bounds to zero and flattens 11.6% of curves, which
            looked like a loss of resolution the ledger could not afford. That
            reasoning was wrong twice over:

              - The clamping concentrates NEAR FAILURE, where every horizon
                being HIGH is the correct answer, not a lost distinction.
                Measured where deferral is actually a live question - doors
                under a PLAN order - the two modes discriminate IDENTICALLY:
                92.2% of rows quotable at 7 days either way, same median spread
                across horizons.

              - Meanwhile the mismatch broke the product's single voice. The
                card's margin comes from the aspect model's alpha=0.10 bound;
                the ledger's "latest date under 10% risk" is its own. Two valid
                conformal bounds under different modes disagreed by a median of
                2.5 days and up to 32, so a card could read WITHDRAW, margin
                1 day, and directly beneath it "latest date under 10% risk:
                5 days". An engineer reads that once and stops believing both
                numbers.

            So: same mode, same voice, and `latest_date_under(0.10)` is exactly
            the margin already printed - to the decimal, asserted by a test.
    """

    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    horizons: tuple[float, ...] = DEFAULT_HORIZONS
    base: ConformalRUL = field(default_factory=ConformalRUL)

    models_: dict[float, ConformalRUL] = field(default_factory=dict, init=False)
    coverage_: dict[float, float] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ fit
    def fit(self, daily: pd.DataFrame, episodes: pd.DataFrame) -> DeferralLedger:
        """Calibrate one conformal bound per confidence level."""
        for a in sorted(self.alphas):
            m = ConformalRUL(
                alpha=a,
                projection=self.base.projection,
                mode=self.base.mode,
                n_bins=self.base.n_bins,
                difficulty=self.base.difficulty,
                min_per_bin=self.base.min_per_bin,
                min_slope=self.base.min_slope,
                horizon=self.base.horizon,
            ).fit(daily, episodes)
            self.models_[a] = m
            self.coverage_[a] = m.coverage_
        return self

    # -------------------------------------------------------------- curve
    def bounds(self, daily: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_alphas) matrix of lower bounds, ascending in alpha.

        Sampling noise can occasionally invert two adjacent levels - a 75%
        bound landing above a 50% one - which would make the curve
        non-invertible. A running maximum enforces monotonicity, which is the
        conservative repair: it never raises a bound, only holds it up.
        """
        if not self.models_:
            raise RuntimeError("DeferralLedger.fit() must be called before use")
        cols = [self.models_[a].predict(daily)["rul_lower"].to_numpy(float)
                for a in sorted(self.alphas)]
        B = np.column_stack(cols)
        # NaN means "no bound yet" (too little history), not "a small bound".
        # Plain maximum.accumulate would propagate it across the whole row and
        # destroy every later level, so step over them and put them back.
        nan = np.isnan(B)
        filled = np.maximum.accumulate(np.where(nan, -np.inf, B), axis=1)
        return np.where(nan, np.nan, filled)

    def risk(self, daily: pd.DataFrame, horizon_days: float,
             bounds: np.ndarray | None = None) -> np.ndarray:
        """P(asset fails in service before `horizon_days` from now), per row.

        Clamped to the fitted range: below the tightest bound we report the
        smallest calibrated alpha, above the loosest we report the largest. We
        do not extrapolate into tails 9 episodes cannot support.
        """
        B = self.bounds(daily) if bounds is None else bounds
        A = np.array(sorted(self.alphas), dtype=float)

        # Acting now is not deferring. This returns before the already-failed
        # override below, which previously clobbered it and printed HIGH against
        # "act tonight" - advising an engineer that fixing a door immediately
        # was as risky as ignoring it for a month.
        if horizon_days <= 0:
            return np.zeros(len(B), dtype=float)

        out = np.zeros(len(B), dtype=float)
        for i in range(len(B)):
            row = B[i]
            finite = np.isfinite(row)
            if finite.any():
                out[i] = float(np.interp(horizon_days, row[finite], A[finite],
                                         left=A[0], right=A[-1]))
            elif np.isnan(row).all():
                # Too little history to project. That is NOT the same as safe,
                # and reporting 0 would claim a safety we have not established.
                out[i] = np.nan
            else:
                out[i] = 0.0                  # all infinite: no failure path
        # An asset already past its threshold has no future to wait into.
        out = np.where(np.isclose(B[:, -1], 0.0), 1.0, out)
        return out

    def latest_date_under(self, daily: pd.DataFrame, risk_budget: float) -> np.ndarray:
        """Days you may wait while keeping failure risk under `risk_budget`.

        This is exactly L(risk_budget) - so the Aspect Card's existing
        `rul_lower` is already 'the latest date at 10% risk'. Interpolated
        between fitted levels when the budget falls between them.
        """
        B = self.bounds(daily)
        A = np.array(sorted(self.alphas), dtype=float)
        out = np.full(len(B), np.inf)
        for i in range(len(B)):
            row = B[i]
            if not np.isfinite(row).any():
                continue
            finite = np.isfinite(row)
            out[i] = float(np.interp(risk_budget, A[finite], row[finite],
                                     left=row[finite][0], right=row[finite][-1]))
        return out

    # -------------------------------------------------------------- ledger
    def ledger(self, daily: pd.DataFrame,
               horizons: tuple[float, ...] | None = None) -> pd.DataFrame:
        """Add one `risk_<d>d` column per horizon, plus the safe-until date."""
        hs = horizons or self.horizons
        B = self.bounds(daily)
        out = daily.copy()
        for h in hs:
            out[f"risk_{int(h)}d"] = self.risk(daily, h, bounds=B)
        out["safe_days_10pct"] = self.latest_date_under(daily, 0.10)
        return out

    def report(self) -> str:
        lines = [f"DeferralLedger - {len(self.models_)} calibrated confidence levels"]
        lines.append(f"  {'stated':>8} {'held-out coverage':>19} {'error':>8}")
        for a in sorted(self.alphas):
            cov = self.coverage_[a]
            lines.append(f"  {1 - a:>7.0%} {cov:>18.1%} {cov - (1 - a):>+7.1%}")
        return "\n".join(lines)


def format_risk(p: float) -> str:
    """A percentage where we have earned one, a word where we have not."""
    if not np.isfinite(p):
        return "  --"
    if p <= 0.0:
        return "   0%"
    if p >= RISK_CEILING:
        return "HIGH"
    if p < 0.05:
        return "  <5%"
    return f"{p:>4.0%}"


def render(row: pd.Series, horizons: tuple[float, ...] = DEFAULT_HORIZONS) -> str:
    """The ledger for one asset-day, as an engineer would read it."""
    names = {0: "act tonight", 3: "wait 3 days", 7: "wait a week",
             14: "wait a fortnight", 28: "wait a month"}
    lines = [f"  {row['asset_id']}   what does it cost to wait?"]
    for h in horizons:
        key = f"risk_{int(h)}d"
        if key not in row:
            continue
        label = names.get(int(h), f"wait {int(h)} days")
        lines.append(f"    {label:<20} {format_risk(row[key]):>5}")
    safe = row.get("safe_days_10pct", np.nan)
    if np.isfinite(safe):
        lines.append(f"    {'-' * 26}")
        lines.append(f"    latest date under 10% risk: {safe:.0f} days")
    return "\n".join(lines)
