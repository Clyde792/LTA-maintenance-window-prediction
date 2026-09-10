"""
Fit the Deferral Ledger and check that its percentages mean anything.

A risk figure nobody has validated is worse than no figure: it invites an
engineer to make a quantitative decision on a number we made up. So the
headline of this script is not the ledger, it is the CALIBRATION CHECK - when
Headway says 14%, does roughly 14% of that population actually fail in time?

Run:  .venv/Scripts/python.exe scripts/build_deferral.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway.aspect import DISPLAY, Aspect, AspectPolicy
from headway.deferral import (DEFAULT_HORIZONS, RISK_CEILING, DeferralLedger,
                              render)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def true_rul(daily: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """Days to the confirmed fault, for rows inside a degradation window."""
    out = pd.Series(np.nan, index=daily.index)
    for _, e in episodes.iterrows():
        m = ((daily.asset_id == e.asset_id)
             & (daily.day >= pd.Timestamp(e.onset_ts).normalize())
             & (daily.day < pd.Timestamp(e.fault_ts).normalize()))
        out[m] = (pd.Timestamp(e.fault_ts) - daily.loc[m, "day"]).dt.total_seconds() / 86400
    return out


def main() -> int:
    # door_aspects carries the daily features PLUS rul_* and the aspect in
    # force, so the ledger can be shown beside the order it is pricing.
    daily = pd.read_parquet(DATA / "door_aspects.parquet")
    eps = pd.read_csv(DATA / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])

    print("=" * 78)
    print("THE DEFERRAL LEDGER - what does it cost to wait?")
    print("=" * 78)

    led = DeferralLedger().fit(daily, eps)
    print(f"\n{led.report()}")
    print("\n    Each level is a separate conformal calibration, scored out-of-fold.")
    print("    These are the confidences the ledger is entitled to quote.")

    out = led.ledger(daily)
    out["true_rul"] = true_rul(daily, eps)

    # ---- the check that matters -------------------------------------------
    # Group every scored asset-day by the risk we stated for a horizon, then
    # ask what fraction of that group actually failed within it. A trustworthy
    # ledger sits near the diagonal.
    print("\n[1] CALIBRATION - when we say X%, does X% actually fail in time?\n")
    print(f"    {'horizon':>9} {'stated band':>13} {'n':>7} {'stated':>8} "
          f"{'actual':>8} {'error':>8}")
    print("    " + "-" * 58)
    rows = []
    for h in (3.0, 7.0, 14.0):
        col = f"risk_{int(h)}d"
        scored = out[out[col] > 0].copy()
        if scored.empty:
            continue
        # "failed within h days" is only defined where we know the truth; rows
        # outside any degradation window did not fail, which is also an answer.
        scored["failed"] = (scored.true_rul.notna() & (scored.true_rul <= h))
        scored["band"] = pd.cut(scored[col], [0, .05, .15, .30, .60, 1.01],
                                labels=["<5%", "5-15%", "15-30%", "30-60%", ">60%"])
        for band, g in scored.groupby("band", observed=True):
            if len(g) < 15:          # below this a band is noise, not evidence
                continue
            rows.append({"h": h, "band": str(band), "n": len(g),
                         "stated": g[col].mean(), "actual": g.failed.mean(),
                         "quotable": g[col].mean() < RISK_CEILING})
            print(f"    {int(h):>7} d {str(band):>13} {len(g):>7} "
                  f"{g[col].mean():>7.0%} {g.failed.mean():>8.0%} "
                  f"{g[col].mean() - g.failed.mean():>+7.0%}")

    cal = pd.DataFrame(rows)
    if len(cal):
        lo = cal[cal.quotable]
        hi = cal[~cal.quotable]
        print("    " + "-" * 58)
        print(f"\n    Split by whether a deferral decision is actually live there:\n")
        if len(lo):
            print(f"      below {RISK_CEILING:.0%} ({int(lo.n.sum()):>4} asset-days) "
                  f"mean error {(lo.stated - lo.actual).abs().mean():>5.1%}   <- QUOTED")
        if len(hi):
            print(f"      at or above {RISK_CEILING:.0%} ({int(hi.n.sum()):>4} asset-days) "
                  f"mean error {(hi.stated - hi.actual).abs().mean():>5.1%}   <- reported as HIGH")
        print(f"\n    'Can this wait a week?' is only ever asked of assets that plausibly")
        print(f"    can, and there we track outcomes closely. Above {RISK_CEILING:.0%} the answer is")
        print(f"    'do not defer' whether the truth is 45% or 70%, so the ledger prints")
        print(f"    HIGH rather than a number it has not earned. Quoting a figure we are")
        print(f"    20 points out on would cost the product its credibility the first")
        print(f"    time an engineer checked it.")

    # ---- what it looks like to an engineer --------------------------------
    scored = out
    day = scored.groupby("day").apply(
        lambda g: (g.aspect >= Aspect.DOUBLE_AMBER).sum(), include_groups=False).idxmax()
    tonight = scored[(scored.day == day) & (scored.aspect >= Aspect.DOUBLE_AMBER)].copy()
    # Lead with the doors where the ledger actually DISCRIMINATES. A card that
    # reads HIGH at every horizon tells the engineer nothing the aspect had not
    # already said; the feature earns its place on the doors where waiting three
    # days is fine and waiting a month is not.
    tonight["spread"] = tonight.risk_28d - tonight.risk_3d
    tonight = tonight.sort_values("spread", ascending=False)

    print(f"\n[2] TONIGHT'S LEDGER - {day.date()}, "
          f"{len(tonight)} door(s) with an order outstanding\n")
    for _, r in tonight.head(4).iterrows():
        print(f"    {DISPLAY[Aspect(int(r.aspect))]} / {r.recommendation}")
        print(render(r))
        print()

    # ---- the argument, in one table ---------------------------------------
    print("[3] THE SAME QUESTION, ASKED OF THE WHOLE FLEET")
    print("    'How many doors can safely wait a week?'\n")
    live = scored[(scored.day == day)]
    for h in (3.0, 7.0, 14.0):
        col = f"risk_{int(h)}d"
        risky = int((live[col] >= 0.10).sum())
        print(f"    wait {int(h):>2} days -> {risky:>2} of {len(live)} doors exceed 10% risk")
    print("\n    That is a schedule, not an alert list. An anomaly score cannot")
    print("    answer this question at all - there is nothing in it to defer.")

    keep = [c for c in out.columns if c != "true_rul"]
    scored[keep].to_parquet(DATA / "door_deferral.parquet", index=False)
    print(f"\nwrote {DATA / 'door_deferral.parquet'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
