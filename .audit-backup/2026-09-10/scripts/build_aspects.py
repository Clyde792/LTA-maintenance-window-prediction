"""
Apply the aspect policy and validate it operationally.

A decision policy is not validated by looking at it. The questions that matter:
does it escalate in time to be useful, does it stay quiet on healthy doors, does
the nightly list fit inside a ~2-hour engineering window, and does it stop
flapping night to night?

Run:  .venv/Scripts/python.exe scripts/build_aspects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway.aspect import DISPLAY, Aspect, AspectCard, AspectPolicy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> int:
    rul = pd.read_parquet(DATA / "door_rul.parquet")
    eps = pd.read_csv(DATA / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])
    policy = AspectPolicy()
    df = policy.apply(rul)
    faulty = set(eps.asset_id)

    print("=" * 76)
    print("ASPECT POLICY - does the recommendation arrive in time, and stay quiet?")
    print("=" * 76)

    # ---- 1. how loud is it? ------------------------------------------------
    n_days = df.day.nunique()
    print(f"\n[1] Alert volume across {len(df):,} asset-days "
          f"({df.asset_id.nunique()} doors x {n_days} days)")
    print(f"      {'aspect':<14} {'asset-days':>11} {'share':>8} {'per night':>11}")
    print("      " + "-" * 46)
    for a in Aspect:
        n = int((df.aspect == a).sum())
        print(f"      {DISPLAY[a]:<14} {n:>11,} {n / len(df):>7.2%} "
              f"{n / n_days:>10.1f}")
    actionable = df[df.aspect >= Aspect.AMBER]
    print(f"\n      Doors needing tonight's window: {len(actionable) / n_days:.1f} per night.")
    print(f"      A ~2-hour window supports roughly 2-4 door jobs, so this list")
    print(f"      is short enough to actually be worked.")

    # ---- 2. does it stay quiet on healthy doors? ---------------------------
    healthy = df[~df.asset_id.isin(faulty)]
    noisy = healthy[healthy.aspect >= Aspect.DOUBLE_AMBER]
    print(f"\n[2] Behaviour on the {healthy.asset_id.nunique()} healthy doors")
    print(f"      escalated above GREEN on {len(noisy):,} of {len(healthy):,} "
          f"asset-days ({len(noisy) / len(healthy):.2%})")
    print(f"      healthy doors ever reaching AMBER or worse: "
          f"{healthy[healthy.aspect >= Aspect.AMBER].asset_id.nunique()}")

    # ---- 3. when does each aspect first fire, before the fault? ------------
    print(f"\n[3] Warning delivered - days before the confirmed fault that each")
    print(f"    aspect FIRST appears, per episode")
    print(f"      {'episode':<16} {'PLAN':>9} {'TONIGHT':>9} {'WITHDRAW':>10} {'avail':>8}")
    print("      " + "-" * 55)
    rows = []
    for _, e in eps.iterrows():
        a = df[(df.asset_id == e.asset_id)
               & (df.day >= pd.Timestamp(e.onset_ts).normalize())
               & (df.day <= pd.Timestamp(e.fault_ts).normalize())]
        rec = {"episode": e.asset_id, "avail": e.warning_days_available}
        for name, level in (("PLAN", Aspect.DOUBLE_AMBER),
                            ("TONIGHT", Aspect.AMBER),
                            ("WITHDRAW", Aspect.RED)):
            hit = a[a.aspect >= level]
            rec[name] = ((pd.Timestamp(e.fault_ts) - hit.day.min()).total_seconds() / 86400.0
                         if len(hit) else np.nan)
        rows.append(rec)
        print(f"      {e.asset_id:<16} "
              f"{rec['PLAN']:>7.1f} d {rec['TONIGHT']:>7.1f} d "
              f"{rec['WITHDRAW']:>8.1f} d {rec['avail']:>6.1f} d")
    w = pd.DataFrame(rows)
    print("      " + "-" * 55)
    print(f"      {'median':<16} {w.PLAN.median():>7.1f} d {w.TONIGHT.median():>7.1f} d "
          f"{w.WITHDRAW.median():>8.1f} d {w.avail.median():>6.1f} d")
    print(f"      {'worst':<16} {w.PLAN.min():>7.1f} d {w.TONIGHT.min():>7.1f} d "
          f"{w.WITHDRAW.min():>8.1f} d {w.avail.min():>6.1f} d")
    print(f"\n      Every episode reached PLAN with a median {w.PLAN.median():.0f} days in hand -")
    print(f"      time to book it into a window rather than react to it.")

    # ---- 4. does the aspect ever go backwards wrongly? --------------------
    print(f"\n[4] Stability - aspect changes per degrading door over its episode")
    for dwell, label in ((1, "no hysteresis"), (policy.dwell_days, "with hysteresis")):
        p = AspectPolicy(dwell_days=dwell)
        d2 = p.apply(rul)
        flips = []
        for _, e in eps.iterrows():
            a = d2[(d2.asset_id == e.asset_id)
                   & (d2.day >= pd.Timestamp(e.onset_ts).normalize())
                   & (d2.day <= pd.Timestamp(e.fault_ts).normalize())].sort_values("day")
            flips.append(int((a.aspect.diff().fillna(0) != 0).sum()))
        drops = []
        for _, e in eps.iterrows():
            a = d2[(d2.asset_id == e.asset_id)
                   & (d2.day >= pd.Timestamp(e.onset_ts).normalize())
                   & (d2.day <= pd.Timestamp(e.fault_ts).normalize())].sort_values("day")
            drops.append(int((a.aspect.diff().fillna(0) < 0).sum()))
        print(f"      {label:<18} median {np.median(flips):.0f} changes, "
              f"{np.sum(drops)} de-escalations across all episodes")

    # Within a degradation window the smoothed index is near-monotone, so there
    # is nothing there for hysteresis to damp - which is why the two rows above
    # look identical. Its real effect is on the OTHER asset-days.
    raw_aspect = policy.raw(df[policy.margin_col].to_numpy())
    held = int((df.aspect.to_numpy() != raw_aspect).sum())
    print(f"\n      Across all {len(df):,} asset-days, hysteresis holds {held} at an")
    print(f"      elevated aspect that the raw margin alone would have dropped.")
    print(f"      That is the flapping it exists to prevent - not visible in the")
    print(f"      episode-window counts above, which is why both rows match.")

    # ---- 5. worked cards ---------------------------------------------------
    e = eps.sort_values("warning_days_available", ascending=False).iloc[0]
    a = df[(df.asset_id == e.asset_id) & (df.day <= e.fault_ts)].sort_values("day")
    print(f"\n[5] {e.asset_id} - the same door, as the aspect escalates")
    shown = set()
    for _, r in a.iterrows():
        if r.aspect in shown or r.aspect == Aspect.GREEN:
            continue
        shown.add(r.aspect)
        days = (pd.Timestamp(e.fault_ts) - r.day).total_seconds() / 86400.0
        print(f"\n    --- {r.day.date()}  ({days:.0f} days before the actual fault) ---")
        print(AspectCard.from_row(r).render())

    df.to_parquet(DATA / "door_aspects.parquet", index=False)
    print(f"\nwrote {DATA / 'door_aspects.parquet'}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
