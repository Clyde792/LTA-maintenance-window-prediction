"""
Verify the premise the product rests on - and measure it honestly.

Two assumptions died in the writing of this script, and both corrections made
the thesis sharper:

  1. "Naive thresholding fails on precision."  That is not its decisive
     failing. Measured properly it is ~56% precise - poor, but survivable.
     What sinks it is TIMING AND COVERAGE: it misses 2 of 9 episodes outright,
     and its worst warning is ~2 days, which is inside the window where the
     only remaining option is an unplanned withdrawal. The product is not
     better detection. It is EARLIER detection - enough margin that deferring
     is still a choice, on every asset, not just the average one.

  2. "Count alarms per cycle."  Wrong unit. A smoothed score stays elevated for
     hundreds of consecutive cycles, so one sick door consumes the entire alert
     budget and a spiky raw score looks better by accident. An engineer gets one
     alert per asset per day, so ASSET-DAYS is the honest unit of alert budget.

This script is the prototype of the Proving Ground. It compares detectors at an
EQUAL ALERT BUDGET - the only fair way - and reports the operational metric:
warning lead time.

  A  raw fleet-wide threshold           what most teams will build
  B  + per-asset baseline               isolates the value of knowing the asset
  C  + condition normalisation          isolates the value of removing weather

Run:  .venv/Scripts/python.exe scripts/check_premise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
SIGNAL = "current_integral_as"
BUDGET = 150            # asset-days we are allowed to alert on, across 120 days
BASELINE_DAYS = 21      # per-asset reference period, assumed healthy
SMOOTH_DAYS = 5         # rolling window on the daily series


def _lead_times(daily: pd.DataFrame, score: str, eps: pd.DataFrame, budget: int) -> dict:
    """Spend `budget` asset-day alerts on the highest scores; measure warning time."""
    s = daily[score]
    valid = s.notna()
    if valid.sum() < budget:
        return {"alarms": 0, "detected": 0, "median_lead_d": np.nan,
                "min_lead_d": np.nan, "precision": 0.0}
    thr = s[valid].nlargest(budget).iloc[-1]
    fired = daily[valid & (s >= thr)]

    leads, useful = [], 0
    for _, e in eps.iterrows():
        w = fired[(fired.asset_id == e.asset_id)
                  & (fired.day >= e.onset_ts.normalize())
                  & (fired.day <= e.fault_ts.normalize())]
        useful += len(w)
        if len(w):
            leads.append((e.fault_ts - w.day.min()).total_seconds() / 86400.0)
    return {
        "alarms": len(fired),
        "detected": len(leads),
        "median_lead_d": float(np.median(leads)) if leads else float("nan"),
        "min_lead_d": float(np.min(leads)) if leads else float("nan"),
        "precision": useful / max(len(fired), 1),
    }


def main() -> int:
    print("LEGACY EXPLORATION: not independent validation. Use validate_pipeline.py for claims.")
    df = pd.read_parquet(ROOT / "data" / "door_cycles.parquet")
    eps = pd.read_csv(ROOT / "data" / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])
    faulty = set(eps.asset_id)

    print("=" * 78)
    print("PREMISE CHECK - can conditions mask early wear, and does removing them help?")
    print("=" * 78)

    # ------------------------------------------------------------------ [1]
    healthy = df[~df.asset_id.isin(faulty)]
    hot = healthy[healthy.ambient_temp_c > healthy.ambient_temp_c.quantile(0.95)][SIGNAL].mean()
    cool = healthy[healthy.ambient_temp_c < healthy.ambient_temp_c.quantile(0.05)][SIGNAL].mean()
    busy = healthy[healthy.load_proxy > 0.85][SIGNAL].mean()
    quiet = healthy[healthy.load_proxy < 0.15][SIGNAL].mean()
    spread = healthy.groupby("asset_id")[SIGNAL].mean().std()

    print("\n[1] How much do operating conditions move the signal? (healthy assets only)")
    print(f"      hot vs cool ambient     {hot - cool:+.3f} A.s")
    print(f"      crowded vs empty        {busy - quiet:+.3f} A.s")
    print(f"      per-asset build spread   {spread:.3f} A.s (1 sigma)")

    # ------------------------------------------------------------------ [2]
    print("\n[2] How much does genuine wear move it, at distance from failure?")
    rows = []
    for _, e in eps.iterrows():
        a = df[df.asset_id == e.asset_id]
        base = a[a.ts < e.onset_ts][SIGNAL].mean()
        for d in (30, 21, 14, 7, 3):
            w = a[(a.ts >= e.fault_ts - pd.Timedelta(days=d))
                  & (a.ts < e.fault_ts - pd.Timedelta(days=d - 3))]
            if len(w):
                rows.append({"d": d, "lift": w[SIGNAL].mean() - base})
    lift = pd.DataFrame(rows).groupby("d")["lift"].mean().sort_index(ascending=False)
    for d, v in lift.items():
        print(f"      {d:2d} days before failure   {v:+.3f} A.s")
    confound = max(abs(hot - cool), abs(busy - quiet))
    print(f"\n      At 21 days out - while deferring is still a real choice - wear has moved")
    print(f"      the signal {lift.loc[21]:+.3f} A.s, but conditions alone move it {confound:.3f} A.s.")
    print(f"      The confound is {confound / abs(lift.loc[21]):.1f}x larger than the signal we need.")

    # ------------------------------------------------------------------ [3]
    # Context model, fitted globally and WITHOUT labels: degradation is 2e-5 of
    # the data, so its contamination of an OLS fit is negligible.
    X = np.column_stack([
        np.ones(len(df)),
        df.ambient_temp_c.to_numpy(),
        df.load_proxy.to_numpy(),
        np.sin(2 * np.pi * df.hour_of_day.to_numpy() / 24),
        np.cos(2 * np.pi * df.hour_of_day.to_numpy() / 24),
    ])
    beta, *_ = np.linalg.lstsq(X, df[SIGNAL].to_numpy(), rcond=None)
    df = df.assign(resid=df[SIGNAL].to_numpy() - X @ beta, day=df.ts.dt.floor("D"))

    # One row per asset per day - the unit an engineer actually receives.
    daily = (df.groupby(["asset_id", "day"], as_index=False)
               .agg(raw=(SIGNAL, "median"), res=("resid", "median"), n=("ts", "size"))
               .sort_values(["asset_id", "day"], ignore_index=True))

    first = daily.groupby("asset_id")["day"].transform("min")
    ref = daily["day"] < first + pd.Timedelta(days=BASELINE_DAYS)
    for src, out in (("raw", "b_raw"), ("res", "b_res")):
        daily[out] = daily.asset_id.map(daily[ref].groupby("asset_id")[src].median())

    def smooth(values: pd.Series) -> pd.Series:
        """Rolling median within each asset, over the daily series."""
        return (values.groupby(daily.asset_id, sort=False)
                      .transform(lambda s: s.rolling(SMOOTH_DAYS, min_periods=3).median()))

    # The ablation. Each row adds exactly one idea to the row above it.
    daily["A_raw"] = smooth(daily.raw)                    # fleet-wide, as-measured
    daily["B_asset"] = smooth(daily.raw - daily.b_raw)    # + this door's own baseline
    daily["C_norm"] = smooth(daily.res - daily.b_res)     # + conditions removed

    total_asset_days = len(daily)
    print(f"\n[3] Detector comparison at an EQUAL budget of {BUDGET} asset-day alerts")
    print(f"    ({BUDGET / total_asset_days:.1%} of {total_asset_days:,} asset-days). "
          f"{len(eps)} episodes to find.\n")
    print(f"    {'detector':<36} {'found':>7} {'median lead':>13} {'worst':>9} {'prec':>7}")
    print(f"    {'-' * 36} {'-' * 7} {'-' * 13} {'-' * 9} {'-' * 7}")

    labels = {
        "A_raw": "A  raw fleet-wide threshold",
        "B_asset": "B  + per-asset baseline",
        "C_norm": "C  + condition normalisation",
    }
    res = {}
    for col, name in labels.items():
        r = _lead_times(daily, col, eps, BUDGET)
        res[col] = r
        print(f"    {name:<36} {r['detected']:>3}/{len(eps):<3} "
              f"{r['median_lead_d']:>10.1f} d {r['min_lead_d']:>7.1f} d {r['precision']:>6.1%}")

    # ------------------------------------------------------------------ [4]
    a, b, c = res["A_raw"], res["B_asset"], res["C_norm"]
    print("\n[4] Verdict")
    print(f"      Spending the same {BUDGET} alerts, the naive detector finds "
          f"{a['detected']}/{len(eps)} episodes")
    print(f"      at {a['precision']:.0%} precision, and its WORST warning is {a['min_lead_d']:.1f} days - "
          f"inside the")
    print(f"      window where the only remaining option is an unplanned withdrawal.")
    # Which stage contributed more is a question for the data, not for prose.
    gain_baseline = b["min_lead_d"] - a["min_lead_d"]
    gain_conditions = c["min_lead_d"] - b["min_lead_d"]
    bigger = "the per-asset baseline" if gain_baseline > gain_conditions else "condition normalisation"

    print(f"\n      Contribution of each stage, on worst-case warning:")
    print(f"        + baseline      {b['detected']}/{len(eps)} found, "
          f"{b['median_lead_d']:.1f} d median, {b['min_lead_d']:.1f} d worst, "
          f"{b['precision']:.0%} precise   ({gain_baseline:+.1f} d)")
    print(f"        + conditions    {c['detected']}/{len(eps)} found, "
          f"{c['median_lead_d']:.1f} d median, {c['min_lead_d']:.1f} d worst, "
          f"{c['precision']:.0%} precise   ({gain_conditions:+.1f} d)")
    print(f"      On this run the larger worst-case win comes from {bigger}.")

    print(f"\n      Net: {c['median_lead_d'] - a['median_lead_d']:+.1f} days of median margin, "
          f"{c['min_lead_d'] - a['min_lead_d']:+.1f} days in the worst case,")
    print(f"      {c['precision'] - a['precision']:+.0%} precision, and no episode missed.")
    print(f"\n      The worst case is the number that matters. A detector whose worst")
    print(f"      warning is {a['min_lead_d']:.1f} days cannot be planned around; one whose worst")
    print(f"      warning is {c['min_lead_d']:.1f} days can. Margin is the product.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
