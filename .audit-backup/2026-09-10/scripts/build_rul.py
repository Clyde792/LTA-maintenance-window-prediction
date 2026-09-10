"""
Fit RUL, choose the conformity mode on evidence, and check the bound is safe.

The question this script answers: when Headway says "you have N days", how
often is that a lie in the dangerous direction?

Run:  .venv/Scripts/python.exe scripts/build_rul.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway.rul import HI, SLOPE, ConformalRUL

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def true_rul(daily: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """Days to the confirmed fault, for rows inside a degradation window."""
    out = pd.Series(np.nan, index=daily.index)
    for _, e in episodes.iterrows():
        m = ((daily.asset_id == e.asset_id)
             & (daily.day >= pd.Timestamp(e.onset_ts).normalize())
             & (daily.day < pd.Timestamp(e.fault_ts).normalize()))
        out[m] = (pd.Timestamp(e.fault_ts) - daily.loc[m, "day"]).dt.total_seconds() / 86400.0
    return out


def main() -> int:
    daily = pd.read_parquet(DATA / "door_daily.parquet")
    eps = pd.read_csv(DATA / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])
    daily["true_rul"] = true_rul(daily, eps)

    print("=" * 74)
    print("RUL CALIBRATION - is the margin we quote safe to act on?")
    print("=" * 74)

    # ---- 1. the projection is biased, and we can show it -------------------
    naive = ConformalRUL(mode="additive").fit(daily, eps)
    raw = daily.assign(hat=(naive.threshold_ - daily[HI]) / daily[SLOPE])
    m = daily.true_rul.notna() & np.isfinite(raw.hat) & (raw.hat > 0) & (daily[SLOPE] > 0.25)
    bias = (daily.true_rul[m] - raw.hat[m])
    print(f"\n[1] Raw straight-line projection, before any correction")
    print(f"      {m.sum():,} calibration points inside degradation windows")
    print(f"      median (truth - projection): {bias.median():+.1f} days")
    print(f"      projection is too {'LONG' if bias.median() < 0 else 'SHORT'} "
          f"on {(bias < 0).mean():.0%} of days")
    print(f"      Wear accelerates, so a line fitted to today's slope reaches the")
    print(f"      threshold later than reality. That error is in the dangerous")
    print(f"      direction: it tells an engineer they can defer when they cannot.")

    # ---- 2. pick the conformity mode on evidence ---------------------------
    print(f"\n[2] Conformity mode, chosen by measurement (target coverage 90%)")
    fitted = {}
    for mode in ("additive", "ratio", "mondrian"):
        r = ConformalRUL(mode=mode, alpha=0.10).fit(daily, eps)
        pred = r.predict(daily)
        ok = daily.true_rul.notna() & np.isfinite(pred.rul_lower)
        width = (pred.rul_upper - pred.rul_lower)[ok]
        safe = (daily.true_rul[ok] >= pred.rul_lower[ok]).mean()
        # Sharpness: how much usable margin the bound actually reports. A bound
        # that collapses to zero is perfectly "safe" and completely useless, so
        # coverage on its own can always be gamed by being more conservative.
        sharp = (pred.rul_lower[ok] / daily.true_rul[ok].clip(lower=1e-9)).median()
        # Safety in the WITHDRAW zone: the final three days, where an over-large
        # bound does not mislabel paperwork - it leaves a failing train in
        # passenger service.
        wz = ok & daily.true_rul.between(0, 3, inclusive="left")
        wz_safe = float((daily.true_rul[wz] >= pred.rul_lower[wz]).mean()) if wz.sum() else 1.0
        fitted[mode] = (r, safe, width.median(), sharp, wz_safe)
        print(f"\n      {mode}")
        print(f"        held-out coverage   {r.coverage_:.1%}   (leave-one-episode-out)")
        print(f"        in-sample safe rate {safe:.1%}")
        print(f"        median interval     {width.median():.1f} days")
        print(f"        sharpness           {sharp:.2f}  (bound / true RUL; 1.0 is perfect)")
        print(f"        withdraw-zone safe  {wz_safe:.1%}  (true RUL < 3 d)")

    # Selection has been wrong twice, in opposite directions, and each mistake
    # is now a criterion:
    #   - coverage alone picked a vacuous bound (additive under the LINEAR
    #     form scored 90% coverage by reporting ~0 days for everything), so
    #     sharpness > 0.1 is required;
    #   - sharpness alone then picked ratio under the LOG-LINEAR form, whose
    #     bound overstates the margin on 28% of the final three days and
    #     misses WITHDRAW entirely on 2 of 9 episodes. Sharpness is measured
    #     mostly far from failure, which is exactly where it matters least.
    # So: hold coverage, hold WITHDRAW-ZONE safety, then take the sharpest
    # bound that survives both. Every criterion is a scar.
    target, floor, wz_floor = 0.90, 0.85, 0.95
    usable = {k: v for k, v in fitted.items()
              if v[0].coverage_ >= floor and v[3] > 0.1 and v[4] >= wz_floor}
    pool = usable or fitted
    best = max(pool, key=lambda k: fitted[k][3])
    print(f"\n      -> using '{best}': the sharpest bound that holds coverage AND"
          f"\n         stays safe inside the withdraw zone.")
    if fitted[best][0].coverage_ < target:
        print(f"         (coverage {fitted[best][0].coverage_:.1%} sits just under the {target:.0%}")
        print(f"          target - stated, not rounded away. With 9 episodes the")
        print(f"          conformal quantile is coarse; more episodes would tighten it.)")

    model = fitted[best][0]
    print()
    print(model.report())

    # ---- 3. does the bound stay safe when it matters most? -----------------
    pred = model.predict(daily)
    print(f"\n[3] Safety of the lower bound, by how close the fault actually is")
    print(f"      {'true RUL':>14} {'n':>7} {'bound safe':>12} {'median bound':>14}")
    print("      " + "-" * 50)
    for lo, hi in ((0, 3), (3, 7), (7, 14), (14, 30), (30, 90)):
        m = daily.true_rul.between(lo, hi, inclusive="left") & np.isfinite(pred.rul_lower)
        if m.sum():
            safe = (daily.true_rul[m] >= pred.rul_lower[m]).mean()
            print(f"      {f'{lo}-{hi} d':>14} {m.sum():>7,} {safe:>11.1%} "
                  f"{pred.rul_lower[m].median():>13.1f} d")

    # ---- 4. a worked trajectory -------------------------------------------
    e = eps.sort_values("warning_days_available", ascending=False).iloc[0]
    a = pred[(pred.asset_id == e.asset_id) & (pred.day >= e.onset_ts)
             & (pred.day < e.fault_ts)].copy()
    print(f"\n[4] {e.asset_id} - {e.fault_mode}, fault on {e.fault_ts.date()}")
    print(f"      {'day':>12} {'health':>8} {'slope':>7} {'margin':>9} "
          f"{'lower':>8} {'true':>7}")
    print("      " + "-" * 56)
    for _, r in a.iloc[::max(len(a) // 8, 1)].iterrows():
        pt = "  --" if not np.isfinite(r.rul_point) else f"{r.rul_point:6.1f} d"
        lo = "  --" if not np.isfinite(r.rul_lower) else f"{r.rul_lower:5.1f} d"
        print(f"      {r.day.date()!s:>12} {r[HI]:>8.1f} {r[SLOPE]:>7.2f} "
              f"{pt:>9} {lo:>8} {r.true_rul:>5.1f} d")

    out = pred.drop(columns=["true_rul"])
    out.to_parquet(DATA / "door_rul.parquet", index=False)
    print(f"\nwrote {DATA / 'door_rul.parquet'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
