"""
Run the full health pipeline and measure what each stage is worth.

  cycles
    -> ConditionNormaliser   per signal, removes heat / crowding / time-of-day
    -> daily features        the asset-day, which is the unit engineers receive
    -> AssetBaseline         per signal, removes each door's build tolerance
    -> MultivariateHealth    one directional index over all signals
    -> trend                 trailing smooth / slope / volatility

Writes data/door_daily.parquet, which is the frame everything downstream
(RUL, Aspect Cards, Possession Planner, Hindsight) reads.

Run:  .venv/Scripts/python.exe scripts/build_health.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract, features
from headway.evaluate import metrics
from headway.normalise import (AssetBaseline, ConditionNormaliser,
                               MultivariateHealthIndex)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBSYSTEM = "door"
BUDGET = 150


def main() -> int:
    cycles = pd.read_parquet(DATA / "door_cycles.parquet")
    eps = pd.read_csv(DATA / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])
    contract.validate(cycles, SUBSYSTEM).raise_if_bad()
    sub = contract.get(SUBSYSTEM)
    levels = [c for c in sub.signals if not c.endswith(("_flag", "_count"))]

    print(f"cycles: {len(cycles):,}   assets: {cycles.asset_id.nunique()}   "
          f"episodes: {len(eps)}   signals: {len(levels)}")

    # ---- 1. remove the operating conditions, signal by signal --------------
    # Fitted on the FIRST 30 DAYS ONLY, then transforming the whole record.
    # This frame feeds lead-time claims, and a context model fitted through
    # day 120 has seen the future relative to every score it produces. The
    # coefficients barely move (the model is label-free and the physics is
    # stationary) - but that is now a property of the protocol, not a hope.
    fit_window = cycles[cycles.ts < cycles.ts.min() + pd.Timedelta(days=30)]
    for s in levels:
        cn = ConditionNormaliser(signal=s, context=contract.CONTEXT).fit(fit_window)
        cycles[f"res_{s}"] = cn.transform(cycles)["residual"]
        if s == sub.primary:
            print(f"\ncondition model on '{s}' - largest standardised effects:")
            print(cn.explain().head(4).to_string(index=False,
                                                float_format=lambda v: f"{v:+.4f}"))

    # ---- 2. collapse to the asset-day --------------------------------------
    daily = features.to_daily(cycles, SUBSYSTEM,
                              extra=tuple(f"res_{s}" for s in levels))
    print(f"\ndaily rows: {len(daily):,}  "
          f"({daily.day.min().date()} to {daily.day.max().date()})")

    # ---- 3. baseline each signal against the door's own history ------------
    for s in levels:
        daily[f"{s}_hx"] = AssetBaseline(value=f"res_{s}").fit_transform(daily)["health_index"]

    # ---- 4. combine into one directional index -----------------------------
    # Orientation matters: a worn door draws MORE current but travels LESS far,
    # so combining raw directions would cancel real evidence.
    mv = MultivariateHealthIndex(
        signals=[f"{s}_hx" for s in levels],
        orientation={f"{s}_hx": sub.sign(s) for s in levels},
    ).fit(daily)
    daily["health_index"] = mv.transform(daily)["health_index"]

    # Keep the univariate index alongside, so the ablation below can measure
    # exactly what going multivariate bought.
    daily["uni_index"] = daily[f"{sub.primary}_hx"]

    # ---- 5. trend ----------------------------------------------------------
    daily = features.add_trend(daily, col="health_index")
    daily = features.add_trend(daily, col="uni_index")

    # ---- 6. what is each stage worth? --------------------------------------
    daily["_A_raw"] = features.add_trend(
        daily.assign(_v=daily[sub.primary]), col="_v")["_v_smooth"]
    daily["_B_asset"] = features.add_trend(
        AssetBaseline(value=sub.primary).fit_transform(daily).rename(
            columns={"health_index": "_vb"}), col="_vb")["_vb_smooth"]

    table = metrics.compare(
        daily,
        {
            "A  raw fleet-wide threshold": "_A_raw",
            "B  + per-asset baseline": "_B_asset",
            "C  + condition normalisation": "uni_index_smooth",
            "D  + multivariate (all signals)": "health_index_smooth",
        },
        eps,
        BUDGET,
    )
    print(f"\nablation at {BUDGET} asset-day alerts "
          f"({BUDGET / len(daily):.1%} of {len(daily):,} asset-days):\n")
    print(metrics.format_table(table))

    c = table[table.name.str.startswith("C")].iloc[0]
    d = table[table.name.str.startswith("D")].iloc[0]
    print(f"\n    Going multivariate, at THIS budget: {d.worst_lead_d - c.worst_lead_d:+.1f} d worst "
          f"case, {d.median_lead_d - c.median_lead_d:+.1f} d median,")
    print(f"    {(d.precision - c.precision) * 100:+.1f} points of precision, "
          f"{d.worst_capture - c.worst_capture:+.0%} capture.")
    if abs(d.worst_lead_d - c.worst_lead_d) < 0.5:
        print(f"    Essentially a wash at 150 alerts - both are already at the ceiling")
        print(f"    the budget allows. The multivariate index earns its keep at LARGER")
        print(f"    budgets, where there is room to separate: see the curve below and")
        print(f"    compare against the univariate figures in the README.")

    # ---- 7. the trade-off curve that replaces ROC --------------------------
    curve = metrics.budget_curve(daily, "health_index_smooth", eps,
                                 [50, 100, 150, 250, 400, 800])
    print("\nalert-budget curve for the full pipeline:")
    print(f"    {'budget':>7} {'found':>7} {'median':>9} {'worst':>9} "
          f"{'prec':>7} {'capture':>9}")
    print("    " + "-" * 53)
    for _, r in curve.iterrows():
        print(f"    {r['budget']:>7,} {r['detected']:>3}/{r['n_episodes']:<3} "
              f"{r['median_lead_d']:>7.1f} d {r['worst_lead_d']:>7.1f} d "
              f"{r['precision']:>6.1%} {r['worst_capture']:>8.0%}")

    out = daily.drop(columns=[c for c in daily.columns if c.startswith("_")])
    out.to_parquet(DATA / "door_daily.parquet", index=False)
    print(f"\nwrote {DATA / 'door_daily.parquet'}  ({len(out):,} rows, {out.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
