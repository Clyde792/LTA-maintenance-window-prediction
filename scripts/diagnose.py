"""
Where is the remaining headroom, and is it in the model or in the physics?

The capture metric says we deliver ~36% of the available warning at a 150-alert
budget, which reads as 64% headroom. This script asks whether that reading is
honest, then measures which knobs actually move the result.

Run:  .venv/Scripts/python.exe scripts/diagnose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract, features
from headway.evaluate import metrics
from headway.normalise import AssetBaseline, ConditionNormaliser, MultivariateHealthIndex

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
from headway.synth.doors import SynthConfig

BUDGET = 250
SIGNAL = "current_integral_as"
# Physics constants come from the generator itself. Hardcoded copies drift the
# moment someone tunes SynthConfig, and the detectability-floor calculation in
# [1] would then be quietly wrong.
_CFG = SynthConfig()
WEAR_COEFF = _CFG.wear_coeff
NOISE = _CFG.noise


def build(cycles: pd.DataFrame, smooth_days=3, slope_days=14, ref_days=21) -> pd.DataFrame:
    """The production pipeline, with the knobs exposed."""
    sub = contract.get("door")
    levels = [c for c in sub.signals if not c.endswith(("_flag", "_count"))]
    cyc = cycles.copy()
    extra = []
    for s in levels:
        cyc[f"res_{s}"] = ConditionNormaliser(signal=s, context=contract.CONTEXT)\
            .fit_transform(cyc)["residual"]
        extra.append(f"res_{s}")
    daily = features.to_daily(cyc, "door", extra=tuple(extra))
    for s in levels:
        daily[f"{s}_hx"] = AssetBaseline(value=f"res_{s}", reference_days=ref_days)\
            .fit_transform(daily)["health_index"]
    mv = MultivariateHealthIndex(
        signals=[f"{s}_hx" for s in levels],
        orientation={f"{s}_hx": sub.sign(s) for s in levels},
        reference_days=ref_days,
    )
    daily["health_index"] = mv.fit_transform(daily)["health_index"]
    return features.add_trend(daily, col="health_index",
                              smooth_days=smooth_days, slope_days=slope_days)


def main() -> int:
    print("LEGACY EXPLORATION: not independent validation. Use validate_pipeline.py for claims.")
    cycles = pd.read_parquet(DATA / "door_cycles.parquet")
    eps = pd.read_csv(DATA / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])
    base = build(cycles)

    print("=" * 78)
    print("DIAGNOSIS - where is the headroom, and what actually moves it?")
    print("=" * 78)

    # ---------------------------------------------------------------- [1]
    # Capture is measured against the window from ONSET. But at onset the wear
    # is exactly zero by construction, so the first days of every window are
    # below the chosen synthetic noise threshold. If that is a large share of the window, capture
    # is flattering the headroom and we are chasing a number we cannot reach.
    print("\n[1] Is the 'available warning' actually available?")
    print("    Detectability floor: the day the expected wear signal first clears")
    print("    2 standard errors of that day's aggregate. Physics, not our model.\n")
    print(f"    {'episode':<16} {'window':>8} {'detectable':>11} {'lost':>8} {'usable':>8}")
    print("    " + "-" * 55)
    rows = []
    for _, e in eps.iterrows():
        span = (e.fault_ts - e.onset_ts).total_seconds() / 86400.0
        a = base[base.asset_id == e.asset_id]
        n_per_day = a.n_cycles.median()
        se = NOISE / np.sqrt(max(n_per_day, 1))
        # h(t) = progress^shape ; detectable when WEAR_COEFF*h > 2*se
        h_min = 2 * se / WEAR_COEFF
        prog_min = h_min ** (1.0 / e.wear_shape)
        detectable_from = span * (1 - prog_min)      # days before fault
        rows.append({"span": span, "det": detectable_from})
        print(f"    {e.asset_id:<16} {span:>6.1f} d {detectable_from:>9.1f} d "
              f"{span - detectable_from:>6.1f} d {detectable_from / span:>7.0%}")
    r = pd.DataFrame(rows)
    usable = (r.det / r.span).mean()
    print("    " + "-" * 55)
    print(f"    {'mean':<16} {r.span.mean():>6.1f} d {r.det.mean():>9.1f} d "
          f"{(r.span - r.det).mean():>6.1f} d {usable:>7.0%}")
    print(f"\n    Only {usable:.0%} of each window is above the chosen synthetic noise threshold. Capture")
    print(f"    measured against the full window therefore has a ceiling near")
    print(f"    {usable:.2f}, not 1.00 - so our 36% at budget 150 is really about")
    print(f"    {0.36 / usable:.0%} of what is reachable, and 51% at 250 is about {0.51 / usable:.0%}.")

    # ---------------------------------------------------------------- [2]
    print("\n[2] What does the smoothing window cost in warning time?")
    print("    A trailing median lags a rising signal by roughly half its width.\n")
    print(f"    {'smooth_days':>12} {'found':>7} {'median':>9} {'worst':>9} {'prec':>7}")
    print("    " + "-" * 47)
    for sd in (1, 3, 5, 9, 15):
        d = build(cycles, smooth_days=sd)
        res = metrics.evaluate(d, "health_index_smooth", eps, BUDGET)
        print(f"    {sd:>12} {res.detected:>3}/{res.n_episodes:<3} "
              f"{res.median_lead_d:>7.1f} d {res.worst_lead_d:>7.1f} d {res.precision:>6.1%}")

    # ---------------------------------------------------------------- [3]
    print("\n[3] Which signal carries the early warning?")
    print("    Each normalised signal alone, against the multivariate index.\n")
    sub = contract.get("door")
    levels = [c for c in sub.signals if not c.endswith(("_flag", "_count"))]
    scores = {}
    for s in levels:
        col = f"{s}_hx"
        d = features.add_trend(base.assign(_v=base[col] * sub.sign(s)), col="_v")
        scores[s] = metrics.evaluate(d, "_v_smooth", eps, BUDGET, name=s)
    scores["MULTIVARIATE"] = metrics.evaluate(base, "health_index_smooth", eps, BUDGET,
                                              name="MULTIVARIATE")
    print(f"    {'signal':<24} {'found':>7} {'median':>9} {'worst':>9} {'prec':>7}")
    print("    " + "-" * 59)
    for k, v in sorted(scores.items(), key=lambda kv: -(kv[1].worst_lead_d or 0)):
        print(f"    {k:<24} {v.detected:>3}/{v.n_episodes:<3} "
              f"{v.median_lead_d:>7.1f} d {v.worst_lead_d:>7.1f} d {v.precision:>6.1%}")

    # ---------------------------------------------------------------- [4]
    print("\n[4] Does dispersion warn earlier than level?")
    print("    A worn mechanism binds intermittently before it binds always.\n")
    disp = [c for c in base.columns if c.endswith("_iqr")]
    for c in disp[:4]:
        d = features.add_trend(base.assign(_v=base[c]), col="_v")
        res = metrics.evaluate(d, "_v_smooth", eps, BUDGET)
        print(f"    {c:<32} {res.detected:>3}/9 {res.median_lead_d:>7.1f} d "
              f"{res.worst_lead_d:>7.1f} d {res.precision:>6.1%}")

    # ---------------------------------------------------------------- [5]
    print("\n[5] Reference window - how much healthy history does a baseline need?\n")
    print(f"    {'ref_days':>9} {'found':>7} {'median':>9} {'worst':>9} {'prec':>7}")
    print("    " + "-" * 44)
    for rd in (7, 14, 21, 35, 50):
        d = build(cycles, ref_days=rd)
        res = metrics.evaluate(d, "health_index_smooth", eps, BUDGET)
        print(f"    {rd:>9} {res.detected:>3}/{res.n_episodes:<3} "
              f"{res.median_lead_d:>7.1f} d {res.worst_lead_d:>7.1f} d {res.precision:>6.1%}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
