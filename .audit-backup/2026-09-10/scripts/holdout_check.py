"""
Does the tuning survive data it has never seen?

scripts/diagnose.py sweeps hyperparameters against the same 9 episodes it
scores on. That is tuning on the test set, and with 9 positives it is very easy
to pick a setting that fits this particular draw of noise and nothing else.

So: regenerate the fleet under different seeds - new assets, new build
tolerances, new onsets, new episodes - and re-run the sweep. A setting that only
wins on the original seed is one we invented, not one we found.

The expensive step is condition normalisation (five Ridge fits over ~460k rows),
and it does not depend on any knob being swept. So it is done ONCE per fleet and
the sweeps reuse it - the difference between ~40 minutes and ~2.

Run:  .venv/Scripts/python.exe scripts/holdout_check.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract, features
from headway.evaluate import metrics
from headway.normalise import AssetBaseline, ConditionNormaliser, MultivariateHealthIndex
from headway.synth.doors import SynthConfig, generate

BUDGET = 250
SEEDS = [20260918, 7, 101, 2024]
SUB = contract.get("door")
LEVELS = [c for c in SUB.signals if not c.endswith(("_flag", "_count"))]


def residual_daily(cycles: pd.DataFrame) -> pd.DataFrame:
    """The expensive half, done once: normalise every signal, aggregate to days."""
    cyc = cycles.copy()
    for s in LEVELS:
        cyc[f"res_{s}"] = ConditionNormaliser(signal=s, context=contract.CONTEXT)\
            .fit_transform(cyc)["residual"]
    return features.to_daily(cyc, "door", extra=tuple(f"res_{s}" for s in LEVELS))


def index_of(daily: pd.DataFrame, ref_days=21, drop=()) -> pd.Series:
    """The cheap half: baseline, combine. Depends on ref_days and the signal set."""
    use = [s for s in LEVELS if s not in drop]
    d = daily.copy()
    for s in use:
        d[f"{s}_hx"] = AssetBaseline(value=f"res_{s}", reference_days=ref_days)\
            .fit_transform(d)["health_index"]
    mv = MultivariateHealthIndex(
        signals=[f"{s}_hx" for s in use],
        orientation={f"{s}_hx": SUB.sign(s) for s in use},
        reference_days=ref_days,
    )
    return mv.fit_transform(d)["health_index"]


def worst(daily: pd.DataFrame, idx: pd.Series, eps: pd.DataFrame, smooth_days=5) -> float:
    d = features.add_trend(daily.assign(health_index=idx), col="health_index",
                           smooth_days=smooth_days)
    return metrics.evaluate(d, "health_index_smooth", eps, BUDGET).worst_lead_d


def show(title: str, keys, table: dict, unit="d") -> None:
    print(f"\n{title}\n")
    print(f"    {'seed':>10}" + "".join(f"{str(k):>12}" for k in keys))
    print("    " + "-" * (10 + 12 * len(keys)))
    n = len(next(iter(table.values())))
    for i in range(n):
        print(f"    {SEEDS[i]:>10}" + "".join(f"{table[k][i]:>10.1f} {unit}" for k in keys))
    print("    " + "-" * (10 + 12 * len(keys)))
    print(f"    {'mean':>10}" + "".join(f"{np.mean(table[k]):>10.1f} {unit}" for k in keys))
    best = max(keys, key=lambda k: np.mean(table[k]))
    tuned = max(keys, key=lambda k: table[k][0])
    print(f"\n    best on mean across fleets: {best}")
    print(f"    best on seed 20260918 alone: {tuned}"
          + ("   <-- the tuned pick does NOT generalise" if tuned != best else "   (agrees)"))


def main() -> int:
    print("=" * 78)
    print("HOLDOUT CHECK - do the tuned settings hold on unseen fleets?")
    print("=" * 78)
    print(f"\n{len(SEEDS)} independent fleets, budget {BUDGET}, metric = worst-case warning.")
    print("Seed 20260918 is the one every earlier number was tuned on.")

    fleets = []
    for sd in SEEDS:
        cycles, eps = generate(replace(SynthConfig(), seed=sd))
        fleets.append((residual_daily(cycles), eps))
        print(f"  fleet seed={sd} built")

    smooth_keys = [1, 3, 5, 9]
    t1 = {k: [] for k in smooth_keys}
    for daily, eps in fleets:
        idx = index_of(daily)
        for k in smooth_keys:
            t1[k].append(worst(daily, idx, eps, smooth_days=k))
    show("[1] Smoothing window (currently 5)", smooth_keys, t1)

    ref_keys = [14, 21, 35, 50]
    t2 = {k: [] for k in ref_keys}
    for daily, eps in fleets:
        for k in ref_keys:
            t2[k].append(worst(daily, index_of(daily, ref_days=k), eps))
    show("[2] Reference window (currently 21)", ref_keys, t2)

    print("\n[3] Does dropping mean_current_a (derived, noisy) help?\n")
    print(f"    {'seed':>10} {'all':>12} {'without':>12} {'delta':>10}")
    print("    " + "-" * 46)
    deltas = []
    for (daily, eps), sd in zip(fleets, SEEDS):
        a = worst(daily, index_of(daily), eps)
        b = worst(daily, index_of(daily, drop=("mean_current_a",)), eps)
        deltas.append(b - a)
        print(f"    {sd:>10} {a:>10.1f} d {b:>10.1f} d {b - a:>+8.1f} d")
    print("    " + "-" * 46)
    print(f"    {'mean':>10} {'':>12} {'':>12} {np.mean(deltas):>+8.1f} d")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
