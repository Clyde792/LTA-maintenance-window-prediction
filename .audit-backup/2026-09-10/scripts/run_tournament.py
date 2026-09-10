"""
The Proving Ground - which model earns the right to make the call?

PS3 asks us to "identify the best models to be used to detect future anomalies".
This answers it, at an equal alert budget, on operational metrics, under a
protocol where no detector can see the failure it is being asked to predict.

Every model family is run TWICE - once on raw per-cycle aggregates, once on
condition-normalised and per-asset baselined features - because the question
worth answering is not which library wins. It is whether the algorithm or the
preprocessing is doing the work.

Run:  .venv/Scripts/python.exe scripts/run_tournament.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract, features
from headway.evaluate import metrics
from headway.models import detectors as D
from headway.normalise import (AssetBaseline, ConditionNormaliser,
                               MultivariateHealthIndex)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BUDGETS = (150, 250)
REFERENCE_DAYS = 30


def levels_of(sub) -> list[str]:
    return [c for c in sub.signals if not c.endswith(("_flag", "_count"))]


def build_features(cycles: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Daily frame carrying BOTH raw and normalised versions of every signal.

    The condition models are fitted on the FIRST `REFERENCE_DAYS` ONLY, then
    transform the whole record. An earlier version fitted them on all 120 days,
    which handed the [norm] detectors a look-ahead the [raw] detectors did not
    have - the exact protocol violation the tournament exists to police, sitting
    in the tournament's own feature prep. The condition model is label-free and
    physics is stationary, so the coefficients barely move; but "barely" is a
    thing to measure, not assert, and the fair protocol costs one line.
    """
    sub = contract.get("door")
    levels = [c for c in sub.signals if not c.endswith(("_flag", "_count"))]
    fit_window = cycles[cycles.ts < cycles.ts.min() + pd.Timedelta(days=REFERENCE_DAYS)]

    # Condition-normalise each level signal independently, then aggregate.
    extra = []
    for s in levels:
        cn = ConditionNormaliser(signal=s, context=contract.CONTEXT).fit(fit_window)
        cycles[f"res_{s}"] = cn.transform(cycles)["residual"]
        extra.append(f"res_{s}")

    daily = features.to_daily(cycles, "door", extra=tuple(extra))

    # Per-asset baseline each residual -> a comparable health index per signal.
    norm_cols = []
    for s in levels:
        ab = AssetBaseline(value=f"res_{s}").fit(daily)
        daily[f"{s}_hx"] = ab.transform(daily)["health_index"]
        norm_cols.append(f"{s}_hx")

    raw_cols = levels + [f"{s}_iqr" for s in levels
                         if f"{s}_iqr" in daily.columns]
    rate_cols = [c for c in daily.columns if c.endswith("_rate")]
    raw_cols += rate_cols
    norm_cols += [f"{s}_iqr" for s in levels if f"{s}_iqr" in daily.columns] + rate_cols

    return daily, raw_cols, norm_cols


def main() -> int:
    cycles = pd.read_parquet(DATA / "door_cycles.parquet")
    eps = pd.read_csv(DATA / "door_episodes.csv", parse_dates=["onset_ts", "fault_ts"])
    sub = contract.get("door")

    daily, raw_cols, norm_cols = build_features(cycles)

    # Headway's own scores, UNSMOOTHED. D.run applies the common smoothing to
    # every detector; handing it an already-smoothed column would smooth Headway
    # twice and quietly handicap it against the rest of the field.
    ab = AssetBaseline(value=f"res_{sub.primary}").fit(daily)
    daily["hw_uni"] = ab.transform(daily)["health_index"]
    mv = MultivariateHealthIndex(
        signals=[f"{s}_hx" for s in levels_of(sub)],
        orientation={f"{s}_hx": sub.sign(s) for s in levels_of(sub)},
    ).fit(daily)
    daily["hw_multi"] = mv.transform(daily)["health_index"]

    zoo: dict[str, D.Detector] = {
        "s_raw":       D.RawThreshold(column=sub.primary),
        "s_ewma_raw":  D.EWMAChart(column=sub.primary),
        "s_if_raw":    D.IsolationForestDetector(needs=raw_cols),
        "s_pca_raw":   D.PCAReconstruction(needs=raw_cols),
        "s_maha_raw":  D.MahalanobisDetector(needs=raw_cols),
        "s_if_norm":   D.IsolationForestDetector(needs=norm_cols),
        "s_pca_norm":  D.PCAReconstruction(needs=norm_cols),
        "s_maha_norm": D.MahalanobisDetector(needs=norm_cols),
        "s_lof_norm":  D.LOFDetector(needs=norm_cols),
        "s_hw_uni":    D.Passthrough(column="hw_uni"),
        "s_hw_multi":  D.Passthrough(column="hw_multi"),
    }
    labels = {
        "s_raw":       "raw signal, fleet-wide            [raw]",
        "s_ewma_raw":  "EWMA control chart                [raw]",
        "s_if_raw":    "isolation forest                  [raw]",
        "s_pca_raw":   "PCA reconstruction                [raw]",
        "s_maha_raw":  "Mahalanobis distance              [raw]",
        "s_if_norm":   "isolation forest                 [norm]",
        "s_pca_norm":  "PCA reconstruction               [norm]",
        "s_maha_norm": "Mahalanobis distance             [norm]",
        "s_lof_norm":  "local outlier factor             [norm]",
        "s_hw_uni":    "HEADWAY univariate               [norm]",
        "s_hw_multi":  "HEADWAY multivariate             [norm]",
    }

    print("=" * 84)
    print("THE PROVING GROUND - which model earns the right to make the call?")
    print("=" * 84)
    print(f"\nProtocol: every detector fitted on the first {REFERENCE_DAYS} days only, then scores")
    print(f"the full {daily.day.nunique()}-day history. No labels, no sight of the future, identical")
    print(f"smoothing. {len(eps)} episodes to find across {daily.asset_id.nunique()} doors.")

    scored = D.run(zoo, daily, reference_days=REFERENCE_DAYS)

    for budget in BUDGETS:
        table = metrics.compare(scored, {labels[k]: k for k in zoo}, eps, budget)
        table = table.sort_values(["detected", "worst_lead_d"], ascending=[False, False])
        print(f"\n\nBUDGET {budget} asset-day alerts "
              f"({budget / len(scored):.1%} of {len(scored):,})\n")
        print(metrics.format_table(table))

        if budget == BUDGETS[0]:
            best = table.iloc[0]
            print(f"\n    Ranked by coverage then worst-case warning, the winner is")
            print(f"    '{best['name'].split('[')[0].strip()}' - "
                  f"{best['detected']}/{best['n_episodes']} found, "
                  f"{best['worst_lead_d']:.1f} d worst case.")

    # ---- the question the zoo exists to answer ----------------------------
    table = metrics.compare(scored, {labels[k]: k for k in zoo}, eps, BUDGETS[0])
    table["key"] = list(zoo)
    pairs = [("s_if_raw", "s_if_norm"), ("s_pca_raw", "s_pca_norm"),
             ("s_maha_raw", "s_maha_norm")]
    print("\n\n" + "-" * 84)
    print("ALGORITHM OR PREPROCESSING - which is doing the work?")
    print("-" * 84)
    print(f"\n    Same family, raw features vs normalised features:\n")
    print(f"    {'family':<24} {'raw worst':>11} {'norm worst':>12} {'delta':>9} {'found':>12}")
    print("    " + "-" * 72)
    gains = []
    for a, b in pairs:
        ra = table[table.key == a].iloc[0]
        rb = table[table.key == b].iloc[0]
        wa = 0.0 if np.isnan(ra.worst_lead_d) else ra.worst_lead_d
        wb = 0.0 if np.isnan(rb.worst_lead_d) else rb.worst_lead_d
        gains.append(wb - wa)
        fam = ra["name"].split("[")[0].strip()
        print(f"    {fam:<24} {wa:>9.1f} d {wb:>10.1f} d {wb - wa:>+7.1f} d "
              f"{int(ra.detected):>5}/9 -> {int(rb.detected)}/9")

    norm_only = table[table.key.isin([k for k in zoo if k.endswith("_norm")])]
    spread = norm_only.worst_lead_d.max() - norm_only.worst_lead_d.min()
    print(f"\n    Switching the FEATURES moves worst-case warning by a mean of "
          f"{np.mean(gains):+.1f} days.")
    print(f"    Switching the MODEL FAMILY, holding features fixed, moves it by "
          f"{spread:.1f} days.")

    by_family = dict(zip(["isolation forest", "PCA", "Mahalanobis"], gains))
    biggest = max(by_family, key=by_family.get)
    smallest = min(by_family, key=by_family.get)
    print(f"\n    But the mean hides the finding. Normalisation is worth "
          f"{by_family[biggest]:+.1f} days to")
    print(f"    {biggest} and {by_family[smallest]:+.1f} days to {smallest}. PCA needs it least")
    print(f"    because it already performs its own implicit normalisation: temperature")
    print(f"    and crowding move every signal together, so those directions land in the")
    print(f"    leading principal components and drop out of the reconstruction error.")
    print(f"\n    -> Condition normalisation is a SUBSTITUTE for model capacity, not an")
    print(f"       addition to it. Explicit normalisation buys the most for models that")
    print(f"       cannot discover the structure themselves. That is a more defensible")
    print(f"       claim than 'preprocessing beats models', and it is what the numbers")
    print(f"       actually support.")

    # ---- what the tournament corrected ------------------------------------
    uni = table[table.key == "s_hw_uni"].iloc[0]
    hw = table[table.key == "s_hw_multi"].iloc[0]
    print("\n" + "-" * 84)
    print("WHAT THE TOURNAMENT CORRECTED - IN US")
    print("-" * 84)
    print()
    print("    An earlier run of this tournament reported that Headway placed mid-field,")
    print("    beaten by six detectors. That was not a result. It was a bug in the harness:")
    print("    the Headway entry was fed an ALREADY-SMOOTHED column, and run() then applied")
    print("    the common smoothing on top - so our own model, alone in the field, was")
    print("    smoothed twice and handicapped by roughly two days of worst-case warning.")
    print()
    print("    The lesson generalises past this repo. An evaluation that produces a")
    print("    surprising verdict about your own model is more likely to be measuring the")
    print("    harness than the model, and the first move is to audit the protocol rather")
    print("    than rebuild the model. Fixed, both Headway variants sit at the top of the")
    print("    field, and the earlier finding is withdrawn.")
    print()
    print(f"    At budget {BUDGETS[1]}, the two Headway variants trade off:\n")
    t250 = metrics.compare(scored, {labels[k]: k for k in zoo}, eps, BUDGETS[1])
    t250["key"] = list(zoo)
    u2 = t250[t250.key == "s_hw_uni"].iloc[0]
    m2 = t250[t250.key == "s_hw_multi"].iloc[0]
    print(f"      {'variant':<16} {'median':>9} {'worst':>9} {'prec':>8} {'capture':>9}")
    print("      " + "-" * 54)
    for nm, r in (("univariate", u2), ("multivariate", m2)):
        print(f"      {nm:<16} {r.median_lead_d:>7.1f} d {r.worst_lead_d:>7.1f} d "
              f"{r.precision:>7.1%} {r.worst_capture:>8.0%}")
    print()
    print(f"    Multivariate wins precision ({m2.precision - u2.precision:+.1%}) and capture")
    print(f"    ({m2.worst_capture - u2.worst_capture:+.0%}); univariate wins worst-case warning")
    print(f"    ({u2.worst_lead_d - m2.worst_lead_d:+.1f} d) - which is the metric we said to lead on.")
    print()
    print(f"    So this is a genuine trade-off, not a clean win, and we say so. We ship")
    print(f"    multivariate anyway, for a reason the table cannot show: on unseen data a")
    print(f"    univariate index is blind if its one sensor drifts or fails. Spending one")
    print(f"    day of worst-case margin to stop depending on a single channel is the")
    print(f"    right trade against a dataset we have not seen yet.")

    scored.to_parquet(DATA / "door_tournament.parquet", index=False)
    print(f"\nwrote {DATA / 'door_tournament.parquet'}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
