"""Fixed, paired onboarding/calibration experiment. Does not change the demo.

Run all arms before comparing outcomes. The previously inspected fleet is
development evidence. A fixed new seed is a simulator replication, not external
validation. No test-fold outcome selects parameters or a deployment winner.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from headway.pipeline import HealthPipeline
from headway.onboarding import onboard
from headway.rul import ConformalRUL, prediction_time
from headway.calibration_experiments import GroupBalancedRUL
from headway.validation import coverage_summary, true_rul
from headway.synth.doors import SynthConfig, generate

REFERENCE_DAYS = 21
FLEET_REFERENCE_DAYS = 30
REPLICATION_SEED = 20260910
ARMS = ("fleet_pooled", "onboard_pooled", "onboard_balanced", "onboard_ratio")


def model_for(arm):
    if arm == "onboard_balanced":
        return GroupBalancedRUL(projection="loglinear", mode="additive")
    return ConformalRUL(projection="loglinear", mode="ratio" if arm == "onboard_ratio" else "additive")


def summary(pred, episodes):
    stats = coverage_summary(pred, episodes)
    truth = true_rul(pred, episodes)
    eligible = truth.notna()
    scored = eligible & np.isfinite(pred.rul_lower)
    useful = scored & pred.rul_lower.gt(0) & pred.rul_lower.le(truth)
    stats["positive_and_covered_fraction_all_episode_rows"] = float(useful.sum() / eligible.sum()) if eligible.any() else None
    over = scored & pred.rul_lower.gt(truth)
    stats["overstated_margin_fraction_all_episode_rows"] = float(over.sum() / eligible.sum()) if eligible.any() else None
    stats["mean_overstatement_days_when_overstated"] = float((pred.rul_lower[over] - truth[over]).mean()) if over.any() else 0.
    return stats


def run_fleet(cycles, episodes, name, outdir):
    start = cycles.ts.min().floor("D")
    ref_end = start + pd.Timedelta(days=FLEET_REFERENCE_DAYS)
    onboard_end = start + pd.Timedelta(days=REFERENCE_DAYS)
    end = cycles.ts.max().ceil("D")
    cutoff = (start + (end - start) * .7).floor("D")
    predictions = {p: {a: [] for a in ARMS} for p in ("train_holdout", "new_train_future")}
    enrollment = []
    for group in episodes.train_id.unique():
        training = cycles[cycles.train_id != group]
        target = cycles[cycles.train_id == group]
        pipe = HealthPipeline().fit(training[training.ts < ref_end])
        train_daily = pipe.transform(training)
        base_daily = pipe.transform(target)
        adapted = onboard(pipe, target[target.ts < onboard_end], experimental=True,as_of=onboard_end,
                          reference_days=REFERENCE_DAYS)
        adapted_daily = adapted.transform(target)
        enrollment.append({"group":group,"accepted":sorted(adapted.ready_at),"rejected":adapted.rejected})
        train_eps = episodes[episodes.train_id != group]
        held_eps = episodes[episodes.train_id == group]
        for protocol in predictions:
            td, te = train_daily, train_eps
            if protocol == "new_train_future":
                if not held_eps.fault_ts.gt(cutoff).any():
                    continue
                td = td[prediction_time(td) <= cutoff]
                te = te[te.fault_ts <= cutoff]
            for arm in ARMS:
                target_daily = base_daily if arm == "fleet_pooled" else adapted_daily
                target_daily = target_daily[prediction_time(target_daily) > ref_end]
                if protocol == "new_train_future":
                    target_daily = target_daily[prediction_time(target_daily) > cutoff]
                model = model_for(arm).fit(td, te, evaluate=False)
                pred = model.predict(target_daily)
                pred["evaluation_group"] = group
                pred["arm"] = arm
                pred["training_fault_groups"] = te.train_id.nunique()
                predictions[protocol][arm].append(pred)
        print(f"{name}: completed held-out train {group}", flush=True)
    reports = {}
    for protocol, arms in predictions.items():
        relevant = episodes if protocol == "train_holdout" else episodes[episodes.fault_ts > cutoff]
        reports[protocol] = {}
        for arm, pieces in arms.items():
            if not pieces:
                reports[protocol][arm] = {"error":"no eligible fault groups"}
                continue
            pred = pd.concat(pieces, ignore_index=True)
            pred.to_parquet(outdir / f"{name}_{protocol}_{arm}.parquet", index=False)
            result = summary(pred, relevant)
            per_group = {}
            for group, g in pred.groupby("evaluation_group"):
                per_group[group] = summary(g, relevant[relevant.train_id == group])
            result["per_group"] = per_group
            reports[protocol][arm] = result
    return {"cutoff":str(cutoff),"fault_groups":int(episodes.train_id.nunique()),
            "onboarding":enrollment,"results":reports}


def clean(v):
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, list): return [clean(x) for x in v]
    if isinstance(v, (float,np.floating)) and not np.isfinite(v): return None
    if isinstance(v,np.generic): return v.item()
    return v


def main():
    outdir = ROOT / "data/onboarding_experiment"
    outdir.mkdir(exist_ok=True)
    protocol = {"arms":ARMS,"asset_reference_days":REFERENCE_DAYS,
        "fleet_reference_days":FLEET_REFERENCE_DAYS,"alpha":.1,
        "projection":"loglinear","replication_seed":REPLICATION_SEED,
        "selection":"none; fixed paired comparisons, no automatic promotion",
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations":["Existing fleet was already inspected; this is development evidence.",
            "New seed uses the same simulator family, not real or external fleet validation.",
            "Onboarding assumes the initial reference is suitable; completeness does not establish health.",
            "Group-balanced quantiles are empirical; no conformal or conditional probability guarantee.",
            "Train holdout uses other trains' later outcomes; only new_train_future also enforces time cutoff."]}
    # Written before outcomes are evaluated, making the configured experiment reviewable.
    (outdir/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    report = {"protocol":protocol,"fleets":{}}
    c = pd.read_parquet(ROOT/"data/door_cycles.parquet")
    e = pd.read_csv(ROOT/"data/door_episodes.csv",parse_dates=["onset_ts","fault_ts"])
    report["fleets"]["development"] = run_fleet(c,e,"development",outdir)
    (outdir/"results.json").write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding="utf-8")
    del c,e
    c,e = generate(SynthConfig(seed=REPLICATION_SEED))
    report["fleets"]["replication"] = run_fleet(c,e,"replication",outdir)
    (outdir/"results.json").write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding="utf-8")
    rows = []
    for fleet, data in report["fleets"].items():
        for protocol, arms in data["results"].items():
            for arm, stats in arms.items():
                rows.append({"fleet":fleet,"protocol":protocol,"arm":arm,
                             **{k:v for k,v in stats.items() if k!="per_group"}})
    table = pd.DataFrame(rows)
    table.to_csv(outdir/"comparison.csv",index=False)
    print(table[["fleet","protocol","arm","macro_lower_coverage","abstention_rate",
                 "point_mae_days","positive_and_covered_fraction_all_episode_rows"]].to_string(index=False),flush=True)
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
