"""Baseline vs outage-tolerant warning score, on fresh synthetic seeds.

Two stages, deliberately separated so parameters cannot be chosen using the
numbers that are later reported:

  --stage dev    sweep candidate window policies on the DEVELOPMENT seeds and
                 write dev_results.json plus the policy the frozen criteria pick.
  --stage eval   run only the FROZEN policy on the EVALUATION seeds.

The development and evaluation seeds are disjoint, and both differ from the seeds
in ROBUSTNESS_RESULTS.md, so nothing here is tuned against a result that is then
presented as an untouched test. `protocol.json` is written before either stage
runs and records the parameters, splits, budget, acceptance criteria and module
hashes.

Nothing in this script promotes a model. The dashboard's detector, thresholds and
smoother are untouched.
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.evaluate.metrics import evaluate_replay
from headway.models import detectors as D
from headway.outage_tolerant import (SUPPORTED, WindowPolicy, availability,
                                     recovery_days, recovery_summary,
                                     score_outage_tolerant)
from headway.pipeline import HealthPipeline
from headway.robustness import align_to_expected, perturb, score_frozen
from headway.synth.doors import SynthConfig, generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outage_experiment"

# Disjoint from each other and from ROBUSTNESS_RESULTS.md's (20260911, 20260912).
DEV_SEEDS = (20260913, 20260914)
EVAL_SEEDS = (20260915, 20260916)

FIT_DAYS, CUTOFF_DAY = 30, 84
THRESHOLD_QUANTILE, DAILY_BUDGET, COOLDOWN_DAYS = .99, 2, 3.
BLACKOUT_START, BLACKOUT_LENGTH = 5, 10       # elapsed days after the cutoff

SCENARIOS = ("clean", "missing_channel", "missing_days", "blackout",
             "sensor_offset", "unsupported_context")

# Candidate grid for the development stage only.
GRID = [WindowPolicy(min_observations=m, max_age_days=a, max_gap_days=g)
        for m in (2, 3) for a in (5., 7., 10.) for g in (2., 3., 4.) if g <= a]

# There is no a-priori "correct" policy. The development stage selects one from
# GRID using CRITERIA, breaks ties with TIE_BREAK, and writes it to
# frozen_policy.json. The evaluation stage REFUSES to run without that file, so a
# policy can never be chosen while looking at evaluation-seed numbers.
FROZEN_FILE = "frozen_policy.json"
TIE_BREAK = [
    ("outage availability on affected assets", "max"),   # the failure being fixed
    ("min_observations", "max"),                          # then require more evidence
    ("max_age_days", "min"),                              # then prefer fresher evidence
    ("max_gap_days", "min"),                              # then tolerate smaller holes
]

# Every criterion is decided on the DEVELOPMENT seeds and then applied unchanged.
CRITERIA = {
    "outage_availability_affected_min": .50,
    "clean_availability_regression_max": .02,
    "clean_detection_regression_max": 0,
    "clean_unmatched_alert_increase_max": .5,
    # Coverage is checked BEFORE speed so a candidate cannot qualify by
    # abandoning the assets it finds hard and recovering the rest quickly.
    "recovery_coverage_not_worse_than_baseline": True,
    "recovery_not_slower_than_baseline": True,
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def code_hashes():
    """Everything whose change could alter a result produced under this policy."""
    return {
        "experiment_script": sha256_file(Path(__file__)),
        "detector_module": sha256_file(ROOT / "headway/outage_tolerant.py"),
        "baseline_module": sha256_file(ROOT / "headway/robustness.py"),
        "metrics_module": sha256_file(ROOT / "headway/evaluate/metrics.py"),
    }


def clean_json(v):
    if isinstance(v, dict): return {k: clean_json(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean_json(x) for x in v]
    if isinstance(v, (float, np.floating)) and not np.isfinite(v): return None
    if isinstance(v, np.generic): return v.item()
    return v


def blackout(cycles, *, after, assets, start_offset, length):
    """Remove a CONTIGUOUS telemetry block, so recovery has a defined resume."""
    begin = pd.Timestamp(after).floor("D") + pd.Timedelta(days=start_offset)
    end = begin + pd.Timedelta(days=length)
    drop = cycles.asset_id.isin(assets) & cycles.ts.ge(begin) & cycles.ts.lt(end)
    return cycles[~drop].copy(), end


def stress(cycles, scenario, *, after, assets, seed):
    if scenario == "blackout":
        return blackout(cycles, after=after, assets=assets,
                        start_offset=BLACKOUT_START, length=BLACKOUT_LENGTH)
    return perturb(cycles, scenario, after=after, assets=assets, seed=seed), None


def fit_seed(cfg):
    """Fit preprocessing and detectors once per seed, on training data only."""
    cycles, eps = generate(cfg)
    start = cycles.ts.min().floor("D")
    ref_end = start + pd.Timedelta(days=FIT_DAYS)
    cutoff = start + pd.Timedelta(days=CUTOFF_DAY)
    held = sorted(cycles.train_id.unique())[0]          # fixed identity, not by outcome
    training = cycles[cycles.train_id != held]
    pipe = HealthPipeline().fit(training[training.ts < ref_end])
    tr = pipe.transform(training)
    names = [f"{s}_hx" for s in pipe.levels]
    zoo = {"raw": D.RawThreshold("current_integral_as"),
           "directional": D.Passthrough("health_index"),
           "mahalanobis": D.MahalanobisDetector(needs=names)}
    fit = tr[tr.day < ref_end]
    for model in zoo.values():
        needs = model.needs if hasattr(model, "needs") else [model.column]
        model.fit(fit[np.isfinite(fit[needs]).all(axis=1)])

    expected = cycles[["asset_id", "train_id", "ts"]].assign(day=cycles.ts.dt.floor("D"))
    expected = expected[["asset_id", "train_id", "day"]].drop_duplicates()
    expected["available_at"] = expected.day + pd.Timedelta(days=1)
    expected = expected[expected.available_at > cutoff]
    affected = sorted(cycles.loc[cycles.train_id.isin(sorted(cycles.train_id.unique())[:3]),
                                 "asset_id"].unique())
    return dict(cfg=cfg, cycles=cycles, eps=eps, pipe=pipe, zoo=zoo, tr=tr,
                ref_end=ref_end, cutoff=cutoff, held=held, expected=expected,
                affected=affected, future_eps=eps[eps.fault_ts > cutoff])


def score_with(ctx, daily, policy):
    """`policy=None` selects the preserved baseline smoother."""
    return score_frozen(ctx["zoo"], daily) if policy is None else \
        score_outage_tolerant(ctx["zoo"], daily, policy)


def thresholds_for(ctx, policy):
    """Each method gets its OWN frozen threshold from the same calibration span:
    the two produce different score distributions, so sharing one would rig the
    comparison. Calibration ends at the cutoff, before any replay."""
    scored = score_with(ctx, ctx["tr"], policy)
    cal = scored[(scored.available_at > ctx["ref_end"]) & (scored.available_at <= ctx["cutoff"])]
    return {name: float(pd.to_numeric(cal[name], errors="coerce").quantile(THRESHOLD_QUANTILE))
            for name in ctx["zoo"]}


def measure(ctx, scenario, policy, thresholds, daily, resumed_at, label):
    zoo, expected = ctx["zoo"], ctx["expected"]
    scored = score_with(ctx, daily, policy)
    scored = scored[scored.available_at > ctx["cutoff"]]
    aligned = align_to_expected(scored, expected, zoo)
    rows = []
    for population, ids in (("whole_fleet", set(expected.asset_id)),
                            ("affected_assets", set(ctx["affected"]))):
        frame = aligned[aligned.asset_id.isin(ids)]
        episodes = ctx["future_eps"][ctx["future_eps"].asset_id.isin(ids)]
        for name in zoo:
            stats = evaluate_replay(frame, name, episodes, thresholds[name],
                                    daily_budget=DAILY_BUDGET, cooldown_days=COOLDOWN_DAYS).as_row()
            alarms = stats["alarms"]
            matched = int(round(stats["precision"] * alarms)) if alarms else 0
            rec = None
            if resumed_at is not None and population == "affected_assets":
                # Every affected asset is in the denominator, and unrecovered
                # assets censor the median rather than dropping out of it.
                rec = recovery_summary(recovery_days(scored, name, resumed_at=resumed_at,
                                                     assets=ctx["affected"]))
            rows.append({
                "seed": ctx["cfg"].seed, "scenario": scenario, "method": label,
                "policy": policy.as_dict() if policy else None,
                "population": population, "detector": name,
                "expected_asset_days": len(frame),
                "supported_fraction_of_expected": availability(frame, name, len(frame)),
                "mechanical_episodes": int(len(episodes)),
                "episodes_detected": int(stats["detected"]),
                "median_lead_d": stats["median_lead_d"], "worst_lead_d": stats["worst_lead_d"],
                "alerts_total": int(alarms), "alerts_matched_mechanical": matched,
                "alerts_unmatched": int(alarms - matched),
                "unmatched_per_asset_month": stats["false_alerts_per_asset_month"],
                "unmatched_attribution": ("sensor perturbation injected on the affected assets"
                                          if scenario == "sensor_offset" else "not attributed"),
                "recovery": rec, "threshold": thresholds[name],
            })
    return rows


def run_seed(cfg, policies, stage):
    """`policies` maps a label to a WindowPolicy or None for the baseline."""
    ctx = fit_seed(cfg)
    thresholds = {label: thresholds_for(ctx, p) for label, p in policies.items()}
    rows = []
    for scenario in SCENARIOS:
        changed, resumed = stress(ctx["cycles"], scenario, after=ctx["cutoff"],
                                  assets=ctx["affected"], seed=cfg.seed)
        daily = ctx["pipe"].transform(changed)
        for label, policy in policies.items():
            rows += measure(ctx, scenario, policy, thresholds[label], daily, resumed, label)
        print(f"  seed {cfg.seed} [{stage}] {scenario} done", flush=True)
    return rows, {"held_train": ctx["held"], "affected_assets": ctx["affected"],
                  "thresholds": thresholds}


def pick(rows):
    """Apply the frozen criteria to development rows and report every candidate."""
    df = pd.DataFrame(rows)
    base = df[df.method == "baseline"]
    verdicts = []
    for label in sorted(set(df.method) - {"baseline"}):
        cand = df[df.method == label]

        def stat(frame, scenario, population, column):
            sel = frame[(frame.scenario == scenario) & (frame.population == population)]
            return float(pd.to_numeric(sel[column], errors="coerce").mean())

        outage = stat(cand, "missing_days", "affected_assets", "supported_fraction_of_expected")
        clean_av = stat(cand, "clean", "whole_fleet", "supported_fraction_of_expected")
        base_av = stat(base, "clean", "whole_fleet", "supported_fraction_of_expected")
        clean_det = stat(cand, "clean", "whole_fleet", "episodes_detected")
        base_det = stat(base, "clean", "whole_fleet", "episodes_detected")
        clean_un = stat(cand, "clean", "whole_fleet", "unmatched_per_asset_month")
        base_un = stat(base, "clean", "whole_fleet", "unmatched_per_asset_month")

        def rec_stats(frame):
            """Coverage first, then speed; a censored median is unresolved, not fast."""
            sel = frame[(frame.scenario == "blackout") & (frame.population == "affected_assets")]
            recs = [r for r in sel["recovery"] if r]
            if not recs:
                return None, None
            cov = float(np.mean([r["assets_recovered"] / max(r["assets_total"], 1) for r in recs]))
            meds = [r["median_days"] for r in recs]
            return cov, (None if any(m is None for m in meds) else float(np.mean(meds)))

        (cand_cov, cand_rec), (base_cov, base_rec) = rec_stats(cand), rec_stats(base)
        checks = {
            "outage_availability_affected": (outage, outage >= CRITERIA["outage_availability_affected_min"]),
            "clean_availability_regression": (base_av - clean_av,
                                              base_av - clean_av <= CRITERIA["clean_availability_regression_max"]),
            "clean_detection_regression": (base_det - clean_det,
                                           base_det - clean_det <= CRITERIA["clean_detection_regression_max"]),
            "clean_unmatched_increase": (clean_un - base_un,
                                         clean_un - base_un <= CRITERIA["clean_unmatched_alert_increase_max"]),
            "recovery_coverage": (None if cand_cov is None or base_cov is None else cand_cov - base_cov,
                                  cand_cov is not None and base_cov is not None and cand_cov >= base_cov - 1e-9),
            "recovery_not_slower": (None if cand_rec is None or base_rec is None else cand_rec - base_rec,
                                    cand_rec is not None and base_rec is not None and cand_rec <= base_rec + 1e-9),
        }
        verdicts.append({"policy_label": label, "policy": cand.policy.iloc[0],
                         "checks": {k: {"value": v, "pass": bool(ok)} for k, (v, ok) in checks.items()},
                         "qualifies": all(ok for _, ok in checks.values())})
    return verdicts


def select_policy(verdicts):
    """Deterministic choice among qualifying candidates; see TIE_BREAK."""
    qualifying = [v for v in verdicts if v["qualifies"]]
    if not qualifying:
        return None
    def key(v):
        p = v["policy"]
        return (-v["checks"]["outage_availability_affected"]["value"],
                -p["min_observations"], p["max_age_days"], p["max_gap_days"])
    return sorted(qualifying, key=key)[0]


def _norm_policy(p):
    """Compare policies by value, so 3 and 3.0 are not treated as different."""
    if not isinstance(p, dict):
        return None
    try:
        return (int(p["min_observations"]), float(p["max_age_days"]), float(p["max_gap_days"]))
    except (KeyError, TypeError, ValueError):
        return None


def protocol():
    return {
        "purpose": "Compare the preserved three-consecutive-day smoother with an "
                   "experimental elapsed-time window that tolerates irregular observations.",
        "development_seeds": list(DEV_SEEDS), "evaluation_seeds": list(EVAL_SEEDS),
        "seeds_differ_from_reported_robustness_run": True,
        "fleet": {"n_trains": 12, "doors_per_train": 2, "n_days": 120,
                  "cycles_per_day": 35, "n_episodes": 6},
        "splits": {"fit_days": FIT_DAYS, "calibration": f"days {FIT_DAYS}-{CUTOFF_DAY}",
                   "replay": f"after day {CUTOFF_DAY}",
                   "held_train": "first train identity, excluded from all fitting"},
        "alerting": {"daily_budget": DAILY_BUDGET, "cooldown_days": COOLDOWN_DAYS,
                     "threshold_quantile": THRESHOLD_QUANTILE,
                     "note": "each method is calibrated to its own threshold on the same "
                             "pre-replay span; sharing one threshold would rig the comparison"},
        "scenarios": list(SCENARIOS),
        "blackout": {"start_offset_days": BLACKOUT_START, "length_days": BLACKOUT_LENGTH},
        "candidate_grid": [p.as_dict() for p in GRID],
        "acceptance_criteria": CRITERIA,
        "selection_rule": {
            "stage": "development seeds only",
            "qualify": "every acceptance criterion must pass",
            "tie_break": [f"{k} ({how})" for k, how in TIE_BREAK],
            "frozen_to": FROZEN_FILE,
            "note": "The evaluation stage loads the selected policy from that file and "
                    "will not run without it. Acceptance criteria were fixed before the "
                    "development sweep ran; the tie-break order was added once the sweep "
                    "showed several qualifiers, and was applied without reference to any "
                    "evaluation-seed result.",
        },
        "detectors": ["raw", "directional", "mahalanobis"],
        "code": code_hashes(),
        "limitations": [
            "Small fixed synthetic fleets; not external validation and not evidence of rail performance.",
            "Fault labels are unchanged by sensor and context perturbations, so an unmatched alert is not necessarily a pointless inspection.",
            "Whole-fleet and affected-asset replays hold separate alert budgets; their counts are not additive.",
            "Correlated daily rows; episode counts are small and are not independent failures.",
            "No model is promoted and no operational threshold is changed by this experiment.",
        ],
    }


def run_dir(out, run_id):
    return Path(out) / "runs" / run_id


def make_run_id(proto):
    stamp = pd.Timestamp.now(tz="Asia/Singapore").strftime("%Y%m%dT%H%M%S")
    return f"dev-{stamp}-{sha256_bytes(canonical(proto))[:8]}"


def canonical(obj):
    """Stable bytes for hashing a JSON document, independent of key order."""
    return json.dumps(clean_json(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_frozen(out, run_id):
    """Load a frozen policy and REFUSE it unless it still binds to its evidence.

    A frozen policy is only meaningful together with the protocol it was chosen
    under, the development results that chose it, and the code that produced
    those results. Any of the three drifting makes the policy stale, so each is
    hashed into the freeze and re-checked here.
    """
    d = run_dir(out, run_id)
    if not d.is_dir():
        raise SystemExit(f"no such run: {d}. Run --stage dev first.")
    path = d / FROZEN_FILE
    if not path.exists():
        raise SystemExit(f"{path} is missing: that development run froze no policy "
                         "(no candidate qualified). Evaluation cannot proceed.")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not frozen.get("policy"):
        raise SystemExit(f"{path} records no qualifying policy. Evaluation cannot proceed.")

    problems = []
    if frozen.get("run_id") != run_id:
        problems.append(f"freeze names run {frozen.get('run_id')!r}, loaded from {run_id!r}")
    for label, fname in (("protocol", "protocol.json"), ("development results", "dev_results.json")):
        key = f"{fname}_sha256"
        actual = sha256_file(d / fname) if (d / fname).exists() else None
        if actual is None:
            problems.append(f"{label} missing from the run directory")
        elif frozen.get(key) != actual:
            problems.append(f"{label} has changed since the policy was frozen")
    # The protocol this code would write NOW must match the one in the run.
    if (d / "protocol.json").exists() and sha256_bytes(canonical(protocol())) != frozen.get("protocol_canonical_sha256"):
        problems.append("parameters or acceptance criteria differ from the frozen protocol")
    now, then = code_hashes(), frozen.get("code", {})
    for k, v in now.items():
        if then.get(k) != v:
            problems.append(f"{k} changed since the policy was frozen")
    if problems:
        raise SystemExit("Refusing to evaluate a stale policy:\n  - " + "\n  - ".join(problems)
                         + "\nRe-run --stage dev to freeze a policy against the current code.")

    # Nothing above hashes the freeze file itself, so its policy could be edited
    # in place after the run and every hash would still agree. Re-derive the
    # winner from the now hash-verified development results and require an exact
    # match: the freeze may only name the policy the sweep actually selected.
    dev = json.loads((d / "dev_results.json").read_text(encoding="utf-8"))
    verdicts, label = dev.get("verdicts"), frozen.get("policy_label")
    if not verdicts:
        problems.append("the development results record no candidate verdicts, so the "
                        "frozen policy cannot be re-derived from evidence")
    else:
        named = next((v for v in verdicts if v.get("policy_label") == label), None)
        winner = select_policy(verdicts)
        if named is None:
            problems.append(f"policy {label!r} was never evaluated in development")
        elif not named.get("qualifies"):
            problems.append(f"policy {label!r} did not qualify under the pre-registered criteria")
        if winner is None:
            problems.append("no candidate qualified in the development results")
        elif label != winner.get("policy_label"):
            problems.append(f"freeze names policy {label!r} but the development results "
                            f"select {winner.get('policy_label')!r}")
        elif _norm_policy(frozen.get("policy")) != _norm_policy(winner.get("policy")):
            problems.append(f"frozen parameters {frozen.get('policy')} differ from those of "
                            f"the selected candidate {winner.get('policy')}")
    if problems:
        raise SystemExit("Refusing a frozen policy that does not match its evidence:\n  - "
                         + "\n  - ".join(problems)
                         + "\nThe freeze may only name the policy the development sweep selected.")
    return frozen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("dev", "eval"), required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--run", help="run id; required for --stage eval, so evaluation "
                                  "can never silently pick up an older frozen policy")
    ap.add_argument("--reproduction", help="identifier for re-running an evaluation whose "
                                           "artifacts already exist; results are written to "
                                           "runs/<run>/reproductions/<id>/ and the original "
                                           "is left untouched")
    args = ap.parse_args(argv)
    proto = protocol()
    if args.reproduction and args.stage != "eval":
        raise SystemExit("--reproduction applies to --stage eval only. A development sweep "
                         "that needs repeating is a new run, not a reproduction of an old one.")

    if args.stage == "dev":
        run_id = args.run or make_run_id(proto)
        d = run_dir(args.out, run_id)
        if d.exists():
            # A run directory is the record of what was decided and on what
            # evidence. Re-using one would silently rewrite that record.
            raise SystemExit(f"{d} already exists. Development runs never overwrite an "
                             "earlier one: choose a new --run id, or leave --run unset "
                             "to have one generated.")
        d.mkdir(parents=True)
        # Written before the sweep, so parameters and criteria are on record
        # ahead of any number they will be judged against.
        (d / "protocol.json").write_text(json.dumps(clean_json(proto), indent=2), encoding="utf-8")
        policies = {"baseline": None}
        policies.update({f"m{p.min_observations}_a{p.max_age_days:g}_g{p.max_gap_days:g}": p for p in GRID})
        seeds, name = DEV_SEEDS, "dev"
    else:
        if not args.run:
            raise SystemExit("--run is required for --stage eval: name the development run "
                             "whose frozen policy you are evaluating.")
        run_id = args.run
        d = run_dir(args.out, run_id)
        frozen = load_frozen(args.out, run_id)
        if args.reproduction:
            d = d / "reproductions" / args.reproduction
            if d.exists():
                raise SystemExit(f"{d} already exists. Name a different --reproduction id.")
            d.mkdir(parents=True)
        policies = {"baseline": None, "frozen": WindowPolicy(**frozen["policy"])}
        seeds, name = EVAL_SEEDS, "eval"
        print(f"Evaluating frozen policy {frozen['policy_label']} from run {run_id}")

    # Checked before the sweep runs, so a refusal costs nothing.
    clash = [p.name for p in (d / f"{name}_results.json", d / f"{name}_comparison.csv")
             if p.exists()]
    if clash:
        raise SystemExit(f"{', '.join(clash)} already exist in {d}. Published results are "
                         "never overwritten: re-run with --reproduction <id> to record this "
                         "as a separately identified reproduction alongside the original.")

    rows, meta = [], {}
    for seed in seeds:
        cfg = SynthConfig(n_trains=12, doors_per_train=2, n_days=120,
                          cycles_per_day=35, n_episodes=6, seed=seed)
        got, info = run_seed(cfg, policies, name)
        rows += got
        meta[str(seed)] = info

    report = {"stage": name, "run_id": run_id, "reproduction": args.reproduction,
              "seeds": list(seeds), "blinding": "unasserted",
              "code": code_hashes(), "meta": meta, "rows": rows}
    if name == "dev":
        report["verdicts"] = pick(rows)
        qualifying = [v for v in report["verdicts"] if v["qualifies"]]
        report["qualifying_policies"] = [v["policy_label"] for v in qualifying]
        chosen = select_policy(report["verdicts"])
        report["selected"] = chosen
        print(f"\nQualifying candidate policies: {report['qualifying_policies'] or 'none'}")

    (d / f"{name}_results.json").write_text(
        json.dumps(clean_json(report), indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(d / f"{name}_comparison.csv", index=False)

    if name == "dev":
        chosen = report["selected"]
        if chosen:
            # Bind the policy to the exact protocol, development results and code
            # that produced it. Evaluation re-checks every one of these.
            (d / FROZEN_FILE).write_text(json.dumps(clean_json({
                "run_id": run_id,
                "policy": chosen["policy"], "policy_label": chosen["policy_label"],
                "selected_on_seeds": list(DEV_SEEDS),
                "qualifying_candidates": report["qualifying_policies"],
                "tie_break": [f"{k} ({how})" for k, how in TIE_BREAK],
                "development_checks": chosen["checks"],
                "protocol.json_sha256": sha256_file(d / "protocol.json"),
                "protocol_canonical_sha256": sha256_bytes(canonical(proto)),
                "dev_results.json_sha256": sha256_file(d / "dev_results.json"),
                "code": code_hashes(),
                "frozen_at": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
                # This script cannot know what has already been scored on this
                # machine, so it never asserts blindness on the operator's behalf.
                "blinding": "unasserted",
                "note": "Frozen from the development seeds under the recorded protocol and "
                        "code. Whether the evaluation seeds were still unscored when this "
                        "was written is not something this script can determine; any "
                        "blindness claim must be asserted and evidenced by the operator.",
            }), indent=2), encoding="utf-8")
            print(f"Froze {chosen['policy_label']} -> {d / FROZEN_FILE}")
            print(f"Evaluate with:  --stage eval --run {run_id}")
        else:
            print("No candidate qualified; nothing frozen and evaluation cannot proceed.")

    print(f"Wrote {name}_results.json and {name}_comparison.csv to {d}")
    return 0


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
