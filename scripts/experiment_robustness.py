"""Fixed synthetic stress protocol. No stress result selects a deployment model."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.robustness import SCENARIOS, perturb, score_frozen, align_to_expected
from headway.pipeline import HealthPipeline
from headway.onboarding import onboard
from headway.models import detectors as D
from headway.evaluate.metrics import evaluate_replay
from headway.rul import ConformalRUL
from headway.validation import true_rul
from headway.synth.doors import SynthConfig, generate

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260911, 20260912)


def clean(v):
    if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)) and not np.isfinite(v): return None
    if isinstance(v, np.generic): return v.item()
    return v


def matched_rul(left, right, episodes):
    keys = ["asset_id", "day", "available_at"]
    l = left[keys + ["rul_point"]].copy()
    l["truth"] = true_rul(left, episodes)
    pair = l.merge(right[keys + ["rul_point"]], on=keys, suffixes=("_base", "_onboard"), validate="one_to_one")
    ok = np.isfinite(pair[["rul_point_base", "rul_point_onboard", "truth"]]).all(axis=1)
    p = pair[ok]
    return {"matched_episode_rows": len(p),
        "base_mae_days": float((p.rul_point_base - p.truth).abs().mean()) if len(p) else None,
        "onboard_mae_days": float((p.rul_point_onboard - p.truth).abs().mean()) if len(p) else None}


def run(cfg, out):
    cycles, eps = generate(cfg)
    start = cycles.ts.min().floor("D")
    ref_end, enroll_end = start + pd.Timedelta(days=30), start + pd.Timedelta(days=21)
    cutoff = start + pd.Timedelta(days=84)
    # Fixed identity, not chosen by fault outcome. Entire train omitted at fit.
    held = sorted(cycles.train_id.unique())[0]
    target = cycles[cycles.train_id == held]
    training = cycles[cycles.train_id != held]
    pipe = HealthPipeline().fit(training[training.ts < ref_end])
    tr = pipe.transform(training)
    names = [f"{s}_hx" for s in pipe.levels]
    zoo = {"raw": D.RawThreshold("current_integral_as"),
           "directional": D.Passthrough("health_index"),
           "mahalanobis": D.MahalanobisDetector(needs=names),
           "isolation_forest": D.IsolationForestDetector(needs=names, n_estimators=100)}
    fit = tr[tr.day < ref_end]
    for model in zoo.values():
        needs = model.needs if hasattr(model, "needs") else [model.column]
        model.fit(fit[np.isfinite(fit[needs]).all(axis=1)])
    train_scores = score_frozen(zoo, tr)
    calibration = train_scores[(train_scores.available_at > ref_end) & (train_scores.available_at <= cutoff)]
    thresholds = {name: float(calibration[name].quantile(.99)) for name in zoo}
    expected = cycles[["asset_id", "train_id", "ts"]].assign(day=cycles.ts.dt.floor("D"))
    expected = expected[["asset_id", "train_id", "day"]].drop_duplicates()
    expected["available_at"] = expected.day + pd.Timedelta(days=1)
    expected = expected[expected.available_at > cutoff]
    future_eps = eps[eps.fault_ts > cutoff]
    rows = []
    # Freeze membership and magnitudes before results: first three train IDs.
    affected = sorted(cycles.loc[cycles.train_id.isin(sorted(cycles.train_id.unique())[:3]), "asset_id"].unique())
    baseline_pred = None
    for scenario in SCENARIOS:
        changed = perturb(cycles, scenario, after=cutoff, assets=affected, seed=cfg.seed)
        daily = pipe.transform(changed)
        if scenario == "clean": baseline_pred = daily
        scored = score_frozen(zoo, daily)
        scored = scored[scored.available_at > cutoff]
        aligned = align_to_expected(scored, expected, zoo)
        for population, ids in (("whole_fleet", set(expected.asset_id)), ("affected_assets", set(affected))):
            frame = aligned[aligned.asset_id.isin(ids)]
            episodes = future_eps[future_eps.asset_id.isin(ids)]
            for name in zoo:
                stats = evaluate_replay(frame, name, episodes, thresholds[name], daily_budget=2, cooldown_days=3).as_row()
                rows.append({"seed": cfg.seed, "scenario": scenario, "population": population,
                    **stats, "expected_asset_days": len(frame), "score_fraction": float(np.isfinite(frame[name]).mean()),
                    "threshold": thresholds[name]})
        print(f"seed {cfg.seed}: {scenario} done", flush=True)
    # Retrospective held-train RUL comparison, distinct from forward stress replay.
    # Other trains' future labels are used here; target labels are evaluation only.
    adapted = onboard(pipe, target[target.ts < enroll_end], experimental=True,as_of=enroll_end)
    td = adapted.transform(target)
    bd = baseline_pred[baseline_pred.asset_id.isin(target.asset_id)]
    m = ConformalRUL(projection="loglinear", mode="additive").fit(tr, eps[eps.train_id != held], evaluate=False)
    matched = matched_rul(m.predict(bd[bd.available_at > ref_end]),
                          m.predict(td[td.available_at > ref_end]), eps[eps.train_id == held])
    # Same post-reference signal for BOTH baselines. A fixed +0.6 A.s offset
    # exists from the beginning; contaminated onboarding can absorb it as normal.
    shifted = target.copy()
    shifted["current_integral_as"] += .6
    contaminated = onboard(pipe, shifted[shifted.ts < enroll_end], experimental=True,as_of=enroll_end)
    clean_ref_view, contaminated_view = adapted.transform(shifted), contaminated.transform(shifted)
    key = "current_integral_as_hx"
    mask = clean_ref_view.available_at > ref_end
    contamination = {"accepted_assets": sorted(contaminated.ready_at), "rejected": contaminated.rejected,
        "offset_as": .6, "matched_post_reference_days": int(mask.sum()),
        "median_hx_with_clean_reference": float(clean_ref_view.loc[mask, key].median()),
        "median_hx_with_offset_reference": float(contaminated_view.loc[mask, key].median()),
        "interpretation": "A stable channel offset can be absorbed by onboarding; this does not distinguish wear from sensor bias."}
    report = {"config": asdict(cfg), "held_train": held, "affected_assets": affected,
              "thresholds": thresholds, "detector_results": rows,
              "matched_held_train_rul": matched, "reference_contamination": contamination}
    (out / f"seed_{cfg.seed}.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    out = ROOT / "data/robustness_experiment"
    out.mkdir(exist_ok=True)
    protocol = {"seeds": SEEDS, "scenarios": SCENARIOS, "fit_days": 30, "cutoff_day": 84,
        "threshold_quantile": .99, "daily_alert_budget": 2, "cooldown_days": 3,
        "missing_channel_fraction": .3, "offset_as": .6, "context_input_shift_c": 20,
        "outage": "remove every third post-cutoff telemetry day from affected assets",
        "affected": "first three train identities; first train held out of all fitting",
        "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_module_hash": hashlib.sha256((ROOT / "headway/robustness.py").read_bytes()).hexdigest(),
        "limitations": ["Fixed small synthetic fleets; not external validation or a model-selection exercise.",
            "Fault labels unchanged during sensor/context perturbations; unlabelled alerts are not necessarily pointless inspections.",
            "Population replays have separate alert budgets; their counts are not additive.",
            "RUL paired comparison is retrospective held-train evaluation, not chronological testing."]}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    reports = [run(SynthConfig(n_trains=12, doors_per_train=2, n_days=120, cycles_per_day=35,
                              n_episodes=6, seed=seed), out) for seed in SEEDS]
    pd.DataFrame([r for report in reports for r in report["detector_results"]]).to_csv(out / "comparison.csv", index=False)
    print("Robustness reports written.", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
