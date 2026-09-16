"""Reviewed-data PS3 selection on temporary files.

A small generated door fleet is written to a temporary workspace as canonical
Parquet with a valid provenance sidecar, and episodes are placed by hand so that
every boundary case is deliberate: complete validation and test episodes, one
episode crossing the calibration boundary and one crossing test_end. Labels are
written as two files - development (confirmed by validation_end) and test.

The configuration uses paths RELATIVE to its own file, so a workspace can be
copied wholesale and keep an identical configuration hash - which is what lets
the tamper tests change exactly one thing at a time.

Nothing here measures detector performance; the fleet is tiny and synthetic.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from headway import ps3_run as P  # noqa: E402
from headway.detector_selection import SelectionPolicy, select_detector  # noqa: E402
from headway.models import detectors as D  # noqa: E402
from headway.normalise import AssetBaseline  # noqa: E402
from headway.pipeline import HealthPipeline  # noqa: E402
from headway.synth.doors import SynthConfig, generate  # noqa: E402
import ps3_readiness as R  # noqa: E402

SCRIPT = ROOT / "scripts" / "select_ps3_detector.py"
BOUNDS = {"reference_start": "2026-01-01", "reference_end": "2026-02-01",
          "calibration_end": "2026-02-20", "validation_end": "2026-03-15",
          "test_end": "2026-04-10"}
PERMISSIVE = {"threshold_quantile": 0.99, "daily_budget": 2, "cooldown_days": 3.0,
              "minimum_fault_groups": 1, "minimum_recall": 0.0,
              "maximum_false_alerts_per_asset_month": 1e6, "minimum_score_fraction": 0.0}
STRICT = dict(PERMISSIVE, minimum_fault_groups=99, minimum_recall=1.0)
DEV, TEST = "data/dev_episodes.csv", "data/test_episodes.csv"
SIGNALS = ("current_integral_as", "peak_current_a", "mean_current_a", "cycle_duration_s")


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------

def fleet():
    cycles, _ = generate(SynthConfig(n_trains=6, doors_per_train=2, n_days=100,
                                     cycles_per_day=12, n_episodes=2, start="2026-01-01", seed=11))
    cycles = cycles.drop_duplicates(["asset_id", "ts"]).reset_index(drop=True)
    cycles["context_supported"] = True
    return cycles


def episodes_for(cycles):
    doors = cycles.groupby("train_id").asset_id.unique().sort_index()
    trains = list(doors.index)
    rows = []
    for t in trains[:3]:                     # complete inside validation
        rows.append((doors[t][0], t, "2026-02-24", "2026-03-06"))
    for t in trains[3:6]:                    # complete inside test
        rows.append((doors[t][0], t, "2026-03-20", "2026-04-02"))
    rows.append((doors[trains[0]][1], trains[0], "2026-02-15", "2026-02-27"))  # crosses calibration_end
    rows.append((doors[trains[1]][1], trains[1], "2026-04-05", "2026-04-20"))  # crosses test_end
    return pd.DataFrame(rows, columns=["asset_id", "train_id", "onset_ts", "fault_ts"])


def split(eps):
    fault = pd.to_datetime(eps.fault_ts)
    end = pd.Timestamp(BOUNDS["validation_end"])
    return eps[fault <= end], eps[fault > end]


def sidecar_for(parquet, **over):
    body = {"input": {"path": "vendor.csv", "sha256": "a" * 64, "rows_read": None,
                      "complete_input": True},
            "output": {"path": parquet.name, "sha256": R.sha256_file(parquet)},
            "mapping": {"canonical_sha256": "b" * 64}, "attestations": {},
            "context_supported": True, "unit_conversions": None}
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(body.get(key), dict):
            body[key] = dict(body[key], **value)
        else:
            body[key] = value
    parquet.with_suffix(".provenance.json").write_text(json.dumps(body), encoding="utf-8")


def config_body(**over):
    body = {
        "run_kind": "external", "subsystem": "door",
        "telemetry": "data/cycles.parquet", "development_episodes": DEV, "test_episodes": TEST,
        "output_dir": "out", "timezone": "Asia/Singapore", "boundaries": dict(BOUNDS),
        "expected_grid": "from_first_observation",
        "reference_eligibility": {"assets": "all", "declared_by": "test reviewer",
                                  "basis": "fixture: inspection record for every door",
                                  "evidence_id": "FIXTURE-INSPECTION-1"},
        "attestations": {"units_verified": True, "identities_verified": True,
                         "fault_labels_verified": True, "reference_eligibility_verified": True,
                         "declared_by": "test reviewer"},
        "label_semantics": {"onset_ts": "fixture onset", "fault_ts": "fixture fault"},
        "data_exposure": {"test_period_previously_inspected": True, "declared_by": "test"},
        "policy": dict(PERMISSIVE), "smooth_days": 3, "min_reference_days_per_asset": 14,
    }
    body.update(over)
    return body


def workspace(root, *, cycles=None, episodes=None, dev=None, test=None, sidecar=True, **config_over):
    root = Path(root)
    (root / "data").mkdir(parents=True, exist_ok=True)
    cycles = fleet() if cycles is None else cycles
    parquet = root / "data" / "cycles.parquet"
    cycles.to_parquet(parquet, index=False)
    if sidecar:
        sidecar_for(parquet)
    split_dev, split_test = split(episodes_for(cycles) if episodes is None else episodes)
    (split_dev if dev is None else dev).to_csv(root / DEV, index=False)
    (split_test if test is None else test).to_csv(root / TEST, index=False)
    config = root / "config.json"
    config.write_text(json.dumps(config_body(**config_over), indent=2), encoding="utf-8")
    return config


def write_config(root, **over):
    config = Path(root) / "config.json"
    config.write_text(json.dumps(config_body(**over), indent=2), encoding="utf-8")
    return config


def select(config, run_id="base"):
    return P.stage_select(P.load_config(config), run_id)


def refused(fn, fragment):
    with pytest.raises(SystemExit) as info:
        fn()
    assert fragment in str(info.value.code), str(info.value.code)
    return str(info.value.code)


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """One successful select stage, reused by copying the whole workspace."""
    root = tmp_path_factory.mktemp("frozen_ws")
    config = workspace(root)
    assert select(config) == 0
    return root


def clone(frozen, tmp_path):
    target = tmp_path / "ws"
    shutil.copytree(frozen, target)
    return target, target / "config.json", target / "out" / "runs" / "base"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_select_freezes_then_test_evaluates_once(frozen, tmp_path):
    root, config, run = clone(frozen, tmp_path)
    for name in ("config.json", "inputs.json", "preflight.json", "scores.parquet",
                 "selection.json", "frozen.json"):
        assert (run / name).exists(), name
    assert not (run / "test.json").exists(), "selection must not write a test"

    assert P.stage_test(P.load_config(config), "base") == 0
    test = read(run / "test.json")
    assert test["stage"] == "test"
    assert "NOT an untouched hold-out" in test["exposure_statement"]
    if test["status"] == "evaluated":
        assert test["test_labels_sha256"] == P.sha256_file(root / TEST)
    refused(lambda: P.stage_test(P.load_config(config), "base"), "a final test is run once")


def test_configuration_and_hashes_are_recorded_before_fitting(tmp_path, monkeypatch):
    config = workspace(tmp_path)

    def no_fitting(*a, **kw):
        raise RuntimeError("fitting reached")

    monkeypatch.setattr(P.HealthPipeline, "fit", no_fitting)
    with pytest.raises(RuntimeError, match="fitting reached"):
        select(config)
    run = tmp_path / "out" / "runs" / "base"
    recorded = read(run / "config.json")
    inputs = read(run / "inputs.json")
    assert recorded["boundaries"]["reference_end"] == "2026-02-01T00:00:00"
    assert "AVAILABLE" in recorded["timezone_interpretation"]
    assert recorded["alert_budget"] == {"daily_budget": 2, "cooldown_days": 3.0}
    assert recorded["qualification_policy"] == PERMISSIVE
    assert recorded["automatic_promotion"] is False
    assert recorded["expected_grid"]["rule"] == "from_first_observation"
    assert set(inputs["inputs"]) == {"telemetry_sha256", "development_episodes_sha256",
                                     "provenance_sidecar_sha256"}
    assert all(len(v) == 64 for v in inputs["inputs"].values())
    assert inputs["test_episodes"]["opened_by_selection_stage"] is False
    assert set(inputs["code"]) == set(P.CODE_FILES)
    assert read(run / "preflight.json")["status"] == "passed"
    assert not (run / "selection.json").exists()


def test_the_selection_report_is_honest_about_what_was_counted(frozen):
    report = read(frozen / "out" / "runs" / "base" / "selection.json")
    eps = report["episodes"]
    assert eps["complete_in_validation"] == 3
    assert eps["excluded_crossing_calibration_boundary"] == 1
    assert len(eps["excluded_crossing_assets"]) == 1
    assert "test_episodes was not opened" in report["labels_used"]
    monitoring = report["validation_monitoring"]
    for cand in report["selection"]["candidates"]:
        if "score_fraction" in cand:
            # The gate and the monitoring report use one denominator.
            assert cand["score_fraction"] == pytest.approx(monitoring["availability"][cand["name"]])
            assert cand["observation_rows"] == monitoring["expected_asset_days"]
        if "alarms" in cand:
            assert cand["unmatched_alerts"] == cand["alarms"] - round(cand["precision"] * cand["alarms"])
    assert report["selection"]["calibration_start"] == "2026-02-01 00:00:00"


def test_the_freeze_note_states_what_selection_actually_read(frozen):
    body = read(frozen / "out" / "runs" / "base" / "frozen.json")
    assert "opened only the development label file" in body["note"]
    assert "not opened, read or hashed" in body["note"]
    assert "provenance_sidecar_sha256" in body["inputs"]
    assert not any("test" in key for key in body["inputs"])


def test_the_test_report_counts_crossings_at_both_ends(frozen, tmp_path):
    root, config, run = clone(frozen, tmp_path)
    assert P.stage_test(P.load_config(config), "base") == 0
    test = read(run / "test.json")
    if test["status"] == "not_evaluated":
        pytest.skip("no candidate qualified on this fixture")
    assert test["episodes"]["excluded_crossing_test_end"] == 1
    assert test["episodes"]["complete_in_test"] == 3
    assert test["result"]["score_fraction"] == pytest.approx(
        next(iter(test["test_monitoring"]["availability"].values())))


# ---------------------------------------------------------------------------
# item 1: expected asset-days in qualification and final metrics
# ---------------------------------------------------------------------------

def two_of_five():
    """One asset; 10 scored calibration days; 5 expected validation days, 2 scored."""
    bounds = {"reference_start": pd.Timestamp("2025-12-20"), "reference_end": pd.Timestamp("2026-01-01"),
              "calibration_end": pd.Timestamp("2026-01-11"), "validation_end": pd.Timestamp("2026-01-16")}
    observed = list(pd.date_range("2025-12-20", "2026-01-10", freq="D")) + \
        [pd.Timestamp("2026-01-11"), pd.Timestamp("2026-01-13")]
    cycles = pd.DataFrame({"asset_id": "A", "ts": [d + pd.Timedelta(hours=8) for d in observed]})
    scored = [d for d in observed if d >= pd.Timestamp("2026-01-01")]
    scores = pd.DataFrame({"asset_id": "A", "day": scored, "det": np.linspace(0, 1, len(scored))})
    scores["available_at"] = scores.day + pd.Timedelta(days=1)
    episodes = pd.DataFrame({"asset_id": ["A"], "train_id": ["T"],
                             "onset_ts": [pd.Timestamp("2026-01-12")],
                             "fault_ts": [pd.Timestamp("2026-01-15")]})
    return bounds, cycles, scores, episodes


def test_two_scored_days_of_five_expected_is_availability_0_4_and_fails_a_0_9_gate():
    bounds, cycles, scores, episodes = two_of_five()
    grid = P.expected_grid(cycles, bounds["reference_end"], bounds["validation_end"],
                           "from_first_observation", bounds["reference_start"])
    aligned = P.align_scores(scores, grid, ["det"])
    validation = aligned[aligned.available_at > bounds["calibration_end"]]
    assert len(validation) == 5 and int(validation.det.notna().sum()) == 2

    policy = dict(PERMISSIVE, minimum_score_fraction=0.9)
    cand = P.qualify(aligned, episodes, ["det"], bounds, policy)["candidates"][0]
    assert cand["score_fraction"] == pytest.approx(0.4)
    assert "too many observations without a score" in cand["reasons"]
    assert not cand["qualified"]
    report = P.availability(aligned, ["det"], bounds["calibration_end"], bounds["validation_end"], set())
    assert report == {"expected_asset_days": 5, "availability": {"det": pytest.approx(0.4)}}

    # The same gate at 0.4 no longer cites availability: the gate reads the grid.
    lenient = P.qualify(aligned, episodes, ["det"], bounds, dict(policy, minimum_score_fraction=0.4))
    assert "too many observations without a score" not in lenient["candidates"][0]["reasons"]

    # The defect being corrected: on observed rows only, the same data looked 100% available.
    observed_only = select_detector(scores, episodes, ["det"], calibration_start=bounds["reference_end"],
                                    calibration_end=bounds["calibration_end"],
                                    validation_end=bounds["validation_end"],
                                    policy=SelectionPolicy(**policy))
    assert observed_only["candidates"][0]["score_fraction"] == 1.0


def test_absent_days_in_the_final_test_count_against_availability():
    bounds, cycles, scores, episodes = two_of_five()
    grid = P.expected_grid(cycles, bounds["reference_end"], bounds["validation_end"],
                           "from_first_observation", bounds["reference_start"])
    aligned = P.align_scores(scores, grid, ["det"])
    selection = {"selected_model": "det", "selected_threshold": 0.5,
                 "validation_end": str(bounds["calibration_end"]), "policy": PERMISSIVE}
    result = P.assess_selected(selection, aligned, episodes)
    assert result["score_fraction"] == pytest.approx(0.4)
    assert result["observation_rows"] == 5


def test_the_grid_rule_is_declared_not_inferred():
    days = pd.date_range("2026-01-01", periods=10, freq="D")
    cycles = pd.DataFrame({"asset_id": ["A"] * 10 + ["B"] * 5,
                           "ts": list(days + pd.Timedelta(hours=8)) + list(days[5:] + pd.Timedelta(hours=8))})
    lo, hi = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-11")
    late = P.expected_grid(cycles, lo, hi, "from_first_observation", lo)
    whole = P.expected_grid(cycles, lo, hi, "from_reference_start", lo)
    assert (late.asset_id == "B").sum() == 5 and (whole.asset_id == "B").sum() == 10
    assert len(whole) == 20


def test_alignment_refuses_duplicate_asset_days_and_drops_off_grid_rows():
    bounds, cycles, scores, _ = two_of_five()
    grid = P.expected_grid(cycles, bounds["reference_end"], bounds["calibration_end"],
                           "from_first_observation", bounds["reference_start"])
    aligned = P.align_scores(scores, grid, ["det"])
    assert aligned.available_at.max() <= bounds["calibration_end"]
    with pytest.raises(ValueError, match="duplicate"):
        P.align_scores(pd.concat([scores, scores.iloc[:1]]), grid, ["det"])


def test_a_missing_telemetry_day_is_carried_as_unavailable_end_to_end(frozen, tmp_path):
    cycles = fleet()
    asset = sorted(cycles.asset_id.unique())[-1]
    gone = (cycles.asset_id == asset) & (cycles.ts >= "2026-02-25") & (cycles.ts < "2026-03-05")
    config = workspace(tmp_path, cycles=cycles[~gone].reset_index(drop=True))
    assert select(config) == 0
    scores = pd.read_parquet(tmp_path / "out" / "runs" / "base" / "scores.parquet")
    missing = scores[(scores.asset_id == asset) & (scores.day >= "2026-02-25") & (scores.day < "2026-03-05")]
    assert len(missing) == 8 and missing.drop(columns=P.KEYS).isna().all().all()
    clean = read(frozen / "out" / "runs" / "base" / "selection.json")["validation_monitoring"]
    thinner = read(tmp_path / "out" / "runs" / "base" / "selection.json")["validation_monitoring"]
    assert thinner["expected_asset_days"] == clean["expected_asset_days"]
    assert thinner["availability"]["raw"] < clean["availability"]["raw"]


# ---------------------------------------------------------------------------
# item 2: validity before smoothing, explicit reference quality
# ---------------------------------------------------------------------------

class Spy(D.Detector):
    """Records the frame it was fitted on."""

    def __init__(self):
        self.column, self.seen = "value", None

    def fit(self, reference):
        self.seen = reference.copy()
        return self

    def score(self, daily):
        return daily["value"].to_numpy(float)


def daily_frame():
    days = pd.date_range("2026-01-01", periods=60, freq="D")
    rows = [{"asset_id": a, "day": d, "available_at": d + pd.Timedelta(days=1), "value": 1.0}
            for a in ("A", "B", "C") for d in days]
    return pd.DataFrame(rows)


def ewma_history(spike):
    rng = np.random.default_rng(3)
    frame = daily_frame()
    frame["value"] = rng.normal(0, 1, len(frame))
    frame["valid"] = True
    bad = (frame.asset_id == "A") & (frame.day == pd.Timestamp("2026-02-10"))
    frame.loc[bad, "valid"] = False
    frame.loc[bad, "value"] = spike
    frame["data_quality_ok"] = frame.valid
    return frame, bad


def test_an_invalid_historical_reading_cannot_move_later_scores():
    kw = dict(smooth_days=3, reference_start="2025-12-31", reference_end="2026-02-01",
              score_valid_column="valid", reference_valid_column="valid")
    scored = {}
    for spike in (0.0, 500.0):
        frame, bad = ewma_history(spike)
        scored[spike] = D.run({"ewma": D.EWMAChart("value")}, frame, **kw)
    assert scored[0.0].loc[bad, "ewma"].isna().all()
    pd.testing.assert_series_equal(scored[0.0]["ewma"], scored[500.0]["ewma"])


def test_the_legacy_masking_order_is_preserved_and_does_leak():
    """Without a validity column the historical path masks after scoring: kept, labelled, leaky."""
    kw = dict(smooth_days=3, reference_start="2025-12-31", reference_end="2026-02-01")
    later = {}
    for spike in (0.0, 500.0):
        frame, bad = ewma_history(spike)
        out = D.run({"ewma": D.EWMAChart("value")}, frame, **kw)
        later[spike] = out.loc[(out.asset_id == "A") & (out.day > pd.Timestamp("2026-02-10")), "ewma"]
    assert not np.allclose(later[0.0], later[500.0], equal_nan=True)


def test_validity_columns_need_an_explicit_reference():
    frame, _ = ewma_history(0.0)
    with pytest.raises(ValueError, match="validity columns require an explicit reference"):
        D.run({"s": Spy()}, frame, score_valid_column="valid")


def test_reference_quality_is_explicit_and_not_the_readiness_flag():
    spy = Spy()
    frame = daily_frame()
    frame["ok"] = frame.day.dt.day % 2 == 0
    D.run({"s": spy}, frame, reference_start="2025-12-31", reference_end="2026-02-01",
          reference_valid_column="ok")
    assert spy.seen.ok.all() and len(spy.seen) == int(frame[frame.available_at <= "2026-02-01"].ok.sum())

    row = pd.DataFrame({"a_completeness": [0.95, 0.95, 0.5], "n_cycles": [12, 3, 12],
                        "health_inputs_complete": [True, True, True],
                        "context_supported": [True, True, True],
                        # Pre-fit rows: the pipeline's readiness flags are False by construction.
                        "data_quality_ok": [False, False, False], "quality_ready": [False, False, False]})
    assert P.reference_quality(row).tolist() == [True, False, False]


def degrade(cycles, mask, *, spike=1.0):
    """Blank a fifth of the peak-current readings on masked cycles: completeness 0.8 < 0.9."""
    cycles = cycles.copy()
    idx = cycles.index[mask]
    cycles.loc[idx[::5], "peak_current_a"] = np.nan
    cycles.loc[idx, "current_integral_as"] *= spike
    return cycles


def test_invalid_history_cannot_influence_later_selection_scores(tmp_path):
    base = fleet()
    asset = sorted(base.asset_id.unique())[2]
    day = (base.asset_id == asset) & (base.ts >= "2026-02-10") & (base.ts < "2026-02-11")
    scores, selections = {}, {}
    for spike in (1.0, 50.0):
        root = tmp_path / f"spike{int(spike)}"
        assert select(workspace(root, cycles=degrade(base, day, spike=spike))) == 0
        run = root / "out" / "runs" / "base"
        scores[spike] = pd.read_parquet(run / "scores.parquet")
        selections[spike] = read(run / "selection.json")["selection"]
    row = (scores[1.0].asset_id == asset) & (scores[1.0].day == pd.Timestamp("2026-02-10"))
    assert row.sum() == 1 and scores[1.0].loc[row].drop(columns=P.KEYS).isna().all().all()
    pd.testing.assert_frame_equal(scores[1.0], scores[50.0])
    assert selections[1.0] == selections[50.0]


def test_insufficient_valid_reference_is_refused(tmp_path):
    cycles = fleet()
    asset = sorted(cycles.asset_id.unique())[0]
    # 31 declared reference days; 20 of them fail completeness, leaving 11 < 14.
    poor = (cycles.asset_id == asset) & (cycles.ts >= "2026-01-01") & (cycles.ts < "2026-01-21")
    config = workspace(tmp_path, cycles=degrade(cycles, poor))
    message = refused(lambda: select(config), "VALID reference days")
    assert asset in message
    preflight = read(tmp_path / "out" / "runs" / "base" / "preflight.json")
    assert preflight["reference"]["excluded_for_quality"] >= 20
    assert not (tmp_path / "out" / "runs" / "base" / "scores.parquet").exists()


def test_poor_reference_days_are_excluded_from_fitting(frozen, tmp_path):
    cycles = fleet()
    asset = sorted(cycles.asset_id.unique())[-1]
    poor = (cycles.asset_id == asset) & (cycles.ts >= "2026-01-10") & (cycles.ts < "2026-01-15")
    assert select(workspace(tmp_path, cycles=degrade(cycles, poor))) == 0
    used = read(tmp_path / "out" / "runs" / "base" / "selection.json")["reference_used"]
    clean = read(frozen / "out" / "runs" / "base" / "selection.json")["reference_used"]
    # The clean fleet already has short (< MIN_CYCLES) reference days; five more are added.
    assert (used["preprocessing"]["asset_days_excluded_for_quality"]
            == clean["preprocessing"]["asset_days_excluded_for_quality"] + 5)
    assert used["detectors"]["asset_days"] <= clean["detectors"]["asset_days"] - 5


# ---------------------------------------------------------------------------
# item 3: the test-label boundary
# ---------------------------------------------------------------------------

OPENED = {"watch": None, "hits": []}


def _audit(event, args):
    watch = OPENED["watch"]
    if watch is None or event != "open" or not args:
        return
    target = args[0]
    if isinstance(target, (str, bytes, os.PathLike)):
        path = os.fsdecode(target)
        if os.path.normcase(os.path.abspath(path)) == watch:
            OPENED["hits"].append(path)


sys.addaudithook(_audit)


def watching(path):
    OPENED["watch"], OPENED["hits"] = os.path.normcase(os.path.abspath(path)), []


def test_selection_never_opens_the_test_label_input(tmp_path):
    config = workspace(tmp_path)
    test_labels = tmp_path / TEST
    watching(tmp_path / DEV)                       # the hook itself sees label reads
    try:
        P.read_episodes(tmp_path / DEV, scope="development",
                        validation_end=pd.Timestamp(BOUNDS["validation_end"]))
        assert OPENED["hits"], "the audit hook does not observe pandas opening a label file"
        watching(test_labels)
        assert select(config) == 0
        assert OPENED["hits"] == [], f"selection opened the test labels: {OPENED['hits']}"
        assert P.stage_test(P.load_config(config), "base") == 0
        status = read(tmp_path / "out" / "runs" / "base" / "test.json")["status"]
        assert bool(OPENED["hits"]) == (status == "evaluated")
    finally:
        OPENED["watch"] = None


def test_selection_does_not_need_the_test_labels_to_exist(tmp_path):
    config = workspace(tmp_path)
    (tmp_path / TEST).unlink()
    assert select(config) == 0
    assert read(tmp_path / "out" / "runs" / "base" / "preflight.json")["test_labels"]["opened"] is False


def test_development_labels_from_the_test_period_are_refused(tmp_path):
    cycles = fleet()
    eps = episodes_for(cycles)
    config = workspace(tmp_path, cycles=cycles, dev=eps, test=eps.iloc[:0])
    refused(lambda: select(config), "confirmed after validation_end")


def test_test_labels_from_the_development_period_are_refused(frozen, tmp_path):
    root, config, run = clone(frozen, tmp_path)
    if read(run / "selection.json")["selection"]["selected_model"] is None:
        pytest.skip("no candidate qualified on this fixture")
    episodes_for(fleet()).to_csv(root / TEST, index=False)
    refused(lambda: P.stage_test(P.load_config(config), "base"), "on or before validation_end")
    assert not (run / "test.json").exists()


def test_one_file_for_both_label_scopes_is_refused(tmp_path):
    refused(lambda: P.load_config(write_config(tmp_path, test_episodes=DEV)), "must be different files")


def test_a_single_combined_episodes_key_is_refused(tmp_path):
    body = config_body()
    body["episodes"] = body.pop("development_episodes")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    refused(lambda: P.load_config(path), "supply development_episodes and test_episodes")


def test_external_eligibility_needs_independent_evidence(tmp_path):
    elig = {"assets": "all", "declared_by": "reviewer", "basis": "no labels on these doors"}
    refused(lambda: P.load_config(write_config(tmp_path, reference_eligibility=elig)),
            "reference_eligibility.evidence_id")


def test_test_labels_that_reveal_reference_contamination_are_disclosed(frozen, tmp_path):
    root, config, run = clone(frozen, tmp_path)
    if read(run / "selection.json")["selection"]["selected_model"] is None:
        pytest.skip("no candidate qualified on this fixture")
    test = pd.read_csv(root / TEST)
    extra = test.iloc[:1].assign(onset_ts="2026-01-20", fault_ts="2026-03-30")
    pd.concat([test, extra]).to_csv(root / TEST, index=False)
    assert P.stage_test(P.load_config(config), "base") == 0
    body = read(run / "test.json")
    assert body["reference_contamination_found_in_test_labels"] == [extra.asset_id.iloc[0]]
    assert "reference was not clean" in body["reference_contamination_note"]


# ---------------------------------------------------------------------------
# split boundaries and configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bounds,fragment", [
    (dict(BOUNDS, calibration_end="2026-01-15"), "strictly ordered"),
    (dict(BOUNDS, validation_end=BOUNDS["calibration_end"]), "strictly ordered"),
    (dict(BOUNDS, reference_end="2026-02-01 06:00"), "must be a midnight"),
    (dict(BOUNDS, test_end="not a date"), "not a readable date"),
    (dict(BOUNDS, reference_end="2026-01-05"), "shorter than min_reference_days"),
    ({k: v for k, v in BOUNDS.items() if k != "test_end"}, "must declare exactly"),
])
def test_invalid_boundaries_are_refused(tmp_path, bounds, fragment):
    config = write_config(tmp_path, boundaries=bounds)
    refused(lambda: P.load_config(config), fragment)


def test_an_offset_boundary_is_converted_not_misread(tmp_path):
    config = write_config(tmp_path, boundaries=dict(BOUNDS, reference_end="2026-01-31T16:00:00+00:00"))
    assert P.load_config(config)["boundaries"]["reference_end"] == pd.Timestamp("2026-02-01")


def test_a_different_timezone_is_refused(tmp_path):
    refused(lambda: P.load_config(write_config(tmp_path, timezone="UTC")), "timezone must be")


def test_an_undeclared_grid_rule_is_refused(tmp_path):
    refused(lambda: P.load_config(write_config(tmp_path, expected_grid="observed_rows")),
            "expected_grid must be one of")


def test_the_policy_is_never_defaulted(tmp_path):
    policy = {k: v for k, v in PERMISSIVE.items() if k != "minimum_recall"}
    refused(lambda: P.load_config(write_config(tmp_path, policy=policy)), "not defaulted")


def test_an_invalid_policy_is_refused(tmp_path):
    refused(lambda: P.load_config(write_config(tmp_path, policy=dict(PERMISSIVE, threshold_quantile=1.5))),
            "invalid qualification policy")


def test_bogie_evaluation_is_refused(tmp_path):
    message = refused(lambda: P.load_config(write_config(tmp_path, subsystem="bogie")),
                      "not supported by this runner")
    assert "must not be transferred" in message


def test_unknown_configuration_keys_are_refused(tmp_path):
    body = config_body()
    body["split_fraction"] = 0.7
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    refused(lambda: P.load_config(path), "unknown configuration key")


# ---------------------------------------------------------------------------
# reference leakage
# ---------------------------------------------------------------------------

def test_every_detector_fits_only_on_the_declared_reference():
    spies = {"one": Spy(), "two": Spy()}
    D.run(spies, daily_frame(), reference_start="2026-01-10", reference_end="2026-02-01",
          eligible_assets=["A", "B"])
    for spy in spies.values():
        seen = spy.seen
        assert seen.available_at.min() > pd.Timestamp("2026-01-10")
        assert seen.available_at.max() <= pd.Timestamp("2026-02-01")
        assert set(seen.asset_id) == {"A", "B"}
    assert spies["one"].seen.equals(spies["two"].seen), "detectors saw different references"


def test_the_historical_reference_default_is_unchanged():
    spy = Spy()
    D.run({"s": spy}, daily_frame(), reference_days=30)
    assert spy.seen.day.max() == pd.Timestamp("2026-01-30")
    assert set(spy.seen.asset_id) == {"A", "B", "C"}


@pytest.mark.parametrize("kw,fragment", [
    ({"reference_start": "2026-01-10"}, "needs both"),
    ({"eligible_assets": ["A"]}, "requires an explicit reference"),
    ({"reference_start": "2026-02-01", "reference_end": "2026-01-10"}, "must precede"),
])
def test_ambiguous_reference_arguments_are_refused(kw, fragment):
    with pytest.raises(ValueError, match=fragment):
        D.run({"s": Spy()}, daily_frame(), **kw)


def test_an_explicit_reference_uses_every_reference_day_per_asset():
    """The hidden 21-day per-asset truncation is gone when the caller bounds the frame."""
    days = pd.date_range("2026-01-01", periods=51, freq="D")
    frame = pd.DataFrame({"asset_id": "A", "day": days,
                          "residual": np.r_[np.zeros(21), np.full(30, 10.0)]})
    frame.loc[::2, "residual"] += 0.5
    truncated = AssetBaseline(value="residual").fit(frame)
    whole = AssetBaseline(value="residual", reference_days=None).fit(frame)
    assert truncated.loc_["A"] < 1.0
    assert whole.loc_["A"] > truncated.loc_["A"]


def test_preprocessing_cannot_be_declared_available_before_its_reference_ends():
    cycles = fleet()
    reference = cycles[cycles.ts < "2026-02-01"]
    with pytest.raises(ValueError, match="precedes the last reference aggregate"):
        HealthPipeline(baseline_reference_days=None).fit(reference, fitted_at="2026-01-15")


def test_later_telemetry_cannot_change_fitting_or_selection(tmp_path, frozen):
    """Spike every signal after validation_end: selection must be byte-identical."""
    cycles = fleet()
    late = cycles.ts >= pd.Timestamp(BOUNDS["validation_end"])
    for column in SIGNALS:
        cycles.loc[late, column] *= 50.0
    assert select(workspace(tmp_path, cycles=cycles)) == 0

    clean = read(frozen / "out" / "runs" / "base" / "selection.json")
    spiked = read(tmp_path / "out" / "runs" / "base" / "selection.json")
    assert spiked["selection"] == clean["selection"]
    assert spiked["validation_monitoring"] == clean["validation_monitoring"]
    assert spiked["reference_used"] == clean["reference_used"]


def test_non_eligible_assets_never_enter_the_reference(tmp_path):
    cycles = fleet()
    assets = sorted(cycles.asset_id.unique())
    eligible = assets[: len(assets) // 2]
    config = workspace(tmp_path, cycles=cycles,
                       reference_eligibility={"assets": eligible, "declared_by": "reviewer",
                                              "basis": "fixture", "evidence_id": "FIXTURE-2"})
    assert select(config) == 0
    used = read(tmp_path / "out" / "runs" / "base" / "selection.json")["reference_used"]
    assert used["preprocessing"]["assets"] == len(eligible)
    assert used["detectors"]["assets"] == len(eligible)
    assert used["detectors"]["last_available_at"] <= "2026-02-01T00:00:00"
    assert used["preprocessing"]["last_ts"] < "2026-02-01T00:00:00"


def test_labels_that_contradict_declared_eligibility_are_refused(tmp_path):
    cycles = fleet()
    eps = episodes_for(cycles)
    eps.loc[0, ["onset_ts", "fault_ts"]] = ["2026-01-20", "2026-02-10"]
    config = workspace(tmp_path, cycles=cycles, episodes=eps)
    refused(lambda: select(config), "contradict the declared eligibility")
    assert not (tmp_path / "out" / "runs" / "base" / "scores.parquet").exists()


# ---------------------------------------------------------------------------
# provenance and attestations
# ---------------------------------------------------------------------------

def test_external_telemetry_without_a_sidecar_is_refused(tmp_path):
    config = workspace(tmp_path, sidecar=False)
    refused(lambda: select(config), "provenance is not verified")
    assert read(tmp_path / "out" / "runs" / "base" / "preflight.json")["status"] == "refused"


def test_a_sidecar_that_does_not_match_the_telemetry_is_refused(tmp_path):
    config = workspace(tmp_path)
    parquet = tmp_path / "data" / "cycles.parquet"
    frame = pd.read_parquet(parquet)
    frame.loc[0, "peak_current_a"] = 999.0
    frame.to_parquet(parquet, index=False)
    refused(lambda: select(config), "provenance is not verified")


def test_a_sampled_export_is_refused(tmp_path):
    config = workspace(tmp_path)
    sidecar_for(tmp_path / "data" / "cycles.parquet", input={"complete_input": False, "rows_read": 5000})
    refused(lambda: select(config), "SAMPLED export")


def test_missing_attestations_are_refused_and_named(tmp_path):
    config = workspace(tmp_path, attestations={
        "units_verified": True, "identities_verified": False, "fault_labels_verified": True,
        "reference_eligibility_verified": False, "declared_by": "reviewer"})
    message = refused(lambda: select(config), "human attestation(s) not given")
    assert "identities_verified" in message and "reference_eligibility_verified" in message


def test_a_synthetic_run_records_rather_than_enforces_provenance(tmp_path):
    config = workspace(tmp_path, sidecar=False, run_kind="synthetic",
                       reference_eligibility={"assets": "all", "declared_by": "generator",
                                              "basis": "synthetic fixture"},
                       attestations={"units_verified": False, "identities_verified": False,
                                     "fault_labels_verified": False,
                                     "reference_eligibility_verified": False,
                                     "declared_by": "n/a: synthetic"})
    report, _, _ = P.preflight(P.load_config(config))
    assert report["status"] == "passed"
    assert any("provenance is not verified" in w for w in report["synthetic_waived"])
    assert "SYNTHETIC RUN" in report["synthetic_note"]


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def test_a_missing_development_label_file_is_refused(tmp_path):
    config = workspace(tmp_path)
    (tmp_path / DEV).unlink()
    refused(lambda: select(config), "no development label file")


def test_confirmation_only_labels_are_refused_with_the_evaluator_reason(tmp_path):
    cycles = fleet()
    eps = episodes_for(cycles).drop(columns=["onset_ts"])
    config = workspace(tmp_path, cycles=cycles, episodes=eps)
    message = refused(lambda: select(config), "[onset_ts, fault_ts)")
    assert "not invented" in message
    assert "does not implement" in message


def test_blank_onset_values_are_refused(tmp_path):
    cycles = fleet()
    eps = episodes_for(cycles)
    eps.loc[2, "onset_ts"] = None
    refused(lambda: select(workspace(tmp_path, cycles=cycles, episodes=eps)), "onset times are missing")


def test_onset_after_fault_is_refused(tmp_path):
    cycles = fleet()
    eps = episodes_for(cycles)
    eps.loc[0, ["onset_ts", "fault_ts"]] = ["2026-03-10", "2026-03-01"]
    refused(lambda: select(workspace(tmp_path, cycles=cycles, episodes=eps)), "must precede its fault_ts")


# ---------------------------------------------------------------------------
# no qualified candidate
# ---------------------------------------------------------------------------

def test_no_qualified_candidate_is_a_valid_outcome(tmp_path):
    config = workspace(tmp_path, policy=dict(STRICT),
                       data_exposure={"test_period_previously_inspected": False,
                                      "declared_by": "reviewer"})
    assert select(config) == 0
    run = tmp_path / "out" / "runs" / "base"
    report = read(run / "selection.json")
    assert report["selection"]["selected_model"] is None
    assert "valid outcome" in report["outcome"]
    assert all(not c["qualified"] for c in report["selection"]["candidates"])

    watching(tmp_path / TEST)
    try:
        assert P.stage_test(P.load_config(config), "base") == 0
        assert OPENED["hits"] == [], "no winner: the test labels have no reason to be opened"
    finally:
        OPENED["watch"] = None
    test = read(run / "test.json")
    assert test["status"] == "not_evaluated"
    assert "cannot verify that declaration" in test["exposure_statement"]


# ---------------------------------------------------------------------------
# item 4: frozen artifacts, every input validated
# ---------------------------------------------------------------------------

def test_a_run_directory_is_never_reused(frozen, tmp_path):
    root, config, run = clone(frozen, tmp_path)
    refused(lambda: select(config, "base"), "already exists")


def test_a_test_without_a_frozen_selection_is_refused(tmp_path):
    config = write_config(tmp_path)
    refused(lambda: P.stage_test(P.load_config(config), "missing"), "no frozen selection")


@pytest.mark.parametrize("target,fragment", [
    ("selection", "selection.json has changed"),
    ("scores", "scores.parquet has changed"),
    ("telemetry", "telemetry input has changed"),
    ("development_episodes", "development episodes input has changed"),
    ("sidecar", "provenance sidecar input has changed"),
    ("sidecar_removed", "provenance sidecar input has changed"),
    ("config", "configuration has changed"),
])
def test_anything_changed_after_the_freeze_blocks_the_test(frozen, tmp_path, target, fragment):
    root, config, run = clone(frozen, tmp_path)
    sidecar = root / "data" / "cycles.provenance.json"
    if target == "selection":
        body = read(run / "selection.json")
        body["selection"]["selected_model"] = "raw"
        (run / "selection.json").write_text(json.dumps(body), encoding="utf-8")
    elif target == "scores":
        frame = pd.read_parquet(run / "scores.parquet")
        frame.loc[0, "raw"] = 1e9
        frame.to_parquet(run / "scores.parquet", index=False)
    elif target == "telemetry":
        frame = pd.read_parquet(root / "data" / "cycles.parquet")
        frame.loc[0, "peak_current_a"] = 7.0
        frame.to_parquet(root / "data" / "cycles.parquet", index=False)
    elif target == "development_episodes":
        with open(root / DEV, "a", encoding="utf-8") as fh:
            fh.write("X,X,2026-03-01,2026-03-10\n")
    elif target == "sidecar":
        body = read(sidecar)
        body["mapping"]["canonical_sha256"] = "c" * 64    # still verifies; still a different input
        sidecar.write_text(json.dumps(body), encoding="utf-8")
    elif target == "sidecar_removed":
        sidecar.unlink()
    elif target == "config":
        write_config(root, smooth_days=5)
    refused(lambda: P.stage_test(P.load_config(config), "base"), fragment)
    assert not (run / "test.json").exists()


def test_a_freeze_that_predates_complete_input_binding_is_refused(frozen, tmp_path):
    root, config, run = clone(frozen, tmp_path)
    body = read(run / "frozen.json")
    del body["inputs"]["provenance_sidecar_sha256"]
    (run / "frozen.json").write_text(json.dumps(body), encoding="utf-8")
    refused(lambda: P.stage_test(P.load_config(config), "base"), "predates complete input binding")


# ---------------------------------------------------------------------------
# label split helper
# ---------------------------------------------------------------------------

SPLITTER = ROOT / "scripts" / "split_episode_labels.py"


def test_the_label_splitter_scopes_by_confirmation_and_refuses_overwrite(tmp_path):
    src = tmp_path / "all.csv"
    episodes_for(fleet()).to_csv(src, index=False)
    dev, test = tmp_path / "dev.csv", tmp_path / "test.csv"
    args = [sys.executable, "-B", str(SPLITTER), str(src), "--validation-end", BOUNDS["validation_end"],
            "--development-out", str(dev), "--test-out", str(test)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    expected_dev, expected_test = split(pd.read_csv(src))
    assert len(pd.read_csv(dev)) == len(expected_dev) and len(pd.read_csv(test)) == len(expected_test)
    again = subprocess.run(args, capture_output=True, text=True, timeout=120)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr
    same = subprocess.run(args[:-4] + ["--development-out", str(src), "--test-out", str(tmp_path / "t2.csv")],
                          capture_output=True, text=True, timeout=120)
    assert same.returncode != 0 and "Traceback" not in same.stderr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli(*args):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)


def test_the_cli_refuses_bogie_without_a_traceback(tmp_path):
    r = cli("--config", write_config(tmp_path, subsystem="bogie"), "--stage", "select")
    assert r.returncode != 0
    assert "not supported by this runner" in r.stderr
    assert "Traceback" not in r.stderr


def test_the_cli_requires_explicit_modes(tmp_path):
    assert cli().returncode != 0
    r = cli("--config", write_config(tmp_path), "--stage", "test")
    assert r.returncode != 0 and "--run" in r.stderr
    assert cli("--legacy-synthetic").returncode != 0


def test_legacy_mode_refuses_to_overwrite_its_artifacts(tmp_path):
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    r = cli("--legacy-synthetic", "--out", tmp_path)
    assert r.returncode != 0
    assert "refusing to overwrite legacy artifacts" in r.stderr
    assert (tmp_path / "selection.json").read_text(encoding="utf-8") == "{}"
