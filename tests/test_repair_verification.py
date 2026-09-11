import sqlite3

import numpy as np
import pandas as pd
import pytest

from headway.repair_verification import (RepairStore, VerificationPolicy,
    make_reference, maintenance_record, verify_repair)


@pytest.fixture
def case():
    days = pd.date_range("2026-01-01", periods=30)
    daily = pd.DataFrame({"asset_id": "A", "day": days,
        "available_at": days + pd.Timedelta(days=1), "data_quality_ok": True,
        "context_supported": True, "motor_hx": np.tile([-1., 1.], 15),
        "travel_hx": np.tile([1., -1.], 15), "load_proxy_mean": np.tile([.3, .7], 15)})
    ref = make_reference(daily.iloc[:14], asset_id="A", channels=["motor_hx", "travel_hx"],
        contexts=["load_proxy_mean"], model_version="frozen-1", reviewed_by="test inspector",
        reviewed_at="2026-01-15")
    record = maintenance_record(job_id="job-1", asset_id="A", started_at="2026-01-16 10:00",
        completed_at="2026-01-16 12:00", recorded_at="2026-01-16 13:00",
        operator="test engineer", finding="roller binding", work_performed="roller replaced")
    return daily, ref, record


def assess(case, **kwargs):
    daily, ref, record = case
    return verify_repair(record, ref, daily, as_of=kwargs.pop("as_of", "2026-01-21"),
        model_version=kwargs.pop("model_version", "frozen-1"), **kwargs)


def test_recovered_requires_full_window_and_never_clears_service(case):
    result = assess(case)
    assert result["status"] == "signal_recovered"
    assert result["observed_days"] == 3
    assert not result["release_to_service"] and not result["follow_up_required"]
    assert assess(case, as_of="2026-01-20")["status"] == "insufficient_evidence"


def test_one_persistent_channel_cannot_be_cancelled_by_another(case):
    daily, ref, record = case
    daily.loc[daily.day.between("2026-01-18", "2026-01-20"), "travel_hx"] = -15
    result = assess(case)
    assert result["status"] == "abnormality_persists"
    assert result["channels"]["travel_hx"]["abnormal_days"] == 3
    assert result["follow_up_required"]


@pytest.mark.parametrize("kind", ["missing", "nan", "unsupported", "quality", "text_false", "late", "context"])
def test_bad_recent_evidence_never_recovers(case, kind):
    daily, ref, record = case
    idx = daily.index[daily.day.eq("2026-01-20")][0]
    if kind == "missing":
        daily = daily.drop(index=idx)
    elif kind == "nan":
        daily.loc[idx, "motor_hx"] = np.nan
    elif kind == "unsupported":
        daily.loc[idx, "context_supported"] = False
    elif kind == "quality":
        daily.loc[idx, "data_quality_ok"] = False
    elif kind == "text_false":
        daily["data_quality_ok"] = daily.data_quality_ok.astype(object)
        daily.loc[idx, "data_quality_ok"] = "false"
    elif kind == "late":
        daily.loc[idx, "available_at"] = pd.Timestamp("2026-01-22")
    else:
        daily.loc[idx, "load_proxy_mean"] = .99
    assert assess((daily, ref, record))["status"] == "insufficient_evidence"


def test_isolated_departure_is_mixed_evidence(case):
    case[0].loc[case[0].day.eq("2026-01-20"), "motor_hx"] = 15
    assert assess(case)["status"] == "insufficient_evidence"


def test_future_telemetry_and_labels_do_not_change_past_verification(case):
    before = assess(case)
    case[0].loc[case[0].day.ge("2026-01-21"), "motor_hx"] = 9999
    case[0]["fault_confirmed"] = True
    assert assess(case) == before


def test_changed_model_and_later_intervention_abstain(case):
    assert assess(case, model_version="new-model")["status"] == "insufficient_evidence"
    assert assess(case, subsequent_maintenance_at="2026-01-19")["status"] == "insufficient_evidence"


def test_reference_integrity(case):
    case[1]["scale"]["motor_hx"] = 100
    with pytest.raises(ValueError, match="fingerprint"):
        assess(case)


def test_reference_requires_valid_reviewed_history(case):
    daily, _, _ = case
    kwargs = dict(asset_id="A", channels=["motor_hx"], contexts=["load_proxy_mean"],
        model_version="m", reviewed_by="engineer", reviewed_at="2026-01-15")
    with pytest.raises(ValueError, match="available"):
        make_reference(daily, **kwargs)
    with pytest.raises(ValueError, match="degenerate"):
        make_reference(daily.iloc[:14].assign(motor_hx=1), **kwargs)
    with pytest.raises(ValueError, match="reviewed_by"):
        make_reference(daily.iloc[:14], **{**kwargs, "reviewed_by": ""})


def test_duplicate_days_rejected(case):
    daily, ref, record = case
    with pytest.raises(ValueError, match="duplicate"):
        assess((pd.concat([daily, daily.iloc[-1:]]), ref, record))


def test_persistent_history_survives_reopening_and_repeated_jobs(case, tmp_path):
    daily, ref, record = case
    path = tmp_path / "repairs.sqlite"
    store = RepairStore(path)
    store.record(**record)
    with pytest.raises(sqlite3.IntegrityError):
        store.record(**record)
    bad = daily.copy()
    bad.loc[bad.day.between("2026-01-18", "2026-01-20"), "motor_hx"] = 20
    first = store.assess("job-1", ref, bad, as_of="2026-01-21", model_version="frozen-1")
    store = RepairStore(path)
    store.assess("job-1", ref, daily, as_of="2026-01-22", model_version="frozen-1")
    assert len(store.history("job-1")) == 2
    assert store.followups() == [{"job_id": "job-1", "assessment_id": first["assessment_id"], "state": "open"}]
    next_record = {**record, "job_id": "job-2", "started_at": "2026-01-22",
        "completed_at": "2026-01-22 01:00", "recorded_at": "2026-01-22 02:00"}
    store.record(**next_record)
    result = store.assess("job-1", ref, daily, as_of="2026-01-25", model_version="frozen-1")
    assert result["status"] == "insufficient_evidence"
    assert "subsequent" in result["reason"]


def test_timezone_equivalence(case):
    assert assess(case, as_of="2026-01-20T16:00:00Z") == assess(case)


def test_policy_validation():
    with pytest.raises(ValueError):
        VerificationPolicy(persistent_days=4)
    with pytest.raises(ValueError):
        VerificationPolicy(post_days=2.5)


def test_cli_round_trip(case, tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    daily, _, record = case
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_repair.py"
    ref_daily, all_daily = tmp_path / "ref.parquet", tmp_path / "all.parquet"
    daily.iloc[:14].to_parquet(ref_daily)
    daily.to_parquet(all_daily)
    record_path, reference_path, db = tmp_path / "record.json", tmp_path / "reference.json", tmp_path / "store.sqlite"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    def run(*args):
        result = subprocess.run([sys.executable, "-B", str(script), *map(str, args)],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=30)
        return json.loads(result.stdout)

    run("reference", "--daily", ref_daily, "--asset", "A", "--channels", "motor_hx", "travel_hx",
        "--contexts", "load_proxy_mean", "--model-version", "frozen-1", "--reviewed-by", "test inspector",
        "--reviewed-at", "2026-01-15", "--output", reference_path)
    run("record", "--db", db, "--record", record_path)
    result = run("assess", "--db", db, "--job", "job-1", "--reference", reference_path,
        "--daily", all_daily, "--as-of", "2026-01-21", "--model-version", "frozen-1")
    assert result["status"] == "signal_recovered"
    assert len(run("history", "--db", db, "--job", "job-1")) == 1
    assert run("followups", "--db", db) == []
