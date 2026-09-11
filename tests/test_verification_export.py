"""The build-time repair-verification export, and the replay clock that gates it.

The dashboard is a static file, so assessments are exported at build time. Three
properties matter and are easy to lose:

  * an assessment must never surface before its own `as_of`;
  * an open follow-up must survive a later favourable result;
  * exporting must never touch the OPERATIONAL store. Only the separate
    demonstration store may be rebuilt.

Every test that runs the script does so against temporary databases.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from headway.repair_verification import RepairStore
from scripts.build_ui import _visible_from, _verification

EXPORT = ROOT / "data" / "verification_export.json"
pytestmark = pytest.mark.skipif(
    not EXPORT.exists(), reason="run scripts/build_verification.py --seed-demo first")


def run_export(tmp_path, seed=True, store_db=None):
    out = tmp_path / "export.json"
    cmd = [sys.executable, "-B", str(ROOT / "scripts" / "build_verification.py"),
           "--demo-db", str(tmp_path / "demo.sqlite"), "--out", str(out),
           "--store-db", str(store_db or tmp_path / "absent.sqlite")]
    if seed:
        cmd.append("--seed-demo")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-900:]
    return json.loads(out.read_text(encoding="utf-8")), r.stdout


@pytest.fixture(scope="module")
def export():
    return json.loads(EXPORT.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- replay clock
def test_visible_from_never_precedes_the_assessment():
    """A day's aggregate lands the next midnight, so night i knows things up to
    days[i] + 1 day and no further."""
    days = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"])
    # 2026-07-02 06:00 SGT is past night 0's landing time, so night 1 is earliest
    assert _visible_from("2026-07-02T06:00:00+08:00", days) == 1
    # exactly on the boundary: night 0's aggregate lands at 07-02 00:00, which is
    # the assessment instant itself, so night 0 already knows it
    assert _visible_from("2026-07-02T00:00:00+08:00", days) == 0
    # one second later and night 0 no longer suffices
    assert _visible_from("2026-07-02T00:00:01+08:00", days) == 1
    # after the record ends there is no night that may show it
    assert _visible_from("2026-08-01T00:00:00+08:00", days) is None


def test_every_assessment_carries_a_replay_gate():
    days = sorted(pd.read_parquet(ROOT / "data" / "door_deferral.parquet").day.unique())
    v = _verification(days)
    assert v is not None and v["assessments"]
    for a in v["assessments"]:
        assert "visibleFrom" in a
        if a["visibleFrom"] is not None:
            gate = pd.Timestamp(days[a["visibleFrom"]]) + pd.Timedelta(days=1)
            as_of = pd.Timestamp(a["as_of"]).tz_convert("Asia/Singapore").tz_localize(None)
            assert gate >= as_of, f"{a['job_id']} would be shown before it existed"


# -------------------------------------------------------------------- contents
def test_all_three_outcomes_are_demonstrated(export):
    got = {a["status"] for a in export["assessments"]}
    assert {"signal_recovered", "abnormality_persists", "insufficient_evidence"} <= got


def test_a_later_recovered_result_does_not_close_an_open_followup(export):
    by_job = {}
    for a in export["assessments"]:
        by_job.setdefault(a["job_id"], []).append(a)
    both = [j for j, xs in by_job.items()
            if any(x["status"] == "abnormality_persists" for x in xs)
            and any(x["status"] == "signal_recovered" for x in xs)]
    assert both, "the export should demonstrate a job that recovered after a persistent result"
    open_jobs = {f["job_id"] for f in export["followups"] if f["state"] == "open"}
    assert set(both) <= open_jobs, "a favourable assessment silently closed a follow-up"


def test_verification_never_releases_to_service(export):
    assert all(a["release_to_service"] is False for a in export["assessments"])


def test_records_are_labelled_simulated_not_attested(export):
    """No invented engineer names: the demonstration says what it is."""
    for job_id, rec in export["records"].items():
        if job_id.startswith("DEMO-"):
            assert "SIMULATED" in rec["operator"]
            assert "SIMULATED" in rec["finding"]


def test_every_assessment_is_joinable_to_a_job_and_asset(export):
    """The dashboard matches on BOTH ids, so both must be present and resolvable."""
    for a in export["assessments"]:
        assert a["job_id"] and a["asset_id"]
        assert a["job_id"] in export["records"]
        assert export["records"][a["job_id"]]["asset_id"] == a["asset_id"]


# ------------------------------------------------------------ store separation
def test_seeding_the_demo_store_never_touches_the_operational_store(tmp_path):
    """The regression that matters: a normal rebuild must not erase real history."""
    live = tmp_path / "operational.sqlite"
    store = RepairStore(live)
    store.record(job_id="REAL-1", asset_id="TRN001-DOOR-1",
                 started_at=pd.Timestamp("2026-07-01 01:00"),
                 completed_at=pd.Timestamp("2026-07-01 03:00"),
                 recorded_at=pd.Timestamp("2026-07-01 04:00"),
                 operator="probe", finding="probe", work_performed="probe")
    before = live.read_bytes()

    export, _ = run_export(tmp_path, seed=True, store_db=live)

    with RepairStore(live).connect() as db:
        jobs = [r[0] for r in db.execute("SELECT job_id FROM repair_records")]
    assert jobs == ["REAL-1"], "the operational store lost records"
    assert live.read_bytes() == before, "the operational store was rewritten"
    assert "REAL-1" in export["records"], "operational records were not exported"


def test_export_without_seeding_creates_no_database(tmp_path):
    export, _ = run_export(tmp_path, seed=False)
    assert not (tmp_path / "demo.sqlite").exists()
    assert not (tmp_path / "absent.sqlite").exists(), "read of a missing store created one"
    assert export["assessments"] == []


def test_seeding_is_reproducible_and_isolated(tmp_path):
    export, stdout = run_export(tmp_path, seed=True)
    assert "signal_recovered" in stdout and "abnormality_persists" in stdout
    assert (tmp_path / "demo.sqlite").exists()
    assert all(a["demo"] is True for a in export["assessments"])


# --------------------------------------------------------- database safeguards
def test_reading_an_unrelated_database_is_refused_and_leaves_it_untouched(tmp_path):
    """Opening a store through RepairStore would CREATE its tables — a write. The
    exporter must open read-only and reject anything that is not a store."""
    import sqlite3
    foreign = tmp_path / "foreign.sqlite"
    db = sqlite3.connect(foreign)
    db.execute("CREATE TABLE unrelated(x)")
    db.commit()
    db.close()
    before = foreign.read_bytes()

    r = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "build_verification.py"),
         "--store-db", str(foreign), "--demo-db", str(tmp_path / "d.sqlite"),
         "--out", str(tmp_path / "o.json")],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode != 0
    assert "not a repair-verification store" in r.stderr + r.stdout
    assert foreign.read_bytes() == before, "an unrelated database was modified"


def test_the_readonly_connection_actually_refuses_writes(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_verification import open_readonly
    live = tmp_path / "live.sqlite"
    RepairStore(live).record(
        job_id="R-1", asset_id="A", started_at=pd.Timestamp("2026-07-01 01:00"),
        completed_at=pd.Timestamp("2026-07-01 03:00"), recorded_at=pd.Timestamp("2026-07-01 04:00"),
        operator="op", finding="f", work_performed="w")
    db = open_readonly(live)
    try:
        with pytest.raises(Exception) as exc:
            db.execute("INSERT INTO repair_records VALUES('x','y','{}')")
            db.commit()
        assert "readonly" in str(exc.value).lower()
    finally:
        db.close()


@pytest.mark.parametrize("alias", ["store", "out"])
def test_aliased_paths_are_refused_before_anything_is_deleted(tmp_path, alias):
    """`--seed-demo --demo-db <operational>` would wipe the real store."""
    demo = tmp_path / "demo.sqlite"
    demo.write_bytes(b"sentinel")
    args = ["--seed-demo", "--demo-db", str(demo)]
    args += ["--store-db", str(demo)] if alias == "store" else ["--out", str(demo)]
    if alias == "store":
        args += ["--out", str(tmp_path / "o.json")]
    else:
        args += ["--store-db", str(tmp_path / "absent.sqlite")]
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "build_verification.py")] + args,
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode != 0
    assert "resolve to the same path" in r.stderr + r.stdout
    assert demo.read_bytes() == b"sentinel", "a path was deleted despite the alias check"


def test_colliding_job_ids_between_stores_are_refused(tmp_path):
    """job_id is the dashboard's join key; a silent overwrite would hand one
    store's maintenance record to the other store's assessment."""
    export, _ = run_export(tmp_path, seed=True)
    demo_job = export["assessments"][0]["job_id"]
    live = tmp_path / "live.sqlite"
    RepairStore(live).record(
        job_id=demo_job, asset_id="TRN001-DOOR-1", started_at=pd.Timestamp("2026-07-01 01:00"),
        completed_at=pd.Timestamp("2026-07-01 03:00"), recorded_at=pd.Timestamp("2026-07-01 04:00"),
        operator="op", finding="f", work_performed="w")
    r = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / "build_verification.py"),
         "--store-db", str(live), "--demo-db", str(tmp_path / "demo.sqlite"),
         "--out", str(tmp_path / "o2.json")],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode != 0
    assert "present in both" in r.stderr + r.stdout and demo_job in r.stderr + r.stdout
