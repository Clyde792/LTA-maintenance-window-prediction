"""Reviewed mapping -> canonical Parquet -> readiness, end to end on temp files.

The path this covers is the one where a wrong answer is silent: a vendor export
in milliamps and UTC, bound by a mapping somebody guessed, written over an
earlier export, and read downstream as though it were the whole dataset in local
time. Every step below is checked against files actually written to disk.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from headway.adapters.mapping import MappingError, load_mapping, parse_mapping  # noqa: E402
from headway.data_readiness import assess_readiness  # noqa: E402

BIND = ROOT / "scripts" / "bind_data.py"
READINESS = ROOT / "scripts" / "ps3_readiness.py"

# A plausible vendor export: milliamps, metres, day-first text timestamps in UTC,
# Y/N labels, and two columns nobody asked for.
VENDOR_COLUMNS = {
    "Event_Time": "ts", "EQUIPMENT_ID": "asset_id", "Car No": "train_id",
    "operation_time": "cycle_duration_s", "I_Peak (mA)": "peak_current_a",
    "avg_current_mA": "mean_current_a", "leaf_travel_m": "travel_mm",
    "Obstruction_Detected": "obstruction_flag", "recycle_count": "retry_count",
    "saloon_temp": "ambient_temp_c", "load_weigh": "load_proxy",
    "operations_since_service": "cycles_since_service",
    "verified_fault": "fault_confirmed", "failure_mode": "fault_mode",
}

MAPPING = {
    "reviewed": True,
    "subsystem": "door",
    "columns": {v: k for k, v in VENDOR_COLUMNS.items()},
    "unit_scales": {"peak_current_a": 0.001, "mean_current_a": 0.001, "travel_mm": 1000.0},
    "source_timezone": "UTC",
    "timestamp_format": "%d/%m/%Y %H:%M:%S",
    "boolean_tokens": {"Y": True, "N": False},
    "reviewed_by": "test reviewer",
    "notes": "fixture",
}


def vendor_csv(path, *, days=40, assets=3, per_day=6, labels="YN", context=True):
    """Write a vendor-shaped CSV. Timestamps are UTC text, day-first."""
    start = pd.Timestamp("2026-03-01 00:30:00")
    rows = []
    for a in range(assets):
        for d in range(days):
            for c in range(per_day):
                ts = start + pd.Timedelta(days=d, hours=c * 3)
                rows.append({
                    "Event_Time": ts.strftime("%d/%m/%Y %H:%M:%S"),
                    "EQUIPMENT_ID": f"TRN00{a}-DOOR-1", "Car No": f"TRN00{a}",
                    "operation_time": 2.5 + 0.01 * c,
                    "I_Peak (mA)": 3200.0 + c,          # milliamps
                    "avg_current_mA": 1800.0 + c,       # milliamps
                    "leaf_travel_m": 1.3,               # metres
                    "Obstruction_Detected": 0, "recycle_count": 0,
                    "saloon_temp": 28.0 + c * 0.1, "load_weigh": 0.4,
                    "operations_since_service": d * per_day + c,
                    "verified_fault": labels[1], "failure_mode": "",
                    "Notes": "", "record_id": len(rows),
                })
    frame = pd.DataFrame(rows)
    if not context:
        frame = frame.drop(columns=["saloon_temp", "load_weigh"])
    frame.to_csv(path, index=False)
    return frame


def write_mapping(path, **over):
    body = dict(MAPPING)
    body.update(over)
    Path(path).write_text(json.dumps(body, indent=2), encoding="utf-8")
    return body


def run_bind(*args):
    return subprocess.run([sys.executable, "-B", str(BIND), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


@pytest.fixture
def vendor(tmp_path):
    source = tmp_path / "vendor_export.csv"
    raw = vendor_csv(source)
    mapping = tmp_path / "door_mapping.json"
    write_mapping(mapping)
    return {"dir": tmp_path, "source": source, "mapping": mapping, "raw": raw,
            "out": tmp_path / "canonical_door_cycles.parquet"}


# --- end to end ----------------------------------------------------------------

def test_vendor_csv_to_canonical_parquet_to_readiness(vendor):
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert vendor["out"].exists()
    prov_path = vendor["out"].with_suffix(".provenance.json")
    assert prov_path.exists()

    frame = pd.read_parquet(vendor["out"])
    assert len(frame) == len(vendor["raw"])
    assert {"ts", "asset_id", "train_id", "peak_current_a", "context_supported"} <= set(frame)

    report = assess_readiness(frame)
    assert report["telemetry_rows"] == len(frame)
    assert report["observed_days"] == 40
    assert report["invalid_timestamps"] == 0
    assert report["missing_columns"] == []
    # The only outstanding blockers are the human attestations this tool
    # deliberately does not make on anybody's behalf.
    assert set(report["current_pipeline"]["blockers"]) == {
        "source units and conversions need review",
        "telemetry-to-asset identities need review"}


def test_the_input_file_is_never_modified(vendor):
    before = vendor["source"].read_bytes()
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    assert vendor["source"].read_bytes() == before


# --- unit conversion, timezone, labels -----------------------------------------

def test_declared_unit_scales_are_applied(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    frame = pd.read_parquet(vendor["out"])
    # 3200 mA -> 3.2 A, 1.3 m -> 1300 mm
    assert frame.peak_current_a.min() == pytest.approx(3.2, abs=1e-9)
    assert frame.mean_current_a.min() == pytest.approx(1.8, abs=1e-9)
    assert frame.travel_mm.unique().tolist() == [1300.0]
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert body["unit_conversions"] == {"mean_current_a": 0.001, "peak_current_a": 0.001,
                                        "travel_mm": 1000.0}


def test_unit_scales_are_applied_before_derivation(vendor):
    """Charge is mean current x duration; converting after would be 1000x wrong."""
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    frame = pd.read_parquet(vendor["out"])
    expected = frame.mean_current_a * frame.cycle_duration_s
    assert frame.current_integral_as.to_numpy() == pytest.approx(expected.to_numpy())
    assert frame.current_integral_as.max() < 10.0


def test_utc_source_is_converted_to_naive_local(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    frame = pd.read_parquet(vendor["out"])
    assert frame.ts.dt.tz is None, "canonical timestamps must be naive local"
    # 00:30 UTC is 08:30 Asia/Singapore, so the first cycle lands on the same day.
    assert frame.ts.min() == pd.Timestamp("2026-03-01 08:30:00")
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert body["timestamps"]["source_timezone"] == "UTC"
    assert body["timestamps"]["target_timezone"] == "Asia/Singapore"


def test_an_undeclared_timezone_is_recorded_as_such(vendor):
    write_mapping(vendor["mapping"], source_timezone=None)
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    frame = pd.read_parquet(vendor["out"])
    assert frame.ts.min() == pd.Timestamp("2026-03-01 00:30:00"), "naive input was shifted"
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert "no source_timezone declared" in body["timestamps"]["note"]


def test_declared_yn_tokens_are_honoured(tmp_path):
    source = tmp_path / "vendor.csv"
    vendor_csv(source, labels="YY")          # every row labelled "Y"
    mapping = tmp_path / "m.json"
    write_mapping(mapping)
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    frame = pd.read_parquet(out)
    assert frame.fault_confirmed.all()
    body = json.loads(out.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert body["boolean_tokens_declared"] == {"n": False, "y": True}


def test_an_undeclared_token_fails_rather_than_becoming_false(tmp_path):
    """The failure mode this guards: a fault quietly read as 'not a fault'."""
    source = tmp_path / "vendor.csv"
    vendor_csv(source, labels="YN")
    frame = pd.read_csv(source)
    frame.loc[0, "verified_fault"] = "CONFIRMED"
    frame.to_csv(source, index=False)
    mapping = tmp_path / "m.json"
    write_mapping(mapping)
    out = tmp_path / "c.parquet"

    r = run_bind(source, "--mapping", mapping, "--out", out)
    assert r.returncode != 0
    assert "unrecognised fault_confirmed value" in (r.stdout + r.stderr)
    assert "confirmed" in (r.stdout + r.stderr).lower()
    assert not out.exists(), "a partial export survived a label failure"

    write_mapping(mapping, boolean_tokens={"Y": True, "N": False, "CONFIRMED": True})
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    assert pd.read_parquet(out).fault_confirmed.sum() == 1


# --- invalid mappings -----------------------------------------------------------

def test_an_unreviewed_mapping_is_refused(vendor):
    write_mapping(vendor["mapping"], reviewed=False)
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"])
    assert r.returncode != 0
    assert "not marked reviewed" in (r.stdout + r.stderr)
    assert not vendor["out"].exists()


@pytest.mark.parametrize("body,message", [
    ({"reviewed": True, "subsystem": "door"}, "missing required key"),
    ({"reviewed": True, "subsystem": "wagon", "columns": {"ts": "a"}}, "unknown subsystem"),
    ({"reviewed": True, "subsystem": "door", "columns": {}}, "non-empty object"),
    ({"reviewed": True, "subsystem": "door", "columns": {"not_a_column": "a"}}, "not contract columns"),
    ({"reviewed": True, "subsystem": "door", "columns": {"ts": "a", "asset_id": "a"}},
     "mapped to several contract columns"),
    ({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"}, "unit_scale": {}},
     "unknown mapping key"),
    ({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
      "unit_scales": {"peak_current_a": 0}}, "finite positive"),
    ({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
      "unit_scales": {"asset_id": 2}}, "not a signal or context"),
    ({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
      "source_timezone": "Mars/Olympus"}, "unknown source_timezone"),
    ({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
      "boolean_tokens": {"Y": "yes"}}, "must map to true or false"),
])
def test_invalid_mappings_are_refused_with_a_reason(body, message):
    with pytest.raises(MappingError, match=message):
        parse_mapping(body)


def test_a_mapping_for_another_subsystem_is_refused(vendor):
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--subsystem", "bogie",
                 "--out", vendor["out"])
    assert r.returncode != 0
    assert "they must agree" in (r.stdout + r.stderr)


def test_a_missing_mapping_file_is_refused(vendor):
    r = run_bind(vendor["source"], "--mapping", vendor["dir"] / "absent.json",
                 "--out", vendor["out"])
    assert r.returncode != 0
    assert "mapping file not found" in (r.stdout + r.stderr)


def test_the_canonical_hash_ignores_key_order_and_formatting(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(MAPPING, indent=2, sort_keys=True), encoding="utf-8")
    b.write_text(json.dumps(dict(reversed(list(MAPPING.items()))), separators=(",", ":")),
                 encoding="utf-8")
    first, second = load_mapping(a), load_mapping(b)
    assert first.canonical_sha256 == second.canonical_sha256
    assert first.file_sha256 != second.file_sha256


# --- guessed mappings are never exported ----------------------------------------

def test_out_without_a_mapping_is_refused(vendor):
    r = run_bind(vendor["source"], "--out", vendor["out"])
    assert r.returncode != 0
    assert "--out requires --mapping" in (r.stdout + r.stderr)
    assert not vendor["out"].exists()


def test_the_suggester_writes_a_draft_that_nothing_accepts(tmp_path):
    source = tmp_path / "vendor.csv"
    vendor_csv(source, labels="10")          # tokens the adapter already knows
    drafted = tmp_path / "draft.json"
    r = run_bind(source, "--suggest-out", drafted)
    assert r.returncode == 0, r.stdout + r.stderr
    body = json.loads(drafted.read_text(encoding="utf-8"))
    assert body["reviewed"] is False
    assert "GUESSES" in body["notes"]
    with pytest.raises(MappingError, match="not marked reviewed"):
        load_mapping(drafted)


def test_the_suggester_reports_an_unbindable_file_and_still_drafts(vendor):
    """A review aid must survive data it cannot yet interpret.

    Y/N labels are exactly the case you need a draft mapping for, so crashing
    with a traceback would withhold the one output that helps.
    """
    drafted = vendor["dir"] / "draft.json"
    r = run_bind(vendor["source"], "--suggest-out", drafted)
    assert r.returncode != 0, "an unbindable file reported success"
    assert "CANNOT BIND WITH THE GUESSED MAPPING" in r.stdout
    assert "unrecognised fault_confirmed value" in r.stdout
    assert "boolean_tokens" in r.stdout
    assert "Traceback" not in r.stderr
    assert drafted.exists(), "the draft was withheld by the failure it explains"
    assert json.loads(drafted.read_text(encoding="utf-8"))["reviewed"] is False


# --- sampling and overwrites -----------------------------------------------------

def test_a_sampled_export_is_marked_and_surfaced(vendor):
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"],
                 "--rows", "50")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SAMPLED" in r.stdout
    frame = pd.read_parquet(vendor["out"])
    assert len(frame) == 50
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert body["input"]["complete_input"] is False
    assert body["input"]["rows_read"] == 50
    assert "not the complete dataset" in body["input"]["sampling_note"]

    check = subprocess.run([sys.executable, "-B", str(READINESS), str(vendor["out"])],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert check.returncode == 0, check.stderr
    report = json.loads(check.stdout)
    assert report["source_provenance"]["complete_input"] is False
    assert "SAMPLED EXPORT" in report["source_provenance"]["warning"]
    assert "SAMPLED EXPORT" in check.stderr


def test_a_complete_export_is_not_marked_as_sampled(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert body["input"]["complete_input"] is True
    assert body["input"]["sampling_note"] is None


def test_an_existing_output_is_not_overwritten_by_accident(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    first = vendor["out"].read_bytes()

    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"],
                 "--rows", "50")
    assert r.returncode != 0
    assert "refusing to overwrite" in (r.stdout + r.stderr)
    assert vendor["out"].read_bytes() == first, "the earlier export was damaged"

    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"],
                 "--rows", "50", "--overwrite")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(pd.read_parquet(vendor["out"])) == 50


def test_an_orphaned_provenance_file_also_blocks(vendor):
    """Both halves are the artifact; a stale sidecar must not be silently replaced."""
    vendor["out"].with_suffix(".provenance.json").write_text("{}", encoding="utf-8")
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"])
    assert r.returncode != 0
    assert "refusing to overwrite" in (r.stdout + r.stderr)


def test_no_partial_files_survive_a_failure(vendor):
    """A bind that fails must leave neither half, nor a .partial."""
    write_mapping(vendor["mapping"], columns={"ts": "Event_Time", "asset_id": "NOT_A_COLUMN"})
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"])
    assert r.returncode != 0
    assert not vendor["out"].exists()
    assert not vendor["out"].with_suffix(".provenance.json").exists()
    assert list(vendor["dir"].glob("*.partial")) == []


# --- missing context keeps its unsupported status ---------------------------------

def test_missing_context_stays_unsupported_through_to_readiness(tmp_path):
    source = tmp_path / "vendor.csv"
    vendor_csv(source, context=False)
    mapping = tmp_path / "m.json"
    columns = {k: v for k, v in MAPPING["columns"].items()
               if k not in ("ambient_temp_c", "load_proxy")}
    write_mapping(mapping, columns=columns,
                  unit_scales={"peak_current_a": 0.001, "mean_current_a": 0.001,
                               "travel_mm": 1000.0})
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0

    frame = pd.read_parquet(out)
    assert not frame.context_supported.any(), "filled context was marked supported"
    body = json.loads(out.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert body["context_supported"] is False
    assert set(body["columns"]["context_filled"]) == {"ambient_temp_c", "load_proxy"}

    report = assess_readiness(frame)
    assert "imputed or unsupported context requires review before current-pipeline fitting" \
        in report["current_pipeline"]["blockers"]


# --- the export invents nothing ----------------------------------------------------

def test_the_export_creates_no_episodes_or_eligibility(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    produced = {p.name for p in vendor["dir"].iterdir()}
    assert produced == {vendor["source"].name, vendor["mapping"].name,
                        vendor["out"].name, vendor["out"].with_suffix(".provenance.json").name}
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert any("onset" in item for item in body["not_provided_by_this_export"])
    assert any("episodes" in item for item in body["not_provided_by_this_export"])
    assert any("eligibility" in item for item in body["not_provided_by_this_export"])


def test_schema_configuration_is_not_a_units_or_label_attestation(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    attested = body["attestations"]
    assert attested["units_verified"] is False
    assert attested["identities_verified"] is False
    assert attested["fault_labels_verified"] is False
    assert "A reviewed mapping is not those reviews" in attested["note"]
    assert body["mapping"]["reviewed"] is True, "the mapping itself was reviewed"


def test_provenance_records_both_input_and_mapping_identity(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert len(body["input"]["sha256"]) == 64
    assert len(body["mapping"]["file_sha256"]) == 64
    assert body["mapping"]["canonical_sha256"] == load_mapping(vendor["mapping"]).canonical_sha256
    assert body["input"]["modified_by_this_tool"] is False
    assert body["rows"]["input"] == body["rows"]["output"] == len(pd.read_parquet(vendor["out"]))
    assert body["validation"]["ok"] is True


# --- 1 · an overwrite that fails must leave the previous export intact ----------
# Two sequential renames are not an atomic pair. Publication is exercised here by
# failing at each step in turn, with a good export already on disk.

def published(vendor):
    """A good export, and the bytes of both halves."""
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    prov = vendor["out"].with_suffix(".provenance.json")
    return prov, vendor["out"].read_bytes(), prov.read_bytes()


@pytest.mark.parametrize("fail_on_call", [1, 2, 3, 4])
def test_a_failed_overwrite_restores_both_halves(vendor, monkeypatch, fail_on_call):
    """os.replace is called four times on an overwrite: two moves aside, two
    publishes. Failing at each one must leave the earlier export exactly as it
    was, and leave no partials or backups behind."""
    import bind_data

    prov, first_parquet, first_sidecar = published(vendor)
    real_replace, calls = bind_data.os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == fail_on_call:
            raise OSError(f"injected failure on replace #{fail_on_call}")
        return real_replace(src, dst)

    monkeypatch.setattr(bind_data.os, "replace", flaky)
    frame = pd.read_parquet(vendor["out"]).head(5)
    with pytest.raises(OSError, match="injected failure"):
        bind_data.write_canonical(vendor["out"], frame, lambda h: {"output": {"sha256": h}},
                                  overwrite=True)

    assert vendor["out"].read_bytes() == first_parquet, "the previous Parquet was lost"
    assert prov.read_bytes() == first_sidecar, "the previous sidecar was lost"
    assert list(vendor["dir"].glob("*.partial")) == []
    assert list(vendor["dir"].glob("*.backup")) == []


def test_a_failed_frame_write_leaves_the_previous_export_alone(vendor, monkeypatch):
    import bind_data

    prov, first_parquet, first_sidecar = published(vendor)

    class Unwritable(pd.DataFrame):
        @property
        def _constructor(self):
            return Unwritable

        def to_parquet(self, *a, **kw):
            raise OSError("injected failure writing the frame")

    with pytest.raises(OSError, match="injected failure"):
        bind_data.write_canonical(vendor["out"], Unwritable(pd.read_parquet(vendor["out"]).head(3)),
                                  lambda h: {"output": {"sha256": h}}, overwrite=True)
    assert vendor["out"].read_bytes() == first_parquet
    assert prov.read_bytes() == first_sidecar
    assert list(vendor["dir"].glob("*.partial")) == []


def test_a_failed_sidecar_build_leaves_the_previous_export_alone(vendor):
    import bind_data

    prov, first_parquet, first_sidecar = published(vendor)

    def explode(_hash):
        raise RuntimeError("injected failure building provenance")

    with pytest.raises(RuntimeError, match="injected failure"):
        bind_data.write_canonical(vendor["out"], pd.read_parquet(vendor["out"]).head(3),
                                  explode, overwrite=True)
    assert vendor["out"].read_bytes() == first_parquet
    assert prov.read_bytes() == first_sidecar
    assert list(vendor["dir"].glob("*.partial")) == []


def test_leftovers_from_an_interrupted_run_block_the_next_one(vendor):
    """The documented crash case: backups on disk mean a run did not finish."""
    published(vendor)
    (vendor["dir"] / (vendor["out"].name + ".backup")).write_bytes(b"stale")
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["out"],
                 "--overwrite")
    assert r.returncode != 0
    assert "did not finish" in (r.stdout + r.stderr)


# --- 2 · source files are never a write target ----------------------------------

def test_the_input_can_never_be_the_output(vendor):
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", vendor["source"],
                 "--overwrite")
    assert r.returncode != 0
    assert "are the same file" in (r.stdout + r.stderr)
    assert "never authorises" in (r.stdout + r.stderr)
    assert vendor["source"].exists()


def test_the_reviewed_mapping_can_never_be_the_output(vendor):
    for flag in ("--out", "--suggest-out"):
        r = run_bind(vendor["source"], "--mapping", vendor["mapping"], flag, vendor["mapping"],
                     "--overwrite")
        assert r.returncode != 0, flag
        assert "are the same file" in (r.stdout + r.stderr)
        assert json.loads(vendor["mapping"].read_text(encoding="utf-8"))["reviewed"] is True


def test_an_alias_by_a_different_spelling_is_still_refused(vendor):
    """Same file, different path text: resolve first, compare after."""
    aliased = vendor["dir"] / "sub" / ".." / vendor["source"].name
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", aliased,
                 "--overwrite")
    assert r.returncode != 0
    assert "are the same file" in (r.stdout + r.stderr)


def test_the_sidecar_can_never_collide_with_a_source(vendor):
    r = run_bind(vendor["source"], "--mapping", vendor["mapping"],
                 "--out", vendor["dir"] / "door_mapping.parquet", "--overwrite")
    assert r.returncode == 0, r.stdout + r.stderr
    clash = vendor["dir"] / "collide.parquet"
    sidecar_target = clash.with_suffix(".provenance.json")
    sidecar_target.write_text("{}", encoding="utf-8")
    r = run_bind(vendor["source"], "--mapping", sidecar_target, "--out", clash)
    assert r.returncode != 0
    assert "are the same file" in (r.stdout + r.stderr)


def test_a_draft_is_not_overwritten_by_accident(vendor):
    drafted = vendor["dir"] / "draft.json"
    drafted.write_text('{"keep": "me"}', encoding="utf-8")
    r = run_bind(vendor["source"], "--suggest-out", drafted)
    assert r.returncode != 0
    assert "refusing to overwrite the draft mapping" in (r.stdout + r.stderr)
    assert json.loads(drafted.read_text(encoding="utf-8")) == {"keep": "me"}


# --- 3 · mixed timezones ---------------------------------------------------------

def mixed_csv(path, values, **over):
    frame = pd.read_csv(over.pop("template"))
    frame = frame.head(len(values)).copy()
    frame["Event_Time"] = values
    frame.to_csv(path, index=False)
    return frame


def test_offset_aware_and_naive_rows_each_keep_their_meaning(tmp_path):
    """All three name the same instant; all three must land on the same local time."""
    template = tmp_path / "t.csv"
    vendor_csv(template)
    source = tmp_path / "mixed.csv"
    mixed_csv(source, ["2026-03-01T00:30:00+00:00", "2026-03-01 00:30:00",
                       "2026-03-01T08:30:00+08:00"], template=template)
    mapping = tmp_path / "m.json"
    write_mapping(mapping, source_timezone="UTC", timestamp_format=None)
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    ts = pd.read_parquet(out).ts
    assert ts.dt.tz is None
    assert ts.nunique() == 1, f"rows drifted apart: {sorted(ts.unique())}"
    assert ts.iloc[0] == pd.Timestamp("2026-03-01 08:30:00")


def test_a_naive_row_is_not_read_as_utc_because_a_neighbour_has_an_offset(tmp_path):
    """The reported defect: one offset row made the whole column parse as UTC."""
    template = tmp_path / "t.csv"
    vendor_csv(template)
    source = tmp_path / "mixed.csv"
    mixed_csv(source, ["2026-03-01T10:00:00+00:00", "2026-03-01 12:00:00"], template=template)
    mapping = tmp_path / "m.json"
    write_mapping(mapping, source_timezone="Asia/Singapore", timestamp_format=None)
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    ts = sorted(pd.read_parquet(out).ts)
    # 10:00 UTC is 18:00 SGT; the naive 12:00 is already SGT and must not move.
    assert ts == [pd.Timestamp("2026-03-01 12:00:00"), pd.Timestamp("2026-03-01 18:00:00")]


def test_mixed_input_without_a_declared_zone_is_refused(tmp_path):
    template = tmp_path / "t.csv"
    vendor_csv(template)
    source = tmp_path / "mixed.csv"
    mixed_csv(source, ["2026-03-01T00:30:00+00:00", "2026-03-01 00:30:00"], template=template)
    mapping = tmp_path / "m.json"
    write_mapping(mapping, source_timezone=None, timestamp_format=None)
    out = tmp_path / "c.parquet"
    r = run_bind(source, "--mapping", mapping, "--out", out)
    assert r.returncode != 0
    assert "mixes offset-aware and naive values" in (r.stdout + r.stderr)
    assert not out.exists()


def test_a_dst_ambiguous_local_time_becomes_invalid_rather_than_a_guess(tmp_path):
    """01:30 on 1 Nov 2026 happens twice in New York. Guessing is not available."""
    template = tmp_path / "t.csv"
    vendor_csv(template)
    source = tmp_path / "dst.csv"
    mixed_csv(source, ["2026-11-01 01:30:00", "2026-11-02 01:30:00"], template=template)
    mapping = tmp_path / "m.json"
    write_mapping(mapping, source_timezone="America/New_York", timestamp_format=None)
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    ts = pd.read_parquet(out).ts
    assert ts.isna().sum() == 1, "the ambiguous local time was resolved by guessing"
    assert assess_readiness(pd.read_parquet(out))["invalid_timestamps"] == 1


def test_a_nonexistent_local_time_becomes_invalid(tmp_path):
    """02:30 on 8 Mar 2026 never happens in New York."""
    template = tmp_path / "t.csv"
    vendor_csv(template)
    source = tmp_path / "dst.csv"
    mixed_csv(source, ["2026-03-08 02:30:00", "2026-03-09 02:30:00"], template=template)
    mapping = tmp_path / "m.json"
    write_mapping(mapping, source_timezone="America/New_York", timestamp_format=None)
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    assert pd.read_parquet(out).ts.isna().sum() == 1


def test_a_dst_transition_is_handled_in_mixed_columns_too(tmp_path):
    template = tmp_path / "t.csv"
    vendor_csv(template)
    source = tmp_path / "dst.csv"
    mixed_csv(source, ["2026-11-01T05:30:00+00:00", "2026-11-01 01:30:00"], template=template)
    mapping = tmp_path / "m.json"
    write_mapping(mapping, source_timezone="America/New_York", timestamp_format=None)
    out = tmp_path / "c.parquet"
    assert run_bind(source, "--mapping", mapping, "--out", out).returncode == 0
    ts = pd.read_parquet(out).ts
    assert ts.isna().sum() == 1, "the ambiguous naive value was guessed"
    assert ts.dropna().iloc[0] == pd.Timestamp("2026-11-01 13:30:00")


# --- 4 · conflicting and malformed token configuration ----------------------------

def test_tokens_that_normalise_together_with_opposite_values_are_refused():
    with pytest.raises(MappingError, match="both normalise to 'y' but map to different values"):
        parse_mapping({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
                       "boolean_tokens": {"Y": True, " y ": False}})


def test_redefining_a_token_the_adapter_already_knows_is_refused():
    with pytest.raises(MappingError, match="would invert every row"):
        parse_mapping({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
                       "boolean_tokens": {"YES": False}})


def test_agreeing_duplicates_are_accepted():
    config = parse_mapping({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
                            "boolean_tokens": {"Y": True, " y ": True}})
    assert config.boolean_tokens == {"y": True}


@pytest.mark.parametrize("value", [False, 0, "", [], ["Y"], "Y"])
@pytest.mark.parametrize("key", ["boolean_tokens", "unit_scales"])
def test_a_falsy_or_malformed_configuration_object_is_refused(key, value):
    body = {"reviewed": True, "subsystem": "door", "columns": {"ts": "a"}, key: value}
    with pytest.raises(MappingError, match="must be a JSON object"):
        parse_mapping(body)


@pytest.mark.parametrize("key", ["boolean_tokens", "unit_scales"])
def test_null_and_absent_still_mean_not_set(key):
    assert parse_mapping({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"},
                          key: None}) is not None
    assert parse_mapping({"reviewed": True, "subsystem": "door", "columns": {"ts": "a"}}) \
        is not None


# --- 5 · the sidecar is bound to the artifact -------------------------------------

def readiness(path):
    check = subprocess.run([sys.executable, "-B", str(READINESS), str(path)],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert check.returncode == 0, check.stderr
    return json.loads(check.stdout), check.stderr


def test_a_matching_sidecar_is_verified(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    body = json.loads(vendor["out"].with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert len(body["output"]["sha256"]) == 64
    assert body["output"]["path"] == vendor["out"].name

    report, _ = readiness(vendor["out"])
    assert report["source_provenance"]["verified"] is True
    assert report["source_provenance"]["complete_input"] is True


def test_a_modified_parquet_makes_its_sidecar_unverified(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    frame = pd.read_parquet(vendor["out"])
    frame.loc[frame.index[0], "peak_current_a"] = 99.0
    frame.to_parquet(vendor["out"], index=False)

    report, stderr = readiness(vendor["out"])
    block = report["source_provenance"]
    assert block["verified"] is False
    assert any("output hash mismatch" in p for p in block["problems"])
    assert "complete_input" not in block, "an unverified claim was reported as fact"
    assert block["unverified_claims"]["complete_input"] is True
    assert "UNVERIFIED PROVENANCE" in stderr


def test_a_swapped_sidecar_is_unverified(vendor):
    """A sidecar from another export, placed beside this one."""
    other = vendor["dir"] / "other.parquet"
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"], "--out", other,
                    "--rows", "50").returncode == 0
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0

    swapped = other.with_suffix(".provenance.json").read_text(encoding="utf-8")
    vendor["out"].with_suffix(".provenance.json").write_text(swapped, encoding="utf-8")

    report, stderr = readiness(vendor["out"])
    block = report["source_provenance"]
    assert block["verified"] is False
    assert any("output hash mismatch" in p or "names" in p for p in block["problems"])
    # The swapped sidecar claims a sample; that claim must not be taken as fact
    # about this complete export, in either direction.
    assert "complete_input" not in block
    assert block["unverified_claims"]["complete_input"] is False
    assert "UNVERIFIED PROVENANCE" in stderr


def test_a_malformed_sidecar_is_unverified(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    vendor["out"].with_suffix(".provenance.json").write_text("{not json", encoding="utf-8")
    report, stderr = readiness(vendor["out"])
    assert report["source_provenance"]["verified"] is False
    assert "not readable JSON" in report["source_provenance"]["note"]
    assert "UNVERIFIED PROVENANCE" in stderr


def test_a_missing_sidecar_is_unverified(vendor):
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    vendor["out"].with_suffix(".provenance.json").unlink()
    report, stderr = readiness(vendor["out"])
    block = report["source_provenance"]
    assert block == {"available": False, "verified": False, "note": block["note"]}
    assert "unknown" in block["note"]
    assert "UNVERIFIED PROVENANCE" in stderr


def test_a_structurally_malformed_sidecar_is_unverified_through_the_cli(vendor):
    """The reported traceback case, end to end: a list where an object belongs."""
    assert run_bind(vendor["source"], "--mapping", vendor["mapping"],
                    "--out", vendor["out"]).returncode == 0
    vendor["out"].with_suffix(".provenance.json").write_text(
        json.dumps({"input": ["bad"], "output": {}}), encoding="utf-8")
    report, stderr = readiness(vendor["out"])
    assert "Traceback" not in stderr
    assert report["source_provenance"]["verified"] is False
    assert "UNVERIFIED PROVENANCE" in stderr
