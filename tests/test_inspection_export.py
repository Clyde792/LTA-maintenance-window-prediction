"""Build-time inspection-evidence export and its dashboard integration.

The export is READ-ONLY evidence computed from the dashboard's own telemetry. The
things that must not happen: evidence attached to the wrong door, evidence shown
before it existed, an older assessment passed off as current, baseline metadata
rendered as a measured adaptation effect, or page text that escapes its markup.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_inspection_evidence as X  # noqa: E402
import build_ui  # noqa: E402

EXPORT = ROOT / "data" / "inspection_evidence_export.json"
PAGE = ROOT / "ui" / "headway.html"


def payload():
    if not PAGE.exists():
        pytest.skip("dashboard not built")
    html = PAGE.read_text(encoding="utf-8")
    body = re.search(r'<script type="application/json" id="payload">(.*?)</script>',
                     html, re.S).group(1)
    return json.loads(body), html


@pytest.fixture(scope="module")
def built():
    return payload()


@pytest.fixture(scope="module")
def export():
    if not EXPORT.exists():
        pytest.skip("inspection evidence has not been exported")
    return json.loads(EXPORT.read_text(encoding="utf-8"))


def frame(n_assets=3, n_days=40, aspect=0, duty="full_service"):
    """A deferral-shaped frame complete enough for the page's display rules."""
    days = pd.date_range("2026-06-01", periods=n_days, freq="D")
    rows = []
    for i in range(n_assets):
        rows.append(pd.DataFrame({
            "asset_id": f"TRN{i:03d}-DOOR-1", "train_id": f"TRN{i:03d}", "day": days,
            "available_at": days + pd.Timedelta(days=1),
            "aspect": aspect, "raw_aspect": aspect, "duty": duty,
            "prediction_state": "valid", "data_quality_ok": True, "context_supported": True,
            "health_index_smooth": 1.0, "health_index_slope": 0.0,
            "rul_lower": 30.0, "rul_point": 40.0,
            "res_peak_current_a_completeness": 1.0, "n_cycles": 40,
            "baseline_source": "asset_reference", "decision_stability": "stable",
            "load_sensitive": False}))
    df = pd.concat(rows, ignore_index=True)
    for h in (0., 3., 7., 14., 28.):
        df[f"window_{int(h)}d"] = "within_margin"
    return df


def last_day(df, asset):
    return df.day.max()


# --- which doors get a record -------------------------------------------------
# Selection is the page's own Status/Review rule applied to the page's own
# display rows, so these exercise build_ui.surfaced_assets, which the exporter
# calls directly.

def test_a_quiet_fleet_produces_no_records():
    """Regression: aspect 0 is falsy, and once selected every door in the fleet."""
    assert build_ui.surfaced_assets(frame(aspect=0)) == []
    assert X.surfaced_assets is build_ui.surfaced_assets, "exporter re-implemented the rule"


def test_an_escalated_door_is_selected():
    df = frame(aspect=0)
    df.loc[df.asset_id == "TRN001-DOOR-1", "aspect"] = 3
    assert build_ui.surfaced_assets(df) == ["TRN001-DOOR-1"]


def test_a_door_that_only_escalated_inside_the_review_window_is_selected():
    df = frame(aspect=0)
    recent = df.day > df.day.max() - pd.Timedelta(days=5)
    df.loc[df.asset_id.eq("TRN001-DOOR-1") & recent & df.day.ne(df.day.max()), "aspect"] = 2
    assert build_ui.surfaced_assets(df) == ["TRN001-DOOR-1"]


def test_a_duty_restricted_door_is_selected():
    df = frame(aspect=0)
    df.loc[df.asset_id == "TRN002-DOOR-1", "duty"] = "off_peak_only"
    assert build_ui.surfaced_assets(df) == ["TRN002-DOOR-1"]


def test_a_door_the_dashboard_shows_as_unknown_is_selected():
    """aspect === -1 is a Status condition: unknown evidence is not quiet."""
    df = frame(aspect=0)
    last = df.day.max()
    df.loc[df.asset_id.eq("TRN000-DOOR-1") & df.day.eq(last), "aspect"] = None
    assert "TRN000-DOOR-1" in build_ui.surfaced_assets(df)


def test_unknown_states_introduced_by_the_dashboard_are_accounted_for():
    """The raw artifact says aspect 0; the page shows -1 because evidence failed.

    Selecting from the artifact would miss this door entirely.
    """
    df = frame(aspect=0)
    last = df.day.max()
    unsupported = df.asset_id.eq("TRN000-DOOR-1") & df.day.eq(last)
    df.loc[unsupported, "context_supported"] = False
    assert df.loc[unsupported, "aspect"].eq(0).all(), "the artifact still says green"

    rows, _ = build_ui.display_rows(df)
    assert rows["TRN000-DOOR-1"][-1]["aspect"] == -1, "display rules did not force unknown"
    assert rows["TRN000-DOOR-1"][-1]["duty"] == "not_assessed"
    assert "TRN000-DOOR-1" in build_ui.surfaced_assets(df)


def test_a_held_escalation_is_selected():
    """held(r): the aspect is above the raw reading, so Review keeps it."""
    df = frame(aspect=0)
    last = df.day.max()
    df.loc[df.asset_id.eq("TRN001-DOOR-1") & df.day.eq(last), ["aspect", "raw_aspect"]] = [1, 0]
    assert "TRN001-DOOR-1" in build_ui.surfaced_assets(df)


def test_the_export_covers_a_subset_of_the_fleet(export, built):
    page, _ = built
    doors = {a["id"] for a in page["assets"]}
    covered = {r["assetId"] for r in export["records"]}
    assert covered and covered < doors, "every door got a record; the filter is not filtering"


# --- asset, time and model matching -------------------------------------------

def test_every_record_names_its_door_time_and_source(export, built):
    page, _ = built
    doors = {a["id"] for a in page["assets"]}
    for r in export["records"]:
        assert r["assetId"] in doors
        assert pd.Timestamp(r["asOf"]).tzinfo is not None
        assert r["sourceArtifact"] == export["sourceArtifact"]
        assert r["source"] == export["source"]
        assert r["evidenceHash"]


def test_a_telemetry_hash_is_not_presented_as_a_model_version(export, built):
    """A dataset hash identifies the frame, not the model that produced it."""
    _, html = built
    assert export["sourceArtifact"].startswith("sha256:")
    assert export["modelVersion"] is None, "a model identity was invented"
    assert export["modelVersionReason"]
    for r in export["records"]:
        assert r["modelVersion"] is None
        assert r["sourceArtifact"] != r["modelVersion"]
    assert "source artifact " in html
    assert "not of a trained model" in html


def test_assessment_times_are_real_availability_instants(export):
    """An assessment is made when an aggregate lands, not at an arbitrary time."""
    for r in export["records"]:
        local = pd.Timestamp(r["asOf"]).tz_convert("Asia/Singapore").tz_localize(None)
        assert local == local.normalize(), "assessment time is not a replay midnight"


def test_the_source_artifact_is_the_hash_of_the_frame_in_use(export):
    source = ROOT / "data" / export["source"]
    if not source.exists():
        pytest.skip("source artifact absent")
    assert export["sourceArtifact"] == build_ui.source_artifact(source)


def test_the_exported_door_set_is_exactly_what_the_page_surfaces(export):
    source = ROOT / "data" / export["source"]
    if not source.exists():
        pytest.skip("source artifact absent")
    df = pd.read_parquet(source)
    assert {r["assetId"] for r in export["records"]} == set(build_ui.surfaced_assets(df))


# --- the replay clock ---------------------------------------------------------

def test_visible_from_never_precedes_the_assessment(built):
    page, _ = built
    if not page.get("inspection"):
        pytest.skip("no inspection payload")
    days = page["days"]
    for r in page["inspection"]["records"]:
        assert r["visibleFrom"] is not None
        landing = pd.Timestamp(days[r["visibleFrom"]]) + pd.Timedelta(days=1)
        assert landing >= pd.Timestamp(r["asOf"]).tz_convert("Asia/Singapore").tz_localize(None)


def test_the_night_before_is_too_early(built):
    page, _ = built
    if not page.get("inspection"):
        pytest.skip("no inspection payload")
    days = page["days"]
    for r in page["inspection"]["records"]:
        i = r["visibleFrom"]
        if i == 0:
            continue
        earlier = pd.Timestamp(days[i - 1]) + pd.Timedelta(days=1)
        assert earlier < pd.Timestamp(r["asOf"]).tz_convert("Asia/Singapore").tz_localize(None)


def test_a_different_replay_date_moves_the_visibility_night():
    days = pd.date_range("2026-06-01", periods=10, freq="D")
    early = build_ui._visible_from("2026-06-03T16:00:00+00:00", days)   # 4 Jun 00:00 SGT
    late = build_ui._visible_from("2026-06-08T16:00:00+00:00", days)
    assert early == 2 and late == 7
    assert build_ui._visible_from("2030-01-01T00:00:00+00:00", days) is None


def test_the_page_carries_the_clock_and_the_staleness_policy(built):
    page, html = built
    if not page.get("inspection"):
        pytest.skip("no inspection payload")
    assert page["inspection"]["staleAfterDays"] >= 1
    assert "not recomputed for" in html
    assert "stale" in html


# --- source matching, enforced by the page build -------------------------------

def written(tmp_path, export, **over):
    """Write an export variant and point build_ui at a temporary data directory."""
    body = dict(export)
    body.update(over)
    (tmp_path / "inspection_evidence_export.json").write_text(
        json.dumps(body, default=str), encoding="utf-8")
    return body


@pytest.fixture
def isolated(tmp_path, export, monkeypatch):
    """A temporary DATA dir and a temporary source artifact with a known hash."""
    source = tmp_path / "door_deferral.parquet"
    source.write_bytes(b"telemetry-A")
    monkeypatch.setattr(build_ui, "DATA", tmp_path)
    days = pd.date_range("2026-04-01", periods=120, freq="D")
    # Every exported door has supported nights here, so these tests isolate the
    # identity checks from the per-door coverage checks.
    doors = sorted({r["assetId"] for r in export["records"]})
    frame = pd.concat([pd.DataFrame({"asset_id": d, "day": days,
                                     "available_at": days + pd.Timedelta(days=1),
                                     "data_quality_ok": True, "context_supported": True})
                       for d in doors], ignore_index=True)
    return tmp_path, source, days, export, frame


def test_a_matching_export_is_accepted(isolated):
    tmp, source, days, export, frame = isolated
    doors = sorted({r["assetId"] for r in export["records"]})
    written(tmp, export, source=source.name,
            sourceArtifact=build_ui.source_artifact(source),
            records=[dict(r, source=source.name,
                          sourceArtifact=build_ui.source_artifact(source))
                     for r in export["records"]])
    out = build_ui._inspection(days, source, doors, frame)
    assert out["unavailable"] is False
    assert len(out["records"]) == len(export["records"])
    assert all(r["visibleFrom"] is not None for r in out["records"])


def test_a_different_source_artifact_is_refused(isolated):
    """The reported gap: the export must match the frame the page is built from."""
    tmp, source, days, export, frame = isolated
    doors = sorted({r["assetId"] for r in export["records"]})
    written(tmp, export, source=source.name, sourceArtifact="sha256:0000000000000000",
            records=[dict(r, source=source.name, sourceArtifact="sha256:0000000000000000")
                     for r in export["records"]])
    out = build_ui._inspection(days, source, doors, frame)
    assert out["unavailable"] is True
    assert out["records"] == []
    assert any("source artifact hash differs" in p for p in out["problems"])


def test_a_changed_artifact_invalidates_a_previously_good_export(isolated):
    """Same export, rebuilt telemetry: the evidence is about the old frame."""
    tmp, source, days, export, frame = isolated
    doors = sorted({r["assetId"] for r in export["records"]})
    stamp = build_ui.source_artifact(source)
    written(tmp, export, source=source.name, sourceArtifact=stamp,
            records=[dict(r, source=source.name, sourceArtifact=stamp)
                     for r in export["records"]])
    assert build_ui._inspection(days, source, doors, frame)["unavailable"] is False
    source.write_bytes(b"telemetry-B")
    out = build_ui._inspection(days, source, doors, frame)
    assert out["unavailable"] is True


def test_a_different_source_name_is_refused(isolated):
    tmp, source, days, export, frame = isolated
    doors = sorted({r["assetId"] for r in export["records"]})
    written(tmp, export, source="some_other.parquet",
            sourceArtifact=build_ui.source_artifact(source))
    out = build_ui._inspection(days, source, doors, frame)
    assert out["unavailable"] is True
    assert any("some_other.parquet" in p for p in out["problems"])


def test_a_record_without_the_export_identity_is_dropped(isolated):
    """A foreign record is never shown, and never condemns the rest."""
    tmp, source, days, export, frame = isolated
    stamp = build_ui.source_artifact(source)
    records = [dict(r, source=source.name, sourceArtifact=stamp) for r in export["records"]]
    foreign = dict(records[0], sourceArtifact="sha256:deadbeefdeadbeef",
                   evidenceHash="FOREIGNHASH")
    records[0] = foreign
    doors = sorted({r["assetId"] for r in records})
    written(tmp, export, source=source.name, sourceArtifact=stamp, records=records)
    out = build_ui._inspection(days, source, doors, frame)
    assert out["unavailable"] is False
    assert any("does not carry this export's identity" in p for p in out["doorProblems"])
    assert not any(r["evidenceHash"] == "FOREIGNHASH" for r in out["records"])
    assert len(out["records"]) == len(records) - 1, "other records were lost"


def test_a_door_whose_only_record_is_foreign_becomes_an_omission(isolated):
    tmp, source, days, export, frame = isolated
    stamp = build_ui.source_artifact(source)
    records = [dict(r, source=source.name, sourceArtifact=stamp) for r in export["records"]]
    target = records[0]["assetId"]
    records = [dict(r, sourceArtifact="sha256:deadbeefdeadbeef") if r["assetId"] == target else r
               for r in records]
    doors = sorted({r["assetId"] for r in records})
    written(tmp, export, source=source.name, sourceArtifact=stamp, records=records)
    out = build_ui._inspection(days, source, doors, frame)
    assert out["unavailableDoors"][target]["reasonCode"] == "unexplained_omission"
    assert target not in {r["assetId"] for r in out["records"]}
    assert {r["assetId"] for r in out["records"]} == set(doors) - {target}


def test_a_door_set_that_no_longer_matches_is_reported_per_door(isolated):
    """Unexpected doors are dropped; a surfaced door with nothing is an omission.

    Neither is build-wide: a coverage failure on one door must not withhold the
    evidence of doors that are fine.
    """
    tmp, source, days, export, frame = isolated
    stamp = build_ui.source_artifact(source)
    written(tmp, export, source=source.name, sourceArtifact=stamp,
            records=[dict(r, source=source.name, sourceArtifact=stamp)
                     for r in export["records"]])
    out = build_ui._inspection(days, source, ["TRN999-DOOR-9"], frame)
    assert out["unavailable"] is False
    assert out["records"] == [], "records for doors this build does not surface were kept"
    assert any("not surfaced by this build" in p for p in out["doorProblems"])
    assert out["unavailableDoors"]["TRN999-DOOR-9"]["reasonCode"] == "unexplained_omission"


def test_an_absent_export_is_simply_absent(isolated):
    tmp, source, days, _, frame = isolated
    assert build_ui._inspection(days, source, [], frame) is None


# --- unavailable evidence and metadata-only references ------------------------

def test_doors_without_a_record_are_stated_not_defaulted(built):
    page, html = built
    if not page.get("inspection"):
        pytest.skip("no inspection payload")
    covered = {r["assetId"] for r in page["inspection"]["records"]}
    assert {a["id"] for a in page["assets"]} - covered
    assert "No inspection evidence for this door" in html
    assert "No inspection evidence had been produced" in html


def test_the_reference_section_states_its_reason(export):
    """The demo pipeline runs no onboarding; the page must say so, not invent one."""
    for r in export["records"]:
        ref = r["reference"]
        if not ref["quantifiable"]:
            assert ref["reason"], "an unavailable reference gave no reason"
            assert ref["channels"] == {} or all(
                v["locationOffset"] is None for v in ref["channels"].values())


def test_metadata_only_references_never_carry_a_measured_effect():
    """Slimming must not turn baseline metadata into an adaptation effect."""
    record = {
        "asset_id": "A", "as_of": "2026-06-01T00:00:00+00:00", "model_version": "sha256:x",
        "evidence_hash": "h", "quality_basis": {"basis": "global_data_quality_ok",
                                                "blocking_metadata": []},
        "window": {}, "observed_changes": {}, "explanation_order": "", "missing_evidence": [],
        "observations": [], "explanations": [], "inspection_suggestions_status": "illustrative",
        "safeguards": {"identifies_root_cause": False}, "limitation": "Experimental",
        "cross_channel": {"shifted_channels": [], "unchanged_channels": [], "unknown_channels": [],
                          "independent_families": [], "ambiguous_channels": [],
                          "single_shifted_channel": False, "isolated_channel_shift": False,
                          "all_shifts_wear_consistent": False},
        "peer_comparison": {"comparability_status": "unknown", "comparability_reason": "none",
                            "peers_comparable": [], "peers_considered": 0, "shared_channels": [],
                            "not_shared_channels": [], "unknown_channels": [], "channels": {}},
        "reference_comparison": {
            "quantifiable": False, "reason": "no usable telemetry",
            "verified_model_identity": True, "parameter_source": None, "demonstration": None,
            "channels": {"peak_current_a": {
                "status": "metadata_only", "reason": "shown as metadata only",
                "unit": "A", "location_offset_removed_by_adaptation": None,
                "location_offset_in_fleet_scale": None, "scale_ratio_asset_over_fleet": None,
                "asset_reference": {"parameter_source": "exported"}}}},
    }
    out = X.slim(record)
    entry = out["reference"]["channels"]["peak_current_a"]
    assert entry["status"] == "metadata_only"
    assert entry["locationOffset"] is None and entry["scaleRatio"] is None
    assert out["reference"]["quantifiable"] is False


# --- structure the page depends on --------------------------------------------

def test_explanations_keep_their_fixed_order_and_carry_no_score(export):
    order = export["explanationOrderNames"]
    assert order[-1] == "cannot_distinguish"
    for r in export["records"]:
        assert [e["explanation"] for e in r["explanations"]] == order
        for e in r["explanations"]:
            assert set(e) == {"explanation", "statement", "supported_by", "contradicted_by"}


def test_suggestions_are_carried_once_not_per_record(export):
    assert set(export["suggestions"]) == set(export["explanationOrderNames"])
    assert all("suggested_inspection_evidence" not in e
               for r in export["records"] for e in r["explanations"])


def test_peer_evidence_keeps_shared_not_shared_and_unknown_distinct(export):
    for r in export["records"]:
        for c, v in r["peers"]["channels"].items():
            if v["status"] == "observed":
                assert v["shared"] in (True, False)
            else:
                assert v["shared"] is None, "unknown peer evidence presented as not shared"


def test_no_record_authorises_anything(export):
    for r in export["records"]:
        assert all(v is False for v in r["safeguards"].values())
        assert "illustrative" in r["suggestionsStatus"]


def test_the_payload_stays_practical(export):
    size = EXPORT.stat().st_size
    assert size < 2_000_000, f"export is {size / 1024:.0f} KiB; trim the record or the door set"
    assert len(export["records"]) <= X.PER_ASSET * len({r["assetId"] for r in export["records"]})


# --- escaping and existing behaviour ------------------------------------------

def test_payload_text_cannot_break_out_of_its_script_block():
    """Anything a record carries is inert in the page, including markup."""
    hostile = "</script><script>alert(1)</script>"
    html = build_ui.render({"days": ["2026-06-01"], "assets": [], "scope": hostile})
    body = re.search(r'<script type="application/json" id="payload">(.*?)</script>',
                     html, re.S).group(1)
    assert "</script><script>" not in body
    assert "\\u003c/script>" in body
    assert json.loads(body)["scope"] == hostile


RAW_CONCAT = re.compile(
    r"\+\s*[A-Za-z_$][\w$]*(?:\[[^\]]*\])?\."
    r"(reason|statement|observation|assetId|limitation|modelVersion|qualityBasis)")


def test_the_page_escapes_record_text_through_one_helper(built):
    """No record text is concatenated into markup without esc() or iplain()."""
    _, html = built
    region = html[html.index("const INSP ="):html.index("const matches =")]
    raw = [m.group(0) for m in RAW_CONCAT.finditer(region)]
    assert not raw, f"record text concatenated unescaped: {raw}"
    assert "function iplain(" in region and "esc(text)" in region


def test_the_section_is_read_only_and_says_so(built):
    _, html = built
    assert "Inspection evidence" in html
    assert "does not identify a cause" in html
    assert "authorise maintenance" in html
    # The sentence is built by concatenation in the source, so match its halves.
    assert "Listed in a fixed order. Neither the order nor the number of supporting" in html
    assert "points is a ranking, and no explanation here is a confirmed cause." in html
    assert "most likely" not in html.lower()
    assert "confidence" not in html.lower() or "no confidence" in html.lower()


def test_the_existing_planner_and_verification_survive(built):
    page, html = built
    assert page.get("verification"), "repair verification disappeared from the payload"
    assert page["verification"]["assessments"], "no verification assessments remain"
    assert "does not authorise return to service" in html
    for marker in ('id="v-planner"', 'id="plan-list"', 'id="pl-clear"', 'data-view="planner"'):
        assert marker in html, f"planner markup missing: {marker}"
    assert "Open follow-up from job" in html
    assert 'data-theme="light"' in html


def test_no_new_top_level_view_was_added(built):
    page, html = built
    assert html.count('data-view="') == 4, "a new top-level view was added"
    assert [m for m in re.findall(r'id="v-(\w+)"', html)] == [
        "status", "review", "planner", "fleet", "asset"]


# --- per-door coverage ---------------------------------------------------------
# One surfaced door that could never be assessed is a fact about that door. It
# must not remove every other door's evidence, and its own reason must reach the
# page.

def two_door_frame():
    """Six doors: one assessable and surfaced, one surfaced but never supported.

    Built from the same helper the module fixtures use, so the channels, context
    means and quality columns are the ones the backend actually reads.
    """
    from headway.synth.inspection_cases import base_frame

    df, as_of = base_frame(n_assets=6, n_days=40, seed=4242)
    doors = sorted(df.asset_id.unique())
    good, never = doors[0], doors[1]
    df["train_id"] = df.asset_id.str.slice(0, 6)
    df["aspect"] = 0
    df["raw_aspect"] = 0
    df["duty"] = "full_service"
    df["prediction_state"] = "valid"
    df["health_index_smooth"] = 1.0
    df["health_index_slope"] = 0.0
    df["rul_lower"] = 30.0
    df["rul_point"] = 40.0
    df["decision_stability"] = "stable"
    df["baseline_source"] = "asset_reference"
    df["load_sensitive"] = False
    for h in (0., 3., 7., 14., 28.):
        df[f"window_{int(h)}d"] = "within_margin"
    # The assessable door escalates, so Status lists it.
    df.loc[df.asset_id == good, "aspect"] = 3
    df.loc[df.asset_id == good, "raw_aspect"] = 3
    # The other never has a supported aggregate, so the page shows it as unknown
    # (aspect -1) and Status lists it too - with nothing to assess.
    df.loc[df.asset_id == never, "data_quality_ok"] = False
    return df, good, never, as_of


@pytest.fixture(scope="module")
def two_doors(tmp_path_factory):
    """Run the real exporter over that frame and load what it wrote."""
    tmp = tmp_path_factory.mktemp("coverage")
    df, good, never, _ = two_door_frame()
    source = tmp / "door_deferral.parquet"
    df.to_parquet(source)
    out = tmp / "inspection_evidence_export.json"
    assert X.main(["--source", str(source), "--out", str(out)]) == 0
    export = json.loads(out.read_text(encoding="utf-8"))
    days = pd.date_range(df.day.min(), df.day.max(), freq="D")
    return {"tmp": tmp, "df": df, "good": good, "never": never, "source": source,
            "out": out, "export": export, "days": days}


def loaded(ctx, monkeypatch, **over):
    body = dict(ctx["export"])
    body.update(over)
    (ctx["tmp"] / "inspection_evidence_export.json").write_text(
        json.dumps(body, default=str), encoding="utf-8")
    monkeypatch.setattr(build_ui, "DATA", ctx["tmp"])
    return build_ui._inspection(ctx["days"], ctx["source"],
                               build_ui.surfaced_assets(ctx["df"]), ctx["df"])


def test_both_doors_are_surfaced_but_only_one_is_assessable(two_doors):
    ctx = two_doors
    assert set(build_ui.surfaced_assets(ctx["df"])) == {ctx["good"], ctx["never"]}
    assert len(build_ui.supported_nights(ctx["df"][ctx["df"].asset_id == ctx["good"]])) > 0
    assert len(build_ui.supported_nights(ctx["df"][ctx["df"].asset_id == ctx["never"]])) == 0


def test_the_exporter_records_the_unassessable_door_with_a_reason(two_doors):
    ctx = two_doors
    export = ctx["export"]
    assert {r["assetId"] for r in export["records"]} == {ctx["good"]}
    entries = {e["assetId"]: e for e in export["unavailableDoors"]}
    assert set(entries) == {ctx["never"]}
    assert entries[ctx["never"]]["reasonCode"] == "no_supported_assessment"
    assert "supported daily aggregate" in entries[ctx["never"]]["reason"]
    assert "skipped" not in export


def test_one_unassessable_door_does_not_invalidate_the_other(two_doors, monkeypatch):
    """The reported defect: coverage ignored `skipped`, so this failed build-wide."""
    ctx = two_doors
    out = loaded(ctx, monkeypatch)
    assert out["unavailable"] is False, out.get("problems")
    assert out["doorProblems"] == []
    assert {r["assetId"] for r in out["records"]} == {ctx["good"]}
    assert out["records"], "the assessable door lost its evidence"
    assert set(out["unavailableDoors"]) == {ctx["never"]}
    assert out["unavailableDoors"][ctx["never"]]["reasonCode"] == "no_supported_assessment"


def test_an_unverifiable_claim_is_refused(two_doors, monkeypatch):
    """The exporter's word is not enough: the claim is checked against the frame."""
    ctx = two_doors
    lying = [{"assetId": ctx["good"], "reasonCode": "no_supported_assessment",
              "reason": "not really"}]
    out = loaded(ctx, monkeypatch, unavailableDoors=lying,
                 records=[r for r in ctx["export"]["records"] if r["assetId"] != ctx["good"]])
    assert out["unavailable"] is False
    assert out["records"] == []
    assert any("supported night(s) for this door" in p for p in out["doorProblems"])
    assert out["unavailableDoors"][ctx["good"]]["reasonCode"] == "unverified_claim"
    assert "could not confirm" in out["unavailableDoors"][ctx["good"]]["reason"]


def test_an_invented_reason_code_is_refused(two_doors, monkeypatch):
    ctx = two_doors
    invented = [{"assetId": ctx["never"], "reasonCode": "because_i_said_so",
                 "reason": "trust me"}]
    out = loaded(ctx, monkeypatch, unavailableDoors=invented)
    assert any("unrecognised unavailable reason code" in p for p in out["doorProblems"])
    assert out["unavailableDoors"][ctx["never"]]["reasonCode"] == "unverified_claim"
    assert {r["assetId"] for r in out["records"]} == {ctx["good"]}, "valid evidence was lost"


def test_an_unexplained_omission_is_refused(two_doors, monkeypatch):
    ctx = two_doors
    out = loaded(ctx, monkeypatch, unavailableDoors=[])
    assert any("neither assessed it nor explained why" in p for p in out["doorProblems"])
    assert out["unavailableDoors"][ctx["never"]]["reasonCode"] == "unexplained_omission"
    assert {r["assetId"] for r in out["records"]} == {ctx["good"]}


def test_conflicting_entries_are_refused(two_doors, monkeypatch):
    ctx = two_doors
    both = list(ctx["export"]["unavailableDoors"]) + [
        {"assetId": ctx["good"], "reasonCode": "no_supported_assessment", "reason": "conflict"}]
    out = loaded(ctx, monkeypatch, unavailableDoors=both)
    assert any("both assessed this door and declared it unassessable" in p
               for p in out["doorProblems"])
    assert out["unavailableDoors"][ctx["good"]]["reasonCode"] == "conflicting_entries"
    assert out["records"] == [], "a conflicted door kept its records"
    assert set(out["unavailableDoors"]) == {ctx["good"], ctx["never"]}


def test_an_unexpected_door_is_dropped_without_harming_the_rest(two_doors, monkeypatch):
    ctx = two_doors
    stranger = dict(ctx["export"]["records"][0], assetId="TRN999-DOOR-9")
    out = loaded(ctx, monkeypatch, records=ctx["export"]["records"] + [stranger])
    assert any("not surfaced by this build" in p for p in out["doorProblems"])
    assert {r["assetId"] for r in out["records"]} == {ctx["good"]}


def test_a_source_mismatch_is_still_build_wide(two_doors, monkeypatch):
    """Per-door handling must not weaken the artifact check."""
    ctx = two_doors
    out = loaded(ctx, monkeypatch, sourceArtifact="sha256:0000000000000000")
    assert out["unavailable"] is True
    assert out["records"] == [] and out["unavailableDoors"] == {}
    assert any("source artifact hash differs" in p for p in out["problems"])
