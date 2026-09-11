"""Executable behaviour tests for the rendered inspection-evidence JavaScript.

These run the ACTUAL code from `ui/headway.html`, not a copy: the renderer region
is extracted from the built page, given a stub `DATA`, and exercised under Node.
Markup-presence assertions cannot establish that an unknown channel is worded
correctly or that one door's record never reaches another door; these can.

Skipped where Node is unavailable. When that happens the skip is not a pass, and
must be reported as a check that did not run.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "ui" / "headway.html"

START = "const INSP ="
END = "const matches ="

# Everything the extracted region depends on and does not define itself.
STUBS = r"""
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const when = t => 'WHEN(' + String(t) + ')';
const whenDate = t => 'DATE(' + String(t) + ')';
function assert(cond, msg) { if (!cond) { throw new Error(msg); } }
"""


def node():
    exe = shutil.which("node")
    if not exe:
        pytest.skip("Node is unavailable in this environment; this check did NOT run")
    return exe


def region():
    if not PAGE.exists():
        pytest.skip("dashboard not built")
    html = PAGE.read_text(encoding="utf-8")
    return html[html.index(START):html.index(END)]


def record(**over):
    """A minimally complete export record, shaped like the real one."""
    channels = ["current_integral_as", "cycle_duration_s", "peak_current_a",
                "mean_current_a", "travel_mm"]
    base = {
        "assetId": "TRN001-DOOR-1", "asOf": "2026-07-29T16:00:00+00:00", "visibleFrom": 10,
        "sourceArtifact": "sha256:abc123", "source": "door_deferral.parquet",
        "modelVersion": None, "modelVersionReason": "no model identity available",
        "evidenceHash": "deadbeefdeadbeef", "qualityBasis": "channel_completeness",
        "qualityBlocking": [], "window": {"reference_start": "r0", "recent_end": "r1"},
        "changes": {c: {"unit": "A", "status": "unknown",
                        "reason": "insufficient supported recent days",
                        "direction": "unknown", "change": None, "sigma": None,
                        "referenceDays": 0, "recentDays": 0, "referenceMedian": None,
                        "recentMedian": None, "referenceScale": None,
                        "qualityBasis": "channel_completeness"} for c in channels},
        "cross": {"shifted": [], "unchanged": [], "unknown": channels, "families": [],
                  "ambiguous": [], "singleShifted": False, "isolated": False,
                  "wearConsistent": False},
        "peers": {"status": "unknown", "reason": "only 1 comparable peers; 3 required",
                  "comparable": 1, "considered": 5, "shared": [], "notShared": [],
                  "unknown": channels,
                  "channels": {c: {"status": "unknown", "reason": "unknown",
                                   "peers": 0, "median": None, "shared": None}
                               for c in channels}},
        "reference": {"quantifiable": False, "reason": "no reference views supplied",
                      "verifiedModelIdentity": False, "parameterSource": None,
                      "demonstration": None, "channels": {}},
        "observations": [], "explanations": [
            {"explanation": "cannot_distinguish", "statement": "Nothing separates these.",
             "supported_by": [], "contradicted_by": []}],
        "missingEvidence": ["peer comparison: too few peers"],
        "explanationOrder": "fixed and unranked", "suggestionsStatus": "illustrative",
        "safeguards": {"identifies_root_cause": False}, "limitation": "Experimental",
    }
    base.update(over)
    return base


def observed(channel, **over):
    base = {"unit": "A", "status": "observed", "reason": None, "direction": "no_detected_change",
            "change": 0.01, "sigma": 0.3, "referenceDays": 21, "recentDays": 5,
            "referenceMedian": 0.0, "recentMedian": 0.01, "referenceScale": 0.08,
            "qualityBasis": "channel_completeness"}
    base.update(over)
    return base


def run(records, body, *, days=None, unavailable=None, unavailable_doors=None):
    days = days or [f"2026-07-{d:02d}" for d in range(1, 32)]
    data = {"signalNames": {"current_integral_as": "Motor charge per cycle",
                            "cycle_duration_s": "Cycle time",
                            "peak_current_a": "Peak current",
                            "mean_current_a": "Average current",
                            "travel_mm": "Door travel"},
            "days": days,
            "inspection": unavailable or {"records": records, "staleAfterDays": 7,
                                          "suggestions": {"cannot_distinguish": ["Keep watching"]},
                                          "exportedAt": "2026-07-30T00:00:00+08:00",
                                          "source": "door_deferral.parquet",
                                          "unavailableDoors": unavailable_doors or {},
                                          "doorProblems": [],
                                          "unavailable": False}}
    script = ("const DATA = " + json.dumps(data) + ";\n" + STUBS + region() + "\n" + body)
    proc = subprocess.run([node()], input=script, text=True, encoding="utf-8",
                          capture_output=True, timeout=60)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.strip()[-2000:])
    return proc.stdout


# --- the region must at least parse and load -----------------------------------

def test_the_rendered_region_is_valid_javascript():
    proc = subprocess.run([node(), "--check"], input="const DATA={signalNames:{},days:[]};\n"
                          + STUBS + region(), text=True, encoding="utf-8", capture_output=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr


# --- unknown-state wording ------------------------------------------------------

def test_all_channels_unknown_reports_insufficient_evidence():
    run([record()], """
      const out = inspectionBlock('TRN001-DOOR-1', 10);
      assert(out.includes('Insufficient evidence to assess change.'), 'missing insufficient-evidence wording');
      assert(!out.includes('No change beyond the reporting threshold'), 'claimed a no-change finding with nothing assessed');
      assert(!out.includes('Assessed and steady'), 'called unassessed channels steady');
      assert(out.includes('not assessable'), 'unknown channels not labelled not assessable');
      assert(!out.includes('not measured'), 'used "not measured" for a failed-quality reading');
    """)


def test_mixed_coverage_limits_the_no_change_statement_to_assessed_channels():
    r = record()
    r["changes"]["peak_current_a"] = observed("peak_current_a")
    r["changes"]["travel_mm"] = observed("travel_mm")
    r["cross"] = dict(r["cross"], unchanged=["peak_current_a", "travel_mm"],
                      unknown=["current_integral_as", "cycle_duration_s", "mean_current_a"])
    run([r], """
      const out = inspectionBlock('TRN001-DOOR-1', 10);
      assert(out.includes('No change beyond the reporting threshold in the 2 channels'),
             'no-change statement not limited to the assessed channels');
      assert(out.includes('3 not assessable'), 'did not say how many channels were unassessable');
      assert(out.includes('Assessed and steady: Peak current, Door travel'),
             'steady list is not the assessed-and-unchanged channels');
      assert(!out.includes('Insufficient evidence to assess change.'),
             'claimed no evidence while two channels were assessed');
    """)


def test_a_shifted_channel_is_reported_in_physical_units():
    r = record()
    r["changes"]["peak_current_a"] = observed("peak_current_a", direction="increase",
                                              change=0.6687, sigma=6.1)
    r["cross"] = dict(r["cross"], shifted=["peak_current_a"],
                      unknown=["current_integral_as", "cycle_duration_s",
                               "mean_current_a", "travel_mm"])
    run([r], """
      const out = inspectionBlock('TRN001-DOOR-1', 10);
      assert(out.includes('Peak current'), 'plain channel name missing');
      assert(out.includes('+0.6687 A'), 'physical units missing');
      assert(out.includes('higher than its own recent history'), 'direction missing');
      assert(!out.includes('peak_current_a'), 'raw column name leaked into the page');
    """)


# --- asset and time matching ----------------------------------------------------

def test_a_record_never_renders_on_another_door():
    run([record(assetId="TRN001-DOOR-1", evidenceHash="AAAA1111")], """
      const mine = inspectionBlock('TRN001-DOOR-1', 10);
      const other = inspectionBlock('TRN002-DOOR-9', 10);
      assert(mine.includes('TRN001-DOOR-1'), 'own record did not render');
      assert(other.includes('No inspection evidence for this door'), 'no explicit empty state');
      assert(!other.includes('AAAA1111'), 'another door\\'s evidence leaked');
      assert(!other.includes('Assessed WHEN('), 'another door showed an assessment time');
    """)


def test_evidence_is_invisible_before_the_night_it_landed():
    run([record(visibleFrom=10)], """
      const before = inspectionBlock('TRN001-DOOR-1', 9);
      const at = inspectionBlock('TRN001-DOOR-1', 10);
      assert(before.includes('No inspection evidence had been produced'), 'shown before it existed');
      assert(!before.includes('Assessed WHEN('), 'assessment time shown before it existed');
      assert(at.includes('current for this date'), 'not marked current on its own night');
    """)


def test_an_older_assessment_is_labelled_not_recomputed_then_stale():
    run([record(visibleFrom=10)], """
      const later = inspectionBlock('TRN001-DOOR-1', 13);
      const old = inspectionBlock('TRN001-DOOR-1', 22);
      assert(later.includes('not recomputed for 2026-07-14 — 3 days later'), 'gap wording wrong: ' + later.slice(0, 400));
      assert(!later.includes('current for this date'), 'stale assessment claimed as current');
      assert(old.includes('stale — 12 days old'), 'beyond the staleness policy but not labelled stale');
    """)


def test_the_most_recent_visible_record_is_the_one_shown():
    run([record(visibleFrom=10, evidenceHash="OLDER000"),
         record(visibleFrom=20, evidenceHash="NEWER000")], """
      const mid = inspectionBlock('TRN001-DOOR-1', 15);
      const late = inspectionBlock('TRN001-DOOR-1', 25);
      assert(mid.includes('OLDER000') && !mid.includes('NEWER000'), 'used a future record');
      assert(late.includes('NEWER000') && !late.includes('OLDER000'), 'kept an older record once superseded');
    """)


# --- source mismatch ------------------------------------------------------------

def test_the_unavailable_state_replaces_all_evidence():
    run([], """
      const out = inspectionBlock('TRN001-DOOR-1', 10);
      assert(out.includes('Unavailable — rebuild required.'), 'no rebuild-required state');
      assert(out.includes('different telemetry'), 'reason not shown');
      assert(out.includes('record for'), 'per-record problem not listed');
      assert(!out.includes('What changed'), 'rendered evidence sections anyway');
    """, unavailable={"unavailable": True,
                      "reason": "the exported evidence was computed from different telemetry",
                      "problems": ["source artifact hash differs",
                                   "record for 'TRN001-DOOR-1' does not carry this export's identity"],
                      "records": [], "staleAfterDays": 0, "suggestions": {}})


# --- escaping -------------------------------------------------------------------

def test_hostile_record_text_cannot_inject_markup():
    hostile = "<img src=x onerror=alert(1)><script>alert(2)</script>"
    r = record()
    r["peers"] = dict(r["peers"], reason=hostile)
    r["missingEvidence"] = [hostile]
    run([r], """
      const out = inspectionBlock('TRN001-DOOR-1', 10);
      assert(!out.includes('<script>'), 'script tag survived escaping');
      assert(!out.includes('<img src=x'), 'img tag survived escaping');
      assert(out.includes('&lt;script&gt;'), 'hostile text was dropped instead of escaped');
    """)


# --- per-door unavailable rendering ---------------------------------------------

def test_an_unassessable_door_shows_its_own_reason_while_others_keep_evidence():
    """One door with no possible assessment, one with evidence, in one payload."""
    run([record(assetId="DOOR-GOOD", visibleFrom=5, evidenceHash="GOODHASH")], """
      const good = inspectionBlock('DOOR-GOOD', 10);
      const never = inspectionBlock('DOOR-NEVER', 10);
      assert(good.includes('GOODHASH'), 'the assessable door lost its evidence');
      assert(good.includes('Assessed WHEN('), 'the assessable door lost its assessment time');
      assert(never.includes('No assessment available for this door.'), 'no per-door state');
      assert(never.includes('supported daily aggregate'), 'the specific reason is not shown');
      assert(!never.includes('GOODHASH'), "another door's evidence leaked");
      assert(!never.includes('Evidence is exported only for doors'),
             'an unassessable door was described as simply not exported');
    """, unavailable_doors={"DOOR-NEVER": {
        "reasonCode": "no_supported_assessment",
        "reason": "No day in the replay produced a supported daily aggregate for this door, "
                  "so no change could be assessed."}})


def test_a_door_nobody_exported_still_reads_differently():
    run([record(assetId="DOOR-GOOD", visibleFrom=5)], """
      const quiet = inspectionBlock('DOOR-QUIET', 10);
      assert(quiet.includes('Evidence is exported only for doors this dashboard surfaces'),
             'a never-surfaced door lost its distinct wording');
      assert(!quiet.includes('No assessment available for this door.'),
             'a never-surfaced door claimed a verified unavailable reason');
    """, unavailable_doors={"DOOR-NEVER": {"reasonCode": "no_supported_assessment",
                                           "reason": "never supported"}})


def test_an_unverified_claim_is_rendered_as_rebuild_required():
    run([], """
      const out = inspectionBlock('DOOR-BAD', 10);
      assert(out.includes('No assessment available for this door.'), 'no per-door state');
      assert(out.includes('Rebuild required.'), 'an unverified claim did not ask for a rebuild');
    """, unavailable_doors={"DOOR-BAD": {
        "reasonCode": "unverified_claim",
        "reason": "The export declared this door unassessable, but this build could not "
                  "confirm it: claims no supported assessment, but this build's telemetry "
                  "has 30 supported night(s) for this door. Rebuild required."}})
