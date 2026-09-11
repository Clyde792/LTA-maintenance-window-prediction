"""Check generated data/HTML consistency without browser access."""
import argparse,json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))
from headway.deferral import compare_window,horizon_key
import build_ui

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-js",action="store_true",
                    help="skip the Node-dependent JavaScript syntax and formatter probes. "
                         "Use ONLY where Node is unavailable, and say so when reporting.")
    args=ap.parse_args(argv)
    d=pd.read_parquet(ROOT/"data/door_deferral.parquet")
    assert not any(c.startswith("risk_") or c=="safe_days_10pct" for c in d)
    unknown=~d.prediction_state.isin(["valid","threshold_exceeded","no_worsening_trend"])
    assert not d.loc[unknown,"aspect"].eq(0).any(), "unknown evidence displayed as green"
    assert d.loc[unknown,"rul_lower"].isna().all()
    assert d.loc[unknown,"duty"].eq("not_assessed").all(), "unknown evidence received a duty assessment"
    assert d.available_at.eq(d.day+pd.Timedelta(days=1)).all()
    for h in (0.,3.,7.,14.,28.):
        expected=[compare_window(m,s,h) for m,s in zip(d.rul_lower,d.prediction_state)]
        assert d[horizon_key(h)].tolist()==expected
    assert set(d.duty.unique())<={"full_service","off_peak_only","withdraw","not_assessed"}
    assert not (d.duty.eq("withdraw")&d.prediction_state.ne("threshold_exceeded")).any(), "withdraw duty without threshold exceeded"
    assert not (d.duty.eq("off_peak_only")&~d.load_sensitive).any(), "restriction without measured load sensitivity"
    html=(ROOT/"ui/headway.html").read_text(encoding="utf-8")
    payload=json.loads(re.search(r'<script type="application/json" id="payload">(.*?)</script>',html,re.S).group(1))
    assert len(payload["assets"])==d.asset_id.nunique()
    assert 'data-theme="light"' in html
    assert "riskCeiling" not in html and "Latest date under 10% risk" not in html
    assert "Estimated lower margin" in html and "simulated" in payload["scope"].lower()
    # Repair verification is EVIDENCE, exported at build time. It must never be
    # shown before its own as_of, never claim service release, and never default
    # a door with no assessment to a favourable reading.
    v=payload.get("verification")
    if v:
        assert "refresh requires rebuild" in html, "verification snapshot is not labelled stale-able"
        assert "does not authorise return to service" in html
        assert "Not assessed" in html, "missing default for doors without an assessment"
        days=payload["days"]
        for a_ in v["assessments"]:
            assert a_["release_to_service"] is False, "verification claimed service release"
            assert a_["visibleFrom"] is None or (
                pd.Timestamp(days[a_["visibleFrom"]])+pd.Timedelta(days=1)
                >= pd.Timestamp(a_["as_of"]).tz_convert("Asia/Singapore").tz_localize(None)
            ), "assessment would be shown before it existed"
            assert a_["job_id"] and a_["asset_id"], "assessment is not joinable to a job and asset"
            assert a_["job_id"] in v["records"], "assessment has no maintenance record"
        assert {x["status"] for x in v["assessments"]} <= {
            "signal_recovered","abnormality_persists","insufficient_evidence"}
    # Inspection evidence is READ-ONLY build-time evidence from this build's own
    # telemetry. It must be joined to a door by asset id, must respect the replay
    # clock, must never present baseline metadata as a measured effect, and must
    # never carry an operational authorisation.
    insp=payload.get("inspection")
    if insp:
        known={a["id"] for a in payload["assets"]}
        days=payload["days"]
        order=insp["explanationOrderNames"]
        assert "Inspection evidence" in html
        assert "does not identify a cause" in html
        assert "refresh requires rebuild" in html
        assert "Cannot distinguish" in html, "the cannot-distinguish option is not rendered"
        assert insp.get("unavailable") is False, "the page was built with unusable evidence"
        # Every surfaced door is explained: records, or a verified reason. An
        # unassessable door must not silently take the others' evidence with it.
        surfaced=set(build_ui.surfaced_assets(d))
        covered={r["assetId"] for r in insp["records"]}
        explained=set(insp["unavailableDoors"])
        assert covered|explained==surfaced, (
            f"coverage gap: unexplained {sorted(surfaced-covered-explained)}, "
            f"unexpected {sorted((covered|explained)-surfaced)}")
        assert not covered&explained, "a door is both assessed and declared unassessable"
        assert insp["doorProblems"]==[], f"per-door problems remain: {insp['doorProblems']}"
        for door,entry in insp["unavailableDoors"].items():
            assert entry["reasonCode"] in build_ui.UNAVAILABLE_REASONS,                 f"{door}: unverified reason code {entry['reasonCode']!r} reached the page"
            assert entry["reason"].strip(), f"{door}: no reason to show"
            assert not len(build_ui.supported_nights(d[d.asset_id==door])),                 f"{door}: declared unassessable but the telemetry has supported nights"
        assert insp["modelVersion"] is None or insp["modelVersionReason"] is None,             "model provenance carries both a value and a why-absent reason"
        assert "not of a trained model" in html, "the telemetry hash is not labelled as such"
        assert "most likely" not in html.lower(), "a likelihood ranking leaked into the page"
        for r in insp["records"]:
            assert r["assetId"] in known, "evidence for a door that is not in the payload"
            assert r["visibleFrom"] is None or (
                pd.Timestamp(days[r["visibleFrom"]])+pd.Timedelta(days=1)
                >= pd.Timestamp(r["asOf"]).tz_convert("Asia/Singapore").tz_localize(None)
            ), "inspection evidence would be shown before it existed"
            assert all(v is False for v in r["safeguards"].values()), "evidence claimed authority"
            assert [e["explanation"] for e in r["explanations"]]==order, "explanation order changed"
            assert not any("suggested_inspection_evidence" in e for e in r["explanations"])
            # A dataset hash identifies the telemetry, not the model. Both are
            # carried, separately, and neither stands in for the other.
            assert r["sourceArtifact"]==insp["sourceArtifact"], "record is from another export"
            assert r["source"]==insp["source"], "record names another source artifact"
            assert r["sourceArtifact"].startswith("sha256:"), "source identity is not a hash"
            assert r["modelVersion"] is None or not r["modelVersion"].startswith("sha256:"),                 "a telemetry hash is being presented as a model version"
            for c,v in r["reference"]["channels"].items():
                if v["status"]!="recovered":
                    assert v["locationOffset"] is None and v["scaleRatio"] is None, \
                        "an unquantified reference reported a measured adaptation effect"
            for c,v in r["peers"]["channels"].items():
                if v["status"]!="observed":
                    assert v["shared"] is None, "unknown peer evidence reported as not shared"
        # The scenario fixtures are illustrative and must not reach fleet doors.
        assert not (ROOT/"data/inspection_evidence").is_dir() or not any(
            r["assetId"].startswith("TRN1") for r in insp["records"]), \
            "handcrafted scenario assets appear in the fleet export"

    for asset in payload["assets"]:
        for row in asset["rows"]:
            assert set(row["windows"].values())<={"unknown","within_margin","exceeds_margin","no_positive_margin","threshold_exceeded"}
            assert row["duty"] in {"full_service","off_peak_only","withdraw","not_assessed"}
    node=shutil.which("node")
    if not node and args.skip_js:
        print("WARNING: Node is unavailable — the JavaScript syntax check and the "
              "lower-margin formatter probe were SKIPPED. Report them as not run.")
        print(f"Artifact checks passed (JS probes skipped): {len(d)} rows, "
              f"{len(payload['assets'])} assets.")
        return 0
    if not node:
        raise RuntimeError("Node required for generated JavaScript syntax check "
                           "(pass --skip-js only where Node is unavailable)")
    # Behaviour, not markup presence: run the page's own inspection renderer.
    if payload.get("inspection"):
        region=html[html.index("const INSP ="):html.index("const matches =")]
        stub=("const DATA={signalNames:{peak_current_a:'Peak current'},days:['2026-07-01','2026-07-02'],"
              "inspection:{records:[{assetId:'A',asOf:'2026-07-01T00:00:00+08:00',visibleFrom:0,"
              "sourceArtifact:'sha256:x',evidenceHash:'HASH1234',qualityBasis:'b',qualityBlocking:[],"
              "window:{reference_start:'a',recent_end:'b'},"
              "changes:{peak_current_a:{unit:'A',status:'unknown',reason:'insufficient supported recent days',"
              "direction:'unknown',change:null,sigma:null,referenceDays:0,recentDays:0,qualityBasis:'b'}},"
              "cross:{shifted:[],unchanged:[],unknown:['peak_current_a'],families:[],ambiguous:[]},"
              "peers:{status:'unknown',reason:'too few',comparable:0,considered:1,shared:[],notShared:[],"
              "unknown:[],channels:{}},"
              "reference:{quantifiable:false,reason:'none',verifiedModelIdentity:false,channels:{}},"
              "observations:[],explanations:[],missingEvidence:[],suggestionsStatus:'illustrative',"
              "safeguards:{},limitation:'Experimental'}],staleAfterDays:7,suggestions:{},"
              "exportedAt:'2026-07-01T00:00:00+08:00',source:'s',unavailable:false}};\n"
              "const esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));\n"
              "const when=t=>String(t);const whenDate=t=>String(t);\n")
        probe=stub+region+"""
        const mine=inspectionBlock('A',0), other=inspectionBlock('B',0);
        if(!mine.includes('Insufficient evidence to assess change.')) throw new Error('unknown state mis-worded');
        if(mine.includes('not measured')) throw new Error('"not measured" used for a failed-quality reading');
        if(!mine.includes('not assessable')) throw new Error('unknown channel not labelled not assessable');
        if(!other.includes('No inspection evidence for this door')) throw new Error('no explicit empty state');
        if(other.includes('HASH1234')) throw new Error("another door's evidence leaked");
        """
        subprocess.run([node],input=probe,text=True,encoding="utf-8",
                       check=True,capture_output=True,timeout=30)
    scripts=re.findall(r'<script>(.*?)</script>',html,re.S)
    assert scripts
    for script in scripts:
        subprocess.run([node,"--check"],input=script,text=True,encoding="utf-8",
                       check=True,capture_output=True,timeout=30)
    # Exercise the actual rendered card expression, not a second formatter.
    expression=re.search(r"const safe=(.*?);", "\n".join(scripts)).group(1)
    probe="const format=(r)=>" + expression + ";\n" + """
    for (const margin of [1.96, 2.99, 10.09]) {
      const shown=Number(format({margin}).match(/[0-9]+(?:\\.[0-9]+)?/)[0]);
      if (shown > margin) throw new Error('display overstates lower margin');
    }
    if (format({margin:0}) !== 'no positive margin') throw new Error('zero margin mislabelled');
    """
    subprocess.run([node],input=probe,text=True,encoding="utf-8",
                   check=True,capture_output=True,timeout=30)
    print(f"Artifact checks passed: {len(d)} rows, {len(payload['assets'])} assets, matching window states, valid JavaScript.")
    return 0

if __name__=="__main__": raise SystemExit(main())
