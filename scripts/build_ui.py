"""Render the dashboard from the current retrospective demo pipeline.

The page is emitted as one self-contained HTML file, but the source is not: the
shell, stylesheet and script live under scripts/ui/ so they can be read and
reviewed as ordinary files. This module only builds the payload and inlines them.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.aspect import RECOMMENDATION,AspectPolicy
from headway.deferral import DEFAULT_HORIZONS,horizon_key
from headway.evidence import evidence_status
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
UI=Path(__file__).resolve().parent/"ui"
OUT=ROOT/"ui/headway.html"

def source_artifact(path):
    """Identity of the TELEMETRY ARTIFACT, not of a trained model.

    A dataset hash says which frame a number was computed from. It does not
    identify the fitted normalisation or failure model, and must never be
    labelled as if it did.
    """
    return "sha256:"+hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

def _clean(v):
    return round(float(v),3) if v is not None and pd.notna(v) and np.isfinite(v) else None

FRIENDLY_MEANING={-1:"Check the data or inspect — no reliable countdown.",
    0:"Nothing to do. Keep monitoring.",1:"Plan a maintenance slot.",
    2:"Book the next suitable maintenance slot.",3:"Take out of service for review."}

SIGNAL_NAMES={"current_integral_as":"Motor charge per cycle","cycle_duration_s":"Cycle time",
    "peak_current_a":"Peak current","mean_current_a":"Average current","travel_mm":"Door travel"}

# Illustrative operator inputs, NOT model outputs. Nothing in the telemetry
# implies how long a repair takes, how many crews a depot fields, or when the
# engineering window opens; fault_mode only exists at the confirmed fault. These
# are depot facts to be confirmed on site, and the dashboard labels them as such.
WINDOW={"startHour":1.0,"minutes":120,"crew":1,"defaultJobMinutes":45}

def _visible_from(as_of,days):
    """First replay night at which an assessment may be shown.

    A day's aggregate only lands the following midnight, so the dashboard's clock
    on night i is days[i] + 1 day. An assessment must never appear before its own
    `as_of`, or the replay would show knowledge nobody had yet.
    """
    t=pd.Timestamp(as_of)
    t=t.tz_convert("Asia/Singapore").tz_localize(None) if t.tzinfo is not None else t
    for i,d in enumerate(days):
        if pd.Timestamp(d)+pd.Timedelta(days=1)>=t: return i
    return None

# The reference snapshot is large and already persisted in the store; the page
# carries its id and hash for provenance instead of the whole thing.
_DROP={"reference_snapshot"}

def _verification(days):
    path=DATA/"verification_export.json"
    if not path.exists(): return None
    v=json.loads(path.read_text(encoding="utf-8"))
    out=[]
    for a in v.get("assessments",[]):
        a={k:x for k,x in a.items() if k not in _DROP}
        a["visibleFrom"]=_visible_from(a["as_of"],days)
        out.append(a)
    v["assessments"]=out
    return v

def _unavailable(reason,problems):
    """A BUILD-WIDE rebuild-required state, rendered by the page.

    Reserved for failures that make the whole export untrustworthy - currently
    only a source-artifact mismatch. A single door that could not be assessed is
    a per-door fact and must not take the other doors' evidence down with it.
    """
    return {"unavailable":True,"reason":reason,"problems":problems,"records":[],
            "unavailableDoors":{},"doorProblems":[],
            "suggestions":{},"explanationOrderNames":[],"staleAfterDays":0}

# Reason codes this build knows how to CHECK. An export may not invent one: an
# unverifiable claim is a way to skip a door out of the completeness check.
UNAVAILABLE_REASONS={"no_supported_assessment"}

def _verify_unavailable(claim,frame,door):
    """Confirm an unavailable claim against the telemetry now being built.

    The exporter's word is not enough. A door is only accepted as unassessable
    if this build's own frame agrees that it never had a supported aggregate.
    """
    code=claim.get("reasonCode")
    if code not in UNAVAILABLE_REASONS:
        return False,(f"unrecognised unavailable reason code {code!r}; this build can only "
                      f"verify {sorted(UNAVAILABLE_REASONS)}")
    if not str(claim.get("reason") or "").strip():
        return False,"unavailable entry carries no reason to show"
    nights=supported_nights(frame[frame.asset_id==door])
    if len(nights):
        return False,(f"claims no supported assessment, but this build's telemetry has "
                      f"{len(nights)} supported night(s) for this door")
    return True,None

def _inspection(days,source,expected_doors,frame):
    """Inline the build-time evidence export, validated against THIS build.

    An export is only evidence about the frame it was computed from. Checked
    here, at the moment the page is assembled, rather than by a separate script
    that inspects whatever files happen to be on disk afterwards:

      * the export must name the same source artifact, by name AND by hash, as
        the frame this page is being built from;
      * every record must carry that same export identity, or it is dropped;
      * the door set must match the selection these very rules produce.

    Any mismatch yields an explicit unavailable/rebuild-required state instead
    of stale evidence presented as current.
    """
    path=DATA/"inspection_evidence_export.json"
    if not path.exists(): return None
    v=json.loads(path.read_text(encoding="utf-8"))
    actual=source_artifact(source)
    problems=[]
    if v.get("source")!=Path(source).name:
        problems.append(f"export was built from {v.get('source')!r}, this page from {Path(source).name!r}")
    if v.get("sourceArtifact")!=actual:
        problems.append(f"source artifact hash differs: export {v.get('sourceArtifact')!r}, "
                        f"this build {actual!r}")
    if problems:
        return _unavailable("the exported evidence was computed from different telemetry",problems)
    # Identity is per record: one record from another export does not condemn
    # the rest, but it is never shown.
    by_door,door_problems={},[]
    for r in v.get("records",[]):
        if r.get("source")!=v.get("source") or r.get("sourceArtifact")!=v.get("sourceArtifact"):
            door_problems.append(f"{r.get('assetId')}: a record does not carry this export's "
                                 "identity and was dropped")
            continue
        r["visibleFrom"]=_visible_from(r["asOf"],days)
        by_door.setdefault(r["assetId"],[]).append(r)

    claimed={e.get("assetId"):e for e in v.get("unavailableDoors",[]) if e.get("assetId")}
    expected=set(expected_doors)
    kept,unavailable_doors=[],{}
    # Every surfaced door must be explained: assessed, or declared unassessable
    # with a reason this build can verify. Neither is an omission.
    for door in sorted(expected):
        records,claim=by_door.get(door),claimed.get(door)
        if records and claim:
            door_problems.append(f"{door}: the export both assessed this door and declared it "
                                 "unassessable; neither entry is trusted")
            unavailable_doors[door]={"reasonCode":"conflicting_entries",
                "reason":"The export contains both assessment records and an unavailable entry "
                         "for this door, so neither was used. Rebuild required."}
        elif records:
            kept+=records
        elif claim:
            ok,why=_verify_unavailable(claim,frame,door)
            if ok:
                unavailable_doors[door]={"reasonCode":claim["reasonCode"],
                                         "reason":str(claim["reason"])}
            else:
                door_problems.append(f"{door}: {why}")
                unavailable_doors[door]={"reasonCode":"unverified_claim",
                    "reason":"The export declared this door unassessable, but this build could "
                             f"not confirm it: {why}. Rebuild required."}
        else:
            door_problems.append(f"{door}: surfaced by this build, but the export neither "
                                 "assessed it nor explained why")
            unavailable_doors[door]={"reasonCode":"unexplained_omission",
                "reason":"This door is shown by Status or Review, but the exported evidence "
                         "neither assessed it nor recorded why. Rebuild required."}
    for door in sorted((set(by_door)|set(claimed))-expected):
        door_problems.append(f"{door}: not surfaced by this build; its entries were dropped")

    v["records"]=kept
    v["unavailableDoors"]=unavailable_doors
    v["doorProblems"]=door_problems
    v["unavailable"]=False
    v.pop("skipped",None)
    return v

def _why(r):
    out=[]
    hi=r.get("health_index_smooth"); sl=r.get("health_index_slope")
    if pd.notna(hi) and np.isfinite(hi): out.append(f"Wear level {hi:.0f}")
    if pd.notna(sl) and np.isfinite(sl):
        out.append(f"rising ~{sl:.1f} a day" if sl>0.05 else f"falling ~{abs(sl):.1f} a day" if sl<-0.05 else "roughly flat now")
    return [", ".join(out)] if out else []

# Mirrors RVWIN in scripts/ui/app.js: the Review view's look-back window.
REVIEW_WINDOW=21

def _asset_rows(g,contribs):
    """The rows the page actually receives for one asset, already reindexed."""
    rows=[]
    last_aspect=-1
    last_supported=None
    for d,r in g.iterrows():
        monitoring=evidence_status(r,as_of=pd.Timestamp(d)+pd.Timedelta(days=1),last_supported=last_supported)
        last_supported=monitoring["lastSupportedAt"]
        state=r.get("prediction_state")
        if pd.isna(state): state="missing_observation"
        asp=int(r.aspect) if pd.notna(r.get("aspect")) else (last_aspect if last_aspect>0 else -1)
        last_aspect=asp
        raw=int(r.raw_aspect) if pd.notna(r.get("raw_aspect")) else -1
        if monitoring["status"] != "supported":
            asp=asp if asp>0 else -1
            raw=-1
            if state != "missing_observation":
                state="evidence_"+monitoring["status"]
        rows.append({"aspect":asp,"raw":raw,"state":state,
            "monitoring":monitoring,
            "margin":_clean(r.get("rul_lower")),"point":_clean(r.get("rul_point")),
            "hi":_clean(r.get("health_index_smooth")),"slope":_clean(r.get("health_index_slope")),
            "stability":r.get("decision_stability") if pd.notna(r.get("decision_stability")) else "n/a",
            "evidence":_why(r),"windows":{str(int(h)):(r.get(horizon_key(h)) if pd.notna(r.get(horizon_key(h))) else "unknown") for h in DEFAULT_HORIZONS},
            "baseline":r.get("baseline_source") if pd.notna(r.get("baseline_source")) else "unavailable",
            "duty":r.get("duty") if pd.notna(r.get("duty")) else "not_assessed",
            "peakHi":_clean(r.get("peak_index")),"offHi":_clean(r.get("offpeak_index")),
            "loadT":_clean(r.get("load_sensitivity_t")),"sensitive":bool(r.get("load_sensitive")) if pd.notna(r.get("load_sensitive")) else False,
            "contributions":{c.removeprefix("contribution_"):_clean(r[c]) for c in contribs}})
        if monitoring["status"] != "supported":
            rows[-1].update(margin=None,point=None,hi=None,slope=None,evidence=[],
                            duty="not_assessed",peakHi=None,offHi=None,loadT=None,sensitive=False,
                            contributions={c:None for c in rows[-1]["contributions"]},
                            windows={str(int(h)):"unknown" for h in DEFAULT_HORIZONS})
    return rows

def display_rows(df):
    """Every asset's rendered rows, keyed by asset id, plus the replay days.

    This is the ONLY derivation of what the dashboard shows. The evidence
    exporter selects doors from these rows, not from the raw artifact, so
    unknown states introduced here - an unsupported day forcing aspect to -1
    and duty to not_assessed - are accounted for by construction.
    """
    days=pd.date_range(df.day.min(),df.day.max(),freq="D")
    contribs=[c for c in df if c.startswith("contribution_")]
    return {aid:_asset_rows(g.set_index("day").reindex(days),contribs)
            for aid,g in df.groupby("asset_id",sort=True)},days

def surfaced(rows,review_window=REVIEW_WINDOW):
    """True when Status or Review would list this door. Mirrors app.js exactly:

        statusAssets: aspect >= 2 or aspect === -1 or duty === 'off_peak_only'
        reviewAssets: aspect === 1, or held (aspect > 0 and aspect > raw), or
                      aspect >= 2 / duty off_peak_only inside the window
    """
    if not rows: return False
    r=rows[-1]
    if r["aspect"]>=2 or r["aspect"]==-1 or r["duty"]=="off_peak_only": return True
    if r["aspect"]==1 or (r["aspect"]>0 and r["aspect"]>r["raw"]): return True
    return any(x["aspect"]>=2 or x["duty"]=="off_peak_only"
               for x in rows[max(0,len(rows)-review_window):])

def supported_nights(g):
    """Replay nights on which this door had a supported daily aggregate.

    The predicate behind a door being assessable at all. It lives here, beside
    the display rules, so the evidence exporter and this validator share one
    definition instead of each asserting their own.
    """
    s=g[g.data_quality_ok.eq(True)&g.context_supported.eq(True)]
    return pd.to_datetime(s.sort_values("day").available_at)

def surfaced_assets(df):
    rows,_=display_rows(df)
    return sorted(a for a,r in rows.items() if surfaced(r))

def build_payload(df,policy):
    per_asset,days=display_rows(df)
    assets=[]
    for aid,g in df.groupby("asset_id",sort=True):
        assets.append({"id":aid,"train":str(g.train_id.dropna().iloc[0]),"rows":per_asset[aid]})
    dstr=[pd.Timestamp(d).strftime("%Y-%m-%d") for d in days]
    last=pd.Timestamp(days[-1])
    threshold=_clean(df.threshold.dropna().iloc[0]) if "threshold" in df and df.threshold.notna().any() else None
    return {"days":dstr,"assets":assets,
        "rec":RECOMMENDATION,"meaning":FRIENDLY_MEANING,"signalNames":SIGNAL_NAMES,
        "thresholds":{"withdraw":policy.withdraw_days,"tonight":policy.tonight_days,"plan":policy.plan_days},
        "threshold":threshold,
        "window":WINDOW,
        # The daily aggregate for a telemetry day only lands the following midnight;
        # the header says so rather than implying the page is live.
        "replay":{"through":last.strftime("%Y-%m-%d"),
                  "availableAt":(last+pd.Timedelta(days=1)).strftime("%Y-%m-%d 00:00 SGT")},
        "verification":_verification(days),
        "inspection":_inspection(days,DATA/"door_deferral.parquet",surfaced_assets(df),df),
        "duty":{"bands":str(df.duty_bands.dropna().iloc[0]) if "duty_bands" in df and df.duty_bands.notna().any() else "no peak identified",
            "threshold":threshold},
        "scope":"Demonstration on simulated door data."}

HEAD=('<!doctype html>\n<html lang="en" data-theme="light"><head><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
      '<title>Headway — Fleet decisions</title>\n'
      '<link rel="preconnect" href="https://fonts.googleapis.com">'
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
      '<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">\n')

def render(payload):
    css=(UI/"app.css").read_text(encoding="utf-8")
    shell=(UI/"shell.html").read_text(encoding="utf-8")
    js=(UI/"app.js").read_text(encoding="utf-8")
    encoded=json.dumps(payload,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    return (HEAD+"<style>\n"+css+"</style></head><body>"+shell
            +'<script type="application/json" id="payload">'+encoded+"</script>\n"
            +"<script>\n"+js+"\n</script></body></html>")

def main():
    df=pd.read_parquet(DATA/"door_deferral.parquet")
    payload=build_payload(df,AspectPolicy())
    insp=payload.get("inspection")
    if insp and insp.get("unavailable"):
        print("Inspection evidence UNAVAILABLE — rebuild required:")
        for p in insp["problems"]: print("  - "+p)
    elif insp:
        if insp["unavailableDoors"]:
            print(f"Inspection evidence: {len(insp['unavailableDoors'])} surfaced door(s) "
                  "carry an unavailable reason instead of an assessment.")
        for p in insp["doorProblems"]: print("  - "+p)
    selection_path=DATA/"rul_selection.json"
    if selection_path.exists():
        selection=json.loads(selection_path.read_text(encoding="utf-8"))
        if not selection.get("qualified",False):
            payload["scope"] += " Timing model is a fallback; read timings as indicative."
    OUT.write_text(render(payload),encoding="utf-8")
    print(f"Dashboard refreshed: {len(payload['assets'])} doors; explicit unknown states and window comparisons.")
    return 0
if __name__=="__main__": raise SystemExit(main())
