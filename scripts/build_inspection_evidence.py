"""Export inspection evidence for the dashboard, from the dashboard's own pipeline.

The static page cannot call Python, so this reads the SAME daily artifact the
dashboard is built from - `data/door_deferral.parquet` - runs the real
`headway.inspection_evidence` backend over it, and writes a JSON snapshot that
`scripts/build_ui.py` inlines.

Nothing here is handcrafted. The synthetic scenario fixtures in
`headway/synth/inspection_cases.py` are a separate, clearly labelled illustrative
set (`data/inspection_evidence/`) and are never attached to a fleet door.

EXPORT STRATEGY - why this is bounded in both runtime and payload:

  COVERAGE      Every surfaced door is explained: it gets assessment records, or
                an entry in `unavailableDoors` carrying a reason code that
                `build_ui` re-checks against the telemetry it is building from. A
                door that could never be assessed is a fact about that door, not
                a reason to withhold every other door's evidence.

  WHICH DOORS   Only doors the dashboard actually surfaces. The selection is not
                re-implemented here: `build_ui.surfaced_assets` derives the rows
                the page renders - including the unknown states the dashboard's
                own evidence handling introduces, where an unsupported day forces
                aspect to -1 and duty to not_assessed - and applies the Status
                and Review rules to them. A door that never asks for attention
                does not need an evidence page.

  WHEN          At most `PER_ASSET` assessment times per door - the first night
                it escalates and its last supported night. Recomputing on all
                ~120 replay nights would multiply runtime and payload by sixty
                for no extra information: the evidence summarises a five-day
                window against a twenty-one-day reference, so adjacent nights
                differ only marginally, and the page shows the assessment time
                rather than pretending otherwise.

  HOW MUCH      Each record is slimmed to the fields the page renders and floats
                are rounded. The full records stay reproducible from this script.

  REPLAY CLOCK  Every record carries `visibleFrom`, the first replay night at
                which it could have existed (`day + 1 day >= as_of`). The page
                must not show a record before that, must not show one door's
                record on another door, and must label an older assessment as
                not recomputed rather than presenting it as current.

IDENTITY: `sourceArtifact` is a hash of the TELEMETRY the evidence was computed
from. It is not a model version, and is never labelled as one - a dataset hash
says nothing about which model was fitted. `modelVersion` is carried separately
and is null when no artifact in the build records a trained-model identity, with
the reason stated. `scripts/build_ui.py` re-checks both the export's identity and
every record's identity against the frame it is actually building the page from,
and shows an explicit rebuild-required state on any mismatch.

REFERENCE VIEWS: the demo pipeline fits one fleet model and runs no onboarding,
so there is no adapted view to compare against a frozen one. The backend
therefore reports `quantifiable: false` with its reason, and the page displays
that reason. No model identifier or inspection provenance is invented to fill
the gap; `modelVersion` is a real hash of the artifact the evidence was computed
from.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from headway.inspection_evidence import (  # noqa: E402
    EXPLANATION_ORDER, SUGGESTIONS, EvidencePolicy, collect_evidence)
# The page's own display rules, imported rather than restated.
from build_ui import source_artifact, supported_nights, surfaced_assets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = DATA / "door_deferral.parquet"
OUT = DATA / "inspection_evidence_export.json"

PER_ASSET = 2
POLICY = EvidencePolicy()
# Display policy for the page, not a model parameter: beyond this many days an
# assessment is called stale rather than merely "not recomputed".
STALE_DAYS = 7


NO_MODEL_IDENTITY = ("no artifact in this build records a trained-model identity; "
                     "sourceArtifact identifies the telemetry the evidence was computed "
                     "from, not the model that produced it")


def model_provenance(source):
    """Separate the telemetry identity from the model identity, and say so.

    Nothing this pipeline writes carries a trained-model identifier, so
    `modelVersion` is null with the reason attached rather than a dataset hash
    dressed up as one.
    """
    return {"sourceArtifact": source_artifact(source), "source": Path(source).name,
            "modelVersion": None, "modelVersionReason": NO_MODEL_IDENTITY}


def assessment_times(g):
    """First escalation night and the last supported night, as available_at."""
    g = g.sort_values("day")
    if not len(supported_nights(g)):
        return []
    supported = g[g.data_quality_ok.eq(True) & g.context_supported.eq(True)]
    times = []
    escalated = supported[supported.aspect.fillna(-1).ge(2)]
    if not escalated.empty:
        times.append(pd.Timestamp(escalated.available_at.iloc[0]))
    times.append(pd.Timestamp(supported.available_at.iloc[-1]))
    seen, unique = set(), []
    for t in times:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[-PER_ASSET:]


def _num(v, digits=4):
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), digits)


def slim(record):
    """Keep exactly what the page renders, rounded. Nothing is re-derived here."""
    changes = {}
    for c, v in record["observed_changes"].items():
        changes[c] = {"unit": v["unit"], "status": v["status"], "reason": v["reason"],
                      "direction": v["direction"], "change": _num(v["change"]),
                      "sigma": _num(v["change_in_reference_sigma"], 2),
                      "referenceDays": v["reference_days_used"],
                      "recentDays": v["recent_days_used"],
                      "referenceMedian": _num(v["reference_median"]),
                      "recentMedian": _num(v["recent_median"]),
                      "referenceScale": _num(v["reference_scale"]),
                      "qualityBasis": v["quality_basis"]["basis"]}
    cross = record["cross_channel"]
    peers = record["peer_comparison"]
    peer_channels = {c: {"status": v["status"], "reason": v.get("reason"),
                         "peers": v["peers_with_observed_change"],
                         "median": _num(v["peer_median_change_in_sigma"], 2),
                         "shared": v["shared_with_peers"]}
                     for c, v in peers["channels"].items()}
    ref = record["reference_comparison"]
    ref_channels = {}
    for c, v in (ref.get("channels") or {}).items():
        ref_channels[c] = {"status": v["status"], "reason": v.get("reason"),
                           "unit": v.get("unit"),
                           "locationOffset": _num(v.get("location_offset_removed_by_adaptation")),
                           "offsetInFleetScale": _num(v.get("location_offset_in_fleet_scale"), 2),
                           "scaleRatio": _num(v.get("scale_ratio_asset_over_fleet"), 3),
                           "parameterSource": (v.get("asset_reference") or {}).get("parameter_source")}
    return {
        "assetId": record["asset_id"],
        "asOf": record["as_of"],
        "evidenceHash": record["evidence_hash"],
        "qualityBasis": record["quality_basis"]["basis"],
        "qualityBlocking": record["quality_basis"]["blocking_metadata"],
        "window": record["window"],
        "changes": changes,
        "cross": {"shifted": cross["shifted_channels"], "unchanged": cross["unchanged_channels"],
                  "unknown": cross["unknown_channels"],
                  "families": cross["independent_families"],
                  "ambiguous": cross["ambiguous_channels"],
                  "singleShifted": cross["single_shifted_channel"],
                  "isolated": cross["isolated_channel_shift"],
                  "wearConsistent": cross["all_shifts_wear_consistent"]},
        "peers": {"status": peers["comparability_status"], "reason": peers["comparability_reason"],
                  "comparable": len(peers["peers_comparable"]),
                  "considered": peers["peers_considered"],
                  "shared": peers["shared_channels"], "notShared": peers["not_shared_channels"],
                  "unknown": peers["unknown_channels"], "channels": peer_channels},
        "reference": {"quantifiable": ref["quantifiable"], "reason": ref.get("reason"),
                      "verifiedModelIdentity": ref.get("verified_model_identity", False),
                      "parameterSource": ref.get("parameter_source"),
                      "demonstration": (ref.get("demonstration") or {}).get("status"),
                      "channels": ref_channels},
        "observations": record["observations"],
        # Suggestions are static per explanation; carried once in the export's
        # `suggestions` catalogue rather than repeated in every record.
        "explanations": [{k: v for k, v in e.items() if k != "suggested_inspection_evidence"}
                         for e in record["explanations"]],
        "missingEvidence": record["missing_evidence"],
        "explanationOrder": record["explanation_order"],
        "suggestionsStatus": record["inspection_suggestions_status"],
        "safeguards": record["safeguards"],
        "limitation": record["limitation"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of doors (0 = no cap)")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.source)
    identity = model_provenance(args.source)
    assets = surfaced_assets(df)
    if args.limit:
        assets = assets[:args.limit]
    print(f"Inspection evidence from {args.source.name}: {len(assets)} of "
          f"{df.asset_id.nunique()} doors surfaced by the dashboard.")

    records, unavailable = [], []
    for asset in assets:
        times = assessment_times(df[df.asset_id == asset])
        if not times:
            # A surfaced door that could never be assessed is a fact about this
            # door, recorded with a reason code build_ui can independently check.
            # It must not remove the other doors' evidence.
            unavailable.append({
                "assetId": asset, "reasonCode": "no_supported_assessment",
                "reason": ("No day in the replay produced a supported daily aggregate for this "
                           "door, so no change could be assessed."),
                "source": identity["source"], "sourceArtifact": identity["sourceArtifact"]})
            print(f"  {asset}: no supported assessment night — recorded as unavailable",
                  flush=True)
            continue
        for as_of in times:
            record = collect_evidence(df, asset_id=asset, as_of=as_of, policy=POLICY,
                                      model_version=identity["modelVersion"])
            # Every record carries the export's identity, so build_ui can reject
            # one that arrived from a different export or a different frame.
            records.append(dict(slim(record), source=identity["source"],
                                sourceArtifact=identity["sourceArtifact"],
                                modelVersion=identity["modelVersion"],
                                modelVersionReason=identity["modelVersionReason"]))
        print(f"  {asset}: {len(times)} assessment time(s), "
              f"latest {times[-1]:%Y-%m-%d}", flush=True)

    export = {
        "exportedAt": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "source": identity["source"],
        "sourceArtifact": identity["sourceArtifact"],
        "modelVersion": identity["modelVersion"],
        "modelVersionReason": identity["modelVersionReason"],
        "staleAfterDays": STALE_DAYS,
        "explanationOrderNames": list(EXPLANATION_ORDER),
        "policy": {k: v for k, v in vars(POLICY).items() if not isinstance(v, dict)},
        "contextTolerance": POLICY.context_tolerance,
        "unavailableDoors": unavailable,
        "suggestions": SUGGESTIONS,
        "note": ("Read-only evidence computed from this build's own telemetry. It does not "
                 "identify a cause, does not change alerts, urgency or service status, and "
                 "does not authorise maintenance. Suggested evidence is illustrative and "
                 "pending operator review."),
        "records": records,
    }
    args.out.write_text(json.dumps(export, indent=2, allow_nan=False, default=str),
                        encoding="utf-8")
    size = args.out.stat().st_size
    print(f"Wrote {args.out.name}: {len(records)} record(s) for "
          f"{len(assets) - len(unavailable)} door(s), {len(unavailable)} door(s) unavailable, "
          f"{size / 1024:.0f} KiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
