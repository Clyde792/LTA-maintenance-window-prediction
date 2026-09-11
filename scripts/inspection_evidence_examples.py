"""Reproducible inspection-evidence examples over the labelled synthetic cases.

Writes one evidence record per case, plus an index pairing each record with the
effect that was injected. The injected label is written to the INDEX, never into
the frame the module sees, so the records are produced without it.

These are fixtures exercising the module's own rules. They demonstrate output
shape and abstention behaviour. They are not a measurement of diagnostic
accuracy, and no accuracy claim may be drawn from them.

    python scripts/inspection_evidence_examples.py [--out data/inspection_evidence]
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from headway.inspection_evidence import collect_evidence  # noqa: E402
from headway.synth.inspection_cases import (  # noqa: E402
    cases, pipeline_case, reference_variants)

OUT = ROOT / "data" / "inspection_evidence"


def summarise(record):
    """The few fields a reader checks first, without opening the whole record."""
    cross, peers, ref = (record["cross_channel"], record["peer_comparison"],
                         record["reference_comparison"])
    baselines = {}
    if ref["quantifiable"]:
        for c, v in ref["channels"].items():
            if v["status"] == "recovered":
                baselines[c] = {
                    "location_offset": v["location_offset_removed_by_adaptation"],
                    "unit": v["unit"],
                    "location_offset_in_fleet_scale": v["location_offset_in_fleet_scale"],
                    "scale_ratio_asset_over_fleet": v["scale_ratio_asset_over_fleet"]}
    return {
        "quality_basis": record["quality_basis"]["basis"],
        "shifted_channels": cross["shifted_channels"],
        "independent_families": cross["independent_families"],
        "ambiguous_channels": cross["ambiguous_channels"],
        "unknown_channels": cross["unknown_channels"],
        "peer_comparability": peers["comparability_status"],
        "comparable_peers": len(peers["peers_comparable"]),
        "shared_with_peers": peers["shared_channels"],
        "not_shared_with_peers": peers["not_shared_channels"],
        "reference_quantifiable": ref["quantifiable"],
        "reference_verified_model_identity": ref["verified_model_identity"],
        "reference_parameter_source": ref["parameter_source"],
        "reference_demonstration": (ref["demonstration"] or {}).get("status"),
        "reference_reason": ref["reason"],
        "recovered_baselines": baselines,
        "supported_explanations": [e["explanation"] for e in record["explanations"]
                                   if e["supported_by"]],
        "missing_evidence": record["missing_evidence"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="omit the HealthPipeline-built example, which costs seconds")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    every = list(cases(seed=args.seed)) + list(reference_variants(seed=args.seed))
    if not args.skip_pipeline:
        # The one case that is not handcrafted: real pipeline quality and
        # completeness columns, so the channel-level resolution is exercised
        # against what the pipeline actually emits.
        every.append(pipeline_case())

    index = []
    for case in every:
        record = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                                  reference_views=case.views, peer_ids=case.peer_ids,
                                  model_version=f"synthetic-cases-seed-{args.seed}",
                                  allow_unverified_reference_demonstration=(
                                      case.allow_unverified_demonstration))
        path = args.out / f"{case.name}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        index.append({"case": case.name, "injected_effect": case.injected,
                      "expected_reading": case.expectation, "asset_id": case.asset_id,
                      "as_of": record["as_of"], "record": path.name,
                      "evidence_hash": record["evidence_hash"], "summary": summarise(record)})
        got = index[-1]["summary"]
        print(f"{case.name:36s} -> {path.name}")
        print(f"  shifted={got['shifted_channels']} families={got['independent_families']} "
              f"unknown={got['unknown_channels']}")
        print(f"  peers={got['peer_comparability']} ({got['comparable_peers']}) "
              f"shared={got['shared_with_peers']} | quality={got['quality_basis']}")
        print(f"  reference quantifiable={got['reference_quantifiable']} "
              f"verified={got['reference_verified_model_identity']} "
              f"params={got['reference_parameter_source']}")
        print(f"  supported={got['supported_explanations']}")

    (args.out / "index.json").write_text(json.dumps({
        "seed": args.seed,
        "purpose": "Illustrative examples of the experimental inspection-evidence module.",
        "warning": ("Fixtures constructed to exercise the module's own rules. They show "
                    "output shape and abstention behaviour, not diagnostic accuracy. The "
                    "injected effect is recorded here and was never an input to the module."),
        "explanation_order": ("Explanations appear in a fixed order in every record. It is "
                              "not a ranking, and 'supported_explanations' below is a list of "
                              "what has any supporting observation at all - not a verdict."),
        "cases": index,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {len(index)} records and index.json to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
