"""Inspect mapped Parquet telemetry; report blockers without fitting models.

If the input was produced by `scripts/bind_data.py --out`, its sidecar
provenance is surfaced alongside the report - in particular whether the export
was a SAMPLE. A readiness verdict on the first 50,000 rows is a verdict on the
first 50,000 rows, and must not be quoted as one on the dataset.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.data_readiness import assess_readiness


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


SHA256_HEX = re.compile(r"[0-9a-f]{64}")
REQUIRED_SECTIONS = ("input", "output", "mapping", "attestations")


def _get(obj, key):
    """A field of an object, or None when the container is not an object at all."""
    return obj.get(key) if isinstance(obj, dict) else None


def _is_sha256(value):
    return isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None


def _reject_constant(name):
    # json.loads accepts NaN and Infinity by default; a sidecar carrying them is
    # malformed, and would otherwise crash the report's allow_nan=False output.
    raise ValueError(f"non-standard JSON constant {name}")


def sidecar_problems(body, parquet):
    """Every reason this sidecar cannot be trusted for `parquet`. Empty = verified.

    Types are checked before anything is read from a nested field, so a malformed
    sidecar yields reasons rather than a traceback. The checks follow the schema
    `scripts/bind_data.py` writes:

      output   a non-empty filename equal to this Parquet's, and a lowercase-hex
               SHA-256 equal to this Parquet's contents. Both are required: a hash
               alone does not show the sidecar was written for this file name,
               and a name alone shows nothing about its contents.
      input    `complete_input` a real boolean - the string "false" is not false,
               and accepting it would let a sampled export skip its warning;
               `rows_read` null for a complete export and a non-negative integer
               for a sample; `sha256` a valid hash.
      mapping  `canonical_sha256` a valid hash.
      top      `context_supported` boolean or null; `unit_conversions` null or an
               object of finite positive numbers; `attestations` an object.
    """
    if not isinstance(body, dict):
        return [f"the sidecar must be a JSON object, got {type(body).__name__}"]
    problems = []
    for key in REQUIRED_SECTIONS:
        if key not in body or body[key] is None:
            problems.append(f"missing required section {key!r}")
        elif not isinstance(body[key], dict):
            problems.append(f"section {key!r} must be an object, got {type(body[key]).__name__}")

    output = body.get("output")
    if isinstance(output, dict):
        name = output.get("path")
        if not isinstance(name, str) or not name.strip():
            problems.append("output.path must be a non-empty filename; without it the sidecar "
                            "cannot be shown to have been written for this file")
        elif name != Path(parquet).name:
            problems.append(f"output.path names {name!r}, not {Path(parquet).name!r}; this "
                            "sidecar belongs to another export")
        digest = output.get("sha256")
        if not _is_sha256(digest):
            problems.append("output.sha256 must be a 64-character lowercase hex SHA-256, got "
                            f"{digest!r}")
        elif not Path(parquet).exists():
            problems.append(f"{Path(parquet).name} does not exist, so its hash cannot be checked")
        elif digest != sha256_file(parquet):
            problems.append(f"output hash mismatch: the sidecar describes a file hashing to "
                            f"{digest[:16]}..., and this file hashes differently. Either the "
                            "Parquet changed after export, or this sidecar belongs to another "
                            "export.")

    given = body.get("input")
    if isinstance(given, dict):
        complete = given.get("complete_input")
        rows = given.get("rows_read")
        if not isinstance(complete, bool):
            problems.append("input.complete_input must be true or false, got "
                            f"{complete!r}; whether this export is a sample cannot be established")
        if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int) or rows < 0):
            problems.append(f"input.rows_read must be a non-negative integer or null, got {rows!r}")
        elif complete is False and rows is None:
            problems.append("input.complete_input is false but input.rows_read is null; the "
                            "sample size is unrecorded")
        elif complete is True and rows is not None:
            problems.append("input.complete_input is true but input.rows_read records a sample "
                            "size; the two claims contradict each other")
        if not _is_sha256(given.get("sha256")):
            problems.append("input.sha256 must be a 64-character lowercase hex SHA-256, got "
                            f"{given.get('sha256')!r}")

    mapping = body.get("mapping")
    if isinstance(mapping, dict) and not _is_sha256(mapping.get("canonical_sha256")):
        problems.append("mapping.canonical_sha256 must be a 64-character lowercase hex SHA-256, "
                        f"got {mapping.get('canonical_sha256')!r}")

    context = body.get("context_supported")
    if context is not None and not isinstance(context, bool):
        problems.append(f"context_supported must be true, false or null, got {context!r}")

    conversions = body.get("unit_conversions")
    if conversions is not None:
        if not isinstance(conversions, dict):
            problems.append("unit_conversions must be an object or null, got "
                            f"{type(conversions).__name__}")
        else:
            for column, scale in conversions.items():
                if isinstance(scale, bool) or not isinstance(scale, (int, float)) \
                        or not math.isfinite(scale) or scale <= 0:
                    problems.append(f"unit_conversions[{column!r}] must be a finite positive "
                                    f"number, got {scale!r}")
    return problems


def _claims(body):
    """The sidecar's assertions, extracted without assuming any of its types."""
    given, mapping = _get(body, "input"), _get(body, "mapping")
    return {"input_sha256": _get(given, "sha256"),
            "mapping_canonical_sha256": _get(mapping, "canonical_sha256"),
            "complete_input": _get(given, "complete_input"),
            "rows_read": _get(given, "rows_read"),
            "context_supported": _get(body, "context_supported"),
            "unit_conversions": _get(body, "unit_conversions"),
            "attestations_recorded_at_export": _get(body, "attestations")}


def source_provenance(path):
    """Read the sidecar beside a canonical export, and trust it only if it checks out.

    A sidecar can be missing, unreadable, malformed, incomplete, left over from a
    different export, or accurate about a Parquet that has since been replaced.
    In every one of those cases its sampling and conversion claims are reported
    under `unverified_claims`, with the reasons, and never as established fact.
    """
    path = Path(path)
    sidecar = path.with_suffix(".provenance.json")
    if not sidecar.exists():
        return {"available": False, "verified": False,
                "note": ("no provenance sidecar beside this file: its origin, whether it is "
                         "a sample, and any unit conversions are unrecorded and unknown")}
    try:
        body = json.loads(sidecar.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"available": True, "verified": False, "file": sidecar.name,
                "problems": [f"not readable JSON: {exc}"],
                "note": f"UNVERIFIED: {sidecar.name} is not readable JSON: {exc}"}

    problems = sidecar_problems(body, path)
    claims = _claims(body)
    if problems:
        return {"available": True, "verified": False, "file": sidecar.name,
                "problems": problems,
                "note": ("UNVERIFIED: this sidecar has not been shown to describe this file. "
                         "Its sampling, conversion and context claims must not be relied on. "
                         "Re-export with scripts/bind_data.py to produce a valid one."),
                "unverified_claims": claims}

    out = {"available": True, "verified": True, "file": sidecar.name,
           "output_sha256": body["output"]["sha256"], **claims}
    # Verified implies complete_input is a real boolean, so `is False` is exact.
    if claims["complete_input"] is False:
        out["warning"] = ("SAMPLED EXPORT: this readiness verdict covers only the "
                          f"{claims['rows_read']:,} sampled rows, not the dataset.")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("daily_or_cycles", help="Mapped cycle telemetry Parquet (not daily aggregate)")
    p.add_argument("--episodes", help="Episode CSV with onset_ts and fault_ts")
    p.add_argument("--subsystem", choices=["door", "bogie"], default="door")
    p.add_argument("--units-verified", action="store_true")
    p.add_argument("--identities-verified", action="store_true")
    p.add_argument("--fault-labels-verified", action="store_true")
    args = p.parse_args()
    report = assess_readiness(pd.read_parquet(args.daily_or_cycles),
        pd.read_csv(args.episodes) if args.episodes else None, subsystem=args.subsystem,
        units_verified=args.units_verified, identities_verified=args.identities_verified,
        fault_labels_verified=args.fault_labels_verified)
    provenance = source_provenance(args.daily_or_cycles)
    report["source_provenance"] = provenance
    print(json.dumps(report, indent=2, allow_nan=False))
    if provenance.get("warning"):
        print(provenance["warning"], file=sys.stderr)
    if not provenance.get("verified"):
        print("UNVERIFIED PROVENANCE: " + provenance.get("note", ""), file=sys.stderr)


if __name__ == "__main__":
    main()
