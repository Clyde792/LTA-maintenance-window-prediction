"""
The Friday-night tool. Point it at whatever NEBULA X hand over.

    .venv/Scripts/python.exe scripts/bind_data.py <file> [--subsystem door]
    .venv/Scripts/python.exe scripts/bind_data.py --demo

It inspects the file, guesses the column mapping, applies it, runs the contract
validator, and prints a dict ready to paste into headway/adapters/nebulax.py.

The point is to convert the riskiest unknown of the whole weekend - "what shape
is their data?" - from an engineering problem into a fifteen-minute review of
about fifteen guesses. Everything downstream already speaks the contract.

TWO MODES, and the difference is who is accountable for the mapping:

  SUGGEST (default)   Guess, report, print a paste-able dict. A review aid.
                      `--suggest-out` writes the guesses to a DRAFT mapping file
                      carrying `reviewed: false`, which nothing will accept.
  EXPORT  (--mapping) Apply a REVIEWED mapping file and, with `--out`, write the
                      canonical Parquet plus a provenance record. `--out` is
                      refused without `--mapping`: a guess must never be exported
                      as though somebody had checked it.

The export never touches the input, refuses to overwrite an existing output
unless told to, and publishes both files or neither. An overwrite moves the
existing pair aside first and puts them back if publication fails, so a failed
re-export cannot destroy a good one.

NOT a transaction. If the process is killed between the two final renames, the
new Parquet can be left beside the old sidecar. The `.backup` files are the
recovery path, and their presence on a later run is treated as evidence that an
earlier run did not finish.

--demo proves the machinery works end to end by taking our own synthetic data,
disguising it with plausible vendor column names and units, and binding it back.
If the demo passes, the seam is sound and only the guesses are in question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract
from headway.adapters.base import Adapter, read_any, suggest_mapping
from headway.adapters.mapping import MappingError, draft, load_mapping

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# A deliberately awkward disguise for --demo: different names, different case,
# units in mA and metres, a timestamp as text, and two columns we do not want.
DISGUISE = {
    "ts": "Event_Time",
    "asset_id": "EQUIPMENT_ID",
    "train_id": "Car No",
    "cycle_duration_s": "operation_time",
    "peak_current_a": "I_Peak (mA)",
    "mean_current_a": "avg_current_mA",
    "current_integral_as": "charge",
    "travel_mm": "leaf_travel_m",
    "obstruction_flag": "Obstruction_Detected",
    "retry_count": "recycle_count",
    "ambient_temp_c": "saloon_temp",
    "load_proxy": "load_weigh",
    "cycles_since_service": "operations_since_service",
    "fault_confirmed": "verified_fault",
    "fault_mode": "failure_mode",
}


def make_demo_file() -> Path:
    """Disguise our own data as a plausible third-party export."""
    src = pd.read_parquet(DATA / "door_cycles.parquet").head(60_000).copy()
    out = pd.DataFrame()
    for ours, theirs in DISGUISE.items():
        if ours not in src.columns:
            continue
        col = src[ours]
        if ours in ("peak_current_a", "mean_current_a"):
            col = col * 1000.0                      # A -> mA
        if ours == "travel_mm":
            col = col / 1000.0                      # mm -> m
        if ours == "ts":
            col = col.dt.strftime("%d/%m/%Y %H:%M:%S")
        out[theirs] = col
    out["hour_of_day"] = np.nan                     # present but useless
    out["Notes"] = ""                               # noise
    out["record_id"] = np.arange(len(out))          # noise
    path = DATA / "_demo_vendor_export.csv"
    out.to_csv(path, index=False)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_aliases(candidates: list[tuple[str, Path | None]]) -> None:
    """No two roles may be the same file on disk.

    The input telemetry and the reviewed mapping are things this tool reads and
    must never write. `--overwrite` authorises replacing an export; it does not
    and cannot authorise replacing a source, so this check runs first and is not
    conditional on it.
    """
    seen: dict[str, str] = {}
    for label, path in candidates:
        if path is None:
            continue
        resolved = Path(path).resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            raise SystemExit(
                f"{label} and {seen[key]} are the same file: {resolved}\n"
                "Refusing. --overwrite replaces an export; it never authorises writing over "
                "source telemetry or a reviewed mapping.")
        seen[key] = label


def provenance(*, source: Path, mapping, config, report, validation, frame,
               rows_requested, complete_input, output_path: Path,
               output_sha256: str) -> dict:
    """Everything needed to say what this file is and where it came from.

    Deliberately includes what was NOT done: no episodes were built, no onset
    dates invented, no reference window declared healthy, and no unit or label
    review attested. Those absences are as load-bearing as the hashes.
    """
    return {
        "artifact": "canonical cycle telemetry",
        "tool": "scripts/bind_data.py",
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "exported_at": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "subsystem": config.subsystem,
        "input": {
            "path": source.name,
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "rows_read": int(rows_requested) if rows_requested else None,
            "complete_input": bool(complete_input),
            "sampling_note": (None if complete_input else
                              "SAMPLED EXPORT - only the first rows of the source were read. "
                              "This is not the complete dataset and must not be reported as one."),
            "modified_by_this_tool": False,
        },
        "output": {
            "path": output_path.name,
            "sha256": output_sha256,
            "note": ("this sidecar describes the file named above and no other; a consumer "
                     "must re-hash it before trusting anything here"),
        },
        "mapping": config.as_provenance(),
        "columns": {
            "mapped": dict(sorted(report.mapped.items())),
            "derived": dict(sorted(report.derived.items())),
            "context_filled": dict(sorted(report.filled.items())),
            "dropped_all_null": sorted(report.empty),
            "source_columns_ignored": sorted(report.unmatched_source),
        },
        "unit_conversions": dict(sorted(config.unit_scales.items())) or None,
        "timestamps": {
            "source_timezone": config.source_timezone,
            "timestamp_format": config.timestamp_format,
            "target_timezone": Adapter.TARGET_TIMEZONE,
            "output": "naive local time in the target timezone",
            "note": (None if config.source_timezone else
                     "no source_timezone declared; naive timestamps were passed through "
                     "unchanged and are interpreted downstream as " + Adapter.TARGET_TIMEZONE),
        },
        "boolean_tokens_declared": dict(sorted(config.boolean_tokens.items())) or None,
        "context_supported": bool(frame["context_supported"].all())
        if "context_supported" in frame else None,
        "validation": {"ok": validation.ok, "errors": list(validation.errors),
                       "warnings": list(validation.warnings)},
        "rows": {"input": int(len(mapping)), "output": int(len(frame))},
        "not_provided_by_this_export": [
            "fault episodes (asset_id, train_id, onset_ts, fault_ts)",
            "degradation onset dates",
            "healthy-reference eligibility for any asset",
        ],
        "attestations": {
            "units_verified": False, "identities_verified": False, "fault_labels_verified": False,
            "note": ("This file records SCHEMA CONFIGURATION only. Whether units, asset "
                     "identities and fault-label semantics have actually been reviewed is a "
                     "separate human judgement, asserted with the --units-verified, "
                     "--identities-verified and --fault-labels-verified flags on "
                     "scripts/ps3_readiness.py. A reviewed mapping is not those reviews."),
        },
    }


def write_canonical(out: Path, frame: pd.DataFrame, build_body, overwrite: bool) -> Path:
    """Publish the frame and its sidecar, or leave what was there untouched.

    Two renames are not an atomic pair, so an overwrite cannot simply replace
    and hope. Instead:

      1. write both files completely, to `.partial` names;
      2. move any existing pair aside to `.backup` (a move, never a delete);
      3. rename the new pair into place;
      4. on ANY failure in 2 or 3, remove whatever was published and put the
         backups back, so the previous export survives intact;
      5. drop the backups only once both files are published.

    The honest limitation: this is recoverable, not atomic. A process killed
    between the two renames in step 3 leaves the new Parquet beside the old
    sidecar, with both `.backup` files still on disk. A later run refuses to
    start while they exist, because their presence means a previous run did not
    finish and somebody should look at which pair is the good one.
    """
    prov = out.with_suffix(".provenance.json")
    existing = [p for p in (out, prov) if p.exists()]
    if existing and not overwrite:
        raise SystemExit("refusing to overwrite: " + ", ".join(str(p) for p in existing)
                         + "\nChoose another --out, or pass --overwrite if replacing it is intended.")
    leftovers = [p for p in (out.with_name(out.name + ".backup"),
                             prov.with_name(prov.name + ".backup"),
                             out.with_name(out.name + ".partial"),
                             prov.with_name(prov.name + ".partial")) if p.exists()]
    if leftovers:
        raise SystemExit(
            "an earlier export did not finish; these remain: "
            + ", ".join(str(p) for p in leftovers)
            + "\nCheck which pair is the good one, move the leftovers aside, and re-run.")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_name(out.name + ".partial")
    tmp_prov = prov.with_name(prov.name + ".partial")
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        frame.to_parquet(tmp_out, index=False)
        # The sidecar names the artifact by hash, so the artifact must exist first.
        tmp_prov.write_text(json.dumps(build_body(sha256_file(tmp_out)), indent=2, default=str),
                            encoding="utf-8")
        for target in (out, prov):
            if target.exists():
                backup = target.with_name(target.name + ".backup")
                os.replace(target, backup)
                backups.append((target, backup))
        os.replace(tmp_out, out)
        published.append(out)
        os.replace(tmp_prov, prov)
        published.append(prov)
    except BaseException:
        for target in published:
            target.unlink(missing_ok=True)
        for target, backup in backups:
            if backup.exists():
                target.unlink(missing_ok=True)
                os.replace(backup, target)
        raise
    finally:
        tmp_out.unlink(missing_ok=True)
        tmp_prov.unlink(missing_ok=True)
    for _, backup in backups:
        backup.unlink(missing_ok=True)
    return prov


def main() -> int:
    ap = argparse.ArgumentParser(description="Bind a source file to the Headway contract.")
    ap.add_argument("path", nargs="?", help="the file they gave us")
    ap.add_argument("--subsystem", default="door", choices=sorted(contract.SUBSYSTEMS))
    ap.add_argument("--rows", type=int, default=None, help="read only the first N rows")
    ap.add_argument("--demo", action="store_true", help="prove the seam on disguised data")
    ap.add_argument("--mapping", type=Path,
                    help="reviewed mapping configuration (JSON). Required for --out.")
    ap.add_argument("--out", type=Path,
                    help="write the canonical Parquet here, plus <out>.provenance.json")
    ap.add_argument("--suggest-out", type=Path,
                    help="write the guesses to a DRAFT mapping file for review; "
                         "it carries reviewed:false and nothing will accept it as-is")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing --out and its provenance")
    args = ap.parse_args()

    if args.out and not args.mapping:
        ap.error("--out requires --mapping. A suggested mapping is a guess; review it into "
                 "a mapping file (see --suggest-out) before exporting anything from it.")
    if args.overwrite and not (args.out or args.suggest_out):
        ap.error("--overwrite only applies to --out or --suggest-out")

    if args.demo:
        args.path = str(make_demo_file())
        print(f"demo: wrote a disguised export to {args.path}\n")
    if not args.path:
        ap.error("give me a file, or --demo")

    # Before reading anything: no role may be the same file as another. This is
    # what stops an export writing over the telemetry it was reading.
    refuse_aliases([
        ("the input file", Path(args.path)),
        ("--mapping", args.mapping),
        ("--out", args.out),
        ("the provenance sidecar", args.out.with_suffix(".provenance.json") if args.out else None),
        ("--suggest-out", args.suggest_out),
    ])
    if args.suggest_out and args.suggest_out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite the draft mapping {args.suggest_out}.\n"
                         "Choose another --suggest-out, or pass --overwrite if replacing a "
                         "draft is intended. A REVIEWED mapping should never be a --suggest-out "
                         "target.")

    config = None
    if args.mapping:
        try:
            config = load_mapping(args.mapping)
        except MappingError as exc:
            raise SystemExit(f"mapping refused: {exc}")
        if config.subsystem != args.subsystem:
            raise SystemExit(f"mapping declares subsystem {config.subsystem!r} but --subsystem "
                             f"is {args.subsystem!r}; they must agree")

    raw = read_any(args.path, nrows=args.rows) if args.rows else read_any(args.path)
    sub = contract.get(args.subsystem)
    complete_input = args.rows is None

    print("=" * 78)
    print(f"BIND  {Path(args.path).name}   ->  subsystem={args.subsystem}")
    print("=" * 78)
    print(f"\n{len(raw):,} rows, {len(raw.columns)} columns")
    print("  " + ", ".join(map(str, raw.columns[:14]))
          + (" ..." if len(raw.columns) > 14 else ""))
    if not complete_input:
        print(f"\n  SAMPLED: only the first {args.rows:,} rows were read. Anything produced "
              "from this run describes a sample, not the dataset.")

    mapping, score = suggest_mapping(list(raw.columns), args.subsystem)
    failure = None
    if config is not None:
        print(f"\nUsing the reviewed mapping {args.mapping.name} "
              f"(canonical sha256 {config.canonical_sha256[:16]}).")
        adapter = config.adapter()
        bound, report = adapter.apply(raw)
    else:
        # Suggest mode is a REVIEW AID, so a bind failure must arrive as a
        # finding with the draft still written - a traceback is no help when the
        # answer is "declare these label tokens in the mapping".
        adapter = Adapter(subsystem=args.subsystem, column_map=mapping)
        try:
            bound, report = adapter.apply(raw)
            report.low_confidence = {k: v for k, v in score.items() if v < 1.0}
        except ValueError as exc:
            bound, report, failure = None, None, str(exc)

    if report is not None:
        print(f"\n{report}\n")
    else:
        print(f"\n  CANNOT BIND WITH THE GUESSED MAPPING: {failure}\n")

    missing = [c for c in sub.columns if c not in bound.columns] if bound is not None else []
    if missing:
        print(f"  STILL MISSING, and not derivable: {missing}")
        print(f"  -> map these by hand in nebulax.py, or accept the loss and say so.\n")

    validation = contract.validate(bound, args.subsystem) if bound is not None else None
    if validation is not None:
        print(validation)

    if args.suggest_out:
        body = draft(args.subsystem, mapping, score)
        args.suggest_out.parent.mkdir(parents=True, exist_ok=True)
        args.suggest_out.write_text(json.dumps(body, indent=2), encoding="utf-8")
        print(f"\nWrote DRAFT mapping {args.suggest_out} with reviewed:false. "
              "Review every entry, then set reviewed to true.")

    if failure is not None:
        print("\nNothing was exported. Resolve the finding above - most often by declaring "
              "boolean_tokens in a reviewed mapping file - and re-run.")
        return 1

    if args.out:
        if not validation.ok:
            raise SystemExit("refusing to export: the bound frame does not satisfy the "
                             "contract. Fix the mapping and re-run.\n" + str(validation))
        def build_body(output_sha256: str) -> dict:
            return provenance(source=Path(args.path), mapping=raw, config=config, report=report,
                              validation=validation, frame=bound, rows_requested=args.rows,
                              complete_input=complete_input, output_path=args.out,
                              output_sha256=output_sha256)

        prov = write_canonical(args.out, bound, build_body, args.overwrite)
        print(f"\nWrote {args.out}  ({len(bound):,} rows)")
        print(f"Wrote {prov}")
        if not complete_input:
            print("  marked complete_input=false: this is a SAMPLE, not the dataset.")
        print("  No fault episodes, onset dates or reference eligibility were produced.")
        print(f"\nNext:  .venv\\Scripts\\python.exe -B scripts\\ps3_readiness.py {args.out}")
        return 0

    print("\n" + "-" * 78)
    print(f"Paste into headway/adapters/nebulax.py as {args.subsystem.upper()}_MAP:")
    print("-" * 78)
    width = max((len(k) for k in mapping), default=0) + 3
    print("{")
    for target in sub.columns:
        if target in mapping:
            flag = "" if score.get(target, 1.0) >= 1.0 else f"  # CHECK ({score[target]:.2f})"
            print(f'    "{target}":{" " * (width - len(target))}"{mapping[target]}",{flag}')
    print("}")

    if report.filled:
        print(f"\nNOTE: {len(report.filled)} context column(s) were filled with a constant.")
        print("Those confounds will NOT be corrected for. Say so in the write-up rather")
        print("than letting the numbers imply a correction that did not happen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
