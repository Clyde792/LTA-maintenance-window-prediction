"""
The Friday-night tool. Point it at whatever NEBULA X hand over.

    .venv/Scripts/python.exe scripts/bind_data.py <file> [--subsystem door]
    .venv/Scripts/python.exe scripts/bind_data.py --demo

It inspects the file, guesses the column mapping, applies it, runs the contract
validator, and prints a dict ready to paste into headway/adapters/nebulax.py.

The point is to convert the riskiest unknown of the whole weekend - "what shape
is their data?" - from an engineering problem into a fifteen-minute review of
about fifteen guesses. Everything downstream already speaks the contract.

--demo proves the machinery works end to end by taking our own synthetic data,
disguising it with plausible vendor column names and units, and binding it back.
If the demo passes, the seam is sound and only the guesses are in question.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract
from headway.adapters.base import Adapter, read_any, suggest_mapping

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


def main() -> int:
    ap = argparse.ArgumentParser(description="Bind a source file to the Headway contract.")
    ap.add_argument("path", nargs="?", help="the file they gave us")
    ap.add_argument("--subsystem", default="door", choices=sorted(contract.SUBSYSTEMS))
    ap.add_argument("--rows", type=int, default=None, help="read only the first N rows")
    ap.add_argument("--demo", action="store_true", help="prove the seam on disguised data")
    args = ap.parse_args()

    if args.demo:
        args.path = str(make_demo_file())
        print(f"demo: wrote a disguised export to {args.path}\n")
    if not args.path:
        ap.error("give me a file, or --demo")

    raw = read_any(args.path, nrows=args.rows) if args.rows else read_any(args.path)
    sub = contract.get(args.subsystem)

    print("=" * 78)
    print(f"BIND  {Path(args.path).name}   ->  subsystem={args.subsystem}")
    print("=" * 78)
    print(f"\n{len(raw):,} rows, {len(raw.columns)} columns")
    print("  " + ", ".join(map(str, raw.columns[:14]))
          + (" ..." if len(raw.columns) > 14 else ""))

    mapping, score = suggest_mapping(list(raw.columns), args.subsystem)
    adapter = Adapter(subsystem=args.subsystem, column_map=mapping)
    bound, report = adapter.apply(raw)
    report.low_confidence = {k: v for k, v in score.items() if v < 1.0}

    print(f"\n{report}\n")

    missing = [c for c in sub.columns if c not in bound.columns]
    if missing:
        print(f"  STILL MISSING, and not derivable: {missing}")
        print(f"  -> map these by hand in nebulax.py, or accept the loss and say so.\n")

    print(contract.validate(bound, args.subsystem))

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
