"""Inspect mapped Parquet telemetry; report blockers without fitting models."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.data_readiness import assess_readiness


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
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
