"""CLI for the isolated repair-verification backend; JSON output for UI binding."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.repair_verification import RepairStore, make_reference


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--daily", required=True, help="Parquet containing only the selected healthy reference window")
    ref.add_argument("--asset", required=True)
    ref.add_argument("--channels", nargs="+", required=True)
    ref.add_argument("--contexts", nargs="+", required=True)
    ref.add_argument("--model-version", required=True)
    ref.add_argument("--reviewed-by", required=True)
    ref.add_argument("--reviewed-at", required=True)
    ref.add_argument("--output", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--db", required=True)
    rec.add_argument("--record", required=True, help="Maintenance record JSON")
    check = sub.add_parser("assess")
    check.add_argument("--db", required=True)
    check.add_argument("--job", required=True)
    check.add_argument("--reference", required=True)
    check.add_argument("--daily", required=True)
    check.add_argument("--as-of", required=True)
    check.add_argument("--model-version", required=True)
    history = sub.add_parser("history")
    history.add_argument("--db", required=True)
    history.add_argument("--job", required=True)
    followups = sub.add_parser("followups")
    followups.add_argument("--db", required=True)
    args = p.parse_args()
    if args.command == "reference":
        result = make_reference(pd.read_parquet(args.daily), asset_id=args.asset,
            channels=args.channels, contexts=args.contexts, model_version=args.model_version,
            reviewed_by=args.reviewed_by, reviewed_at=args.reviewed_at)
        # A reviewed reference is a frozen artifact. Never silently overwrite it.
        with Path(args.output).open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    else:
        store = RepairStore(args.db)
        if args.command == "record":
            result = store.record(**read_json(args.record))
        elif args.command == "assess":
            result = store.assess(args.job, read_json(args.reference), pd.read_parquet(args.daily),
                                  as_of=args.as_of, model_version=args.model_version)
        elif args.command == "history":
            result = store.history(args.job)
        else:
            result = store.followups()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
