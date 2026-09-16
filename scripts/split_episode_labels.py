"""Split one episode label file into development and test label files.

    .venv/Scripts/python.exe scripts/split_episode_labels.py <episodes.csv> \
        --validation-end 2026-06-24 --development-out <dev.csv> --test-out <test.csv>

The reviewed-data runner (scripts/select_ps3_detector.py --config) takes labels
as two inputs, so the selection stage can be shown never to open test labels.
An episode belongs to the development file when its fault was confirmed on or
before validation_end (a midnight, local Asia/Singapore time), and to the test
file otherwise. Rows are copied verbatim; nothing is re-timed or invented.

Do this split BEFORE anyone inspects test-period labels. Splitting a file that
has already been read end to end produces the right shape but not an untouched
test set, and the run configuration's data_exposure must say so.

Refuses to overwrite either output, to write an output onto the input, or to
write both scopes to one file.
"""
import argparse
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway import ps3_run as P  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("episodes", type=Path)
    ap.add_argument("--validation-end", required=True)
    ap.add_argument("--development-out", type=Path, required=True)
    ap.add_argument("--test-out", type=Path, required=True)
    args = ap.parse_args(argv)

    source = args.episodes.resolve()
    outputs = {"development": args.development_out.resolve(), "test": args.test_out.resolve()}
    names = [os.path.normcase(str(p)) for p in (source, *outputs.values())]
    if len(set(names)) != 3:
        raise SystemExit("the input and both outputs must be three different files")
    for role, path in outputs.items():
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {role} label file {path}")
    if not source.exists():
        raise SystemExit(f"label file not found: {source}")

    end = P._midnight("validation_end", args.validation_end)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    if "fault_ts" not in raw:
        raise SystemExit("label file has no fault_ts column; episodes cannot be scoped")
    fault = P.local_times(raw["fault_ts"])
    if fault.isna().any():
        raise SystemExit(f"{int(fault.isna().sum())} fault_ts value(s) are blank, unparseable or "
                         "DST-ambiguous; they cannot be assigned to a scope")
    development = (fault <= end).to_numpy()
    for role, path, rows in (("development", outputs["development"], raw[development]),
                             ("test", outputs["test"], raw[~development])):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial")
        rows.to_csv(tmp, index=False)
        if path.exists():
            tmp.unlink()
            raise SystemExit(f"refusing to overwrite existing {role} label file {path}")
        os.replace(tmp, path)
        print(f"{role}: {len(rows)} episode(s) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
