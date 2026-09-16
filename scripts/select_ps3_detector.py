"""PS3 detector selection: reviewed external data, or the legacy synthetic experiment.

Two modes, deliberately separate.

REVIEWED DATA (--config)
    Every split, the alert budget, the qualification policy, reference
    eligibility, attestations and data exposure are declared in a JSON
    configuration. Two stages, run in order:

      --stage select   record config and hashes, preflight, fit on the declared
                       reference, select on validation, and FREEZE the result
      --stage test     verify nothing has changed since the freeze, then run the
                       final test exactly once

    See headway/ps3_run.py for what is checked, and DATA_HANDOVER_PLAN.md for
    the configuration format and Windows commands. Door only.

LEGACY SYNTHETIC (--legacy-synthetic --out DIR)
    The original fixed synthetic experiment, unchanged so its published result
    stays reproducible: it fits on the first 30 days of the synthetic fleet and
    uses the historical per-asset baseline windows. Those are the assumptions
    the reviewed-data mode removes, which is why the two modes are not merged.
    It writes into a directory that must not already hold its artifacts. It is
    never pointed at organiser data.

Neither mode promotes a detector into the dashboard.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.detector_selection import SelectionPolicy, select_detector, assess_selected  # noqa: E402
from headway.models import detectors as D  # noqa: E402
from headway.pipeline import HealthPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ARTIFACTS = ("protocol.json", "selection.json", "test.json")


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x): return None
    if isinstance(x, np.generic): return x.item()
    return x


def legacy_synthetic(out: Path) -> int:
    """The original synthetic experiment, byte-for-byte in what it computes."""
    existing = [name for name in LEGACY_ARTIFACTS if (out / name).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite legacy artifacts in {out}: {', '.join(existing)}. "
                         "Choose an empty --out; the published result in data/ps3_selection is kept.")
    cpath, epath = ROOT / "data/door_cycles.parquet", ROOT / "data/door_episodes.csv"
    cycles = pd.read_parquet(cpath)
    eps = pd.read_csv(epath, parse_dates=["onset_ts", "fault_ts"])
    start, end = cycles.ts.min().floor("D"), cycles.ts.max().ceil("D")
    ref_end = start + pd.Timedelta(days=30)
    calibration_end = (start + (end - start) * .5).floor("D")
    validation_end = (start + (end - start) * .7).floor("D")
    if not ref_end < calibration_end < validation_end:
        raise ValueError("not enough calendar history for the fixed protocol")
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"reference_end": str(ref_end), "calibration_end": str(calibration_end),
        "validation_end": str(validation_end), "test_end": str(end),
        "data": "synthetic development evidence; previously inspected simulator family",
        "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cpath, epath)},
        "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": "legacy synthetic: first-30-day reference and historical baseline windows",
        "no_automatic_promotion": True}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    pipeline = HealthPipeline().fit(cycles[cycles.ts < ref_end])
    daily = pipeline.transform(cycles)
    norm = [f"{s}_hx" for s in pipeline.levels]
    zoo = {"raw": D.RawThreshold("current_integral_as"),
        "ewma": D.EWMAChart("current_integral_as"),
        "isolation_forest": D.IsolationForestDetector(needs=norm),
        "pca": D.PCAReconstruction(needs=norm),
        "mahalanobis": D.MahalanobisDetector(needs=norm), "lof": D.LOFDetector(needs=norm),
        "headway_directional": D.Passthrough("health_index"),
        "headway_distance": D.Passthrough("anomaly_distance")}
    scored = D.run(zoo, daily, reference_days=30, smooth_days=3)
    selection = select_detector(scored, eps, list(zoo), calibration_start=ref_end,
        calibration_end=calibration_end, validation_end=validation_end, policy=SelectionPolicy())
    # The selection artifact is saved BEFORE test labels enter the final evaluation.
    (out / "selection.json").write_text(json.dumps(clean(selection), indent=2, allow_nan=False), encoding="utf-8")
    test = assess_selected(selection, scored, eps)
    (out / "test.json").write_text(json.dumps(clean(test), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean({"selection": selection, "test": test}), indent=2, allow_nan=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", type=Path, help="reviewed-data run configuration (JSON)")
    ap.add_argument("--stage", choices=("select", "test"), help="which reviewed-data stage to run")
    ap.add_argument("--run-id", help="name for a new select run (default: timestamp + config hash)")
    ap.add_argument("--run", help="existing run id, required for --stage test")
    ap.add_argument("--legacy-synthetic", action="store_true",
                    help="reproduce the original fixed synthetic experiment")
    ap.add_argument("--out", type=Path, help="output directory for --legacy-synthetic")
    args = ap.parse_args(argv)

    if args.legacy_synthetic:
        if args.config or args.stage or args.run or args.run_id:
            ap.error("--legacy-synthetic takes only --out; it does not read a configuration")
        if not args.out:
            ap.error("--legacy-synthetic requires --out")
        return legacy_synthetic(args.out)

    if not args.config or not args.stage:
        ap.error("give --config and --stage, or --legacy-synthetic --out DIR")
    from headway import ps3_run
    cfg = ps3_run.load_config(args.config)
    if args.stage == "select":
        if args.run:
            ap.error("--run names an existing run; a select stage always creates a new one")
        return ps3_run.stage_select(cfg, args.run_id)
    if not args.run:
        ap.error("--stage test requires --run <run id> from the select stage")
    if args.run_id:
        ap.error("--run-id names a new run; the test stage takes --run")
    return ps3_run.stage_test(cfg, args.run)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
