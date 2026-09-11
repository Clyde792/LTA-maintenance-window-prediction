"""Fixed temporal model-selection experiment; does not modify the demo winner."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from headway.detector_selection import SelectionPolicy, select_detector, assess_selected
from headway.models import detectors as D
from headway.pipeline import HealthPipeline

ROOT = Path(__file__).resolve().parents[1]


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x): return None
    if isinstance(x, np.generic): return x.item()
    return x


def main():
    cpath, epath = ROOT / "data/door_cycles.parquet", ROOT / "data/door_episodes.csv"
    cycles = pd.read_parquet(cpath)
    eps = pd.read_csv(epath, parse_dates=["onset_ts", "fault_ts"])
    start, end = cycles.ts.min().floor("D"), cycles.ts.max().ceil("D")
    ref_end = start + pd.Timedelta(days=30)
    calibration_end = (start + (end - start) * .5).floor("D")
    validation_end = (start + (end - start) * .7).floor("D")
    if not ref_end < calibration_end < validation_end:
        raise ValueError("not enough calendar history for the fixed protocol")
    out = ROOT / "data/ps3_selection"
    out.mkdir(exist_ok=True)
    protocol = {"reference_end": str(ref_end), "calibration_end": str(calibration_end),
        "validation_end": str(validation_end), "test_end": str(end),
        "data": "synthetic development evidence; previously inspected simulator family",
        "input_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cpath, epath)},
        "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
