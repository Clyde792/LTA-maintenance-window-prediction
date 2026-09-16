"""Reviewed-data PS3 detector selection: explicit splits, preflight, frozen runs.

This is the path for evaluating detectors on external telemetry. Nothing about an
evaluation is decided by the data's own shape or by its outcomes:

  * every boundary is declared in a configuration file and recorded before
    anything is fitted;
  * labels arrive as TWO inputs. Development labels (faults confirmed by
    validation_end) are the only labels the selection stage opens - for
    preflight, eligibility consistency and model choice. Test labels are opened
    for the first time by the test stage;
  * a preflight checks schema, provenance, sampling, development labels and
    reference eligibility, and refuses rather than degrading. Human attestations
    are recorded separately and never stand in for those checks;
  * preprocessing and every detector are fitted on the same declared reference,
    on the availability clock, restricted to eligible assets and to asset-days
    that pass EXPLICIT reference-quality checks;
  * rows that fail quality or context support are excluded BEFORE scoring and
    smoothing, so an invalid reading cannot move a later valid score;
  * scores are aligned onto a declared grid of EXPECTED asset-days before
    qualification and final metrics, so a day with no telemetry counts as
    unavailable in the qualification gate itself, not only in a side report;
  * selection is frozen with every input, score and code hash it depended on,
    and the test stage refuses to run if any of them has changed.

Door only. A bogie evaluation is refused: no bogie detector, threshold or
episode semantics has been validated, and door ones must not be transferred.

The existing evaluator matches alerts to episodes on `[onset_ts, fault_ts)`.
Labels without onset are refused with that reason; onset times are not invented
and no other matching protocol is substituted.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import contract, features
from .adapters.base import Adapter
from .adapters.mapping import canonical_bytes
from .data_readiness import assess_readiness
from .detector_selection import SelectionPolicy, assess_selected, select_detector
from .features import MIN_COMPLETENESS, MIN_CYCLES
from .models import detectors as D
from .pipeline import HealthPipeline

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = "Asia/Singapore"
BOUNDARY_NAMES = ("reference_start", "reference_end", "calibration_end", "validation_end", "test_end")
RUN_KINDS = ("external", "synthetic")
GRID_RULES = {
    "from_reference_start": "every asset in the telemetry is expected on every day of the window",
    "from_first_observation": ("each asset is expected from its first observed day onward; later "
                               "days with no telemetry stay expected"),
}
ATTESTATIONS = ("units_verified", "identities_verified", "fault_labels_verified",
                "reference_eligibility_verified")
POLICY_FIELDS = ("threshold_quantile", "daily_budget", "cooldown_days", "minimum_fault_groups",
                 "minimum_recall", "maximum_false_alerts_per_asset_month", "minimum_score_fraction")
CONFIG_KEYS = {"run_kind", "subsystem", "telemetry", "development_episodes", "test_episodes",
               "output_dir", "timezone", "boundaries", "expected_grid", "reference_eligibility",
               "attestations", "label_semantics", "data_exposure", "policy", "smooth_days",
               "min_reference_days_per_asset", "notes"}
REQUIRED_KEYS = CONFIG_KEYS - {"notes", "smooth_days", "min_reference_days_per_asset"}
CODE_FILES = ("scripts/select_ps3_detector.py", "headway/ps3_run.py", "headway/detector_selection.py",
              "headway/models/detectors.py", "headway/pipeline.py", "headway/normalise.py",
              "headway/features.py", "headway/evaluate/metrics.py", "headway/data_readiness.py",
              "scripts/ps3_readiness.py")
KEYS = ["asset_id", "day", "available_at"]


class RunRefused(SystemExit):
    """A refusal with a reason. Exits non-zero; never a traceback."""


# ---------------------------------------------------------------------------
# hashing and artifact writing
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes() -> dict:
    return {rel: sha256_file(ROOT / rel) for rel in CODE_FILES}


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


def write_new(path: Path, text: str) -> None:
    """Write a file that must not already exist. Artifacts are never overwritten."""
    if path.exists():
        raise RunRefused(f"refusing to overwrite existing artifact {path}")
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text, encoding="utf-8")
    try:
        if path.exists():
            raise RunRefused(f"refusing to overwrite existing artifact {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, body) -> str:
    write_new(path, json.dumps(clean(body), indent=2, allow_nan=False))
    return sha256_file(path)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def _midnight(name, value):
    try:
        t = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise RunRefused(f"boundaries.{name} is not a readable date: {value!r}") from None
    if pd.isna(t):
        raise RunRefused(f"boundaries.{name} is not a readable date: {value!r}")
    if t.tzinfo is not None:
        t = t.tz_convert(TIMEZONE).tz_localize(None)
    if t != t.normalize():
        raise RunRefused(f"boundaries.{name} must be a midnight in {TIMEZONE}, got {t}; daily "
                         "aggregates land at midnight, so a boundary elsewhere is ambiguous")
    return t


def _text(obj, key, where):
    value = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise RunRefused(f"{where}.{key} must be a non-empty string")
    return value


def load_config(path) -> dict:
    """Read and validate a run configuration. Refuses rather than defaulting.

    The test-label path is joined as text only - no filesystem call touches it
    here - because the selection stage must never open that input.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise RunRefused(f"configuration not found: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunRefused(f"{path} is not readable JSON: {exc}") from None
    if not isinstance(body, dict):
        raise RunRefused("the configuration must be a JSON object")
    unknown = sorted(set(body) - CONFIG_KEYS)
    if unknown:
        hint = (" (a single 'episodes' file is no longer accepted: supply development_episodes "
                "and test_episodes separately)") if "episodes" in unknown else ""
        raise RunRefused(f"unknown configuration key(s): {', '.join(unknown)}{hint}")
    missing = sorted(REQUIRED_KEYS - set(body))
    if missing:
        raise RunRefused(f"configuration is missing required key(s): {', '.join(missing)}")

    if body["run_kind"] not in RUN_KINDS:
        raise RunRefused(f"run_kind must be one of {RUN_KINDS}, got {body['run_kind']!r}")
    external = body["run_kind"] == "external"
    if body["subsystem"] != "door":
        raise RunRefused(
            f"subsystem {body['subsystem']!r} is not supported by this runner. Only door "
            "evaluation is implemented: no bogie detector, threshold or episode semantics has "
            "been validated, and door thresholds and semantics must not be transferred.")
    if body["timezone"] != TIMEZONE:
        raise RunRefused(
            f"timezone must be {TIMEZONE!r}: canonical telemetry from scripts/bind_data.py is "
            "naive local time in that zone, and boundaries in another zone would silently "
            "shift every split")

    base = path.parent
    resolved = {}
    for key in ("telemetry", "development_episodes", "test_episodes", "output_dir"):
        if not isinstance(body[key], str) or not body[key].strip():
            raise RunRefused(f"{key} must be a non-empty path")
        joined = base / body[key]
        # Text-only for the test labels: Path.resolve() can touch the filesystem.
        resolved[key] = Path(os.path.abspath(joined)) if key == "test_episodes" else joined.resolve()
    if os.path.normcase(str(resolved["development_episodes"])) == \
            os.path.normcase(str(resolved["test_episodes"])):
        raise RunRefused("development_episodes and test_episodes must be different files; one "
                         "label file for both would put test labels in front of selection")

    raw = body["boundaries"]
    if not isinstance(raw, dict) or set(raw) != set(BOUNDARY_NAMES):
        raise RunRefused(f"boundaries must declare exactly {', '.join(BOUNDARY_NAMES)}")
    bounds = {name: _midnight(name, raw[name]) for name in BOUNDARY_NAMES}
    for earlier, later in zip(BOUNDARY_NAMES, BOUNDARY_NAMES[1:]):
        if not bounds[earlier] < bounds[later]:
            raise RunRefused(f"boundaries must be strictly ordered: {earlier} ({bounds[earlier].date()}) "
                             f"is not before {later} ({bounds[later].date()})")

    if body["expected_grid"] not in GRID_RULES:
        raise RunRefused(f"expected_grid must be one of {sorted(GRID_RULES)}; availability is "
                         "measured against it, so it is declared rather than inferred")

    min_days = body.get("min_reference_days_per_asset", 14)
    if type(min_days) is not int or min_days < 1:
        raise RunRefused("min_reference_days_per_asset must be a positive integer")
    if (bounds["reference_end"] - bounds["reference_start"]).days < min_days:
        raise RunRefused(f"the reference window is shorter than min_reference_days_per_asset ({min_days})")
    smooth = body.get("smooth_days", 3)
    if type(smooth) is not int or smooth < 1:
        raise RunRefused("smooth_days must be a positive integer")

    elig = body["reference_eligibility"]
    if not isinstance(elig, dict):
        raise RunRefused("reference_eligibility must be an object")
    assets = elig.get("assets")
    if not (assets == "all" or (isinstance(assets, list) and assets
                                and all(isinstance(a, str) and a.strip() for a in assets))):
        raise RunRefused('reference_eligibility.assets must be "all" or a non-empty list of asset ids')
    _text(elig, "declared_by", "reference_eligibility")
    _text(elig, "basis", "reference_eligibility")
    if external:
        # Eligibility rests on independent reviewed evidence - an inspection or
        # maintenance record - never on the absence of labels.
        _text(elig, "evidence_id", "reference_eligibility")

    att = body["attestations"]
    if not isinstance(att, dict):
        raise RunRefused("attestations must be an object")
    for key in ATTESTATIONS:
        if not isinstance(att.get(key), bool):
            raise RunRefused(f"attestations.{key} must be true or false")
    _text(att, "declared_by", "attestations")

    sem = body["label_semantics"]
    if not isinstance(sem, dict):
        raise RunRefused("label_semantics must be an object")
    _text(sem, "onset_ts", "label_semantics")
    _text(sem, "fault_ts", "label_semantics")

    exposure = body["data_exposure"]
    if not isinstance(exposure, dict) or not isinstance(exposure.get("test_period_previously_inspected"), bool):
        raise RunRefused("data_exposure.test_period_previously_inspected must be true or false")
    _text(exposure, "declared_by", "data_exposure")

    policy = body["policy"]
    if not isinstance(policy, dict) or set(policy) != set(POLICY_FIELDS):
        raise RunRefused("policy must declare exactly: " + ", ".join(POLICY_FIELDS)
                         + ". Qualification gates are not defaulted for reviewed data.")
    try:
        SelectionPolicy(**policy)
    except (ValueError, TypeError) as exc:
        raise RunRefused(f"invalid qualification policy: {exc}") from None

    return {"path": str(path), "body": body, "resolved": resolved, "boundaries": bounds,
            "min_reference_days": min_days, "smooth_days": smooth,
            "canonical_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}


def recorded_config(cfg) -> dict:
    """What the run records about its configuration, before anything is fitted."""
    body = cfg["body"]
    return {
        "configuration_file": cfg["path"],
        "configuration_canonical_sha256": cfg["canonical_sha256"],
        "run_kind": body["run_kind"], "subsystem": body["subsystem"],
        "inputs": {k: str(v) for k, v in cfg["resolved"].items()},
        "timezone_interpretation": (
            f"Boundaries, telemetry and naive label timestamps are local {TIMEZONE} time. "
            "Offset-aware label timestamps are converted to it. Every boundary is a midnight, "
            "and a row belongs to a window by when its daily aggregate became AVAILABLE "
            "(day + 1 day), not by its observation day."),
        "boundaries": {k: v.isoformat() for k, v in cfg["boundaries"].items()},
        "windows": {
            "reference": ("reference_start < available_at <= reference_end, eligible assets, "
                          "asset-days passing explicit reference-quality checks"),
            "calibration": "reference_end < available_at <= calibration_end",
            "validation": "calibration_end < available_at <= validation_end",
            "test": "validation_end < available_at <= test_end",
        },
        "label_scope": {
            "development_episodes": "faults confirmed on or before validation_end; the only labels "
                                    "the selection stage opens",
            "test_episodes": "faults confirmed after validation_end; opened only by the test stage",
        },
        "expected_grid": {"rule": body["expected_grid"], "meaning": GRID_RULES[body["expected_grid"]],
                          "use": ("scores are aligned onto this grid before qualification and final "
                                  "metrics; an expected day with no score is unavailable")},
        "validity": {
            "scoring": ("a row is scored only if data_quality_ok and context_supported are both True; "
                        "invalid rows are removed before scoring and smoothing"),
            "reference": (f"an asset-day is reference material only if every completeness column is "
                          f">= {MIN_COMPLETENESS}, n_cycles >= {MIN_CYCLES}, health inputs are "
                          "complete and context is supported. The prediction-readiness flag is not "
                          "used, because it rejects every date before the fit."),
        },
        "reference_eligibility": body["reference_eligibility"],
        "attestations": body["attestations"],
        "label_semantics": body["label_semantics"],
        "data_exposure": body["data_exposure"],
        "alert_budget": {"daily_budget": body["policy"]["daily_budget"],
                         "cooldown_days": body["policy"]["cooldown_days"]},
        "qualification_policy": body["policy"],
        "smooth_days": cfg["smooth_days"],
        "min_reference_days_per_asset": cfg["min_reference_days"],
        "automatic_promotion": False,
    }


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

_LOCAL = Adapter(subsystem="door", source_timezone=TIMEZONE)


def local_times(series: pd.Series) -> pd.Series:
    """Label timestamps as naive local time, with ingestion's mixed-offset rules."""
    return _LOCAL._parse_ts(series)


def read_episodes(path: Path, *, scope: str, validation_end):
    """Episodes as a frame, or the reasons they cannot be used.

    `scope` is "development" (faults confirmed by validation_end) or "test"
    (confirmed after it). A file carrying labels outside its scope is refused:
    test-period faults in the development file would put them in front of
    selection.
    """
    role = f"{scope} label file"
    if not path.exists():
        return None, [f"no {role} at {path}; labelled evaluation cannot run"]
    try:
        eps = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return None, [f"{role} {path.name} is not readable CSV: {exc}"]
    problems = []
    for column in ("asset_id", "train_id", "fault_ts"):
        if column not in eps:
            problems.append(f"{role} is missing required column {column!r}")
    if "onset_ts" not in eps or eps["onset_ts"].isna().any() \
            or eps["onset_ts"].astype(str).str.strip().eq("").any():
        problems.append(
            f"{role}: episode onset times are missing. The current evaluator credits an alert only "
            "when it falls in [onset_ts, fault_ts), so it cannot compute detection or lead time "
            "without onset. Confirmation-only labels need a separately agreed event-matching "
            "protocol, which this runner does not implement; onset times are not invented.")
    if problems:
        return None, problems
    eps = eps.copy()
    eps["onset_ts"] = local_times(eps["onset_ts"].astype(str))
    eps["fault_ts"] = local_times(eps["fault_ts"].astype(str))
    if eps[["onset_ts", "fault_ts"]].isna().any().any():
        problems.append(f"{role}: timestamps include unparseable or DST-ambiguous values")
    elif (eps.onset_ts >= eps.fault_ts).any():
        problems.append(f"{role}: every episode's onset_ts must precede its fault_ts")
    if eps[["asset_id", "train_id"]].isna().any().any():
        problems.append(f"{role}: episodes have missing asset or train identities")
    if not problems:
        end = pd.Timestamp(validation_end)
        if scope == "development" and (eps.fault_ts > end).any():
            problems.append(f"{role} contains {int((eps.fault_ts > end).sum())} episode(s) confirmed "
                            "after validation_end; test-period labels must be supplied only in "
                            "test_episodes")
        if scope == "test" and (eps.fault_ts <= end).any():
            problems.append(f"{role} contains {int((eps.fault_ts <= end).sum())} episode(s) confirmed "
                            "on or before validation_end; those belong in development_episodes")
    return (None, problems) if problems else (eps, [])


# ---------------------------------------------------------------------------
# quality, grid and alignment
# ---------------------------------------------------------------------------

def reference_day_quality(ref_cycles: pd.DataFrame) -> pd.DataFrame:
    """Which reference asset-days are usable to fit preprocessing, from raw cycles.

    Uses `features.to_daily`'s own data-quality flag - completeness and cycle
    count - which carries NO fit-readiness gate, plus supported context. Nothing
    has been fitted yet, so nothing here depends on a model.
    """
    if ref_cycles.empty:
        return pd.DataFrame(columns=["asset_id", "day", "reference_valid"])
    daily = features.to_daily(ref_cycles, "door", extra=())
    valid = daily.data_quality_ok.eq(True)
    if "context_supported" in daily:
        valid &= daily.context_supported.eq(True)
    return daily.assign(reference_valid=valid)[["asset_id", "day", "reference_valid"]]


def reference_quality(daily: pd.DataFrame) -> pd.Series:
    """Explicit reference-quality checks on transformed daily rows.

    Deliberately NOT `data_quality_ok` or `quality_ready`: after HealthPipeline
    both include `available_at >= fitted_at`, which is False for every reference
    date by construction and would discard the entire reference.
    """
    completeness = [c for c in daily if c.endswith("_completeness")]
    ok = daily[completeness].min(axis=1) >= MIN_COMPLETENESS
    ok &= pd.to_numeric(daily.n_cycles, errors="coerce") >= MIN_CYCLES
    ok &= daily.health_inputs_complete.eq(True)
    ok &= daily.context_supported.eq(True)
    return ok.fillna(False)


def expected_grid(cycles, lo, hi, rule, reference_start) -> pd.DataFrame:
    """Expected asset-days whose aggregate would land in (lo, hi]."""
    first = cycles.groupby("asset_id").ts.min().dt.floor("D")
    if rule == "from_reference_start":
        first = pd.Series(pd.Timestamp(reference_start), index=first.index)
    days = pd.date_range(lo, pd.Timestamp(hi) - pd.Timedelta(days=1), freq="D")
    grid = pd.DataFrame([(a, d) for a, f in first.items() for d in days if d >= f],
                        columns=["asset_id", "day"])
    grid["available_at"] = grid.day + pd.Timedelta(days=1)
    return grid


def align_scores(scores: pd.DataFrame, grid: pd.DataFrame, names) -> pd.DataFrame:
    """Scores placed on the expected grid; an expected day with no score is NaN.

    This is what qualification and the final test see. Rows off the grid are
    dropped, and a duplicate asset-day is an error rather than a double count.
    """
    if scores.duplicated(["asset_id", "day"]).any():
        raise ValueError("duplicate asset-days in scores")
    aligned = grid[KEYS].merge(scores[KEYS + list(names)], on=KEYS, how="left", validate="one_to_one")
    return aligned.sort_values(["asset_id", "day"]).reset_index(drop=True)


def crossing(episodes, boundary):
    b = pd.Timestamp(boundary)
    return episodes[(episodes.onset_ts <= b) & (episodes.fault_ts > b)]


def availability(aligned, names, lo, hi, excluded) -> dict:
    rows = aligned[(aligned.available_at > lo) & (aligned.available_at <= hi)
                   & ~aligned.asset_id.isin(set(excluded))]
    return {"expected_asset_days": len(rows),
            "availability": {n: (float(np.isfinite(rows[n].astype(float)).mean()) if len(rows) else None)
                             for n in names}}


def qualify(aligned, dev_episodes, names, bounds, policy):
    """Model choice on the expected grid, from development labels only."""
    return select_detector(aligned, dev_episodes, list(names),
                           calibration_start=bounds["reference_end"],
                           calibration_end=bounds["calibration_end"],
                           validation_end=bounds["validation_end"],
                           policy=SelectionPolicy(**policy))


def unmatched(candidate):
    if "alarms" not in candidate:
        return None
    return int(candidate["alarms"] - round(candidate["precision"] * candidate["alarms"]))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(cfg):
    """Every reason the run must not fit anything. Development labels only."""
    body, paths, b = cfg["body"], cfg["resolved"], cfg["boundaries"]
    synthetic = body["run_kind"] == "synthetic"
    problems, waived, warnings = [], [], []
    record = {"run_kind": body["run_kind"],
              "test_labels": {"path": str(paths["test_episodes"]), "opened": False}}

    def strict(message):
        (waived if synthetic else problems).append(message)

    telemetry = paths["telemetry"]
    if not telemetry.exists():
        return {**record, "status": "refused", "problems": [f"telemetry not found: {telemetry}"]}, None, None
    try:
        cycles = pd.read_parquet(telemetry)
    except Exception as exc:  # pyarrow raises a zoo of types for a bad file
        return {**record, "status": "refused",
                "problems": [f"telemetry is not readable Parquet: {exc}"]}, None, None
    validation = contract.validate(cycles, "door")
    record["schema"] = {"ok": validation.ok, "errors": validation.errors,
                        "warnings": validation.warnings, "rows": len(cycles)}
    problems += [f"schema: {e}" for e in validation.errors]
    if "subsystem" in cycles and not cycles.subsystem.eq("door").all():
        problems.append("telemetry contains non-door rows; only door evaluation is supported")
    if not validation.ok:
        return {**record, "status": "refused", "problems": problems}, None, None
    cycles = cycles[cycles.ts < b["test_end"]].copy()
    record["telemetry_extent"] = {"first_ts": cycles.ts.min(), "last_ts": cycles.ts.max()}
    if cycles.ts.max() < b["validation_end"] - pd.Timedelta(days=1):
        problems.append("telemetry ends before validation_end; the declared splits cannot be evaluated")

    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import ps3_readiness  # noqa: E402
    provenance = ps3_readiness.source_provenance(telemetry)
    record["provenance"] = provenance
    if not provenance.get("verified"):
        strict("telemetry provenance is not verified: " + provenance.get("note", "")
               + " ".join(provenance.get("problems", [])))
    else:
        if provenance.get("complete_input") is not True:
            strict("telemetry is a SAMPLED export; a sample cannot be evaluated as the dataset")
        if provenance.get("context_supported") is not True:
            strict("the export records unsupported operating context")

    dev, label_problems = read_episodes(paths["development_episodes"], scope="development",
                                        validation_end=b["validation_end"])
    problems += label_problems
    if dev is not None:
        unknown = sorted(set(dev.asset_id) - set(cycles.asset_id))
        if unknown:
            problems.append(f"development labels name assets absent from the telemetry: {unknown[:5]}")
        record["development_labels"] = {"count": len(dev),
                                        "fault_bearing_trains": int(dev.train_id.nunique())}

    att = body["attestations"]
    record["attestations"] = att
    unattested = [k for k in ATTESTATIONS if att[k] is not True]
    if unattested:
        strict("human attestation(s) not given: " + ", ".join(unattested))

    # The review flags are passed as True solely to separate assess_readiness's
    # data checks from its attestation blockers, which are evaluated from the
    # configuration above. No review is being asserted here.
    readiness = assess_readiness(cycles, dev, subsystem="door", units_verified=True,
                                 identities_verified=True, fault_labels_verified=True)
    record["readiness"] = {"current_pipeline": readiness["current_pipeline"]["blockers"],
                           "labelled_evaluation": readiness["labelled_evaluation"]["blockers"]}
    for message in readiness["current_pipeline"]["blockers"]:
        strict(f"readiness: {message}")
    for message in readiness["labelled_evaluation"]["blockers"]:
        if "onset_ts" in message:
            continue                       # already reported with its reason
        if "illustrative minimum" in message:
            warnings.append(f"readiness: {message}")   # the declared policy is the gate
        else:
            strict(f"readiness: {message}")

    eligible = eligible_assets(cfg, cycles)
    absent = sorted(set(eligible) - set(cycles.asset_id))
    if absent:
        problems.append(f"reference_eligibility names assets absent from the telemetry: {absent[:5]}")
    ref = cycles[(cycles.ts >= b["reference_start"]) & (cycles.ts < b["reference_end"])
                 & cycles.asset_id.isin(eligible)]
    quality = reference_day_quality(ref)
    valid_days = quality[quality.reference_valid].groupby("asset_id").day.nunique()
    thin = sorted(a for a in eligible if valid_days.get(a, 0) < cfg["min_reference_days"])
    if thin:
        problems.append(f"{len(thin)} eligible asset(s) have fewer than {cfg['min_reference_days']} "
                        f"VALID reference days: {thin[:5]}")
    if dev is not None:
        # A consistency check, not a basis: labels can contradict eligibility,
        # but the absence of a label never establishes it.
        overlap = dev[dev.asset_id.isin(eligible) & (dev.onset_ts < b["reference_end"])
                      & (dev.fault_ts > b["reference_start"])]
        if len(overlap):
            problems.append(
                "labelled degradation overlaps the declared reference window for eligible "
                f"asset(s) {sorted(overlap.asset_id.unique())[:5]}; the labels contradict the "
                "declared eligibility")
    record["reference"] = {"eligible_assets": len(eligible), "declared": body["reference_eligibility"],
                           "declared_asset_days": len(quality),
                           "valid_asset_days": int(quality.reference_valid.sum()),
                           "excluded_for_quality": int((~quality.reference_valid).sum())}

    record.update(status="refused" if problems else "passed", problems=problems, warnings=warnings)
    if synthetic:
        record["synthetic_waived"] = waived
        record["synthetic_note"] = ("SYNTHETIC RUN. Provenance, sampling, readiness and attestation "
                                    "checks above were recorded but not enforced. Nothing in this "
                                    "run is evidence about real equipment.")
    return record, cycles, dev


def eligible_assets(cfg, cycles) -> list[str]:
    assets = cfg["body"]["reference_eligibility"]["assets"]
    return sorted(cycles.asset_id.unique()) if assets == "all" else sorted(assets)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def zoo(pipeline):
    norm = [f"{s}_hx" for s in pipeline.levels]
    return {"raw": D.RawThreshold("current_integral_as"),
            "ewma": D.EWMAChart("current_integral_as"),
            "isolation_forest": D.IsolationForestDetector(needs=norm),
            "pca": D.PCAReconstruction(needs=norm),
            "mahalanobis": D.MahalanobisDetector(needs=norm), "lof": D.LOFDetector(needs=norm),
            "headway_directional": D.Passthrough("health_index"),
            "headway_distance": D.Passthrough("anomaly_distance")}


def run_dir(cfg, run_id):
    return cfg["resolved"]["output_dir"] / "runs" / run_id


def _hash_or_none(path: Path):
    return sha256_file(path) if path.exists() else None


def frozen_inputs(cfg) -> dict:
    """Inputs the selection stage depends on. The test labels are NOT among them."""
    paths = cfg["resolved"]
    return {"telemetry_sha256": _hash_or_none(paths["telemetry"]),
            "development_episodes_sha256": _hash_or_none(paths["development_episodes"]),
            "provenance_sidecar_sha256": _hash_or_none(paths["telemetry"].with_suffix(".provenance.json"))}


def stage_select(cfg, run_id=None) -> int:
    body, b = cfg["body"], cfg["boundaries"]
    run_id = run_id or ("ps3-" + pd.Timestamp.now(tz=TIMEZONE).strftime("%Y%m%dT%H%M%S")
                        + "-" + cfg["canonical_sha256"][:8])
    directory = run_dir(cfg, run_id)
    if directory.exists():
        raise RunRefused(f"run directory already exists: {directory}. Every run uses a new directory.")
    directory.mkdir(parents=True)

    inputs = frozen_inputs(cfg)
    config_sha = write_json(directory / "config.json", {**recorded_config(cfg), "run_id": run_id})
    write_json(directory / "inputs.json", {
        "inputs": inputs, "code": code_hashes(),
        "test_episodes": {"path": str(cfg["resolved"]["test_episodes"]),
                          "opened_by_selection_stage": False,
                          "note": "not opened, read or hashed by this stage"}})

    report, cycles, dev = preflight(cfg)
    write_json(directory / "preflight.json", report)
    if report["status"] != "passed":
        raise RunRefused("preflight refused this run:\n  - " + "\n  - ".join(report["problems"])
                         + f"\nRecord kept in {directory}")

    eligible = eligible_assets(cfg, cycles)
    ref_cycles = cycles[(cycles.ts >= b["reference_start"]) & (cycles.ts < b["reference_end"])
                        & cycles.asset_id.isin(eligible)]
    quality = reference_day_quality(ref_cycles)
    keep = quality.loc[quality.reference_valid, ["asset_id", "day"]]
    ref_cycles = (ref_cycles.assign(day=ref_cycles.ts.dt.floor("D"))
                  .merge(keep, on=["asset_id", "day"]).drop(columns="day"))
    pipeline = HealthPipeline(subsystem="door", baseline_reference_days=None).fit(
        ref_cycles, fitted_at=b["reference_end"])
    labels = [c for c in ("fault_confirmed", "fault_mode") if c in cycles]
    daily = pipeline.transform(cycles.drop(columns=labels))
    daily["score_valid"] = daily.data_quality_ok.eq(True) & daily.context_supported.eq(True)
    daily["reference_valid"] = reference_quality(daily)

    candidates = zoo(pipeline)
    names = list(candidates)
    scored = D.run(candidates, daily, smooth_days=cfg["smooth_days"],
                   reference_start=b["reference_start"], reference_end=b["reference_end"],
                   eligible_assets=eligible, score_valid_column="score_valid",
                   reference_valid_column="reference_valid")
    detector_ref = D.reference_frame(daily, b["reference_start"], b["reference_end"], eligible)
    detector_ref = detector_ref[detector_ref.reference_valid]
    reference_used = {
        "preprocessing": {"cycles": len(ref_cycles), "first_ts": ref_cycles.ts.min(),
                          "last_ts": ref_cycles.ts.max(), "assets": int(ref_cycles.asset_id.nunique()),
                          "asset_days_excluded_for_quality": int((~quality.reference_valid).sum())},
        "detectors": {"asset_days": len(detector_ref), "first_available_at": detector_ref.available_at.min(),
                      "last_available_at": detector_ref.available_at.max(),
                      "assets": int(detector_ref.asset_id.nunique())},
        "preprocessing_available_from": pipeline.fitted_at,
    }
    per_asset = detector_ref.groupby("asset_id").day.nunique()
    thin = sorted(a for a in eligible if per_asset.get(a, 0) < cfg["min_reference_days"])
    if thin:
        raise RunRefused(
            f"insufficient valid reference: {len(thin)} eligible asset(s) have fewer than "
            f"{cfg['min_reference_days']} asset-days passing the reference-quality checks after "
            f"preprocessing: {thin[:5]}. Nothing was selected; record kept in {directory}")
    if not (ref_cycles.ts.min() >= b["reference_start"] and ref_cycles.ts.max() < b["reference_end"]
            and detector_ref.available_at.min() > b["reference_start"]
            and detector_ref.available_at.max() <= b["reference_end"]
            and set(detector_ref.asset_id) <= set(eligible)):
        raise RunRefused("internal check failed: a fitting frame left the declared reference window")

    grid = expected_grid(cycles, b["reference_end"], b["test_end"], body["expected_grid"],
                         b["reference_start"])
    aligned = align_scores(scored, grid, names)
    scores_path = directory / "scores.parquet"
    tmp = scores_path.with_name(scores_path.name + ".partial")
    aligned.to_parquet(tmp, index=False)
    os.replace(tmp, scores_path)
    scores_sha = sha256_file(scores_path)

    selection = qualify(aligned, dev, names, b, body["policy"])
    c1, v1 = b["calibration_end"], b["validation_end"]
    crossed = crossing(dev, c1)
    excluded = set(crossed.asset_id)
    monitoring = availability(aligned, names, c1, v1, excluded)
    for cand in selection["candidates"]:
        cand["unmatched_alerts"] = unmatched(cand)
        # One source of truth: the gate's score_fraction IS availability over
        # expected asset-days. Refuse to report if the two ever disagree.
        if "score_fraction" in cand and not np.isclose(
                cand["score_fraction"], monitoring["availability"][cand["name"]] or 0.0):
            raise RunRefused(f"internal check failed: qualification and availability disagree for "
                             f"{cand['name']}")
    complete = dev[(dev.onset_ts > c1) & ~dev.asset_id.isin(excluded)]
    report_body = {
        "run_id": run_id, "stage": "select", "selection": selection,
        "labels_used": "development_episodes only; test_episodes was not opened",
        "episodes": {"development_labels": len(dev),
                     "before_calibration_end": int((dev.fault_ts <= c1).sum()),
                     "complete_in_validation": len(complete),
                     "excluded_crossing_calibration_boundary": len(crossed),
                     "excluded_crossing_assets": sorted(excluded)},
        "validation_monitoring": {**monitoring, "window": {"after": c1, "through": v1},
                                  "excluded_assets": sorted(excluded),
                                  "note": ("identical to each candidate's score_fraction: both are "
                                           "supported scores over EXPECTED asset-days")},
        "reference_used": reference_used,
        "outcome": ("a candidate qualified on validation only; it is not promoted anywhere"
                    if selection["selected_model"] else
                    "no candidate met the qualification policy. This is a valid outcome, and the "
                    "final test will record that nothing was evaluated."),
    }
    selection_sha = write_json(directory / "selection.json", report_body)
    write_json(directory / "frozen.json", {
        "run_id": run_id, "frozen_at": pd.Timestamp.now(tz=TIMEZONE),
        "config_json_sha256": config_sha, "configuration_canonical_sha256": cfg["canonical_sha256"],
        "selection_json_sha256": selection_sha, "scores_parquet_sha256": scores_sha,
        "inputs": inputs, "code": code_hashes(),
        "note": ("Selection was frozen here. The selection stage opened only the development "
                 "label file; the test label file named in the configuration was not opened, "
                 "read or hashed before this point. The test stage refuses to run if any hash "
                 "below has changed.")})
    print(f"Selection frozen in {directory}")
    print(f"  selected: {selection['selected_model'] or 'none qualified'}")
    return 0


def stage_test(cfg, run_id) -> int:
    body, b = cfg["body"], cfg["boundaries"]
    directory = run_dir(cfg, run_id)
    frozen_path = directory / "frozen.json"
    if not frozen_path.exists():
        raise RunRefused(f"no frozen selection in {directory}; run --stage select first")
    if (directory / "test.json").exists():
        raise RunRefused(f"{directory / 'test.json'} already exists; a final test is run once")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    problems = []
    for label, now, then in (
            ("configuration", cfg["canonical_sha256"], frozen.get("configuration_canonical_sha256")),
            ("config.json", _hash_or_none(directory / "config.json"), frozen.get("config_json_sha256")),
            ("selection.json", _hash_or_none(directory / "selection.json"), frozen.get("selection_json_sha256")),
            ("scores.parquet", _hash_or_none(directory / "scores.parquet"), frozen.get("scores_parquet_sha256"))):
        if now is None or now != then:
            problems.append(f"{label} has changed or is missing since selection was frozen")
    recorded = frozen.get("inputs", {})
    for key, now in frozen_inputs(cfg).items():
        if key not in recorded:
            problems.append(f"the freeze does not record {key}; it predates complete input binding")
        elif now != recorded[key]:
            problems.append(f"{key.removesuffix('_sha256').replace('_', ' ')} input has changed "
                            "since selection was frozen")
    for rel, digest in code_hashes().items():
        if frozen.get("code", {}).get(rel) != digest:
            problems.append(f"code changed since selection was frozen: {rel}")
    if problems:
        raise RunRefused("refusing the final test; the frozen selection is incompatible:\n  - "
                         + "\n  - ".join(problems))

    selected = json.loads((directory / "selection.json").read_text(encoding="utf-8"))["selection"]
    exposure = body["data_exposure"]
    statement = ("This test period was previously inspected, so this is NOT an untouched hold-out."
                 if exposure["test_period_previously_inspected"] else
                 "Declared by " + exposure["declared_by"] + " as not previously inspected. This "
                 "runner cannot verify that declaration.")
    out = {"run_id": run_id, "stage": "test", "exposure_statement": statement,
           "window": {"after": b["validation_end"], "through": b["test_end"]}}
    if selected["selected_model"] is None:
        out.update(status="not_evaluated",
                   reason="no candidate qualified on validation; there is nothing to test, and "
                          "the test labels were not opened")
        write_json(directory / "test.json", out)
        print("No qualified candidate; test recorded as not evaluated.")
        return 0

    test_path, dev_path = cfg["resolved"]["test_episodes"], cfg["resolved"]["development_episodes"]
    if test_path.exists() and dev_path.exists() and os.path.samefile(test_path, dev_path):
        raise RunRefused("test_episodes and development_episodes are the same file on disk")
    test_eps, label_problems = read_episodes(test_path, scope="test", validation_end=b["validation_end"])
    if label_problems:
        raise RunRefused("test labels are not usable:\n  - " + "\n  - ".join(label_problems))
    out["test_labels_sha256"] = sha256_file(test_path)

    aligned = pd.read_parquet(directory / "scores.parquet")
    t0, t1 = b["validation_end"], b["test_end"]
    known = test_eps[test_eps.fault_ts <= t1]
    crossed_start, crossed_end = crossing(known, t0), crossing(test_eps, t1)
    excluded = set(crossed_start.asset_id) | set(crossed_end.asset_id)
    scoped = aligned[(aligned.available_at <= t1) & ~aligned.asset_id.isin(set(crossed_end.asset_id))]
    result = assess_selected(selected, scoped, known)
    name = selected["selected_model"]
    monitoring = availability(aligned, [name], t0, t1, excluded)
    if "score_fraction" in result:
        result["unmatched_alerts"] = unmatched(result)
        if not np.isclose(result["score_fraction"], monitoring["availability"][name] or 0.0):
            raise RunRefused("internal check failed: final metrics and availability disagree")

    eligible = body["reference_eligibility"]["assets"]
    contaminated = test_eps[(test_eps.onset_ts < b["reference_end"]) & (test_eps.fault_ts > b["reference_start"])]
    if eligible != "all":
        contaminated = contaminated[contaminated.asset_id.isin(eligible)]
    out.update(
        status=result.get("status"), result=result,
        episodes={"complete_in_test": int(len(known[(known.onset_ts > t0) & ~known.asset_id.isin(excluded)])),
                  "excluded_crossing_validation_boundary": len(crossed_start),
                  "excluded_crossing_test_end": len(crossed_end),
                  "excluded_assets": sorted(excluded)},
        test_monitoring={**monitoring, "excluded_assets": sorted(excluded),
                         "note": "identical to score_fraction: supported scores over EXPECTED asset-days"},
        reference_contamination_found_in_test_labels=sorted(contaminated.asset_id.unique()),
    )
    if len(contaminated):
        out["reference_contamination_note"] = (
            "Test labels show degradation beginning inside the declared reference window for these "
            "eligible assets. The selection cannot be revised, but its reference was not clean for "
            "them, and that must be reported alongside this result.")
    write_json(directory / "test.json", out)
    print(f"Final test recorded in {directory / 'test.json'}")
    return 0
