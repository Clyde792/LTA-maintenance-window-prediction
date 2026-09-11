"""Read-only readiness checks for mapped organiser data, before fitting models."""
import numpy as np
import pandas as pd
from .contract import get


def assess_readiness(cycles, episodes=None, *, subsystem="door", units_verified=False,
                     identities_verified=False, fault_labels_verified=False, minimum_fault_groups=5):
    signals = list(get(subsystem).signals)
    contexts = ["ambient_temp_c", "load_proxy", "hour_of_day"]
    required = ["ts", "asset_id", "train_id", *signals, *contexts]
    missing = sorted(set(required) - set(cycles))
    blockers, warnings = [], []
    if missing:
        blockers.append("mapped columns missing: " + ", ".join(missing))
    if not units_verified:
        blockers.append("source units and conversions need review")
    if not identities_verified:
        blockers.append("telemetry-to-asset identities need review")
    if cycles.empty:
        blockers.append("no telemetry rows")
    null_fraction = {}
    for c in signals + contexts:
        if c in cycles:
            v = pd.to_numeric(cycles[c], errors="coerce")
            null_fraction[c] = float((~np.isfinite(v)).mean()) if len(v) else None
            if len(v) and not np.isfinite(v).any():
                blockers.append(f"no finite values for {c}")
            elif len(v) and (~np.isfinite(v)).any():
                warnings.append(f"{c}: missing/nonfinite observations need explicit abstention")
    bad_time, duplicates, observed_days = 0, 0, 0
    if "ts" in cycles:
        ts = pd.to_datetime(cycles.ts, errors="coerce", utc=True)
        bad_time = int(ts.isna().sum())
        observed_days = int(ts.dt.floor("D").nunique())
        if bad_time:
            blockers.append("unparseable or missing telemetry timestamps")
        if observed_days < 30:
            blockers.append("less than 30 observed days for the current reference configuration")
        if "asset_id" in cycles:
            duplicates = int(pd.DataFrame({"asset_id": cycles.asset_id, "ts": ts}).duplicated().sum())
            if duplicates:
                blockers.append("duplicate asset timestamps require cycle/event identity resolution")
    for c in ("asset_id", "train_id"):
        if c in cycles and (cycles[c].isna() | cycles[c].astype(str).str.strip().eq("")).any():
            blockers.append(f"missing {c}")
    if {"asset_id", "train_id"} <= set(cycles):
        if cycles.groupby("asset_id").train_id.nunique().gt(1).any():
            blockers.append("asset maps to multiple trains; resolve identity/service history")
    if "context_supported" in cycles and not cycles.context_supported.eq(True).all():
        blockers.append("imputed or unsupported context requires review before current-pipeline fitting")
    evaluation_blockers = []
    groups = 0
    if not fault_labels_verified:
        evaluation_blockers.append("fault meaning, confirmation time and annotation coverage need review")
    label_columns = {"asset_id", "train_id", "onset_ts", "fault_ts"}
    if episodes is None or episodes.empty:
        evaluation_blockers.append("no verified fault episodes supplied")
    elif not label_columns <= set(episodes):
        evaluation_blockers.append("episode-based lead-time evaluation needs asset/train identity, onset_ts and fault_ts")
    else:
        onset = pd.to_datetime(episodes.onset_ts, errors="coerce", utc=True)
        fault = pd.to_datetime(episodes.fault_ts, errors="coerce", utc=True)
        if (onset.isna() | fault.isna() | onset.ge(fault)).any():
            evaluation_blockers.append("invalid onset/fault ordering or timestamps")
        if episodes[["asset_id", "train_id"]].isna().any().any():
            evaluation_blockers.append("fault records have missing asset/train identities")
        if "asset_id" in cycles and not episodes.asset_id.isin(cycles.asset_id).all():
            evaluation_blockers.append("fault assets cannot all be joined to telemetry")
        groups = int(episodes.train_id.nunique())
    if groups < minimum_fault_groups:
        evaluation_blockers.append(f"only {groups} fault-bearing train groups; illustrative minimum is {minimum_fault_groups}")
    return {"subsystem": subsystem, "telemetry_rows": len(cycles), "observed_days": observed_days,
        "missing_columns": missing, "nonfinite_fraction": null_fraction,
        "invalid_timestamps": bad_time, "duplicate_asset_timestamps": duplicates,
        "current_pipeline": {"ready_for_reviewed_trial": not blockers, "blockers": blockers},
        "labelled_evaluation": {"ready_for_reviewed_trial": not blockers and not evaluation_blockers,
                                "fault_groups": groups, "blockers": evaluation_blockers},
        "warnings": warnings,
        "limitations": ["Passing is not evidence that the reference is healthy or the model is applicable.",
            "Current thresholds, context assumptions and cycle aggregation must be reviewed against the supplied schema.",
            "No labels means no supervised accuracy, warning-time or RUL validation claim.",
            "Bogie contract availability is not a demonstrated bogie detector."]}
