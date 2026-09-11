"""Experimental repair outcome checks, independent of dashboard job state.

Inputs are daily condition-normalised channels from one frozen model. Thresholds
are illustrative policy, not calibrated probabilities or release-to-service rules.
Naive input timestamps mean Asia/Singapore; internal timestamps are UTC.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pandas as pd


def timestamp(value):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("timestamp is required")
    return t.tz_localize("Asia/Singapore").tz_convert("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def text_required(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class VerificationPolicy:
    reference_days: int = 14
    post_days: int = 3
    settling_days: int = 1
    deviation_limit: float = 3.0
    persistent_days: int = 2

    def __post_init__(self):
        for name in ("reference_days", "post_days", "settling_days", "persistent_days"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if (self.reference_days < 5 or self.post_days < 2 or self.settling_days < 0
                or not 1 <= self.persistent_days <= self.post_days
                or not np.isfinite(self.deviation_limit) or self.deviation_limit <= 0):
            raise ValueError("invalid verification policy")


def daily_frame(daily, asset_id, channels, contexts):
    required = {"asset_id", "day", "available_at", "data_quality_ok", "context_supported", *channels, *contexts}
    if required - set(daily):
        raise ValueError(f"missing columns: {sorted(required - set(daily))}")
    d = daily.loc[daily.asset_id == asset_id, sorted(required)].copy()
    for col in ("day", "available_at"):
        d[col] = pd.to_datetime(d[col].map(timestamp), utc=True)
    if d.day.duplicated().any():
        raise ValueError("duplicate asset-days")
    if (d.available_at < d.day + pd.Timedelta(days=1)).any():
        raise ValueError("full-day telemetry cannot be available before its day ends")
    if not d.day.eq(d.day.dt.tz_convert("Asia/Singapore").dt.floor("D").dt.tz_convert("UTC")).all():
        raise ValueError("day must be midnight in Asia/Singapore")
    return d.sort_values("day")


def usable(d, channels, contexts):
    # Do not turn text "false" or missing booleans into true.
    quality = d.data_quality_ok.eq(True) & d.context_supported.eq(True)
    values = d[list(channels) + list(contexts)].apply(pd.to_numeric, errors="coerce")
    return quality & np.isfinite(values).all(axis=1)


def make_reference(daily, *, asset_id, channels, contexts, model_version,
                   reviewed_by, reviewed_at, policy=VerificationPolicy()):
    """Freeze a caller-selected, inspection-reviewed healthy reference.

    Review is an explicit attestation supplied by the caller, not inferred from
    quiet telemetry. Daily rows must be consecutive, supported and available.
    """
    for name, value in (("asset_id", asset_id), ("model_version", model_version), ("reviewed_by", reviewed_by)):
        text_required(value, name)
    if not channels or not contexts or len(set(channels)) != len(channels) or set(channels) & set(contexts):
        raise ValueError("distinct channels and operating-context columns are required")
    d = daily_frame(daily, asset_id, channels, contexts)
    when = timestamp(reviewed_at)
    if len(d) < policy.reference_days or not usable(d, channels, contexts).all():
        raise ValueError("insufficient or invalid healthy reference")
    if not d.day.diff().dropna().eq(pd.Timedelta(days=1)).all() or (d.available_at > when).any():
        raise ValueError("reference must be consecutive and available at review")
    location, scale = {}, {}
    for c in channels:
        values = pd.to_numeric(d[c])
        location[c] = float(values.median())
        scale[c] = float(1.4826 * (values - location[c]).abs().median())
        if not np.isfinite(scale[c]) or scale[c] <= 1e-9:
            raise ValueError(f"degenerate reference scale: {c}")
    rows = json.loads(d.to_json(orient="records", date_format="iso"))
    ref = {"asset_id": asset_id, "model_version": model_version, "channels": list(channels),
           "contexts": list(contexts), "location": location, "scale": scale,
           "context_ranges": {c: [float(pd.to_numeric(d[c]).min()), float(pd.to_numeric(d[c]).max())] for c in contexts},
           "start": d.day.min().isoformat(), "end": d.available_at.max().isoformat(),
           "reference_days": len(d), "reviewed_by": reviewed_by, "reviewed_at": when.isoformat(),
           "source_hash": fingerprint(rows), "policy": asdict(policy)}
    ref["reference_id"] = fingerprint(ref)
    return ref


def maintenance_record(*, job_id, asset_id, started_at, completed_at, recorded_at,
                       operator, finding, work_performed):
    values = locals().copy()
    for name in ("job_id", "asset_id", "operator", "finding", "work_performed"):
        text_required(values[name], name)
    times = [timestamp(values[k]) for k in ("started_at", "completed_at", "recorded_at")]
    if not times[0] <= times[1] <= times[2]:
        raise ValueError("maintenance timestamps must be start <= completion <= recording")
    for name, t in zip(("started_at", "completed_at", "recorded_at"), times):
        values[name] = t.isoformat()
    return values


def verify_repair(record, reference, daily, *, as_of, model_version, subsequent_maintenance_at=None):
    """Return review evidence only. Never alter aspects, RUL or service status."""
    record = maintenance_record(**record)
    as_of = timestamp(as_of)
    ref_content = {k: v for k, v in reference.items() if k != "reference_id"}
    if fingerprint(ref_content) != reference.get("reference_id"):
        raise ValueError("reference fingerprint mismatch")
    policy = VerificationPolicy(**reference["policy"])
    result = {"job_id": record["job_id"], "asset_id": record["asset_id"],
              "as_of": as_of.isoformat(), "model_version": model_version,
              "reference_id": reference["reference_id"], "policy": asdict(policy),
              "status": "insufficient_evidence", "reason": "", "channels": {},
              "observed_days": 0, "follow_up_required": False,
              "release_to_service": False,
              "limitation": "Experimental signal check; not causal proof of repair effectiveness or service clearance."}

    def insufficient(reason):
        result["reason"] = reason
        return result

    if reference["asset_id"] != record["asset_id"]:
        return insufficient("reference belongs to another asset")
    if model_version != reference["model_version"]:
        return insufficient("model changed; rebuild and review a comparable reference")
    if timestamp(record["recorded_at"]) > as_of:
        return insufficient("maintenance record was not available at assessment time")
    if max(timestamp(reference["end"]), timestamp(reference["reviewed_at"])) > timestamp(record["started_at"]):
        return insufficient("healthy reference must be available and reviewed before this maintenance")
    if subsequent_maintenance_at is not None and timestamp(subsequent_maintenance_at) <= as_of:
        return insufficient("subsequent maintenance prevents attributing this observation window to this job")
    # Start with full days after completion and an explicit settling interval.
    local_completion = timestamp(record["completed_at"]).tz_convert("Asia/Singapore")
    first = local_completion.ceil("D") + pd.Timedelta(days=policy.settling_days)
    end = as_of.tz_convert("Asia/Singapore").floor("D")
    days = pd.date_range(end=end - pd.Timedelta(days=1), periods=policy.post_days, freq="D").tz_convert("UTC")
    if days[0] < first.tz_convert("UTC"):
        return insufficient("waiting for settling period and complete post-maintenance days")
    d = daily_frame(daily, record["asset_id"], reference["channels"], reference["contexts"])
    d = d[d.day.isin(days) & (d.available_at <= as_of)]
    result["observed_days"] = len(d)
    result["window_start"], result["window_end"] = days[0].isoformat(), end.tz_convert("UTC").isoformat()
    if len(d) != policy.post_days or not usable(d, reference["channels"], reference["contexts"]).all():
        return insufficient("recent window contains missing, incomplete or unsupported telemetry")
    for c, (lo, hi) in reference["context_ranges"].items():
        if not pd.to_numeric(d[c]).between(lo, hi).all():
            return insufficient(f"operating context is outside the healthy reference: {c}")
    result["evidence_hash"] = fingerprint(json.loads(d.to_json(orient="records", date_format="iso")))
    recovered, persistent = True, []
    for c in reference["channels"]:
        z = (pd.to_numeric(d[c]) - reference["location"][c]).abs() / reference["scale"][c]
        count = int(z.gt(policy.deviation_limit).sum())
        result["channels"][c] = {"median_absolute_deviation": float(z.median()),
                                 "max_absolute_deviation": float(z.max()), "abnormal_days": count}
        recovered &= count == 0
        if count >= policy.persistent_days:
            persistent.append(c)
    if persistent:
        result.update(status="abnormality_persists", reason="Repeated reference departures: " + ", ".join(persistent),
                      follow_up_required=True)
    elif recovered:
        result.update(status="signal_recovered", reason="All monitored channels remain within reference limits throughout the recent window")
    else:
        result["reason"] = "mixed post-maintenance readings; continue observation or inspect"
    return result


class RepairStore:
    """Local SQLite maintenance records, assessment history and open follow-ups.

    Transactions support multiple writers. This is not authenticated or tamper
    proof; no method edits/deletes a stored maintenance record or assessment.
    """
    def __init__(self, path):
        self.path = str(Path(path))
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS repair_records(job_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS repair_assessments(id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES repair_records(job_id), body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS repair_followups(job_id TEXT PRIMARY KEY REFERENCES repair_records(job_id), assessment_id TEXT NOT NULL REFERENCES repair_assessments(id), state TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def record(self, **kwargs):
        record = maintenance_record(**kwargs)
        with self.connect() as db:
            db.execute("INSERT INTO repair_records VALUES (?, ?, ?)",
                       (record["job_id"], record["asset_id"], canonical(record)))
        return record

    def assess(self, job_id, reference, daily, *, as_of, model_version):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM repair_records WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            record = json.loads(row[0])
            other = [json.loads(r[0]) for r in db.execute(
                "SELECT body FROM repair_records WHERE asset_id=? AND job_id<>?", (record["asset_id"], job_id))]
            later = [timestamp(r["started_at"]) for r in other
                     if timestamp(r["recorded_at"]) <= timestamp(as_of)
                     and timestamp(r["started_at"]) >= timestamp(record["started_at"])]
            result = verify_repair(record, reference, daily, as_of=as_of, model_version=model_version,
                                   subsequent_maintenance_at=min(later) if later else None)
            result["assessment_id"] = str(uuid.uuid4())
            # Persist exact reference parameters with the evidence, not just a pointer.
            result["reference_snapshot"] = reference
            db.execute("INSERT INTO repair_assessments VALUES (?, ?, ?)",
                       (result["assessment_id"], job_id, canonical(result)))
            if result["follow_up_required"]:
                db.execute("INSERT OR IGNORE INTO repair_followups VALUES (?, ?, 'open')",
                           (job_id, result["assessment_id"]))
        return result

    def history(self, job_id):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT body FROM repair_assessments WHERE job_id=? ORDER BY rowid", (job_id,))]

    def followups(self):
        with self.connect() as db:
            return [dict(zip(("job_id", "assessment_id", "state"), r)) for r in db.execute("SELECT * FROM repair_followups ORDER BY rowid")]
