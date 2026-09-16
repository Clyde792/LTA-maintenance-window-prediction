"""The reviewed mapping configuration: the one file a human signs off.

A suggested mapping is a guess. This module is the seam where a guess becomes a
decision somebody is accountable for, so the format is deliberately strict:

  * `reviewed: true` is REQUIRED. A draft written by the suggester carries
    `reviewed: false` and is refused until a person changes it, having checked
    the entries the suggester flagged.
  * Unknown top-level keys are an error, not ignored. A typo in `unit_scales`
    that silently did nothing would be a wrong-units export that looked fine.
  * Every declared target must be a real contract column for that subsystem, and
    no source column may be claimed twice.

What this file does NOT carry, on purpose: any attestation that units, asset
identities or fault labels have been *verified*. Those are separate human
reviews, asserted on the `scripts/ps3_readiness.py` command line, and conflating
them with schema configuration would let one review stand in for three.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from .. import contract
from .base import Adapter

ALLOWED_KEYS = {"reviewed", "subsystem", "columns", "unit_scales", "source_timezone",
                "timestamp_format", "boolean_tokens", "reviewed_by", "reviewed_at", "notes"}
REQUIRED_KEYS = {"reviewed", "subsystem", "columns"}


class MappingError(ValueError):
    """A mapping configuration that must be corrected before anything is exported."""


def _object_or_absent(body: dict, key: str) -> dict:
    """An optional object field, where only absence or null means "not set".

    `body.get(key) or {}` would quietly accept `false`, `0`, `""` and `[]` as an
    empty configuration. A mapping that says `"boolean_tokens": false` has a
    mistake in it, and a mistake in a label configuration is the kind that
    reaches an evaluation before anybody notices.
    """
    if key not in body or body[key] is None:
        return {}
    value = body[key]
    if not isinstance(value, dict):
        raise MappingError(f"{key} must be a JSON object, or null/absent; got "
                           f"{type(value).__name__}")
    return value


@dataclass(frozen=True)
class MappingConfig:
    subsystem: str
    columns: dict[str, str]
    unit_scales: dict[str, float] = field(default_factory=dict)
    source_timezone: str | None = None
    timestamp_format: str | None = None
    boolean_tokens: dict[str, bool] = field(default_factory=dict)
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    notes: str | None = None
    file_sha256: str | None = None
    canonical_sha256: str | None = None
    path: str | None = None

    def adapter(self) -> Adapter:
        return Adapter(subsystem=self.subsystem, column_map=dict(self.columns),
                       unit_scales=dict(self.unit_scales),
                       source_timezone=self.source_timezone,
                       timestamp_format=self.timestamp_format,
                       boolean_tokens=dict(self.boolean_tokens))

    def as_provenance(self) -> dict:
        """What the export records about the mapping it was told to use."""
        return {"path": self.path, "file_sha256": self.file_sha256,
                "canonical_sha256": self.canonical_sha256, "reviewed": True,
                "reviewed_by": self.reviewed_by, "reviewed_at": self.reviewed_at,
                "subsystem": self.subsystem, "notes": self.notes}


def canonical_bytes(body: dict) -> bytes:
    """Key-order-independent bytes, so a reformatted file hashes the same."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def parse_mapping(body: dict, *, path: str | None = None,
                  file_sha256: str | None = None) -> MappingConfig:
    if not isinstance(body, dict):
        raise MappingError("mapping configuration must be a JSON object")
    unknown = sorted(set(body) - ALLOWED_KEYS)
    if unknown:
        raise MappingError(f"unknown mapping key(s): {', '.join(unknown)}. "
                           f"Allowed: {', '.join(sorted(ALLOWED_KEYS))}")
    missing = sorted(REQUIRED_KEYS - set(body))
    if missing:
        raise MappingError(f"mapping is missing required key(s): {', '.join(missing)}")

    if body["reviewed"] is not True:
        raise MappingError(
            "this mapping is not marked reviewed. A suggested mapping is a guess: "
            "check every entry the suggester flagged, then set \"reviewed\": true.")

    subsystem = body["subsystem"]
    if subsystem not in contract.SUBSYSTEMS:
        raise MappingError(f"unknown subsystem {subsystem!r}; "
                           f"registered: {sorted(contract.SUBSYSTEMS)}")
    sub = contract.get(subsystem)

    columns = body["columns"]
    if not isinstance(columns, dict) or not columns:
        raise MappingError("columns must be a non-empty object of contract_column -> source_column")
    bad_targets = sorted(set(columns) - set(sub.columns))
    if bad_targets:
        raise MappingError(f"not contract columns for subsystem {subsystem!r}: "
                           f"{', '.join(bad_targets)}")
    for target, source in columns.items():
        if not isinstance(source, str) or not source.strip():
            raise MappingError(f"source column for {target!r} must be a non-empty string")
    duplicates = sorted({s for s in columns.values() if list(columns.values()).count(s) > 1})
    if duplicates:
        raise MappingError("one source column is mapped to several contract columns: "
                           + ", ".join(duplicates))

    scales = _object_or_absent(body, "unit_scales")
    convertible = set(sub.signals) | set(contract.CONTEXT)
    for col, value in scales.items():
        if col not in convertible:
            raise MappingError(f"unit_scales names {col!r}, which is not a signal or context "
                               f"column for subsystem {subsystem!r}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not np.isfinite(value) or value <= 0:
            raise MappingError(f"unit scale for {col!r} must be a finite positive number")

    tz = body.get("source_timezone")
    if tz is not None:
        if not isinstance(tz, str) or not tz.strip():
            raise MappingError("source_timezone must be an IANA zone name, e.g. \"UTC\"")
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            raise MappingError(f"unknown source_timezone {tz!r}; use an IANA name such as "
                               "\"UTC\" or \"Asia/Singapore\"") from None

    fmt = body.get("timestamp_format")
    if fmt is not None and (not isinstance(fmt, str) or not fmt.strip()):
        raise MappingError("timestamp_format must be a non-empty strptime format string")

    tokens = _object_or_absent(body, "boolean_tokens")
    # Tokens are matched after strip+lower, so "Y" and " y " are the same token.
    # Letting the later one win would decide, silently and by file order, whether
    # a labelled fault is a fault.
    normalised: dict[str, bool] = {}
    origin: dict[str, str] = {}
    for token, value in tokens.items():
        if not isinstance(token, str) or not token.strip():
            raise MappingError("boolean_tokens keys must be non-empty strings")
        if not isinstance(value, bool):
            raise MappingError(f"boolean token {token!r} must map to true or false")
        key = token.strip().lower()
        if key in normalised and normalised[key] != value:
            raise MappingError(
                f"boolean tokens {origin[key]!r} and {token!r} both normalise to {key!r} "
                f"but map to different values; remove one")
        if key in Adapter.BASE_BOOLEAN_TOKENS and Adapter.BASE_BOOLEAN_TOKENS[key] != value:
            raise MappingError(
                f"boolean token {token!r} normalises to {key!r}, which the adapter already "
                f"reads as {Adapter.BASE_BOOLEAN_TOKENS[key]}; redefining it would invert "
                "every row that uses it")
        normalised[key] = value
        origin[key] = token

    return MappingConfig(
        subsystem=subsystem, columns=dict(columns), unit_scales={k: float(v) for k, v in scales.items()},
        source_timezone=tz, timestamp_format=fmt, boolean_tokens=normalised,
        reviewed_by=body.get("reviewed_by"), reviewed_at=body.get("reviewed_at"),
        notes=body.get("notes"), path=path, file_sha256=file_sha256,
        canonical_sha256=hashlib.sha256(canonical_bytes(body)).hexdigest())


def load_mapping(path: str | Path) -> MappingConfig:
    p = Path(path)
    if not p.exists():
        raise MappingError(f"mapping file not found: {p}")
    raw = p.read_bytes()
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MappingError(f"{p} is not readable JSON: {exc}") from None
    return parse_mapping(body, path=p.name,
                         file_sha256=hashlib.sha256(raw).hexdigest())


def draft(subsystem: str, columns: dict[str, str], scores: dict[str, float] | None = None) -> dict:
    """A DRAFT mapping for review. Never usable as-is: `reviewed` is false.

    The point of writing it at all is that a reviewer edits a file rather than
    retyping fifteen column names, and the entries the suggester was unsure of
    arrive already listed.
    """
    scores = scores or {}
    flagged = sorted(t for t, v in scores.items() if v < 1.0 and t in columns)
    return {
        "reviewed": False,
        "subsystem": subsystem,
        "columns": dict(columns),
        "unit_scales": {},
        "source_timezone": None,
        "timestamp_format": None,
        "boolean_tokens": {},
        "reviewed_by": None,
        "reviewed_at": None,
        "notes": ("DRAFT produced by scripts/bind_data.py. These are GUESSES. Check every "
                  "entry against the source, add unit_scales for any column not already in "
                  "contract units, declare source_timezone and timestamp_format, then set "
                  "reviewed to true."
                  + (" Low-confidence guesses to check first: " + ", ".join(flagged) + "."
                     if flagged else "")),
    }
