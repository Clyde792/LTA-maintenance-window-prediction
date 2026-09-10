"""
Headway - the adapter layer. The only code that touches vendor column names.

This exists so that Friday 18 Sep at 20:00 is a MAPPING exercise, not an
engineering one. Everything downstream of here speaks the contract; the adapter
is the single seam where somebody else's schema meets ours.

Three things it does, in order of how much time each saves on the night:

  SUGGEST    given a real file, guess the mapping from an alias table plus fuzzy
             matching, and print a ready-to-paste dict. Turns "read 40 column
             names and think" into "check 15 guesses".
  DERIVE     fill contract columns that are absent but computable. Missing
             current_integral_as but have mean current and duration? Multiply.
             Missing hour_of_day? Take it from the timestamp. Cheap, and each
             one is a column we do not have to go without.
  DEGRADE    when a required CONTEXT column simply does not exist in their data,
             fill it with a constant and SAY SO, loudly. A constant column makes
             its normalisation term inert rather than crashing the pipeline -
             we lose the correction for that confound, we do not lose the run.
             At 02:00 the right failure mode is "works, worse, and tells you",
             not "raises".

Nothing here guesses silently. Every substitution is reported.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .. import contract

# --------------------------------------------------------------------------
# Alias table
# --------------------------------------------------------------------------
# Plausible vendor names for each contract column, drawn from the door-cycle and
# rolling-stock condition-monitoring literature. Matching is done on a squashed
# form (lowercase, alphanumerics only), so "Peak Current (A)" hits "peakcurrenta".

ALIASES: dict[str, list[str]] = {
    "ts": ["timestamp", "time", "datetime", "date_time", "cycle_start", "event_time",
           "recorded_at", "log_time", "sample_time", "start_time"],
    "asset_id": ["asset", "asset_no", "asset_code", "equipment_id", "door_id", "unit_id",
                 "component_id", "device_id", "dcu_id", "serial"],
    "train_id": ["train", "train_no", "train_number", "vehicle_id", "car_id", "car_no",
                 "set_id", "set_no", "rake_id", "trainset", "fleet_no"],
    "subsystem": ["system", "asset_type", "component", "equipment_type"],

    "cycle_duration_s": ["duration", "cycle_time", "open_time", "close_time", "travel_time",
                         "t_cycle", "operation_time", "movement_time", "elapsed"],
    "peak_current_a": ["i_peak", "max_current", "peak_i", "current_max", "imax",
                       "peak_motor_current", "current_peak"],
    "mean_current_a": ["i_mean", "avg_current", "current_avg", "imean", "average_current",
                       "mean_motor_current"],
    "current_integral_as": ["i_integral", "charge", "current_integral", "integral_current",
                            "amp_seconds", "as", "work_done", "energy"],
    "travel_mm": ["travel", "distance", "leaf_travel", "stroke", "position_max",
                  "displacement", "door_travel"],
    "obstruction_flag": ["obstruction", "obstacle", "blocked", "reopen_flag", "obstruct",
                         "obstruction_detected", "is_obstructed"],
    "retry_count": ["retries", "reopen_count", "attempts", "n_retry", "retry",
                    "recycle_count"],

    "bearing_temp_rise_c": ["bearing_temp", "temp_rise", "brg_temp", "bearing_temperature"],
    "vibration_rms_g": ["vib_rms", "rms", "vibration", "accel_rms", "rms_g"],
    "vibration_kurtosis": ["kurtosis", "vib_kurtosis", "kurt"],
    "suspension_defl_mm": ["deflection", "susp_defl", "suspension", "air_spring"],
    "wheel_impact_g": ["impact", "wheel_impact", "shock", "peak_accel"],

    "ambient_temp_c": ["temp", "temperature", "ambient", "saloon_temp", "t_ambient",
                       "air_temp", "ambient_temperature", "cabin_temp"],
    "load_proxy": ["load", "crowding", "occupancy", "passenger_load", "dwell", "dwell_time",
                   "load_weigh", "passenger_count", "pax", "loading"],
    "hour_of_day": ["hour", "hr", "tod", "time_of_day"],
    "cycles_since_service": ["cycles_since_maint", "service_cycles", "usage", "cycle_count",
                             "operations_since_service", "km_since_service", "mileage"],

    "fault_confirmed": ["fault", "failure", "confirmed_fault", "is_fault", "defect",
                        "fault_flag", "verified_fault", "breakdown"],
    "fault_mode": ["mode", "failure_mode", "fault_type", "defect_type", "fault_desc",
                   "failure_cause", "root_cause"],
}


def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def suggest_mapping(columns: list[str], subsystem: str,
                    cutoff: float = 0.72) -> tuple[dict[str, str], dict[str, float]]:
    """Guess contract_column -> source_column. Returns (mapping, confidence).

    Exact alias hits score 1.0; fuzzy hits carry their similarity ratio so the
    dry-run report can flag the ones worth checking by eye. A source column is
    never assigned twice.
    """
    sub = contract.get(subsystem)
    squashed = {_squash(c): c for c in columns}
    mapping: dict[str, str] = {}
    score: dict[str, float] = {}
    taken: set[str] = set()

    for target in sub.columns:
        candidates = [target] + ALIASES.get(target, [])
        # 1. exact match against the target name or any alias
        for cand in candidates:
            key = _squash(cand)
            if key in squashed and squashed[key] not in taken:
                mapping[target] = squashed[key]
                score[target] = 1.0
                taken.add(squashed[key])
                break
        if target in mapping:
            continue
        # 2. fuzzy, against the same candidate pool
        pool = [k for k in squashed if squashed[k] not in taken]
        best, best_ratio = None, 0.0
        for cand in candidates:
            for m in difflib.get_close_matches(_squash(cand), pool, n=1, cutoff=cutoff):
                r = difflib.SequenceMatcher(None, _squash(cand), m).ratio()
                if r > best_ratio:
                    best, best_ratio = m, r
        if best is not None:
            mapping[target] = squashed[best]
            score[target] = round(best_ratio, 2)
            taken.add(squashed[best])
    return mapping, score


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------
# Each entry: target -> (required source columns, function, human explanation).
# Applied only when the target is missing and every input is present.

Derivation = tuple[list[str], Callable[[pd.DataFrame], pd.Series], str]

DERIVATIONS: dict[str, Derivation] = {
    "hour_of_day": (
        ["ts"],
        lambda d: d["ts"].dt.hour + d["ts"].dt.minute / 60.0,
        "from the timestamp",
    ),
    "current_integral_as": (
        ["mean_current_a", "cycle_duration_s"],
        lambda d: d["mean_current_a"] * d["cycle_duration_s"],
        "mean current x duration",
    ),
    "mean_current_a": (
        ["current_integral_as", "cycle_duration_s"],
        lambda d: d["current_integral_as"] / d["cycle_duration_s"].replace(0, np.nan),
        "current integral / duration",
    ),
    "train_id": (
        ["asset_id"],
        lambda d: d["asset_id"].astype(str).str.split(r"[-_ ]").str[0],
        "leading token of asset_id",
    ),
    "bearing_temp_rise_c": (
        ["bearing_temp_c", "ambient_temp_c"],
        lambda d: d["bearing_temp_c"] - d["ambient_temp_c"],
        "bearing temp minus ambient",
    ),
}


@dataclass
class BindReport:
    """What the adapter did, and what it had to invent."""

    subsystem: str
    n_rows: int
    mapped: dict[str, str] = field(default_factory=dict)
    derived: dict[str, str] = field(default_factory=dict)
    filled: dict[str, float] = field(default_factory=dict)
    unmatched_source: list[str] = field(default_factory=list)
    low_confidence: dict[str, float] = field(default_factory=dict)
    empty: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        L = [f"bind report - subsystem={self.subsystem} rows={self.n_rows:,}"]
        L.append(f"  mapped   {len(self.mapped)} columns from the source")
        for t, s in sorted(self.mapped.items()):
            flag = ""
            if t in self.low_confidence:
                flag = f"   <-- CHECK, fuzzy match {self.low_confidence[t]:.2f}"
            L.append(f"             {t:<22} <- {s}{flag}")
        if self.empty:
            L.append(f"  DROPPED  {len(self.empty)} mapped columns that were entirely null")
            for t in sorted(self.empty):
                L.append(f"             {t:<22} -- present in the source but unusable")
        if self.derived:
            L.append(f"  derived  {len(self.derived)} columns that were absent but computable")
            for t, how in sorted(self.derived.items()):
                L.append(f"             {t:<22} =  {how}")
        if self.filled:
            L.append(f"  FILLED   {len(self.filled)} columns that do not exist in this data")
            for t, v in sorted(self.filled.items()):
                L.append(f"             {t:<22} := {v}   (its correction is now inert)")
        if self.unmatched_source:
            L.append(f"  ignored  {len(self.unmatched_source)} source columns not in the contract")
            L.append(f"             {', '.join(self.unmatched_source[:12])}"
                     + (" ..." if len(self.unmatched_source) > 12 else ""))
        return "\n".join(L)


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

@dataclass
class Adapter:
    """Maps one source schema onto the contract.

    Args:
        subsystem: "door" or "bogie".
        column_map: contract_column -> source_column. Anything omitted is
            derived, filled, or reported missing - in that order.
        rename_only: skip derivation and filling. Use when you want a hard
            failure rather than a degraded run.
    """

    subsystem: str
    column_map: dict[str, str] = field(default_factory=dict)
    rename_only: bool = False

    # Constants used when a context column is absent entirely. Chosen so the
    # term is inert: zero variance means the normaliser's coefficient is
    # meaningless but harmless, and standardisation guards the divide.
    FILL_DEFAULTS = {
        "ambient_temp_c": 28.0,   # a plausible SGT ambient
        "load_proxy": 0.5,        # mid-scale
        "cycles_since_service": 0.0,
        "hour_of_day": 12.0,
    }

    def apply(self, raw: pd.DataFrame) -> tuple[pd.DataFrame, BindReport]:
        sub = contract.get(self.subsystem)
        rep = BindReport(subsystem=self.subsystem, n_rows=len(raw))
        out = pd.DataFrame(index=raw.index)

        # 1. rename
        for target, source in self.column_map.items():
            if source in raw.columns:
                out[target] = raw[source]
                rep.mapped[target] = source
        rep.unmatched_source = [c for c in raw.columns if c not in set(self.column_map.values())]

        # 2. types, before derivation - hour_of_day needs a real datetime
        if "ts" in out.columns and not pd.api.types.is_datetime64_any_dtype(out["ts"]):
            out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        for col in sub.signals + contract.CONTEXT:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # 2b. A column that exists but is entirely null is worse than an absent
        # one: the name matches, so it wins the mapping and silently blocks the
        # derivation that would have produced a real value. Real exports are
        # full of these. Drop them and let step 3 do its job.
        for col in list(out.columns):
            if col in ("fault_confirmed", "fault_mode"):
                continue
            if out[col].isna().all():
                out = out.drop(columns=[col])
                rep.mapped.pop(col, None)
                rep.empty.append(col)

        if self.rename_only:
            return self._finalise(out, sub, rep)

        # 3. derive what is computable
        for target, (needs, fn, how) in DERIVATIONS.items():
            if target in sub.columns and target not in out.columns:
                if all(n in out.columns for n in needs):
                    out[target] = fn(out)
                    rep.derived[target] = how

        # 4. fill what is not, and say so
        for target in contract.CONTEXT:
            if target not in out.columns:
                v = self.FILL_DEFAULTS.get(target, 0.0)
                out[target] = v
                rep.filled[target] = v

        return self._finalise(out, sub, rep)

    def _finalise(self, out: pd.DataFrame, sub: contract.Subsystem,
                  rep: BindReport) -> tuple[pd.DataFrame, BindReport]:
        if "subsystem" not in out.columns:
            out["subsystem"] = self.subsystem
        for lbl, default in (("fault_confirmed", False), ("fault_mode", None)):
            if lbl not in out.columns:
                out[lbl] = default
        if "fault_confirmed" in out.columns:
            out["fault_confirmed"] = out["fault_confirmed"].fillna(False).astype(bool)

        keep = [c for c in sub.columns if c in out.columns]
        out = out[keep]
        if "ts" in out.columns and "asset_id" in out.columns:
            out = out.sort_values(["asset_id", "ts"], ignore_index=True)
        return out, rep

    # ---------------------------------------------------------------- helpers
    @classmethod
    def suggest(cls, raw: pd.DataFrame, subsystem: str) -> tuple["Adapter", dict[str, float]]:
        """Build an adapter by guessing. Always review the report before trusting it."""
        mapping, score = suggest_mapping(list(raw.columns), subsystem)
        return cls(subsystem=subsystem, column_map=mapping), score


def read_any(path: str | Path, **kw) -> pd.DataFrame:
    """Read whatever they hand us, by extension. CSV is the likely case."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(p, **kw)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, **kw)
    if ext in (".json",):
        return pd.read_json(p, **kw)
    return pd.read_csv(p, **kw)
