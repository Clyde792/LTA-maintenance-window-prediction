"""
THE FILE WE EDIT ON THE NIGHT.

Friday 18 Sep, ~20:00. NEBULA X hand over the real door and bogie data. The job
is to fill in COLUMN_MAP below and nothing else. Every other module in this
repository reads the contract, so if the mapping is right the whole pipeline -
normalisation, health index, RUL, aspects, tournament, UI - runs unchanged.

HOW TO USE THIS, IN ORDER
-------------------------
  1. Run the dry-run against their file. It prints a suggested mapping:

         .venv/Scripts/python.exe scripts/bind_data.py <their_file.csv>

  2. Paste the suggested dict into DOOR_MAP (or BOGIE_MAP) below.
  3. Fix the guesses it flagged with "CHECK". Delete any that are wrong.
  4. Verify units, timestamp semantics, identity joins and data quality.
     A contract PASS checks schema; it does not validate inference suitability.

WHAT TO WATCH FOR, FROM THE SITE VISITS
---------------------------------------
  - The join key. Telemetry may be keyed on a DCU serial while the verified
    fault records are keyed on a car number. If so, the mapping is not enough
    and we need a lookup - build it as a dict here, not in the pipeline.
  - Open vs close. If each row is a half-cycle, `direction` will be a column.
    Filter to close movements only (they carry the load) rather than mixing.
  - Units. Verify and convert mA to A, milliseconds to seconds, and metres
    to mm BEFORE derivation or inference. Configure unit_scales on Adapter.
    Scale-free normalisation does not make mixed units safe.
  - Missing context. If there is no temperature or crowding column, the adapter
    fills a constant and says so. We lose that correction, not the run.
"""

from __future__ import annotations

import pandas as pd

from .base import Adapter, read_any

# --------------------------------------------------------------------------
# THE MAPPING.  contract column  ->  their column name
# --------------------------------------------------------------------------
# Left side: our names, from headway/contract.py. Do not change these.
# Right side: theirs. This is the only side that gets edited.
#
# Anything left out is derived if possible (hour_of_day from ts,
# current_integral_as from mean current x duration, train_id from asset_id),
# then filled with a constant if it is a required context column.

DOOR_MAP: dict[str, str] = {
    # --- identity -----------------------------------------------------------
    # "ts":                   "",
    # "asset_id":             "",
    # "train_id":             "",

    # --- signals ------------------------------------------------------------
    # "cycle_duration_s":     "",
    # "peak_current_a":       "",
    # "mean_current_a":       "",
    # "current_integral_as":  "",
    # "travel_mm":            "",
    # "obstruction_flag":     "",
    # "retry_count":          "",

    # --- operating context (what we normalise against) ----------------------
    # "ambient_temp_c":       "",
    # "load_proxy":           "",
    # "cycles_since_service": "",

    # --- labels (the manually verified fault data) --------------------------
    # "fault_confirmed":      "",
    # "fault_mode":           "",
}

BOGIE_MAP: dict[str, str] = {
    # "ts":                   "",
    # "asset_id":             "",
    # "bearing_temp_rise_c":  "",
    # "vibration_rms_g":      "",
    # "vibration_kurtosis":   "",
    # "suspension_defl_mm":   "",
    # "wheel_impact_g":       "",
    # "ambient_temp_c":       "",
    # "load_proxy":           "",
}


def door_adapter() -> Adapter:
    return Adapter(subsystem="door", column_map=DOOR_MAP)


def bogie_adapter() -> Adapter:
    return Adapter(subsystem="bogie", column_map=BOGIE_MAP)


def load(path: str, subsystem: str = "door", **read_kw) -> pd.DataFrame:
    """Read their file and return a contract-shaped frame.

    Raises if the result does not satisfy the contract, because a silently
    half-bound frame is worse than a stop: every number downstream would be
    computed on whatever happened to map.
    """
    from .. import contract

    # Check the mapping BEFORE touching the file. An empty map is by far the
    # likelier problem on the night, and its message is the actionable one -
    # a FileNotFoundError sends you looking in the wrong place.
    adapter = door_adapter() if subsystem == "door" else bogie_adapter()
    if not adapter.column_map:
        raise ValueError(
            f"{subsystem.upper()}_MAP in headway/adapters/nebulax.py is still empty.\n"
            f"Run:  .venv/Scripts/python.exe scripts/bind_data.py {path}\n"
            f"and paste the suggested mapping in."
        )

    raw = read_any(path, **read_kw)
    out, report = adapter.apply(raw)
    print(report)
    contract.validate(out, subsystem).raise_if_bad()
    return out
