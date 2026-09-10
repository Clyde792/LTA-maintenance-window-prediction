"""The adapter is the seam where someone else's schema meets ours, and it gets
edited at 20:00 on the night under time pressure. These tests exist so that the
failure modes are loud rather than silent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway import contract
from headway.adapters.base import Adapter, suggest_mapping


def _vendor(n=400, seed=0) -> pd.DataFrame:
    """A plausible third-party export: their names, their units, their types."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-04-01 06:00", periods=n, freq="7min")
    return pd.DataFrame({
        "Event_Time": ts.strftime("%d/%m/%Y %H:%M:%S"),
        "EQUIPMENT_ID": ["TRN001-DOOR-1"] * n,
        "Car No": ["TRN001"] * n,
        "operation_time": rng.normal(3.2, .1, n),
        "I_Peak (mA)": rng.normal(5800, 200, n),
        "avg_current_mA": rng.normal(1300, 60, n),
        "charge": rng.normal(4.1, .1, n),
        "leaf_travel_m": rng.normal(1.3, .002, n),
        "Obstruction_Detected": rng.integers(0, 2, n),
        "recycle_count": rng.integers(0, 2, n),
        "saloon_temp": rng.normal(28, 2, n),
        "load_weigh": rng.uniform(0, 1, n),
        "operations_since_service": np.arange(n, dtype=float),
        "verified_fault": [False] * n,
        "failure_mode": [None] * n,
        "record_id": np.arange(n),
    })


# ------------------------------------------------------------------ suggestion

def test_suggestion_finds_every_column_the_source_actually_has():
    """hour_of_day is deliberately absent from the export - it is DERIVED from
    the timestamp, not suggested, so it is not part of this contract."""
    mapping, score = suggest_mapping(list(_vendor().columns), "door")
    for required in contract.IDENTITY + contract.CONTEXT:
        if required in ("subsystem", "hour_of_day"):
            continue
        assert required in mapping, f"failed to suggest a source for {required}"


def test_suggestion_scores_exact_hits_higher_than_fuzzy():
    mapping, score = suggest_mapping(list(_vendor().columns), "door")
    assert score["ambient_temp_c"] == 1.0            # saloon_temp is a listed alias
    assert score["travel_mm"] < 1.0                  # leaf_travel_m is only close


def test_a_source_column_is_never_used_twice():
    mapping, _ = suggest_mapping(list(_vendor().columns), "door")
    used = list(mapping.values())
    assert len(used) == len(set(used))


def test_suggestion_is_case_and_punctuation_insensitive():
    cols = ["TIMESTAMP", "asset-id", "Peak Current (A)"]
    mapping, _ = suggest_mapping(cols, "door")
    assert mapping.get("ts") == "TIMESTAMP"
    assert mapping.get("asset_id") == "asset-id"
    assert mapping.get("peak_current_a") == "Peak Current (A)"


def test_unknown_columns_are_left_alone():
    mapping, _ = suggest_mapping(["totally_unrelated", "xyzzy"], "door")
    assert "totally_unrelated" not in mapping.values() or len(mapping) == 0


# ---------------------------------------------------------------------- apply

def test_binds_a_vendor_export_to_a_valid_contract_frame():
    raw = _vendor()
    adapter, _ = Adapter.suggest(raw, "door")
    out, report = adapter.apply(raw)
    assert contract.validate(out, "door").ok
    assert pd.api.types.is_datetime64_any_dtype(out.ts)
    assert out.subsystem.eq("door").all()


def test_text_timestamps_are_parsed():
    out, _ = Adapter.suggest(_vendor(), "door")[0].apply(_vendor())
    assert out.ts.notna().all()
    assert out.ts.dt.year.eq(2026).all()


def test_unmapped_source_columns_are_reported_not_smuggled_through():
    raw = _vendor()
    out, report = Adapter.suggest(raw, "door")[0].apply(raw)
    assert "record_id" not in out.columns
    assert "record_id" in report.unmatched_source


def test_missing_context_is_filled_and_declared():
    """Losing a confound correction is survivable. Losing it silently is not."""
    raw = _vendor().drop(columns=["saloon_temp", "load_weigh"])
    out, report = Adapter.suggest(raw, "door")[0].apply(raw)
    assert contract.validate(out, "door").ok, "a missing confound must not stop the run"
    assert "ambient_temp_c" in report.filled
    assert "load_proxy" in report.filled
    assert out.ambient_temp_c.nunique() == 1, "the filled term must be inert"


def test_hour_of_day_is_derived_from_the_timestamp():
    raw = _vendor()
    out, report = Adapter.suggest(raw, "door")[0].apply(raw)
    assert "hour_of_day" in report.derived
    assert out.hour_of_day.between(0, 24, inclusive="left").all()


def test_an_all_null_column_is_dropped_so_derivation_can_run():
    """Regression test.

    A column that EXISTS but is entirely null is worse than an absent one: the
    name wins the mapping and silently blocks the derivation that would have
    produced a real value. The demo export had exactly this - an empty
    `hour_of_day` - and it bound to 100% nulls.
    """
    raw = _vendor()
    raw["hour_of_day"] = np.nan
    out, report = Adapter.suggest(raw, "door")[0].apply(raw)
    assert "hour_of_day" in report.empty
    assert "hour_of_day" in report.derived
    assert out.hour_of_day.notna().all()


def test_current_integral_is_derived_when_absent():
    raw = _vendor().drop(columns=["charge"])
    out, report = Adapter.suggest(raw, "door")[0].apply(raw)
    assert "current_integral_as" in report.derived
    assert out.current_integral_as.notna().all()


def test_train_id_is_derived_from_asset_id():
    raw = _vendor().drop(columns=["Car No"])
    out, report = Adapter.suggest(raw, "door")[0].apply(raw)
    assert "train_id" in report.derived
    assert out.train_id.eq("TRN001").all()


def test_rename_only_does_not_invent_anything():
    """Escape hatch for when we want a hard failure instead of a degraded run."""
    raw = _vendor().drop(columns=["saloon_temp"])
    adapter, _ = Adapter.suggest(raw, "door")
    adapter.rename_only = True
    out, report = adapter.apply(raw)
    assert not report.filled and not report.derived
    assert not contract.validate(out, "door").ok


def test_report_names_every_substitution_it_made():
    raw = _vendor().drop(columns=["saloon_temp", "charge"])
    _, report = Adapter.suggest(raw, "door")[0].apply(raw)
    text = str(report)
    assert "ambient_temp_c" in text and "current_integral_as" in text
    assert "FILLED" in text and "derived" in text


def test_empty_mapping_produces_an_actionable_error():
    from headway.adapters import nebulax
    with pytest.raises(ValueError, match="still empty"):
        nebulax.load("nonexistent.csv", subsystem="door")
