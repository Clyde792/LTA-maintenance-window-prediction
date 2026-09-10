"""The contract is the interface to the real NEBULA X data. If its validator is
wrong, a bad adapter mapping fails silently at 02:00 on the night. These tests
exist mostly to protect the error messages."""

from __future__ import annotations

import pandas as pd
import pytest

from headway import contract


def test_generated_data_satisfies_contract(cycles):
    assert contract.validate(cycles, "door").ok


def test_missing_column_fails_and_names_it(cycles):
    r = contract.validate(cycles.drop(columns=["peak_current_a"]), "door")
    assert not r.ok
    assert any("peak_current_a" in e for e in r.errors)


def test_non_datetime_ts_fails_with_actionable_message(cycles):
    bad = cycles.assign(ts=cycles.ts.astype(str))
    r = contract.validate(bad, "door")
    assert not r.ok
    assert any("to_datetime" in e for e in r.errors)


def test_non_numeric_signal_fails(cycles):
    bad = cycles.assign(peak_current_a=cycles.peak_current_a.astype(str))
    r = contract.validate(bad, "door")
    assert not r.ok
    assert any("peak_current_a" in e and "numeric" in e for e in r.errors)


def test_extra_columns_are_a_warning_not_an_error(cycles):
    r = contract.validate(cycles.assign(vendor_debug_col=1), "door")
    assert r.ok
    assert any("vendor_debug_col" in w for w in r.warnings)


def test_out_of_range_load_proxy_warns_but_still_runs(cycles):
    r = contract.validate(cycles.assign(load_proxy=cycles.load_proxy * 100), "door")
    assert r.ok, "out-of-range context must not block a run on real data"
    assert any("load_proxy" in w for w in r.warnings)


def test_fractional_hour_23_point_4_is_valid(cycles):
    """hour_of_day is a float in [0,24). 23.4 is a cycle at 23:24, not an error."""
    r = contract.validate(cycles.assign(hour_of_day=23.4), "door")
    assert not any("hour_of_day" in w for w in r.warnings)


def test_hour_of_day_24_is_flagged(cycles):
    r = contract.validate(cycles.assign(hour_of_day=24.0), "door")
    assert any("hour_of_day" in w for w in r.warnings)


def test_base_rate_is_reported_so_nobody_reports_accuracy(cycles):
    r = contract.validate(cycles, "door")
    assert any("base rate" in w and "never accuracy" in w for w in r.warnings)


def test_empty_frame_does_not_crash():
    cols = contract.get("door").columns
    df = pd.DataFrame({c: pd.Series(dtype="float") for c in cols})
    df["ts"] = pd.Series(dtype="datetime64[ns]")
    r = contract.validate(df, "door")
    assert r.n_rows == 0


def test_unknown_subsystem_lists_the_valid_ones():
    with pytest.raises(KeyError, match="door"):
        contract.get("pantograph")


def test_raise_if_bad_raises_on_failure(cycles):
    with pytest.raises(ValueError):
        contract.validate(cycles.drop(columns=["ts"]), "door").raise_if_bad()


def test_primary_signal_is_declared_in_signals():
    for name, sub in contract.SUBSYSTEMS.items():
        assert sub.primary in sub.signals, f"{name}.primary must be one of its signals"
