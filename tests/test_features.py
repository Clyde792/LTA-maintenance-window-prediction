"""Feature extraction. The alignment and leakage tests here guard defects that
would be invisible in output but fatal to every number we quote."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway import contract, features


def test_one_row_per_asset_day(cycles):
    daily = features.to_daily(cycles, "door")
    expected = cycles.assign(day=cycles.ts.dt.floor("D")).groupby(
        ["asset_id", "day"]).ngroups
    assert len(daily) == expected
    assert not daily.duplicated(["asset_id", "day"]).any()


def test_level_dispersion_and_rate_families_all_present(cycles):
    daily = features.to_daily(cycles, "door")
    assert "current_integral_as" in daily            # level
    assert "current_integral_as_iqr" in daily        # dispersion
    assert "obstruction_flag_rate" in daily          # rate
    assert "retry_count_rate" in daily
    assert "residual" not in daily, "residual only appears if the caller supplied it"


def test_rates_are_rates_and_dispersion_is_non_negative(cycles):
    daily = features.to_daily(cycles, "door")
    assert daily.obstruction_flag_rate.between(0, 1).all()
    assert (daily.current_integral_as_iqr.dropna() >= 0).all()


def test_context_is_carried_through_for_downstream_explanation(cycles):
    daily = features.to_daily(cycles, "door")
    for c in contract.CONTEXT:
        assert f"{c}_mean" in daily


def test_fault_label_survives_aggregation(cycles):
    daily = features.to_daily(cycles, "door")
    assert daily.fault_confirmed.sum() == cycles.fault_confirmed.sum()


# --------------------------------------------------------------- add_trend

def _toy(n_assets=3, n_days=40, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        for d in range(n_days):
            rows.append({
                "asset_id": f"A{a}",
                "day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=d),
                "health_index": 0.05 * d + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_add_trend_is_order_independent():
    """Regression test.

    add_trend used to sort with ignore_index=True and return a re-ordered frame.
    Callers assign the result back into their own frame, so that silently
    misaligned every value whenever the input was not already sorted.
    """
    df = _toy()
    ordered = features.add_trend(df, col="health_index")
    shuffled_in = df.sample(frac=1.0, random_state=42)
    shuffled = features.add_trend(shuffled_in, col="health_index")

    merged = ordered.merge(
        shuffled[["asset_id", "day", "health_index_smooth"]],
        on=["asset_id", "day"], suffixes=("_a", "_b"),
    )
    pd.testing.assert_series_equal(
        merged.health_index_smooth_a, merged.health_index_smooth_b,
        check_names=False,
    )


def test_add_trend_preserves_caller_index_and_order():
    df = _toy().sample(frac=1.0, random_state=1)
    out = features.add_trend(df, col="health_index")
    assert out.index.equals(df.index)
    pd.testing.assert_series_equal(out.health_index, df.health_index)


def test_trend_windows_are_trailing_and_cannot_see_the_future():
    """If a window peeked ahead, every lead-time number we report is inflated."""
    df = _toy(n_days=60)
    full = features.add_trend(df, col="health_index")
    cutoff = pd.Timestamp("2026-04-01") + pd.Timedelta(days=39)
    truncated = features.add_trend(df[df.day <= cutoff], col="health_index")

    a = full[full.day == cutoff].set_index("asset_id")
    b = truncated[truncated.day == cutoff].set_index("asset_id")
    for col in ("health_index_smooth", "health_index_slope", "health_index_vol"):
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


def test_slope_recovers_a_known_gradient():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "asset_id": "A",
        "day": pd.date_range("2026-04-01", periods=40, freq="D"),
        "health_index": 0.20 * np.arange(40) + rng.normal(0, 0.01, 40),
    })
    out = features.add_trend(df, col="health_index", slope_days=14)
    assert out.health_index_slope.dropna().iloc[-1] == pytest.approx(0.20, abs=0.03)


def test_slope_is_flat_for_a_healthy_asset():
    rng = np.random.default_rng(10)
    df = pd.DataFrame({
        "asset_id": "A",
        "day": pd.date_range("2026-04-01", periods=40, freq="D"),
        "health_index": rng.normal(0, 0.1, 40),
    })
    out = features.add_trend(df, col="health_index")
    assert abs(out.health_index_slope.dropna().iloc[-1]) < 0.05


def test_slope_handles_short_series_without_raising():
    df = pd.DataFrame({
        "asset_id": "A",
        "day": pd.date_range("2026-04-01", periods=2, freq="D"),
        "health_index": [0.1, 0.2],
    })
    out = features.add_trend(df, col="health_index")
    assert out.health_index_slope.isna().all(), "too little history must be NaN, not a guess"
