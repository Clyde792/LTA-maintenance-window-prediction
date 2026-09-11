"""Safety properties of the experimental outage-tolerant score.

These are the failure modes that would make the challenger worse than useless:
inventing evidence, reading the future, or quietly relabelling a stale score as a
current assessment. Each is pinned here.
"""
import numpy as np
import pandas as pd
import pytest

from headway.outage_tolerant import (GAP, INCOMPLETE, NO_SCORE, SUPPORTED, TOO_FEW,
                                     UNSUPPORTED, WindowPolicy, availability,
                                     recovery_days, recovery_summary,
                                     score_outage_tolerant)


class Column:
    """Minimal fitted-model stand-in: the score IS the column, so any change in
    the output can only have come from the windowing rule."""
    def __init__(self, column="x"):
        self.column = column

    def score(self, frame):
        return pd.to_numeric(frame[self.column]).to_numpy(float)


def frame(days, values, *, supported=None, quality=None, asset="A", start="2026-01-01"):
    base = pd.Timestamp(start)
    n = len(days)
    return pd.DataFrame({
        "asset_id": asset,
        "day": [base + pd.Timedelta(days=int(d)) for d in days],
        "x": values,
        "available_at": [base + pd.Timedelta(days=int(d) + 1) for d in days],
        "data_quality_ok": [True] * n if quality is None else quality,
        "context_supported": [True] * n if supported is None else supported,
    })


DET = {"s": Column()}


# ------------------------------------------------------------------- policy
@pytest.mark.parametrize("kw", [
    {"min_observations": 1}, {"min_observations": 3.0},
    {"max_age_days": 0}, {"max_gap_days": -1}, {"max_age_days": float("nan")},
    {"max_age_days": 2, "max_gap_days": 5},          # a gap wider than the window
])
def test_incoherent_policies_are_refused(kw):
    with pytest.raises(ValueError):
        WindowPolicy(**kw)


# ------------------------------------------------------- irregular timestamps
def test_irregular_spacing_still_supports_a_score():
    """The baseline needs three rows inside three calendar days. Observations on
    days 0, 2, 4 satisfy this policy and must produce a score."""
    out = score_outage_tolerant(DET, frame([0, 2, 4], [1., 2., 3.]),
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    last = out.iloc[-1]
    assert last.s_status == SUPPORTED
    assert last.s == pytest.approx(2.0)          # median of 1, 2, 3
    assert last.s_obs == 3
    assert last.s_span_days == pytest.approx(4.0)
    assert last.s_max_gap_days == pytest.approx(2.0)


def test_every_third_day_outage_is_scored_where_the_baseline_cannot():
    """The exact ROBUSTNESS_RESULTS.md failure: elapsed%3==0 removed."""
    days = [d for d in range(15) if d % 3 != 0]
    out = score_outage_tolerant(DET, frame(days, np.arange(len(days), dtype=float)),
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    assert (out.s_status == SUPPORTED).sum() >= len(days) - 3
    assert np.isfinite(out.s.iloc[-1])


# --------------------------------------------------------------- excessive gaps
def test_a_gap_wider_than_policy_ends_the_run():
    """Days 0,1 then a 9-day hole then day 10: the old pair must not be welded
    onto today's reading to manufacture enough evidence."""
    out = score_outage_tolerant(DET, frame([0, 1, 10], [1., 1., 9.]),
                                WindowPolicy(min_observations=3, max_age_days=30, max_gap_days=3))
    last = out.iloc[-1]
    assert last.s_status == GAP
    assert not np.isfinite(last.s)
    assert last.s_obs == 1


def test_observations_older_than_max_age_are_not_admitted():
    out = score_outage_tolerant(DET, frame([0, 1, 2, 20], [1., 1., 1., 5.]),
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=7))
    last = out.iloc[-1]
    assert last.s_status == TOO_FEW
    assert not np.isfinite(last.s)


# ------------------------------------------------- missing current observation
def test_missing_current_observation_is_unavailable_not_carried_forward():
    """Day 3 is unsupported. Its score must be NaN — never day 2's number."""
    out = score_outage_tolerant(
        DET, frame([0, 1, 2, 3], [1., 2., 3., 99.], supported=[True, True, True, False]),
        WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    assert out.iloc[2].s_status == SUPPORTED and np.isfinite(out.iloc[2].s)
    last = out.iloc[-1]
    assert last.s_status == UNSUPPORTED
    assert not np.isfinite(last.s)
    assert last.s != out.iloc[2].s or not np.isfinite(last.s)


def test_incomplete_channel_is_distinguished_from_unsupported_context():
    out = score_outage_tolerant(DET, frame([0, 1, 2, 3], [1., 2., 3., np.nan]),
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    assert out.iloc[-1].s_status == INCOMPLETE
    assert not np.isfinite(out.iloc[-1].s)


# ------------------------------------------- unsupported history must not leak
def test_unsupported_history_never_enters_the_window():
    """Days 1 and 2 are out of support. Day 3 then has only itself and day 0,
    which is fewer than three observations — it must abstain rather than borrow
    readings taken outside the supported operating range."""
    out = score_outage_tolerant(
        DET, frame([0, 1, 2, 3], [1., 100., 100., 2.], supported=[True, False, False, True]),
        WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    last = out.iloc[-1]
    assert last.s_status == TOO_FEW
    assert last.s_obs == 2
    assert not np.isfinite(last.s)


def test_failed_quality_is_excluded_from_history_too():
    out = score_outage_tolerant(
        DET, frame([0, 1, 2, 3], [1., 100., 100., 2.], quality=[True, False, False, True]),
        WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    assert out.iloc[-1].s_obs == 2
    assert not np.isfinite(out.iloc[-1].s)


# ------------------------------------------------------------ future leakage
def test_future_rows_cannot_change_an_earlier_assessment():
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    days, vals = [0, 1, 2, 3, 4], [1., 2., 3., 4., 5.]
    early = score_outage_tolerant(DET, frame(days[:3], vals[:3]), policy)
    full = score_outage_tolerant(DET, frame(days, vals), policy)
    for col in ("s", "s_status", "s_obs", "s_span_days", "s_max_gap_days"):
        pd.testing.assert_series_equal(
            early[col].reset_index(drop=True), full[col].head(3).reset_index(drop=True),
            check_names=False)


def test_a_huge_future_value_does_not_reach_back():
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    out = score_outage_tolerant(DET, frame([0, 1, 2, 3], [1., 1., 1., 1e6]), policy)
    assert out.iloc[2].s == pytest.approx(1.0)


def test_assets_do_not_borrow_each_others_history():
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    a = frame([0, 1, 2], [1., 1., 1.], asset="A")
    b = frame([2], [9.], asset="B")
    out = score_outage_tolerant(DET, pd.concat([a, b], ignore_index=True), policy)
    row = out[out.asset_id == "B"].iloc[0]
    assert row.s_status == TOO_FEW and row.s_obs == 1


# ------------------------------------------------------------------ reporting
def test_duplicate_asset_days_are_refused():
    with pytest.raises(ValueError):
        score_outage_tolerant(DET, frame([1, 1], [1., 2.]), WindowPolicy())


def test_availability_keeps_missing_days_in_the_denominator():
    out = score_outage_tolerant(DET, frame([0, 1, 2], [1., 2., 3.]),
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    assert availability(out, "s", expected_rows=3) == pytest.approx(1 / 3)
    assert availability(out, "s", expected_rows=6) == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        availability(out, "s", expected_rows=0)


def test_recovery_is_measured_when_the_score_can_be_RECEIVED():
    """A day's aggregate lands the next midnight. Observations resume on day 10;
    day 12 is the first with three, and its aggregate arrives on day 13 — three
    elapsed days after the resume, not two."""
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    out = score_outage_tolerant(DET, frame([0, 1, 2, 10, 11, 12], [1.] * 6), policy)
    got = recovery_days(out, "s", resumed_at="2026-01-11", assets=["A"])
    assert got["A"] == pytest.approx(3.0)


def test_recovery_preserves_fractional_days():
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    out = score_outage_tolerant(DET, frame([0, 1, 2], [1., 1., 1.]), policy)
    got = recovery_days(out, "s", resumed_at="2026-01-02 12:00", assets=["A"])
    assert got["A"] == pytest.approx(1.5)          # available 2026-01-03 00:00


def test_recovery_requires_the_availability_clock():
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    out = score_outage_tolerant(DET, frame([0, 1, 2], [1., 1., 1.]), policy)
    with pytest.raises(ValueError, match="operational recovery"):
        recovery_days(out.drop(columns=["available_at"]), "s",
                      resumed_at="2026-01-01", assets=["A"])


def test_every_requested_asset_appears_even_with_no_rows():
    """An asset that produced nothing must not vanish from the denominator."""
    policy = WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3)
    out = score_outage_tolerant(DET, frame([0, 1, 2], [1., 1., 1.]), policy)
    got = recovery_days(out, "s", resumed_at="2026-01-01", assets=["A", "GHOST"])
    assert set(got) == {"A", "GHOST"}
    assert got["GHOST"] is None


def test_recovery_reports_none_when_no_score_returns():
    policy = WindowPolicy(min_observations=3, max_age_days=2, max_gap_days=1)
    out = score_outage_tolerant(DET, frame([0, 1, 2, 10], [1., 1., 1., 1.]), policy)
    assert recovery_days(out, "s", resumed_at="2026-01-11", assets=["A"])["A"] is None


def test_recovering_fewer_assets_faster_cannot_win():
    """Censoring: a method that abandons most assets must not look quick."""
    thorough = recovery_summary({"a": 3.0, "b": 3.0, "c": 3.0})
    partial = recovery_summary({"a": 1.0, "b": None, "c": None})
    assert thorough["median_days"] == pytest.approx(3.0)
    assert partial["median_days"] is None and partial["median_is_censored"]
    assert partial["assets_recovered"] == 1 and partial["assets_total"] == 3


def test_recovery_summary_counts_every_requested_asset():
    got = recovery_summary({"a": 2.0, "b": None})
    assert got["assets_total"] == 2 and got["assets_recovered"] == 1
    assert got["median_days"] is None and got["median_is_censored"]


# ------------------------------------------- non-finite model output
class BadAt:
    """Finite inputs, deliberately unusable OUTPUT on chosen rows — the case
    input-completeness checks cannot see."""
    column = "x"

    def __init__(self, rows, value):
        self.rows, self.value = set(rows), value

    def score(self, frame):
        v = pd.to_numeric(frame[self.column]).to_numpy(float).copy()
        for k, pos in enumerate(frame.index):
            if pos in self.rows:
                v[k] = self.value
        return v


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_model_output_is_never_published_as_supported(bad):
    """Before this guard a model returning NaN today produced status='supported'
    with score=NaN: the inputs were finite, so nothing upstream objected."""
    data = frame([0, 1, 2], [1., 2., 3.])          # every input finite
    out = score_outage_tolerant({"s": BadAt([2], bad)}, data,
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    last = out.iloc[-1]
    assert last.s_status == NO_SCORE
    assert not np.isfinite(last.s)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_history_output_is_excluded_from_the_window(bad):
    """A bad output in HISTORY must be dropped from the window, never averaged
    into today's median."""
    data = frame([0, 1, 2, 3], [1., 2., 3., 4.])
    out = score_outage_tolerant({"s": BadAt([1], bad)}, data,
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    last = out.iloc[-1]
    assert last.s_status == SUPPORTED
    assert last.s_obs == 3                          # days 0, 2, 3 only
    assert last.s == pytest.approx(3.0)             # median of 1, 3, 4
    assert np.isfinite(last.s)


def test_a_wholly_unusable_model_yields_no_supported_rows():
    class Broken:
        column = "x"
        def score(self, frame):
            return np.full(len(frame), np.nan)
    out = score_outage_tolerant({"s": Broken()}, frame([0, 1, 2, 3], [1., 2., 3., 4.]),
                                WindowPolicy(min_observations=3, max_age_days=7, max_gap_days=3))
    assert (out.s_status == NO_SCORE).all()
    assert not np.isfinite(out.s).any()
