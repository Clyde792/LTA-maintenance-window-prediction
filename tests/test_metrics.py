"""The Proving Ground. Every comparative claim in the pitch is produced here, so
these tests are the ones that stop us saying something false on stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway.evaluate import metrics


@pytest.fixture
def toy():
    """3 assets x 60 days. A2 degrades from day 20, fails day 50, then is repaired.

    The repair matters: without it A2 stays elevated to day 59, those post-fault
    rows outrank the genuine warning window, and the budget gets spent telling
    you about a train that has already failed.
    """
    days = pd.date_range("2026-04-01", periods=60, freq="D")
    rows = []
    for a in ("A0", "A1", "A2"):
        for i, d in enumerate(days):
            hi = 0.0
            if a == "A2" and 20 <= i <= 50:
                hi = (i - 20) / 30.0
            rows.append({"asset_id": a, "day": d, "hi": hi, "noise": (i % 7) / 100.0})
    daily = pd.DataFrame(rows)
    episodes = pd.DataFrame([{
        "asset_id": "A2",
        "onset_ts": days[20],
        "fault_ts": days[50],
    }])
    return daily, episodes


def test_budget_is_spent_exactly(toy):
    """Regression test.

    Selecting with `score >= threshold` admitted ties, so one detector could
    fire 152 alarms against another's 150 - breaking the equal-budget
    comparison the whole module exists to make.
    """
    daily, episodes = toy
    for budget in (5, 17, 40):
        r = metrics.evaluate(daily, "noise", episodes, budget)
        assert r.alarms == budget, f"budget {budget} spent {r.alarms}"


def test_tied_scores_do_not_inflate_the_budget(toy):
    daily, episodes = toy
    daily = daily.assign(flat=1.0)                 # every row tied
    r = metrics.evaluate(daily, "flat", episodes, 25)
    assert r.alarms == 25


def test_selection_is_deterministic_under_row_order(toy):
    daily, episodes = toy
    a = metrics.evaluate(daily, "hi", episodes, 20)
    b = metrics.evaluate(daily.sample(frac=1.0, random_state=3), "hi", episodes, 20)
    assert (a.detected, a.median_lead_d, a.precision) == \
           (b.detected, b.median_lead_d, b.precision)


def test_oracle_detector_captures_all_available_warning(toy):
    """A detector that ranks the degradation window first should score capture 1."""
    daily, episodes = toy
    r = metrics.evaluate(daily, "hi", episodes, 30)
    assert r.detected == 1
    assert r.worst_capture == pytest.approx(1.0, abs=0.05)
    assert r.median_lead_d == pytest.approx(30.0, abs=1.0)


def test_capture_uses_the_detectable_window_when_declared(toy):
    """The full onset-to-fault window overstates what any detector could do -
    at onset the wear is exactly zero. When the ground truth declares a
    detectable_days ceiling, capture must be measured against it."""
    daily, episodes = toy
    base = metrics.evaluate(daily, "hi", episodes, 30)
    declared = episodes.assign(detectable_days=15.0)   # half the 30-day window
    strict = metrics.evaluate(daily, "hi", declared, 30)
    # Same alerts, same leads - only the denominator changes, so capture rises
    # (clipped at 1.0) and nothing else moves.
    assert strict.median_lead_d == base.median_lead_d
    assert strict.worst_capture >= base.worst_capture
    assert strict.worst_capture <= 1.0
    # A declared ceiling larger than the window is ignored rather than trusted.
    weird = episodes.assign(detectable_days=500.0)
    assert metrics.evaluate(daily, "hi", weird, 30).worst_capture == \
        pytest.approx(base.worst_capture)


def test_capture_never_exceeds_one(toy):
    daily, episodes = toy
    for budget in (5, 20, 60, 120):
        r = metrics.evaluate(daily, "hi", episodes, budget)
        if r.detected:
            assert r.worst_capture <= 1.0 and r.median_capture <= 1.0


def test_alerts_after_the_fault_are_not_credited_as_warnings():
    """Telling someone a train has already failed is not a warning."""
    days = pd.date_range("2026-04-01", periods=40, freq="D")
    daily = pd.DataFrame({
        "asset_id": "A0", "day": days,
        # score rises only AFTER the fault on day 20
        "s": [0.0] * 21 + [1.0] * 19,
    })
    episodes = pd.DataFrame([{"asset_id": "A0", "onset_ts": days[5], "fault_ts": days[20]}])
    r = metrics.evaluate(daily, "s", episodes, 10)
    assert r.detected == 0
    assert r.precision == 0.0


def test_a_detector_that_finds_nothing_is_reported_honestly(toy):
    daily, episodes = toy
    daily = daily.assign(useless=-daily.hi)        # ranks the healthy assets first
    r = metrics.evaluate(daily, "useless", episodes, 10)
    assert r.detected == 0
    assert r.missed == 1
    assert np.isnan(r.median_lead_d)


def test_all_nan_score_does_not_crash(toy):
    daily, episodes = toy
    daily = daily.assign(broken=np.nan)
    r = metrics.evaluate(daily, "broken", episodes, 10)
    assert r.alarms == 0 and r.detected == 0


def test_budget_larger_than_the_data_is_clamped(toy):
    daily, episodes = toy
    r = metrics.evaluate(daily, "hi", episodes, 10_000)
    assert r.alarms == len(daily)


def test_compare_returns_one_row_per_detector(toy):
    daily, episodes = toy
    out = metrics.compare(daily, {"good": "hi", "noise": "noise"}, episodes, 20)
    assert len(out) == 2
    assert set(out.name) == {"good", "noise"}
    assert out[out.name == "good"].detected.iloc[0] >= out[out.name == "noise"].detected.iloc[0]


def test_budget_curve_spends_each_budget_exactly(toy):
    daily, episodes = toy
    curve = metrics.budget_curve(daily, "hi", episodes, [5, 10, 25])
    assert list(curve.alarms) == [5, 10, 25]


def test_more_budget_never_reduces_detection(toy):
    """Coverage must be monotone in budget - a sanity check on the whole harness."""
    daily, episodes = toy
    curve = metrics.budget_curve(daily, "hi", episodes, [5, 10, 20, 40, 80])
    assert list(curve.detected) == sorted(curve.detected)


def test_format_table_renders_without_error(toy):
    daily, episodes = toy
    out = metrics.compare(daily, {"good": "hi"}, episodes, 20)
    text = metrics.format_table(out)
    assert "good" in text and "capture" in text
