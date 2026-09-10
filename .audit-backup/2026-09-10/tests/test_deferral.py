"""The Deferral Ledger prints probabilities an engineer will act on, so its
failure modes are quantitative and quiet. These tests pin the ones that would
mislead rather than crash."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway import contract, features
from headway.deferral import RISK_CEILING, DeferralLedger, format_risk, render
from headway.normalise import AssetBaseline, ConditionNormaliser
from headway.rul import ConformalRUL


@pytest.fixture(scope="module")
def pipeline(cycles, episodes):
    sub = contract.get("door")
    cyc = ConditionNormaliser(sub.primary, contract.CONTEXT).fit_transform(cycles)
    daily = AssetBaseline().fit_transform(features.to_daily(cyc, "door"))
    return features.add_trend(daily, col="health_index"), episodes


@pytest.fixture(scope="module")
def ledger(pipeline):
    daily, eps = pipeline
    return DeferralLedger().fit(daily, eps), daily, eps


# ------------------------------------------------------------------ the curve

def test_bounds_are_monotone_in_confidence(ledger):
    """A 50% bound must never sit below a 95% one, or the curve cannot be
    inverted and every risk figure downstream is meaningless."""
    led, daily, _ = ledger
    B = led.bounds(daily)
    d = np.diff(B, axis=1)
    assert (d[~np.isnan(d)] >= -1e-9).all()


def test_missing_history_is_priced_as_unknown_not_as_safe(ledger):
    """A row with no bound yet must report NaN, not 0. Zero reads as 'safe to
    wait a month', which is a claim we have not earned on an asset we have
    barely seen."""
    led, daily, _ = ledger
    B = led.bounds(daily)
    unknown = np.isnan(B).all(axis=1)
    if unknown.any():
        r = led.risk(daily, 14.0)
        assert np.isnan(r[unknown]).all()
        assert format_risk(r[unknown][0]) == "  --"


def test_one_bound_per_calibrated_level(ledger):
    led, daily, _ = ledger
    assert led.bounds(daily).shape == (len(daily), len(led.alphas))


def test_use_before_fit_raises(pipeline):
    daily, _ = pipeline
    with pytest.raises(RuntimeError, match="fit"):
        DeferralLedger().bounds(daily)


# -------------------------------------------------------------------- risk

def test_acting_now_is_never_risky(ledger):
    """Regression test.

    'Act tonight' is not a deferral, so its risk is zero by definition. An
    earlier version let the already-failed override clobber the horizon-0 case
    and printed HIGH against 'act tonight' - telling an engineer that fixing a
    door immediately was as dangerous as ignoring it for a month.
    """
    led, daily, _ = ledger
    assert (led.risk(daily, 0.0) == 0.0).all()
    assert (led.risk(daily, -1.0) == 0.0).all()


def test_risk_never_decreases_with_a_longer_wait(ledger):
    """Waiting longer cannot be safer. If this inverts, the ledger would advise
    deferring FURTHER to reduce risk."""
    led, daily, _ = ledger
    B = led.bounds(daily)
    prev = led.risk(daily, 1.0, bounds=B)
    for h in (3.0, 7.0, 14.0, 28.0, 60.0):
        cur = led.risk(daily, h, bounds=B)
        assert (cur >= prev - 1e-9).all(), f"risk fell between {h} and the previous horizon"
        prev = cur


def test_risk_stays_a_probability(ledger):
    led, daily, _ = ledger
    for h in (0.0, 5.0, 30.0, 365.0):
        r = led.risk(daily, h)
        assert ((r >= 0.0) & (r <= 1.0)).all()


def test_a_healthy_door_carries_no_deferral_risk(ledger):
    """Assets with no failure path must not be handed a countdown."""
    led, daily, eps = ledger
    healthy = daily[~daily.asset_id.isin(eps.asset_id)]
    r = led.risk(healthy, 14.0)
    assert (r < 0.10).mean() > 0.95


def test_a_degrading_door_is_riskier_than_a_healthy_one(ledger):
    led, daily, eps = ledger
    d = daily.assign(r=led.risk(daily, 14.0))
    sick = d[d.asset_id.isin(eps.asset_id)].r.mean()
    well = d[~d.asset_id.isin(eps.asset_id)].r.mean()
    assert sick > well


# ------------------------------------------------------- the identity

def test_latest_safe_date_is_the_conformal_bound_itself(pipeline):
    """`latest_date_under(0.10)` should BE the 90% bound - the same number the
    Aspect Card already prints as its margin. The ledger exposes an idea that
    was already in the product rather than adding a new one."""
    daily, eps = pipeline
    led = DeferralLedger(alphas=(0.05, 0.10, 0.25, 0.50)).fit(daily, eps)
    direct = ConformalRUL(alpha=0.10, mode=led.base.mode,
                          projection=led.base.projection).fit(daily, eps)
    expected = direct.predict(daily)["rul_lower"].to_numpy(float)
    got = led.latest_date_under(daily, 0.10)
    ok = np.isfinite(expected) & np.isfinite(got)
    assert ok.sum() > 50
    np.testing.assert_allclose(got[ok], expected[ok], rtol=1e-6, atol=1e-6)


def test_the_ledger_never_contradicts_the_card(pipeline):
    """Regression test for a contradiction that reached the screen.

    The card printed RED / WITHDRAW / margin 1 day, and the ledger directly
    beneath it printed "latest date under 10% risk: 5 days". Both were valid
    conformal bounds - but under DIFFERENT modes, disagreeing by up to 32 days.
    An engineer reads that once and stops believing either number.

    The ledger must therefore share the aspect model's mode, and this asserts
    the agreement rather than trusting the default to stay put.
    """
    daily, eps = pipeline
    led = DeferralLedger().fit(daily, eps)
    card_margin = ConformalRUL(alpha=0.10).fit(daily, eps)        .predict(daily)["rul_lower"].to_numpy(float)
    ledger_date = led.latest_date_under(daily, 0.10)
    ok = np.isfinite(card_margin) & np.isfinite(ledger_date)
    assert ok.sum() > 50
    np.testing.assert_allclose(ledger_date[ok], card_margin[ok], rtol=1e-6, atol=1e-6)


def test_a_tighter_risk_budget_buys_less_time(ledger):
    led, daily, _ = ledger
    strict = led.latest_date_under(daily, 0.05)
    loose = led.latest_date_under(daily, 0.50)
    ok = np.isfinite(strict) & np.isfinite(loose)
    assert (strict[ok] <= loose[ok] + 1e-9).all()


# ------------------------------------------------------------- reporting

def test_high_risk_is_reported_as_a_word_not_a_number():
    """Above the ceiling we measured ourselves ~35 points out. Printing '53%'
    there would be inventing precision, and one engineer checking one number
    against reality would cost the product its credibility."""
    assert format_risk(RISK_CEILING) == "HIGH"
    assert format_risk(0.9) == "HIGH"
    assert "%" not in format_risk(0.55)


def test_quotable_risks_are_printed_as_percentages():
    assert "%" in format_risk(0.12)
    assert format_risk(0.01) == "  <5%"
    assert format_risk(0.0) == "   0%"


def test_no_failure_path_renders_as_unknown_not_zero():
    assert format_risk(np.nan) == "  --"


def test_ledger_adds_one_column_per_horizon(ledger):
    led, daily, _ = ledger
    out = led.ledger(daily, horizons=(3.0, 7.0))
    assert "risk_3d" in out and "risk_7d" in out and "safe_days_10pct" in out
    assert len(out) == len(daily)


def test_render_reads_as_a_menu_of_choices(ledger):
    led, daily, _ = ledger
    out = led.ledger(daily)
    row = out[np.isfinite(out.safe_days_10pct)].iloc[0]
    text = render(row)
    assert "what does it cost to wait" in text
    assert "act tonight" in text and "wait a week" in text


def test_coverage_is_recorded_for_every_level(ledger):
    """Each quoted confidence must carry its own measured coverage - that is
    what entitles the ledger to print a percentage at all."""
    led, _, _ = ledger
    assert set(led.coverage_) == set(led.alphas)
    assert all(np.isfinite(v) for v in led.coverage_.values())
