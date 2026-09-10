"""The decision policy. This is the layer that actually sends people onto
track at 01:00, so the tests here are about the direction of every error and
about the policy never silently reading the wrong number."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway.aspect import (DISPLAY, RECOMMENDATION, Aspect, AspectCard,
                            AspectPolicy, confidence)


@pytest.fixture
def policy():
    return AspectPolicy()


# ------------------------------------------------------------------ thresholds

@pytest.mark.parametrize("margin,expected", [
    (0.0, Aspect.RED),
    (0.9, Aspect.RED),
    (1.0, Aspect.AMBER),          # boundary belongs to the less severe side
    (2.9, Aspect.AMBER),
    (3.0, Aspect.DOUBLE_AMBER),
    (20.9, Aspect.DOUBLE_AMBER),
    (21.0, Aspect.GREEN),
    (90.0, Aspect.GREEN),
])
def test_margin_maps_to_the_expected_aspect(policy, margin, expected):
    assert policy.raw(np.array([margin]))[0] == expected


def test_no_failure_path_is_green_not_unknown(policy):
    """An infinite margin means no meaningful slope. Handing a healthy door a
    countdown would be a false alarm."""
    assert policy.raw(np.array([np.inf]))[0] == Aspect.GREEN
    assert policy.raw(np.array([np.nan]))[0] == Aspect.GREEN


def test_severity_is_monotone_in_margin(policy):
    margins = np.linspace(0, 60, 200)
    aspects = policy.raw(margins)
    assert (np.diff(aspects) <= 0).all(), "less margin must never mean less severity"


def test_every_aspect_has_a_recommendation_and_a_label():
    for a in Aspect:
        assert RECOMMENDATION[a] and DISPLAY[a]


# ----------------------------------------------------------------- hysteresis

def test_escalation_is_immediate():
    p = AspectPolicy(dwell_days=3)
    raw = np.array([0, 0, 3, 3])
    assert list(p._damp(raw)) == [0, 0, 3, 3]


def test_de_escalation_waits_for_the_dwell():
    p = AspectPolicy(dwell_days=3)
    raw = np.array([2, 2, 0, 0, 0, 0])
    #                     hold hold drop
    assert list(p._damp(raw)) == [2, 2, 2, 2, 0, 0]


def test_a_single_quiet_day_does_not_drop_the_aspect():
    """The flapping this exists to prevent: one good day must not clear an
    AMBER, or operators learn to wait it out instead of acting."""
    p = AspectPolicy(dwell_days=3)
    raw = np.array([2, 2, 0, 2, 2])
    assert list(p._damp(raw)) == [2, 2, 2, 2, 2]


def test_dwell_of_one_is_a_no_op():
    p = AspectPolicy(dwell_days=1)
    raw = np.array([0, 2, 0, 3, 1])
    assert list(p._damp(raw)) == list(raw)


def test_hysteresis_only_ever_holds_higher_never_lower():
    """Damping must never make the system quieter than the raw signal."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 4, 500)
    damped = AspectPolicy(dwell_days=4)._damp(raw)
    assert (damped >= raw).all()


# ---------------------------------------------------------------------- apply

def _frame(n=40, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for a in ("A0", "A1"):
        for i in range(n):
            rows.append({
                "asset_id": a, "train_id": a,
                "day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i),
                "rul_point": 40.0 - i, "rul_lower": max(40.0 - i * 1.2, 0.0),
                "rul_upper": 60.0 - i,
            })
    return pd.DataFrame(rows)


def test_apply_is_order_independent(policy):
    """Same class of bug as the add_trend misalignment: apply() sorts to run
    hysteresis, and must reindex back to the caller's row order."""
    df = _frame()
    a = policy.apply(df)
    b = policy.apply(df.sample(frac=1.0, random_state=7))
    merged = a.merge(b[["asset_id", "day", "aspect"]], on=["asset_id", "day"],
                     suffixes=("_a", "_b"))
    assert (merged.aspect_a == merged.aspect_b).all()


def test_apply_preserves_the_caller_index(policy):
    df = _frame().sample(frac=1.0, random_state=3)
    out = policy.apply(df)
    assert out.index.equals(df.index)


def test_policy_reads_the_lower_bound_not_the_point_estimate():
    """THE design property. Acting on rul_point would systematically defer work,
    because the projection overstates remaining life by ~11 days."""
    df = pd.DataFrame({
        "asset_id": ["A"], "train_id": ["A"], "day": [pd.Timestamp("2026-04-01")],
        "rul_point": [30.0],     # GREEN if read directly
        "rul_lower": [2.0],      # AMBER - the honest reading
        "rul_upper": [40.0],
    })
    out = AspectPolicy().apply(df)
    assert out.aspect.iloc[0] == Aspect.AMBER, "policy must consume rul_lower"


def test_two_assets_same_projection_different_bound_get_different_aspects():
    """The behaviour the Aspect Card promises: uncertainty changes the order."""
    df = pd.DataFrame({
        "asset_id": ["CONFIDENT", "UNCERTAIN"], "train_id": ["T", "T"],
        "day": [pd.Timestamp("2026-04-01")] * 2,
        "rul_point": [30.0, 30.0],
        "rul_lower": [25.0, 2.0],
        "rul_upper": [35.0, 80.0],
    })
    out = AspectPolicy().apply(df).set_index("asset_id")
    assert out.loc["CONFIDENT", "aspect"] == Aspect.GREEN
    assert out.loc["UNCERTAIN", "aspect"] == Aspect.AMBER


# ----------------------------------------------------------------- confidence

def test_confidence_is_high_when_the_bound_does_not_change_the_call(policy):
    assert confidence(policy, point=1.5, lower=1.2) == "HIGH"     # both AMBER


def test_confidence_is_medium_when_the_bound_moves_it_one_step(policy):
    assert confidence(policy, point=5.0, lower=2.0) == "MEDIUM"   # DA -> AMBER


def test_confidence_is_low_when_the_bound_moves_it_two_steps(policy):
    assert confidence(policy, point=30.0, lower=2.0) == "LOW"     # GREEN -> AMBER


def test_a_red_card_is_not_labelled_low_just_because_the_margin_is_small(policy):
    """Regression test for the first confidence definition, which used relative
    interval width and labelled a RED card LOW - near failure a +/-1 day
    interval is 100% relative error even though the decision is not in doubt."""
    assert confidence(policy, point=0.8, lower=0.5) == "HIGH"


def test_confidence_is_na_for_a_healthy_asset(policy):
    assert confidence(policy, point=np.inf, lower=np.inf) == "n/a"


# ----------------------------------------------------------------------- card

def test_card_renders_the_decision_first(policy):
    df = _frame()
    row = policy.apply(df).iloc[-1]
    card = AspectCard.from_row(row, policy)
    text = card.render()
    assert card.recommendation in text
    assert DISPLAY[card.aspect] in text
    assert "margin" in text


def test_card_flags_a_hysteresis_hold(policy):
    """A PLAN order with no margin attached needs to say why, or it reads as a
    bug to the engineer looking at it."""
    row = pd.Series({
        "asset_id": "A", "train_id": "A", "day": pd.Timestamp("2026-04-01"),
        "aspect": int(Aspect.DOUBLE_AMBER),
        "rul_point": np.inf, "rul_lower": np.inf, "rul_upper": np.inf,
    })
    card = AspectCard.from_row(row, policy)
    assert card.held
    assert "HELD from an earlier escalation" in card.render()


def test_card_flags_a_hold_when_the_margin_is_large_but_finite(policy):
    """Regression test for a card that read WITHDRAW / 26.7 days / -0.1 sigma.

    After a repair the health index falls from ~40 sigma to ~0 overnight, so the
    margin becomes large but FINITE. The old `held` test only looked for an
    infinite margin, so the card rendered a withdraw order beside evidence
    saying the door was healthier than its own baseline.
    """
    row = pd.Series({
        "asset_id": "A", "train_id": "A", "day": pd.Timestamp("2026-04-01"),
        "aspect": int(Aspect.RED),
        "rul_point": 68.4, "rul_lower": 26.7, "rul_upper": 90.0,
    })
    card = AspectCard.from_row(row, policy)
    assert card.held, "a RED order on a 26.7-day margin must be flagged as held"
    assert card.margin_aspect == Aspect.GREEN
    assert "GREEN" in card.render(), "the card must say what today alone would give"


def test_a_genuine_red_is_not_flagged_as_held(policy):
    row = pd.Series({
        "asset_id": "A", "train_id": "A", "day": pd.Timestamp("2026-04-01"),
        "aspect": int(Aspect.RED),
        "rul_point": 1.2, "rul_lower": 0.4, "rul_upper": 5.0,
    })
    assert not AspectCard.from_row(row, policy).held


def test_card_margin_is_the_lower_bound(policy):
    row = pd.Series({
        "asset_id": "A", "train_id": "A", "day": pd.Timestamp("2026-04-01"),
        "aspect": int(Aspect.AMBER),
        "rul_point": 9.0, "rul_lower": 2.0, "rul_upper": 20.0,
    })
    card = AspectCard.from_row(row, policy)
    assert card.margin_days == 2.0
    assert card.projected_days == 9.0
