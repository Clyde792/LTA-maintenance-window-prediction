"""RUL and its bounds. The lower bound is what the recommendation consumes, so
a bound that is quietly optimistic would send engineers away from trains that
need them. These tests guard the direction of every error."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway import contract, features
from headway.normalise import AssetBaseline, ConditionNormaliser
from headway.rul import HI, SLOPE, ConformalRUL, _project


# ------------------------------------------------------------------ projection

def test_linear_projection_is_simple_division():
    hi = np.array([10.0]); slope = np.array([2.0])
    assert _project(hi, slope, 30.0, 0.25, 90.0, form="linear")[0] == pytest.approx(10.0)


def test_loglinear_projection_compounds_the_growth():
    # k = slope/hi = 0.2/day; time to threshold = ln(30/10)/0.2
    hi = np.array([10.0]); slope = np.array([2.0])
    expected = np.log(3.0) / 0.2
    assert _project(hi, slope, 30.0, 0.25, 90.0)[0] == pytest.approx(expected)


def test_loglinear_is_never_more_optimistic_than_linear():
    """The whole point of the form change: on convex wear the straight line
    overstates the time available, so the log form must always come in at or
    below it wherever both are defined."""
    rng = np.random.default_rng(0)
    hi = rng.uniform(1.0, 29.0, 500)
    slope = rng.uniform(0.3, 3.0, 500)
    lin = _project(hi, slope, 30.0, 0.25, 90.0, form="linear")
    log = _project(hi, slope, 30.0, 0.25, 90.0, form="loglinear")
    ok = np.isfinite(lin) & np.isfinite(log)
    assert (log[ok] <= lin[ok] + 1e-9).all()


def test_loglinear_falls_back_to_linear_below_one_sigma():
    """A barely-elevated index gives k = slope/hi enormous leverage - the log
    form would manufacture a confident countdown out of noise, so it must not
    be used there."""
    hi = np.array([0.5]); slope = np.array([0.4])
    log = _project(hi, slope, 30.0, 0.25, 90.0, form="loglinear")
    lin = _project(hi, slope, 30.0, 0.25, 90.0, form="linear")
    assert log[0] == pytest.approx(lin[0])
    # And a zero/negative index must not produce NaN.
    assert np.isfinite(_project(np.array([0.0]), slope, 30.0, 0.25, 90.0)[0])
    assert np.isfinite(_project(np.array([-2.0]), slope, 30.0, 0.25, 90.0)[0])


def test_flat_asset_is_not_on_a_failure_path():
    out = _project(np.array([1.0]), np.array([0.01]), 30.0, 0.25, 90.0)
    assert np.isinf(out[0]), "a healthy slope must not produce a finite margin"


def test_negative_slope_is_not_a_failure_path():
    out = _project(np.array([5.0]), np.array([-0.5]), 30.0, 0.25, 90.0)
    assert np.isinf(out[0])


def test_already_past_the_threshold_is_zero_margin():
    out = _project(np.array([40.0]), np.array([1.0]), 30.0, 0.25, 90.0)
    assert out[0] == 0.0


def test_projection_is_capped_at_the_horizon():
    out = _project(np.array([0.0]), np.array([0.3]), 100.0, 0.25, 90.0)
    assert out[0] == 90.0


# ------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def pipeline(cycles, episodes):
    """Full pipeline on the small fleet, so RUL sees realistic trajectories."""
    sub = contract.get("door")
    cyc = ConditionNormaliser(sub.primary, contract.CONTEXT).fit_transform(cycles)
    daily = AssetBaseline().fit_transform(features.to_daily(cyc, "door"))
    daily = features.add_trend(daily, col="health_index")
    return daily, episodes


def _true_rul(daily, episodes):
    out = pd.Series(np.nan, index=daily.index)
    for _, e in episodes.iterrows():
        m = ((daily.asset_id == e.asset_id)
             & (daily.day >= pd.Timestamp(e.onset_ts).normalize())
             & (daily.day < pd.Timestamp(e.fault_ts).normalize()))
        out[m] = (pd.Timestamp(e.fault_ts) - daily.loc[m, "day"]).dt.total_seconds() / 86400.0
    return out


# ------------------------------------------------------------------------ fit

def test_threshold_is_learned_near_the_health_index_at_fault(pipeline):
    daily, eps = pipeline
    m = ConformalRUL().fit(daily, eps)
    at_fault = [
        daily[(daily.asset_id == e.asset_id) & (daily.day <= e.fault_ts)][HI].iloc[-1]
        for _, e in eps.iterrows()
    ]
    assert m.threshold_ == pytest.approx(float(np.median(at_fault)), rel=0.01)


def test_predict_before_fit_raises(pipeline):
    daily, _ = pipeline
    with pytest.raises(RuntimeError, match="fit"):
        ConformalRUL().predict(daily)


def test_bounds_are_ordered(pipeline):
    daily, eps = pipeline
    out = ConformalRUL().fit_predict(daily, eps)
    ok = np.isfinite(out.rul_point) & np.isfinite(out.rul_lower) & np.isfinite(out.rul_upper)
    assert (out.rul_lower[ok] <= out.rul_upper[ok]).all()
    assert (out.rul_lower[ok] >= 0).all()


def test_lower_bound_is_conservative_where_it_matters(pipeline):
    """The bound must not overstate the time available - that is the error that
    sends an engineer away from a train that needs them."""
    daily, eps = pipeline
    out = ConformalRUL().fit_predict(daily, eps)
    truth = _true_rul(daily, eps)
    m = truth.notna() & np.isfinite(out.rul_lower)
    assert m.sum() > 20
    safe = (truth[m] >= out.rul_lower[m]).mean()
    assert safe >= 0.80, f"lower bound overstated the margin on {1 - safe:.0%} of days"

    # Strictest where it counts: inside a week of failure.
    close = m & truth.between(0, 7)
    if close.sum() >= 5:
        assert (truth[close] >= out.rul_lower[close]).mean() >= 0.85


def test_coverage_is_measured_out_of_fold_not_tautologically(pipeline):
    """Coverage computed against a quantile of the pooled scores would return
    1 - alpha by construction. It must be able to differ from the target."""
    daily, eps = pipeline
    covs = {a: ConformalRUL(alpha=a).fit(daily, eps).coverage_ for a in (0.05, 0.10, 0.25)}
    assert all(np.isfinite(c) for c in covs.values())
    # Not pinned to exactly 1 - alpha, and ordered the right way.
    assert covs[0.05] >= covs[0.25]
    assert any(abs(c - (1 - a)) > 1e-6 for a, c in covs.items())


def test_mondrian_produces_band_specific_corrections(pipeline):
    """Given enough calibration points per band, corrections must differ - or
    confidence cannot vary between rows and mondrian buys us nothing.

    min_per_bin is lowered here because the test fleet has only 4 episodes.
    The production default (20) deliberately falls back to a global ratio at
    this data volume; that fallback is asserted separately below.
    """
    daily, eps = pipeline
    m = ConformalRUL(mode="mondrian", min_per_bin=8).fit(daily, eps)
    assert len(m.lo_) == m.n_bins
    assert len(set(np.round(m.lo_, 4))) > 1, "bands are degenerate"


def test_thin_bands_fall_back_to_the_global_ratio(pipeline):
    """Safety property: too few points per band must not produce a noisy
    per-band quantile. It must degrade to the pooled one, exactly."""
    daily, eps = pipeline
    m = ConformalRUL(mode="mondrian", min_per_bin=10_000).fit(daily, eps)
    assert len(set(np.round(m.lo_, 6))) == 1, "thin bands must share one quantile"


def test_mondrian_is_sharper_than_a_single_global_ratio(pipeline):
    """The whole reason mondrian exists. If this stops holding, use ratio."""
    daily, eps = pipeline
    truth = _true_rul(daily, eps)

    def sharpness(mode: str, **kw) -> float:
        out = ConformalRUL(mode=mode, **kw).fit_predict(daily, eps)
        m = truth.notna() & np.isfinite(out.rul_lower)
        return float((out.rul_lower[m] / truth[m].clip(lower=1e-9)).median())

    ratio = sharpness("ratio")
    # Never worse is the invariant that must hold at any data volume, including
    # when the bands are too thin to estimate and it falls back to ratio.
    assert sharpness("mondrian") >= ratio - 1e-9
    # Strictly better once the bands have enough points. On the full 9-episode
    # dataset this holds at the production default (0.62 vs 0.46); the test
    # fleet has 4 episodes, so the threshold is lowered to reach the same regime.
    assert sharpness("mondrian", min_per_bin=8) > ratio


def test_additive_mode_was_degenerate_under_the_linear_form(pipeline):
    """Historical regression, pinned to the form it was measured under.

    Under the LINEAR projection, additive bought its coverage by shifting every
    bound to ~zero - safe and useless, sharpness ~0. That vacuousness was
    downstream of the linear form's ~9-day optimistic bias: the conformal shift
    had to be enormous to cover it. With the near-unbiased log-linear point
    estimate the additive correction is small and the mode becomes viable,
    which is exactly why build_rul.py re-selects the mode on evidence rather
    than trusting this history.
    """
    daily, eps = pipeline
    out = ConformalRUL(mode="additive", projection="linear").fit_predict(daily, eps)
    truth = _true_rul(daily, eps)
    m = truth.notna() & np.isfinite(out.rul_lower)
    sharp = (out.rul_lower[m] / truth[m].clip(lower=1e-9)).median()
    assert sharp < 0.1, "the linear-form vacuousness should be reproducible"


def test_loglinear_point_estimate_is_less_biased_than_linear(pipeline):
    """The reason the default changed. Median |error| of the point estimate
    inside degradation windows must be smaller under the log-linear form."""
    daily, eps = pipeline
    truth = _true_rul(daily, eps)

    def mae(form: str) -> float:
        out = ConformalRUL(projection=form).fit_predict(daily, eps)
        m = truth.notna() & np.isfinite(out.rul_point) & (out.rul_point > 0)
        return float((truth[m] - out.rul_point[m]).abs().median())

    assert mae("loglinear") < mae("linear")


def test_healthy_fleet_gets_no_finite_margin(pipeline):
    """Healthy doors must not be handed a countdown - that is a false alarm."""
    daily, eps = pipeline
    out = ConformalRUL().fit_predict(daily, eps)
    healthy = out[~out.asset_id.isin(eps.asset_id)]
    finite = np.isfinite(healthy.rul_point).mean()
    assert finite < 0.10, f"{finite:.0%} of healthy asset-days were given a margin"


def test_no_episodes_degrades_gracefully(pipeline):
    daily, _ = pipeline
    empty = pd.DataFrame(columns=["asset_id", "onset_ts", "fault_ts"])
    m = ConformalRUL().fit(daily, empty)
    assert m.n_calib_ == 0
    assert np.isnan(m.threshold_)
    with pytest.raises(RuntimeError):
        m.predict(daily)
