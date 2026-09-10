"""Fit-for-duty: within-asset load sensitivity and peak-conditioned level rule."""
import numpy as np
import pandas as pd
import pytest
from headway.aspect import AspectPolicy
from headway.duty import (DutyAssessor, cycle_stats, duty_summary, format_bands,
                          load_sensitivity, peak_bands, render, STATS)
from headway.pipeline import HealthPipeline
from headway.rul import ConformalRUL


def _cycles(b, n=200, seed=0):
    rng = np.random.default_rng(seed)
    load = rng.uniform(0, 1, n)
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.uniform(0, 3, n), unit="D")
    return pd.DataFrame({"asset_id": "A", "ts": ts, "load_proxy": load,
                         "cycle_index": 1.0 + b * load + rng.normal(0, .5, n)})


def test_sufficient_statistics_reproduce_within_day_regression():
    c = _cycles(4.)
    stats = cycle_stats(c)
    assert set(STATS) <= set(stats)
    daily = stats.assign(asset_id="A")
    out = load_sensitivity(daily, window_days=3, min_cycles=10)
    last = out.sort_values("day").iloc[-1]
    centered = c[["load_proxy", "cycle_index"]] - c.groupby(c.ts.dt.floor("D"))[["load_proxy", "cycle_index"]].transform("mean")
    ref = np.polyfit(centered.load_proxy, centered.cycle_index, 1)[0]
    assert last.load_sensitivity == pytest.approx(ref, rel=1e-6)
    assert last.load_sensitive


def test_daily_wear_and_crowding_changes_do_not_create_sensitivity():
    frames = []
    for day in range(3):
        frames.append(pd.DataFrame({"asset_id": "A",
            "ts": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day) + pd.to_timedelta(np.arange(100), unit="m"),
            "load_proxy": day * .3 + np.linspace(0, .1, 100),
            "cycle_index": np.full(100, day * 10.)}))
    out = load_sensitivity(cycle_stats(pd.concat(frames)), min_load_sd=.01)
    assert not out.load_sensitive.any()
    assert np.allclose(out.load_sensitivity, 0.)


def test_unsupported_cycles_do_not_supply_load_evidence():
    c = _cycles(4.).assign(context_supported=False)
    assert cycle_stats(c).empty


@pytest.mark.parametrize("state", ["insufficient_data", "stale_data", "outside_training_conditions", "uncalibrated"])
def test_held_escalation_with_unknown_evidence_is_not_duty_clearance(state):
    from types import SimpleNamespace
    from headway.duty import _duty
    row = SimpleNamespace(prediction_state=state, peak_state=state, aspect=3,
                          load_sensitivity=2., load_sensitive=False)
    assert _duty(row, 10.)[0] == "not_assessed"


def test_no_slope_is_not_sensitive_and_negative_slope_is_not_sensitive():
    for b in (0., -4.):
        out = load_sensitivity(cycle_stats(_cycles(b)), window_days=3, min_cycles=10)
        assert not out.load_sensitive.any()


def test_insufficient_contrast_abstains():
    c = _cycles(4.)
    c["load_proxy"] = 0.5  # no crowded/quiet contrast at all
    out = load_sensitivity(cycle_stats(c), window_days=3, min_cycles=10)
    assert out.load_sensitivity.isna().all() and not out.load_sensitive.any()


def test_missing_statistics_rejected():
    with pytest.raises(ValueError, match="duty statistics"):
        load_sensitivity(pd.DataFrame({"asset_id": ["A"], "day": [pd.Timestamp("2026-01-01")]}))


def test_peak_bands_are_derived_from_the_load_profile():
    rng = np.random.default_rng(1)
    hour = rng.uniform(5.5, 24, 20000)
    load = 0.15 + 0.8 * (np.exp(-((hour - 8) ** 2) / 2) + np.exp(-((hour - 18) ** 2) / 2))
    ref = pd.DataFrame({"hour_of_day": hour, "load_proxy": np.clip(load + rng.normal(0, .05, len(hour)), 0, 1)})
    bands, peak, off = peak_bands(ref)
    assert len(bands) == 2 and bands[0][0] <= 8 < bands[0][1] and bands[1][0] <= 18 < bands[1][1]
    assert peak > off
    assert "07:00" in format_bands(bands) or "06:00" in format_bands(bands)


def test_flat_profile_yields_no_bands():
    ref = pd.DataFrame({"hour_of_day": np.linspace(6, 23, 500), "load_proxy": 0.4})
    bands, peak, off = peak_bands(ref)
    assert bands == [] and peak == pytest.approx(off)


@pytest.fixture(scope="module")
def assessed(cycles, episodes):
    ref = cycles[cycles.ts < cycles.ts.min().floor("D") + pd.Timedelta(days=30)]
    p = HealthPipeline().fit(ref)
    d = p.transform(cycles)
    m = ConformalRUL().fit(d, episodes, evaluate=False)
    a = AspectPolicy().apply(m.predict(d))
    out = DutyAssessor(m, AspectPolicy(), load_peak=p.load_peak, load_offpeak=p.load_offpeak,
                       bands=p.peak_bands).assess(a)
    return out, episodes, m


def test_pipeline_emits_duty_statistics(cycles):
    ref = cycles[cycles.ts < cycles.ts.min().floor("D") + pd.Timedelta(days=30)]
    d = HealthPipeline().fit(ref).transform(cycles)
    assert set(STATS) <= set(d) and (d.duty_n >= 0).all()


def test_duty_labels_and_invariants(assessed):
    out, _, m = assessed
    assert set(out.duty) <= {"full_service", "off_peak_only", "withdraw", "not_assessed"}
    # withdraw only when the day-average level is at or beyond the failure level
    assert (out.loc[out.duty == "withdraw", "prediction_state"] == "threshold_exceeded").all()
    # a restriction always rests on measured sensitivity, and the day average is below the level
    r = out[out.duty == "off_peak_only"]
    assert r.load_sensitive.all()
    assert (r.health_index_smooth < m.threshold_).all()
    assert (r.peak_index >= m.threshold_).all() and (r.peak_state == "threshold_exceeded").all()
    # unknown base evidence is never given a duty label
    assert (out.loc[out.aspect == -1, "duty"] == "not_assessed").all()


def test_restriction_precedes_fault_and_never_fires_on_healthy_doors(assessed):
    out, eps, _ = assessed
    s = duty_summary(out, eps)
    assert s["episodes_load_sensitive_before_fault"] >= len(eps) - 1
    assert s["episodes_restricted_before_fault"] >= 1
    assert s["healthy_asset_days_restricted"] == 0
    assert s["healthy_asset_days_flagged_sensitive"] / max(s["healthy_asset_days_assessed"], 1) < .04


def test_repair_does_not_leave_a_stale_restriction(assessed):
    out, eps, _ = assessed
    after = []
    for e in eps.itertuples():
        w = out[(out.asset_id == e.asset_id) & (out.day > e.fault_ts + pd.Timedelta(days=7))]
        after.append(w.duty)
    after = pd.concat(after) if after else pd.Series(dtype=object)
    assert not after.isin(["off_peak_only", "withdraw"]).any()


def test_sensitivity_gate_has_hysteresis():
    """Deterministic t-statistics: noise is built orthogonal to load, so the
    slope is exactly 2 and t = 2*sqrt(Sxx)/sd exactly. Sequence 3, 2, 2, 1, 3, 2."""
    days = pd.date_range("2026-01-01", periods=6)
    load = np.linspace(0, 1, 100)
    z = np.random.default_rng(0).normal(size=100)
    X = np.column_stack([np.ones(100), load])
    z = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    z = z / z.std(ddof=2)
    sxx = ((load - load.mean()) ** 2).sum()
    frames = []
    for i, t_target in enumerate((3., 2., 2., 1., 3., 2.)):
        sd = 2 * np.sqrt(sxx) / t_target
        frames.append(pd.DataFrame({"asset_id": "A", "ts": days[i] + pd.to_timedelta(load, unit="h"),
                                    "load_proxy": load, "cycle_index": 2 * load + sd * z}))
    stats = cycle_stats(pd.concat(frames))
    out = load_sensitivity(stats, window_days=1, min_cycles=10, min_t=2.5, exit_t=1.5).sort_values("day")
    assert out.load_sensitivity_t.round(6).tolist() == [3., 2., 2., 1., 3., 2.]
    assert out.load_sensitive.tolist() == [True, True, True, False, True, True]


def test_assess_requires_card_columns_and_reference_loads(assessed):
    out, _, m = assessed
    with pytest.raises(ValueError, match="reference loads"):
        DutyAssessor(m).assess(out)
    with pytest.raises(ValueError, match="aspect"):
        DutyAssessor(m, load_peak=.9, load_offpeak=.2).assess(out.drop(columns=["aspect", "rul_lower"]))


def test_render_states_the_limitation(assessed):
    out, _, _ = assessed
    text = render(out[out.duty == "off_peak_only"].iloc[0])
    assert "Off-peak service only" in text and "not separately calibrated" in text
    assert "%" not in text
