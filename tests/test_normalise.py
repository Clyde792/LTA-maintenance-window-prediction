"""Normalisation is where the thesis lives. These tests check it actually
removes what it claims to remove, and degrades gracefully on unseen assets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway import contract
from headway.normalise import AssetBaseline, ConditionNormaliser


def test_removes_a_known_linear_effect():
    rng = np.random.default_rng(0)
    n = 4000
    temp = rng.uniform(24, 34, n)
    load = rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "ambient_temp_c": temp,
        "load_proxy": load,
        "hour_of_day": rng.uniform(6, 23, n),
        "cycles_since_service": rng.uniform(0, 2000, n),
        "y": 4.0 + 0.05 * temp + 0.40 * load + rng.normal(0, 0.05, n),
    })
    out = ConditionNormaliser(signal="y", context=contract.CONTEXT).fit_transform(df)

    # The residual must retain no usable linear dependence on the confounds.
    assert abs(np.corrcoef(out.residual, temp)[0, 1]) < 0.05
    assert abs(np.corrcoef(out.residual, load)[0, 1]) < 0.05
    # And it must reach the irreducible noise floor (0.05 by construction).
    # Asserting a fixed fraction of y.std() would be demanding better than the
    # noise, which is unachievable and would make this test a flake.
    assert out.residual.std() == pytest.approx(0.05, rel=0.15)
    assert out.residual.std() < 0.35 * out.y.std()


def test_preserves_a_signal_orthogonal_to_context():
    """Wear is not a function of the weather, so normalisation must not eat it."""
    rng = np.random.default_rng(1)
    n = 4000
    temp = rng.uniform(24, 34, n)
    wear = np.linspace(0, 1, n)                  # a slow drift, independent of temp
    df = pd.DataFrame({
        "ambient_temp_c": temp,
        "load_proxy": rng.uniform(0, 1, n),
        "hour_of_day": rng.uniform(6, 23, n),
        "cycles_since_service": rng.uniform(0, 2000, n),
        "y": 4.0 + 0.05 * temp + 0.6 * wear + rng.normal(0, 0.03, n),
    })
    out = ConditionNormaliser(signal="y", context=contract.CONTEXT).fit_transform(df)
    assert np.corrcoef(out.residual, wear)[0, 1] > 0.9, "degradation must survive"


def test_explain_ranks_the_dominant_conditions_first(cycles):
    sub = contract.get("door")
    cn = ConditionNormaliser(signal=sub.primary, context=contract.CONTEXT).fit(cycles)
    top2 = set(cn.explain().head(2).term)
    assert top2 == {"ambient_temp_c", "load_proxy"}


def test_robust_fit_resists_contamination():
    rng = np.random.default_rng(2)
    n = 3000
    temp = rng.uniform(24, 34, n)
    y = 4.0 + 0.05 * temp + rng.normal(0, 0.05, n)
    y[:150] += 5.0                                    # a badly degraded minority
    df = pd.DataFrame({
        "ambient_temp_c": temp, "load_proxy": rng.uniform(0, 1, n),
        "hour_of_day": rng.uniform(6, 23, n),
        "cycles_since_service": rng.uniform(0, 2000, n), "y": y,
    })
    ctx = contract.CONTEXT
    robust = ConditionNormaliser("y", ctx, robust=True, trim=0.10).fit_transform(df)
    naive = ConditionNormaliser("y", ctx, robust=False).fit_transform(df)
    # The contaminated rows should stand out MORE after a robust fit, because the
    # expected-value model has not been dragged up toward them.
    assert robust.residual[:150].mean() > naive.residual[:150].mean()


def test_transform_before_fit_raises():
    cn = ConditionNormaliser(signal="y", context=contract.CONTEXT)
    with pytest.raises(RuntimeError, match="fit"):
        cn.transform(pd.DataFrame({"y": [1.0]}))


def test_context_layout_change_between_fit_and_transform_raises(cycles):
    cn = ConditionNormaliser(signal="current_integral_as", context=contract.CONTEXT).fit(cycles)
    cn.context = ["ambient_temp_c"]                    # simulate a mismatched adapter
    with pytest.raises(ValueError, match="context layout"):
        cn.transform(cycles)


# ----------------------------------------------------------------- AssetBaseline

def _daily_frame(n_assets=6, n_days=60, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        offset = rng.normal(0, 0.5)
        for d in range(n_days):
            rows.append({
                "asset_id": f"A{a}",
                "day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=d),
                "residual": offset + rng.normal(0, 0.2),
            })
    return pd.DataFrame(rows)


def test_health_index_is_centred_and_scaled_on_the_reference_window():
    df = _daily_frame()
    out = AssetBaseline(value="residual", reference_days=21).fit_transform(df)
    ref = out[out.day < out.day.min() + pd.Timedelta(days=21)]
    per_asset = ref.groupby("asset_id").health_index
    assert per_asset.median().abs().max() < 0.35, "each asset must sit near zero"
    assert 0.5 < ref.health_index.std() < 2.0, "units should be roughly sigma"


def test_per_asset_offsets_are_removed():
    """The whole point: a stiff healthy door must not outrank a degrading one.

    Both spreads are expressed in units of within-asset noise, because the raw
    residual and the health index are in different units - comparing them
    directly (as an earlier version of this test did) compares apples to sigmas.
    """
    df = _daily_frame()
    out = AssetBaseline(value="residual", reference_days=21).fit_transform(df)

    within = df.groupby("asset_id").residual.std().mean()
    raw_spread_sigma = df.groupby("asset_id").residual.mean().std() / within
    idx_spread_sigma = out.groupby("asset_id").health_index.mean().std()

    assert raw_spread_sigma > 2.0, "the fixture must have real between-asset spread"
    assert idx_spread_sigma < 0.5, (
        f"after baselining, assets should sit within half a sigma of each other, "
        f"got {idx_spread_sigma:.2f}"
    )


def test_unseen_asset_falls_back_to_the_fleet_instead_of_crashing():
    df = _daily_frame()
    ab = AssetBaseline(value="residual").fit(df)
    newcomer = pd.DataFrame({
        "asset_id": ["BRAND-NEW-TRAIN"],
        "day": [pd.Timestamp("2026-06-01")],
        "residual": [0.4],
    })
    out = ab.transform(newcomer)
    assert np.isfinite(out.health_index).all()


def test_degrading_asset_scores_higher_near_its_fault(cycles, episodes):
    from headway import features
    sub = contract.get("door")
    cyc = ConditionNormaliser(sub.primary, contract.CONTEXT).fit_transform(cycles)
    daily = AssetBaseline().fit_transform(features.to_daily(cyc, "door"))

    e = episodes.iloc[0]
    a = daily[daily.asset_id == e.asset_id]
    late = a[(a.day > e.fault_ts - pd.Timedelta(days=5)) & (a.day <= e.fault_ts)]
    early = a[a.day < e.onset_ts]
    assert late.health_index.mean() > early.health_index.mean() + 2.0


# ------------------------------------------------ MultivariateHealthIndex

def _mv_frame(n_assets=6, n_days=80, wear_from=50, seed=11):
    """Two correlated 'up with wear' signals and one 'down with wear'."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        for d in range(n_days):
            wear = max(d - wear_from, 0) * 0.06 if a == 0 else 0.0
            shared = rng.normal(0, 1.0)          # a confound both currents share
            rows.append({
                "asset_id": f"A{a}",
                "day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=d),
                "peak_hx": shared + wear + rng.normal(0, 0.2),
                "integral_hx": shared + wear + rng.normal(0, 0.2),
                "travel_hx": -wear + rng.normal(0, 0.2),   # DOWN with wear
            })
    return pd.DataFrame(rows)


SIGS = ["peak_hx", "integral_hx", "travel_hx"]
ORIENT = {"peak_hx": 1, "integral_hx": 1, "travel_hx": -1}


def test_index_is_standard_normal_on_healthy_data():
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame(wear_from=10_000)                    # nobody degrades
    out = MultivariateHealthIndex(signals=SIGS, orientation=ORIENT).fit_transform(df)
    assert abs(out.health_index.mean()) < 0.3
    assert 0.7 < out.health_index.std() < 1.4


def test_index_rises_with_wear():
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame()
    out = MultivariateHealthIndex(signals=SIGS, orientation=ORIENT).fit_transform(df)
    sick = out[(out.asset_id == "A0") & (out.day > out.day.max() - pd.Timedelta(days=10))]
    well = out[out.asset_id != "A0"]
    assert sick.health_index.mean() > well.health_index.mean() + 2.0


def test_orientation_is_load_bearing():
    """A signal that falls with wear must be flipped, or it cancels the signals
    that rise - quietly subtracting the evidence we are trying to accumulate."""
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame()

    def separation(orientation):
        out = MultivariateHealthIndex(signals=SIGS, orientation=orientation).fit_transform(df)
        sick = out[(out.asset_id == "A0") & (out.day > out.day.max() - pd.Timedelta(days=10))]
        well = out[out.asset_id != "A0"]
        return sick.health_index.mean() - well.health_index.mean()

    correct = separation(ORIENT)
    wrong = separation({k: 1 for k in SIGS})            # travel not flipped
    assert correct > wrong, "flipping the falling signal must help, not hurt"


def test_whitening_stops_correlated_signals_double_counting():
    """peak and integral move together. Without whitening they would count as
    two independent votes and the confound would dominate."""
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame(wear_from=10_000)
    out = MultivariateHealthIndex(signals=SIGS, orientation=ORIENT).fit_transform(df)
    naive = (df.peak_hx + df.integral_hx - df.travel_hx) / np.sqrt(3)
    assert out.health_index.std() < naive.std(), "whitening must shrink shared variance"


def test_mahalanobis_mode_is_unsigned():
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame()
    out = MultivariateHealthIndex(signals=SIGS, orientation=ORIENT,
                                  mode="mahalanobis").fit_transform(df)
    assert (out.health_index >= 0).all()


def test_projection_mode_is_signed_unlike_distance():
    """RUL projects a quantity forward to a threshold, so the index must be able
    to be NEGATIVE when an asset is healthier than its baseline. A distance
    cannot, which is why distance is not the default."""
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame(wear_from=10_000)
    out = MultivariateHealthIndex(signals=SIGS, orientation=ORIENT).fit_transform(df)
    assert (out.health_index < 0).any()


def test_transform_before_fit_raises():
    from headway.normalise import MultivariateHealthIndex
    with pytest.raises(RuntimeError, match="fit"):
        MultivariateHealthIndex(signals=SIGS).transform(_mv_frame())


def test_perfectly_collinear_signals_do_not_blow_up():
    from headway.normalise import MultivariateHealthIndex
    df = _mv_frame(wear_from=10_000)
    df["copy_hx"] = df.peak_hx                          # exact duplicate
    out = MultivariateHealthIndex(
        signals=SIGS + ["copy_hx"],
        orientation={**ORIENT, "copy_hx": 1},
    ).fit_transform(df)
    assert np.isfinite(out.health_index).all()


def test_contract_declares_travel_falls_with_wear():
    assert contract.get("door").sign("travel_mm") == -1
    assert contract.get("door").sign("peak_current_a") == 1
    assert contract.get("door").sign("not_a_column") == 1, "default must be +1"
