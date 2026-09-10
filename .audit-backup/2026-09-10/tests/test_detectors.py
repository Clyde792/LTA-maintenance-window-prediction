"""The detector zoo. The protocol tests matter more than the detectors: if a
model can see the failure it is being asked to predict, every lead-time number
in the tournament is fiction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway.models import detectors as D


@pytest.fixture
def toy():
    """4 assets x 60 days. A3 drifts upward from day 30."""
    rng = np.random.default_rng(4)
    rows = []
    for a in ("A0", "A1", "A2", "A3"):
        offset = rng.normal(0, 0.4)
        for i in range(60):
            drift = max(i - 30, 0) * 0.25 if a == "A3" else 0.0
            rows.append({
                "asset_id": a,
                "day": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i),
                "f1": 4.0 + offset + drift + rng.normal(0, 0.1),
                "f2": 1.0 + 0.5 * drift + rng.normal(0, 0.1),
                "f3": rng.normal(0, 0.1),        # pure noise, carries no signal
            })
    return pd.DataFrame(rows)


FEATS = ["f1", "f2", "f3"]


def _all(feats=FEATS):
    return {
        "raw": D.RawThreshold(column="f1"),
        "ewma": D.EWMAChart(column="f1"),
        "iforest": D.IsolationForestDetector(needs=feats),
        "pca": D.PCAReconstruction(needs=feats),
        "maha": D.MahalanobisDetector(needs=feats),
        "lof": D.LOFDetector(needs=feats, n_neighbors=10),
    }


# ------------------------------------------------------------------- protocol

def test_every_detector_scores_every_row(toy):
    out = D.run(_all(), toy, reference_days=25)
    for k in _all():
        assert k in out.columns
        assert len(out[k]) == len(toy)


def test_run_preserves_caller_row_order(toy):
    """Same misalignment hazard as add_trend and AspectPolicy.apply."""
    shuffled = toy.sample(frac=1.0, random_state=11)
    a = D.run(_all(), toy, reference_days=25)
    b = D.run(_all(), shuffled, reference_days=25)
    assert b.index.equals(shuffled.index)
    merged = a.merge(b[["asset_id", "day", "maha"]], on=["asset_id", "day"],
                     suffixes=("_a", "_b"))
    pd.testing.assert_series_equal(merged.maha_a, merged.maha_b, check_names=False)


def test_detectors_never_see_beyond_the_reference_window(toy):
    """THE protocol property. A detector fitted on the whole record has already
    seen the failure it is being congratulated for predicting.

    Truncating everything after the reference window must not change the fitted
    model, and therefore must not change the scores on the reference rows.
    """
    ref_days = 25
    cutoff = toy.day.min() + pd.Timedelta(days=ref_days)
    full = D.run(_all(), toy, reference_days=ref_days)
    truncated = D.run(_all(), toy[toy.day < cutoff], reference_days=ref_days)

    a = full[full.day < cutoff].set_index(["asset_id", "day"]).sort_index()
    b = truncated.set_index(["asset_id", "day"]).sort_index()
    for k in _all():
        np.testing.assert_allclose(a[k].to_numpy(), b[k].to_numpy(), rtol=1e-9,
                                   err_msg=f"{k} changed when the future was removed")


def test_higher_score_means_worse_for_every_detector(toy):
    """Convention the whole tournament depends on. A detector that got this
    backwards would silently rank the healthiest doors as the sickest."""
    out = D.run(_all(), toy, reference_days=25)
    late = out[out.day > out.day.max() - pd.Timedelta(days=10)]
    for k in _all():
        sick = late[late.asset_id == "A3"][k].median()
        well = late[late.asset_id != "A3"][k].median()
        assert sick > well, f"{k} scored the degrading asset below the healthy ones"


def test_smoothing_is_applied_identically(toy):
    """Comparison must measure the detector, not who smoothed harder."""
    out = D.run({"maha": D.MahalanobisDetector(needs=FEATS)}, toy,
                reference_days=25, smooth_days=5)
    a = out[out.asset_id == "A0"].sort_values("day")
    assert a.maha.head(2).isna().all(), "min_periods must suppress the first rows"


# ------------------------------------------------------------- individual bits

def test_raw_threshold_is_the_column_itself(toy):
    det = D.RawThreshold(column="f1").fit(toy)
    np.testing.assert_allclose(det.score(toy), toy.f1.to_numpy())


def test_ewma_centres_each_asset_on_its_own_baseline(toy):
    """A per-asset chart must not rank a stiff healthy door above a sick one."""
    det = D.EWMAChart(column="f1").fit(toy[toy.day < toy.day.min() + pd.Timedelta(days=25)])
    toy = toy.assign(s=det.score(toy))
    early = toy[toy.day < toy.day.min() + pd.Timedelta(days=20)]
    spread = early.groupby("asset_id").s.mean()
    assert spread.abs().max() < 1.5, "per-asset offsets should be centred out"


def test_ewma_handles_an_unseen_asset(toy):
    det = D.EWMAChart(column="f1").fit(toy)
    newcomer = pd.DataFrame({
        "asset_id": ["NEW"] * 3,
        "day": pd.date_range("2026-06-01", periods=3),
        "f1": [4.0, 4.1, 4.2],
    })
    assert np.isfinite(det.score(newcomer)).all()


def test_pca_reconstruction_error_is_non_negative(toy):
    det = D.PCAReconstruction(needs=FEATS, n_components=2).fit(toy)
    assert (det.score(toy) >= 0).all()


def test_pca_component_count_is_clamped_to_the_feature_count(toy):
    det = D.PCAReconstruction(needs=["f1"], n_components=5).fit(toy)
    assert det.model_.n_components_ == 1


def test_mahalanobis_is_near_zero_at_the_centre_of_the_reference(toy):
    ref = toy[toy.day < toy.day.min() + pd.Timedelta(days=25)]
    det = D.MahalanobisDetector(needs=FEATS).fit(ref)
    centre = pd.DataFrame([ref[FEATS].mean()])
    assert det.score(centre)[0] < det.score(ref).mean()


def test_nan_features_do_not_crash_a_detector(toy):
    dirty = toy.copy()
    dirty.loc[dirty.index[:20], "f2"] = np.nan
    out = D.run({"maha": D.MahalanobisDetector(needs=FEATS)}, dirty, reference_days=25)
    assert out.maha.notna().sum() > 0


def test_passthrough_returns_the_named_column(toy):
    toy = toy.assign(precomputed=np.arange(len(toy), dtype=float))
    det = D.Passthrough(column="precomputed").fit(toy)
    np.testing.assert_allclose(det.score(toy), toy.precomputed.to_numpy())
