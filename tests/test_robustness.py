import numpy as np
import pandas as pd
import pytest
from headway.robustness import perturb, align_to_expected, score_frozen, SCENARIOS


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stress_cannot_change_history_or_unselected_assets(cycles, scenario):
    before = cycles.copy(deep=True)
    cutoff = cycles.ts.min().floor("D") + pd.Timedelta(days=60)
    asset = cycles.asset_id.iloc[0]
    changed = perturb(cycles, scenario, after=cutoff, assets=[asset], seed=1)
    unchanged = (cycles.ts < cutoff) | cycles.asset_id.ne(asset)
    pd.testing.assert_frame_equal(cycles.loc[unchanged], changed.loc[unchanged[unchanged].index])
    pd.testing.assert_frame_equal(cycles, before)


def test_missing_days_remain_in_score_denominator():
    d = pd.DataFrame({"asset_id": ["A"] * 3, "day": pd.date_range("2026-01-01", periods=3)})
    d["available_at"] = d.day + pd.Timedelta(days=1)
    result = align_to_expected(d.iloc[:2].assign(score=1.), d, ["score"])
    assert len(result) == 3
    assert np.isfinite(result.score).mean() == pytest.approx(2 / 3)


def test_frozen_scoring_never_refits_and_masks_unsupported_inputs():
    class Frozen:
        column = "value"
        def fit(self, _): raise AssertionError("must not refit")
        def score(self, d): return d.value.to_numpy()
    d = pd.DataFrame({"asset_id": ["A"] * 4, "day": pd.date_range("2026-01-01", periods=4),
        "value": [1., 2., 3., 4.], "data_quality_ok": True, "context_supported": [True, True, True, False]})
    s = score_frozen({"score": Frozen()}, d)
    assert s.score.iloc[2] == 2.
    assert np.isnan(s.score.iloc[3])


def test_invalid_history_cannot_enter_later_smoothed_scores():
    class Frozen:
        column = "value"
        def score(self, d): return d.value.to_numpy()
    d = pd.DataFrame({"asset_id": ["A"] * 6, "day": pd.date_range("2026-01-01", periods=6),
        "value": [1., 999., 3., 4., 5., 6.], "data_quality_ok": True,
        "context_supported": [True, False, True, True, True, True]})
    s = score_frozen({"score": Frozen()}, d)
    assert s.score.iloc[:4].isna().all()
    assert s.score.iloc[4] == 4.
    changed = d.copy()
    changed.loc[5, "value"] = 10000.
    pd.testing.assert_series_equal(s.score.iloc[:5], score_frozen({"score": Frozen()}, changed).score.iloc[:5])


def test_rul_comparison_uses_same_finite_rows(monkeypatch):
    from scripts import experiment_robustness as experiment
    d = pd.DataFrame({"asset_id": ["A"] * 3, "day": pd.date_range("2026-01-01", periods=3),
                      "available_at": pd.date_range("2026-01-02", periods=3), "rul_point": [5., 100., np.nan]})
    other = d.assign(rul_point=[4., np.nan, 3.])
    monkeypatch.setattr(experiment, "true_rul", lambda *_: np.array([3., 2., 1.]))
    result = experiment.matched_rul(d, other, None)
    assert result == {"matched_episode_rows": 1, "base_mae_days": 2., "onboard_mae_days": 1.}
