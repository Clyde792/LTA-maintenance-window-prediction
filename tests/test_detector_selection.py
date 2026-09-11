import copy
import numpy as np
import pandas as pd
import pytest
from headway.detector_selection import SelectionPolicy, select_detector, assess_selected


@pytest.fixture
def sample():
    rows = []
    for a in ("A", "B", "C"):
        for i, day in enumerate(pd.date_range("2026-01-01", periods=30)):
            score = 1 + i / 10 if i < 9 else 10. if 12 <= i <= 16 or 24 <= i <= 27 else 0.
            rows.append({"asset_id": a, "day": day, "available_at": day + pd.Timedelta(days=1),
                         "good": score, "missing": np.nan})
    eps = pd.DataFrame([{"asset_id": a, "train_id": a, "onset_ts": pd.Timestamp(onset),
        "fault_ts": pd.Timestamp(fault)} for a in ("A", "B", "C")
        for onset, fault in (("2026-01-12", "2026-01-19"), ("2026-01-24", "2026-01-29"))])
    return pd.DataFrame(rows), eps


def select(sample):
    scores, episodes = sample
    return select_detector(scores, episodes, ["good", "missing"], calibration_start="2026-01-01",
        calibration_end="2026-01-10", validation_end="2026-01-21")


def test_selection_qualifies_and_missing_model_cannot_win(sample):
    selected = select(sample)
    assert selected["selected_model"] == "good"
    assert not selected["candidates"][1]["qualified"]
    assert assess_selected(selected, *sample)["detected"] == 3


def test_future_scores_and_labels_cannot_change_selection(sample):
    before = select(sample)
    scores, eps = sample
    scores.loc[scores.available_at > "2026-01-21", "good"] = 99999
    eps.loc[eps.fault_ts > "2026-01-21", "fault_ts"] = pd.Timestamp("2026-12-01")
    assert select(sample) == before


def test_validation_scores_do_not_retune_threshold(sample):
    before = select(sample)["candidates"][0]["threshold"]
    sample[0].loc[sample[0].available_at > "2026-01-10", "good"] = 100
    assert select(sample)["candidates"][0]["threshold"] == before


def test_no_qualified_model_means_no_test_winner(sample):
    s, e = sample
    result = select((s, e[e.train_id == "A"]))
    assert result["selected_model"] is None
    assert assess_selected(result, s, e)["status"] == "not_evaluated"


def test_final_test_cannot_mutate_selection(sample):
    selected = select(sample)
    original = copy.deepcopy(selected)
    assess_selected(selected, *sample)
    assert selected == original


def test_crossing_episode_is_not_credited_as_fully_observed_validation(sample):
    s, e = sample
    e.loc[(e.train_id == "A") & (e.fault_ts < "2026-01-21"), "onset_ts"] = pd.Timestamp("2026-01-05")
    result = select(sample)
    assert result["excluded_crossing_assets"] == ["A"]
    assert result["selected_model"] is None


def test_duplicate_asset_time_is_rejected(sample):
    s, e = sample
    with pytest.raises(ValueError, match="unique"):
        select((pd.concat([s, s.iloc[:1]], ignore_index=True), e))


def test_invalid_policy():
    with pytest.raises(ValueError):
        SelectionPolicy(threshold_quantile=1)
