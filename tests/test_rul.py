"""Regression tests for uncertainty states and fold isolation."""
import numpy as np
import pandas as pd
import pytest
from headway.rul import ConformalRUL, _project, prediction_time

def trajectories(n=5):
    rows=[]; eps=[]
    for j in range(n):
        days=pd.date_range("2026-01-01",periods=25)
        for k,d in enumerate(days):
            rows.append({"asset_id":f"A{j}","train_id":f"T{j}","day":d,
                         "health_index_smooth":1+k*(.7+j*.07),"health_index_slope":.7+j*.07,
                         "health_index_vol":1+k*.1})
        eps.append({"asset_id":f"A{j}","onset_ts":days[3],"fault_ts":days[24]+pd.Timedelta(hours=12)})
    return pd.DataFrame(rows),pd.DataFrame(eps)

def test_projection_forms_and_calendar_units():
    assert _project([10.],[2.],30.,.25,90.,"linear")[0]==pytest.approx(10.)
    assert _project([10.],[2.],30.,.25,90.,"loglinear")[0]==pytest.approx(np.log(3)/.2)

@pytest.mark.parametrize("hi,slope",[(np.nan,np.nan),(1.,.01),(5.,-.5)])
def test_no_projection_is_unknown_not_infinite_life(hi,slope):
    assert np.isnan(_project([hi],[slope],30.,.25,90.)[0])

def test_exceeded_threshold_has_zero_projection():
    assert _project([40.],[0.],30.,.25,90.)[0]==0.

def test_predict_requires_fit():
    with pytest.raises(RuntimeError,match="fit"):
        ConformalRUL().predict(trajectories()[0])

def test_missing_flat_and_elevated_states_are_distinct():
    d,e=trajectories(); m=ConformalRUL().fit(d,e,evaluate=False)
    q=d.iloc[:4].copy()
    q["health_index_smooth"]=[np.nan,0.,5.,m.threshold_+1]
    q["health_index_slope"]=[np.nan,0.,0.,0.]
    p=m.predict(q)
    assert list(p.prediction_state)==["insufficient_history","no_worsening_trend","elevated_no_trend","threshold_exceeded"]
    assert p.rul_lower.iloc[:3].isna().all()
    assert p.rul_lower.iloc[3]==0

def test_data_quality_and_staleness_override_estimates():
    d,e=trajectories(); m=ConformalRUL().fit(d,e,evaluate=False)
    p=m.predict(d.assign(stale_data=True))
    assert p.rul_lower.isna().all()
    assert set(p.prediction_state)=={"stale_data"}
    p=m.predict(d.assign(data_quality_ok=False))
    assert p.rul_lower.isna().all()

def test_fault_day_aggregate_is_not_used_before_it_is_available():
    d,e=trajectories()
    d["available_at"]=d.day+pd.Timedelta(days=1)
    e["fault_ts"]=pd.Timestamp("2026-01-25 12:00")
    baseline=ConformalRUL._threshold_from(d,e)
    d.loc[d.day==pd.Timestamp("2026-01-25"),"health_index_smooth"]=99999.
    assert ConformalRUL._threshold_from(d,e)==baseline

@pytest.mark.parametrize("mode",["additive","ratio","mondrian"])
def test_outer_fold_predictions_do_not_depend_on_held_out_fault_label(mode):
    d,e=trajectories()
    m=ConformalRUL(mode=mode,min_per_bin=5)
    before=m.out_of_fold_predict(d,e)
    e.loc[0,"fault_ts"]-=pd.Timedelta(days=8)
    after=m.out_of_fold_predict(d,e)
    mask=before.asset_id.eq("A0")
    pd.testing.assert_frame_equal(before.loc[mask,["rul_point","rul_lower","rul_upper"]],
                                  after.loc[mask,["rul_point","rul_lower","rul_upper"]])

def test_all_episodes_from_same_train_are_excluded_together():
    d,e=trajectories()
    d.loc[d.asset_id=="A1","train_id"]="T0"
    p=ConformalRUL().out_of_fold_predict(d,e)
    e.loc[1,"fault_ts"]-=pd.Timedelta(days=10)
    q=ConformalRUL().out_of_fold_predict(d,e)
    pd.testing.assert_series_equal(p.loc[p.asset_id=="A0","rul_lower"],q.loc[q.asset_id=="A0","rul_lower"])

def test_uncalibrated_model_cannot_invent_a_bound():
    d,e=trajectories(1)
    p=ConformalRUL().fit(d,e).predict(d.iloc[:5])
    assert p.rul_lower.isna().all()
    assert set(p.prediction_state)=={"uncalibrated"}

def test_rank_correction_is_conservative_at_small_sample_sizes():
    m=ConformalRUL(alpha=.1,min_per_bin=100)
    low,high=m._quantiles(np.arange(5.),np.zeros(5,int))
    assert np.isneginf(low[0]) and np.isposinf(high[0])

def test_empirical_coverage_is_group_counted():
    d,e=trajectories()
    m=ConformalRUL().fit(d,e)
    assert 0<=m.coverage_<=1
    assert m.n_eval_groups_==5 and m.n_groups_==5


def test_prediction_time_gate_marks_stale_and_future_observations():
    d,e=trajectories()
    m=ConformalRUL().fit(d,e,evaluate=False)
    p=m.predict(d.iloc[:1],as_of=pd.Timestamp("2026-01-04"))
    assert p.prediction_state.iloc[0]=="stale_data"
    assert np.isnan(p.rul_lower.iloc[0])
    p=m.predict(d.iloc[:1],as_of=pd.Timestamp("2025-12-31"))
    assert p.prediction_state.iloc[0]=="not_yet_available"


def test_threshold_exceeded_does_not_establish_upper_failure_time():
    d,e=trajectories()
    m=ConformalRUL().fit(d,e,evaluate=False)
    q=d.iloc[:1].copy()
    q["health_index_smooth"]=m.threshold_+5
    p=m.predict(q)
    assert p.rul_lower.iloc[0]==0
    assert np.isnan(p.rul_upper.iloc[0])
