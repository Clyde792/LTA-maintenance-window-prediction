"""Edge cases found in the scientific and implementation audit."""
import numpy as np
import pandas as pd
import pytest
from headway.features import add_trend,to_daily
from headway.normalise import ConditionNormaliser,MultivariateHealthIndex
from headway.aspect import AspectPolicy,Aspect
from headway.evaluate.metrics import replay_alerts,evaluate
from headway.validation import horizon_outcomes

def test_elapsed_calendar_slope_with_missing_days():
    day=pd.Timestamp("2026-01-01")+pd.to_timedelta([0,1,4,5,8,9,12],unit="D")
    d=pd.DataFrame({"asset_id":"A","day":day,"health_index":np.array([0,1,4,5,8,9,12])*.2})
    p=add_trend(d,smooth_days=1,slope_days=14)
    assert p.health_index_slope.iloc[-1]==pytest.approx(.2)

def test_missing_current_reading_is_not_filled_by_smoothing():
    d=pd.DataFrame({"asset_id":"A","day":pd.date_range("2026-01-01",periods=20),
                    "health_index":[1.]*19+[np.nan]})
    p=add_trend(d)
    assert np.isnan(p.health_index_smooth.iloc[-1])
    assert np.isnan(p.health_index_slope.iloc[-1])

def test_context_imputation_is_batch_invariant():
    train=pd.DataFrame({"x":np.arange(100.),"y":2*np.arange(100.)})
    m=ConditionNormaliser("y",["x"]).fit(train)
    a=pd.DataFrame({"x":[np.nan],"y":[3.]})
    b=pd.concat([a,pd.DataFrame({"x":[-10000.,10000.],"y":[0.,0.]})],ignore_index=True)
    assert m.transform(a).expected.iloc[0]==pytest.approx(m.transform(b).expected.iloc[0])
    assert not m.transform(a).context_supported.iloc[0]

def test_missing_signal_cannot_become_zero_health():
    rng=np.random.default_rng(2)
    d=pd.DataFrame({"day":pd.date_range("2026-01-01",periods=60),"x":rng.normal(size=60),"y":rng.normal(size=60)})
    m=MultivariateHealthIndex(["x","y"]).fit(d)
    d.loc[59,"x"]=np.nan
    assert np.isnan(m.transform(d).health_index.iloc[-1])

def test_short_reference_cannot_use_future_data():
    d=pd.DataFrame({"day":pd.to_datetime(["2026-01-01"]+["2026-03-01"]*40),"x":np.arange(41.),"y":np.arange(41.)})
    with pytest.raises(ValueError,match="reference"):
        MultivariateHealthIndex(["x","y"]).fit(d)

def test_worsening_one_oriented_channel_cannot_lower_directional_score():
    rng=np.random.default_rng(3)
    d=pd.DataFrame({"day":pd.date_range("2026-01-01",periods=60),"x":rng.normal(size=60),"y":rng.normal(size=60)})
    m=MultivariateHealthIndex(["x","y"]).fit(d)
    q=d.iloc[:1].copy()
    before=m.transform(q).health_index.iloc[0]
    q["x"]+=5
    assert m.transform(q).health_index.iloc[0]>=before

def test_unknown_cannot_clear_red_hysteresis():
    p=AspectPolicy()
    assert list(p._damp(np.array([3,-1,-1,-1,0,0,0])))==[3,3,3,3,3,3,0]
    assert list(p.raw([np.nan,np.inf]))==[-1,-1]

def test_gaps_break_consecutive_clear_days():
    days=pd.to_datetime(["2026-01-01","2026-01-02","2026-01-08","2026-01-09","2026-01-10"])
    assert list(AspectPolicy()._damp(np.array([3,0,0,0,0]),days))==[3,3,3,3,0]

def test_future_scores_cannot_change_prior_replay_alerts():
    d=pd.DataFrame({"asset_id":["A","B"]*8,"day":np.repeat(pd.date_range("2026-01-01",periods=8),2),
                    "score":np.tile([2.,1.],8)})
    before=replay_alerts(d[d.day<pd.Timestamp("2026-01-05")],"score",.5,daily_budget=1)
    d.loc[d.day>=pd.Timestamp("2026-01-05"),"score"]=100000.
    after=replay_alerts(d,"score",.5,daily_budget=1)
    pd.testing.assert_frame_equal(before,after[after.day<pd.Timestamp("2026-01-05")])

def test_fault_day_aggregate_cannot_warn_about_earlier_fault():
    d=pd.DataFrame({"asset_id":["A"],"day":pd.to_datetime(["2026-01-01"]),
        "available_at":pd.to_datetime(["2026-01-02"]),"score":[9.]})
    e=pd.DataFrame({"asset_id":["A"],"onset_ts":pd.to_datetime(["2025-12-20"]),
        "fault_ts":pd.to_datetime(["2026-01-01 12:00"])})
    assert evaluate(d,"score",e,1).detected==0

def test_censored_horizon_does_not_become_negative_label():
    d=pd.DataFrame({"asset_id":["A","B"],"day":pd.to_datetime(["2026-01-01"]*2)})
    e=pd.DataFrame({"asset_id":["A"],"onset_ts":pd.to_datetime(["2025-12-20"]),
        "fault_ts":pd.to_datetime(["2026-01-03"])})
    out=horizon_outcomes(d,e,7,pd.Timestamp("2026-01-04"))
    assert out.iloc[0]==1 and np.isnan(out.iloc[1])

def test_capture_ignores_unproven_physical_ceiling():
    d=pd.DataFrame({"asset_id":["A"],"day":pd.to_datetime(["2026-01-05"]),"score":[9.]})
    e=pd.DataFrame({"asset_id":["A"],"onset_ts":pd.to_datetime(["2026-01-01"]),
        "fault_ts":pd.to_datetime(["2026-01-11"]),"detectable_days":[1.]})
    assert evaluate(d,"score",e,1).worst_capture==pytest.approx(.6)

def test_zero_margin_model_is_not_selected_as_a_validated_winner(monkeypatch):
    from headway.rul import ConformalRUL
    from headway.validation import select_rul
    d=pd.DataFrame({"asset_id":["A","B","C","D"],"train_id":["A","B","C","D"],
        "day":pd.to_datetime(["2026-01-05"]*4)})
    e=pd.DataFrame({"asset_id":d.asset_id,"onset_ts":pd.to_datetime(["2026-01-01"]*4),
        "fault_ts":pd.to_datetime(["2026-01-11"]*4)})
    def fake_oof(self,daily,episodes):
        return daily.assign(rul_point=6.,rul_lower=0. if self.projection=="linear" else 7.)
    monkeypatch.setattr(ConformalRUL,"out_of_fold_predict",fake_oof)
    monkeypatch.setattr(ConformalRUL,"fit",lambda self,*args,**kwargs:self)
    model,report=select_rul(d,e)
    assert not report["qualified"]
    assert "fixed baseline" in report["selection_reason"]
    assert model.projection=="loglinear"
