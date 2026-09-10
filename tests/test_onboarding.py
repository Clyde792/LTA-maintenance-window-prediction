import numpy as np
import pandas as pd
import pytest
from headway.pipeline import HealthPipeline
from headway.onboarding import onboard
from headway.calibration_experiments import weighted_quantile, GroupBalancedRUL
from headway.rul import ConformalRUL


@pytest.fixture(scope="module")
def setup(cycles):
    start=cycles.ts.min().floor("D")
    group=cycles.train_id.iloc[0]
    training=cycles[cycles.train_id!=group]
    target=cycles[cycles.train_id==group]
    pipe=HealthPipeline().fit(training[training.ts<start+pd.Timedelta(days=30)])
    ready=start+pd.Timedelta(days=21)
    return pipe,target,ready


def test_onboarding_does_not_mutate_fleet_or_use_labels(setup):
    pipe,target,ready=setup
    ref=target[target.ts<ready]
    original={s:b.loc_.copy() for s,b in pipe.baselines.items()}
    a=onboard(pipe,ref,as_of=ready)
    b=onboard(pipe,ref.assign(fault_confirmed=True,fault_mode="invented"),as_of=ready)
    assert a.ready_at and a.ready_at==b.ready_at
    for s in pipe.levels:
        pd.testing.assert_series_equal(pipe.baselines[s].loc_,original[s])
        pd.testing.assert_series_equal(a.pipeline.baselines[s].loc_,b.pipeline.baselines[s].loc_)
        np.testing.assert_array_equal(a.pipeline.normalisers[s].model_.coef_,pipe.normalisers[s].model_.coef_)
    np.testing.assert_array_equal(a.pipeline.multivariate.weights_,pipe.multivariate.weights_)


def test_future_reference_is_rejected(setup):
    pipe,target,ready=setup
    with pytest.raises(ValueError,match="unavailable"):
        onboard(pipe,target,as_of=ready)


def test_short_reference_abstains(setup):
    pipe,target,ready=setup
    short=target[target.ts<ready-pd.Timedelta(days=5)]
    a=onboard(pipe,short,as_of=ready)
    assert a.rejected and not a.ready_at
    assert not a.transform(target).data_quality_ok.any()


def test_onboarding_reference_cannot_create_earlier_decisions(setup):
    pipe,target,ready=setup
    a=onboard(pipe,target[target.ts<ready],as_of=ready)
    pred=a.transform(target)
    assert not pred.loc[pred.available_at<ready,"data_quality_ok"].any()
    assert pred.loc[pred.available_at<ready,"onboarding_status"].eq("reference_not_yet_available").all()


def test_future_target_changes_cannot_change_earlier_predictions(setup):
    pipe,target,ready=setup
    a=onboard(pipe,target[target.ts<ready],as_of=ready)
    cutoff=ready+pd.Timedelta(days=15)
    before=a.transform(target)
    modified=target.copy()
    modified.loc[modified.ts>=cutoff,"peak_current_a"]*=100
    after=a.transform(modified)
    cols=["health_index_smooth","health_index_slope"]
    pd.testing.assert_frame_equal(before.loc[before.available_at<=cutoff,cols],after.loc[after.available_at<=cutoff,cols])


def test_equal_group_weight_does_not_change_if_one_group_is_repeated():
    a=[-4.,-3.,-2.]; b=[1.,2.]
    first=weighted_quantile(a+b,[1/3]*3+[1/2]*2,.6)
    repeated=weighted_quantile(a*10+b,[1/30]*30+[1/2]*2,.6)
    assert first==repeated


def test_balanced_calibration_keeps_held_out_labels_out(monkeypatch):
    from test_rul import trajectories
    d,e=trajectories()
    m=GroupBalancedRUL()
    before=m.out_of_fold_predict(d,e)
    e.loc[0,"fault_ts"]-=pd.Timedelta(days=8)
    after=m.out_of_fold_predict(d,e)
    pd.testing.assert_series_equal(before.loc[before.asset_id=="A0","rul_lower"],
                                   after.loc[after.asset_id=="A0","rul_lower"])
