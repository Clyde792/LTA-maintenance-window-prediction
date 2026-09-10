import numpy as np
import pandas as pd
import pytest
from headway.deferral import DeferralLedger,compare_window,horizon_key,render

def frame():
    return pd.DataFrame({"asset_id":["A","B","C","D"],"day":pd.to_datetime(["2026-01-01"]*4),
        "available_at":pd.to_datetime(["2026-01-02"]*4),"rul_lower":[2.,0.,np.nan,0.],
        "prediction_state":["valid","valid","insufficient_history","threshold_exceeded"]})

def test_ledger_consumes_exact_card_margin_and_emits_no_risk():
    out=DeferralLedger().ledger(frame())
    assert list(out.window_3d)==["exceeds_margin","no_positive_margin","unknown","threshold_exceeded"]
    assert not any(c.startswith("risk_") or "safe_days" in c for c in out)
    pd.testing.assert_series_equal(out.rul_lower,frame().rul_lower)

def test_zero_bound_is_not_certain_failure():
    assert compare_window(0.,"valid",7.)=="no_positive_margin"
    assert compare_window(0.,"threshold_exceeded",7.)=="threshold_exceeded"

def test_act_now_does_not_override_unknown_or_threshold_exceeded():
    out=DeferralLedger().ledger(frame())
    assert list(out.window_0d)==["within_margin","no_positive_margin","unknown","threshold_exceeded"]

def test_fractional_windows_do_not_collide():
    out=DeferralLedger().ledger(frame(),horizons=(.25,.5))
    assert "window_0.25d" in out and "window_0.5d" in out

def test_booked_window_uses_hours_from_prediction_availability():
    out=DeferralLedger().ledger(frame(),window_at=pd.Timestamp("2026-01-02 12:00"))
    assert out.hours_to_window.eq(12.).all()
    assert out.scheduled_window.iloc[0]=="within_margin"

def test_past_window_rejected():
    with pytest.raises(ValueError,match="precedes"):
        DeferralLedger().ledger(frame(),window_at=pd.Timestamp("2026-01-01"))

@pytest.mark.parametrize("h",[-1,np.nan,np.inf])
def test_invalid_horizon_rejected(h):
    with pytest.raises(ValueError): horizon_key(h)

def test_probability_api_fails_explicitly():
    with pytest.raises(NotImplementedError,match="not calibrated"):
        DeferralLedger().risk(frame(),7)
    with pytest.raises(NotImplementedError):
        DeferralLedger().latest_date_under(frame(),.1)

def test_render_never_claims_probability_or_safe_date():
    row=DeferralLedger().ledger(frame()).iloc[0]
    text=render(row)
    assert "%" not in text and "safe until" not in text.lower()
    assert "Within estimated margin" in text
