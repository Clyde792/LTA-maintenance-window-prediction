import json
import numpy as np
import pandas as pd
from scripts.build_ui import build_payload
from headway.aspect import AspectPolicy

def test_missing_day_remains_unknown_and_retains_prior_escalation():
    d=pd.DataFrame({
        "asset_id":["A","A","B"],"train_id":["T","T","U"],
        "day":pd.to_datetime(["2026-01-01","2026-01-03","2026-01-02"]),
        "prediction_state":["valid"]*3,"aspect":[3,3,0],"raw_aspect":[3,3,0],
        "rul_lower":[.2,.1,30.],"rul_point":[1.,1.,40.],
        "health_index_smooth":[6.,7.,0.],"health_index_slope":[1.,1.,0.],
        "decision_stability":["HIGH"]*3})
    payload=build_payload(d,AspectPolicy())
    json.dumps(payload,allow_nan=False)
    missing=payload["assets"][0]["rows"][1]
    assert missing["state"]=="missing_observation"
    assert missing["aspect"]==3 and missing["raw"]==-1
    assert missing["margin"] is None
    assert set(missing["windows"].values())=={"unknown"}


def test_duty_fields_default_to_not_assessed_when_absent():
    d=pd.DataFrame({
        "asset_id":["A"],"train_id":["T"],"day":pd.to_datetime(["2026-01-01"]),
        "prediction_state":["valid"],"aspect":[1],"raw_aspect":[1],
        "rul_lower":[5.],"rul_point":[9.],"health_index_smooth":[4.],"health_index_slope":[1.],
        "decision_stability":["HIGH"]})
    payload=build_payload(d,AspectPolicy())
    json.dumps(payload,allow_nan=False)
    row=payload["assets"][0]["rows"][0]
    assert row["duty"]=="not_assessed" and row["sensitive"] is False and row["peakHi"] is None
    assert payload["duty"]["bands"]=="no peak identified"


def test_duty_fields_are_carried_through():
    d=pd.DataFrame({
        "asset_id":["A"],"train_id":["T"],"day":pd.to_datetime(["2026-01-01"]),
        "prediction_state":["valid"],"aspect":[2],"raw_aspect":[2],
        "rul_lower":[2.],"rul_point":[4.],"health_index_smooth":[30.],"health_index_slope":[3.],
        "decision_stability":["HIGH"],"duty":["off_peak_only"],"duty_reason":["level"],
        "peak_index":[41.],"offpeak_index":[28.],"load_sensitivity_t":[6.2],"load_sensitive":[True],
        "duty_bands":["06:00–10:00, 16:00–20:00"],"threshold":[39.1]})
    d["available_at"] = d.day + pd.Timedelta(days=1)
    d["data_quality_ok"] = True
    d["context_supported"] = True
    payload=build_payload(d,AspectPolicy())
    row=payload["assets"][0]["rows"][0]
    assert row["duty"]=="off_peak_only" and row["sensitive"] and row["peakHi"]==41.
    assert payload["duty"]["threshold"]==39.1 and "06:00" in payload["duty"]["bands"]
