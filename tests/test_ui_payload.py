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
