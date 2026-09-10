"""Run chronological and end-to-end held-out-train checks on synthetic data."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.pipeline import HealthPipeline
from headway.validation import select_rul,coverage_summary
from headway.rul import prediction_time
from headway.evaluate.metrics import evaluate_replay
from headway.aspect import AspectPolicy
from headway.duty import DutyAssessor,duty_summary
ROOT=Path(__file__).resolve().parents[1]

def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list): return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
    if isinstance(v,np.generic): return v.item()
    return v

def main():
    cycles=pd.read_parquet(ROOT/"data/door_cycles.parquet")
    eps=pd.read_csv(ROOT/"data/door_episodes.csv",parse_dates=["onset_ts","fault_ts"])
    start=cycles.ts.min().floor("D")
    end=cycles.ts.max().ceil("D")
    cutoff=(start+(end-start)*.7).floor("D")
    ref_end=start+pd.Timedelta(days=30)
    pipeline=HealthPipeline().fit(cycles[cycles.ts<ref_end])
    daily=pipeline.transform(cycles)
    train=daily[prediction_time(daily)<=cutoff]
    test=daily[prediction_time(daily)>cutoff]
    train_eps=eps[eps.fault_ts<=cutoff]
    test_eps=eps[(eps.fault_ts>cutoff)&(eps.fault_ts<=end)]
    model,selection=select_rul(train,train_eps)
    pred=model.predict(test)
    reference=train[(train.available_at>ref_end)&train.data_quality_ok & train.context_supported]
    threshold=float(reference.health_index_smooth.quantile(.99))
    detection=evaluate_replay(test,"health_index_smooth",test_eps,threshold,daily_budget=2,cooldown_days=3)
    duty=DutyAssessor(model,AspectPolicy(),load_peak=pipeline.load_peak,load_offpeak=pipeline.load_offpeak,
        bands=pipeline.peak_bands).assess(AspectPolicy().apply(pred))
    pred.to_parquet(ROOT/"data/chronological_predictions.parquet",index=False)
    report={"data":"synthetic","cutoff":str(cutoff),"training_fault_groups":len(train_eps),
        "future_fault_episodes":len(test_eps),"chronological_rul":coverage_summary(pred,test_eps),
        "chronological_detection":detection.as_row(),"frozen_alert_threshold":threshold,
        "chronological_duty":{**duty_summary(duty,test_eps),"peak_bands":[list(b) for b in pipeline.peak_bands],
            "note":"Level comparison at peak load; wear-load interaction is a simulator assumption."},
        "selection":selection,"limitations":[
            "Same simulator family, not external fleet validation.",
            "Only nine fault episodes; effective sample size is groups, not daily rows.",
            "RUL coverage is evaluated inside known degradation windows; abstention reported separately.",
            "Policy thresholds and reference eligibility require operator validation.",
            "No conditional failure probabilities or guaranteed safe dates are produced."]}
    print("Chronological test:",report["chronological_rul"],flush=True)
    held_predictions=[]
    selections=[]
    for group in eps.train_id.unique():
        train_cycles=cycles[cycles.train_id!=group]
        held_cycles=cycles[cycles.train_id==group]
        pipe=HealthPipeline().fit(train_cycles[train_cycles.ts<ref_end])
        tr=pipe.transform(train_cycles)
        te=pipe.transform(held_cycles)
        tr_eps=eps[eps.train_id!=group]
        m,choice=select_rul(tr,tr_eps)
        hp=m.predict(te)
        hp["evaluation_group"]=group
        held_predictions.append(hp)
        selections.append({"group":group,"mode":m.mode,"projection":m.projection,"reason":choice["selection_reason"]})
        print("Held out",group,flush=True)
    grouped=pd.concat(held_predictions,ignore_index=True)
    grouped.to_parquet(ROOT/"data/group_holdout_predictions.parquet",index=False)
    report["group_holdout_rul"]=coverage_summary(grouped,eps)
    report["group_holdout_selection"]=selections
    report["group_holdout_scope"]="Entire preprocessing and RUL refitted without held-out train; separate from chronological test."
    (ROOT/"data/validation_report.json").write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding="utf-8")
    print("Validation report written.",flush=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
