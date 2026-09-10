"""Assess fit-for-duty at peak load with the exact RUL model and policy the card uses."""
import sys,json
from pathlib import Path
import joblib,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.aspect import AspectPolicy
from headway.duty import DutyAssessor,duty_summary,format_bands
ROOT=Path(__file__).resolve().parents[1]
def main():
    d=pd.read_parquet(ROOT/"data/door_aspects.parquet")
    model=joblib.load(ROOT/"data/rul_model.joblib")
    bands=json.loads((ROOT/"data/peak_bands.json").read_text(encoding="utf-8"))
    out=DutyAssessor(model,AspectPolicy(),load_peak=bands["load_peak"],load_offpeak=bands["load_offpeak"],
        bands=[tuple(b) for b in bands["bands"]]).assess(d)
    out.to_parquet(ROOT/"data/door_duty.parquet",index=False)
    e=pd.read_csv(ROOT/"data/door_episodes.csv",parse_dates=["onset_ts","fault_ts"])
    summary=duty_summary(out,e)
    (ROOT/"data/duty_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(out.duty.value_counts().to_string())
    print(f"Peak bands {format_bands([tuple(b) for b in bands['bands']])}; peak load {bands['load_peak']:.2f}, off-peak {bands['load_offpeak']:.2f}.")
    print(json.dumps(summary))
    print("Duty labels are level comparisons at peak load; the wear-load interaction is a simulator assumption.")
    return 0
if __name__=="__main__": raise SystemExit(main())
