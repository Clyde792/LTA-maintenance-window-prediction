"""Build a labelled retrospective demonstration; not an independent performance test."""
import sys,json
from pathlib import Path
import joblib,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.validation import select_rul, true_rul
ROOT=Path(__file__).resolve().parents[1]
def main():
    d=pd.read_parquet(ROOT/"data/door_daily.parquet")
    e=pd.read_csv(ROOT/"data/door_episodes.csv",parse_dates=["onset_ts","fault_ts"])
    model,report=select_rul(d,e)
    report["scope"]="RUL-stage inner validation; demo model refitted on all available labels"
    joblib.dump(model,ROOT/"data/rul_model.joblib")
    (ROOT/"data/rul_selection.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    model.predict(d).to_parquet(ROOT/"data/door_rul.parquet",index=False)
    print(model.report())
    print(report["selection_reason"])
    return 0
if __name__=="__main__": raise SystemExit(main())
