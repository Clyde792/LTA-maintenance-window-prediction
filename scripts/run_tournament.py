"""Compare frozen detectors with chronological alert replay.

The final-period table is a benchmark comparison, not evidence that a winner
selected from that table will generalise to another fleet.
"""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.pipeline import HealthPipeline
from headway.evaluate import metrics
from headway.models import detectors as D
ROOT=Path(__file__).resolve().parents[1]
def main():
    cycles=pd.read_parquet(ROOT/"data/door_cycles.parquet")
    eps=pd.read_csv(ROOT/"data/door_episodes.csv",parse_dates=["onset_ts","fault_ts"])
    start=cycles.ts.min().floor("D")
    end=cycles.ts.max().ceil("D")
    ref_end=start+pd.Timedelta(days=30)
    cutoff=(start+(end-start)*.7).floor("D")
    pipeline=HealthPipeline().fit(cycles[cycles.ts<ref_end])
    daily=pipeline.transform(cycles)
    norm=[f"{s}_hx" for s in pipeline.levels]
    zoo={"raw":D.RawThreshold("current_integral_as"),
         "ewma":D.EWMAChart("current_integral_as"),
         "isolation_forest":D.IsolationForestDetector(needs=norm),
         "pca":D.PCAReconstruction(needs=norm),
         "mahalanobis":D.MahalanobisDetector(needs=norm),
         "lof":D.LOFDetector(needs=norm),
         "headway_directional":D.Passthrough("health_index"),
         "headway_distance":D.Passthrough("anomaly_distance")}
    scored=D.run(zoo,daily,reference_days=30,smooth_days=3)
    calibration=scored[(scored.available_at>ref_end)&(scored.available_at<=cutoff)]
    test=scored[scored.available_at>cutoff]
    test_eps=eps[eps.fault_ts>cutoff]
    table=[]
    for key in zoo:
        threshold=float(calibration[key].quantile(.99))
        result=metrics.evaluate_replay(test,key,test_eps,threshold,
            daily_budget=2,cooldown_days=3,name=key).as_row()
        result["frozen_threshold"]=threshold
        table.append(result)
    out=pd.DataFrame(table)
    out.to_csv(ROOT/"data/tournament_replay.csv",index=False)
    scored.to_parquet(ROOT/"data/door_tournament.parquet",index=False)
    print("Synthetic future-period comparison; threshold fixed before",cutoff,flush=True)
    print(metrics.format_table(out),flush=True)
    print("No deployment winner selected using this test period.",flush=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
