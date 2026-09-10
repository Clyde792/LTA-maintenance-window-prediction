"""Build synthetic replay features. Independent evaluation: scripts/validate_pipeline.py."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.pipeline import HealthPipeline
ROOT=Path(__file__).resolve().parents[1]
def main():
    cycles=pd.read_parquet(ROOT/"data/door_cycles.parquet")
    cutoff=cycles.ts.min().floor("D")+pd.Timedelta(days=30)
    p=HealthPipeline().fit(cycles[cycles.ts<cutoff])
    d=p.transform(cycles)
    d.to_parquet(ROOT/"data/door_daily.parquet",index=False)
    print(f"Built {len(d)} asset-days; preprocessing available {p.fitted_at}.")
    print("Initial training days are marked unavailable for decisions.")
    return 0
if __name__=="__main__": raise SystemExit(main())
