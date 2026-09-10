"""Apply illustrative policy to retrospective demo estimates."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.aspect import AspectPolicy
ROOT=Path(__file__).resolve().parents[1]
def main():
    d=AspectPolicy().apply(pd.read_parquet(ROOT/"data/door_rul.parquet"))
    d.to_parquet(ROOT/"data/door_aspects.parquet",index=False)
    print(d.groupby(["prediction_state","recommendation"]).size().to_string())
    print("Demo policy thresholds; actions need operator review.")
    return 0
if __name__=="__main__": raise SystemExit(main())
