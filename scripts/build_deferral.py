"""Compare windows against the exact RUL margin used by each card."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.deferral import DeferralLedger
ROOT=Path(__file__).resolve().parents[1]
def main():
    d=pd.read_parquet(ROOT/"data/door_duty.parquet")
    out=DeferralLedger().ledger(d)
    out.to_parquet(ROOT/"data/door_deferral.parquet",index=False)
    print("Built window comparisons using existing card margins. No failure probabilities.")
    return 0
if __name__=="__main__": raise SystemExit(main())
