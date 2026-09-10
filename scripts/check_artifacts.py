"""Check generated data/HTML consistency without browser access."""
import json,re,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from headway.deferral import compare_window,horizon_key

def main():
    d=pd.read_parquet(ROOT/"data/door_deferral.parquet")
    assert not any(c.startswith("risk_") or c=="safe_days_10pct" for c in d)
    unknown=~d.prediction_state.isin(["valid","threshold_exceeded","no_worsening_trend"])
    assert not d.loc[unknown,"aspect"].eq(0).any(), "unknown evidence displayed as green"
    assert d.loc[unknown,"rul_lower"].isna().all()
    assert d.available_at.eq(d.day+pd.Timedelta(days=1)).all()
    for h in (0.,3.,7.,14.,28.):
        expected=[compare_window(m,s,h) for m,s in zip(d.rul_lower,d.prediction_state)]
        assert d[horizon_key(h)].tolist()==expected
    assert set(d.duty.unique())<={"full_service","off_peak_only","withdraw","not_assessed"}
    assert not (d.duty.eq("withdraw")&d.prediction_state.ne("threshold_exceeded")).any(), "withdraw duty without threshold exceeded"
    assert not (d.duty.eq("off_peak_only")&~d.load_sensitive).any(), "restriction without measured load sensitivity"
    html=(ROOT/"ui/headway.html").read_text(encoding="utf-8")
    payload=json.loads(re.search(r'<script type="application/json" id="payload">(.*?)</script>',html,re.S).group(1))
    assert len(payload["assets"])==d.asset_id.nunique()
    assert 'data-theme="light"' in html
    assert "riskCeiling" not in html and "Latest date under 10% risk" not in html
    assert "Safe time left" in html and "simulated" in payload["scope"].lower()
    for asset in payload["assets"]:
        for row in asset["rows"]:
            assert set(row["windows"].values())<={"unknown","within_margin","exceeds_margin","no_positive_margin","threshold_exceeded"}
            assert row["duty"] in {"full_service","off_peak_only","withdraw","not_assessed"}
    node=shutil.which("node")
    if not node:
        raise RuntimeError("Node required for generated JavaScript syntax check")
    scripts=re.findall(r'<script>(.*?)</script>',html,re.S)
    assert scripts
    for script in scripts:
        subprocess.run([node,"--check"],input=script,text=True,check=True,capture_output=True)
    print(f"Artifact checks passed: {len(d)} rows, {len(payload['assets'])} assets, matching window states, valid JavaScript.")
    return 0

if __name__=="__main__": raise SystemExit(main())
