"""Independent evaluation helpers. Label horizons use observation follow-up."""
import numpy as np
import pandas as pd
from .rul import ConformalRUL, prediction_time

def true_rul(daily,episodes):
    out=pd.Series(np.nan,index=daily.index,dtype=float)
    when=prediction_time(daily)
    for e in episodes.itertuples():
        m=(daily.asset_id==e.asset_id)&(when>=e.onset_ts)&(when<e.fault_ts)
        out.loc[m]=(e.fault_ts-when[m]).dt.total_seconds()/86400
    return out

def horizon_outcomes(daily,episodes,horizon_days,observation_end):
    """Censored follow-up stays NaN; no-event windows with complete follow-up are 0."""
    if horizon_days<0:
        raise ValueError("negative horizon")
    when=prediction_time(daily)
    deadline=when+pd.Timedelta(days=horizon_days)
    followup = daily.asset_id.map(observation_end) if isinstance(observation_end,dict) else pd.Series(pd.Timestamp(observation_end),index=daily.index)
    result=pd.Series(np.nan,index=daily.index,dtype=float)
    result[deadline<=followup]=0.
    for e in episodes.itertuples():
        hit=(daily.asset_id==e.asset_id)&(when<e.fault_ts)&(e.fault_ts<=deadline)
        result[hit]=1.
    return result

def coverage_summary(pred,episodes):
    truth=true_rul(pred,episodes)
    eligible=truth.notna()
    finite=eligible & np.isfinite(pred.rul_lower)
    groups=pred.train_id if "train_id" in pred else pred.asset_id
    cov=(pred.loc[finite,"rul_lower"]<=truth[finite]).groupby(groups[finite]).mean()
    close=finite & truth.le(7)
    return {
        "episode_rows":int(eligible.sum()),"scored_rows":int(finite.sum()),
        "abstention_rate":float(1-finite.sum()/eligible.sum()) if eligible.any() else None,
        "evaluated_groups":int(len(cov)),
        "macro_lower_coverage":float(cov.mean()) if len(cov) else None,
        "near_fault_lower_coverage":float((pred.rul_lower[close]<=truth[close]).mean()) if close.any() else None,
        "point_mae_days":float((pred.rul_point[finite]-truth[finite]).abs().mean()) if finite.any() else None,
        "median_lower_days":float(pred.rul_lower[finite].median()) if finite.any() else None,
        "positive_margin_fraction":float(pred.rul_lower[finite].gt(0).mean()) if finite.any() else None,
    }

def select_rul(daily,episodes):
    """Inner group validation only. Caller owns an untouched outer test set."""
    candidates=[]
    for projection,mode in [("linear","additive"),("loglinear","additive"),("loglinear","ratio"),("loglinear","mondrian")]:
        m=ConformalRUL(projection=projection,mode=mode)
        p=m.out_of_fold_predict(daily,episodes)
        stats=coverage_summary(p,episodes) if not p.empty else {}
        candidates.append({"projection":projection,"mode":mode,**stats})
    valid=[c for c in candidates if c.get("evaluated_groups",0)>=3
           and (c.get("macro_lower_coverage") or 0)>=.9
           and (c.get("near_fault_lower_coverage") or 0)>=.9
           and (c.get("median_lower_days") or 0)>0]
    selected=max(valid,key=lambda c:c["median_lower_days"] or 0) if valid else {"projection":"loglinear","mode":"additive"}
    reason="inner-group selection" if valid else "fixed baseline; insufficient qualifying inner validation"
    model=ConformalRUL(projection=selected["projection"],mode=selected["mode"]).fit(daily,episodes)
    return model, {"selection_reason":reason,"qualified":bool(valid),"selected":selected,"candidates":candidates}
