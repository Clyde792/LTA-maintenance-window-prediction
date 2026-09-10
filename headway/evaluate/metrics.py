"""Retrospective ranking and causal alert replay.

Ranking spends a total budget using future score ranks and is NOT a live policy.
Replay uses a frozen threshold, daily capacity and elapsed-time cooldown.
Capture is fraction of onset-to-fault time, not a physical detection limit.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from ..rul import prediction_time

@dataclass
class DetectorResult:
    name: str
    budget: int
    alarms: int
    n_episodes: int
    detected: int
    median_lead_d: float
    worst_lead_d: float
    mean_lead_d: float
    precision: float
    median_capture: float = float("nan")
    worst_capture: float = float("nan")
    false_alerts_per_asset_month: float = float("nan")
    evaluation: str = "retrospective_ranking"

    @property
    def missed(self):
        return self.n_episodes-self.detected

    def as_row(self):
        return {**asdict(self),"missed":self.missed}

def _measure(daily, fired, episodes, budget, name, evaluation):
    useful = np.zeros(len(fired), bool)
    leads, captures = [], []
    when = prediction_time(fired)
    for e in episodes.itertuples():
        inside = ((fired.asset_id == e.asset_id)&(when >= e.onset_ts)&(when < e.fault_ts)).to_numpy()
        useful |= inside
        available = (e.fault_ts-e.onset_ts).total_seconds()/86400
        if inside.any():
            lead = (e.fault_ts-when[inside].min()).total_seconds()/86400
            leads.append(lead)
            captures.append(min(lead/available,1.) if available>0 else 0.)
        else:
            captures.append(0.)
    missed = len(episodes)-len(leads)
    worst = 0. if missed else min(leads) if leads else np.nan
    return DetectorResult(name,budget,len(fired),len(episodes),len(leads),
        float(np.median(leads)) if leads else np.nan, worst,
        float(np.mean(leads)) if leads else np.nan,
        float(useful.mean()) if len(fired) else 0.,
        float(np.median(captures)) if captures else np.nan,
        float(np.min(captures)) if captures else np.nan,
        float((~useful).sum()/max(len(daily)/30,1e-9)), evaluation)

def evaluate(daily, score, episodes, budget, *, name=None, day_col="day", asset_col="asset_id"):
    """Retrospective ranking diagnostic at a total asset-day budget."""
    if budget < 0:
        raise ValueError("budget cannot be negative")
    df = daily.rename(columns={day_col:"day",asset_col:"asset_id"}) if (day_col,asset_col)!=("day","asset_id") else daily
    valid = df[np.isfinite(pd.to_numeric(df[score],errors="coerce"))]
    fired = valid.sort_values([score,"day","asset_id"],ascending=[False,True,True]).head(budget)
    return _measure(df,fired,episodes,budget,name or score,"retrospective_ranking")

def replay_alerts(daily, score, threshold, *, daily_budget=2, cooldown_days=3.):
    """Availability-time order; only already available rows compete for capacity."""
    if not np.isfinite(threshold) or daily_budget<0 or cooldown_days<0:
        raise ValueError("invalid replay configuration")
    df=daily.copy()
    df["_available"]=prediction_time(df)
    if df.duplicated(["asset_id","_available"]).any():
        raise ValueError("duplicate asset prediction times")
    df=df.sort_values(["_available",score,"asset_id"],ascending=[True,False,True])
    last, spent, selected = {}, {}, []
    for idx,r in df.iterrows():
        if not np.isfinite(r[score]) or r[score]<threshold:
            continue
        stamp=r["_available"]
        date=stamp.floor("D")
        if spent.get(date,0)>=daily_budget:
            continue
        if r.asset_id in last and stamp-last[r.asset_id]<pd.Timedelta(days=cooldown_days):
            continue
        selected.append(idx)
        last[r.asset_id]=stamp
        spent[date]=spent.get(date,0)+1
    return daily.loc[selected].copy()

def evaluate_replay(daily, score, episodes, threshold, *, daily_budget=2, cooldown_days=3., name=None):
    fired=replay_alerts(daily,score,threshold,daily_budget=daily_budget,cooldown_days=cooldown_days)
    budget=daily_budget*prediction_time(daily).dt.floor("D").nunique()
    return _measure(daily,fired,episodes,budget,name or score,"chronological_replay")

def compare(daily,scores,episodes,budget,**kw):
    return pd.DataFrame([evaluate(daily,col,episodes,budget,name=label,**kw).as_row() for label,col in scores.items()])

def budget_curve(daily,score,episodes,budgets,**kw):
    return pd.DataFrame([evaluate(daily,score,episodes,b,**kw).as_row() for b in budgets])

def format_table(df):
    cols=[c for c in ["name","evaluation","detected","missed","median_lead_d","worst_lead_d","precision","worst_capture","false_alerts_per_asset_month"] if c in df]
    return df[cols].to_string(index=False,float_format=lambda v:f"{v:.3f}")
