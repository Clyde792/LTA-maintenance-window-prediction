"""Deterministic, label-free stress transformations for experimental replay."""
import numpy as np
import pandas as pd

SCENARIOS = ("clean", "missing_channel", "sensor_offset", "unsupported_context", "missing_days")


def perturb(cycles, scenario, *, after, assets, seed=0):
    """Change only selected assets at/after a fixed time; never inspect labels.

    Offset is a sensor perturbation, not an asserted physical wear mechanism.
    Context corruption is an out-of-support input test, not simulated hot weather.
    """
    if scenario not in SCENARIOS:
        raise ValueError("unknown stress scenario")
    c = cycles.copy(deep=True)
    selected = c.asset_id.isin(assets) & c.ts.ge(pd.Timestamp(after))
    if scenario == "missing_channel":
        rng = np.random.default_rng(seed)
        missing = selected & (rng.random(len(c)) < .3)
        c.loc[missing, "current_integral_as"] = np.nan
    elif scenario == "sensor_offset":
        c.loc[selected, "current_integral_as"] += .6
    elif scenario == "unsupported_context":
        c.loc[selected, "ambient_temp_c"] += 20.
    elif scenario == "missing_days":
        elapsed = (c.ts.dt.floor("D") - pd.Timestamp(after).floor("D")).dt.days
        c = c[~(selected & elapsed.mod(3).eq(0))].copy()
    return c


def score_frozen(detectors, daily, smooth_days=3):
    """Score already fitted models without fitting or filling missing channels."""
    out = daily.sort_values(["asset_id", "day"]).copy()
    if not out.index.is_unique or out.duplicated(["asset_id", "day"]).any():
        raise ValueError("unique daily rows required")
    for name, model in detectors.items():
        needs = model.needs if hasattr(model, "needs") else [model.column]
        supported = out.data_quality_ok.fillna(False) & out.context_supported.fillna(False)
        complete = np.isfinite(out[needs].to_numpy(float)).all(axis=1) & supported.to_numpy()
        raw = pd.Series(np.nan, index=out.index)
        if complete.any():
            raw.loc[complete] = model.score(out.loc[complete])
        out[name] = np.nan
        for _, g in out.groupby("asset_id", sort=False):
            s = pd.Series(raw.loc[g.index].to_numpy(), index=pd.DatetimeIndex(g.day))
            out.loc[g.index, name] = s.rolling(f"{smooth_days}D", min_periods=smooth_days).median().where(s.notna()).to_numpy()
        out.loc[~supported, name] = np.nan
    return out


def align_to_expected(scored, expected, score_names):
    """Keep absent asset-days in the denominator instead of making outages vanish."""
    keys = ["asset_id", "day", "available_at"]
    return expected[keys].merge(scored[keys + list(score_names)], on=keys, how="left", validate="one_to_one")
