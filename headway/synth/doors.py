"""
Synthetic Door Control Unit telemetry.

Purpose: let us build and prove the whole Headway pipeline - normalisation,
tournament, RUL, Aspect Cards, Hindsight replay - BEFORE 18 Sep, so that the
real NEBULA X data only has to be plugged into an adapter on the night.

The generator is deliberately adversarial toward naive anomaly detection, and
that is the entire point. Three properties are engineered in:

  1. CONFOUNDS DOMINATE EARLY DEGRADATION.
     A hot afternoon and a crowded peak-hour run each move the primary signal by
     ~0.36 A.s. Three weeks before failure - while there is still time to plan -
     genuine wear has moved it by under 0.10 A.s, inside the per-cycle noise.
     End-of-life wear (0.60 A.s) is comparable to the operating envelope rather
     than far above it, so the top percentiles of the raw signal are a genuine
     mix of sick doors and healthy doors having a hot, crowded day.
     Threshold a raw signal and you get a false-alarm generator. Model
     E[signal | context] and score the residual, and the wear emerges cleanly.
     This is the "that's not a fault, that's a Tuesday" demo, in the data.

  2. PER-ASSET OFFSETS.
     Every door has its own build tolerance. A fleet-wide threshold flags the
     stiffest healthy door before it flags the worst degrading one, so
     normalisation has to be per-asset as well as per-condition.

  3. A BRUTAL BASE RATE.
     Tens of confirmed faults against hundreds of thousands of cycles. Any model
     reporting 99.99% accuracy is reporting that it predicted "healthy" every
     time. The harness refuses to show accuracy for exactly this reason.

  4. WEAR AND LOAD INTERACT.
     A worn mechanism is not just higher on average; it is disproportionately
     worse UNDER LOAD. A crowded peak-hour cycle costs a healthy door
     `load_coeff`; it costs a door at end of life `load_coeff + load_wear_coeff`.
     This is the operational claim behind the duty restriction ("fit for
     off-peak service, not for the peaks"): a door can look tolerable across a
     day's average and still be the one that fails at 08:15. It is a modelling
     ASSUMPTION drawn from operator experience, not a measured coefficient, and
     it must be checked against real telemetry before any duty claim is made.

Degradation follows a convex wear curve (slow onset, accelerating failure),
which is what gives Hindsight its story: the aspect escalates GREEN -> DOUBLE
AMBER -> AMBER over days before the recorded failure. A confirmed fault sends
the door to the depot, so health resets afterwards - the fleet recovers, and the
Logbook has a visible before/after to show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SERVICE_START_H = 5.5   # first passenger service
SERVICE_END_H = 24.0    # last service before the engineering window opens


@dataclass
class SynthConfig:
    """Knobs for the generator. Defaults produce ~0.5M cycles over 120 days."""

    n_trains: int = 40
    doors_per_train: int = 2
    n_days: int = 120
    cycles_per_day: int = 60          # per door, before dropout
    dropout: float = 0.15             # random missing cycles, as in real telemetry
    n_episodes: int = 9               # degradation episodes ending in confirmed fault
    start: str = "2026-04-01"
    seed: int = 20260918             # the hackathon date, because why not

    # Every degrading asset must have a clean healthy stretch before onset, or it
    # has no baseline to be compared against. Keep this >= AssetBaseline's
    # reference_days (21) with margin, otherwise the reference window silently
    # includes degradation and the asset baselines itself as already-sick.
    min_healthy_days: float = 25.0

    # Repair is not instant. After a confirmed fault the door goes to the depot
    # and its health decays back to baseline over a few days rather than snapping
    # to zero. Side effect, and the point of it: a door that failed in the last
    # days of the record is still visibly degraded "today" - the realistic fleet
    # snapshot, where something is always mid-life.
    repair_days: float = 2.5

    # --- effect sizes: the confound/degradation balance that makes the demo work
    temp_coeff: float = 0.055         # A.s per degree C above reference
    load_coeff: float = 0.42          # A.s at full crowding
    wear_coeff: float = 0.60          # A.s at end-of-life (h = 1)
    load_wear_coeff: float = 0.45     # EXTRA A.s at full crowding AND end-of-life
    asset_spread: float = 0.15        # per-asset build tolerance, 1 sigma
    noise: float = 0.09               # per-cycle measurement noise, 1 sigma

    ref_temp_c: float = 28.0


def _duty_curve(hours: np.ndarray) -> np.ndarray:
    """Relative service intensity by hour - the AM and PM peaks."""
    am = 0.90 * np.exp(-((hours - 8.0) ** 2) / (2 * 1.4**2))
    pm = 1.00 * np.exp(-((hours - 18.2) ** 2) / (2 * 1.8**2))
    return 0.35 + am + pm


def generate(cfg: SynthConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate contract-shaped door telemetry.

    Returns:
        cycles: the contract frame, ready for validate(df, "door").
        episodes: synthetic ground truth (onset/fault times per episode). This is
            a sidecar for building and checking the evaluation harness - it is NOT
            part of the contract and no model is ever allowed to see it.
    """
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)

    # ---------------------------------------------------------------- assets
    trains = [f"TRN{i:03d}" for i in range(1, cfg.n_trains + 1)]
    assets: list[tuple[str, str]] = [
        (t, f"{t}-DOOR-{d}") for t in trains for d in range(1, cfg.doors_per_train + 1)
    ]
    n_assets = len(assets)
    asset_offset = rng.normal(0.0, cfg.asset_spread, n_assets)   # build tolerance
    service_period = rng.integers(26, 36, n_assets)              # days between depot touches
    service_phase = rng.integers(0, 30, n_assets)

    # ------------------------------------------------------- degradation plan
    # Episodes are spread across the window so Hindsight always has a fault to
    # replay, and so that some are still mid-degradation at the end of the data
    # (the realistic case: the fleet is never uniformly healthy).
    ep_assets = rng.choice(n_assets, size=cfg.n_episodes, replace=False)
    fault_day = rng.uniform(0.45, 0.98, cfg.n_episodes) * cfg.n_days
    onset_lead_raw = rng.uniform(18, 45, cfg.n_episodes)
    # Clamp so onset never precedes the data, or the baseline reference window.
    # An episode that would have started too early simply gets a shorter ramp -
    # realistic, and it keeps `warning_days_available` an honest ceiling for the
    # capture metric rather than a number the detector never had access to.
    onset_day = np.maximum(fault_day - onset_lead_raw, cfg.min_healthy_days)
    onset_lead = fault_day - onset_day
    wear_shape = rng.uniform(1.8, 2.8, cfg.n_episodes)  # convexity of the wear curve
    modes = rng.choice(
        ["roller_wear", "gearbox_wear", "belt_tension_loss", "guide_contamination"],
        size=cfg.n_episodes,
    )

    # Pull the latest few episodes right up to the final days of the record, so
    # "today" always shows doors mid-degradation or freshly repaired - the
    # realistic fleet snapshot, and a non-empty first screen. Drawn last so the
    # rest of the degradation plan is identical to the base generator.
    n_recent = min(cfg.n_episodes, max(2, cfg.n_episodes // 4))
    late = np.argsort(fault_day)[-n_recent:] if n_recent else np.array([], dtype=int)
    fault_day[late] = cfg.n_days - rng.uniform(0.5, 4.5, n_recent)
    onset_day[late] = np.maximum(fault_day[late] - onset_lead_raw[late], cfg.min_healthy_days)
    onset_lead[late] = fault_day[late] - onset_day[late]

    onset_by_asset = np.full(n_assets, np.nan)
    fault_by_asset = np.full(n_assets, np.nan)
    shape_by_asset = np.full(n_assets, np.nan)
    mode_by_asset = np.array([None] * n_assets, dtype=object)
    for k, a in enumerate(ep_assets):
        onset_by_asset[a] = onset_day[k]
        fault_by_asset[a] = fault_day[k]
        shape_by_asset[a] = wear_shape[k]
        mode_by_asset[a] = modes[k]

    # ------------------------------------------------------------ time grid
    n_per_asset = cfg.n_days * cfg.cycles_per_day
    total = n_assets * n_per_asset

    asset_idx = np.repeat(np.arange(n_assets), n_per_asset)
    day_idx = np.tile(np.repeat(np.arange(cfg.n_days), cfg.cycles_per_day), n_assets)

    # Sample cycle times from the duty curve by inverse-CDF over a fine grid.
    grid = np.linspace(SERVICE_START_H, SERVICE_END_H, 512)
    w = _duty_curve(grid)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    hour = np.interp(rng.random(total), cdf, grid)

    # Weekends run a thinner service: drop more cycles and flatten the peaks.
    start_ts = pd.Timestamp(cfg.start)
    dow = (start_ts.dayofweek + day_idx) % 7
    is_weekend = dow >= 5

    keep = rng.random(total) > (cfg.dropout + 0.20 * is_weekend)
    asset_idx, day_idx, hour, is_weekend = (
        asset_idx[keep], day_idx[keep], hour[keep], is_weekend[keep]
    )
    n = len(hour)

    ts = start_ts + pd.to_timedelta(day_idx, unit="D") + pd.to_timedelta(hour, unit="h")
    t_days = day_idx + hour / 24.0   # continuous time, for the wear curve

    # ------------------------------------------------------------- context
    # Ambient: Singapore diurnal swing plus a slow multi-week drift. Peaks ~15:30.
    ambient = (
        27.0
        + 3.8 * np.sin(2 * np.pi * (hour - 9.5) / 24.0)
        + 1.2 * np.sin(2 * np.pi * day_idx / 60.0)
        + rng.normal(0, 0.6, n)
    )

    # Crowding tracks the duty curve, damped at weekends.
    duty = _duty_curve(hour)
    peakness = (duty - duty.min()) / (np.ptp(duty) + 1e-9)
    load = np.clip(
        (0.12 + 0.85 * peakness) * np.where(is_weekend, 0.62, 1.0) + rng.normal(0, 0.07, n),
        0.0, 1.0,
    )

    cycles_since_service = (
        (day_idx + service_phase[asset_idx]) % service_period[asset_idx]
    ) * cfg.cycles_per_day + rng.integers(0, cfg.cycles_per_day, n)

    # ---------------------------------------------------------- health state
    # h = 0 healthy, h = 1 at the confirmed fault. Convex: slow, then accelerating.
    onset = onset_by_asset[asset_idx]
    fault = fault_by_asset[asset_idx]
    shape = shape_by_asset[asset_idx]

    has_ep = ~np.isnan(onset)
    progress = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = (t_days - onset) / np.maximum(fault - onset, 1e-9)
    # Only inside the degradation window. Before onset the asset is healthy;
    # AFTER the confirmed fault it has been to the depot and is healthy again.
    in_window = has_ep & (t_days >= onset) & (t_days <= fault)
    progress[in_window] = np.clip(raw[in_window], 0.0, 1.0)
    h = np.where(in_window, progress ** np.where(has_ep, shape, 1.0), 0.0)

    # After the fault, decay from h=1 back to baseline over `repair_days` rather
    # than an instant reset (see SynthConfig.repair_days).
    post = has_ep & (t_days > fault)
    h = np.where(post, np.exp(-(t_days - fault) / max(cfg.repair_days, 1e-6)), h)

    # A slow benign random walk on every asset, so "healthy" is not a flat line.
    h = h + np.abs(rng.normal(0, 0.006, n)) * (t_days / cfg.n_days)

    # ------------------------------------------------------------- signals
    dtemp = ambient - cfg.ref_temp_c

    current_integral = (
        4.10
        + cfg.temp_coeff * dtemp
        + cfg.load_coeff * load
        + 0.08 * np.sin(2 * np.pi * hour / 24.0)
        + cfg.wear_coeff * h
        + cfg.load_wear_coeff * load * h      # property 4: worse under load
        + asset_offset[asset_idx]
        + rng.normal(0, cfg.noise, n)
    )

    duration = (
        3.15
        + 0.30 * load
        + 0.012 * dtemp
        + 0.26 * h
        + 0.20 * load * h
        + 0.35 * asset_offset[asset_idx]
        + rng.normal(0, 0.06, n)
    )

    peak_current = (
        5.80
        + 0.90 * load
        + 0.030 * dtemp
        + 0.72 * h
        + 0.55 * load * h
        + 0.60 * asset_offset[asset_idx]
        + rng.normal(0, 0.18, n)
    )

    mean_current = current_integral / np.maximum(duration, 0.5) + rng.normal(0, 0.05, n)

    # A worn mechanism under-travels very slightly before it fails outright.
    travel = 1300.0 - 5.5 * h + rng.normal(0, 2.0, n)

    # Obstruction is driven overwhelmingly by CROWDING, not by wear. Any model
    # that chases obstruction_flag will learn the peak-hour timetable instead of
    # the fault - a trap we want present in the data so the tournament exposes it.
    p_obs = np.clip(0.008 + 0.100 * load + 0.030 * h + 0.060 * load * h, 0, 1)
    obstruction = (rng.random(n) < p_obs).astype(np.int8)

    p_retry = np.clip(0.004 + 0.35 * obstruction * 0.1 + 0.045 * h, 0, 1)
    retry = rng.poisson(p_retry).astype(np.int16)

    # -------------------------------------------------------------- labels
    # The fault is confirmed on the FIRST cycle at or after the fault time, and
    # only then: maintenance records are single dated events, not spans.
    fault_confirmed = np.zeros(n, dtype=bool)
    fault_mode = np.array([None] * n, dtype=object)

    order = np.lexsort((t_days, asset_idx))
    for k, a in enumerate(ep_assets):
        pos = order[asset_idx[order] == a]
        hit = pos[t_days[pos] >= fault_day[k]]
        if len(hit):
            fault_confirmed[hit[0]] = True
            fault_mode[hit[0]] = modes[k]

    train_arr = np.array([t for t, _ in assets], dtype=object)
    asset_arr = np.array([a for _, a in assets], dtype=object)

    cycles = pd.DataFrame(
        {
            "ts": ts,
            "asset_id": asset_arr[asset_idx],
            "train_id": train_arr[asset_idx],
            "subsystem": "door",
            "cycle_duration_s": duration,
            "peak_current_a": peak_current,
            "mean_current_a": mean_current,
            "current_integral_as": current_integral,
            "travel_mm": travel,
            "obstruction_flag": obstruction,
            "retry_count": retry,
            "ambient_temp_c": ambient,
            "load_proxy": load,
            "hour_of_day": hour % 24.0,
            "cycles_since_service": cycles_since_service.astype(float),
            "fault_confirmed": fault_confirmed,
            "fault_mode": fault_mode,
        }
    ).sort_values(["asset_id", "ts"], ignore_index=True)

    # Detectable warning per episode, declared by the physics the generator
    # itself controls. At onset the wear term is exactly zero, so the opening
    # stretch of every window is below the noise floor of a one-day aggregate
    # NO MATTER how good the detector is. Measuring capture against the full
    # onset-to-fault window therefore flattered the apparent headroom by ~26%
    # (scripts/diagnose.py, section [1], derives the same floor from outside).
    # Floor: the day expected wear first clears 2 standard errors of the daily
    # aggregate. h(t) = progress^shape, detectable when wear_coeff*h > 2*se.
    week_keep = (5 * (1 - cfg.dropout) + 2 * (1 - cfg.dropout - 0.20)) / 7.0
    cycles_per_day_eff = cfg.cycles_per_day * week_keep
    se = cfg.noise / np.sqrt(max(cycles_per_day_eff, 1.0))
    prog_min = (2.0 * se / cfg.wear_coeff) ** (1.0 / wear_shape)
    detectable_days = onset_lead * (1.0 - prog_min)

    episodes = pd.DataFrame(
        {
            "asset_id": asset_arr[ep_assets],
            "train_id": train_arr[ep_assets],
            "fault_mode": modes,
            "onset_ts": start_ts + pd.to_timedelta(onset_day, unit="D"),
            "fault_ts": start_ts + pd.to_timedelta(fault_day, unit="D"),
            "warning_days_available": onset_lead,
            "detectable_days": detectable_days,
            "wear_shape": wear_shape,
        }
    ).sort_values("fault_ts", ignore_index=True)

    return cycles, episodes
