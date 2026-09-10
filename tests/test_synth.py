"""The synthetic data is the evidence base for every claim we will make on
stage. If the generator stops being adversarial, the pitch quietly becomes
false. These tests assert the properties the argument depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from headway.synth.doors import SynthConfig, generate

SIGNAL = "current_integral_as"


def test_deterministic_under_seed(small_cfg):
    a, _ = generate(small_cfg)
    b, _ = generate(small_cfg)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ(small_cfg):
    from dataclasses import replace
    a, _ = generate(small_cfg)
    b, _ = generate(replace(small_cfg, seed=small_cfg.seed + 1))
    assert not a[SIGNAL].equals(b[SIGNAL])


def test_base_rate_is_brutal(cycles):
    """Accuracy must be useless on this data, or the tournament proves nothing."""
    rate = cycles.fault_confirmed.mean()
    assert rate < 1e-3, f"base rate {rate:.2e} is too generous to be realistic"
    always_healthy_accuracy = 1 - rate
    assert always_healthy_accuracy > 0.999


def test_one_confirmed_fault_per_episode(cycles, episodes):
    assert cycles.fault_confirmed.sum() == len(episodes)
    for _, e in episodes.iterrows():
        n = cycles[(cycles.asset_id == e.asset_id) & cycles.fault_confirmed].shape[0]
        assert n == 1, f"{e.asset_id} should have exactly one confirmed fault"


def test_confounds_dominate_early_wear(cycles, episodes):
    """THE core property. If this inverts, condition normalisation is decoration.

    Three weeks before failure - while deferring is still a real choice - a hot
    or crowded day must move the signal MORE than the genuine degradation does.
    """
    healthy = cycles[~cycles.asset_id.isin(episodes.asset_id)]
    hot = healthy[healthy.ambient_temp_c > healthy.ambient_temp_c.quantile(0.95)][SIGNAL].mean()
    cool = healthy[healthy.ambient_temp_c < healthy.ambient_temp_c.quantile(0.05)][SIGNAL].mean()
    busy = healthy[healthy.load_proxy > 0.85][SIGNAL].mean()
    quiet = healthy[healthy.load_proxy < 0.15][SIGNAL].mean()
    confound = max(hot - cool, busy - quiet)

    # Measuring early wear is itself a confounded measurement, which is the
    # product's whole thesis showing up in its own test suite. Two traps:
    #
    #   - a fixed "21 days before the fault" samples from before onset once an
    #     episode's window is shorter than that, and averages in zeros;
    #   - a raw before/after comparison spans different calendar periods, so the
    #     generator's multi-week seasonal temperature drift lands in the answer.
    #
    # So: sample at a fixed fraction of each episode's OWN window, and take a
    # difference-in-differences against the healthy fleet over the same dates,
    # which cancels anything that moved the whole fleet.
    lifts = []
    for _, e in episodes.iterrows():
        a = cycles[cycles.asset_id == e.asset_id]
        span = e.fault_ts - e.onset_ts
        lo, hi = e.onset_ts + 0.25 * span, e.onset_ts + 0.45 * span

        def _delta(frame: pd.DataFrame) -> float:
            pre = frame[frame.ts < e.onset_ts][SIGNAL].mean()
            win = frame[(frame.ts >= lo) & (frame.ts < hi)][SIGNAL].mean()
            return win - pre

        asset_delta, fleet_delta = _delta(a), _delta(healthy)
        if np.isfinite(asset_delta) and np.isfinite(fleet_delta):
            lifts.append(asset_delta - fleet_delta)
    early_wear = float(np.mean(lifts))

    assert early_wear > 0, "wear must move the signal in the positive direction"
    assert confound > 2 * early_wear, (
        f"confound {confound:.3f} must dominate 21-day-out wear {early_wear:.3f}"
    )


def test_wear_is_monotone_approaching_failure(cycles, episodes):
    """Convex wear curve: closer to the fault must mean a larger lift."""
    means = {}
    for d in (21, 14, 7, 3):
        lifts = []
        for _, e in episodes.iterrows():
            a = cycles[cycles.asset_id == e.asset_id]
            base = a[a.ts < e.onset_ts][SIGNAL].mean()
            w = a[(a.ts >= e.fault_ts - pd.Timedelta(days=d))
                  & (a.ts < e.fault_ts - pd.Timedelta(days=d - 3))]
            if len(w):
                lifts.append(w[SIGNAL].mean() - base)
        means[d] = float(np.mean(lifts))
    ordered = [means[d] for d in (21, 14, 7, 3)]
    assert ordered == sorted(ordered), f"wear must accelerate toward failure: {means}"


def test_asset_is_repaired_after_confirmed_fault(cycles, episodes):
    """A confirmed fault sends the door to the depot; it returns healthy.

    Without this the degraded asset runs elevated forever, dominates the upper
    percentiles with physically impossible data, and poisons every evaluation.
    """
    checked = 0
    for _, e in episodes.iterrows():
        a = cycles[cycles.asset_id == e.asset_id]
        before = a[a.ts < e.onset_ts][SIGNAL].mean()
        just_before_fault = a[(a.ts >= e.fault_ts - pd.Timedelta(days=3))
                              & (a.ts <= e.fault_ts)][SIGNAL].mean()
        after = a[a.ts > e.fault_ts + pd.Timedelta(days=3)][SIGNAL]
        if len(after) < 50:
            continue          # episode ran to the end of the window
        checked += 1
        assert after.mean() < just_before_fault, "signal must fall after repair"
        assert abs(after.mean() - before) < 0.25, (
            "a repaired door must return to roughly its own pre-onset baseline"
        )
    assert checked > 0, "no episode had enough post-fault data to verify repair"


def test_per_asset_build_tolerance_exists(cycles, episodes):
    """A fleet-wide threshold must be a bad idea, or AssetBaseline is pointless."""
    healthy = cycles[~cycles.asset_id.isin(episodes.asset_id)]
    spread = healthy.groupby("asset_id")[SIGNAL].mean().std()
    assert spread > 0.05, f"per-asset spread {spread:.3f} is too small to matter"


def test_obstruction_is_driven_by_crowding_not_wear(cycles):
    """The trap we want present: a model chasing obstruction learns the timetable."""
    busy = cycles[cycles.load_proxy > 0.8].obstruction_flag.mean()
    quiet = cycles[cycles.load_proxy < 0.2].obstruction_flag.mean()
    assert busy > 3 * quiet, "crowding must dominate the obstruction rate"


def test_no_cycles_outside_service_hours(cycles):
    h = cycles.hour_of_day
    assert h.between(0, 24, inclusive="left").all()
    # Service runs 05:30 to ~00:30; nothing should sit in the small hours.
    assert ((h > 1.0) & (h < 5.0)).sum() == 0


def test_context_ranges_are_physical(cycles):
    assert cycles.load_proxy.between(0, 1).all()
    assert cycles.ambient_temp_c.between(18, 40).all(), "Singapore ambient, not Siberia"
    assert (cycles.cycles_since_service >= 0).all()
    assert (cycles.retry_count >= 0).all()
    assert cycles.obstruction_flag.isin([0, 1]).all()


def test_episodes_sidecar_matches_labels(cycles, episodes):
    labelled = set(cycles[cycles.fault_confirmed].asset_id)
    assert labelled == set(episodes.asset_id)
    assert (episodes.fault_ts > episodes.onset_ts).all()
    assert (episodes.warning_days_available > 0).all()
