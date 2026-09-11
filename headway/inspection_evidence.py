"""Experimental inspection evidence: what the telemetry supports, not what is wrong.

This module answers one question for an anomalous door: *which observations
support which explanations, and what would tell them apart?* It never returns a
root cause, a probability, a likelihood or a confidence. Explanations are listed
in a FIXED order that is not a ranking, and the counts of supporting observations
are not a score - several statements can restate one underlying fact.

Three comparisons do the work:

  CROSS-CHANNEL   A shift in one measurement is one measurement. A shift across
                  physically different measurements is harder to explain by a
                  single faulty transducer. Current-derived features are NOT
                  independent of one another, and the charge integral is not
                  independent of the movement duration either, so neither can
                  corroborate the other. Independent channels moving together
                  fails to corroborate a sensor fault; it does not exclude one.

  PEER            Contemporaneous neighbouring doors under comparable conditions,
                  matched on overlapping dates. Fails closed: absent context
                  columns, thin context support, non-overlapping dates or too few
                  peers all yield UNKNOWN comparability, never a silent pass. A
                  channel is called "not shared" only when that channel has
                  enough comparable peer evidence to say so.

  REFERENCE       An onboarded asset is scored against its own adapted baseline.
                  The adapted and frozen indices are standardised by DIFFERENT
                  scales, so their difference is not a quantity. Instead this
                  module recovers each view's baseline location and scale from
                  `residual = location + scale * index` on matched asset-days,
                  reports the LOCATION difference in physical residual units and
                  the SCALE ratio separately, and refuses to quantify anything
                  when the parameters, matched days or provenance do not permit
                  it. A scale-only difference is never contamination evidence.

SAFEGUARDS, and they are the point of the module rather than a disclaimer on it:
nothing here suppresses an alert, lowers an aspect, authorises deferral, clears
an asset for service or touches the dashboard's detector or thresholds. Only data
available at the assessment time is read; anything missing or unsupported stays
`unknown` instead of being filled, inferred or quietly dropped.

Suggested inspection evidence is ILLUSTRATIVE and pending operator review. It
lists records and measurements that would discriminate between explanations. It
is not a maintenance procedure and must not be read as one.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import contract
from .features import MIN_COMPLETENESS, READINESS_FLAG
from .normalise import MAD_TO_SIGMA
from .repair_verification import fingerprint, timestamp

# --------------------------------------------------------------------------
# Physical grouping
# --------------------------------------------------------------------------
# Which physically distinct measurement each channel comes from. Three current
# features share one transducer and one signal path: they are one measurement
# reported three ways, and must never be counted as three agreeing witnesses.
FAMILY = {
    "peak_current_a": "motor_current",
    "mean_current_a": "motor_current",
    "current_integral_as": "motor_current",
    "cycle_duration_s": "timing",
    "travel_mm": "position",
}

# Channels computed FROM other families. Charge is the time integral of current,
# so it moves when the current moves, when the movement takes longer, or both.
# It is therefore consistent with either family and attributable to neither.
DERIVED = {"current_integral_as": ("motor_current", "timing")}

UNITS = {
    "cycle_duration_s": "s", "peak_current_a": "A", "mean_current_a": "A",
    "current_integral_as": "A.s", "travel_mm": "mm",
}

PEER_CONTEXT = ("ambient_temp_c_mean", "load_proxy_mean", "hour_of_day_mean")

# Absolute comparability tolerances, in each context's own units. These are a
# transparent screening policy, not calibrated equivalence limits.
CONTEXT_TOLERANCE = {"ambient_temp_c_mean": 3.0, "load_proxy_mean": .15, "hour_of_day_mean": 2.0}

# Onboarding states in which a row must not be used as evidence at all.
REFUSED_ONBOARDING = ("reference_rejected", "reference_not_yet_available")

# Columns that may carry the identity of the frozen condition-normalisation
# model. A fitting TIME is not an identity: two different models can be fitted
# at the same instant, and the same model can be re-fitted later.
MODEL_ID_COLUMNS = ("normalisation_model_id", "model_version")

# Explanations are ALWAYS emitted in this order. It is alphabetically arbitrary
# with respect to strength: nothing about the position of an entry, or the number
# of observations attached to it, expresses diagnostic priority.
EXPLANATION_ORDER = (
    "sensor_or_measurement_change_on_this_door",
    "mechanical_change_on_this_door",
    "common_influence_across_doors",
    "reference_contamination",
    "cannot_distinguish",
)

LIMITATION = (
    "Experimental evidence summary. It does not identify a cause, estimate a "
    "probability, or establish that any explanation is correct. It does not "
    "change alerts, aspects, urgency, deferral or service status."
)


def attributable_family(channel):
    """The single family a channel can be attributed to, or None if derived."""
    return None if channel in DERIVED else FAMILY.get(channel)


@dataclass(frozen=True)
class EvidencePolicy:
    """Window and screening parameters. Illustrative thresholds, not calibrated."""
    recent_days: int = 5
    reference_days: int = 21
    settling_days: int = 0
    min_recent_days: int = 3
    min_reference_days: int = 10
    shift_sigma: float = 2.0
    peer_share_sigma: float = 1.0
    min_peers: int = 3
    min_context_days: int = 3
    min_overlap_days: int = 3
    min_matched_days: int = 5
    # Imported from features, not restated: the completeness gate belongs to the
    # daily quality contract and must not drift from it. The channel-agnostic
    # terms are not restated at all - the pipeline publishes them as
    # features.READINESS_FLAG.
    min_channel_completeness: float = MIN_COMPLETENESS
    location_offset_sigma: float = 1.0
    affine_tolerance: float = 1e-6
    context_tolerance: dict = field(default_factory=lambda: dict(CONTEXT_TOLERANCE))

    def __post_init__(self):
        ints = ("recent_days", "reference_days", "settling_days", "min_recent_days",
                "min_reference_days", "min_peers", "min_context_days",
                "min_overlap_days", "min_matched_days")
        for name in ints:
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.recent_days < 1 or self.reference_days < 1 or self.min_peers < 1:
            raise ValueError("windows and the peer minimum must be positive")
        if self.min_recent_days > self.recent_days or self.min_reference_days > self.reference_days:
            raise ValueError("a window cannot require more days than it spans")
        if self.min_context_days > self.recent_days or self.min_overlap_days > self.recent_days:
            raise ValueError("peer screening cannot require more days than the recent window")
        for name in ("shift_sigma", "peer_share_sigma", "location_offset_sigma", "affine_tolerance"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.min_channel_completeness <= 1:
            raise ValueError("min_channel_completeness must lie in (0, 1]")
        if not self.context_tolerance:
            raise ValueError("peer comparability requires at least one context tolerance")
        for k, v in self.context_tolerance.items():
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"context tolerance for {k} must be finite and non-negative")


# --------------------------------------------------------------------------
# Quality: channel-level completeness without discarding global safeguards
# --------------------------------------------------------------------------

def _completeness_column(frame, channel, value_col):
    for c in (f"{value_col}_completeness", f"{channel}_completeness"):
        if c in frame.columns:
            return c
    return None


def quality_plan(frame, channels, value_prefix, policy):
    """Decide ONCE whether channel-level eligibility may be reconstructed at all.

    `data_quality_ok` folds the cross-channel completeness minimum together with
    every channel-agnostic readiness term, so one degraded channel voids the
    whole day. Measured on a 30% sample-loss replay: `data_quality_ok` was False
    on 100% of affected days while four of five channels sat at 1.00
    completeness.

    Completeness columns ALONE are not enough to override that. Without the
    pipeline's own readiness flag there is no way to know WHY the global flag is
    False, and assuming it was the completeness term would be a guess. So
    reconstruction is permitted only when the frame carries

        features.READINESS_FLAG   - the channel-agnostic half of the contract
        a completeness column     - for every declared channel

    and otherwise the global rejection stands, with the missing metadata named.
    """
    blocking = []
    if READINESS_FLAG not in frame.columns:
        blocking.append(f"{READINESS_FLAG} (the pipeline's channel-agnostic readiness flag)")
    columns = {}
    for c in channels:
        col = _completeness_column(frame, c, f"{value_prefix}{c}")
        if col is None:
            blocking.append(f"a completeness column for {c}")
        else:
            columns[c] = col
    every = sorted(c for c in frame.columns if c.endswith("_completeness"))
    if blocking:
        return {"basis": "global_data_quality_ok", "columns": {},
                "all_completeness_columns": every, "blocking_metadata": blocking,
                "reason": "a channel-specific assessment needs metadata this frame does not "
                          "carry, so the global rejection stands: " + "; ".join(blocking)}
    return {"basis": "channel_completeness", "columns": columns,
            "all_completeness_columns": every, "blocking_metadata": [], "reason": None}


def _channel_quality(frame, channel, policy, plan):
    """Rows usable FOR THIS CHANNEL, and an explicit account of what was checked."""
    checks = ["context_supported"]
    mask = frame.context_supported.eq(True)
    inconsistent = 0
    if plan["basis"] == "global_data_quality_ok":
        mask &= frame.data_quality_ok.eq(True)
        checks.append(f"data_quality_ok (global; {plan['reason']})")
    else:
        ready = frame[READINESS_FLAG]
        comp = pd.to_numeric(frame[plan["columns"][channel]], errors="coerce")
        every = frame[plan["all_completeness_columns"]].apply(pd.to_numeric, errors="coerce")
        # The contract features.to_daily documents. A frame that breaks it has
        # metadata we cannot interpret, so those rows are dropped rather than
        # resolved on an assumption about which term failed.
        expected = ready.eq(True) & (every.min(axis=1) >= policy.min_channel_completeness)
        consistent = every.notna().all(axis=1) & frame.data_quality_ok.eq(True).eq(expected)
        inconsistent = int((~consistent).sum())
        mask &= (ready.isin([True, False]) & ready.eq(True) & comp.between(0., 1.)
                 & (comp >= policy.min_channel_completeness) & consistent)
        checks += [f"{READINESS_FLAG} (pipeline-published channel-agnostic readiness)",
                   f"{plan['columns'][channel]} in [0,1] and >= {policy.min_channel_completeness}",
                   "data_quality_ok consistent with readiness and completeness"]
    # Reference safeguards apply on either basis: a rejected or not-yet-available
    # onboarding reference is not evidence, however complete the telemetry is.
    if "onboarding_status" in frame.columns:
        mask &= ~frame.onboarding_status.isin(REFUSED_ONBOARDING)
        checks.append("onboarding_status not rejected or pending")
    return mask.fillna(False), {"basis": plan["basis"], "checks_applied": checks,
                                "blocking_metadata": plan["blocking_metadata"],
                                "rows_rejected_for_inconsistent_quality_metadata": inconsistent}


def _supported(frame, col, channel, policy, plan):
    """Finite, quality-passing values of one channel."""
    if frame.empty:
        return pd.Series(dtype=float), {"basis": plan["basis"], "checks_applied": [],
                                        "blocking_metadata": plan["blocking_metadata"],
                                        "rows_rejected_for_inconsistent_quality_metadata": 0}
    mask, basis = _channel_quality(frame, channel, policy, plan)
    v = pd.to_numeric(frame[col], errors="coerce")
    return v[mask & np.isfinite(v)], basis


# --------------------------------------------------------------------------
# Windows and robust change
# --------------------------------------------------------------------------

def _required(daily, columns):
    missing = sorted(set(columns) - set(daily.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _visible(daily, as_of):
    """Only rows whose aggregate had actually landed by the assessment time.

    The replay clock is `available_at`, never the observation date: a day's
    aggregate is not evidence until the day has ended and it has been produced.
    """
    a = pd.to_datetime(daily.available_at.map(timestamp), utc=True)
    return daily[a <= as_of].assign(_available=a[a <= as_of])


def _windows(frame, as_of, policy):
    recent_start = as_of - pd.Timedelta(days=policy.recent_days)
    ref_end = recent_start - pd.Timedelta(days=policy.settling_days)
    ref_start = ref_end - pd.Timedelta(days=policy.reference_days)
    recent = frame[frame._available > recent_start]
    reference = frame[(frame._available > ref_start) & (frame._available <= ref_end)]
    return reference, recent, {"reference_start": ref_start.isoformat(),
                               "reference_end": ref_end.isoformat(),
                               "recent_start": recent_start.isoformat(),
                               "recent_end": as_of.isoformat()}


def _change(reference, recent, channel, value_col, policy, plan):
    """Robust change in a condition-normalised channel, in its own units and in
    reference sigma. Anything the data cannot support is returned as unknown."""
    r, basis = _supported(reference, value_col, channel, policy, plan)
    n, _ = _supported(recent, value_col, channel, policy, plan)
    out = {"channel": channel, "unit": UNITS.get(channel, "unknown"),
           "family": FAMILY.get(channel, "unknown"),
           "attributable_family": attributable_family(channel),
           "quality_basis": basis, "reference_days_used": int(len(r)),
           "recent_days_used": int(len(n)), "status": "unknown", "reason": None,
           "direction": "unknown", "reference_median": None, "recent_median": None,
           "change": None, "reference_scale": None, "change_in_reference_sigma": None}
    if len(n) < policy.min_recent_days:
        out["reason"] = "insufficient supported recent days"
        return out
    if len(r) < policy.min_reference_days:
        out["reason"] = "insufficient supported reference days"
        return out
    loc = float(r.median())
    scale = float(MAD_TO_SIGMA * (r - loc).abs().median())
    if not np.isfinite(scale) or scale <= 1e-9:
        out["reason"] = "degenerate reference scale; change is not interpretable"
        return out
    delta = float(n.median()) - loc
    z = delta / scale
    out.update(status="observed", reference_median=loc, recent_median=float(n.median()),
               change=delta, reference_scale=scale, change_in_reference_sigma=z,
               direction=("increase" if z >= policy.shift_sigma else
                          "decrease" if z <= -policy.shift_sigma else "no_detected_change"))
    return out


# --------------------------------------------------------------------------
# Cross-channel
# --------------------------------------------------------------------------

def cross_channel(changes, subsystem="door"):
    """Distinguish an isolated channel shift from agreement across families.

    A single shift alongside UNKNOWN channels is not an isolated shift: the
    unknown channels were not observed to stay put, they were not observed.
    """
    sub = contract.get(subsystem)
    shifted = sorted(c for c, v in changes.items() if v["direction"] in ("increase", "decrease"))
    unknown = sorted(c for c, v in changes.items() if v["status"] == "unknown")
    unchanged = sorted(c for c, v in changes.items() if v["direction"] == "no_detected_change")
    families = sorted({attributable_family(c) for c in shifted} - {None})
    ambiguous = sorted(c for c in shifted if attributable_family(c) is None)
    # Wear orientation from the contract: +1 means larger is worse, so a
    # degradation-consistent move has the same sign as the orientation.
    coherent = {c: bool(np.sign(changes[c]["change_in_reference_sigma"]) == np.sign(sub.sign(c)))
                for c in shifted}
    return {
        "shifted_channels": shifted,
        "unchanged_channels": unchanged,
        "unknown_channels": unknown,
        "independent_families": families,
        "n_independent_families": len(families),
        "ambiguous_channels": ambiguous,
        "single_shifted_channel": len(shifted) == 1,
        "isolated_channel_shift": len(shifted) == 1 and not unknown,
        "confined_to_one_family": bool(shifted) and len(families) <= 1,
        "wear_orientation_consistent": coherent,
        "all_shifts_wear_consistent": bool(shifted) and all(coherent.values()),
        "note": ("Current-derived channels share one transducer and signal path and are "
                 "counted as a single family. The charge integral depends on both current "
                 "and duration, so it corroborates neither. Unknown channels are not "
                 "channels that stayed put."),
    }


# --------------------------------------------------------------------------
# Peers
# --------------------------------------------------------------------------

def _context_medians(frame, cols, policy):
    """Median of each context column over SUPPORTED rows, or None if too thin."""
    out = {}
    supported = frame.context_supported.eq(True) if len(frame) else pd.Series(dtype=bool)
    for c in cols:
        v = pd.to_numeric(frame[c], errors="coerce")[supported] if len(frame) else pd.Series(dtype=float)
        v = v[np.isfinite(v)]
        out[c] = float(v.median()) if len(v) >= policy.min_context_days else None
    return out


def peer_comparison(daily, *, asset_id, as_of, changes, value_prefix, policy, plan,
                    peer_ids=None, context_cols=PEER_CONTEXT):
    """Do comparable neighbouring doors show the same change?

    Fails closed at every step. The target is excluded from its own peer
    baseline, always and by construction.
    """
    required = list(context_cols)
    absent = [c for c in required if c not in daily.columns]
    untoleranced = [c for c in required if c not in policy.context_tolerance]
    out = {"peer_scope": "all other assets in the frame" if peer_ids is None else "caller-supplied",
           "target_excluded_from_peers": True,
           "required_context_columns": required,
           "comparability_status": "unknown", "comparability_reason": None,
           "peers_considered": 0, "peers_comparable": [], "peers_excluded": {},
           "sufficient_peer_coverage": False, "channels": {},
           "shared_channels": [], "not_shared_channels": [], "unknown_channels": sorted(changes),
           "note": ("A change shared with comparable peers is evidence of a common "
                    "influence - duty, environment, timetable or a fleet-wide change. "
                    "It is not proof that the target's sensor failed. A channel is "
                    "reported as not shared only where that channel has enough "
                    "comparable peer evidence; otherwise it is unknown.")}

    def unresolved(reason):
        out["comparability_reason"] = reason
        out["channels"] = {c: {"status": "unknown", "reason": reason,
                               "peers_with_observed_change": 0,
                               "peer_median_change_in_sigma": None,
                               "shared_with_peers": None} for c in changes}
        return out

    if absent or untoleranced:
        return unresolved("required operating-context columns are unavailable: "
                          + ", ".join(sorted(set(absent) | set(untoleranced))))

    pool = daily[daily.asset_id != asset_id]
    if peer_ids is not None:
        pool = pool[pool.asset_id.isin([p for p in peer_ids if p != asset_id])]
    out["peers_considered"] = int(pool.asset_id.nunique())
    target = daily[daily.asset_id == asset_id]
    _, target_recent, _ = _windows(target, as_of, policy)
    target_ctx = _context_medians(target_recent, required, policy)
    thin = sorted(c for c, v in target_ctx.items() if v is None)
    if thin:
        return unresolved("the target has too few supported observations of: " + ", ".join(thin))
    target_days = set(pd.to_datetime(target_recent.day))

    per_channel = {c: [] for c in changes}
    for peer, g in pool.groupby("asset_id", sort=True):
        _, recent, _ = _windows(g, as_of, policy)
        overlap = len(target_days & set(pd.to_datetime(recent.day)))
        if overlap < policy.min_overlap_days:
            out["peers_excluded"][str(peer)] = (
                f"only {overlap} recent days overlap the target's window; "
                f"{policy.min_overlap_days} required")
            continue
        ctx, why = _context_medians(recent, required, policy), None
        for c in required:
            if ctx[c] is None:
                why = f"too few supported observations of {c}"
                break
            if abs(ctx[c] - target_ctx[c]) > policy.context_tolerance[c]:
                why = f"operating context {c} not comparable"
                break
        if why:
            out["peers_excluded"][str(peer)] = why
            continue
        reference, _, _ = _windows(g, as_of, policy)
        got = {c: _change(reference, recent, c, f"{value_prefix}{c}", policy, plan)
               for c in changes}
        if all(v["status"] == "unknown" for v in got.values()):
            out["peers_excluded"][str(peer)] = "no supported channel in either window"
            continue
        out["peers_comparable"].append(str(peer))
        for c, v in got.items():
            if v["status"] == "observed":
                per_channel[c].append(v["change_in_reference_sigma"])

    out["sufficient_peer_coverage"] = len(out["peers_comparable"]) >= policy.min_peers
    out["comparability_status"] = "resolved" if out["sufficient_peer_coverage"] else "unknown"
    if not out["sufficient_peer_coverage"]:
        out["comparability_reason"] = (
            f"only {len(out['peers_comparable'])} comparable peers; {policy.min_peers} required")

    for c in sorted(changes):
        zs, t = per_channel[c], changes[c]
        entry = {"peers_with_observed_change": len(zs), "peer_median_change_in_sigma": None,
                 "shared_with_peers": None, "status": "unknown", "reason": None}
        if not out["sufficient_peer_coverage"]:
            entry["reason"] = out["comparability_reason"]
        elif len(zs) < policy.min_peers:
            entry["reason"] = (f"only {len(zs)} comparable peers have a supported change in "
                               f"this channel; {policy.min_peers} required")
        elif t["status"] != "observed":
            entry["reason"] = "the target's own change in this channel is unknown"
            entry["peer_median_change_in_sigma"] = float(np.median(zs))
        else:
            m = float(np.median(zs))
            entry.update(status="observed", peer_median_change_in_sigma=m,
                         shared_with_peers=bool(abs(m) >= policy.peer_share_sigma
                                                and np.sign(m) == np.sign(t["change_in_reference_sigma"])))
            (out["shared_channels"] if entry["shared_with_peers"]
             else out["not_shared_channels"]).append(c)
        out["channels"][c] = entry
    out["unknown_channels"] = sorted(c for c, v in out["channels"].items() if v["status"] == "unknown")
    return out


# --------------------------------------------------------------------------
# Reference views
# --------------------------------------------------------------------------

def _validated(location, scale, channel, source, **extra):
    """A baseline is only usable with a finite location and a STRICTLY POSITIVE
    finite scale.

    A negative scale means the index falls as the residual rises. That is not an
    orientation convention this module may adopt on the caller's behalf -
    negating both views would otherwise flip every sign in the comparison while
    every other check still passed - so it is rejected.
    """
    if not np.isfinite(location):
        return {"status": "unavailable", "reason": f"{source} location is not finite"}
    if not np.isfinite(scale) or scale <= 0:
        return {"status": "unavailable",
                "reason": f"{source} scale is {float(scale):.6g}: a baseline scale must be finite "
                          "and strictly positive. A non-positive scale is rejected, not "
                          "reinterpreted as a different orientation convention."}
    return {"status": "recovered", "location": float(location), "scale": float(scale),
            "unit": UNITS.get(channel, "unknown"), "parameter_source": source, **extra}


def _exported_baseline(parameters, view, channel):
    """Baseline parameters supplied by the caller from the fitted pipeline."""
    entry = ((parameters or {}).get(view) or {}).get(channel)
    if entry is None:
        return None
    try:
        location, scale = float(entry["location"]), float(entry["scale"])
    except (KeyError, TypeError, ValueError):
        return {"status": "unavailable",
                "reason": "exported parameters must carry numeric 'location' and 'scale'"}
    return _validated(location, scale, channel, "exported")


def _admit(on, channel, value_col, index_col, policy, plan):
    """The ONE admission gate, applied before either parameter path.

    Returns the observations both views may be compared on: matched asset-days
    on which this channel is quality-supported and both the residual and the
    index are finite, in BOTH views. Exported parameters do not skip this -
    supplying numbers is not evidence that there is telemetry to apply them to.
    """
    parts = {}
    for name, frame in on.items():
        if value_col not in frame.columns or index_col not in frame.columns:
            return None, f"{name}: {value_col} or {index_col} is absent"
        mask, _ = _channel_quality(frame, channel, policy, plan)
        d = frame[mask]
        res = pd.to_numeric(d[value_col], errors="coerce").to_numpy(float)
        idx = pd.to_numeric(d[index_col], errors="coerce").to_numpy(float)
        ok = np.isfinite(res) & np.isfinite(idx)
        days = pd.DatetimeIndex(pd.to_datetime(d.day))[ok]
        if days.has_duplicates:
            return None, f"{name}: duplicate asset-days in the reference view"
        parts[name] = pd.DataFrame({"residual": res[ok], "index": idx[ok]}, index=days)
    days = parts["asset_reference"].index.intersection(parts["fleet_reference"].index)
    if len(days) < policy.min_matched_days:
        return None, (f"only {len(days)} matched, quality-supported asset-days carry this "
                      f"channel in both views; {policy.min_matched_days} required")
    admitted = {n: parts[n].loc[days].sort_index() for n in parts}
    gap = float(np.max(np.abs(admitted["asset_reference"].residual.to_numpy()
                              - admitted["fleet_reference"].residual.to_numpy())))
    if gap > policy.affine_tolerance:
        return None, ("the underlying residuals differ between views, so the two baselines "
                      "were not applied to the same telemetry")
    return admitted, None


def _infer_baseline(observations, channel):
    """Infer (location, scale) from `residual = location + scale * index`.

    INFERRED from the published index, not read from the fitted pipeline. Pass
    `reference_parameters` to use exported values instead; the integration test
    checks these against a real fitted pipeline's AssetBaseline.
    """
    x = observations["index"].to_numpy(float)
    y = observations["residual"].to_numpy(float)
    if np.ptp(x) <= 1e-12:
        return {"status": "unavailable", "reason": "the index does not vary; scale is unidentifiable"}
    scale, location = np.polyfit(x, y, 1)
    return _validated(location, scale, channel, "inferred_from_index")


def _consistent_with(got, observations, policy):
    """Both parameter paths must reproduce the admitted observations.

    Tolerance is absolute, in residual units, scaled by the largest magnitude in
    play so it means the same thing for millimetres and ampere-seconds:

        max |residual - (location + scale * index)|
            <= affine_tolerance * max(1, |scale|, max|residual|)

    with `affine_tolerance` defaulting to 1e-6. Exported parameters that do not
    satisfy this describe a different baseline from the one that produced the
    index, and are rejected rather than applied.
    """
    if got["status"] != "recovered":
        return got
    x = observations["index"].to_numpy(float)
    y = observations["residual"].to_numpy(float)
    error = float(np.max(np.abs(y - (got["location"] + got["scale"] * x))))
    limit = policy.affine_tolerance * max(1., abs(got["scale"]),
                                          float(np.max(np.abs(y))) if len(y) else 1.)
    if not np.isfinite(error) or error > limit:
        return {"status": "unavailable",
                "reason": f"{got['parameter_source']} parameters do not satisfy "
                          f"residual = location + scale x index on the {len(y)} admitted "
                          f"observations (max error {error:.3g}, tolerance {limit:.3g})"}
    return dict(got, days_used=int(len(y)), max_fit_error=error, fit_tolerance=limit,
                days=[t.isoformat() for t in observations.index])


def _view_identity(frame, name, declared):
    """Provenance for one view, validated across EVERY row used - not the last.

    A fitting time is not an identity. A caller declaration does not excuse the
    rows from validation either: every populated identity column must be
    constant, the columns must agree with each other, and all of them must agree
    with whatever the caller declared.
    """
    out = {"rows": int(len(frame)), "model_id": None, "model_id_source": None,
           "preprocessing_fitted_at": None, "consistent": True, "problems": []}
    found = {}
    for col in MODEL_ID_COLUMNS:
        if col not in frame.columns:
            continue
        values = sorted(frame[col].dropna().astype(str).unique())
        if not values:
            continue
        found[col] = values
        if len(values) > 1:
            out["problems"].append(f"{name}: {col} is not constant across the rows used "
                                   f"({len(values)} distinct values: {', '.join(values[:3])})")
    distinct = sorted({v for values in found.values() for v in values})
    if len(found) > 1 and len({tuple(v) for v in found.values()}) > 1:
        out["problems"].append(f"{name}: identity columns disagree "
                               + "; ".join(f"{c}={v}" for c, v in sorted(found.items())))
    if declared is not None:
        out["model_id"], out["model_id_source"] = str(declared), "caller"
        conflicting = [v for v in distinct if v != str(declared)]
        if conflicting:
            out["problems"].append(f"{name}: the caller declared {str(declared)!r} but the rows "
                                   f"carry {', '.join(repr(v) for v in conflicting)}")
    elif len(distinct) == 1:
        out["model_id"] = distinct[0]
        out["model_id_source"] = ", ".join(sorted(found))
    if "preprocessing_fitted_at" in frame.columns:
        stamps = pd.to_datetime(frame.preprocessing_fitted_at.dropna()).unique()
        if len(stamps) > 1:
            out["problems"].append(f"{name}: preprocessing_fitted_at is not constant across the "
                                   f"rows used ({len(stamps)} distinct values)")
        elif len(stamps) == 1:
            out["preprocessing_fitted_at"] = pd.Timestamp(stamps[0]).isoformat()
    if not (out["model_id"] or "").strip():
        out["problems"].append(f"{name}: no frozen condition-normalisation model identifier "
                               f"(looked for {', '.join(MODEL_ID_COLUMNS)}, or a caller-supplied "
                               "reference_model_versions entry)")
        out["model_id"] = None
    for col in ("onboarding_status", "baseline_source", "reference_verified",
                "reference_evidence_id"):
        if col in frame.columns and len(frame):
            values = frame[col].dropna().unique()
            v = values[0] if len(values) else None
            out[col] = (None if v is None else
                        bool(v) if isinstance(v, (bool, np.bool_)) else str(v))
            if len(values) > 1:
                out[col] = f"varies across rows: {len(values)} distinct values"
    out["consistent"] = not out["problems"]
    return out


def reference_comparison(views, *, asset_id, as_of, channels, policy, value_prefix, plan,
                         index_suffix="_hx", model_versions=None, parameters=None,
                         allow_unverified_demonstration=False):
    """Preserve both views and quantify only what provenance, telemetry and
    algebra permit.

    Never subtracts two independently standardised indices. Reports the baseline
    LOCATION difference in residual units and the SCALE ratio separately, and
    says plainly when neither can be established.
    """
    out = {"quantifiable": False, "reason": None, "views": {}, "provenance": {},
           "matched_days": {}, "channels": {}, "parameter_source": None,
           "verified_model_identity": False, "demonstration": None,
           "note": ("A stable adapted baseline is not proof of health. The adapted and "
                    "frozen indices are standardised by different scales, so their "
                    "difference is not a quantity; only a baseline location difference in "
                    "residual units, or on an explicitly named shared scale, is meaningful. "
                    "A scale-only difference is not contamination evidence.")}
    required = ("asset_reference", "fleet_reference")
    if not views:
        out["reason"] = "no reference views supplied; the adaptation effect is not visible"
        return out

    frames = {}
    for name, frame in views.items():
        f = _visible(frame[frame.asset_id == asset_id], as_of)
        frames[name] = f
        out["provenance"][name] = _view_identity(f, name, (model_versions or {}).get(name))
        out["views"][name] = {"rows_visible": int(len(f)), "index_suffix": index_suffix}

    missing = [n for n in required if n not in views]
    if missing:
        out["reason"] = ("both 'asset_reference' and 'fleet_reference' views are required to "
                         f"quantify the adaptation effect; missing: {', '.join(missing)}")
        return out

    a, f = out["provenance"]["asset_reference"], out["provenance"]["fleet_reference"]
    problems = list(a["problems"]) + list(f["problems"])
    if a["model_id"] and f["model_id"] and a["model_id"] != f["model_id"]:
        problems.append(f"the views name different condition-normalisation models "
                        f"({a['model_id']!r} vs {f['model_id']!r})")
    # Checked in ADDITION to identity, never instead of it.
    if a["preprocessing_fitted_at"] != f["preprocessing_fitted_at"]:
        problems.append(f"the views carry a different preprocessing_fitted_at "
                        f"({a['preprocessing_fitted_at']!r} vs {f['preprocessing_fitted_at']!r})")

    days = {n: set(pd.to_datetime(frames[n].day)) for n in required}
    matched = days["asset_reference"] & days["fleet_reference"]
    out["matched_days"] = {"n_matched": len(matched),
                           "asset_reference_days": len(days["asset_reference"]),
                           "fleet_reference_days": len(days["fleet_reference"])}
    if len(matched) < policy.min_matched_days:
        problems.append(f"the views share only {len(matched)} asset-days; "
                        f"{policy.min_matched_days} required to compare like with like")

    out["verified_model_identity"] = not problems
    if problems:
        out["reason"] = "; ".join(problems)
        if not (allow_unverified_demonstration and len(matched) >= policy.min_matched_days):
            return out

    on = {n: frames[n][pd.to_datetime(frames[n].day).isin(matched)] for n in required}
    computed, sources = {}, set()
    for c in channels:
        value_col, index_col = f"{value_prefix}{c}", f"{c}{index_suffix}"
        entry = {"status": "unavailable", "reason": None, "unit": UNITS.get(c, "unknown"),
                 "asset_reference": None, "fleet_reference": None,
                 "admitted_observations": 0,
                 "location_offset_removed_by_adaptation": None,
                 "location_offset_in_fleet_scale": None,
                 "location_offset_in_asset_scale": None,
                 "scale_ratio_asset_over_fleet": None, "scale_changed": None}
        admitted, why = _admit(on, c, value_col, index_col, policy, plan)
        if admitted is None:
            entry["reason"] = why
            # Baseline metadata may still be worth showing, but it is metadata:
            # no adaptation or contamination evidence comes out of it.
            supplied = {n: _exported_baseline(parameters, n, c) for n in required}
            if all(v is not None for v in supplied.values()):
                entry["status"] = "metadata_only"
                entry["asset_reference"], entry["fleet_reference"] = (
                    supplied["asset_reference"], supplied["fleet_reference"])
                entry["reason"] = (f"{why}. The supplied baseline parameters are shown as "
                                   "metadata only and are not evidence of an adaptation effect.")
            computed[c] = entry
            continue
        entry["admitted_observations"] = int(len(admitted["asset_reference"]))
        rec = {n: _consistent_with(_exported_baseline(parameters, n, c)
                                   or _infer_baseline(admitted[n], c), admitted[n], policy)
               for n in required}
        entry["asset_reference"], entry["fleet_reference"] = rec["asset_reference"], rec["fleet_reference"]
        if any(r["status"] != "recovered" for r in rec.values()):
            entry["reason"] = "; ".join(f"{n}: {r['reason']}" for n, r in rec.items()
                                        if r["status"] != "recovered")
            computed[c] = entry
            continue
        sources.update(r["parameter_source"] for r in rec.values())
        al, fl = rec["asset_reference"]["location"], rec["fleet_reference"]["location"]
        as_, fs = rec["asset_reference"]["scale"], rec["fleet_reference"]["scale"]
        entry.update(status="recovered",
                     location_offset_removed_by_adaptation=float(al - fl),
                     location_offset_in_fleet_scale=float((al - fl) / fs),
                     location_offset_in_asset_scale=float((al - fl) / as_),
                     scale_ratio_asset_over_fleet=float(as_ / fs),
                     scale_changed=bool(abs(as_ / fs - 1.) > .05))
        computed[c] = entry

    source = "+".join(sorted(sources)) if sources else None
    resolved = [c for c, v in computed.items() if v["status"] == "recovered"]
    if out["verified_model_identity"]:
        out["channels"] = computed
        if resolved:
            out.update(quantifiable=True, parameter_source=source)
        else:
            # Provenance was satisfied, but nothing could actually be quantified.
            out["reason"] = ("no channel could be quantified: "
                             + "; ".join(f"{c}: {v['reason']}" for c, v in sorted(computed.items())
                                         if v["reason"]))
    else:
        # Algebra only: the numbers follow from the published indices, but the
        # views have not been shown to share a frozen model, so this is not an
        # operational measurement of an adaptation effect.
        out["demonstration"] = {
            "status": "unverified_algebraic_demonstration", "verified": False,
            "parameter_source": source, "channels": computed,
            "warning": ("Computed from the supplied indices alone. The two views were not "
                        "shown to share a frozen condition-normalisation model, so these "
                        "numbers do not establish an operational adaptation effect and must "
                        "not be reported as one.")}
    return out


# --------------------------------------------------------------------------
# Observations and explanations
# --------------------------------------------------------------------------

SUGGESTIONS = {
    "sensor_or_measurement_change_on_this_door": [
        "Independent re-measurement of the affected channel on a few cycles",
        "Records of any door control unit, sensor, wiring or firmware change since the reference window",
        "Whether the same channel on the opposite leaf or paired door shows the shift",
    ],
    "mechanical_change_on_this_door": [
        "Recorded observations of movement quality on a sample of cycles",
        "Maintenance and parts history for this door since the reference window",
        "Whether the affected channels continue to move together over the next few days",
    ],
    "common_influence_across_doors": [
        "Whether the shift starts on the same date across the comparable peers",
        "Timetable, duty roster or service pattern changes covering the same dates",
        "Fleet-wide software, calibration or configuration change records",
    ],
    "reference_contamination": [
        "Inspection or maintenance evidence covering the reference window itself",
        "The frozen fleet baseline parameters for the same asset-days, alongside the adapted ones",
        "Any commissioning or post-overhaul record explaining a persistent offset",
    ],
    "cannot_distinguish": [
        "The evidence listed as missing for this assessment",
        "Continued observation until the unknown channels become supported",
    ],
}

BASE_STATEMENT = {
    "sensor_or_measurement_change_on_this_door":
        "A measurement or sensor change on this door would explain a shift confined to "
        "one measurement family.",
    "mechanical_change_on_this_door":
        "A mechanical change on this door would explain a coherent shift across "
        "physically independent measurements.",
    "common_influence_across_doors":
        "Something common to these doors would explain a change they share - duty, "
        "environment, timetable, or a fleet-wide calibration or configuration change. "
        "This does not identify which, and does not establish a sensor fault.",
    "reference_contamination":
        "Reference adaptation may have absorbed a baseline level difference that the "
        "frozen fleet baseline still carries.",
    "cannot_distinguish":
        "The available observations do not separate the explanations above.",
}


def _statements(cross, peers, reference):
    """Statement clauses are appended ONLY where the evidence for them exists."""
    s = {k: [v] for k, v in BASE_STATEMENT.items()}
    peer_known = peers["comparability_status"] == "resolved"
    shifted_shared = sorted(set(peers["shared_channels"]) & set(cross["shifted_channels"]))
    shifted_not_shared = sorted(set(peers["not_shared_channels"]) & set(cross["shifted_channels"]))

    if cross["shifted_channels"] and cross["confined_to_one_family"]:
        s["sensor_or_measurement_change_on_this_door"].append(
            "The observed shift is confined to one measurement family.")
    if cross["n_independent_families"] >= 2:
        s["sensor_or_measurement_change_on_this_door"].append(
            "Independent channels also moved. That does not corroborate a single-transducer "
            "fault, and it does not exclude a sensor problem occurring alongside another change.")
    if cross["unknown_channels"]:
        for k in ("sensor_or_measurement_change_on_this_door", "mechanical_change_on_this_door"):
            s[k].append(f"{len(cross['unknown_channels'])} channel(s) were not observed, so "
                        "they neither support nor contradict this.")
    if not peer_known:
        clause = f"Peer evidence is not available ({peers['comparability_reason']}), so a "\
                 "change common to other doors has not been excluded."
        for k in ("sensor_or_measurement_change_on_this_door", "mechanical_change_on_this_door",
                  "common_influence_across_doors"):
            s[k].append(clause)
    else:
        if shifted_not_shared:
            s["mechanical_change_on_this_door"].append(
                "Comparable peers did not show the change in: " + ", ".join(shifted_not_shared) + ".")
        if shifted_shared:
            s["common_influence_across_doors"].append(
                "Comparable peers show the same direction of change in: "
                + ", ".join(shifted_shared) + ".")
    if reference.get("quantifiable"):
        offsets = {c: v for c, v in reference["channels"].items()
                   if v["status"] == "recovered" and v["location_offset_removed_by_adaptation"] is not None}
        scaled = sorted(c for c, v in offsets.items() if v["scale_changed"])
        if scaled:
            s["reference_contamination"].append(
                "The two baselines also differ in scale for " + ", ".join(scaled)
                + "; a scale difference alone is not evidence of contamination.")
    else:
        s["reference_contamination"].append(
            "The adaptation effect cannot be quantified: " + str(reference.get("reason")) + ".")
        if reference.get("demonstration"):
            s["reference_contamination"].append(
                "An algebraic demonstration from the supplied indices is recorded and "
                "explicitly unverified; it is not evidence of an adaptation effect.")
    return {k: " ".join(v) for k, v in s.items()}


def _observations(changes, cross, peers, reference, policy):
    """Numbered, quotable observations. Explanations may only cite these."""
    obs, missing = [], []

    def add(kind, text):
        obs.append({"id": f"OBS{len(obs) + 1}", "kind": kind, "observation": text})
        return obs[-1]["id"]

    for c in sorted(changes):
        v = changes[c]
        if v["status"] == "observed" and v["direction"] != "no_detected_change":
            add("change", f"{c} {v['direction']} of {v['change']:+.4g} {v['unit']} "
                          f"({v['change_in_reference_sigma']:+.2f} reference sigma) against its "
                          f"own {v['reference_days_used']}-day reference")
        elif v["status"] == "unknown":
            missing.append(f"{c}: {v['reason']}")

    if cross["shifted_channels"]:
        if cross["confined_to_one_family"]:
            others = len(cross["unchanged_channels"])
            tail = (f"{others} other channel(s) were observed and did not move"
                    if others else "no other channel was observed to stay put")
            if cross["unknown_channels"]:
                tail += f"; {len(cross['unknown_channels'])} channel(s) are unknown"
            add("single_family_shift", "the shift is confined to one measurement family ("
                + ", ".join(cross["shifted_channels"]) + f"); {tail}")
        if cross["n_independent_families"] >= 2:
            add("multi_family", f"channels from {cross['n_independent_families']} physically "
                "distinct families shifted together: " + ", ".join(cross["independent_families"]))
            if cross["all_shifts_wear_consistent"]:
                add("wear_consistent", "every shift is in the direction the contract associates "
                                       "with wear for that channel")
            else:
                against = sorted(c for c, ok in cross["wear_orientation_consistent"].items() if not ok)
                add("wear_inconsistent", "shifts are not all in the wear direction; against: "
                    + ", ".join(against))
    if cross["ambiguous_channels"]:
        add("ambiguous", "shifted but not attributable to one measurement family: "
            + ", ".join(cross["ambiguous_channels"]))

    if peers["comparability_status"] != "resolved":
        missing.append(f"peer comparison: {peers['comparability_reason']}")
    else:
        shared = sorted(set(peers["shared_channels"]) & set(cross["shifted_channels"]))
        not_shared = sorted(set(peers["not_shared_channels"]) & set(cross["shifted_channels"]))
        for label, names in (("peer_shared", shared), ("peer_not_shared", not_shared)):
            for c in names:
                n = peers["channels"][c]["peers_with_observed_change"]
                verb = "show" if label == "peer_shared" else "do not show"
                add(label, f"{n} comparable peers with supported evidence in {c} {verb} the "
                           f"same direction of change")
        for c in cross["shifted_channels"]:
            if peers["channels"][c]["status"] != "observed":
                missing.append(f"peer comparison for {c}: {peers['channels'][c]['reason']}")

    if reference.get("quantifiable"):
        agreeing = []
        for c, v in sorted(reference["channels"].items()):
            if v["status"] != "recovered":
                missing.append(f"reference comparison for {c}: {v['reason']}")
            elif abs(v["location_offset_in_fleet_scale"]) >= policy.location_offset_sigma:
                add("location_offset", f"the adapted baseline for {c} sits "
                    f"{v['location_offset_removed_by_adaptation']:+.4g} {v['unit']} from the frozen "
                    f"fleet baseline ({v['location_offset_in_fleet_scale']:+.2f} in the fleet scale); "
                    f"scale ratio {v['scale_ratio_asset_over_fleet']:.3f}")
            else:
                agreeing.append(f"{c} ({v['location_offset_removed_by_adaptation']:+.3g} {v['unit']}, "
                                f"scale ratio {v['scale_ratio_asset_over_fleet']:.3f})")
        if agreeing:
            # A scale difference with no location difference is a different fact
            # from an absorbed offset, and is recorded as one.
            add("no_location_offset", "the adapted and frozen baselines agree in location within "
                f"{policy.location_offset_sigma:g} fleet sigma for: " + ", ".join(agreeing))
    else:
        missing.append(f"reference comparison: {reference.get('reason')}")
        for c, v in sorted((reference.get("channels") or {}).items()):
            if v.get("status") != "recovered" and v.get("reason"):
                missing.append(f"reference comparison for {c}: {v['reason']}")
        demo = reference.get("demonstration")
        if demo:
            # Recorded, and deliberately cited by nothing: algebra on the
            # supplied indices is not an operational adaptation effect.
            shown = [f"{c} {v['location_offset_removed_by_adaptation']:+.4g} {v['unit']}"
                     for c, v in sorted(demo["channels"].items()) if v["status"] == "recovered"]
            add("unverified_demonstration", "UNVERIFIED algebraic demonstration only, from the "
                "supplied indices, with no shared frozen model established: "
                + (", ".join(shown) if shown else "no channel resolved"))

    for m in missing:
        add("missing", f"not established: {m}")
    return obs, missing


def _explanations(cross, peers, reference, obs, statements):
    """Fixed order, never sorted by support. Counts are not a ranking, and a
    statement derived from another statement is not a second piece of evidence."""
    def ids(kind):
        return [o["id"] for o in obs if o["kind"] == kind]

    support = {
        "sensor_or_measurement_change_on_this_door": ids("single_family_shift") + ids("peer_not_shared"),
        "mechanical_change_on_this_door": ids("multi_family") + ids("wear_consistent") + ids("peer_not_shared"),
        "common_influence_across_doors": ids("peer_shared"),
        "reference_contamination": ids("location_offset"),
        "cannot_distinguish": ids("missing"),
    }
    # A multi-family shift is an ABSENCE of corroboration for a single-transducer
    # fault, not evidence against one: two problems can coexist. Only a change
    # shared across doors actually contradicts a fault in THIS door's sensor.
    against = {
        "sensor_or_measurement_change_on_this_door": ids("peer_shared"),
        "mechanical_change_on_this_door": ids("peer_shared") + ids("wear_inconsistent"),
        "common_influence_across_doors": ids("peer_not_shared"),
        "reference_contamination": ids("no_location_offset"),
        "cannot_distinguish": [],
    }
    return [{"explanation": name, "statement": statements[name],
             "supported_by": sorted(set(support[name])),
             "contradicted_by": sorted(set(against[name])),
             "suggested_inspection_evidence": list(SUGGESTIONS[name])}
            for name in EXPLANATION_ORDER]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def collect_evidence(daily, *, asset_id, as_of, subsystem="door", channels=None,
                     value_prefix="res_", policy=EvidencePolicy(), peer_ids=None,
                     reference_views=None, context_cols=PEER_CONTEXT, model_version=None,
                     reference_model_versions=None, reference_parameters=None,
                     allow_unverified_reference_demonstration=False):
    """Return a structured evidence record for one asset at one assessment time.

    Reads ONLY the columns it declares: identifiers, times, quality and
    completeness columns, the condition-normalised channels and the peer context
    means. Fault labels and injected scenario columns are never inputs, so a
    frame carrying them produces the same record as one without.
    """
    sub = contract.get(subsystem)
    channels = list(channels) if channels else [
        s for s in sub.signals if not s.endswith(("_flag", "_count"))]
    unknown = [c for c in channels if c not in FAMILY]
    if unknown:
        raise ValueError(f"no physical family declared for: {unknown}; add it to FAMILY "
                         "before these channels can be reasoned about")
    as_of = timestamp(as_of)
    value_cols = [f"{value_prefix}{c}" for c in channels]
    _required(daily, ["asset_id", "day", "available_at", "data_quality_ok",
                      "context_supported", *value_cols])
    if asset_id not in set(daily.asset_id):
        raise KeyError(f"{asset_id} is not present in the supplied frame")
    if daily.duplicated(["asset_id", "day"]).any():
        raise ValueError("duplicate asset-days in the supplied frame; peer statistics would "
                         "double-count them")

    visible = _visible(daily, as_of)
    target = visible[visible.asset_id == asset_id]
    reference_frame, recent, window = _windows(target, as_of, policy)
    plan = quality_plan(visible, channels, value_prefix, policy)

    changes = {c: _change(reference_frame, recent, c, f"{value_prefix}{c}", policy, plan)
               for c in channels}
    cross = cross_channel(changes, subsystem)
    peers = peer_comparison(visible, asset_id=asset_id, as_of=as_of, changes=changes,
                            value_prefix=value_prefix, policy=policy, plan=plan,
                            peer_ids=peer_ids, context_cols=context_cols)
    reference = reference_comparison(
        reference_views or {}, asset_id=asset_id, as_of=as_of, channels=channels, policy=policy,
        value_prefix=value_prefix, plan=plan, model_versions=reference_model_versions,
        parameters=reference_parameters,
        allow_unverified_demonstration=allow_unverified_reference_demonstration)
    obs, missing = _observations(changes, cross, peers, reference, policy)
    explanations = _explanations(cross, peers, reference, obs,
                                 _statements(cross, peers, reference))

    record = {
        "asset_id": asset_id, "subsystem": subsystem, "as_of": as_of.isoformat(),
        "model_version": model_version, "policy": asdict(policy), "window": window,
        "quality_basis": {k: v for k, v in plan.items() if k != "all_completeness_columns"},
        "observed_changes": changes,
        "cross_channel": cross,
        "peer_comparison": peers,
        "reference_comparison": reference,
        "observations": obs,
        "missing_evidence": missing,
        "explanations": explanations,
        "explanation_order": ("fixed and unranked; the number of supporting observations is "
                              "not a score, a probability or a diagnostic priority, and "
                              "several statements may restate one underlying fact"),
        "inspection_suggestions_status": "illustrative, pending operator review",
        "safeguards": {
            "identifies_root_cause": False,
            "suppresses_alerts": False,
            "reduces_urgency": False,
            "authorises_deferral": False,
            "release_to_service": False,
            "changes_detector_or_thresholds": False,
            "prescribes_maintenance_procedure": False,
        },
        "limitation": LIMITATION,
    }
    # Covers every input-derived field, so an identical record is provably
    # identical - the property the future-data and label-invariance tests assert.
    record["evidence_hash"] = fingerprint(record)
    return record
