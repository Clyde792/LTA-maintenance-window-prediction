"""Labelled synthetic cases for the inspection-evidence module.

These are FIXTURES, not evidence. Each case injects a known effect into an
otherwise quiet daily frame so the module's behaviour can be inspected and
regression-tested. They are constructed to exercise the rules the module
implements, so agreement between the two says the code does what it says - it
says nothing whatever about diagnostic accuracy on real door telemetry.

The injected label lives on the `Case` object and NEVER in the frame handed to
the module: `Case.daily` carries identifiers, times, quality flags, the
condition-normalised channels and the context means, and nothing else.

`pipeline_case()` is the exception to "handcrafted": it runs the real
HealthPipeline over generated door cycles, so the module is exercised against
the quality and completeness columns the pipeline actually emits.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CHANNELS = ["cycle_duration_s", "peak_current_a", "mean_current_a",
            "current_integral_as", "travel_mm"]

# Per-channel day-to-day scale of the condition-normalised residual, in the
# channel's own units. Loosely ordered like the door generator's noise.
SIGMA = {"cycle_duration_s": .05, "peak_current_a": .08, "mean_current_a": .05,
         "current_integral_as": .10, "travel_mm": 1.2}

CONTEXT = {"ambient_temp_c_mean": 29.0, "load_proxy_mean": .45, "hour_of_day_mean": 13.0}

FITTED_AT = pd.Timestamp("2026-08-01")
MODEL_ID = "synthetic-door-normaliser-2026-08-01"


@dataclass(frozen=True)
class Case:
    """One labelled scenario. `injected` is the answer key, held outside `daily`."""
    name: str
    injected: str
    expectation: str
    asset_id: str
    as_of: pd.Timestamp
    daily: pd.DataFrame
    views: dict = field(default_factory=dict)
    peer_ids: object = None
    # Only for cases that deliberately lack a model identity: lets the example
    # runner show the algebra, explicitly labelled unverified.
    allow_unverified_demonstration: bool = False


def base_frame(*, n_assets=6, n_days=40, as_of="2026-09-11", seed=20260911):
    """A quiet fleet: every channel is noise around zero, every day supported."""
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(as_of)
    days = pd.date_range(end=as_of - pd.Timedelta(days=1), periods=n_days, freq="D")
    rows = []
    for i in range(n_assets):
        asset = f"TRN{100 + i // 2:03d}-DOOR-{i % 2 + 1}"
        frame = pd.DataFrame({"asset_id": asset, "day": days})
        for c in CHANNELS:
            frame[f"res_{c}"] = rng.normal(0., SIGMA[c], len(days))
        for c, v in CONTEXT.items():
            frame[c] = v + rng.normal(0., .2, len(days))
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    out["available_at"] = out.day + pd.Timedelta(days=1)
    out["data_quality_ok"] = True
    out["context_supported"] = True
    return out, as_of


def _recent(frame, as_of, days=5):
    return frame.available_at > as_of - pd.Timedelta(days=days)


def _shift(frame, asset_ids, as_of, moves, *, days=5):
    """Add `moves` (channel -> multiples of that channel's sigma) to recent days."""
    out = frame.copy()
    sel = out.asset_id.isin(asset_ids) & _recent(out, as_of, days)
    for c, k in moves.items():
        out.loc[sel, f"res_{c}"] += k * SIGMA[c]
    return out


def views(frame, asset_id, *, asset_scale_factor=1., asset_days=None, fleet_days=None,
          fitted_at=(FITTED_AT, FITTED_AT), model_id=MODEL_ID, negate_indices=False,
          supported=(None, None)):
    """Adapted and frozen views of the SAME residuals, as onboarding produces.

    Both views standardise the identical `res_*` columns; they differ only in the
    baseline applied. The adapted view centres on the asset's own median, so an
    offset present from the first reference day is absorbed and reads near zero.
    The frozen fleet view centres on zero and still shows it.
    """
    out = {}
    for (name, fitted), days_supported in zip(
            zip(("asset_reference", "fleet_reference"), fitted_at), supported):
        v = frame.copy()
        keep = asset_days if name == "asset_reference" else fleet_days
        if keep is not None:
            v = v[v.day.isin(keep)].copy()
        for c in CHANNELS:
            if name == "asset_reference":
                loc = float(v.loc[v.asset_id == asset_id, f"res_{c}"].median())
                scale = SIGMA[c] * asset_scale_factor
            else:
                loc, scale = 0., SIGMA[c]
            v[f"{c}_hx"] = (v[f"res_{c}"] - loc) / scale
            if negate_indices:
                # Both views negated: every downstream check still passes, and
                # the recovered scales come out negative. Must be rejected.
                v[f"{c}_hx"] = -v[f"{c}_hx"]
        # The frozen view is the fleet baseline for every asset, including the
        # onboarded one; only the adapted view claims an asset reference.
        if name == "asset_reference":
            v["onboarding_status"] = np.where(v.asset_id == asset_id, "asset_reference", "not_onboarded")
            v["baseline_source"] = np.where(v.asset_id == asset_id,
                                            "onboarded_asset_reference", "fleet_fallback")
        else:
            v["onboarding_status"] = "not_onboarded"
            v["baseline_source"] = "fleet_fallback"
        v["reference_verified"] = False
        v["reference_evidence_id"] = None
        v["preprocessing_fitted_at"] = fitted
        if days_supported is not None:
            v["context_supported"] = v.day.isin(list(days_supported))
        # The identity of the frozen condition-normalisation model. A caller
        # attestation, exactly like the onboarding provenance record: this code
        # does not authenticate it, it only refuses to proceed without it.
        if model_id is not None:
            v["normalisation_model_id"] = model_id
        out[name] = v
    return out


def cases(seed=20260911):
    """The eight labelled example cases, deterministic for a given seed."""
    frame, as_of = base_frame(seed=seed)
    target = sorted(frame.asset_id.unique())[0]
    peers = sorted(frame.asset_id.unique())[1:]
    out = []

    # 1. A single channel steps, exactly as robustness.perturb's sensor offset
    #    does. The charge integral is derived, so it cannot even be attributed
    #    to one measurement family.
    out.append(Case(
        name="isolated_sensor_offset", injected="sensor offset on current_integral_as only",
        expectation="one shifted channel, no independent corroboration, peers unaffected",
        asset_id=target, as_of=as_of,
        daily=_shift(frame, [target], as_of, {"current_integral_as": 6.})))

    # 2. Physically different measurements move together, in the wear direction:
    #    more current, longer movement, less travel.
    out.append(Case(
        name="coherent_multi_channel", injected="coherent mechanical change on the target",
        expectation="three independent families shift in the wear direction, peers unaffected",
        asset_id=target, as_of=as_of,
        daily=_shift(frame, [target], as_of, {"peak_current_a": 5., "mean_current_a": 5.,
                                              "cycle_duration_s": 4., "travel_mm": -4.})))

    # 3. The whole comparable population moves. Something common moved it; this
    #    case deliberately does not say what.
    out.append(Case(
        name="shared_peer_shift", injected="common influence across the target and its peers",
        expectation="peers share the change; sensor and local-mechanical are contradicted",
        asset_id=target, as_of=as_of,
        daily=_shift(frame, [target] + peers, as_of, {"peak_current_a": 4., "mean_current_a": 4.})))

    # 4. Nothing moves recently, but the target's residual carries a constant
    #    offset from day one. The adapted baseline centres on it; the frozen
    #    fleet baseline does not - the ROBUSTNESS_RESULTS.md contamination case.
    contaminated = frame.copy()
    contaminated.loc[contaminated.asset_id == target, "res_current_integral_as"] += \
        8. * SIGMA["current_integral_as"]
    out.append(Case(
        name="contaminated_reference", injected="offset present from the first reference day",
        expectation=("no recent change; the two baselines differ in LOCATION by the injected "
                     "offset in A.s, with the scale ratio at 1"),
        asset_id=target, as_of=as_of, daily=contaminated,
        views=views(contaminated, target)))

    # 5. Most channels unsupported, one sub-threshold move, and a peer set too
    #    thin to say anything. The honest answer is that nothing is separated.
    weak = _shift(frame, [target], as_of, {"peak_current_a": 2.2})
    unsupported = weak.asset_id.eq(target) & _recent(weak, as_of, 5)
    for c in ("cycle_duration_s", "mean_current_a", "current_integral_as", "travel_mm"):
        weak.loc[unsupported, f"res_{c}"] = np.nan
    thin = weak.asset_id.isin(peers[2:]) & _recent(weak, as_of, 5)
    weak.loc[thin, "context_supported"] = False
    out.append(Case(
        name="ambiguous_missing_evidence",
        injected="a sub-threshold move on peak_current_a; four channels unsupported",
        expectation=("no shift is claimed - the move lands below the shift threshold; the "
                     "unsupported channels and the thin peer set stay unknown and are carried "
                     "as missing evidence under cannot_distinguish"),
        asset_id=target, as_of=as_of, daily=weak, peer_ids=peers[:2]))

    # 6. Both at once: a step on the derived charge channel AND a genuine
    #    coherent mechanical move. Neither explanation may erase the other.
    out.append(Case(
        name="sensor_and_mechanical_overlap",
        injected="sensor offset on current_integral_as plus a coherent mechanical change",
        expectation="multi-family shift and an ambiguous derived channel are both reported",
        asset_id=target, as_of=as_of,
        daily=_shift(frame, [target], as_of, {"current_integral_as": 8., "cycle_duration_s": 4.,
                                              "peak_current_a": 4., "travel_mm": -4.})))

    # 7. A shared shift on top of a local one. The peer signal must not delete
    #    the target's own independent corroboration.
    shared_plus = _shift(frame, [target] + peers, as_of, {"peak_current_a": 4., "mean_current_a": 4.})
    out.append(Case(
        name="shared_shift_with_local_mechanical",
        injected="fleet-wide current shift plus a target-only timing and travel change",
        expectation="peer sharing and local independent families are both reported",
        asset_id=target, as_of=as_of,
        daily=_shift(shared_plus, [target], as_of, {"cycle_duration_s": 4., "travel_mm": -4.})))

    # 8. Quiet. A module that finds a story here is broken.
    out.append(Case(
        name="quiet_control", injected="nothing injected",
        expectation="no shifted channels, no explanation supported except cannot_distinguish",
        asset_id=target, as_of=as_of, daily=frame))
    return out


def reference_variants(seed=20260911):
    """Reference-view cases that must NOT be reported as contamination."""
    frame, as_of = base_frame(seed=seed)
    target = sorted(frame.asset_id.unique())[0]
    out = []

    # Same baseline location, different scale. A scale difference is not an
    # absorbed offset and must never be counted as contamination evidence.
    out.append(Case(
        name="scale_only_difference", injected="adapted scale three times the fleet scale",
        expectation="location offset ~0, scale ratio ~3, no contamination observation",
        asset_id=target, as_of=as_of, daily=frame,
        views=views(frame, target, asset_scale_factor=3.)))

    # The two views cover disjoint days, so there is nothing matched to compare.
    # The answer is "cannot be quantified", not zero.
    days = sorted(frame.day.unique())
    out.append(Case(
        name="views_cover_different_dates",
        injected="frozen view on the first ten days, adapted view on the last twenty-five",
        expectation="no matched asset-days; the adaptation effect is not quantifiable",
        asset_id=target, as_of=as_of, daily=frame,
        views=views(frame, target, fleet_days=days[:10], asset_days=days[-25:])))

    # Different condition-normalisation model behind each view.
    out.append(Case(
        name="incompatible_reference_provenance",
        injected="the two views were produced under different preprocessing",
        expectation="provenance mismatch refuses the comparison before any arithmetic",
        asset_id=target, as_of=as_of, daily=frame,
        views=views(frame, target, fitted_at=(FITTED_AT, FITTED_AT + pd.Timedelta(days=1)))))

    # No model identifier on either side. Absence on both sides is not agreement.
    contaminated = frame.copy()
    contaminated.loc[contaminated.asset_id == target, "res_current_integral_as"] +=         8. * SIGMA["current_integral_as"]
    out.append(Case(
        name="unidentified_reference_model",
        injected="a real offset, but neither view names its normalisation model",
        expectation=("the operational effect is not quantified; the algebra is recorded "
                     "only as an explicitly unverified demonstration"),
        asset_id=target, as_of=as_of, daily=contaminated,
        views=views(contaminated, target, model_id=None),
        allow_unverified_demonstration=True))

    # Both indices negated: the recovered scales are negative and must be refused.
    out.append(Case(
        name="negated_indices", injected="both views' indices multiplied by -1",
        expectation="non-positive recovered scale is rejected, not reinterpreted",
        asset_id=target, as_of=as_of, daily=frame,
        views=views(frame, target, negate_indices=True)))

    # Provenance and calendar coverage are fine, but no row in either view is
    # quality-supported, so there is nothing to apply a baseline to.
    out.append(Case(
        name="all_reference_rows_unsupported",
        injected="every reference row marked unsupported in both views",
        expectation="admission gate finds no usable observation; nothing is quantified",
        asset_id=target, as_of=as_of, daily=frame,
        views=views(frame, target, supported=([], []))))

    # The views cover the same calendar days but support disjoint ones, so the
    # coarse matched-day count passes and the real intersection is empty.
    days = sorted(frame.day.unique())
    out.append(Case(
        name="disjoint_supported_days",
        injected="alternate days supported in each view, with no day supported in both",
        expectation="matched calendar days are not matched SUPPORTED days; nothing is quantified",
        asset_id=target, as_of=as_of, daily=frame,
        views=views(frame, target, supported=(days[0::2], days[1::2]))))
    return out


def pipeline_case(*, seed=7, n_trains=6, n_days=70, stress="missing_channel"):
    """A case built through the REAL HealthPipeline, not a handcrafted frame.

    Imported lazily: generating cycles and fitting the pipeline costs seconds,
    and the handcrafted cases must stay fast.
    """
    from ..pipeline import HealthPipeline
    from ..robustness import perturb
    from .doors import SynthConfig, generate

    cfg = SynthConfig(n_trains=n_trains, doors_per_train=2, n_days=n_days,
                      cycles_per_day=40, n_episodes=2, seed=seed)
    cycles, _ = generate(cfg)
    cut = cycles.ts.min() + pd.Timedelta(days=30)
    pipe = HealthPipeline().fit(cycles[cycles.ts < cut])
    affected = sorted(cycles.asset_id.unique())[:2]
    stressed = perturb(cycles, stress, after=cut, assets=affected, seed=1) if stress else cycles
    daily = pipe.transform(stressed)
    # Fault labels come out of the pipeline; drop them so the answer key cannot
    # reach the module even by accident.
    daily = daily.drop(columns=[c for c in ("fault_confirmed", "fault_mode") if c in daily])
    as_of = pd.Timestamp(daily.available_at.max())
    return Case(
        name=f"real_pipeline_{stress or 'clean'}",
        injected=f"HealthPipeline output with the '{stress or 'clean'}' stress on {affected[0]}",
        expectation=("channel-level completeness keeps the four intact channels usable even "
                     "though the pipeline's global data_quality_ok is False on every "
                     "affected day"),
        asset_id=affected[0], as_of=as_of, daily=daily)
