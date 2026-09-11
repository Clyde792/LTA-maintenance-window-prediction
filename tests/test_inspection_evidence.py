"""Inspection-evidence module: invariances, fail-closed screening and abstention.

These fixtures are built to exercise the module's own rules, so passing tests
show the code behaves as documented. They are NOT evidence of diagnostic
accuracy on real telemetry, and nothing here should be quoted as such.
"""
import json

import numpy as np
import pandas as pd
import pytest

from headway.inspection_evidence import (
    EXPLANATION_ORDER, EvidencePolicy, attributable_family, collect_evidence)
from headway.features import MIN_COMPLETENESS, MIN_CYCLES, READINESS_FLAG
from headway.synth.inspection_cases import (
    CHANNELS, MODEL_ID, SIGMA, base_frame, cases, pipeline_case, reference_variants, views)


def run(case, **kw):
    return collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                            reference_views=case.views, peer_ids=case.peer_ids, **kw)


def by_name(name, source=cases):
    return next(c for c in source() if c.name == name)


def shift(frame, assets, as_of, moves, days=5):
    out = frame.copy()
    sel = out.asset_id.isin(assets) & (out.available_at > as_of - pd.Timedelta(days=days))
    for c, k in moves.items():
        out.loc[sel, f"res_{c}"] += k * SIGMA[c]
    return out


def with_completeness(frame, *, degraded=None, completeness=.5, ready=True,
                      with_readiness=True):
    """Add the per-channel completeness columns and the readiness flag.

    `data_quality_ok` is set the way `features.to_daily` sets it - the readiness
    flag AND the MINIMUM completeness across all channels - so the invariant the
    module relies on holds unless a test deliberately breaks it.
    """
    out = frame.copy()
    for c in CHANNELS:
        out[f"res_{c}_completeness"] = completeness if c == degraded else 1.
    cols = [f"res_{c}_completeness" for c in CHANNELS]
    out["n_cycles"] = 40
    if with_readiness:
        out[READINESS_FLAG] = ready
    out["data_quality_ok"] = (out[cols].min(axis=1) >= MIN_COMPLETENESS) & ready
    return out


@pytest.fixture
def fleet():
    return base_frame()


@pytest.fixture(scope="module")
def pipeline_stressed():
    return pipeline_case()


# --- invariances --------------------------------------------------------------

def test_future_observations_cannot_change_the_record(fleet):
    """Rows that had not landed by as_of are not evidence, however dramatic."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    before = collect_evidence(frame, asset_id=target, as_of=as_of)

    future = frame[frame.available_at == frame.available_at.max()].copy()
    future["day"] = future.day + pd.Timedelta(days=1)
    future["available_at"] = future.available_at + pd.Timedelta(days=1)
    for c in SIGMA:
        future[f"res_{c}"] += 40 * SIGMA[c]
    after = collect_evidence(pd.concat([frame, future], ignore_index=True),
                             asset_id=target, as_of=as_of)
    assert after["evidence_hash"] == before["evidence_hash"]


def test_an_aggregate_landing_exactly_at_the_assessment_instant_is_visible(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    assert frame.available_at.max() == as_of
    trimmed = frame[frame.available_at < as_of]
    assert (collect_evidence(frame, asset_id=target, as_of=as_of)["evidence_hash"]
            != collect_evidence(trimmed, asset_id=target, as_of=as_of)["evidence_hash"])


def test_scenario_and_fault_labels_are_not_inference_inputs(fleet):
    """The module reads only the columns it declares; labels must not reach it."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    before = collect_evidence(frame, asset_id=target, as_of=as_of)
    labelled = frame.assign(fault_confirmed=True, fault_mode="roller_wear",
                            injected_scenario="sensor_offset", scenario_label="answer key")
    assert collect_evidence(labelled, asset_id=target,
                            as_of=as_of)["evidence_hash"] == before["evidence_hash"]


def test_the_record_is_deterministic(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    a = collect_evidence(frame, asset_id=target, as_of=as_of)
    b = collect_evidence(frame.sample(frac=1., random_state=3), asset_id=target, as_of=as_of)
    assert a["evidence_hash"] == b["evidence_hash"]


# --- peer comparison fails closed ---------------------------------------------

def test_absent_context_columns_make_comparability_unknown_not_free(fleet):
    """Dropping the screening columns must not drop the screening criteria."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    stripped = frame.drop(columns=["ambient_temp_c_mean"])
    peers = collect_evidence(stripped, asset_id=target, as_of=as_of)["peer_comparison"]
    assert peers["comparability_status"] == "unknown"
    assert "ambient_temp_c_mean" in peers["comparability_reason"]
    assert peers["peers_comparable"] == []
    assert all(v["status"] == "unknown" for v in peers["channels"].values())


def test_a_context_column_without_a_tolerance_is_not_silently_ignored(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    policy = EvidencePolicy(context_tolerance={"ambient_temp_c_mean": 3.})
    peers = collect_evidence(frame, asset_id=target, as_of=as_of,
                             policy=policy)["peer_comparison"]
    assert peers["comparability_status"] == "unknown"
    assert "load_proxy_mean" in peers["comparability_reason"]


def test_thin_context_support_makes_comparability_unknown(fleet):
    """A peer needs enough SUPPORTED context observations, not merely a column."""
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    thin = frame.copy()
    recent = thin.available_at > as_of - pd.Timedelta(days=5)
    thin.loc[thin.asset_id.isin(assets[1:]) & recent, "context_supported"] = False
    peers = collect_evidence(thin, asset_id=assets[0], as_of=as_of)["peer_comparison"]
    assert peers["sufficient_peer_coverage"] is False
    assert all("supported observations" in why for why in peers["peers_excluded"].values())


def test_the_target_itself_needs_supported_context(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    thin = frame.copy()
    recent = thin.available_at > as_of - pd.Timedelta(days=5)
    thin.loc[thin.asset_id.eq(target) & recent, "context_supported"] = False
    peers = collect_evidence(thin, asset_id=target, as_of=as_of)["peer_comparison"]
    assert peers["comparability_status"] == "unknown"
    assert "the target has too few supported observations" in peers["comparability_reason"]


def test_peers_whose_dates_do_not_overlap_are_excluded(fleet):
    """Contemporaneous means contemporaneous; a peer on other days is not one."""
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    target, offset = assets[0], assets[1]
    moved = frame.copy()
    sel = moved.asset_id == offset
    moved.loc[sel, "day"] = moved.loc[sel, "day"] - pd.Timedelta(days=60)
    moved.loc[sel, "available_at"] = moved.loc[sel, "available_at"] - pd.Timedelta(days=60)
    peers = collect_evidence(moved, asset_id=target, as_of=as_of)["peer_comparison"]
    assert "overlap" in peers["peers_excluded"][offset]


def test_duplicate_asset_days_are_refused(fleet):
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    dup = pd.concat([frame, frame[frame.asset_id == assets[1]].tail(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate asset-days"):
        collect_evidence(dup, asset_id=assets[0], as_of=as_of)


def test_a_channel_is_not_shared_only_where_that_channel_has_peer_evidence(fleet):
    """Unknown is not 'no change'. One channel's thin evidence stays unknown."""
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    target = assets[0]
    blind = shift(frame, [target], as_of, {"peak_current_a": 6.})
    # Every peer loses this one channel; the others remain fully comparable.
    blind.loc[blind.asset_id != target, "res_peak_current_a"] = np.nan
    peers = collect_evidence(blind, asset_id=target, as_of=as_of)["peer_comparison"]
    entry = peers["channels"]["peak_current_a"]
    assert entry["status"] == "unknown"
    assert entry["shared_with_peers"] is None
    assert "peak_current_a" not in peers["not_shared_channels"]
    assert peers["channels"]["travel_mm"]["status"] == "observed"


def test_peer_counts_are_channel_specific_not_the_admitted_pool(fleet):
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    target = assets[0]
    partial = shift(frame, [target], as_of, {"peak_current_a": 6.})
    # Two of five peers lose the shifted channel: enough remain to resolve it,
    # but the reported count must be three, not the five admitted peers.
    partial.loc[partial.asset_id.isin(assets[1:3]), "res_peak_current_a"] = np.nan
    r = collect_evidence(partial, asset_id=target, as_of=as_of)
    peers = r["peer_comparison"]
    assert len(peers["peers_comparable"]) == 5
    assert peers["channels"]["peak_current_a"]["peers_with_observed_change"] == 3
    text = next(o["observation"] for o in r["observations"] if o["kind"] == "peer_not_shared")
    assert text.startswith("3 comparable peers")


def test_the_target_is_never_its_own_peer(fleet):
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    r = collect_evidence(frame, asset_id=assets[0], as_of=as_of, peer_ids=assets)
    assert assets[0] not in r["peer_comparison"]["peers_comparable"]
    assert assets[0] not in r["peer_comparison"]["peers_excluded"]
    assert r["peer_comparison"]["target_excluded_from_peers"] is True


def test_the_targets_own_change_does_not_move_the_peer_baseline(fleet):
    """A large target-only shift must leave every peer statistic untouched."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    quiet = collect_evidence(frame, asset_id=target, as_of=as_of)
    loud = collect_evidence(shift(frame, [target], as_of, {"peak_current_a": 25.}),
                            asset_id=target, as_of=as_of)
    for c, entry in loud["peer_comparison"]["channels"].items():
        assert (entry["peer_median_change_in_sigma"]
                == quiet["peer_comparison"]["channels"][c]["peer_median_change_in_sigma"])


def test_peers_in_different_operating_conditions_are_excluded(fleet):
    frame, as_of = fleet
    assets = sorted(frame.asset_id.unique())
    hot = frame.copy()
    hot.loc[hot.asset_id == assets[1], "ambient_temp_c_mean"] += 12.
    peers = collect_evidence(hot, asset_id=assets[0], as_of=as_of)["peer_comparison"]
    assert "ambient_temp_c_mean" in peers["peers_excluded"][assets[1]]


# --- cross-channel independence -----------------------------------------------

def test_current_features_are_one_family_not_three(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    cross = collect_evidence(shift(frame, [target], as_of,
                                   {"peak_current_a": 6., "mean_current_a": 6.}),
                             asset_id=target, as_of=as_of)["cross_channel"]
    assert cross["shifted_channels"] == ["mean_current_a", "peak_current_a"]
    assert cross["independent_families"] == ["motor_current"]
    assert cross["confined_to_one_family"] is True


def test_the_charge_integral_corroborates_nothing_on_its_own():
    cross = run(by_name("isolated_sensor_offset"))["cross_channel"]
    assert cross["shifted_channels"] == ["current_integral_as"]
    assert cross["ambiguous_channels"] == ["current_integral_as"]
    assert cross["independent_families"] == []
    assert attributable_family("current_integral_as") is None


def test_physically_distinct_channels_count_separately():
    cross = run(by_name("coherent_multi_channel"))["cross_channel"]
    assert cross["n_independent_families"] == 3
    assert cross["all_shifts_wear_consistent"] is True


def test_a_shift_against_the_wear_direction_is_flagged(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    cross = collect_evidence(shift(frame, [target], as_of,
                                   {"peak_current_a": 6., "travel_mm": 6.}),
                             asset_id=target, as_of=as_of)["cross_channel"]
    assert cross["n_independent_families"] == 2
    assert cross["all_shifts_wear_consistent"] is False
    assert cross["wear_orientation_consistent"]["travel_mm"] is False


def test_unknown_channels_are_not_channels_that_did_not_move(fleet):
    """A lone shift beside unobserved channels is not an ISOLATED shift."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    partial = shift(frame, [target], as_of, {"peak_current_a": 6.})
    partial.loc[partial.asset_id == target, "res_travel_mm"] = np.nan
    r = collect_evidence(partial, asset_id=target, as_of=as_of)
    cross = r["cross_channel"]
    assert cross["single_shifted_channel"] is True
    assert cross["isolated_channel_shift"] is False
    assert "travel_mm" in cross["unknown_channels"]
    assert "travel_mm" not in cross["unchanged_channels"]
    text = next(o["observation"] for o in r["observations"] if o["kind"] == "single_family_shift")
    assert "1 channel(s) are unknown" in text


# --- reference mathematics ----------------------------------------------------

def test_a_location_offset_is_reported_in_physical_units():
    """The quantity is a baseline location difference, not a difference of indices."""
    ref = run(by_name("contaminated_reference"))["reference_comparison"]
    entry = ref["channels"]["current_integral_as"]
    assert entry["status"] == "recovered"
    injected = 8. * SIGMA["current_integral_as"]
    assert entry["location_offset_removed_by_adaptation"] == pytest.approx(injected, rel=.1)
    assert entry["unit"] == "A.s"
    assert entry["scale_ratio_asset_over_fleet"] == pytest.approx(1., abs=1e-6)
    assert entry["location_offset_in_fleet_scale"] == pytest.approx(8., rel=.1)
    assert "adaptation_removed" not in ref


def test_both_baselines_are_preserved_with_their_own_parameters():
    ref = run(by_name("contaminated_reference"))["reference_comparison"]
    entry = ref["channels"]["current_integral_as"]
    for view in ("asset_reference", "fleet_reference"):
        assert entry[view]["status"] == "recovered"
        assert set(entry[view]) >= {"location", "scale", "unit", "days_used"}
    assert ref["provenance"]["asset_reference"]["baseline_source"] == "onboarded_asset_reference"
    assert ref["provenance"]["fleet_reference"]["baseline_source"] == "fleet_fallback"


def test_a_scale_difference_alone_is_not_contamination():
    r = run(by_name("scale_only_difference", reference_variants))
    entry = r["reference_comparison"]["channels"]["current_integral_as"]
    assert entry["scale_ratio_asset_over_fleet"] == pytest.approx(3., rel=1e-3)
    assert entry["scale_changed"] is True
    assert abs(entry["location_offset_in_fleet_scale"]) < 1.
    assert not [o for o in r["observations"] if o["kind"] == "location_offset"]
    named = {e["explanation"]: e for e in r["explanations"]}
    assert named["reference_contamination"]["supported_by"] == []
    assert "scale difference alone is not evidence" in named["reference_contamination"]["statement"]


def test_views_over_different_dates_cannot_be_quantified():
    r = run(by_name("views_cover_different_dates", reference_variants))
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "share only 0 asset-days" in ref["reason"]
    assert ref["channels"] == {}
    assert any("reference comparison" in m for m in r["missing_evidence"])


def test_incompatible_provenance_refuses_the_comparison():
    r = run(by_name("incompatible_reference_provenance", reference_variants))
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "different preprocessing_fitted_at" in ref["reason"]
    assert not [o for o in r["observations"] if o["kind"] == "location_offset"]


def test_a_declared_model_version_mismatch_refuses_the_comparison():
    case = by_name("contaminated_reference")
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=case.views,
                         reference_model_versions={"asset_reference": "v1",
                                                   "fleet_reference": "v2"})
    assert r["reference_comparison"]["quantifiable"] is False
    assert "name different condition-normalisation models" in r["reference_comparison"]["reason"]


def test_different_underlying_telemetry_is_refused():
    """Two baselines applied to different residuals are not two views of one thing."""
    case = by_name("contaminated_reference")
    tampered = {k: v.copy() for k, v in case.views.items()}
    tampered["fleet_reference"]["res_current_integral_as"] += .5
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=tampered)
    entry = r["reference_comparison"]["channels"]["current_integral_as"]
    assert entry["status"] == "unavailable"
    assert "not applied to the same telemetry" in entry["reason"]


def test_one_view_alone_cannot_quantify_the_adaptation_effect():
    case = by_name("contaminated_reference")
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views={"asset_reference": case.views["asset_reference"]})
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "fleet_reference" in ref["reason"]
    assert ref["views"]["asset_reference"]["rows_visible"] > 0


def test_absent_reference_views_are_missing_evidence_not_a_clean_bill(fleet):
    frame, as_of = fleet
    r = collect_evidence(frame, asset_id=sorted(frame.asset_id.unique())[0], as_of=as_of)
    assert r["reference_comparison"]["quantifiable"] is False
    assert any("reference" in m for m in r["missing_evidence"])


# --- explanation text is conditional on evidence ------------------------------

def test_without_peers_no_statement_claims_anything_about_peers():
    case = by_name("coherent_multi_channel")
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of, peer_ids=[])
    named = {e["explanation"]: e for e in r["explanations"]}
    mechanical = named["mechanical_change_on_this_door"]["statement"]
    assert "not on comparable peers" not in mechanical
    assert "did not show the change" not in mechanical
    assert "Peer evidence is not available" in mechanical
    assert r["peer_comparison"]["comparability_status"] == "unknown"


def test_peer_clauses_appear_only_where_peer_evidence_exists():
    r = run(by_name("coherent_multi_channel"))
    mechanical = next(e for e in r["explanations"]
                      if e["explanation"] == "mechanical_change_on_this_door")
    assert "Comparable peers did not show the change in:" in mechanical["statement"]
    assert "Peer evidence is not available" not in mechanical["statement"]


def test_unknown_channels_are_named_as_unobserved_in_the_statements(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    partial = shift(frame, [target], as_of, {"peak_current_a": 6.})
    partial.loc[partial.asset_id == target, "res_travel_mm"] = np.nan
    r = collect_evidence(partial, asset_id=target, as_of=as_of)
    sensor = next(e for e in r["explanations"]
                  if e["explanation"] == "sensor_or_measurement_change_on_this_door")
    assert "neither support nor contradict" in sensor["statement"]


def test_independent_channels_moving_does_not_contradict_a_sensor_problem():
    """Absence of corroboration is not contradiction: two faults can coexist."""
    r = run(by_name("sensor_and_mechanical_overlap"))
    named = {e["explanation"]: e for e in r["explanations"]}
    multi = [o["id"] for o in r["observations"] if o["kind"] == "multi_family"]
    assert multi
    assert not set(multi) & set(named["sensor_or_measurement_change_on_this_door"]["contradicted_by"])
    assert "does not exclude a sensor problem" in named[
        "sensor_or_measurement_change_on_this_door"]["statement"]


def test_only_a_shared_change_contradicts_this_doors_sensor():
    r = run(by_name("shared_peer_shift"))
    named = {e["explanation"]: e for e in r["explanations"]}
    shared = [o["id"] for o in r["observations"] if o["kind"] == "peer_shared"]
    assert shared
    assert set(shared) <= set(named["sensor_or_measurement_change_on_this_door"]["contradicted_by"])
    assert "does not establish a sensor fault" in named["common_influence_across_doors"]["statement"]


def test_the_explanation_order_is_fixed_and_unranked():
    for case in list(cases()) + list(reference_variants()):
        r = run(case)
        assert [e["explanation"] for e in r["explanations"]] == list(EXPLANATION_ORDER)
        assert "not a score" in r["explanation_order"]


def test_a_fleet_wide_shift_does_not_delete_a_local_one():
    named = {e["explanation"]: e for e in run(by_name("shared_shift_with_local_mechanical"))["explanations"]}
    assert named["common_influence_across_doors"]["supported_by"]
    assert named["mechanical_change_on_this_door"]["supported_by"]


def test_a_quiet_asset_produces_no_supported_explanation_but_cannot_distinguish():
    r = run(by_name("quiet_control"))
    assert r["cross_channel"]["shifted_channels"] == []
    supported = {e["explanation"] for e in r["explanations"] if e["supported_by"]}
    assert supported <= {"cannot_distinguish"}


def test_explanations_carry_no_score_or_probability_field():
    for case in list(cases()) + list(reference_variants()):
        for e in run(case)["explanations"]:
            assert set(e) == {"explanation", "statement", "supported_by",
                              "contradicted_by", "suggested_inspection_evidence"}


def test_explanations_may_only_cite_recorded_observations():
    for case in list(cases()) + list(reference_variants()):
        r = run(case)
        known = {o["id"] for o in r["observations"]}
        for e in r["explanations"]:
            assert set(e["supported_by"]) <= known
            assert set(e["contradicted_by"]) <= known


# --- quality flags and the real pipeline --------------------------------------

def test_the_pipeline_global_flag_rejects_every_channel_for_one_missing_channel(pipeline_stressed):
    """The defect this module must not inherit, measured on real pipeline output."""
    case = pipeline_stressed
    d = case.daily
    affected = d[d.asset_id.eq(case.asset_id) & (d.available_at > case.as_of - pd.Timedelta(days=5))]
    assert affected.data_quality_ok.eq(True).sum() == 0
    intact = [c for c in CHANNELS if c != "current_integral_as"]
    for c in intact:
        assert (affected[f"res_{c}_completeness"] >= .9).all()
    assert (affected["res_current_integral_as_completeness"] < .9).all()


def test_channel_completeness_keeps_intact_channels_usable(pipeline_stressed):
    """Per-channel resolution, on the frame the pipeline actually produces."""
    r = collect_evidence(pipeline_stressed.daily, asset_id=pipeline_stressed.asset_id,
                         as_of=pipeline_stressed.as_of)
    changes = r["observed_changes"]
    assert changes["current_integral_as"]["status"] == "unknown"
    assert [c for c in CHANNELS if c != "current_integral_as"
            and changes[c]["status"] == "observed"]
    basis = changes["peak_current_a"]["quality_basis"]
    assert basis["basis"] == "channel_completeness"
    assert any(READINESS_FLAG in c for c in basis["checks_applied"])
    assert basis["rows_rejected_for_inconsistent_quality_metadata"] == 0


def test_channel_completeness_is_used_where_available(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    graded = with_completeness(frame, degraded="current_integral_as")
    assert graded.data_quality_ok.eq(False).all()
    changes = collect_evidence(graded, asset_id=target, as_of=as_of)["observed_changes"]
    assert changes["current_integral_as"]["status"] == "unknown"
    assert changes["peak_current_a"]["status"] == "observed"


def test_completeness_alone_cannot_override_a_global_rejection(fleet):
    """The reported regression: no readiness metadata means the rejection stands."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    blind = with_completeness(frame, with_readiness=False)
    blind["data_quality_ok"] = False
    assert (blind[[f"res_{c}_completeness" for c in CHANNELS]] == 1.).all().all()

    r = collect_evidence(blind, asset_id=target, as_of=as_of)
    assert r["quality_basis"]["basis"] == "global_data_quality_ok"
    assert READINESS_FLAG in r["quality_basis"]["reason"]
    assert all(v["status"] == "unknown" for v in r["observed_changes"].values())


def test_the_missing_metadata_is_named(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    partial = with_completeness(frame).drop(columns=["res_travel_mm_completeness"])
    plan = collect_evidence(partial, asset_id=target, as_of=as_of)["quality_basis"]
    assert plan["basis"] == "global_data_quality_ok"
    assert any("completeness column for travel_mm" in b for b in plan["blocking_metadata"])


def test_a_broken_quality_invariant_falls_back_to_rejection(fleet):
    """Readiness true and every channel complete, yet the global flag is False.

    The metadata contradicts itself, so those rows cannot be interpreted and are
    dropped rather than resolved on a guess about which term failed.
    """
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    broken = with_completeness(frame)
    broken["data_quality_ok"] = False
    changes = collect_evidence(broken, asset_id=target, as_of=as_of)["observed_changes"]
    assert all(v["status"] == "unknown" for v in changes.values())
    basis = changes["peak_current_a"]["quality_basis"]
    assert basis["basis"] == "channel_completeness"
    assert basis["rows_rejected_for_inconsistent_quality_metadata"] > 0


def test_the_global_flag_is_the_fallback_not_a_bypass(fleet):
    """Without completeness columns the global flag still governs, and says so."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    blocked = frame.copy()
    blocked.loc[blocked.available_at > as_of - pd.Timedelta(days=5), "data_quality_ok"] = False
    r = collect_evidence(blocked, asset_id=target, as_of=as_of)
    change = r["observed_changes"]["peak_current_a"]
    assert change["status"] == "unknown"
    assert change["quality_basis"]["basis"] == "global_data_quality_ok"


def test_channel_resolution_still_honours_the_readiness_flag(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    not_ready = with_completeness(frame, ready=False)
    changes = collect_evidence(not_ready, asset_id=target, as_of=as_of)["observed_changes"]
    assert all(v["status"] == "unknown" for v in changes.values())


def test_the_pipeline_publishes_the_readiness_flag_and_keeps_the_invariant(pipeline_stressed):
    """features.to_daily and HealthPipeline.transform must stay in step."""
    d = pipeline_stressed.daily
    assert READINESS_FLAG in d.columns
    completeness = [c for c in d.columns if c.endswith("_completeness")]
    expected = d[READINESS_FLAG] & (d[completeness].min(axis=1) >= MIN_COMPLETENESS)
    assert d.data_quality_ok.eq(expected).all()
    # The readiness flag is strictly weaker than the global flag, and carries
    # the cycle-count term.
    assert d[READINESS_FLAG][d.data_quality_ok].all()
    assert not d[READINESS_FLAG][d.n_cycles < MIN_CYCLES].any()


def test_a_rejected_onboarding_reference_is_never_evidence(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    refused = with_completeness(frame).assign(onboarding_status="reference_rejected")
    changes = collect_evidence(refused, asset_id=target, as_of=as_of)["observed_changes"]
    assert all(v["status"] == "unknown" for v in changes.values())


def test_unsupported_conditions_do_not_become_a_measured_change(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    moved = shift(frame, [target], as_of, {"peak_current_a": 9.})
    moved.loc[moved.asset_id.eq(target)
              & (moved.available_at > as_of - pd.Timedelta(days=5)), "context_supported"] = False
    change = collect_evidence(moved, asset_id=target, as_of=as_of)["observed_changes"]["peak_current_a"]
    assert change["status"] == "unknown"
    assert change["direction"] == "unknown"
    assert change["change"] is None


def test_a_degenerate_reference_scale_is_not_an_infinite_shift(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    flat = frame.copy()
    flat.loc[flat.asset_id.eq(target)
             & (flat.available_at <= as_of - pd.Timedelta(days=5)), "res_peak_current_a"] = 1.
    change = collect_evidence(flat, asset_id=target, as_of=as_of)["observed_changes"]["peak_current_a"]
    assert change["status"] == "unknown"
    assert "degenerate" in change["reason"]


# --- safeguards and inputs ----------------------------------------------------

def test_no_case_authorises_any_operational_decision():
    for case in list(cases()) + list(reference_variants()):
        r = run(case)
        assert all(v is False for v in r["safeguards"].values())
        assert r["inspection_suggestions_status"] == "illustrative, pending operator review"


def test_the_record_is_json_serialisable():
    for case in list(cases()) + list(reference_variants()):
        json.dumps(run(case), allow_nan=False)


def test_a_channel_without_a_declared_family_is_refused(fleet):
    frame, as_of = fleet
    frame = frame.assign(res_mystery_v=1.)
    with pytest.raises(ValueError, match="no physical family declared"):
        collect_evidence(frame, asset_id=sorted(frame.asset_id.unique())[0],
                         as_of=as_of, channels=["peak_current_a", "mystery_v"])


def test_missing_columns_and_unknown_assets_are_refused(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    with pytest.raises(ValueError, match="missing columns"):
        collect_evidence(frame.drop(columns=["res_travel_mm"]), asset_id=target, as_of=as_of)
    with pytest.raises(KeyError):
        collect_evidence(frame, asset_id="TRN999-DOOR-9", as_of=as_of)


@pytest.mark.parametrize("kw", [
    {"recent_days": 0}, {"reference_days": 0}, {"min_peers": 0}, {"shift_sigma": 0.},
    {"shift_sigma": float("nan")}, {"peer_share_sigma": -1.}, {"min_recent_days": 99},
    {"min_reference_days": 99}, {"recent_days": 2.5}, {"context_tolerance": {}},
    {"context_tolerance": {"ambient_temp_c_mean": -1.}}, {"min_context_days": 99},
    {"min_overlap_days": 99}, {"min_channel_completeness": 0.}, {"min_channel_completeness": 1.5},
    {"location_offset_sigma": 0.}, {"affine_tolerance": -1.},
])
def test_invalid_policies_are_refused(kw):
    with pytest.raises(ValueError):
        EvidencePolicy(**kw)


def test_every_case_produces_a_record():
    got = {c.name: run(c) for c in list(cases()) + list(reference_variants())}
    assert len(got) == 15
    for name, r in got.items():
        assert r["limitation"].startswith("Experimental")
        assert r["observations"] or r["missing_evidence"], name


# --- reference provenance and scale validity ----------------------------------

def test_mixed_row_identities_are_rejected_without_a_caller_declaration(fleet):
    """Two identities in one view's rows cannot both be the frozen model."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    v = views(frame, target)
    mixed = v["asset_reference"].copy()
    rows = mixed.index[mixed.asset_id == target]
    mixed.loc[rows[:5], "normalisation_model_id"] = "a-second-model"
    r = collect_evidence(frame, asset_id=target, as_of=as_of,
                         reference_views={"asset_reference": mixed,
                                          "fleet_reference": v["fleet_reference"]})
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "not constant across the rows used" in ref["reason"]


def test_a_caller_declaration_does_not_skip_row_level_validation(fleet):
    """The reported bypass: declaring an id must not excuse the rows from checks."""
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    v = views(frame, target)
    mixed = v["asset_reference"].copy()
    rows = mixed.index[mixed.asset_id == target]
    mixed.loc[rows[:5], "normalisation_model_id"] = "a-second-model"
    r = collect_evidence(frame, asset_id=target, as_of=as_of,
                         reference_views={"asset_reference": mixed,
                                          "fleet_reference": v["fleet_reference"]},
                         reference_model_versions={"asset_reference": MODEL_ID,
                                                   "fleet_reference": MODEL_ID})
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "not constant across the rows used" in ref["reason"]
    assert "a-second-model" in ref["reason"]


def test_a_declaration_conflicting_with_constant_rows_is_rejected(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    r = collect_evidence(frame, asset_id=target, as_of=as_of,
                         reference_views=views(frame, target),
                         reference_model_versions={"asset_reference": "something-else",
                                                   "fleet_reference": "something-else"})
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "the caller declared" in ref["reason"]
    assert MODEL_ID in ref["reason"]


def test_disagreeing_identity_columns_are_rejected(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    v = views(frame, target)
    clashing = v["fleet_reference"].copy()
    clashing["model_version"] = "a-different-identity"
    r = collect_evidence(frame, asset_id=target, as_of=as_of,
                         reference_views={"asset_reference": v["asset_reference"],
                                          "fleet_reference": clashing})
    assert r["reference_comparison"]["quantifiable"] is False
    assert "identity columns disagree" in r["reference_comparison"]["reason"]


# --- the common admission gate ------------------------------------------------

def test_unsupported_reference_rows_quantify_nothing():
    r = run(by_name("all_reference_rows_unsupported", reference_variants))
    ref = r["reference_comparison"]
    assert ref["verified_model_identity"] is True
    assert ref["quantifiable"] is False
    assert "no channel could be quantified" in ref["reason"]
    for entry in ref["channels"].values():
        assert entry["status"] == "unavailable"
        assert "quality-supported asset-days" in entry["reason"]
        assert entry["admitted_observations"] == 0
    assert not [o for o in r["observations"]
                if o["kind"] in ("location_offset", "no_location_offset")]


def test_days_matched_on_the_calendar_are_not_matched_supported_days():
    """Both views cover every day; no day is supported in both."""
    case = by_name("disjoint_supported_days", reference_variants)
    r = run(case)
    ref = r["reference_comparison"]
    assert ref["matched_days"]["n_matched"] >= 5
    assert ref["quantifiable"] is False
    for entry in ref["channels"].values():
        assert entry["admitted_observations"] == 0


def test_exported_parameters_do_not_skip_the_admission_gate():
    """Numbers are not telemetry: without admitted observations there is no effect."""
    case = by_name("all_reference_rows_unsupported", reference_variants)
    parameters = {v: {c: {"location": 0., "scale": .1} for c in CHANNELS}
                  for v in ("asset_reference", "fleet_reference")}
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=case.views, reference_parameters=parameters)
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    entry = ref["channels"]["current_integral_as"]
    assert entry["status"] == "metadata_only"
    assert entry["location_offset_removed_by_adaptation"] is None
    assert "metadata only" in entry["reason"]
    assert entry["asset_reference"]["location"] == 0.
    named = {e["explanation"]: e for e in r["explanations"]}
    assert named["reference_contamination"]["supported_by"] == []


def test_missing_provenance_on_both_sides_is_not_agreement():
    """Two views with no model identifier have not been shown to share a model."""
    case = by_name("unidentified_reference_model", reference_variants)
    ref = run(case)["reference_comparison"]
    assert ref["quantifiable"] is False
    assert ref["verified_model_identity"] is False
    assert "no frozen condition-normalisation model identifier" in ref["reason"]
    assert ref["channels"] == {}
    assert ref["demonstration"] is None


def test_matching_fitting_times_alone_do_not_establish_model_identity(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    v = views(frame, target, model_id=None)
    for f in v.values():
        assert f.preprocessing_fitted_at.nunique() == 1
    r = collect_evidence(frame, asset_id=target, as_of=as_of, reference_views=v)
    assert r["reference_comparison"]["quantifiable"] is False


def test_an_unverified_demonstration_is_labelled_and_cited_by_nothing():
    case = by_name("unidentified_reference_model", reference_variants)
    r = run(case, allow_unverified_reference_demonstration=True)
    ref = r["reference_comparison"]
    demo = ref["demonstration"]
    assert ref["quantifiable"] is False
    assert demo["status"] == "unverified_algebraic_demonstration"
    assert demo["verified"] is False
    assert "must not be reported as one" in demo["warning"]
    entry = demo["channels"]["current_integral_as"]
    assert entry["location_offset_removed_by_adaptation"] == pytest.approx(
        8. * SIGMA["current_integral_as"], rel=.1)

    shown = [o["id"] for o in r["observations"] if o["kind"] == "unverified_demonstration"]
    assert shown
    cited = {i for e in r["explanations"] for i in e["supported_by"] + e["contradicted_by"]}
    assert not set(shown) & cited
    contamination = next(e for e in r["explanations"]
                         if e["explanation"] == "reference_contamination")
    assert contamination["supported_by"] == []
    assert "explicitly unverified" in contamination["statement"]


def test_provenance_is_validated_across_every_row_not_just_the_last(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    v = views(frame, target)
    drifting = v["fleet_reference"].copy()
    drifting.loc[drifting.index[0], "normalisation_model_id"] = "a-different-model"
    assert drifting.normalisation_model_id.iloc[-1] == MODEL_ID
    r = collect_evidence(frame, asset_id=target, as_of=as_of,
                         reference_views={"asset_reference": v["asset_reference"],
                                          "fleet_reference": drifting})
    ref = r["reference_comparison"]
    assert ref["quantifiable"] is False
    assert "not constant across the rows used" in ref["reason"]


def test_a_drifting_fitting_time_within_one_view_is_refused(fleet):
    frame, as_of = fleet
    target = sorted(frame.asset_id.unique())[0]
    v = views(frame, target)
    drifting = v["asset_reference"].copy()
    drifting.loc[drifting.index[0], "preprocessing_fitted_at"] += pd.Timedelta(days=1)
    r = collect_evidence(frame, asset_id=target, as_of=as_of,
                         reference_views={"asset_reference": drifting,
                                          "fleet_reference": v["fleet_reference"]})
    assert "preprocessing_fitted_at is not constant" in r["reference_comparison"]["reason"]


def test_negated_indices_are_rejected_not_reinterpreted():
    """Negating both views keeps every other check passing; the scales go negative."""
    r = run(by_name("negated_indices", reference_variants))
    ref = r["reference_comparison"]
    assert ref["verified_model_identity"] is True
    assert ref["quantifiable"] is False
    assert "no channel could be quantified" in ref["reason"]
    for c, entry in ref["channels"].items():
        assert entry["status"] == "unavailable", c
        assert entry["location_offset_removed_by_adaptation"] is None
        assert "strictly positive" in entry["reason"]
        assert "orientation convention" in entry["reason"]
    assert not [o for o in r["observations"] if o["kind"] == "location_offset"]


def consistent_parameters(case, channel="current_integral_as"):
    """The baselines the fixture's views actually applied to this channel."""
    rows = case.daily[case.daily.asset_id == case.asset_id]
    return {"asset_reference": {channel: {"location": float(rows[f"res_{channel}"].median()),
                                          "scale": SIGMA[channel]}},
            "fleet_reference": {channel: {"location": 0., "scale": SIGMA[channel]}}}


def test_exported_parameters_are_used_and_validated():
    case = by_name("contaminated_reference")
    good = consistent_parameters(case)
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=case.views, reference_parameters=good)
    entry = r["reference_comparison"]["channels"]["current_integral_as"]
    assert entry["asset_reference"]["parameter_source"] == "exported"
    assert entry["location_offset_removed_by_adaptation"] == pytest.approx(
        good["asset_reference"]["current_integral_as"]["location"])

    bad = {"asset_reference": {"current_integral_as": {"location": 0., "scale": -1.}}}
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=case.views, reference_parameters=bad)
    entry = r["reference_comparison"]["channels"]["current_integral_as"]
    assert entry["status"] == "unavailable"
    assert "strictly positive" in entry["reason"]


def test_exported_parameters_inconsistent_with_the_data_are_rejected():
    """Supplying numbers is not evidence that they are the applied baseline."""
    case = by_name("contaminated_reference")
    wrong = consistent_parameters(case)
    wrong["asset_reference"]["current_integral_as"]["location"] += 5.
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=case.views, reference_parameters=wrong)
    entry = r["reference_comparison"]["channels"]["current_integral_as"]
    assert entry["status"] == "unavailable"
    assert "do not satisfy residual = location + scale x index" in entry["reason"]
    assert entry["location_offset_removed_by_adaptation"] is None
    assert not [o for o in r["observations"] if o["kind"] == "location_offset"]


def test_a_wrong_exported_scale_is_rejected_by_the_same_check():
    case = by_name("contaminated_reference")
    wrong = consistent_parameters(case)
    wrong["fleet_reference"]["current_integral_as"]["scale"] *= 4.
    r = collect_evidence(case.daily, asset_id=case.asset_id, as_of=case.as_of,
                         reference_views=case.views, reference_parameters=wrong)
    entry = r["reference_comparison"]["channels"]["current_integral_as"]
    assert entry["status"] == "unavailable"
    assert "fleet_reference" in entry["reason"]


def test_the_affine_tolerance_is_reported_with_the_parameters():
    ref = run(by_name("contaminated_reference"))["reference_comparison"]
    entry = ref["channels"]["current_integral_as"]["asset_reference"]
    assert entry["max_fit_error"] <= entry["fit_tolerance"]
    assert entry["days_used"] > 0


def test_inferred_parameters_are_labelled_as_inferred():
    ref = run(by_name("contaminated_reference"))["reference_comparison"]
    entry = ref["channels"]["current_integral_as"]
    assert entry["asset_reference"]["parameter_source"] == "inferred_from_index"
    assert ref["parameter_source"] == "inferred_from_index"


# --- integration: inferred parameters against a real fitted pipeline ----------

@pytest.fixture(scope="module")
def onboarded_views():
    """Two real views of one asset: the fleet baseline and an adapted one."""
    from headway.onboarding import onboard
    from headway.pipeline import HealthPipeline
    from headway.synth.doors import SynthConfig, generate

    cfg = SynthConfig(n_trains=6, doors_per_train=2, n_days=70, cycles_per_day=40,
                      n_episodes=1, seed=11)
    cycles, _ = generate(cfg)
    held = sorted(cycles.train_id.unique())[0]
    start = cycles.ts.min()
    fleet_pipe = HealthPipeline().fit(
        cycles[(cycles.train_id != held) & (cycles.ts < start + pd.Timedelta(days=30))])
    ref_cut = start + pd.Timedelta(days=25)
    adapted = onboard(fleet_pipe, cycles[(cycles.train_id == held) & (cycles.ts < ref_cut)],
                      as_of=ref_cut, experimental=True)
    target = cycles[cycles.train_id == held]
    frames = {"fleet_reference": fleet_pipe.transform(target),
              "asset_reference": adapted.transform(target)}
    for f in frames.values():
        f["normalisation_model_id"] = "integration-fleet-model"
    asset = sorted(target.asset_id.unique())[0]
    assert asset in adapted.ready_at, adapted.rejected
    return frames, fleet_pipe, adapted.pipeline, asset


def exported_baseline(pipe, signal, asset):
    """Read the parameters AssetBaseline.transform would apply to this asset."""
    b = pipe.baselines[signal]
    loc = float(b.loc_[asset]) if asset in b.loc_.index else float(b.fleet_loc_)
    scale = float(b.scale_[asset]) if asset in b.scale_.index else float(b.fleet_scale_)
    return loc, scale


def test_inferred_parameters_match_the_fitted_pipelines_exported_ones(onboarded_views):
    """Affine recovery must reproduce the baselines the pipeline actually applied."""
    frames, fleet_pipe, asset_pipe, asset = onboarded_views
    daily = frames["fleet_reference"].drop(
        columns=[c for c in ("fault_confirmed", "fault_mode") if c in frames["fleet_reference"]])
    as_of = pd.Timestamp(daily.available_at.max())
    ref = collect_evidence(daily, asset_id=asset, as_of=as_of,
                           reference_views=frames)["reference_comparison"]
    assert ref["quantifiable"] is True, ref["reason"]
    assert ref["parameter_source"] == "inferred_from_index"

    checked = 0
    for c, entry in ref["channels"].items():
        if entry["status"] != "recovered":
            continue
        for view, pipe in (("asset_reference", asset_pipe), ("fleet_reference", fleet_pipe)):
            location, scale = exported_baseline(pipe, c, asset)
            assert entry[view]["location"] == pytest.approx(location, rel=1e-6, abs=1e-9)
            assert entry[view]["scale"] == pytest.approx(scale, rel=1e-6)
            checked += 1
        expected = (exported_baseline(asset_pipe, c, asset)[0]
                    - exported_baseline(fleet_pipe, c, asset)[0])
        assert entry["location_offset_removed_by_adaptation"] == pytest.approx(expected, abs=1e-9)
    assert checked >= 6


def test_exported_parameters_agree_with_the_inferred_ones(onboarded_views):
    frames, fleet_pipe, asset_pipe, asset = onboarded_views
    daily = frames["fleet_reference"]
    as_of = pd.Timestamp(daily.available_at.max())
    parameters = {}
    for view, pipe in (("asset_reference", asset_pipe), ("fleet_reference", fleet_pipe)):
        parameters[view] = {c: dict(zip(("location", "scale"),
                                        exported_baseline(pipe, c, asset))) for c in CHANNELS}
    inferred = collect_evidence(daily, asset_id=asset, as_of=as_of,
                                reference_views=frames)["reference_comparison"]
    used = collect_evidence(daily, asset_id=asset, as_of=as_of, reference_views=frames,
                            reference_parameters=parameters)["reference_comparison"]
    assert used["parameter_source"] == "exported"
    for c, entry in used["channels"].items():
        if entry["status"] != "recovered" or inferred["channels"][c]["status"] != "recovered":
            continue
        assert entry["location_offset_removed_by_adaptation"] == pytest.approx(
            inferred["channels"][c]["location_offset_removed_by_adaptation"], abs=1e-9)
