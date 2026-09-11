from headway.data_readiness import assess_readiness


def test_unreviewed_data_never_claims_ready(cycles, episodes):
    report = assess_readiness(cycles, episodes)
    assert not report["current_pipeline"]["ready_for_reviewed_trial"]
    assert not report["labelled_evaluation"]["ready_for_reviewed_trial"]


def test_reviewed_simulator_contract_can_be_used_for_trial(cycles, episodes):
    cycles = cycles.drop_duplicates(["asset_id", "ts"])
    report = assess_readiness(cycles, episodes, units_verified=True, identities_verified=True,
                             fault_labels_verified=True, minimum_fault_groups=1)
    assert report["current_pipeline"]["ready_for_reviewed_trial"]
    assert report["labelled_evaluation"]["ready_for_reviewed_trial"]


def test_missing_channel_blocks_current_pipeline(cycles):
    report = assess_readiness(cycles.drop(columns="travel_mm"), units_verified=True, identities_verified=True)
    assert "travel_mm" in report["missing_columns"]
    assert not report["current_pipeline"]["ready_for_reviewed_trial"]


def test_fault_free_data_does_not_imply_evaluation_ready(cycles):
    cycles = cycles.drop_duplicates(["asset_id", "ts"])
    report = assess_readiness(cycles, units_verified=True, identities_verified=True, fault_labels_verified=True)
    assert report["current_pipeline"]["ready_for_reviewed_trial"]
    assert not report["labelled_evaluation"]["ready_for_reviewed_trial"]


def test_unjoinable_faults_are_reported(cycles, episodes):
    report = assess_readiness(cycles, episodes.assign(asset_id="unknown"), units_verified=True,
                             identities_verified=True, fault_labels_verified=True)
    assert any("joined" in r for r in report["labelled_evaluation"]["blockers"])


def test_source_is_not_mutated(cycles):
    import pandas as pd
    before = cycles.copy(deep=True)
    assess_readiness(cycles)
    pd.testing.assert_frame_equal(cycles, before)


def test_duplicate_cycle_identity_requires_review(cycles):
    import pandas as pd
    c = pd.concat([cycles, cycles.iloc[:1]], ignore_index=True)
    report = assess_readiness(c, units_verified=True, identities_verified=True)
    assert report["duplicate_asset_timestamps"] > 0
    assert not report["current_pipeline"]["ready_for_reviewed_trial"]
