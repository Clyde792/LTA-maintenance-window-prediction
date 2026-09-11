"""Freeze integrity for the outage-tolerance experiment runner.

A frozen policy only means something together with the protocol it was chosen
under, the development results that chose it, and the code that produced those
results. These tests pin the refusals, because a silently accepted stale policy
would turn the evaluation stage back into an unblinded one.

No experiment is executed here: the runs are stubbed, so these stay fast and do
not depend on Parquet or on synthetic generation.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import experiment_outage_tolerance as X


def verdict(label, policy, outage, qualifies=True):
    return {"policy_label": label, "policy": policy, "qualifies": qualifies,
            "checks": {"outage_availability_affected": {"value": outage, "pass": qualifies}}}


# m3_a5_g3 wins on outage availability; m2_a5_g2 qualifies but loses; the wide
# window does not qualify at all. This mirrors the real sweep's shape.
WINNER = {"min_observations": 3, "max_age_days": 5.0, "max_gap_days": 3.0}
RUNNER_UP = {"min_observations": 2, "max_age_days": 5.0, "max_gap_days": 2.0}
REJECTED = {"min_observations": 3, "max_age_days": 10.0, "max_gap_days": 3.0}
DEV_RESULTS = {"stage": "dev", "rows": [], "verdicts": [
    verdict("m3_a5_g3", WINNER, .66),
    verdict("m2_a5_g2", RUNNER_UP, .60),
    verdict("m3_a10_g3", REJECTED, .70, qualifies=False),
]}


def restamp(d):
    """Re-hash the run's artifacts into the freeze, as a fresh run would."""
    body = json.loads((d / X.FROZEN_FILE).read_text(encoding="utf-8"))
    body["protocol.json_sha256"] = X.sha256_file(d / "protocol.json")
    body["dev_results.json_sha256"] = X.sha256_file(d / "dev_results.json")
    (d / X.FROZEN_FILE).write_text(json.dumps(body, indent=2), encoding="utf-8")


def edit_freeze(d, **changes):
    body = json.loads((d / X.FROZEN_FILE).read_text(encoding="utf-8"))
    body.update(changes)
    (d / X.FROZEN_FILE).write_text(json.dumps(body, indent=2), encoding="utf-8")


@pytest.fixture
def frozen_run(tmp_path):
    """A minimal, internally consistent frozen development run."""
    run_id = "dev-20260911T000000-deadbeef"
    d = X.run_dir(tmp_path, run_id)
    d.mkdir(parents=True)
    proto = X.protocol()
    (d / "protocol.json").write_text(json.dumps(X.clean_json(proto), indent=2), encoding="utf-8")
    (d / "dev_results.json").write_text(json.dumps(DEV_RESULTS, indent=2), encoding="utf-8")
    (d / X.FROZEN_FILE).write_text(json.dumps({
        "run_id": run_id,
        "policy": dict(WINNER),
        "policy_label": "m3_a5_g3",
        "protocol.json_sha256": X.sha256_file(d / "protocol.json"),
        "protocol_canonical_sha256": X.sha256_bytes(X.canonical(proto)),
        "dev_results.json_sha256": X.sha256_file(d / "dev_results.json"),
        "code": X.code_hashes(),
        "blinding": "unasserted",
    }, indent=2), encoding="utf-8")
    return tmp_path, run_id, d


def test_a_consistent_freeze_loads(frozen_run):
    out, run_id, _ = frozen_run
    got = X.load_frozen(out, run_id)
    assert got["policy_label"] == "m3_a5_g3"


def test_missing_run_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no such run"):
        X.load_frozen(tmp_path, "dev-does-not-exist")


def test_a_development_run_with_no_qualifier_cannot_be_evaluated(tmp_path):
    """A sweep that qualified nothing must not leave an earlier policy usable."""
    run_id = "dev-empty"
    d = X.run_dir(tmp_path, run_id)
    d.mkdir(parents=True)
    (d / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="froze no policy"):
        X.load_frozen(tmp_path, run_id)


def test_a_null_policy_is_refused(frozen_run):
    out, run_id, d = frozen_run
    edit_freeze(d, policy=None)
    with pytest.raises(SystemExit, match="no qualifying policy"):
        X.load_frozen(out, run_id)


def test_an_edited_protocol_invalidates_the_freeze(frozen_run):
    out, run_id, d = frozen_run
    proto = json.loads((d / "protocol.json").read_text(encoding="utf-8"))
    proto["acceptance_criteria"]["outage_availability_affected_min"] = .01
    (d / "protocol.json").write_text(json.dumps(proto, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit, match="protocol has changed"):
        X.load_frozen(out, run_id)


def test_edited_development_results_invalidate_the_freeze(frozen_run):
    out, run_id, d = frozen_run
    (d / "dev_results.json").write_text(json.dumps({"stage": "dev", "rows": [1]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="development results has changed"):
        X.load_frozen(out, run_id)


def test_missing_development_results_invalidate_the_freeze(frozen_run):
    out, run_id, d = frozen_run
    (d / "dev_results.json").unlink()
    with pytest.raises(SystemExit, match="missing from the run directory"):
        X.load_frozen(out, run_id)


@pytest.mark.parametrize("component", ["experiment_script", "detector_module",
                                       "baseline_module", "metrics_module"])
def test_changed_code_invalidates_the_freeze(frozen_run, component):
    """Results produced under different code are not results under this policy."""
    out, run_id, d = frozen_run
    body = json.loads((d / X.FROZEN_FILE).read_text(encoding="utf-8"))
    body["code"][component] = "0" * 64
    (d / X.FROZEN_FILE).write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(SystemExit, match=f"{component} changed"):
        X.load_frozen(out, run_id)


def test_a_freeze_copied_into_another_run_is_refused(frozen_run):
    """Guards against lifting a freeze out of the run whose evidence produced it."""
    out, run_id, d = frozen_run
    other = X.run_dir(out, "dev-other")
    other.mkdir(parents=True)
    for f in ("protocol.json", "dev_results.json", X.FROZEN_FILE):
        (other / f).write_text((d / f).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit, match="freeze names run"):
        X.load_frozen(out, "dev-other")


def test_parameter_drift_in_code_invalidates_the_freeze(frozen_run, monkeypatch):
    """The protocol this code would write now must match the frozen one."""
    out, run_id, _ = frozen_run
    monkeypatch.setitem(X.CRITERIA, "outage_availability_affected_min", .99)
    with pytest.raises(SystemExit, match="acceptance criteria differ"):
        X.load_frozen(out, run_id)


def test_evaluation_requires_a_named_run(tmp_path):
    with pytest.raises(SystemExit, match="--run is required"):
        X.main(["--stage", "eval", "--out", str(tmp_path)])


def test_canonical_hash_ignores_key_order():
    a = {"x": 1, "y": {"p": 2, "q": 3}}
    b = {"y": {"q": 3, "p": 2}, "x": 1}
    assert X.sha256_bytes(X.canonical(a)) == X.sha256_bytes(X.canonical(b))


# --- the freeze must match the winner the development evidence selects --------
# No hash covers the freeze file itself, so these edits leave every earlier
# check satisfied. The policy is re-derived from the development results.

def test_loosening_the_frozen_parameters_is_refused(frozen_run):
    """Reproduces the reported bypass: min_observations edited from 3 to 2."""
    out, run_id, d = frozen_run
    edit_freeze(d, policy={"min_observations": 2, "max_age_days": 5.0, "max_gap_days": 3.0})
    with pytest.raises(SystemExit, match="differ from those of the selected candidate"):
        X.load_frozen(out, run_id)


def test_an_equal_valued_policy_still_loads(frozen_run):
    """3 and 3.0 are the same policy; the check compares values, not JSON types."""
    out, run_id, d = frozen_run
    edit_freeze(d, policy={"min_observations": 3, "max_age_days": 5, "max_gap_days": 3})
    assert X.load_frozen(out, run_id)["policy_label"] == "m3_a5_g3"


def test_swapping_in_a_qualifying_but_unselected_policy_is_refused(frozen_run):
    """A runner-up is not the frozen policy, even though it passed the criteria."""
    out, run_id, d = frozen_run
    edit_freeze(d, policy=dict(RUNNER_UP), policy_label="m2_a5_g2")
    with pytest.raises(SystemExit, match="development results select 'm3_a5_g3'"):
        X.load_frozen(out, run_id)


def test_a_nonqualifying_policy_is_refused(frozen_run):
    out, run_id, d = frozen_run
    edit_freeze(d, policy=dict(REJECTED), policy_label="m3_a10_g3")
    with pytest.raises(SystemExit, match="did not qualify"):
        X.load_frozen(out, run_id)


def test_a_policy_absent_from_development_is_refused(frozen_run):
    out, run_id, d = frozen_run
    edit_freeze(d, policy={"min_observations": 2, "max_age_days": 7.0, "max_gap_days": 4.0},
                policy_label="m2_a7_g4")
    with pytest.raises(SystemExit, match="never evaluated in development"):
        X.load_frozen(out, run_id)


def test_a_mismatched_label_alone_is_refused(frozen_run):
    """Parameters untouched, label relabelled: still not what the sweep selected."""
    out, run_id, d = frozen_run
    edit_freeze(d, policy_label="m2_a5_g2")
    with pytest.raises(SystemExit, match="development results select"):
        X.load_frozen(out, run_id)


def test_development_results_without_verdicts_cannot_support_a_freeze(frozen_run):
    out, run_id, d = frozen_run
    (d / "dev_results.json").write_text(json.dumps({"stage": "dev", "rows": []}), encoding="utf-8")
    restamp(d)
    with pytest.raises(SystemExit, match="no candidate verdicts"):
        X.load_frozen(out, run_id)


def test_development_results_where_nothing_qualified_cannot_support_a_freeze(frozen_run):
    out, run_id, d = frozen_run
    body = {"stage": "dev", "rows": [], "verdicts": [
        verdict("m3_a5_g3", WINNER, .66, qualifies=False)]}
    (d / "dev_results.json").write_text(json.dumps(body), encoding="utf-8")
    restamp(d)
    with pytest.raises(SystemExit, match="did not qualify"):
        X.load_frozen(out, run_id)


# --- runs and published artifacts are never overwritten -----------------------

def test_a_development_run_cannot_target_an_existing_run_directory(frozen_run):
    out, run_id, _ = frozen_run
    with pytest.raises(SystemExit, match="never overwrite an earlier one"):
        X.main(["--stage", "dev", "--out", str(out), "--run", run_id])


def test_existing_evaluation_artifacts_are_never_overwritten(frozen_run):
    out, run_id, d = frozen_run
    (d / "eval_results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="--reproduction"):
        X.main(["--stage", "eval", "--out", str(out), "--run", run_id])


def test_a_reproduction_is_written_beside_the_original(frozen_run, monkeypatch):
    """The sweep is stubbed: this pins where artifacts land, not what they contain."""
    out, run_id, d = frozen_run
    original = json.dumps({"stage": "eval", "rows": ["original"]})
    (d / "eval_results.json").write_text(original, encoding="utf-8")
    monkeypatch.setattr(X, "run_seed", lambda cfg, policies, stage: ([], {}))
    X.main(["--stage", "eval", "--out", str(out), "--run", run_id, "--reproduction", "rerun-1"])

    repro = d / "reproductions" / "rerun-1"
    assert (repro / "eval_results.json").exists()
    assert (d / "eval_results.json").read_text(encoding="utf-8") == original
    body = json.loads((repro / "eval_results.json").read_text(encoding="utf-8"))
    assert body["reproduction"] == "rerun-1"
    assert body["blinding"] == "unasserted"


def test_a_reproduction_id_cannot_be_reused(frozen_run, monkeypatch):
    out, run_id, d = frozen_run
    (d / "reproductions" / "rerun-1").mkdir(parents=True)
    monkeypatch.setattr(X, "run_seed", lambda cfg, policies, stage: ([], {}))
    with pytest.raises(SystemExit, match="different --reproduction"):
        X.main(["--stage", "eval", "--out", str(out), "--run", run_id, "--reproduction", "rerun-1"])


def test_reproduction_is_rejected_for_development(tmp_path):
    with pytest.raises(SystemExit, match="applies to --stage eval only"):
        X.main(["--stage", "dev", "--out", str(tmp_path), "--reproduction", "x"])


# --- the runner never asserts blindness it cannot verify ----------------------

def test_the_freeze_makes_no_blindness_claim(frozen_run, monkeypatch):
    """A regenerated run cannot know whether the evaluation seeds are still unseen."""
    out, _, _ = frozen_run
    monkeypatch.setattr(X, "run_seed", lambda cfg, policies, stage: ([], {}))
    monkeypatch.setattr(X, "pick", lambda rows: [verdict("m3_a5_g3", WINNER, .66)])
    X.main(["--stage", "dev", "--out", str(out), "--run", "dev-fresh"])

    frozen = json.loads((X.run_dir(out, "dev-fresh") / X.FROZEN_FILE).read_text(encoding="utf-8"))
    assert frozen["blinding"] == "unasserted"
    assert "not been scored" not in frozen["note"]
    assert "must be asserted and evidenced by the operator" in frozen["note"]
