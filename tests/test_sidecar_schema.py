"""Sidecar schema validation in scripts/ps3_readiness.py, tested in-process.

A sidecar is only worth reading once it has been shown to describe this exact
Parquet and to be well formed. These tests call the validator directly: starting
a subprocess would add no coverage to a pure function over a dict and a file.
The CLI paths are covered in tests/test_mapping_export.py.
"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ps3_readiness as R  # noqa: E402

HASH_A, HASH_B = "a" * 64, "b" * 64


@pytest.fixture
def parquet(tmp_path):
    path = tmp_path / "canonical_door_cycles.parquet"
    pd.DataFrame({"asset_id": ["A"], "ts": [pd.Timestamp("2026-03-01")]}).to_parquet(path)
    return path


def valid(parquet, **over):
    """A sidecar matching what bind_data.py writes for a complete export."""
    body = {
        "input": {"path": "vendor.csv", "sha256": HASH_A, "rows_read": None,
                  "complete_input": True},
        "output": {"path": parquet.name, "sha256": R.sha256_file(parquet)},
        "mapping": {"canonical_sha256": HASH_B, "reviewed": True},
        "attestations": {"units_verified": False},
        "context_supported": True,
        "unit_conversions": {"peak_current_a": 0.001},
    }
    body.update(over)
    return body


def check(parquet, body):
    """Write the sidecar, read it back through the real entry point."""
    parquet.with_suffix(".provenance.json").write_text(json.dumps(body), encoding="utf-8")
    result = R.source_provenance(parquet)
    # The report is printed with allow_nan=False; a result that cannot be
    # serialised would be a traceback in the CLI.
    json.dumps(result, allow_nan=False)
    return result


def assert_unverified(result, fragment):
    assert result["verified"] is False
    assert any(fragment in p for p in result["problems"]), result["problems"]
    for trusted in ("complete_input", "unit_conversions", "context_supported", "warning"):
        assert trusted not in result, f"an unverified {trusted} was reported as fact"


# --- the baseline ----------------------------------------------------------------

def test_a_valid_complete_sidecar_is_verified(parquet):
    result = check(parquet, valid(parquet))
    assert result["verified"] is True
    assert result["complete_input"] is True
    assert "warning" not in result


def test_a_valid_sampled_sidecar_is_verified_and_warns(parquet):
    body = valid(parquet)
    body["input"].update(complete_input=False, rows_read=50000)
    result = check(parquet, body)
    assert result["verified"] is True
    assert "SAMPLED EXPORT" in result["warning"]
    assert "50,000" in result["warning"]


# --- the three reported cases ----------------------------------------------------

def test_a_list_input_section_is_unverified_not_a_traceback(parquet):
    assert_unverified(check(parquet, {"input": ["bad"], "output": {}}),
                      "section 'input' must be an object, got list")


def test_a_list_mapping_section_is_unverified_not_a_traceback(parquet):
    assert_unverified(check(parquet, valid(parquet, mapping=["x"])),
                      "section 'mapping' must be an object, got list")


def test_a_matching_hash_without_a_filename_is_not_verified(parquet):
    body = valid(parquet)
    del body["output"]["path"]
    assert_unverified(check(parquet, body), "output.path must be a non-empty filename")


# --- output identity -------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   ", None, 7, ["x.parquet"]])
def test_a_blank_or_non_string_filename_is_not_verified(parquet, name):
    body = valid(parquet)
    body["output"]["path"] = name
    assert_unverified(check(parquet, body), "output.path must be a non-empty filename")


def test_a_different_filename_is_not_verified(parquet):
    body = valid(parquet)
    body["output"]["path"] = "other_export.parquet"
    assert_unverified(check(parquet, body), "belongs to another export")


@pytest.mark.parametrize("digest", [None, "", "abc", "A" * 64, "g" * 64, 12345, ["x"]])
def test_a_malformed_output_hash_is_not_verified(parquet, digest):
    body = valid(parquet)
    body["output"]["sha256"] = digest
    assert_unverified(check(parquet, body), "output.sha256 must be a 64-character")


def test_a_well_formed_but_wrong_hash_is_not_verified(parquet):
    body = valid(parquet)
    body["output"]["sha256"] = HASH_A
    assert_unverified(check(parquet, body), "output hash mismatch")


def test_a_filename_alone_is_not_enough(parquet):
    body = valid(parquet)
    del body["output"]["sha256"]
    assert_unverified(check(parquet, body), "output.sha256 must be a 64-character")


# --- sampling metadata -----------------------------------------------------------

@pytest.mark.parametrize("flag", ["false", "False", 0, 1, None, "", [], {}])
def test_complete_input_must_be_a_real_boolean(parquet, flag):
    """The string "false" must not let a sample skip its warning."""
    body = valid(parquet)
    body["input"]["complete_input"] = flag
    result = check(parquet, body)
    assert_unverified(result, "input.complete_input must be true or false")
    assert result["unverified_claims"]["complete_input"] == flag


def test_a_missing_complete_input_is_not_verified(parquet):
    body = valid(parquet)
    del body["input"]["complete_input"]
    assert_unverified(check(parquet, body), "input.complete_input must be true or false")


@pytest.mark.parametrize("rows", [True, -1, "50000", 1.5])
def test_rows_read_must_be_a_non_negative_integer(parquet, rows):
    body = valid(parquet)
    body["input"].update(complete_input=False, rows_read=rows)
    assert_unverified(check(parquet, body), "input.rows_read must be a non-negative integer")


def test_a_sample_without_its_size_is_not_verified(parquet):
    body = valid(parquet)
    body["input"].update(complete_input=False, rows_read=None)
    assert_unverified(check(parquet, body), "sample size is unrecorded")


def test_a_complete_export_claiming_a_sample_size_is_not_verified(parquet):
    body = valid(parquet)
    body["input"].update(complete_input=True, rows_read=50)
    assert_unverified(check(parquet, body), "contradict each other")


def test_a_malformed_input_hash_is_not_verified(parquet):
    body = valid(parquet)
    body["input"]["sha256"] = "not-a-hash"
    assert_unverified(check(parquet, body), "input.sha256 must be a 64-character")


# --- conversions, context, mapping, attestations ---------------------------------

@pytest.mark.parametrize("conversions,fragment", [
    (["peak_current_a"], "unit_conversions must be an object or null"),
    ("0.001", "unit_conversions must be an object or null"),
    ({"peak_current_a": True}, "must be a finite positive number"),
    ({"peak_current_a": "0.001"}, "must be a finite positive number"),
    ({"peak_current_a": 0}, "must be a finite positive number"),
    ({"peak_current_a": -1}, "must be a finite positive number"),
])
def test_malformed_unit_conversions_are_not_verified(parquet, conversions, fragment):
    assert_unverified(check(parquet, valid(parquet, unit_conversions=conversions)), fragment)


def test_null_unit_conversions_are_accepted(parquet):
    assert check(parquet, valid(parquet, unit_conversions=None))["verified"] is True


@pytest.mark.parametrize("context", ["true", "yes", 1, []])
def test_context_supported_must_be_boolean_or_null(parquet, context):
    assert_unverified(check(parquet, valid(parquet, context_supported=context)),
                      "context_supported must be true, false or null")


def test_a_malformed_mapping_hash_is_not_verified(parquet):
    assert_unverified(check(parquet, valid(parquet, mapping={"canonical_sha256": 42})),
                      "mapping.canonical_sha256 must be a 64-character")


@pytest.mark.parametrize("section", ["input", "output", "mapping", "attestations"])
def test_each_required_section_is_required(parquet, section):
    body = valid(parquet)
    del body[section]
    assert_unverified(check(parquet, body), f"missing required section {section!r}")


@pytest.mark.parametrize("section", ["input", "output", "mapping", "attestations"])
@pytest.mark.parametrize("value", [["x"], "x", 0, True])
def test_each_section_must_be_an_object(parquet, section, value):
    body = valid(parquet, **{section: value})
    assert_unverified(check(parquet, body), f"section {section!r} must be an object")


# --- whole-document failures ------------------------------------------------------

@pytest.mark.parametrize("text", ["[]", "\"x\"", "42", "null", "true"])
def test_a_non_object_document_is_not_verified(parquet, text):
    parquet.with_suffix(".provenance.json").write_text(text, encoding="utf-8")
    result = R.source_provenance(parquet)
    json.dumps(result, allow_nan=False)
    assert result["verified"] is False
    assert any("must be a JSON object" in p for p in result["problems"])


@pytest.mark.parametrize("text", ["{not json", '{"input": NaN}', '{"x": Infinity}', ""])
def test_unreadable_json_is_not_verified(parquet, text):
    """NaN is accepted by json.loads by default, and would crash allow_nan=False."""
    parquet.with_suffix(".provenance.json").write_text(text, encoding="utf-8")
    result = R.source_provenance(parquet)
    json.dumps(result, allow_nan=False)
    assert result["verified"] is False
    assert "not readable JSON" in result["note"]


def test_every_problem_is_reported_not_just_the_first(parquet):
    """An actionable reason means the whole list, so one fix is not followed by another run."""
    body = valid(parquet)
    body["output"]["path"] = ""
    body["input"]["complete_input"] = "false"
    body["context_supported"] = "yes"
    result = check(parquet, body)
    assert result["verified"] is False
    assert len(result["problems"]) >= 3


def test_malformed_claims_are_quarantined_without_being_type_coerced(parquet):
    body = valid(parquet, unit_conversions=["bad"])
    body["input"]["complete_input"] = "false"
    result = check(parquet, body)
    claims = result["unverified_claims"]
    assert claims["complete_input"] == "false"
    assert claims["unit_conversions"] == ["bad"]


def test_the_validator_never_raises_on_arbitrary_nesting(parquet):
    """A sweep over structurally hostile documents: reasons, never exceptions."""
    base = valid(parquet)
    hostile = [None, [], "", 0, True, 1.5, {"nested": {"deeper": []}}]
    for key in ("input", "output", "mapping", "attestations", "context_supported",
                "unit_conversions"):
        for value in hostile:
            body = copy.deepcopy(base)
            body[key] = value
            problems = R.sidecar_problems(body, parquet)
            assert isinstance(problems, list)
            # A non-object where a section belongs must always be reported. (An
            # arbitrary object is a structurally valid `attestations`, so dicts
            # are only checked for not raising.)
            if key in ("input", "output", "mapping", "attestations") \
                    and not isinstance(value, dict):
                assert problems, f"a {type(value).__name__} {key} section was accepted"
    for inner in ("input", "output", "mapping"):
        for field in ("path", "sha256", "complete_input", "rows_read", "canonical_sha256"):
            for value in hostile:
                body = copy.deepcopy(base)
                body[inner] = dict(body[inner], **{field: value})
                assert isinstance(R.sidecar_problems(body, parquet), list)


def test_a_missing_parquet_is_reported_rather_than_raised(tmp_path):
    missing = tmp_path / "gone.parquet"
    body = {"input": {"sha256": HASH_A, "complete_input": True, "rows_read": None},
            "output": {"path": missing.name, "sha256": HASH_A},
            "mapping": {"canonical_sha256": HASH_B}, "attestations": {}}
    problems = R.sidecar_problems(body, missing)
    assert any("does not exist" in p for p in problems)
