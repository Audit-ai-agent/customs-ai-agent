import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from evals.run_deterministic_validation import _reconstruct_parent
from tools.build_trade_triplet_eval_assets import (
    ADJUDICATION_SCHEMA,
    BUSINESS_FIELD_OPERATOR_IDS,
    EvalAssetError,
    build_adjudications,
    build_certificate_of_origin_fixtures,
    build_evaluation_assets,
    build_extraction_records,
    build_validation_case_index,
    inject_controlled_defects,
    validate_and_fold_adjudication_events,
)


SCHEMA_ROOT = Path(__file__).parents[1] / "evals" / "trade_document_triplets" / "v1" / "schemas"
CO_DESIGN_PATH = (
    Path(__file__).parents[1]
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "certificate_of_origin.evaluation-v2.design.json"
)


def _case(index: int) -> dict:
    case_id = f"trade-triplet-{index:04d}"
    source_hash = f"{index:064x}"[-64:]
    ci = {
        "Shipper / Seller": "Synthetic Seller",
        "Consignee (or For account & risk of Messrs)": "Synthetic Consignee",
        "Departure Date": "2025-08-13",
        "Vessel / Flight": "Synthetic Vessel / V.001E",
        "From": "Busan, South Korea",
        "To": "Los Angeles, USA",
        "Invoice No. and Date": f"KR-INV-{index:04d} / 2025-08-05",
        "Other References (Country of Origin)": "Country of Origin: Republic of Korea",
        "Goods Description": "Synthetic Item",
        "Quantity": "3 PCS",
        "Unit Price": "USD 10.00",
        "Amount": "USD 30.00",
    }
    pl = {
        "Invoice No. and Date": ci["Invoice No. and Date"],
        "Seller": "Synthetic Seller",
        "Consignee": "Synthetic Consignee",
        "Departure Date": "2025-08-13",
        "Vessel / Flight": "Synthetic Vessel / V.001E",
        "From": "Busan, South Korea",
        "To": "Los Angeles, USA",
        "No. & Kind of Packages": "10 Cartons",
        "Goods Details": [
            {
                "item": "Synthetic Item",
                "quantity": "3 PCS",
                "weight": "20.00 KGS",
            }
        ],
        "Gross Weight": "30.00 KGS",
        "Measurement": "1.50 CBM",
    }
    bl = {
        "Shipper / Exporter": "Synthetic Seller",
        "Gross Weight": "30.00 KGS",
        "Consignee": "Synthetic Consignee",
        "Ocean Vessel": "Synthetic Vessel",
        "Voyage No.": "V.001E",
        "Port of Loading": "Busan, South Korea",
        "Port of Discharge": "Los Angeles, USA",
        "No. & Kinds of Containers or Packages": "1x40HQ / 10 Cartons",
        "Revenue Tons": "1.50 RT",
        "Container No.": f"CONT{index:04d}",
        "Seal No. / Marks & No.": f"SEAL{index:04d} // SYNTHETIC",
        "Prepaid": "Y",
        "Collect": "",
    }
    documents = {}
    for document_type, filename, record in (
        ("commercial_invoice", "CI_100_Final.json", ci),
        ("packing_list", "PL_100_Final.json", pl),
        ("bill_of_lading", "BL_100_Final.json", bl),
    ):
        documents[document_type] = {
            "source_relative_path": f"raw/{filename}",
            "source_file_sha256": source_hash,
            "source_record_index_zero_based": index - 1,
            "raw_record": record,
        }
    return {
        "schema_version": "trade-document-triplet-case/v1",
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "case_id": case_id,
        "documents": documents,
        "pairing": {
            "basis": "shared_zero_based_array_index",
            "source_explicit_case_id": False,
            "author_intent_status": "unknown",
            "adjudication_status": "ambiguous",
        },
    }


def _schema(name: str) -> dict:
    return json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8"))


def test_compromised_v2_full_asset_builder_stops_before_input_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EvalAssetError,
        match="release_holdout_v2_publicly_derivable",
    ):
        build_evaluation_assets(
            tmp_path / "must-not-be-read-dataset",
            tmp_path / "must-not-be-written-output",
        )

    assert not (tmp_path / "must-not-be-written-output").exists()


def _ambiguous_observation(case_id: str) -> dict:
    return {
        "schema_version": "trade-document-triplets-anomaly/v1",
        "anomaly_id": f"{case_id}:test_rule",
        "scope": "case",
        "case_id": case_id,
        "rule_id": "test.rule",
        "status": "ambiguous",
        "intent_status": "unknown",
        "observed_condition": "source values differ",
        "expected_agent_behavior": "preserve values and require review",
        "evidence": {"left": "A", "right": "B"},
        "possible_interpretations": ["intentional defect", "generator divergence"],
        "scoring_eligible": False,
        "adjudication_required": True,
    }


def test_extraction_and_validation_gold_are_separate_and_schema_valid():
    cases = [_case(1)]
    extraction = build_extraction_records(cases)
    adjudications = build_adjudications([_ambiguous_observation(cases[0]["case_id"])])
    validation_cases = build_validation_case_index(cases, adjudications)

    Draft202012Validator(_schema("extraction-gold.schema.json")).validate(extraction[0])
    Draft202012Validator(_schema("adjudication.schema.json")).validate(adjudications[0])
    Draft202012Validator(_schema("validation-gold-case.schema.json")).validate(validation_cases[0])
    extraction_text = json.dumps(extraction)
    validation_index_text = json.dumps(validation_cases)
    assert "adjudication_id" not in extraction_text
    assert "expected_outcome" not in extraction_text
    assert "raw_record" not in validation_index_text
    assert "Synthetic Seller" not in validation_index_text
    assert validation_cases[0]["scored_observation_count"] == 0


def test_adjudication_schema_supports_six_states_and_rejects_unsafe_scoring():
    schema = _schema("adjudication.schema.json")
    validator = Draft202012Validator(schema)
    base = build_adjudications([_ambiguous_observation("trade-triplet-0001")])[0]
    records = []
    for state in (
        "confirmed_normal",
        "confirmed_intentional_defect",
        "confirmed_generator_error",
        "ambiguous",
        "coverage_gap",
        "not_applicable",
    ):
        record = copy.deepcopy(base)
        record["state"] = state
        if state in {"confirmed_normal", "confirmed_intentional_defect"}:
            record["scoring_eligible"] = True
            record["intent_status"] = "confirmed"
            record["adjudication_evidence"] = [
                {
                    "kind": "controlled_generator_manifest",
                    "reference": "test-fixture",
                }
            ]
        else:
            record["scoring_eligible"] = False
        record["expected_findings"] = (
            ["test.rule"] if state == "confirmed_intentional_defect" else []
        )
        records.append(record)

    for record in records:
        validator.validate(record)
    unsafe = copy.deepcopy(base)
    unsafe["scoring_eligible"] = True
    with pytest.raises(ValidationError):
        validator.validate(unsafe)
    assert {record["state"] for record in records} == {
        "confirmed_normal",
        "confirmed_intentional_defect",
        "confirmed_generator_error",
        "ambiguous",
        "coverage_gap",
        "not_applicable",
    }


def test_controlled_defects_are_deterministic_single_mutations_and_keep_source_unchanged():
    cases = [_case(index) for index in range(1, 21)]
    original = copy.deepcopy(cases)
    case_ids = [case["case_id"] for case in cases]

    first = inject_controlled_defects(cases, case_ids)
    second = inject_controlled_defects(cases, case_ids)

    assert first == second
    assert cases == original
    assert len(first) == 20
    assert tuple(item["mutation"]["operator_id"] for item in first) == (BUSINESS_FIELD_OPERATOR_IDS)
    assert all(item["mutation_count"] == 1 for item in first)
    assert all(
        item["adjudication"]["schema_version"] == ADJUDICATION_SCHEMA
        and item["adjudication"]["state"] == "confirmed_intentional_defect"
        and item["adjudication"]["scoring_eligible"] is True
        for item in first
    )
    adjudication_validator = Draft202012Validator(_schema("adjudication.schema.json"))
    for item in first:
        adjudication_validator.validate(item["adjudication"])
    freight = {
        item["mutation"]["operator_id"]: item
        for item in first
        if item["mutation"]["operator_id"].startswith("bl_freight_")
    }
    for operator_id, expected_truth_count in (
        ("bl_freight_both_present", 2),
        ("bl_freight_neither_present", 0),
    ):
        item = freight[operator_id]
        parent = _reconstruct_parent(item)
        parent_flags = parent["documents"]["bill_of_lading"]["raw_record"]
        mutated_flags = item["mutated_case"]["documents"]["bill_of_lading"]["raw_record"]
        assert sum(bool(str(parent_flags[name]).strip()) for name in ("Prepaid", "Collect")) == 1
        assert (
            sum(bool(str(mutated_flags[name]).strip()) for name in ("Prepaid", "Collect"))
            == expected_truth_count
        )


def test_certificate_of_origin_design_generates_20_controlled_non_legal_fixtures():
    cases = [_case(index) for index in range(1, 21)]
    splits = {
        "development": [case["case_id"] for case in cases[:15]],
        "validation": [case["case_id"] for case in cases[15:20]],
        "locked_test": [],
    }
    design = json.loads(CO_DESIGN_PATH.read_text(encoding="utf-8"))

    fixtures = build_certificate_of_origin_fixtures(cases, splits, design)

    assert len(fixtures) == 20
    assert len({fixture["fixture_id"] for fixture in fixtures}) == 20
    assert len({fixture["record"]["Goods Details"][0]["hs_code"] for fixture in fixtures}) == 20
    assert all(fixture["synthetic_hs_code"] is True for fixture in fixtures)
    assert all(fixture["hs_evidence_query_allowed"] is False for fixture in fixtures)
    assert all(
        fixture["record"]["Document Disclaimer"] == design["disclaimer"] for fixture in fixtures
    )
    assert {fixture["split"] for fixture in fixtures} == {
        "development",
        "validation",
    }


def _event(
    *,
    event_id: str,
    anomaly_id: str,
    previous_state: str,
    new_state: str,
    timestamp: str,
    previous_hash: str | None,
    evidence_kind: str = "two_reviewer_consensus",
) -> dict:
    return {
        "schema_version": "trade-document-adjudication-event/v1",
        "event_id": event_id,
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "anomaly_id": anomaly_id,
        "previous_state": previous_state,
        "new_state": new_state,
        "evidence": {
            "kind": evidence_kind,
            "reference": f"controlled-review:{event_id}",
            "sha256": "a" * 64,
        },
        "adjudicator_id": "reviewer-1",
        "adjudicated_at_utc": timestamp,
        "explanation": "Controlled adjudication transition.",
        "previous_event_sha256": previous_hash,
    }


def _event_sha256(event: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def test_adjudication_event_chain_folds_multiple_events_without_promoting_scoring():
    adjudications = build_adjudications(
        [
            _ambiguous_observation("trade-triplet-0001"),
            _ambiguous_observation("trade-triplet-0002"),
        ]
    )
    first = _event(
        event_id="event-1",
        anomaly_id="trade-triplet-0001:test_rule",
        previous_state="ambiguous",
        new_state="confirmed_normal",
        timestamp="2026-07-18T01:00:00Z",
        previous_hash=None,
    )
    second = _event(
        event_id="event-2",
        anomaly_id="trade-triplet-0002:test_rule",
        previous_state="ambiguous",
        new_state="coverage_gap",
        timestamp="2026-07-18T01:01:00Z",
        previous_hash=_event_sha256(first),
    )
    third = _event(
        event_id="event-3",
        anomaly_id="trade-triplet-0001:test_rule",
        previous_state="confirmed_normal",
        new_state="ambiguous",
        timestamp="2026-07-18T01:02:00Z",
        previous_hash=_event_sha256(second),
    )

    folded = validate_and_fold_adjudication_events(
        adjudications,
        [first, second, third],
    )

    assert folded["event_count"] == 3
    assert folded["chain_head_sha256"] == _event_sha256(third)
    assert folded["folded_states"] == {
        "trade-triplet-0001:test_rule": "ambiguous",
        "trade-triplet-0002:test_rule": "coverage_gap",
    }
    assert folded["scoring_eligible_count"] == 0
    assert not any(folded["folded_scoring_eligibility"].values())


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("previous_state", "previous_state"),
        ("previous_hash", "previous_event_sha256"),
        ("non_monotonic", "append ordered"),
        ("duplicate_id", "identity"),
        ("unknown_anomaly", "unknown anomaly"),
        ("unsafe_scoring", "JSON Schema"),
        ("unsafe_authority", "scoring-safe authority"),
    ],
)
def test_adjudication_event_chain_rejects_broken_or_unsafe_transitions(tamper, message):
    adjudications = build_adjudications(
        [
            _ambiguous_observation("trade-triplet-0001"),
            _ambiguous_observation("trade-triplet-0002"),
        ]
    )
    first = _event(
        event_id="event-1",
        anomaly_id="trade-triplet-0001:test_rule",
        previous_state="ambiguous",
        new_state="confirmed_normal",
        timestamp="2026-07-18T01:00:00Z",
        previous_hash=None,
    )
    second = _event(
        event_id="event-2",
        anomaly_id="trade-triplet-0002:test_rule",
        previous_state="ambiguous",
        new_state="coverage_gap",
        timestamp="2026-07-18T01:01:00Z",
        previous_hash=_event_sha256(first),
    )
    if tamper == "previous_state":
        second["previous_state"] = "confirmed_normal"
    elif tamper == "previous_hash":
        second["previous_event_sha256"] = "0" * 64
    elif tamper == "non_monotonic":
        second["adjudicated_at_utc"] = "2026-07-18T00:59:00Z"
    elif tamper == "duplicate_id":
        second["event_id"] = first["event_id"]
    elif tamper == "unknown_anomaly":
        second["anomaly_id"] = "trade-triplet-9999:unknown"
    elif tamper == "unsafe_scoring":
        second["scoring_eligible"] = True
    elif tamper == "unsafe_authority":
        second["new_state"] = "confirmed_intentional_defect"
        second["evidence"]["kind"] = "source_correction_record"

    with pytest.raises(EvalAssetError, match=message):
        validate_and_fold_adjudication_events(adjudications, [first, second])
