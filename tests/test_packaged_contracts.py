from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from customs_ai_agent.contracts import (
    ContractSchemaError,
    available_schemas,
    generate_schema,
    get_schema,
    schema_sha256,
    verify_packaged_schemas,
)


EXPECTED_SCHEMAS = {
    ("backend", "v1", "review-request"),
    ("backend", "v1", "review-result"),
    ("backend", "v2", "review-request"),
    ("backend", "v2", "review-result"),
    ("backend", "v3", "review-request"),
    ("backend", "v3", "review-result"),
    ("internal", "v1", "ocr-layout-ledger"),
    ("internal", "v2", "ocr-layout-ledger"),
    ("internal", "v1", "claude-text-candidate"),
    ("internal", "v1", "claude-vision-candidate"),
    ("internal", "v1", "claude-crop-read-candidate"),
    ("local", "v1", "review-manifest"),
}


def _validator(scope: str, version: str, kind: str) -> Draft202012Validator:
    schema = get_schema(scope, version, kind)  # type: ignore[arg-type]
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _property_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        names = set(value.get("properties", {}))
        for nested in value.values():
            names.update(_property_names(nested))
        return names
    if isinstance(value, list):
        names: set[str] = set()
        for nested in value:
            names.update(_property_names(nested))
        return names
    return set()


def _assert_every_object_is_closed(value: Any) -> None:
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
        for nested in value.values():
            _assert_every_object_is_closed(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_every_object_is_closed(nested)


def _referenced_definition_names(
    schema: dict[str, Any],
    root_definition: str,
) -> set[str]:
    definitions = schema["$defs"]
    pending = [root_definition]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        serialized = json.dumps(definitions[name], sort_keys=True)
        for candidate in definitions:
            if f'"$ref": "#/$defs/{candidate}"' in serialized:
                pending.append(candidate)
    return visited


def _image_request() -> dict[str, Any]:
    return {
        "case_id": "case-image-contract",
        "documents": [
            {
                "document_id": "invoice-1",
                "filename": "invoice.png",
                "image_pages": [
                    {
                        "page_number": 1,
                        "media_type": "image/png",
                        "data_base64": "eA==",
                        "byte_length": 1,
                        "sha256": "a" * 64,
                        "width_px": 1,
                        "height_px": 1,
                        "provenance": {
                            "source_media_type": "image/png",
                            "source_sha256": "a" * 64,
                            "source_page_number": 1,
                        },
                    }
                ],
            }
        ],
        "llm_mode": "llm",
    }


def _service_text_request() -> dict[str, Any]:
    return {
        "case_id": "case-service-contract",
        "documents": [
            {
                "document_id": "invoice-1",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-001",
            }
        ],
        "llm_mode": "llm",
    }


def _pdf_request() -> dict[str, Any]:
    source_pdf_sha256 = "b" * 64
    return {
        "case_id": "case-pdf-contract",
        "documents": [
            {
                "document_id": "invoice-pdf-1",
                "filename": "invoice.pdf",
                "pdf_pages": {
                    "source_pdf_sha256": source_pdf_sha256,
                    "source_pdf_byte_length": 1,
                    "page_count": 1,
                    "rasterizer_id": "pypdfium2",
                    "rasterizer_version": "5.12.0",
                    "raster_dpi": 200,
                    "pages": [
                        {
                            "page_number": 1,
                            "media_type": "image/png",
                            "data_base64": "eA==",
                            "byte_length": 1,
                            "sha256": "a" * 64,
                            "width_px": 1,
                            "height_px": 1,
                            "provenance": {
                                "source_media_type": "application/pdf",
                                "source_sha256": source_pdf_sha256,
                                "source_page_number": 1,
                                "transforms": ["pdf_rasterize"],
                                "rasterizer_id": "pypdfium2",
                                "rasterizer_version": "5.12.0",
                                "raster_dpi": 200,
                            },
                        }
                    ],
                },
            }
        ],
        "llm_mode": "llm",
    }


def test_registry_contains_only_the_versioned_release_schemas() -> None:
    assert set(available_schemas()) == EXPECTED_SCHEMAS


def test_every_packaged_schema_is_valid_and_matches_its_model() -> None:
    hashes = verify_packaged_schemas()

    assert set(hashes) == EXPECTED_SCHEMAS
    for key, digest in hashes.items():
        Draft202012Validator.check_schema(get_schema(*key))
        assert digest == schema_sha256(*key)
        assert re.fullmatch(r"[a-f0-9]{64}", digest)


def test_get_schema_returns_a_detached_copy() -> None:
    first = get_schema("backend", "v1", "review-request")
    first["title"] = "mutated by consumer"

    second = get_schema("backend", "v1", "review-request")

    assert second["title"] == "Customs AI Agent V1 review request"


def test_backend_v1_preserves_text_only_request_contract() -> None:
    schema = get_schema("backend", "v1", "review-request")
    validator = Draft202012Validator(schema)

    validator.validate(
        {
            "documents": [
                {
                    "document_id": "invoice-1",
                    "filename": "invoice.txt",
                    "content_text": "COMMERCIAL INVOICE",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        validator.validate(_image_request())
    assert set(schema["$defs"]) == {"DocumentInput"}
    assert set(schema["$defs"]["DocumentInput"]["properties"]) == {
        "document_id",
        "filename",
        "content_text",
        "declared_type",
        "requirement_document_codes",
    }
    assert "content_text" in schema["$defs"]["DocumentInput"]["required"]
    assert not {
        "image_pages",
        "ImagePageInput",
        "ImagePageProvenance",
    }.intersection(_property_names(schema) | set(schema["$defs"]))


def test_backend_v2_accepts_bounded_images_but_not_local_manifest_paths() -> None:
    schema = get_schema("backend", "v2", "review-request")
    Draft202012Validator(schema).validate(_image_request())
    Draft202012Validator(schema).validate(_pdf_request())

    property_names = _property_names(schema)
    assert "path" not in property_names
    assert "ocr_ledger" not in property_names
    assert "contract_version" not in property_names

    invalid = _image_request()
    invalid["documents"][0]["source"] = {
        "kind": "image",
        "path": "documents/invoice.png",
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(invalid)


@pytest.mark.parametrize("schema_source", ["packaged", "generated"])
def test_backend_v2_service_request_requires_real_explicit_inputs(
    schema_source: str,
) -> None:
    schema = (
        get_schema("backend", "v2", "review-request")
        if schema_source == "packaged"
        else generate_schema("backend", "v2", "review-request")
    )
    validator = Draft202012Validator(schema)
    valid = _service_text_request()

    validator.validate(valid)
    assert {"case_id", "documents", "llm_mode"}.issubset(schema["required"])

    invalid_payloads: list[dict[str, Any]] = []
    for missing in ("case_id", "documents", "llm_mode"):
        payload = _service_text_request()
        payload.pop(missing)
        invalid_payloads.append(payload)
    for null_field in ("case_id", "documents", "llm_mode"):
        payload = _service_text_request()
        payload[null_field] = None
        invalid_payloads.append(payload)
    mock_payload = _service_text_request()
    mock_payload["llm_mode"] = "mock"
    invalid_payloads.append(mock_payload)
    scenario_payload = _service_text_request()
    scenario_payload["scenario_id"] = "happy_path"
    invalid_payloads.append(scenario_payload)
    null_scenario_payload = _service_text_request()
    null_scenario_payload["scenario_id"] = None
    invalid_payloads.append(null_scenario_payload)
    unknown_payload = _service_text_request()
    unknown_payload["unknown_policy"] = "allow"
    invalid_payloads.append(unknown_payload)
    too_many_codes = _service_text_request()
    too_many_codes["uploaded_requirement_document_codes"] = [f"REQ-{index}" for index in range(101)]
    invalid_payloads.append(too_many_codes)
    long_code = _service_text_request()
    long_code["uploaded_requirement_document_codes"] = ["R" * 129]
    invalid_payloads.append(long_code)
    duplicate_codes = _service_text_request()
    duplicate_codes["uploaded_requirement_document_codes"] = ["REQ-A", "REQ-A"]
    invalid_payloads.append(duplicate_codes)
    invalid_document_codes = copy.deepcopy(_service_text_request())
    invalid_document_codes["documents"][0]["requirement_document_codes"] = ["REQ VALID WITH SPACES"]
    invalid_payloads.append(invalid_document_codes)
    invalid_case_id = _service_text_request()
    invalid_case_id["case_id"] = "case with spaces"
    invalid_payloads.append(invalid_case_id)
    unsafe_filename = copy.deepcopy(_service_text_request())
    unsafe_filename["documents"][0]["filename"] = "../private.txt"
    invalid_payloads.append(unsafe_filename)

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            validator.validate(payload)


def test_local_manifest_contract_is_separate_and_supports_text_image_and_pdf() -> None:
    schema = get_schema("local", "v1", "review-manifest")
    payload = {
        "schema_version": "customs-agent-local-review-manifest/v1",
        "contract_version": "v2",
        "llm_mode": "llm",
        "documents": [
            {
                "document_id": "invoice-text",
                "filename": "invoice.txt",
                "source": {"kind": "text", "path": "documents/invoice.txt"},
            },
            {
                "document_id": "packing-image",
                "filename": "packing.png",
                "source": {
                    "kind": "image",
                    "path": "documents/packing.png",
                    "ocr_ledger": "ocr/packing.json",
                },
            },
            {
                "document_id": "bl-pdf",
                "filename": "bill-of-lading.pdf",
                "source": {
                    "kind": "pdf",
                    "path": "documents/bill-of-lading.pdf",
                    "ocr_ledger": "ocr/bill-of-lading.json",
                },
            },
        ],
    }

    Draft202012Validator(schema).validate(payload)
    assert schema["$id"].endswith("/local/v1/review-manifest.schema.json")
    assert schema["$id"] != get_schema("backend", "v2", "review-request")["$id"]


def test_v2_result_schema_exposes_auditable_provenance_without_local_inputs() -> None:
    schema = get_schema("backend", "v2", "review-result")
    definitions = schema["$defs"]

    assert "document_results" in schema["properties"]
    assert {
        "output_schema_id",
        "output_schema_version",
        "output_schema_sha256",
    }.issubset(definitions["ModelCallRecordV2"]["required"])
    assert {
        "source",
        "confidence_basis",
    }.issubset(definitions["ReviewFieldV2"]["properties"])
    assert "ocr_layout" in definitions["ReviewDocumentResultV2"]["properties"]
    assert not {
        "content_text",
        "image_pages",
        "data_base64",
        "source_path",
        "manifest_path",
        "ocr_ledger_path",
    }.intersection(_property_names(schema))


def test_backend_v1_result_schema_excludes_all_v2_audit_and_conflict_fields() -> None:
    schema = get_schema("backend", "v1", "review-result")
    serialized = json.dumps(schema, sort_keys=True)

    assert {"ModelCallRecord", "ModelErrorDetail"}.issubset(schema["$defs"])
    assert "ModelCallRecordV2" not in schema["$defs"]
    assert "ModelErrorDetailV2" not in schema["$defs"]
    for forbidden in (
        "prompt_sha256",
        "output_schema_id",
        "output_schema_version",
        "output_schema_sha256",
        "input_modality",
        "ocr_ledger_id",
        "ocr_provider_id",
        "ocr_engine_id",
        "ocr_engine_version",
        "crop_attempted",
        "crop_bbox",
        "crop_image_sha256",
        "model_conflicts",
        "document_results",
    ):
        assert forbidden not in serialized


def test_v2_xai_report_schema_has_only_typed_closed_objects() -> None:
    schema = get_schema("backend", "v2", "review-result")
    definitions = schema["$defs"]

    assert schema["properties"]["report"] == {"$ref": "#/$defs/XaiReportV2"}
    report_definitions = _referenced_definition_names(schema, "XaiReportV2")
    assert {
        "XaiReportSummaryV2",
        "XaiDocumentSummaryV2",
        "XaiValidationIssueV2",
        "XaiRegulatoryFindingV2",
        "XaiRagEvidenceV2",
        "XaiEvidenceScoreV2",
    }.issubset(report_definitions)
    assert "XaiReport" not in report_definitions
    for name in report_definitions:
        _assert_every_object_is_closed(definitions[name])


def test_internal_schemas_are_explicitly_non_backend_contracts() -> None:
    for kind in (
        "ocr-layout-ledger",
        "claude-text-candidate",
        "claude-vision-candidate",
        "claude-crop-read-candidate",
    ):
        schema = get_schema("internal", "v1", kind)  # type: ignore[arg-type]
        Draft202012Validator.check_schema(schema)
        assert schema["$id"].endswith(f"/internal/v1/{kind}.schema.json")
        assert "not a Backend request/result contract" in schema["description"]
    ocr_v2 = get_schema("internal", "v2", "ocr-layout-ledger")
    Draft202012Validator.check_schema(ocr_v2)
    assert ocr_v2["$id"].endswith("/internal/v2/ocr-layout-ledger.schema.json")
    assert ocr_v2["properties"]["reading_order_policy_id"]["const"] == "worker-emitted-sequence/v1"


def test_checkout_backend_contract_copies_match_packaged_resources() -> None:
    root = Path(__file__).resolve().parents[1] / "contracts" / "agent"
    for version in ("v1", "v2"):
        for wire_kind in ("request", "result"):
            checkout = json.loads(
                (root / version / f"review_{wire_kind}.schema.json").read_text(encoding="utf-8")
            )
            assert checkout == get_schema(
                "backend",
                version,  # type: ignore[arg-type]
                f"review-{wire_kind}",  # type: ignore[arg-type]
            )


def test_unknown_schema_selection_fails_closed() -> None:
    with pytest.raises(ContractSchemaError, match="unsupported schema selection"):
        get_schema(
            "backend",
            "v9",  # type: ignore[arg-type]
            "review-request",
        )
