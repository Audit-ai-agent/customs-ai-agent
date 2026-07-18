from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from customs_ai_agent.contracts import generate_schema, schema_sha256
from customs_ai_agent.runner import (
    run_review,
    run_review_v2,
    to_review_run_result,
    to_review_run_result_v2,
)
from customs_ai_agent.schemas.models import (
    DocumentInput,
    ModelCallRecord,
    ModelErrorDetail,
    ReviewRunRequest,
    ReviewRunRequestV2,
    ReviewRunResult,
    ReviewRunResultV2,
)


V1_DOCUMENT_FIELDS = {
    "document_id",
    "filename",
    "content_text",
    "declared_type",
    "requirement_document_codes",
}
V1_REQUEST_FIELDS = {
    "scenario_id",
    "case_id",
    "documents",
    "llm_mode",
    "review_run_id",
    "review_as_of",
    "trade_type",
    "jurisdiction",
    "uploaded_requirement_document_codes",
}
V1_MODEL_CALL_FIELDS = {
    "document_id",
    "provider",
    "model_id",
    "prompt_id",
    "prompt_version",
    "request_id",
    "message_id",
    "stop_reason",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "latency_ms",
}
V1_MODEL_ERROR_FIELDS = {
    "document_id",
    "code",
    "retryable",
    "provider",
    "model_id",
    "prompt_id",
    "prompt_version",
    "latency_ms",
    "provider_request_id",
    "provider_status_code",
}
V1_RESULT_FIELDS = {
    "schema_version",
    "case_id",
    "review_run_id",
    "review_as_of",
    "trade_type",
    "jurisdiction",
    "status",
    "requires_human_review",
    "retrieval_status",
    "retrieval_coverage",
    "knowledge_snapshot_id",
    "retrieval_request",
    "retrieval_results",
    "retrieval_errors",
    "validation_issues",
    "regulatory_findings",
    "rag_evidence",
    "report",
    "audit_trail",
    "model_calls",
    "model_errors",
}
V2_ONLY_WIRE_NAMES = {
    "image_pages",
    "ImagePageInput",
    "ImagePageProvenance",
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
}
V1_REQUEST_SCHEMA_SHA256 = "d0917d6834040705dac019f81a2fe34265d93f2c591c5e605b9307f5257b48e4"
V1_RESULT_SCHEMA_SHA256 = "dfed07c3f00ac9036e195c9c873400cdab99568771d1b4d37c6db47381496de8"


def _extended_model_call() -> dict[str, object]:
    return {
        "document_id": "doc_inv",
        "provider": "anthropic",
        "model_id": "claude-test",
        "prompt_id": "document-analysis",
        "prompt_version": "1.0.0",
        "prompt_sha256": "a" * 64,
        "output_schema_id": "claude-document-candidate",
        "output_schema_version": "1.0",
        "output_schema_sha256": "b" * 64,
        "request_id": "request-1",
        "message_id": "message-1",
        "stop_reason": "end_turn",
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "latency_ms": 12,
        "input_modality": "text",
    }


def _extended_model_error() -> dict[str, object]:
    return {
        "document_id": "doc_inv",
        "code": "structured_output_invalid",
        "retryable": False,
        "provider": "anthropic",
        "model_id": "claude-test",
        "prompt_id": "document-analysis",
        "prompt_version": "1.0.0",
        "prompt_sha256": "a" * 64,
        "latency_ms": 12,
        "provider_request_id": "request-1",
        "provider_status_code": 400,
        "input_modality": "text",
    }


def _model_conflict() -> dict[str, object]:
    return {
        "document_id": "doc_inv",
        "field_path": "fields.invoice_no",
        "status": "crop_not_eligible",
        "ocr_ledger_id": "ocr_" + "c" * 64,
        "ocr_span_ids": ["span-1"],
        "candidate_value_sha256": "d" * 64,
        "ocr_text_sha256": "e" * 64,
        "crop_attempted": False,
    }


def test_v1_python_models_match_the_legacy_text_only_wire_field_sets() -> None:
    assert set(DocumentInput.model_fields) == V1_DOCUMENT_FIELDS
    assert set(ReviewRunRequest.model_fields) == V1_REQUEST_FIELDS
    assert set(ModelCallRecord.model_fields) == V1_MODEL_CALL_FIELDS
    assert set(ModelErrorDetail.model_fields) == V1_MODEL_ERROR_FIELDS
    assert set(ReviewRunResult.model_fields) == V1_RESULT_FIELDS

    request = ReviewRunRequest(
        case_id="case-v1",
        documents=[
            DocumentInput(
                document_id="doc-v1",
                filename="invoice.txt",
                content_text="COMMERCIAL INVOICE",
            )
        ],
    )
    assert set(request.model_dump(mode="json")["documents"][0]) == V1_DOCUMENT_FIELDS


def test_v1_request_model_and_schema_have_no_multimodal_definitions() -> None:
    schema = generate_schema("backend", "v1", "review-request")
    serialized = json.dumps(schema, sort_keys=True)

    assert set(schema["$defs"]) == {"DocumentInput"}
    for forbidden in V2_ONLY_WIRE_NAMES:
        assert forbidden not in serialized

    with pytest.raises(ValidationError):
        ReviewRunRequest.model_validate(
            {
                "case_id": "case-image-forbidden",
                "documents": [
                    {
                        "document_id": "doc-image",
                        "filename": "invoice.png",
                        "image_pages": [{"page_number": 1}],
                    }
                ],
                "llm_mode": "llm",
            }
        )


def test_v1_schema_digests_are_immutable_compatibility_constants() -> None:
    assert schema_sha256("backend", "v1", "review-request") == V1_REQUEST_SCHEMA_SHA256
    assert schema_sha256("backend", "v1", "review-result") == V1_RESULT_SCHEMA_SHA256


def test_v1_result_projects_internal_v2_audit_without_leaking_additive_fields() -> None:
    state = run_review(ReviewRunRequest(scenario_id="happy_path"))
    state["model_calls"] = [_extended_model_call()]
    state["model_errors"] = [_extended_model_error()]
    state["model_conflicts"] = [_model_conflict()]

    result = to_review_run_result(state)
    payload = result.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)

    assert set(payload) == V1_RESULT_FIELDS
    assert set(payload["model_calls"][0]) == V1_MODEL_CALL_FIELDS
    assert set(payload["model_errors"][0]) == V1_MODEL_ERROR_FIELDS
    for forbidden in V2_ONLY_WIRE_NAMES:
        assert forbidden not in serialized


def test_v2_runner_and_result_retain_additive_audit_metadata() -> None:
    state = run_review_v2(ReviewRunRequestV2(scenario_id="happy_path"))
    state["model_calls"] = [_extended_model_call()]
    state["model_errors"] = [_extended_model_error()]

    payload = to_review_run_result_v2(state).model_dump(mode="json")

    assert payload["schema_version"] == "2.0"
    assert payload["model_calls"][0]["prompt_sha256"] == "a" * 64
    assert payload["model_calls"][0]["output_schema_sha256"] == "b" * 64
    assert payload["model_calls"][0]["input_modality"] == "text"
    assert payload["model_errors"][0]["prompt_sha256"] == "a" * 64
    assert payload["model_errors"][0]["input_modality"] == "text"
    assert payload["model_conflicts"] == []


def test_v2_rejects_unknown_nested_bbox_field_without_changing_v1_runtime() -> None:
    state = run_review_v2(ReviewRunRequestV2(scenario_id="invoice_total_mismatch"))
    source_bbox = {
        "page": 1,
        "x1": 1,
        "y1": 2,
        "x2": 10,
        "y2": 20,
    }
    state["validation_issues"][0]["source_bbox"] = source_bbox
    state["report_json"]["issues"][0]["source_bbox"] = source_bbox

    v2_payload = to_review_run_result_v2(state).model_dump(mode="json")
    v2_payload["validation_issues"][0]["source_bbox"]["unexpected_nested_field"] = "reject"
    with pytest.raises(ValidationError):
        ReviewRunResultV2.model_validate(v2_payload)

    del v2_payload["validation_issues"][0]["source_bbox"]["unexpected_nested_field"]
    assert v2_payload["audit_trail"]
    v2_payload["audit_trail"][0]["unexpected_nested_field"] = "reject"
    with pytest.raises(ValidationError):
        ReviewRunResultV2.model_validate(v2_payload)

    del v2_payload["audit_trail"][0]["unexpected_nested_field"]
    v2_payload["regulatory_findings"] = [
        {
            "finding_code": "TEST_FINDING",
            "severity": "high",
            "message": "Test strict regulatory finding.",
            "item_id": "item-1",
            "required_document_code": "TEST-DOC",
            "required_document_name": "Test document",
            "evidence_ids": ["ev_" + "a" * 64],
            "confidence": 0.9,
            "confidence_basis": "structured_requirement_policy",
            "unexpected_nested_field": "reject",
        }
    ]
    with pytest.raises(ValidationError):
        ReviewRunResultV2.model_validate(v2_payload)

    v1_payload = to_review_run_result(state).model_dump(mode="json")
    v1_payload["validation_issues"][0]["source_bbox"]["unexpected_nested_field"] = "legacy"
    legacy_result = ReviewRunResult.model_validate(v1_payload)
    assert (
        "unexpected_nested_field"
        not in legacy_result.model_dump(mode="json")["validation_issues"][0]["source_bbox"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "scenario_id": "happy_path",
            "unexpected_control": "discarding this would be unsafe",
        },
        {
            "case_id": "case-v2-extra-document",
            "documents": [
                {
                    "document_id": "doc-v2",
                    "filename": "invoice.txt",
                    "content_text": "COMMERCIAL INVOICE",
                    "source_path": "private/invoice.txt",
                }
            ],
        },
    ],
)
def test_v2_python_request_models_reject_unknown_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReviewRunRequestV2.model_validate(payload)


def test_v2_result_rejects_cross_layer_status_and_summary_tampering() -> None:
    payload = to_review_run_result_v2(
        run_review_v2(ReviewRunRequestV2(scenario_id="happy_path"))
    ).model_dump(mode="json")

    tampered = json.loads(json.dumps(payload))
    tampered["report"]["overall_status"] = "needs_review"
    with pytest.raises(ValidationError, match="status must match"):
        ReviewRunResultV2.model_validate(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["report"]["summary"]["total_issues"] = 99
    with pytest.raises(ValidationError, match="total_issues"):
        ReviewRunResultV2.model_validate(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["report"]["summary"]["retrieval_coverage"] = "partial"
    with pytest.raises(ValidationError, match="retrieval_coverage"):
        ReviewRunResultV2.model_validate(tampered)


@pytest.mark.parametrize("mutation", ["empty", "partial", "item_error"])
def test_v2_completed_result_requires_real_complete_item_level_retrieval(mutation: str) -> None:
    payload = to_review_run_result_v2(
        run_review_v2(ReviewRunRequestV2(scenario_id="happy_path"))
    ).model_dump(mode="json")

    if mutation == "empty":
        payload["retrieval_results"] = []
        payload["rag_evidence"] = []
        payload["report"]["rag_evidence"] = []
    elif mutation == "partial":
        payload["retrieval_results"][0]["coverage"] = "partial"
    else:
        payload["retrieval_results"][0]["errors"] = [
            {
                "code": "PROVIDER_PARTIAL",
                "message": "Evidence item is incomplete.",
                "retryable": True,
            }
        ]

    with pytest.raises(ValidationError):
        ReviewRunResultV2.model_validate(payload)


def test_v2_result_binds_retrieval_request_and_evidence_to_the_review() -> None:
    payload = to_review_run_result_v2(
        run_review_v2(ReviewRunRequestV2(scenario_id="happy_path"))
    ).model_dump(mode="json")

    wrong_case = json.loads(json.dumps(payload))
    wrong_case["retrieval_request"]["case_id"] = "case-other"
    with pytest.raises(ValidationError, match="retrieval request identity"):
        ReviewRunResultV2.model_validate(wrong_case)

    detached_evidence = json.loads(json.dumps(payload))
    detached_evidence["rag_evidence"] = []
    detached_evidence["report"]["rag_evidence"] = []
    with pytest.raises(ValidationError, match="item-level retrieval evidence"):
        ReviewRunResultV2.model_validate(detached_evidence)


@pytest.mark.parametrize("mutation", ["unknown_document", "wrong_modality"])
def test_v2_model_call_audit_is_bound_to_its_public_document(mutation: str) -> None:
    payload = to_review_run_result_v2(
        run_review_v2(ReviewRunRequestV2(scenario_id="happy_path"))
    ).model_dump(mode="json")
    call = _extended_model_call()
    if mutation == "unknown_document":
        call["document_id"] = "doc-not-present"
    else:
        call.update(
            {
                "input_modality": "image",
                "ocr_ledger_id": "ocr_" + "a" * 64,
                "ocr_provider_id": "unbound-provider",
                "ocr_engine_id": "unbound-engine",
                "ocr_engine_version": "0",
            }
        )
    payload["model_calls"] = [call]

    with pytest.raises(ValidationError):
        ReviewRunResultV2.model_validate(payload)


def test_v2_public_any_values_reject_nested_sensitive_payload_keys() -> None:
    payload = to_review_run_result_v2(
        run_review_v2(ReviewRunRequestV2(scenario_id="happy_path"))
    ).model_dump(mode="json")

    unsafe_audit = json.loads(json.dumps(payload))
    unsafe_audit["audit_trail"][0]["metadata"]["data_base64"] = "raw-media"
    with pytest.raises(ValidationError, match="forbidden sensitive key"):
        ReviewRunResultV2.model_validate(unsafe_audit)

    unsafe_field = json.loads(json.dumps(payload))
    unsafe_field["document_results"][0]["fields"][0]["value"] = {
        "nested": {"raw_response": "provider payload"}
    }
    with pytest.raises(ValidationError, match="forbidden sensitive key"):
        ReviewRunResultV2.model_validate(unsafe_field)
