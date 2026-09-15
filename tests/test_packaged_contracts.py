from __future__ import annotations

from customs_ai_agent.contracts import available_schemas, get_schema, verify_packaged_schemas
from customs_ai_agent.schemas.assistant import AssistantCorrectionDraftV1
from customs_ai_agent.schemas.v3 import ServiceReviewRunRequestV3


def test_only_live_http_schema_bytes_are_packaged() -> None:
    assert available_schemas() == (
        ("backend", "v3", "assistant-answer-request"),
        ("backend", "v3", "assistant-answer-response"),
        ("backend", "v3", "document-extraction-request"),
        ("backend", "v3", "document-extraction-result"),
        ("backend", "v3", "hs-candidates-request"),
        ("backend", "v3", "hs-candidates-response"),
        ("backend", "v3", "knowledge-answer-request"),
        ("backend", "v3", "knowledge-answer-response"),
        ("backend", "v3", "raster-document-extraction-request"),
        ("backend", "v3", "raster-document-extraction-result"),
        ("backend", "v3", "raster-text-extraction-request"),
        ("backend", "v3", "raster-text-extraction-result"),
        ("backend", "v3", "review-request"),
        ("backend", "v3", "review-result"),
        ("evidence", "v3", "guidance-query"),
        ("evidence", "v3", "rag-retrieval"),
        ("evidence", "v3", "readiness"),
    )
    assert set(verify_packaged_schemas()) == set(available_schemas())


def test_optional_correction_targets_are_omittable_or_null_in_models_and_schema() -> None:
    request_schema = get_schema("backend", "v3", "review-request")
    response_schema = get_schema("backend", "v3", "assistant-answer-response")
    knowledge_response_schema = get_schema("backend", "v3", "knowledge-answer-response")
    request_required = request_schema["$defs"]["HumanCorrection"]["required"]
    response_required = response_schema["$defs"]["CorrectionDraft"]["required"]
    assert {"item_id", "fact_id", "unit"}.isdisjoint(request_required)
    assert {"item_id", "fact_id", "unit"}.isdisjoint(response_required)
    assert "parent_review_run_id" not in request_schema["required"]
    assert knowledge_response_schema["$defs"]["GuidanceCitation"]["properties"]["source_kind"][
        "enum"
    ] == [
        "official_law",
        "official_law_catalog",
        "official_agency_guidance",
        "official_requirement_reference",
        "official_public_guidance",
        "development_fixture",
    ]
    assert response_schema["$defs"]["AnswerSection"]["properties"]["kind"]["enum"] == [
        "case_analysis",
        "general_guidance",
        "case_application",
    ]
    assert response_schema["properties"]["answer_sections"]["maxItems"] == 4
    assert "guidance_resolution" in response_schema["properties"]
    assert "guidance_resolutions" in response_schema["properties"]
    assert "answer_format" in response_schema["properties"]
    assert response_schema["properties"]["guidance_registry_sha256"] == {
        "type": ["string", "null"],
        "pattern": "^[a-f0-9]{64}$",
    }
    review_result_schema = get_schema("backend", "v3", "review-result")
    assert review_result_schema["properties"]["temporal_context"] == {
        "anyOf": [{"$ref": "#/$defs/TemporalContext"}, {"type": "null"}]
    }
    assert review_result_schema["properties"]["temporal_coverage"] == {
        "anyOf": [{"$ref": "#/$defs/TemporalCoverage"}, {"type": "null"}]
    }
    assert review_result_schema["$defs"]["TemporalContext"]["required"] == [
        "law_as_of",
        "hsk_snapshot_as_of",
        "requirements_snapshot_as_of",
    ]
    extraction_request_schema = get_schema("backend", "v3", "document-extraction-request")
    extraction_result_schema = get_schema("backend", "v3", "document-extraction-result")
    raster_request_schema = get_schema("backend", "v3", "raster-document-extraction-request")
    raster_result_schema = get_schema("backend", "v3", "raster-document-extraction-result")
    raster_text_request_schema = get_schema("backend", "v3", "raster-text-extraction-request")
    raster_text_result_schema = get_schema("backend", "v3", "raster-text-extraction-result")
    assert extraction_request_schema["required"] == [
        "document_asset_id",
        "extraction_request_id",
        "source_sha256",
        "media_type",
    ]
    assert extraction_result_schema["properties"]["status"]["enum"] == [
        "preview_ready",
        "needs_review",
        "unavailable",
    ]
    assert "base64" not in extraction_request_schema["properties"]
    assert "base64" not in extraction_result_schema["properties"]
    assert raster_request_schema["required"][-1] == "raster_profile_revision"
    assert raster_result_schema["properties"]["schema_version"] == {
        "const": "2.0",
        "type": "string",
    }
    assert "base64" not in raster_request_schema["properties"]
    assert "base64" not in raster_result_schema["properties"]
    assert raster_text_request_schema["required"][-1] == "raster_text_profile_revision"
    assert raster_text_result_schema["properties"]["byte_size"]["maximum"] == 16 * 1024 * 1024
    assert raster_text_result_schema["properties"]["canonical_text"]["maxLength"] == 200_000
    assert raster_text_result_schema["$defs"]["Reason"]["properties"]["code"]["minLength"] == 2
    assert "base64" not in raster_text_request_schema["properties"]
    assert "base64" not in raster_text_result_schema["properties"]
    assert "case_input_notice" in response_schema["properties"]
    assert "guidance_resolution" in knowledge_response_schema["properties"]
    assert "guidance_resolutions" in knowledge_response_schema["properties"]
    assert "answer_format" in knowledge_response_schema["properties"]
    assert knowledge_response_schema["properties"]["guidance_registry_sha256"] == {
        "type": ["string", "null"],
        "pattern": "^[a-f0-9]{64}$",
    }
    assert response_schema["$defs"]["GuidanceCitation"]["properties"]["registry_source_id"][
        "type"
    ] == ["string", "null"]
    assert knowledge_response_schema["$defs"]["GuidanceResolution"]["required"] == [
        "topic",
        "scope",
        "requested_direction",
        "applicable_directions",
        "coverage",
        "snapshot_id",
        "as_of",
        "registry_version",
    ]

    request = ServiceReviewRunRequestV3.model_validate(
        {
            "case_id": "case-optional-correction",
            "review_run_id": "run-optional-correction",
            "review_as_of": "2026-08-05",
            "trade_type": "import",
            "jurisdiction": "KR",
            "documents": [
                {
                    "document_id": "invoice",
                    "filename": "invoice.txt",
                    "content_text": "COMMERCIAL INVOICE",
                }
            ],
            "applied_correction_proposal_ids": ["proposal-1"],
            "human_corrections": [
                {
                    "correction_proposal_id": "proposal-1",
                    "document_id": "invoice",
                    "field_name": "quantity",
                    "new_value": "2 PCS",
                    "reason": "담당자 확인",
                }
            ],
        }
    )
    correction = request.human_corrections[0]
    assert (correction.item_id, correction.fact_id, correction.unit) == (None, None, None)
    draft = AssistantCorrectionDraftV1.model_validate(
        {
            "document_id": "invoice",
            "field_name": "quantity",
            "new_value": "2 PCS",
            "reason": "담당자 확인",
        }
    )
    assert (draft.item_id, draft.fact_id, draft.unit) == (None, None, None)
