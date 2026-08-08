from __future__ import annotations

from customs_ai_agent.contracts import available_schemas, get_schema, verify_packaged_schemas
from customs_ai_agent.schemas.assistant import AssistantCorrectionDraftV1
from customs_ai_agent.schemas.v3 import ServiceReviewRunRequestV3


def test_only_live_http_schema_bytes_are_packaged() -> None:
    assert available_schemas() == (
        ("backend", "v3", "assistant-answer-request"),
        ("backend", "v3", "assistant-answer-response"),
        ("backend", "v3", "knowledge-answer-request"),
        ("backend", "v3", "knowledge-answer-response"),
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
