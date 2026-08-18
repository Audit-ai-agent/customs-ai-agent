from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from typing import Any
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import customs_ai_agent.service.profiles as profile_module
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.schemas.v3 import (
    ReviewRunResultV3,
    ServiceReviewRunRequestV3,
    canonical_json_sha256,
    review_result_v3_sha256,
)
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.guidance_registry import guidance_registry
from customs_ai_agent.service.profiles import (
    _GUIDANCE_LEGACY_SAFE_QUERY,
    _GUIDANCE_LEGACY_TOPIC_BY_SCOPE,
    _GUIDANCE_SAFE_QUERY,
    _guidance_query_request,
    _DeterministicDocumentAnalyzer,
    _actions,
    _dedupe_evidence_records,
    _evidence_from_data,
    _findings_from_evidence,
    _retrieve_request,
    _temporal_coverage_from_data,
    _temporal_context_from_data,
    ProfileEvidenceRetriever,
    ProfileRuntime,
    SERVICE_MODE_HEADER,
)
from customs_ai_agent.service.runtime_errors import RuntimeConfigurationError, RuntimeExecutionError
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
_GUIDANCE_TOKEN = "Zyxwvutsrqponmlkjihgfedcba" + "9876543210ABCDEFGHIJK"


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_TOKEN),
        ),
        **changes,
    )


def test_default_profile_is_local_development() -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        }
    )
    assert settings.service_mode == "local-development"
    assert settings.port == 18090
    assert settings.guidance_base_url is None
    assert settings.guidance_service_token is None
    assert settings.guidance_registry_version is None
    assert settings.readiness_timeout_seconds == 5
    assert settings.evidence_retrieve_timeout_seconds == 90


def test_evidence_retrieve_timeout_is_environment_configurable_and_bounded() -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
        "EVIDENCE_SERVICE_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        "CUSTOMS_AGENT_EVIDENCE_RETRIEVE_TIMEOUT_SECONDS": "91.5",
    }

    assert ServiceSettings.from_environment(source).evidence_retrieve_timeout_seconds == 91.5
    for invalid in (0.9, 120.1):
        with pytest.raises(
            ServiceConfigurationError,
            match="customs_agent_evidence_retrieve_timeout_seconds_invalid",
        ):
            _settings(evidence_retrieve_timeout_seconds=invalid)


def test_temporal_context_is_atomic_and_bound_to_the_requested_law_date() -> None:
    response = {
        "temporal_context": {
            "law_as_of": "2026-08-05",
            "hsk_snapshot_as_of": "2026-08-05",
            "requirements_snapshot_as_of": "2026-08-05",
        }
    }

    context = _temporal_context_from_data(response, expected_law_as_of="2026-08-05")

    assert context is not None
    assert context.law_as_of.isoformat() == "2026-08-05"
    assert context.hsk_snapshot_as_of.isoformat() == "2026-08-05"
    assert context.requirements_snapshot_as_of.isoformat() == "2026-08-05"

    with pytest.raises(RetrievalError, match="does not match the review date"):
        _temporal_context_from_data(response, expected_law_as_of="2026-08-06")
    with pytest.raises(RetrievalError, match="temporal context is invalid"):
        _temporal_context_from_data(
            {"temporal_context": {"law_as_of": "2026-08-05"}},
            expected_law_as_of="2026-08-05",
        )


def test_temporal_coverage_is_atomic_and_never_inferred_from_snapshot_dates() -> None:
    response = {
        "temporal_context": {
            "law_as_of": "2026-08-05",
            "hsk_snapshot_as_of": "2026-08-05",
            "requirements_snapshot_as_of": "2026-08-05",
        },
        "temporal_coverage": {
            "law": "covered",
            "hsk": "unavailable",
            "requirements": "unavailable",
        },
    }

    coverage = _temporal_coverage_from_data(response, required=True)

    assert coverage is not None
    assert coverage.law == "covered"
    assert coverage.hsk == "unavailable"
    assert coverage.requirements == "unavailable"
    with pytest.raises(RetrievalError, match="temporal coverage is invalid"):
        _temporal_coverage_from_data(
            {
                "temporal_context": response["temporal_context"],
                "temporal_coverage": {"law": "covered"},
            },
            required=True,
        )
    with pytest.raises(RetrievalError, match="temporal coverage requires temporal context"):
        _temporal_coverage_from_data({"temporal_coverage": response["temporal_coverage"]})


def test_retriever_rejects_data_temporal_context_for_a_different_review_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ProfileEvidenceRetriever(_settings())
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    monkeypatch.setattr(
        retriever,
        "_request",
        lambda *_args, **_kwargs: {
            "schema_version": "3.0",
            "request_id": "rag-temporal-mismatch",
            "status": "partial",
            "knowledge_snapshot_id": "functional-uat-law-history",
            "temporal_context": {
                "law_as_of": "2026-08-06",
                "hsk_snapshot_as_of": "2026-08-05",
                "requirements_snapshot_as_of": "2026-08-05",
            },
            "results": [],
        },
    )

    with pytest.raises(RetrievalError, match="does not match the review date"):
        retriever.retrieve(
            {
                "review_run_id": "run-temporal-mismatch",
                "request_id": "rag-temporal-mismatch",
                "review_as_of": "2026-08-05",
            }
        )


def test_law_history_profile_requires_a_complete_temporal_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ProfileEvidenceRetriever(_settings(evidence_profile="functional-uat-law-history"))
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    monkeypatch.setattr(
        retriever,
        "_request",
        lambda *_args, **_kwargs: {
            "schema_version": "3.0",
            "request_id": "rag-history-context-required",
            "status": "partial",
            "knowledge_snapshot_id": "snapshot-history",
            "temporal_context": None,
            "results": [],
        },
    )

    with pytest.raises(RetrievalError, match="temporal context is required"):
        retriever.retrieve(
            {
                "review_run_id": "run-history-context-required",
                "request_id": "rag-history-context-required",
                "review_as_of": "2026-08-05",
            }
        )


def test_law_history_profile_requires_temporal_coverage_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ProfileEvidenceRetriever(_settings(evidence_profile="functional-uat-law-history"))
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    monkeypatch.setattr(
        retriever,
        "_request",
        lambda *_args, **_kwargs: {
            "schema_version": "3.0",
            "request_id": "rag-history-coverage-required",
            "status": "partial",
            "knowledge_snapshot_id": "snapshot-history",
            "temporal_context": {
                "law_as_of": "2026-08-05",
                "hsk_snapshot_as_of": "2026-08-05",
                "requirements_snapshot_as_of": "2026-08-05",
            },
            "temporal_coverage": None,
            "results": [],
        },
    )

    with pytest.raises(RetrievalError, match="temporal coverage is required"):
        retriever.retrieve(
            {
                "review_run_id": "run-history-coverage-required",
                "request_id": "rag-history-coverage-required",
                "review_as_of": "2026-08-05",
            }
        )


def test_legacy_evidence_profile_keeps_nullable_temporal_context_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ProfileEvidenceRetriever(_settings(evidence_profile="functional-uat-3law"))
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    monkeypatch.setattr(
        retriever,
        "_request",
        lambda *_args, **_kwargs: {
            "schema_version": "3.0",
            "request_id": "rag-legacy-null-context",
            "status": "partial",
            "knowledge_snapshot_id": "snapshot-3law",
            "temporal_context": None,
            "results": [],
        },
    )

    response = retriever.retrieve(
        {
            "review_run_id": "run-legacy-null-context",
            "request_id": "rag-legacy-null-context",
            "review_as_of": "2026-08-05",
        }
    )

    assert response["temporal_context"] is None


def test_evidence_consumer_accepts_rerank_but_rejects_unregistered_retrieval_methods() -> None:
    response = {
        "results": [
            {
                "evidence": [
                    {
                        "evidence_id": "ev_" + "a" * 64,
                        "item_id": "commodity-rerank",
                        "title": "법령 조문",
                        "text": "검토할 법령 근거입니다.",
                        "source_uri": "https://law.go.kr/example",
                        "source_type": "law_article",
                        "version_hash": "a" * 64,
                        "retrieval_methods": ["dense", "bm25", "rerank"],
                    }
                ]
            }
        ]
    }

    evidence = _evidence_from_data(response)

    assert evidence[0]["evidence_id"] == "ev_" + "a" * 64
    response["results"][0]["evidence"][0]["retrieval_methods"] = ["dense", "shell"]
    with pytest.raises(RetrievalError, match="evidence is invalid"):
        _evidence_from_data(response)


def test_retrieve_request_keeps_legacy_wire_and_enables_law_support_explicitly() -> None:
    request = ServiceReviewRunRequestV3.model_construct(
        case_id="case-retrieve-intent",
        review_run_id="run-retrieve-intent",
        review_as_of=date(2026, 8, 5),
        trade_type="import",
        jurisdiction="KR",
    )

    legacy = _retrieve_request(request, [])
    law_history = _retrieve_request(request, [], include_law_support=True)

    assert "query_intent" not in legacy
    assert law_history["query_intent"] == "required_import_documents_and_law_support"


def test_review_result_preserves_data_temporal_context_for_backend_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def retrieve(_self: ProfileEvidenceRetriever, request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "partial",
            "knowledge_snapshot_id": "functional-uat-law-history",
            "temporal_context": {
                "law_as_of": "2026-08-05",
                "hsk_snapshot_as_of": "2026-08-05",
                "requirements_snapshot_as_of": "2026-08-05",
            },
            "temporal_coverage": {
                "law": "covered",
                "hsk": "unavailable",
                "requirements": "unavailable",
            },
            "results": [
                {
                    "evidence": [
                        {
                            "evidence_id": "ev_" + "a" * 64,
                            "item_id": "commodity-law-history",
                            "title": "법령 조문",
                            "text": "검증된 법령 근거입니다.",
                            "source_uri": "https://www.law.go.kr/example",
                            "source_type": "law_article",
                            "version_hash": "a" * 64,
                            "retrieval_methods": ["dense", "bm25"],
                        }
                    ]
                }
            ],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", retrieve)
    runtime = ProfileRuntime(_settings(evidence_profile="functional-uat-law-history"))
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-law-history",
                "review_run_id": "run-law-history",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
                "documents": [
                    {
                        "document_id": "invoice",
                        "filename": "invoice.txt",
                        "content_text": (
                            "COMMERCIAL INVOICE\nHS Code: 123456\n"
                            "Description: Widget | HS Code: 123456 | Quantity: 1 PCS"
                        ),
                    }
                ],
            }
        ).encode()
    )

    assert result.execution_status == "partial"
    assert result.temporal_context is not None
    assert result.temporal_context.law_as_of.isoformat() == "2026-08-05"
    assert result.temporal_context.hsk_snapshot_as_of.isoformat() == "2026-08-05"
    assert result.temporal_context.requirements_snapshot_as_of.isoformat() == "2026-08-05"
    assert result.temporal_coverage is not None
    assert result.temporal_coverage.law == "covered"
    assert result.temporal_coverage.hsk == "unavailable"
    assert result.temporal_coverage.requirements == "unavailable"
    assert requests[0]["query_intent"] == "required_import_documents_and_law_support"
    assert [item.source_type for item in result.evidence] == ["law_article"]

    # Existing immutable 3-law snapshots did not have the additive field.
    # Their old hashes must remain readable when Backend sends one back for a
    # grounded follow-up conversation.
    legacy_wire = result.model_dump(mode="json", warnings=False)
    legacy_wire.pop("temporal_context")
    legacy_wire.pop("temporal_coverage")
    legacy_wire["result_sha256"] = review_result_v3_sha256(legacy_wire)
    legacy_result = ReviewRunResultV3.model_validate(legacy_wire)
    assert legacy_result.temporal_context is None

    mismatched_wire = result.model_dump(mode="json", warnings=False)
    assert mismatched_wire["temporal_context"] is not None
    mismatched_wire["temporal_context"]["law_as_of"] = "2026-08-06"
    mismatched_wire["result_sha256"] = review_result_v3_sha256(mismatched_wire)
    with pytest.raises(ValidationError, match="law_as_of must match review_as_of"):
        ReviewRunResultV3.model_validate(mismatched_wire)


def test_law_history_runtime_rejects_a_null_temporal_context_even_if_retriever_is_stubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def retrieve(_self: ProfileEvidenceRetriever, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "partial",
            "knowledge_snapshot_id": "snapshot-history",
            "temporal_context": None,
            "results": [],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", retrieve)
    runtime = ProfileRuntime(_settings(evidence_profile="functional-uat-law-history"))

    with pytest.raises(RuntimeExecutionError) as raised:
        runtime.review_v3_json(
            json.dumps(
                {
                    "case_id": "case-history-null-context",
                    "review_run_id": "run-history-null-context",
                    "review_as_of": "2026-08-05",
                    "trade_type": "import",
                    "jurisdiction": "KR",
                    "documents": [
                        {
                            "document_id": "invoice",
                            "filename": "invoice.txt",
                            "content_text": (
                                "COMMERCIAL INVOICE\nHS Code: 123456\n"
                                "Description: Widget | HS Code: 123456 | Quantity: 1 PCS"
                            ),
                        }
                    ],
                }
            ).encode()
        )

    assert raised.value.code == "evidence_dependency_unready"


def test_regulatory_requirement_is_one_finding_and_action_across_linked_documents() -> None:
    evidence = [
        {
            "evidence_id": "evidence-food-1",
            "item_id": "commodity-one",
            "title": "수입식품등 수입신고확인증",
            "eligibility": "finding",
            "law_id": "food-law",
            "required_document_code": "FOOD_IMPORT_CONFIRMATION",
            "required_document_name": "수입식품등 수입신고확인증",
        },
        {
            "evidence_id": "evidence-food-2",
            "item_id": "commodity-two",
            "title": "수입식품등 수입신고확인증",
            "eligibility": "finding",
            "law_id": "food-law",
            "required_document_code": "FOOD_IMPORT_CONFIRMATION",
            "required_document_name": "수입식품등 수입신고확인증",
        },
    ]

    deduped = _dedupe_evidence_records(evidence)
    findings = _findings_from_evidence(
        deduped,
        document_ids_by_item={
            "commodity-one": ["invoice", "packing"],
            "commodity-two": ["certificate"],
        },
    )
    actions = _actions(findings)

    assert len(deduped) == 1
    assert len(findings) == 1
    assert findings[0].document_ids == ["certificate", "invoice", "packing"]
    assert findings[0].evidence_ids == ["evidence-food-1"]
    regulatory_actions = [action for action in actions if action.finding_ids]
    assert len(regulatory_actions) == 1
    assert regulatory_actions[0].finding_ids == [findings[0].finding_id]


def test_deterministic_classifier_keeps_a_packing_header_over_a_later_bl_reference() -> None:
    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "packing",
            "filename": "packing.txt",
            "content_text": "PACKING LIST\nB/L No: BL-001",
        }
    )

    assert analysis.candidate.document_type == "packing_list"


def test_deterministic_extractor_does_not_treat_an_invoice_total_as_a_line_item() -> None:
    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "invoice",
            "filename": "invoice.txt",
            "content_text": (
                "COMMERCIAL INVOICE\n"
                "Total Amount: USD 30\n"
                "Description: Widget | HS Code: 123456 | Quantity: 3 PCS | "
                "Unit Price: USD 10 | Amount: USD 30"
            ),
        }
    )

    assert len(analysis.candidate.line_items) == 1


@pytest.mark.parametrize(
    ("document_id", "content_text", "expected_type", "expected_awb"),
    [
        (
            "packing-awb-number",
            "PACKING LIST\nPacking No: PL-AWB-001\nAWB Number: AWB-EXPORT-001",
            "packing_list",
            "AWB-EXPORT-001",
        ),
        (
            "air-waybill-awb-no",
            "AIR WAYBILL\nAWB No: AWB-EXPORT-001\nGross Weight: 100 KG",
            "air_waybill",
            "AWB-EXPORT-001",
        ),
    ],
)
def test_deterministic_extractor_recognizes_standard_awb_number_labels(
    document_id: str, content_text: str, expected_type: str, expected_awb: str
) -> None:
    """Packing and AWB documents retain the same source-grounded AWB value."""

    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": document_id,
            "filename": "transport.txt",
            "content_text": content_text,
        }
    )

    values = {field.name: field.value for field in analysis.candidate.fields}
    assert analysis.candidate.document_type == expected_type
    assert values["awb_no"] == expected_awb


def test_deterministic_extractor_keeps_missing_awb_as_a_reviewable_absence() -> None:
    """An AWB heading alone must not manufacture an identifier for comparison."""

    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "packing-with-awb-like-reference",
            "filename": "transport.txt",
            "content_text": "PACKING LIST\nPacking No: PL-AWB-001\nTotal Gross Weight: 100 KG",
        }
    )

    assert analysis.candidate.document_type == "packing_list"
    assert "awb_no" not in {field.name for field in analysis.candidate.fields}


def test_deterministic_extractor_supports_korean_export_and_submission_documents() -> None:
    declaration = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "export-declaration",
            "filename": "export.txt",
            "content_text": (
                "수출신고필증\n수출신고번호: ED-2026-001\n신고일자: 2026-08-05\n"
                "처리상태: 수리\n송장번호: INV-2026-001\n총액: USD 20\n원산지: KR\n"
                "목적국: US\n품명: Widget | HS 코드: 123456 | 수량: 2 PCS | 단위: PCS | 금액: USD 20"
            ),
        }
    )
    support = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "support",
            "filename": "support.txt",
            "content_text": "보완 증빙\n접수번호: RECEIPT-001\n제출일자: 2026-08-05\n처리상태: 승인",
        }
    )

    assert declaration.candidate.document_type == "export_declaration"
    assert {field.name for field in declaration.candidate.fields} >= {
        "export_declaration_no",
        "filing_date",
        "submission_status",
        "invoice_no",
        "total_amount",
        "origin_country",
        "destination_country",
    }
    assert declaration.candidate.line_items[0].fields[0].name == "description"
    assert support.candidate.document_type == "supporting_evidence"
    assert {field.name for field in support.candidate.fields} == {
        "submission_receipt_no",
        "filing_date",
        "submission_status",
    }


def test_backend_profile_loader_mode_is_accepted() -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
            "CUSTOMS_AGENT_SERVICE_MODE": "functional-uat",
            "CUSTOMS_AGENT_EVIDENCE_PROFILE": "functional-uat-law-history",
        }
    )
    assert settings.service_mode == "functional-uat"
    assert settings.evidence_profile == "functional-uat-law-history"
    assert settings.requires_temporal_context is True


@pytest.mark.parametrize(
    ("profile", "requires_temporal_context"),
    [
        ("dev-fixture", False),
        ("functional-uat-3law", False),
        ("functional-uat-law-history", True),
    ],
)
def test_evidence_profile_selection_is_explicit_and_nonsecret(
    profile: str, requires_temporal_context: bool
) -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
            "CUSTOMS_AGENT_EVIDENCE_PROFILE": profile,
        }
    )

    assert settings.evidence_profile == profile
    assert settings.requires_temporal_context is requires_temporal_context


def test_evidence_profile_rejects_unknown_value() -> None:
    with pytest.raises(ServiceConfigurationError) as raised:
        ServiceSettings.from_environment(
            {
                "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
                "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
                "EVIDENCE_SERVICE_TOKEN": _TOKEN,
                "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
                "CUSTOMS_AGENT_EVIDENCE_PROFILE": "unknown-profile",
            }
        )
    assert raised.value.code == "customs_agent_evidence_profile_invalid"


def test_functional_uat_requires_an_explicit_evidence_profile_binding() -> None:
    with pytest.raises(ServiceConfigurationError) as raised:
        ServiceSettings.from_environment(
            {
                "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
                "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
                "EVIDENCE_SERVICE_TOKEN": _TOKEN,
                "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
                "CUSTOMS_AGENT_SERVICE_MODE": "functional-uat",
            }
        )
    assert raised.value.code == "customs_agent_evidence_profile_missing"


def test_profile_rejects_non_loopback_evidence() -> None:
    with pytest.raises(RuntimeConfigurationError) as raised:
        ProfileEvidenceRetriever(replace(_settings(), evidence_base_url="https://example.invalid"))
    assert raised.value.code == "evidence_base_url_invalid"


@pytest.mark.parametrize(
    "environment",
    [
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "GUIDANCE_SERVICE_BASE_URL": "http://127.0.0.1:18093",
        },
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "GUIDANCE_SERVICE_TOKEN": _GUIDANCE_TOKEN,
        },
    ],
)
def test_guidance_peer_requires_a_complete_url_and_token_pair(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ServiceConfigurationError) as raised:
        ServiceSettings.from_environment(environment)
    assert raised.value.code == "guidance_service_peer_incomplete"


def test_guidance_registry_version_is_an_explicit_complete_peer_binding() -> None:
    base_environment = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
        "EVIDENCE_SERVICE_TOKEN": _TOKEN,
        "GUIDANCE_SERVICE_BASE_URL": "http://127.0.0.1:18094",
        "GUIDANCE_SERVICE_TOKEN": _GUIDANCE_TOKEN,
        "GUIDANCE_SERVICE_REGISTRY_VERSION": "2.0",
    }
    settings = ServiceSettings.from_environment(base_environment)
    assert settings.guidance_registry_version == "2.0"

    missing_version = dict(base_environment)
    del missing_version["GUIDANCE_SERVICE_REGISTRY_VERSION"]
    with pytest.raises(ServiceConfigurationError) as raised:
        ServiceSettings.from_environment(missing_version)
    assert raised.value.code == "guidance_registry_version_peer_incomplete"

    orphan_version = {
        key: value
        for key, value in base_environment.items()
        if not key.startswith("GUIDANCE_SERVICE_BASE_URL")
        and not key.startswith("GUIDANCE_SERVICE_TOKEN")
    }
    with pytest.raises(ServiceConfigurationError) as raised:
        ServiceSettings.from_environment(orphan_version)
    assert raised.value.code == "guidance_registry_version_peer_incomplete"


def test_profile_rejects_non_loopback_guidance_peer() -> None:
    settings = replace(
        _settings(),
        guidance_base_url="https://example.invalid",
        guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
    )
    with pytest.raises(RuntimeConfigurationError) as raised:
        ProfileEvidenceRetriever(settings)
    assert raised.value.code == "guidance_base_url_invalid"


@pytest.mark.parametrize(
    ("target", "legacy_topic"),
    sorted(_GUIDANCE_LEGACY_TOPIC_BY_SCOPE.items()),
)
def test_versioned_guidance_request_keeps_v2_scope_and_omits_it_for_v1(
    target: tuple[str, str], legacy_topic: str
) -> None:
    """A V1 request must be usable by Data's preserved scope-free endpoint."""

    topic, scope = target
    v1 = _guidance_query_request(
        conversation_id="conversation-v1-request",
        trade_type="import",
        as_of="2026-08-05",
        topic_hint=topic,
        scope_hint=scope,
        registry_version="1.0",
    )
    v2 = _guidance_query_request(
        conversation_id="conversation-v2-request",
        trade_type="import",
        as_of="2026-08-05",
        topic_hint=topic,
        scope_hint=scope,
        registry_version="2.0",
    )

    assert v1["topic_hint"] == legacy_topic
    assert v1["query"] == _GUIDANCE_LEGACY_SAFE_QUERY[legacy_topic]
    assert "scope_hint" not in v1
    assert v2["topic_hint"] == topic
    assert v2["scope_hint"] == scope
    assert v2["query"] == _GUIDANCE_SAFE_QUERY[target]
    # Both forms are built from code-owned labels only.  A question or a
    # conversation turn cannot be supplied to the builder.
    assert "사용자 원문" not in json.dumps({"v1": v1, "v2": v2}, ensure_ascii=False)


def test_v1_legacy_projection_covers_every_v2_target_once() -> None:
    assert set(_GUIDANCE_LEGACY_TOPIC_BY_SCOPE) == set(_GUIDANCE_SAFE_QUERY)
    assert set(_GUIDANCE_LEGACY_TOPIC_BY_SCOPE.values()) == set(_GUIDANCE_LEGACY_SAFE_QUERY)


@pytest.mark.parametrize("scope_hint", ["export_declaration", None])
def test_v1_retriever_rejects_an_additive_scope_hint_before_http_dispatch(
    monkeypatch: pytest.MonkeyPatch, scope_hint: str | None
) -> None:
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="1.0",
        )
    )
    request = _guidance_query_request(
        conversation_id="conversation-v1-invalid",
        trade_type="export",
        as_of="2026-08-05",
        topic_hint="customs_declaration",
        scope_hint="export_declaration",
        registry_version="1.0",
    )
    request["scope_hint"] = scope_hint
    dispatched = False

    def data_request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal dispatched
        dispatched = True
        return {}

    monkeypatch.setattr(retriever, "_request", data_request)
    with pytest.raises(RetrievalError, match="Data guidance request is invalid"):
        retriever.guidance_query(request)
    assert not dispatched


def test_v2_retriever_rejects_a_missing_scope_hint_before_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18094",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="2.0",
        )
    )
    request = _guidance_query_request(
        conversation_id="conversation-v2-invalid",
        trade_type="export",
        as_of="2026-08-05",
        topic_hint="customs_declaration",
        scope_hint="export_declaration",
        registry_version="2.0",
    )
    del request["scope_hint"]
    dispatched = False

    def data_request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal dispatched
        dispatched = True
        return {}

    monkeypatch.setattr(retriever, "_request", data_request)
    with pytest.raises(RetrievalError, match="Data guidance request is invalid"):
        retriever.guidance_query(request)
    assert not dispatched


def test_guidance_uses_its_own_loopback_origin_and_token_while_review_stays_on_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed profile must never redirect document review to guidance."""

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    calls: list[tuple[str, str | None]] = []
    guidance_bodies: list[dict[str, Any]] = []

    def open_request(request: Any, **_kwargs: Any) -> object:
        headers = {key.casefold(): value for key, value in request.header_items()}
        calls.append((request.full_url, headers.get("authorization")))
        if request.full_url.endswith(ProfileEvidenceRetriever.readiness_path):
            payload: dict[str, Any] = {"status": "ready"}
        elif request.full_url.endswith(ProfileEvidenceRetriever.retrieve_path):
            body = json.loads(request.data.decode("utf-8"))
            payload = {
                "schema_version": "3.0",
                "request_id": body["request_id"],
                "status": "ok",
                "knowledge_snapshot_id": "functional-uat-3law",
                "results": [],
            }
        else:
            body = json.loads(request.data.decode("utf-8"))
            guidance_bodies.append(body)
            # A V1 profile must never emit the additive V2 field.  The
            # retriever independently rejects it before HTTP dispatch.
            assert "scope_hint" not in body
            payload = {
                "schema_version": "3.0",
                "request_id": body["request_id"],
                "status": "covered",
                "snapshot_id": "functional-uat-kr-trade",
                "as_of": body["as_of"],
                "coverage": "covered",
                "registry_version": "1.0",
                "registry_sha256": "6fed779ee0ec1cd2358972925fe1b2788838a197d4f33e62e461fcfed2bb7d6c",
                "topic": body["topic_hint"],
                "results": [],
                "errors": [],
            }

        class Response:
            headers = Headers({SERVICE_MODE_HEADER: "functional-uat"})

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return json.dumps(payload).encode("utf-8")

        return Response()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="1.0",
        )
    )
    retriever.retrieve({"request_id": "evidence-request", "review_run_id": "review-run"})
    response = retriever.guidance_query(
        _guidance_query_request(
            conversation_id="guidance-request",
            trade_type="export",
            as_of="2026-08-05",
            topic_hint="customs_declaration",
            scope_hint="export_declaration",
            registry_version="1.0",
        )
    )

    assert calls == [
        ("http://127.0.0.1:18083/internal/rag/v3/health/ready", f"Bearer {_TOKEN}"),
        ("http://127.0.0.1:18083/internal/rag/v3/retrieve", f"Bearer {_TOKEN}"),
        ("http://127.0.0.1:18093/internal/rag/v3/guidance/query", f"Bearer {_GUIDANCE_TOKEN}"),
    ]
    assert response["status"] == "covered"
    assert guidance_bodies == [
        {
            "schema_version": "3.0",
            "request_id": guidance_bodies[0]["request_id"],
            "query": _GUIDANCE_LEGACY_SAFE_QUERY["customs_declaration"],
            "trade_direction": "export",
            "topic_hint": "customs_declaration",
            "as_of": "2026-08-05",
            "top_k": 3,
        }
    ]


def test_evidence_retrieve_uses_dedicated_timeout_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold Evidence model may take longer than readiness, never guidance."""

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    calls: list[tuple[str, str, float]] = []

    def open_request(request: Any, *, timeout: float) -> object:
        if request.full_url.endswith(ProfileEvidenceRetriever.readiness_path):
            kind = "readiness"
            payload: dict[str, Any] = {"status": "ready"}
        elif request.full_url.endswith(ProfileEvidenceRetriever.retrieve_path):
            kind = "retrieve"
            body = json.loads(request.data.decode("utf-8"))
            payload = {
                "schema_version": "3.0",
                "request_id": body["request_id"],
                "status": "ok",
                "knowledge_snapshot_id": "test-snapshot",
                "results": [],
            }
        else:
            kind = "guidance"
            payload = {}
        calls.append((kind, request.get_method(), timeout))

        class Response:
            def __init__(self, response_payload: dict[str, Any]) -> None:
                self.headers = Headers({SERVICE_MODE_HEADER: "local-development"})
                self._payload = response_payload

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return json.dumps(self._payload).encode("utf-8")

        return Response(payload)

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    retriever = ProfileEvidenceRetriever(_settings())
    retriever.retrieve(
        {"schema_version": "3.0", "request_id": "retrieve-timeout", "review_run_id": "run-timeout"}
    )
    retriever._request(
        ProfileEvidenceRetriever.guidance_query_path,
        method="POST",
        peer="guidance",
        payload={"request_id": "guidance-timeout"},
    )

    assert calls == [
        ("readiness", "GET", 5.0),
        ("retrieve", "POST", 90.0),
        ("guidance", "POST", 5.0),
    ]


def test_v1_runtime_uses_scope_free_legacy_request_and_v1_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the Agent HTTP boundary with the preserved V1 profile.

    The local response shape is deliberately the V1 public contract.  This
    checks the builder, retriever, response provenance, and deterministic
    answer path together without importing Data implementation code into this
    repository.
    """

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    registry = guidance_registry()
    observed_bodies: list[dict[str, Any]] = []

    def open_request(request: Any, **_kwargs: Any) -> object:
        body = json.loads(request.data.decode("utf-8"))
        observed_bodies.append(body)
        topic = body["topic_hint"]
        assert "scope_hint" not in body
        source_id = sorted(registry.topic_source_ids[topic])[0]
        payload: dict[str, Any] = {
            "schema_version": "3.0",
            "request_id": body["request_id"],
            "status": "covered",
            "snapshot_id": "dev-fixture-v1",
            "as_of": body["as_of"],
            "coverage": "covered",
            "registry_version": "1.0",
            "registry_sha256": registry.registry_sha256,
            "topic": topic,
            "results": [
                {
                    "source_id": "guide_" + "a" * 64,
                    "registry_source_id": source_id,
                    "source_kind": "development_fixture",
                    "title": registry.topic_titles[topic],
                    "uri": f"https://customs.local.invalid/dev-fixture/guidance/{topic}",
                    "topic": topic,
                    "direction": body["trade_direction"],
                    "excerpt": "등록된 공식 자료의 범위와 기준일을 확인해야 합니다.",
                    "version_hash": "a" * 64,
                    "retrieval_methods": ["registry"],
                }
            ],
            "errors": [],
        }

        class Response:
            headers = Headers({SERVICE_MODE_HEADER: "local-development"})

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return json.dumps(payload).encode("utf-8")

        return Response()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    runtime = ProfileRuntime(
        replace(
            _settings(),
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="1.0",
        )
    )
    question = "수입 통관 절차에서 기본으로 확인할 항목을 알려 주세요. 사용자원문표식"
    answer = runtime.knowledge_answer_json(
        json.dumps(
            {
                "schema_version": "1.0",
                "conversation_id": "knowledge-v1-http",
                "question": question,
                "trade_type": "import",
                "as_of": "2026-08-05",
                "recent_turns": [],
            }
        ).encode("utf-8")
    )

    assert answer.answer_scope == "general_customs_guidance"
    assert answer.coverage == "covered"
    assert answer.guidance_resolution is not None
    assert answer.guidance_resolution.registry_version == "1.0"
    assert answer.guidance_resolution.scope == "legacy_topic"
    assert answer.guidance_registry_sha256 == registry.registry_sha256
    assert len(observed_bodies) == 1
    assert "scope_hint" not in observed_bodies[0]
    assert observed_bodies[0]["topic_hint"] == "customs_clearance"
    assert observed_bodies[0]["query"] == _GUIDANCE_LEGACY_SAFE_QUERY["customs_clearance"]
    assert "사용자원문표식" not in json.dumps(observed_bodies[0], ensure_ascii=False)


def test_v2_guidance_requires_readiness_and_exact_configured_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str]] = []
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18094",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="2.0",
        )
    )

    def data_request(
        path: str,
        *,
        method: str,
        peer: str,
        payload: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((path, method, peer))
        if method == "GET":
            return {"status": "ready"}
        assert payload is not None
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "covered",
            "coverage": "covered",
            "registry_version": "2.0",
            "results": [],
        }

    monkeypatch.setattr(retriever, "_request", data_request)
    request = {
        "schema_version": "3.0",
        "request_id": "guidance-v2-ready",
        "query": "대한민국 수출 신고 절차 공식 안내",
        "trade_direction": "export",
        "topic_hint": "customs_declaration",
        "scope_hint": "export_declaration",
        "as_of": "2026-08-05",
        "top_k": 3,
    }
    retriever.guidance_query(request)
    retriever.guidance_query({**request, "request_id": "guidance-v2-ready-second"})

    assert calls == [
        (ProfileEvidenceRetriever.readiness_path, "GET", "guidance"),
        (ProfileEvidenceRetriever.guidance_query_path, "POST", "guidance"),
        (ProfileEvidenceRetriever.guidance_query_path, "POST", "guidance"),
    ]


@pytest.mark.parametrize(
    ("readiness", "registry_version", "expected_calls"),
    [
        ({"status": "not_ready"}, "2.0", 1),
        ({"status": "ready"}, "1.0", 2),
    ],
)
def test_v2_guidance_fails_closed_when_readiness_or_generation_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
    readiness: dict[str, str],
    registry_version: str,
    expected_calls: int,
) -> None:
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18094",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="2.0",
        )
    )
    calls = 0

    def data_request(
        _path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if method == "GET":
            return readiness
        assert payload is not None
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "covered",
            "coverage": "covered",
            "registry_version": registry_version,
            "results": [],
        }

    monkeypatch.setattr(retriever, "_request", data_request)
    with pytest.raises(RetrievalError):
        retriever.guidance_query(
            {
                "schema_version": "3.0",
                "request_id": "guidance-v2-fail",
                "query": "대한민국 수출 신고 절차 공식 안내",
                "trade_direction": "export",
                "topic_hint": "customs_declaration",
                "scope_hint": "export_declaration",
                "as_of": "2026-08-05",
                "top_k": 3,
            }
        )
    assert calls == expected_calls


@pytest.mark.parametrize("failure", ["mode", "token"])
def test_guidance_peer_mode_or_token_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A separate peer cannot degrade into a successful general answer."""

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    def open_request(request: Any, **_kwargs: Any) -> object:
        if failure == "token":
            assert request.get_header("Authorization") == f"Bearer {_GUIDANCE_TOKEN}"
            raise HTTPError(request.full_url, 401, "", Headers(), None)

        class Response:
            headers = Headers({SERVICE_MODE_HEADER: "local-development"})

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return b"{}"

        return Response()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
            guidance_registry_version="1.0",
        )
    )
    with pytest.raises(RetrievalError):
        retriever.guidance_query(
            {
                "schema_version": "3.0",
                "request_id": "guidance-failed",
                "query": "대한민국 수출 신고 절차 공식 안내",
                "trade_direction": "export",
                "topic_hint": "customs_declaration",
                "scope_hint": "export_declaration",
                "as_of": "2026-08-05",
                "top_k": 3,
            }
        )


def test_review_input_manifest_hashes_the_exact_backend_wire_for_baseline_and_child_runs() -> None:
    baseline_wire = {
        "case_id": "case-input-manifest",
        "review_run_id": "run-input-manifest",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": [
            {
                "document_id": "document-input-manifest",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-001",
            }
        ],
    }
    baseline = ServiceReviewRunRequestV3.model_validate(baseline_wire)
    assert baseline.input_manifest_wire() == baseline_wire
    assert canonical_json_sha256(baseline.input_manifest_wire()) == canonical_json_sha256(
        baseline_wire
    )

    child_wire = {
        **baseline_wire,
        "review_run_id": "run-input-manifest-child",
        "parent_review_run_id": "run-input-manifest",
        "applied_correction_proposal_ids": ["correction-input-manifest"],
        "human_corrections": [
            {
                "correction_proposal_id": "correction-input-manifest",
                "document_id": "document-input-manifest",
                "field_name": "invoice_number",
                "new_value": "INV-001-CORRECTED",
                "reason": "Reviewer confirmed the text document.",
                "source_kind": "human_provided",
            }
        ],
    }
    child = ServiceReviewRunRequestV3.model_validate(child_wire)
    assert child.input_manifest_wire() == child_wire
    assert canonical_json_sha256(child.input_manifest_wire()) == canonical_json_sha256(child_wire)


def test_review_uses_one_readiness_then_one_retrieve_and_stays_human_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def data_request(
        _self: ProfileEvidenceRetriever,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        review_run_id: str | None = None,
    ) -> dict[str, Any]:
        assert review_run_id == "run-profile-v3"
        calls.append((method, path))
        if path == ProfileEvidenceRetriever.readiness_path:
            assert method == "GET"
            assert payload is None
            return {"status": "ready"}
        assert path == ProfileEvidenceRetriever.retrieve_path
        assert method == "POST"
        assert payload is not None
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "_request", data_request)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-profile-v3",
                "documents": [
                    {
                        "document_id": "doc-profile-v3",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-003",
                    }
                ],
                "review_run_id": "run-profile-v3",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )
    assert result.schema_version == "3.2"
    assert result.status == "needs_review"
    assert result.requires_human_review is True
    assert result.engine is not None and result.engine.kind == "deterministic"
    assert any(action.action_code == "human_review_required" for action in result.required_actions)
    assert [stage.stage for stage in result.workflow.stages] == [
        "classify_documents",
        "extract_facts",
        "normalize_facts",
        "match_documents",
        "validate_documents",
        "reextract_documents",
        "retrieve_evidence",
        "generate_result",
    ]
    assert result.document_analyses[0].document_id == "doc-profile-v3"
    assert result.workflow.planner.cycle_count <= 2
    assert len(result.workflow.planner.actions) <= 6
    assert {section.section for section in result.report.sections} == {
        "conclusion",
        "execution_process",
        "document_comparison",
        "issues_and_causes",
        "legal_evidence",
        "impact",
        "next_actions",
        "limitations",
    }
    # No HS-qualified commodity exists in this deliberately incomplete
    # invoice.  Sending an empty or no-HS fallback to Data would manufacture a
    # partial Data status, so the bounded workflow records an input-side skip.
    assert calls == []
    retrieval = next(
        stage for stage in result.workflow.stages if stage.stage == "retrieve_evidence"
    )
    assert retrieval.status == "skipped"
    legal_evidence = next(
        section for section in result.report.sections if section.section == "legal_evidence"
    )
    assert "HS 코드가 확인된 품목이 없어" in legal_evidence.claims[0].text
    assert "Data" not in legal_evidence.claims[0].text


def test_reextraction_failure_is_exposed_as_partial_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Analyzer:
        model_id = "test-reextract-failure"

        def __init__(self) -> None:
            self.calls = 0
            self._fallback = _DeterministicDocumentAnalyzer()

        def analyze(self, document: dict[str, Any], **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("controlled reextract failure")
            return self._fallback.analyze(document)

    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    runtime = ProfileRuntime(_settings())
    analyzer = Analyzer()
    runtime._analyzer = analyzer  # type: ignore[assignment]

    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-reextract-failure",
                "documents": [
                    {
                        "document_id": "doc-reextract-failure",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-RETRY",
                    }
                ],
                "review_run_id": "run-reextract-failure",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )

    assert analyzer.calls == 2
    assert result.workflow.status == "partial"
    assert result.execution_status == "partial"


def test_internal_api_requires_matching_mode_header() -> None:
    class Runtime:
        def review_v3_json(self, _payload: bytes) -> object:
            raise AssertionError("review is not part of this test")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        missing = client.get(
            "/internal/agent/v2/metadata", headers={"Authorization": f"Bearer {_TOKEN}"}
        )
        accepted = client.get(
            "/internal/agent/v2/metadata",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
        )
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "service_mode_mismatch"
    assert accepted.status_code == 200
    assert accepted.json()["human_review_only"] is True


def test_http_knowledge_rejects_document_material_before_runtime_execution() -> None:
    """The public Agent route must not enqueue a pasted Case/document payload."""

    calls = 0

    class Runtime:
        def knowledge_answer_json(self, _payload: bytes) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("knowledge runtime must not receive rejected input")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    unsafe_question = "상업송장\n송장번호: INV-001"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/knowledge/answers",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "schema_version": "1.0",
                "conversation_id": "knowledge-boundary",
                "question": unsafe_question,
                "trade_type": "both",
                "as_of": "2026-08-05",
                "recent_turns": [],
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "knowledge_input_boundary"
    assert "일반적인 통관 지식 질문" in response.json()["error"]["message"]
    assert unsafe_question not in response.text
    assert calls == 0


def test_http_knowledge_rejects_document_material_in_recent_turns_before_execution() -> None:
    """History is an input boundary too, not a trusted provider context."""

    calls = 0

    class Runtime:
        def knowledge_answer_json(self, _payload: bytes) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("knowledge runtime must not receive rejected history")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    unsafe_history = "원산지증명서\nHS 코드: 0303540000"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/knowledge/answers",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "schema_version": "1.0",
                "conversation_id": "knowledge-history-boundary",
                "question": "HS 품목분류의 기본 절차를 알려 주세요.",
                "trade_type": "both",
                "as_of": "2026-08-05",
                "recent_turns": [{"role": "user", "content": unsafe_history}],
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "knowledge_input_boundary"
    assert unsafe_history not in response.text
    assert calls == 0


def test_http_review_keeps_human_only_v3_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "case_id": "case-profile-http",
                "documents": [
                    {
                        "document_id": "doc-profile-http",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-004",
                    }
                ],
                "review_run_id": "run-profile-http",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["requires_human_review"] is True


def test_http_review_accepts_bounded_child_correction_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-valid child run must pass HTTP admission before the runtime."""

    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "case_id": "case-profile-child",
                "documents": [
                    {
                        "document_id": "invoice",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-005",
                    }
                ],
                "review_run_id": "run-profile-child",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
                "parent_review_run_id": "run-profile-parent",
                "applied_correction_proposal_ids": ["proposal-origin"],
                "human_corrections": [
                    {
                        "correction_proposal_id": "proposal-origin",
                        "document_id": "invoice",
                        "field_name": "origin_country",
                        "new_value": "KR",
                        "reason": "승인된 원산지 보완값입니다.",
                        "source_kind": "human_provided",
                    }
                ],
            },
        )

    assert response.status_code == 200
    actions = response.json()["workflow"]["planner"]["actions"]
    correction = next(action for action in actions if action["action"] == "apply_human_corrections")
    assert correction["correction_proposal_ids"] == ["proposal-origin"]


def test_law_search_proxies_a_bounded_reviewer_query(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = ProfileEvidenceRetriever(_settings())
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    captured: dict[str, object] = {}

    def fake_request(path, *, method, payload=None, **_kwargs):
        captured["path"] = path
        captured["method"] = method
        captured["payload"] = payload
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "ok",
            "as_of": payload["as_of"],
            "results": [],
        }

    monkeypatch.setattr(retriever, "_request", fake_request)
    response = retriever.law_search(
        {
            "schema_version": "3.0",
            "request_id": "law-search-1",
            "query": "원산지증명서 제출",
            "top_k": 10,
            "as_of": "2026-08-05",
        }
    )
    assert captured["path"] == "/internal/rag/v3/law-search"
    assert captured["method"] == "POST"
    assert response["status"] == "ok"


def test_law_search_rejects_an_invalid_request() -> None:
    retriever = ProfileEvidenceRetriever(_settings())
    base = {
        "schema_version": "3.0",
        "request_id": "law-search-2",
        "query": "관세",
        "top_k": 10,
        "as_of": "2026-08-05",
    }
    for broken in (
        {**base, "query": ""},
        {**base, "query": "가" * 201},
        {**base, "top_k": 50},
        {**base, "request_id": "bad id!"},
        {**base, "as_of": 20260805},
    ):
        with pytest.raises(RetrievalError, match="law-search request is invalid"):
            retriever.law_search(broken)


def test_law_search_rejects_a_contract_breaking_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = ProfileEvidenceRetriever(_settings())
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    monkeypatch.setattr(
        retriever,
        "_request",
        lambda *_args, **_kwargs: {
            "schema_version": "3.0",
            "request_id": "law-search-3",
            "status": "failed",
            "results": [],
        },
    )
    with pytest.raises(RetrievalError, match="law-search response"):
        retriever.law_search(
            {
                "schema_version": "3.0",
                "request_id": "law-search-3",
                "query": "관세",
                "top_k": 5,
                "as_of": "2026-08-05",
            }
        )


def test_law_catalog_proxies_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = ProfileEvidenceRetriever(_settings())
    monkeypatch.setattr(retriever, "readiness", lambda **_kwargs: {"status": "ready"})
    monkeypatch.setattr(
        retriever,
        "_request",
        lambda path, **_kwargs: {
            "schema_version": "3.0",
            "request_id": "law-catalog-1",
            "status": "ok",
            "as_of": "2026-08-05",
            "laws": [
                {
                    "law_id": "LAW-226",
                    "law_name": "관세법",
                    "hierarchy": "법률",
                    "article_count": 280,
                }
            ],
        },
    )
    response = retriever.law_catalog(
        {"schema_version": "3.0", "request_id": "law-catalog-1", "as_of": "2026-08-05"}
    )
    assert response["laws"][0]["law_name"] == "관세법"
    with pytest.raises(RetrievalError, match="law-catalog request is invalid"):
        retriever.law_catalog({"schema_version": "3.0", "request_id": "law-catalog-2"})
