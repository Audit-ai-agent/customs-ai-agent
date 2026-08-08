from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.schemas.assistant import (
    AssistantAnswerRequestV1,
    GENERAL_GUIDANCE_DISCLAIMER,
    KnowledgeAnswerRequestV1,
)
from customs_ai_agent.schemas.v3 import EngineV3
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.assistant import (
    deterministic_knowledge_answer,
    guidance_context_from_data,
    guidance_topic_for_question,
    response_from_knowledge_provider_candidate,
    response_from_provider_candidate,
)
from customs_ai_agent.service.guidance_registry import guidance_registry
from customs_ai_agent.service.profiles import (
    ProfileEvidenceRetriever,
    ProfileRuntime,
    SERVICE_MODE_HEADER,
)
from customs_ai_agent.service.runtime_errors import RuntimeInputError
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.user_text import (
    contains_internal_result_marker,
    reviewer_text_is_safe,
)


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
_REGISTRY_SHA256 = "6fed779ee0ec1cd2358972925fe1b2788838a197d4f33e62e461fcfed2bb7d6c"
_FIXTURE_SOURCE_BY_TOPIC = {
    "customs_clearance": "law_001556",
    "customs_declaration": "law_001556",
    "customs_duty": "law_001556",
    "hs_classification": "law_001556",
    "origin_verification": "law_001556",
    "trade_documents": "law_001556",
    "import_requirements": "law_001556",
    "export_requirements": "law_kr_trade_fta_scope",
    "transport_documents": "kcs_customs_guidance",
    "fta_procedure": "law_kr_trade_fta_scope",
    "trade_terms": "kotra_trade_guidance",
}
_FIXTURE_TITLE_BY_TOPIC = {
    "customs_clearance": "통관 절차의 기본 안내",
    "customs_declaration": "수출입 신고의 기본 안내",
    "customs_duty": "관세와 과세가격의 기본 안내",
    "hs_classification": "HS 품목분류의 기본 안내",
    "origin_verification": "원산지 확인의 기본 안내",
    "trade_documents": "무역 서류의 역할 안내",
    "import_requirements": "수입 요건 확인의 기본 안내",
    "export_requirements": "수출 요건 확인의 기본 안내",
    "transport_documents": "운송·항공운송 서류의 기본 안내",
    "fta_procedure": "FTA 절차의 기본 안내",
    "trade_terms": "무역 거래조건의 기본 안내",
}


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_TOKEN),
        ),
        **changes,
    )


def _stub_retrieve(_self: ProfileEvidenceRetriever, request: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "request_id": request["request_id"],
        "status": "ok",
        "knowledge_snapshot_id": "dev-fixture",
        "results": [
            {
                "item_id": request["items"][0]["item_id"],
                "evidence": [
                    {
                        "evidence_id": "evidence-article-1",
                        "item_id": request["items"][0]["item_id"],
                        "title": "관세법 확인 항목",
                        "text": "원산지와 품목 정보를 확인해야 합니다.",
                        "source_uri": "https://law.example.invalid/1",
                        "source_type": "law",
                        "version_hash": "a" * 64,
                    }
                ],
            }
        ],
    }


def _stub_guidance(
    _self: ProfileEvidenceRetriever, request: dict[str, Any], **_kwargs: Any
) -> dict[str, Any]:
    topic = str(request.get("topic_hint", "customs_clearance"))
    return {
        "schema_version": "3.0",
        "request_id": request["request_id"],
        "status": "covered",
        "snapshot_id": "dev-fixture-guidance",
        "as_of": request["as_of"],
        "coverage": "covered",
        "registry_version": "1.0",
        "registry_sha256": _REGISTRY_SHA256,
        "topic": topic,
        "results": [
            {
                "source_id": "guide_" + "a" * 64,
                "registry_source_id": _FIXTURE_SOURCE_BY_TOPIC[topic],
                "source_kind": "development_fixture",
                "title": _FIXTURE_TITLE_BY_TOPIC[topic],
                "uri": f"https://customs.local.invalid/dev-fixture/guidance/{topic}",
                "topic": topic,
                "direction": request.get("trade_direction", "both"),
                "excerpt": "통관은 신고 내용과 관련 서류를 확인하는 절차입니다.",
                "version_hash": "a" * 64,
                "retrieval_methods": ["registry"],
            }
        ],
        "errors": [],
    }


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda response: response.__setitem__("as_of", "2026-08-06"), "as_of_mismatch"),
        (lambda response: response.__setitem__("coverage", "partial"), "status_coverage"),
        (
            lambda response: response["results"][0].__setitem__("source_id", "guide-invalid"),
            "source_invalid",
        ),
        (
            lambda response: response["results"][0].__setitem__("retrieval_methods", ["shell"]),
            "source_invalid",
        ),
        (
            lambda response: response["results"][0].__setitem__("direction", "import"),
            "direction_mismatch",
        ),
        (
            lambda response: response.__setitem__("topic", "trade_documents"),
            "topic_mismatch",
        ),
        (
            lambda response: response["results"][0].__setitem__("law_id", "001556"),
            "source_invalid",
        ),
    ],
)
def test_guidance_context_rejects_semantically_invalid_data_response(
    mutate: Any, error: str
) -> None:
    request = {
        "request_id": "guide-test",
        "as_of": "2026-08-05",
    }
    response = deepcopy(_stub_guidance(None, request))
    mutate(response)

    with pytest.raises(ValueError, match=error):
        guidance_context_from_data(
            response,
            profile_mode="local-development",
            expected_as_of=date(2026, 8, 5),
            expected_trade_direction="export",
            expected_topic="customs_clearance",
        )


def test_guidance_context_accepts_registered_official_agency_source() -> None:
    request = {"request_id": "guide-test", "as_of": "2026-08-05"}
    response = deepcopy(_stub_guidance(None, request))
    response["results"][0].update(
        {
            "registry_source_id": "kcs_customs_guidance",
            "source_kind": "official_agency_guidance",
            "title": "관세청 수출입 통관 공개 안내",
            "uri": "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?cntntsId=817&mi=2819",
        }
    )

    context = guidance_context_from_data(
        response,
        profile_mode="local-development",
        expected_as_of=date(2026, 8, 5),
        expected_trade_direction="both",
        expected_topic="customs_clearance",
    )

    assert context.evidence[0].source_kind == "official_agency_guidance"


def _official_law_guidance() -> dict[str, Any]:
    response = _stub_guidance(
        None,
        {
            "request_id": "guide-law",
            "as_of": "2026-08-05",
            "topic_hint": "customs_clearance",
            "trade_direction": "both",
        },
    )
    response["results"][0].update(
        {
            "registry_source_id": "law_001556",
            "source_kind": "official_law",
            "title": "관세법 제1조",
            "uri": "https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EA%B4%80%EC%84%B8%EB%B2%95/%EC%A0%9C1%EC%A1%B0",
            "law_id": "001556",
            "law_name": "관세법",
            "article_no": "제1조",
        }
    )
    return response


def test_guidance_registry_consumer_loads_the_expected_safe_manifest() -> None:
    registry = guidance_registry()

    assert registry.registry_sha256 == _REGISTRY_SHA256
    assert registry.sources_by_id["law_001556"].law_id == "001556"
    assert registry.topic_titles["customs_clearance"] == "통관 절차의 기본 안내"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response.__setitem__("registry_sha256", "0" * 64),
        lambda response: response["results"][0].__setitem__(
            "registry_source_id", "not_a_registry_source"
        ),
        lambda response: response["results"][0].__setitem__("law_id", "002421"),
        lambda response: response["results"][0].__setitem__("law_name", "관세법 시행령"),
        lambda response: response["results"][0].__setitem__("title", "관세법 시행령 제1조"),
        lambda response: response["results"][0].__setitem__(
            "uri", "https://law.example.invalid/법령/관세법/제1조"
        ),
        lambda response: response["results"][0].__setitem__(
            "uri", "https://www.law.go.kr/법령/관세법시행령/제1조"
        ),
        lambda response: response["results"][0].__setitem__(
            "uri", "https://www.law.go.kr/법령/관세법/제1조/other"
        ),
        lambda response: response["results"][0].__setitem__(
            "uri", "https://www.law.go.kr:443/법령/관세법/제1조"
        ),
        lambda response: response["results"][0].__setitem__(
            "uri", "https://www.law.go.kr:/법령/관세법/제1조"
        ),
    ],
)
def test_guidance_context_rejects_unregistered_or_noncanonical_law_provenance(
    mutate: Any,
) -> None:
    response = _official_law_guidance()
    mutate(response)

    with pytest.raises(ValueError, match="provenance_invalid"):
        guidance_context_from_data(
            response,
            profile_mode="functional-uat",
            expected_as_of=date(2026, 8, 5),
            expected_trade_direction="both",
            expected_topic="customs_clearance",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda response: response["results"][0].__setitem__("title", "변조된 기관 안내"),
        lambda response: response["results"][0].__setitem__(
            "uri", "https://www.customs.go.kr/kcs/other"
        ),
        lambda response: response["results"][0].__setitem__(
            "source_kind", "official_public_guidance"
        ),
    ],
)
def test_guidance_context_requires_exact_registered_nonlaw_metadata(mutate: Any) -> None:
    response = _stub_guidance(
        None,
        {
            "request_id": "guide-agency",
            "as_of": "2026-08-05",
            "topic_hint": "customs_clearance",
            "trade_direction": "both",
        },
    )
    response["results"][0].update(
        {
            "registry_source_id": "kcs_customs_guidance",
            "source_kind": "official_agency_guidance",
            "title": "관세청 수출입 통관 공개 안내",
            "uri": "https://www.customs.go.kr/kcs/cm/cntnts/cntntsView.do?cntntsId=817&mi=2819",
        }
    )
    mutate(response)

    with pytest.raises(ValueError, match="provenance_invalid"):
        guidance_context_from_data(
            response,
            profile_mode="functional-uat",
            expected_as_of=date(2026, 8, 5),
            expected_trade_direction="both",
            expected_topic="customs_clearance",
        )


def test_development_fixture_is_never_accepted_by_functional_uat() -> None:
    response = _stub_guidance(
        None,
        {
            "request_id": "guide-fixture",
            "as_of": "2026-08-05",
            "topic_hint": "customs_clearance",
            "trade_direction": "both",
        },
    )

    with pytest.raises(ValueError, match="provenance_invalid"):
        guidance_context_from_data(
            response,
            profile_mode="functional-uat",
            expected_as_of=date(2026, 8, 5),
            expected_trade_direction="both",
            expected_topic="customs_clearance",
        )


def test_guidance_context_rejects_a_direction_outside_the_registered_topic_scope() -> None:
    response = _stub_guidance(
        None,
        {
            "request_id": "guide-direction",
            "as_of": "2026-08-05",
            "topic_hint": "import_requirements",
            "trade_direction": "export",
        },
    )

    with pytest.raises(ValueError, match="provenance_invalid"):
        guidance_context_from_data(
            response,
            profile_mode="local-development",
            expected_as_of=date(2026, 8, 5),
            expected_trade_direction="export",
            expected_topic="import_requirements",
        )


def test_knowledge_provider_can_select_only_code_owned_presentation() -> None:
    context = guidance_context_from_data(
        _official_law_guidance(),
        profile_mode="functional-uat",
        expected_as_of=date(2026, 8, 5),
        expected_trade_direction="both",
        expected_topic="customs_clearance",
    )
    request = KnowledgeAnswerRequestV1(
        conversation_id="knowledge-template",
        question="일반 통관 절차를 알려 주세요.",
        trade_type="both",
        as_of=date(2026, 8, 5),
    )
    engine = EngineV3(kind="claude", model_id="claude-haiku")

    answer = response_from_knowledge_provider_candidate(
        {
            "presentation_style": "source_focus",
            "source_slots": [0],
            "suggestion_topics": ["trade_documents"],
        },
        request,
        engine=engine,
        context=context,
    )

    assert answer.direct_answer.startswith("수출입 통관·신고 절차")
    assert "공식 자료 발췌" in answer.direct_answer
    assert "관세법 제1조" in answer.direct_answer
    assert answer.suggested_questions == ["상업송장과 포장명세서는 어떤 역할을 하나요?"]
    assert answer.answer_sections[0].content == answer.direct_answer

    with pytest.raises(ValueError, match="knowledge_provider_output_invalid"):
        response_from_knowledge_provider_candidate(
            {
                "presentation_style": "overview",
                "source_slots": [0],
                "suggestion_topics": [],
                "direct_answer": "Claude가 만든 임의의 법률 판단",
            },
            request,
            engine=engine,
            context=context,
        )


def test_deterministic_knowledge_template_uses_only_data_source_fields() -> None:
    context = guidance_context_from_data(
        _official_law_guidance(),
        profile_mode="functional-uat",
        expected_as_of=date(2026, 8, 5),
        expected_trade_direction="both",
        expected_topic="customs_clearance",
    )
    answer = deterministic_knowledge_answer(
        KnowledgeAnswerRequestV1(
            conversation_id="knowledge-deterministic",
            question="통관 절차의 기본을 알려 주세요.",
            as_of=date(2026, 8, 5),
        ),
        engine=EngineV3(kind="deterministic", model_id="deterministic"),
        context=context,
    )

    assert answer.answer_scope == "general_customs_guidance"
    assert "확인 절차" in answer.direct_answer
    assert "공식 자료 발췌" in answer.direct_answer
    assert GENERAL_GUIDANCE_DISCLAIMER in answer.direct_answer


def _review_payload() -> dict[str, Any]:
    return {
        "case_id": "case-assistant",
        "review_run_id": "run-assistant",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": [
            {
                "document_id": "invoice",
                "filename": "invoice.txt",
                "content_text": (
                    "COMMERCIAL INVOICE\nInvoice No: INV-009\nHS Code: 123456\n"
                    "Total Amount: USD 10\nDescription: Widget | HS Code: 123456 | "
                    "Quantity: 1 PCS | Unit Price: USD 10 | Amount: USD 10"
                ),
            }
        ],
    }


def _matching_review_payload() -> dict[str, Any]:
    return {
        "case_id": "case-matching",
        "review_run_id": "run-matching",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": [
            {
                "document_id": "invoice",
                "filename": "matching-invoice.txt",
                "content_text": (
                    "COMMERCIAL INVOICE\nInvoice No: INV-MATCH-001\n"
                    "Total Amount: USD 20\nCountry of Origin: KR\n"
                    "Description: Frozen Mackerel | HS Code: 0303540000 | Quantity: 2 PCS | "
                    "Unit Price: USD 10 | Amount: USD 20"
                ),
            },
            {
                "document_id": "packing",
                "filename": "matching-packing.txt",
                "content_text": (
                    "PACKING LIST\nPacking No: PL-MATCH-001\nB/L No: BL-MATCH-001\n"
                    "Container No: ABCD1234567\nTotal Gross Weight: 100 KG\n"
                    "Description: Frozen Mackerel | HS Code: 0303540000 | Quantity: 2 PCS | "
                    "Unit: PCS | Gross Weight: 100 KG"
                ),
            },
            {
                "document_id": "certificate",
                "filename": "matching-origin.txt",
                "content_text": (
                    "CERTIFICATE OF ORIGIN\nCertificate No: COO-MATCH-001\n"
                    "Origin Country: KR\nHS Code: 0303540000"
                ),
            },
            {
                "document_id": "bill",
                "filename": "matching-bill.txt",
                "content_text": (
                    "BILL OF LADING\nB/L No: BL-MATCH-001\nContainer No: ABCD1234567\n"
                    "Gross Weight: 100 KG"
                ),
            },
        ],
    }


def _answer_payload(
    result: Any, *, question: str, recent_turns: list[dict[str, str]] | None = None
) -> bytes:
    return json.dumps(
        {
            "schema_version": "1.0",
            "case_id": result.case_id,
            "review_run_id": result.review_run_id,
            "conversation_id": "conversation-assistant",
            "review_result": result.model_dump(mode="json"),
            "question": question,
            "recent_turns": recent_turns or [],
        }
    ).encode()


def test_deterministic_assistant_scopes_citations_and_turn_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    monkeypatch.setattr(ProfileEvidenceRetriever, "guidance_query", _stub_guidance)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    case_answer = runtime.assistant_answer_json(
        _answer_payload(result, question="현재 문서 비교 결과를 설명해 주세요.")
    )
    assert case_answer.answer_scope == "case_grounded"
    assert case_answer.citations and case_answer.citations[0].kind == "comparison"

    evidence_answer = runtime.assistant_answer_json(
        _answer_payload(result, question="연결된 법령 근거는 무엇인가요?")
    )
    assert evidence_answer.answer_scope == "evidence_grounded"
    assert evidence_answer.citations[0].reference_id == "evidence-article-1"
    assert all(
        "Evidence" not in claim.text and "Review Run" not in claim.text
        for section in result.report.sections
        for claim in section.claims
    )

    case_evidence_answer = runtime.assistant_answer_json(
        _answer_payload(result, question="현재 사건의 법령 근거를 보여 주세요.")
    )
    assert case_evidence_answer.answer_scope == "evidence_grounded"
    assert len(case_evidence_answer.citations) == 1
    assert case_evidence_answer.citations[0].kind == "evidence"
    assert case_evidence_answer.citations[0].reference_id == "evidence-article-1"

    general_evidence_concept = runtime.assistant_answer_json(
        _answer_payload(result, question="일반적인 법령 근거의 개념을 설명해 주세요.")
    )
    assert general_evidence_concept.answer_scope == "out_of_scope"
    assert general_evidence_concept.citations == []

    general_answer = runtime.assistant_answer_json(
        _answer_payload(result, question="일반 통관 절차를 안내해 주세요.")
    )
    assert general_answer.answer_scope == "general_customs_guidance"
    assert len(general_answer.citations) == 1
    assert general_answer.citations[0].kind == "guidance_evidence"
    assert general_answer.answer_sections[0].kind == "general_guidance"
    assert GENERAL_GUIDANCE_DISCLAIMER in general_answer.direct_answer

    unsupported_general = runtime.assistant_answer_json(
        _answer_payload(result, question="일반 관세 환급 전략을 알려 주세요.")
    )
    assert unsupported_general.answer_scope == "insufficient_evidence"

    out_of_scope = runtime.assistant_answer_json(
        _answer_payload(result, question="웹 검색으로 다른 법령도 찾아 주세요.")
    )
    assert out_of_scope.answer_scope == "out_of_scope"

    too_many = [{"role": "user", "content": f"질문 {index}"} for index in range(9)]
    with pytest.raises(RuntimeInputError):
        runtime.assistant_answer_json(
            _answer_payload(result, question="현재 비교를 설명해 주세요.", recent_turns=too_many)
        )


def test_knowledge_answer_is_case_isolated_and_uses_one_official_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guidance_calls: list[dict[str, Any]] = []

    def guidance(
        self: ProfileEvidenceRetriever, request: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        del self, kwargs
        guidance_calls.append(request)
        return _stub_guidance(None, request)

    def review_retrieve(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("standalone guidance must not use the review Evidence endpoint")

    monkeypatch.setattr(ProfileEvidenceRetriever, "guidance_query", guidance)
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", review_retrieve)
    runtime = ProfileRuntime(_settings())

    question = "수출 통관 절차에서 기본으로 확인할 서류를 알려 주세요."
    answer = runtime.knowledge_answer_json(
        json.dumps(
            {
                "schema_version": "1.0",
                "conversation_id": "knowledge-conversation",
                "question": question,
                "trade_type": "export",
                "as_of": "2026-08-05",
                "recent_turns": [],
            }
        ).encode()
    )

    assert answer.answer_scope == "general_customs_guidance"
    assert answer.coverage == "covered"
    assert answer.snapshot_id == "dev-fixture-guidance"
    assert answer.citations and answer.citations[0].kind == "guidance_evidence"
    assert answer.answer_sections and answer.answer_sections[0].kind == "general_guidance"
    assert "case_id" not in answer.model_dump(mode="json")
    assert "review_run_id" not in answer.model_dump(mode="json")
    assert len(guidance_calls) == 1
    assert guidance_calls[0]["trade_direction"] == "export"
    assert guidance_calls[0]["topic_hint"] == "customs_clearance"
    assert guidance_calls[0]["query"] == "대한민국 수출입 통관 절차 공식 안내"
    assert guidance_calls[0]["query"] != question


def test_standalone_export_requirements_narrows_both_to_export_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guidance_calls: list[dict[str, Any]] = []

    def guidance(
        _self: ProfileEvidenceRetriever, request: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        guidance_calls.append(deepcopy(request))
        return _stub_guidance(None, request)

    monkeypatch.setattr(ProfileEvidenceRetriever, "guidance_query", guidance)
    runtime = ProfileRuntime(_settings())

    answer = runtime.knowledge_answer_json(
        json.dumps(
            {
                "schema_version": "1.0",
                "conversation_id": "knowledge-export-requirements",
                "question": "수출 요건을 알려 주세요.",
                "trade_type": "both",
                "as_of": "2026-08-05",
                "recent_turns": [],
            }
        ).encode()
    )

    assert answer.answer_scope == "general_customs_guidance"
    assert len(guidance_calls) == 1
    assert guidance_calls[0]["topic_hint"] == "export_requirements"
    assert guidance_calls[0]["trade_direction"] == "export"


def test_case_question_can_keep_case_and_general_guidance_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    monkeypatch.setattr(ProfileEvidenceRetriever, "guidance_query", _stub_guidance)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    answer = runtime.assistant_answer_json(
        _answer_payload(
            result, question="현재 사건의 수입 통관 절차에서 먼저 확인할 항목을 안내해 주세요."
        )
    )

    assert answer.answer_scope == "case_grounded"
    assert [section.kind for section in answer.answer_sections] == [
        "case_analysis",
        "general_guidance",
    ]
    assert any(citation.kind == "comparison" for citation in answer.citations)
    assert any(citation.kind == "guidance_evidence" for citation in answer.citations)
    assert GENERAL_GUIDANCE_DISCLAIMER in answer.answer_sections[1].content


@pytest.mark.parametrize(
    ("question", "expected_topic"),
    [
        ("AWB는 어떤 서류인가요?", "transport_documents"),
        ("수출 요건을 알려 주세요.", "export_requirements"),
        ("FTA 특혜 적용 절차를 알려 주세요.", "fta_procedure"),
        ("FOB와 CIF의 차이를 알려 주세요.", "trade_terms"),
    ],
)
def test_general_guidance_topic_taxonomy_matches_trade_questions(
    question: str, expected_topic: str
) -> None:
    assert guidance_topic_for_question(question) == expected_topic


def test_unrelated_question_requests_clarification_instead_of_arbitrary_case_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    answer = runtime.assistant_answer_json(
        _answer_payload(result, question="오늘 날씨가 어떤가요?")
    )

    assert answer.answer_scope == "clarification_needed"
    assert answer.citations == []


def test_case_specific_follow_up_questions_do_not_fall_back_to_general_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def regulatory_retrieve(
        _self: ProfileEvidenceRetriever, request: dict[str, Any]
    ) -> dict[str, Any]:
        item_id = request["items"][0]["item_id"]
        return {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [
                {
                    "item_id": item_id,
                    "evidence": [
                        {
                            "evidence_id": "evidence-food-confirmation",
                            "item_id": item_id,
                            "title": "수입식품등 수입신고확인증 확인",
                            "text": "수입식품등 수입신고확인증을 확인해야 합니다.",
                            "source_uri": "https://law.example.invalid/food",
                            "source_type": "law",
                            "version_hash": "b" * 64,
                            "law_id": "food-law",
                            "law_name": "수입식품안전관리 특별법",
                            "eligibility": "finding",
                            "required_document_code": "FOOD_IMPORT_CONFIRMATION",
                            "required_document_name": "수입식품등 수입신고확인증",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", regulatory_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_matching_review_payload()).encode())
    retrieval_stage = next(
        stage for stage in result.workflow.stages if stage.stage == "retrieve_evidence"
    )
    regulatory_findings = [finding for finding in result.findings if finding.evidence_ids]
    regulatory_actions = [action for action in result.required_actions if action.finding_ids]
    assert retrieval_stage.status == "completed"
    assert result.execution_status == "succeeded"
    assert len(regulatory_findings) == 1
    assert regulatory_findings[0].document_ids == ["certificate", "invoice", "packing"]
    assert len(regulatory_actions) == 1
    assert regulatory_actions[0].finding_ids == [regulatory_findings[0].finding_id]

    q1 = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question="원산지증명서(matching-origin.txt)의 원문에 HS 코드가 기재되어 있는지 확인할 수 있습니까?",
        )
    )
    assert q1.answer_scope == "case_grounded"
    assert any(citation.kind == "fact" for citation in q1.citations)

    q2 = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question="수입식품등 수입신고확인증이 제출되었거나 제출 예정인지 확인할 수 있습니까?",
        )
    )
    assert q2.answer_scope == "insufficient_evidence"
    assert q2.citations == []
    assert "제출" in q2.direct_answer and "예정" in q2.direct_answer

    q3 = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question="원산지증명서와 포장명세서, 인보이스 간 HS 코드 0303540000이 동일 품목을 나타내는지 확인되었습니까?",
        )
    )
    assert q3.answer_scope == "case_grounded"
    assert len(q3.citations) == 3

    for answer in (q1, q2, q3):
        assert reviewer_text_is_safe(
            [answer.direct_answer, *answer.suggested_questions],
            filenames=[analysis.filename for analysis in result.document_analyses],
        )


def test_submission_answer_distinguishes_case_attachment_from_document_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    payload = _review_payload()
    payload["documents"].append(
        {
            "document_id": "support",
            "filename": "support.txt",
            "content_text": (
                "SUPPORTING EVIDENCE\nReceipt No: RECEIPT-2026-001\n"
                "Submission Date: 2026-08-05\nAcceptance Status: Accepted"
            ),
        }
    )
    result = runtime.review_v3_json(json.dumps(payload).encode())

    answer = runtime.assistant_answer_json(
        _answer_payload(
            result, question="첨부된 보완 증빙의 제출과 수리·승인 여부를 확인해 주세요."
        )
    )

    assert answer.answer_scope == "case_grounded"
    assert answer.citations
    assert "현재 심사에는" in answer.direct_answer
    assert "외부 기관 시스템의 실시간 조회 결과가 아니라" in answer.direct_answer
    fact_names = {
        fact.fact_id: fact.name
        for analysis in result.document_analyses
        for fact in [
            *analysis.facts,
            *(fact for item in analysis.line_items for fact in item.facts),
        ]
    }
    assert {fact_names[citation.reference_id] for citation in answer.citations} >= {
        "submission_receipt_no",
        "submission_status",
    }

    attached_only_payload = _review_payload()
    attached_only_payload["documents"].append(
        {
            "document_id": "support-empty",
            "filename": "support-empty.txt",
            "content_text": "SUPPORTING EVIDENCE",
        }
    )
    attached_only = runtime.review_v3_json(json.dumps(attached_only_payload).encode())
    insufficient = runtime.assistant_answer_json(
        _answer_payload(
            attached_only,
            question="첨부된 보완 증빙의 기관 제출·승인 여부를 확인해 주세요.",
        )
    )
    assert insufficient.answer_scope == "insufficient_evidence"
    assert insufficient.citations == []
    assert "첨부되어 있습니다" in insufficient.direct_answer


def test_functional_exact_hs_answer_uses_snapshot_not_an_extra_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Responder:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("exact HS snapshot answer must not call the provider")

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_matching_review_payload()).encode())
    responder = Responder()
    runtime.mode = "functional-uat"
    runtime._grounded_responder = responder  # type: ignore[assignment]

    answer = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question="원산지증명서의 원문에 HS 코드가 기재되어 있는지 확인할 수 있습니까?",
        )
    )

    assert answer.answer_scope == "case_grounded"
    assert responder.calls == 0


def test_recent_case_context_keeps_a_short_follow_up_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_matching_review_payload()).encode())

    answer = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question="그 항목은 확인됐나요?",
            recent_turns=[
                {
                    "role": "assistant",
                    "content": "원산지증명서의 HS 코드 확인이 필요합니다.",
                }
            ],
        )
    )

    assert answer.answer_scope == "case_grounded"
    assert answer.citations


def test_deterministic_evidence_answer_keeps_raw_provider_title_in_citation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raw_title_retrieve(
        self: ProfileEvidenceRetriever, request: dict[str, Any]
    ) -> dict[str, Any]:
        response = _stub_retrieve(self, request)
        response["results"][0]["evidence"][0]["title"] = "customs_requireHSK_2026_release"
        return response

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", raw_title_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    answer = runtime.assistant_answer_json(
        _answer_payload(result, question="연결된 법령 근거는 무엇인가요?")
    )

    assert answer.answer_scope == "evidence_grounded"
    assert len(answer.citations) == 1
    assert answer.citations[0].kind == "evidence"
    assert answer.citations[0].reference_id == "evidence-article-1"
    assert "customs_requireHSK_2026_release" not in answer.direct_answer
    assert answer.direct_answer.startswith("현재 심사에는 연결된 법령 근거가 있습니다.")


def test_assistant_draft_is_non_mutating_and_endpoint_is_profile_protected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())
    original_hash = result.result_sha256
    answer = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question="문서: invoice; 필드: quantity; 값: 2 PCS; 사유: 담당자 정정 확인",
        )
    )
    assert answer.correction_draft is not None
    assert answer.correction_draft.new_value == "2 PCS"
    assert result.result_sha256 == original_hash

    english_template = runtime.assistant_answer_json(
        _answer_payload(
            result,
            question=(
                "document=invoice; field=quantity; value=2 PCS; "
                "reason=Invoice and Packing List require a reviewed correction"
            ),
        )
    )
    assert english_template.answer_scope == "case_grounded"
    assert english_template.correction_draft is not None
    assert english_template.correction_draft.document_id == "invoice"

    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        review = client.post(
            "/internal/agent/v2/reviews",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json=_review_payload() | {"review_run_id": "run-assistant-http"},
        )
        assert review.status_code == 200
        request = {
            "schema_version": "1.0",
            "case_id": "case-assistant",
            "review_run_id": "run-assistant-http",
            "conversation_id": "conversation-http",
            "review_result": review.json(),
            "question": "현재 문서 비교 결과를 설명해 주세요.",
            "recent_turns": [],
        }
        missing_mode = client.post(
            "/internal/agent/v2/assistant/answers",
            headers={"Authorization": f"Bearer {_TOKEN}"},
            json=request,
        )
        accepted = client.post(
            "/internal/agent/v2/assistant/answers",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json=request,
        )
    assert missing_mode.status_code == 403
    assert accepted.status_code == 200
    assert accepted.json()["answer_scope"] == "case_grounded"


def test_provider_scope_policy_falls_back_and_enforces_correction_target_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidScopeResponder:
        def respond(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "answer_scope": "general_customs_guidance",
                "direct_answer": f"현재 사건을 자동으로 승인합니다. {GENERAL_GUIDANCE_DISCLAIMER}",
                "citations": [],
                "suggested_questions": [],
                "correction_draft": None,
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())
    assert result.engine is not None
    request = AssistantAnswerRequestV1.model_validate(
        json.loads(_answer_payload(result, question="현재 문서 비교 결과를 설명해 주세요."))
    )

    invalid_general = {
        "answer_scope": "general_customs_guidance",
        "direct_answer": GENERAL_GUIDANCE_DISCLAIMER,
        "citations": [],
        "suggested_questions": [],
        "correction_draft": None,
    }
    with pytest.raises(ValueError, match="assistant_provider_output_invalid"):
        response_from_provider_candidate(invalid_general, request, engine=result.engine)

    draft_request = AssistantAnswerRequestV1.model_validate(
        json.loads(
            _answer_payload(
                result,
                question=("document=invoice; field=quantity; value=2 PCS; reason=담당자 정정 확인"),
            )
        )
    )
    missing_required_draft = {
        "answer_scope": "case_grounded",
        "direct_answer": "문서 비교 항목을 확인해야 합니다.",
        "citations": [{"kind": "comparison", "reference_id": result.comparisons[0].comparison_id}],
        "suggested_questions": [],
        "correction_draft": None,
    }
    with pytest.raises(ValueError, match="assistant_provider_correction_draft_required"):
        response_from_provider_candidate(
            missing_required_draft, draft_request, engine=result.engine
        )

    item = result.document_analyses[0].line_items[0]
    document_fact = result.document_analyses[0].facts[0]
    ownership_mismatch = {
        "answer_scope": "case_grounded",
        "direct_answer": "문서 비교 항목을 확인해야 합니다.",
        "citations": [{"kind": "comparison", "reference_id": result.comparisons[0].comparison_id}],
        "suggested_questions": [],
        "correction_draft": {
            "document_id": "invoice",
            "item_id": item.item_id,
            "fact_id": document_fact.fact_id,
            "field_name": document_fact.name,
            "new_value": "2 PCS",
            "reason": "대상 불일치 검증",
        },
    }
    with pytest.raises(ValueError, match="assistant correction draft fact must belong to its item"):
        response_from_provider_candidate(ownership_mismatch, request, engine=result.engine)

    runtime.mode = "functional-uat"
    runtime._grounded_responder = InvalidScopeResponder()  # type: ignore[assignment]
    fallback = runtime.assistant_answer_json(
        _answer_payload(result, question="현재 문서 비교 결과를 설명해 주세요.")
    )
    assert fallback.answer_scope == "case_grounded"
    assert fallback.citations and fallback.citations[0].kind == "comparison"


def test_functional_policy_only_calls_provider_for_grounded_case_or_evidence_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Responder:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("non-grounded policy answer must not call the provider")

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())
    responder = Responder()
    runtime.mode = "functional-uat"
    runtime._grounded_responder = responder  # type: ignore[assignment]

    answer = runtime.assistant_answer_json(
        _answer_payload(result, question="웹 검색으로 경쟁사 정보를 찾아 주세요.")
    )

    assert answer.answer_scope == "out_of_scope"
    assert responder.calls == 0


def test_functional_report_provider_failure_keeps_deterministic_grounded_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingResponder:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise RuntimeError("provider unavailable")

        def close(self) -> None:
            return None

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    responder = FailingResponder()
    runtime.mode = "functional-uat"
    runtime._grounded_responder = responder  # type: ignore[assignment]

    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    assert responder.calls == 1
    assert len(result.report.sections) == 8
    assert all(section.claims for section in result.report.sections)
    assert all(
        claim.fact_ids or claim.comparison_ids or claim.evidence_ids
        for section in result.report.sections
        for claim in section.claims
    )


def test_functional_report_uses_one_valid_provider_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ValidResponder:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            snapshot = payload["review_snapshot"]
            comparison_id = snapshot["comparisons"][0]["comparison_id"]
            sections = [
                {
                    "section": name,
                    "title": title,
                    "claims": [
                        {
                            "claim_id": f"provider:{name}",
                            "text": f"{title}은 제공된 비교 근거를 사용했습니다.",
                            "fact_ids": [],
                            "comparison_ids": [comparison_id],
                            "evidence_ids": [],
                        },
                        {
                            "claim_id": f"ungrounded:{name}",
                            "text": "근거 없이 추가된 문장은 공개 결과에 포함되면 안 됩니다.",
                            "fact_ids": [],
                            "comparison_ids": [],
                            "evidence_ids": [],
                        },
                    ],
                }
                for name, title in (
                    ("conclusion", "결론"),
                    ("execution_process", "실행 과정"),
                    ("document_comparison", "문서 비교"),
                    ("issues_and_causes", "문제와 원인"),
                    ("legal_evidence", "법령 근거"),
                    ("impact", "영향"),
                    ("next_actions", "다음 조치"),
                    ("limitations", "한계"),
                )
            ]
            impact = next(section for section in sections if section["section"] == "impact")
            impact["claims"] = [impact["claims"][1]]
            return {"sections": sections}

        def close(self) -> None:
            return None

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    responder = ValidResponder()
    runtime.mode = "functional-uat"
    runtime._grounded_responder = responder  # type: ignore[assignment]

    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    assert responder.calls == 1
    assert result.report.sections[0].claims[0].claim_id == "provider:conclusion"
    assert all(len(section.claims) == 1 for section in result.report.sections)
    assert result.report.sections[5].claims[0].claim_id.startswith("report:impact:")
    assert result.result_sha256


def test_functional_assistant_falls_back_when_provider_exposes_internal_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LeakyResponder:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            comparison_id = payload["review_snapshot"]["comparisons"][0]["comparison_id"]
            return {
                "direct_answer": f"확인 결과는 {comparison_id}입니다.",
                "citations": [{"kind": "comparison", "reference_id": comparison_id}],
                "suggested_questions": ["inspect_document_consistency를 실행해 주세요."],
                "correction_draft": None,
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())
    responder = LeakyResponder()
    runtime.mode = "functional-uat"
    runtime._grounded_responder = responder  # type: ignore[assignment]

    answer = runtime.assistant_answer_json(
        _answer_payload(result, question="현재 문서 비교 결과를 설명해 주세요.")
    )

    assert responder.calls == 1
    assert answer.answer_scope == "case_grounded"
    assert len(answer.citations) == 1
    assert answer.citations[0].kind == "comparison"
    assert answer.citations[0].reference_id == result.comparisons[0].comparison_id
    assert not contains_internal_result_marker(answer.direct_answer)
    assert not any(contains_internal_result_marker(value) for value in answer.suggested_questions)
    assert "현재 사건의 관련 비교 결과" in answer.direct_answer


def test_provider_assistant_rejects_english_reviewer_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())
    assert result.engine is not None
    request = AssistantAnswerRequestV1.model_validate(
        json.loads(_answer_payload(result, question="현재 문서 비교 결과를 설명해 주세요."))
    )

    with pytest.raises(ValueError, match="assistant_provider_user_text_unsafe"):
        response_from_provider_candidate(
            {
                "direct_answer": "Document validation is incomplete.",
                "citations": [
                    {"kind": "comparison", "reference_id": result.comparisons[0].comparison_id}
                ],
                "suggested_questions": [],
                "correction_draft": None,
            },
            request,
            engine=result.engine,
        )


def test_functional_report_falls_back_when_provider_exposes_internal_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LeakyReportResponder:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, payload: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            comparison_id = payload["review_snapshot"]["comparisons"][0]["comparison_id"]
            return {
                "sections": [
                    {
                        "section": name,
                        "title": title,
                        "claims": [
                            {
                                "claim_id": f"provider:{name}",
                                "text": f"{comparison_id}와 needs_review를 확인했습니다.",
                                "fact_ids": [],
                                "comparison_ids": [comparison_id],
                                "evidence_ids": [],
                            }
                        ],
                    }
                    for name, title in (
                        ("conclusion", "결론"),
                        ("execution_process", "실행 과정"),
                        ("document_comparison", "문서 비교"),
                        ("issues_and_causes", "문제와 원인"),
                        ("legal_evidence", "법령 근거"),
                        ("impact", "영향"),
                        ("next_actions", "다음 조치"),
                        ("limitations", "한계"),
                    )
                ]
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(ProfileEvidenceRetriever, "retrieve", _stub_retrieve)
    runtime = ProfileRuntime(_settings())
    responder = LeakyReportResponder()
    runtime.mode = "functional-uat"
    runtime._grounded_responder = responder  # type: ignore[assignment]

    result = runtime.review_v3_json(json.dumps(_review_payload()).encode())

    assert responder.calls == 1
    assert result.report.sections[0].claims[0].claim_id.startswith("report:conclusion:")
    assert all(
        not contains_internal_result_marker(value)
        for section in result.report.sections
        for value in (section.title, *(claim.text for claim in section.claims))
    )
    assert any(
        claim.comparison_ids for section in result.report.sections for claim in section.claims
    )
