from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.schemas.assistant import (
    AssistantAnswerRequestV1,
    GENERAL_GUIDANCE_DISCLAIMER,
)
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.assistant import response_from_provider_candidate
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

    general_answer = runtime.assistant_answer_json(
        _answer_payload(result, question="일반 통관 절차를 안내해 주세요.")
    )
    assert general_answer.answer_scope == "general_customs_guidance"
    assert general_answer.citations == []
    assert GENERAL_GUIDANCE_DISCLAIMER in general_answer.direct_answer

    unsupported_general = runtime.assistant_answer_json(
        _answer_payload(result, question="일반 관세 환급 전략을 알려 주세요.")
    )
    assert unsupported_general.answer_scope == "out_of_scope"

    out_of_scope = runtime.assistant_answer_json(
        _answer_payload(result, question="웹 검색으로 다른 법령도 찾아 주세요.")
    )
    assert out_of_scope.answer_scope == "out_of_scope"

    too_many = [{"role": "user", "content": f"질문 {index}"} for index in range(9)]
    with pytest.raises(RuntimeInputError):
        runtime.assistant_answer_json(
            _answer_payload(result, question="현재 비교를 설명해 주세요.", recent_turns=too_many)
        )


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
