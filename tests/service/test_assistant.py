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
    with pytest.raises(ValueError, match="assistant_provider_scope_policy_violation"):
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
    assert result.result_sha256
