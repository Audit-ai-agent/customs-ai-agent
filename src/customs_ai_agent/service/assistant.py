"""One bounded, grounded assistant turn over an immutable review snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from customs_ai_agent.schemas.assistant import (
    AssistantAnswerRequestV1,
    AssistantAnswerResponseV1,
    AssistantCitationV1,
    AssistantCorrectionDraftV1,
    GENERAL_GUIDANCE_DISCLAIMER,
)
from customs_ai_agent.schemas.v3 import EngineV3, ReviewRunResultV3


_OUT_OF_SCOPE = re.compile(
    r"(?:웹\s*검색|인터넷|url|shell|명령(?:어)?|파일\s*시스템|db\s*직접|데이터베이스\s*직접|"
    r"token|api\s*key|비밀번호|cookie)",
    re.IGNORECASE,
)
_EVIDENCE_WORDS = re.compile(r"(?:법령|근거|evidence|조문|출처)", re.IGNORECASE)
_GENERAL_REQUEST = re.compile(
    r"(?:일반|기본|통관|관세|세율|hs\s*code|품목\s*분류|원산지|인보이스|상업송장|"
    r"packing\s*list|b\s*/?\s*l|선하증권)",
    re.IGNORECASE,
)
_GENERAL_TOPIC_ALLOWLIST = (
    re.compile(r"(?:(?:수입|수출)\s*)?통관\s*(?:절차|서류|신고|준비|흐름|기본)", re.I),
    re.compile(r"(?:hs\s*code|품목\s*분류)\s*(?:란|이란|기본|일반|절차|방법|설명|확인)?", re.I),
    re.compile(r"원산지\s*(?:증명(?:서)?|판정|표시|확인\s*절차|란|이란)", re.I),
    re.compile(r"(?:관세|세율)\s*(?:계산|기본|일반|확인\s*방법)", re.I),
    re.compile(
        r"(?:인보이스|상업송장|packing\s*list|b\s*/?\s*l|선하증권)\s*(?:역할|차이|기본|일반)", re.I
    ),
)
_CORRECTION_DRAFT = re.compile(
    r"(?:문서|document)\s*[:=]\s*(?P<document>[A-Za-z0-9._:-]+).*?"
    r"(?:필드|field)\s*[:=]\s*(?P<field>[A-Za-z0-9_:-]+).*?"
    r"(?:값|value)\s*[:=]\s*(?P<value>[^;\n]+).*?"
    r"(?:사유|reason)\s*[:=]\s*(?P<reason>[^;\n]+)",
    re.IGNORECASE | re.DOTALL,
)

_AnswerScope = Literal[
    "case_grounded",
    "evidence_grounded",
    "general_customs_guidance",
    "clarification_needed",
    "insufficient_evidence",
    "out_of_scope",
]


@dataclass(frozen=True)
class _QuestionPolicy:
    """The one allowed answer class for a question before Claude is called."""

    scope: _AnswerScope


def deterministic_assistant_answer(
    request: AssistantAnswerRequestV1, *, engine: EngineV3
) -> AssistantAnswerResponseV1:
    """Return a bounded answer without provider or additional tool calls."""

    question = request.question.strip()
    result = request.review_result
    policy = _classify_question(question, result)
    if policy.scope == "out_of_scope":
        return _answer(
            request,
            engine=engine,
            scope="out_of_scope",
            text=(
                "이 도우미는 현재 심사 결과와 연결된 Evidence 범위에서만 답변합니다. "
                "웹 검색, 시스템 명령, 비밀값 또는 외부 데이터 접근 요청은 처리하지 않습니다."
            ),
            citations=[],
            suggestions=["현재 심사 결과에서 확인이 필요한 비교 항목은 무엇인가요?"],
        )
    if policy.scope == "clarification_needed":
        return _answer(
            request,
            engine=engine,
            scope="clarification_needed",
            text="어느 문서, 비교 항목 또는 법령 근거를 확인할지 구체적으로 알려 주세요.",
            citations=[],
            suggestions=[
                "수량 비교 결과를 설명해 주세요.",
                "연결된 법령 Evidence를 보여 주세요.",
            ],
        )
    if policy.scope == "general_customs_guidance":
        return _answer(
            request,
            engine=engine,
            scope="general_customs_guidance",
            text=(
                "일반적인 통관 안내는 문서·품목·원산지 정보가 서로 일치하는지 확인하고, "
                f"필요한 증빙을 담당자가 검토하는 과정입니다. {GENERAL_GUIDANCE_DISCLAIMER}"
            ),
            citations=[],
            suggestions=["현재 사건의 문서 비교 결과를 설명해 주세요."],
        )
    if policy.scope == "insufficient_evidence":
        return _answer(
            request,
            engine=engine,
            scope="insufficient_evidence",
            text="현재 Review Run에는 이 질문을 뒷받침할 충분한 문서 비교 또는 법령 Evidence가 없습니다.",
            citations=[],
            suggestions=["문서를 보완한 뒤 새 심사를 실행해 주세요."],
        )
    if policy.scope == "evidence_grounded":
        evidence = result.evidence[0] if result.evidence else None
        if evidence is None:
            return _answer(
                request,
                engine=engine,
                scope="insufficient_evidence",
                text="현재 Review Run에는 이 질문을 뒷받침할 연결된 법령 Evidence가 없습니다.",
                citations=[],
                suggestions=["문서 비교 결과에서 사람이 확인해야 할 항목은 무엇인가요?"],
            )
        return _answer(
            request,
            engine=engine,
            scope="evidence_grounded",
            text=(
                f"현재 심사에는 ‘{evidence.title}’ Evidence가 연결되어 있습니다. "
                "이 근거는 담당자가 원문과 함께 확인해야 하며, 자동 승인 근거로 사용되지 않습니다."
            ),
            citations=[AssistantCitationV1(kind="evidence", reference_id=evidence.evidence_id)],
            suggestions=["이 Evidence가 어떤 finding과 연결되는지 설명해 주세요."],
        )
    comparison = _primary_comparison(result)
    if comparison is None:
        return _answer(
            request,
            engine=engine,
            scope="insufficient_evidence",
            text="현재 Review Run에는 질문을 설명할 수 있는 문서 비교 결과가 없습니다.",
            citations=[],
            suggestions=["문서를 보완한 뒤 새 심사를 실행해 주세요."],
        )
    return _answer(
        request,
        engine=engine,
        scope="case_grounded",
        text=(
            f"현재 사건의 관련 비교 결과는 ‘{comparison.message}’입니다. "
            "이 값은 자동 결론이 아니므로 문서 원문과 연결된 근거를 담당자가 함께 확인해야 합니다."
        ),
        citations=[AssistantCitationV1(kind="comparison", reference_id=comparison.comparison_id)],
        suggestions=[
            "이 비교에 사용한 문서 사실을 보여 주세요.",
            "연결된 법령 근거가 있나요?",
        ],
        correction_draft=_draft_from_question(question, result),
    )


def response_from_provider_candidate(
    candidate: Mapping[str, Any],
    request: AssistantAnswerRequestV1,
    *,
    engine: EngineV3,
) -> AssistantAnswerResponseV1:
    """Validate a single Claude candidate against the fixed public envelope."""

    payload = {
        "schema_version": "1.0",
        "case_id": request.case_id,
        "review_run_id": request.review_run_id,
        "conversation_id": request.conversation_id,
        "answer_scope": candidate.get("answer_scope"),
        "direct_answer": candidate.get("direct_answer"),
        "citations": candidate.get("citations", []),
        "suggested_questions": candidate.get("suggested_questions", []),
        "correction_draft": candidate.get("correction_draft"),
        "engine": engine,
    }
    try:
        answer = AssistantAnswerResponseV1.model_validate(payload)
    except ValidationError:
        raise ValueError("assistant_provider_output_invalid") from None
    _validate_scope_policy(answer, request)
    _validate_references(answer, request.review_result)
    return answer


def provider_snapshot(result: ReviewRunResultV3) -> dict[str, Any]:
    """Bound a functional-UAT provider context without changing public evidence.

    The public result remains complete.  Claude sees only enough structured
    fields to compose one answer or report, avoiding an unbounded prompt when a
    Data response contains many long Evidence excerpts.
    """

    analyses: list[dict[str, Any]] = []
    for analysis in result.document_analyses[:20]:
        facts = [
            {
                "fact_id": fact.fact_id,
                "name": fact.name,
                "raw_value": fact.raw_value[:500],
                "source": fact.source.model_dump(mode="json", warnings=False),
            }
            for fact in analysis.facts[:32]
        ]
        analyses.append(
            {
                "document_id": analysis.document_id,
                "filename": analysis.filename,
                "document_type": analysis.document_type,
                "facts": facts,
                "line_items": [
                    {
                        "item_id": item.item_id,
                        "facts": [
                            {
                                "fact_id": fact.fact_id,
                                "name": fact.name,
                                "raw_value": fact.raw_value[:500],
                                "source": fact.source.model_dump(mode="json", warnings=False),
                            }
                            for fact in item.facts[:16]
                        ],
                    }
                    for item in analysis.line_items[:25]
                ],
            }
        )
    return {
        "case_id": result.case_id,
        "review_run_id": result.review_run_id,
        "workflow": result.workflow.model_dump(mode="json", warnings=False),
        "document_analyses": analyses,
        "comparisons": [
            item.model_dump(mode="json", warnings=False) for item in result.comparisons[:100]
        ],
        "evidence": [
            {
                **item.model_dump(mode="json", warnings=False, exclude={"text"}),
                "text": item.text[:1200],
            }
            for item in result.evidence[:30]
        ],
        "findings": [
            item.model_dump(mode="json", warnings=False) for item in result.findings[:100]
        ],
        "required_actions": [
            item.model_dump(mode="json", warnings=False) for item in result.required_actions[:100]
        ],
    }


def _answer(
    request: AssistantAnswerRequestV1,
    *,
    engine: EngineV3,
    scope: str,
    text: str,
    citations: list[AssistantCitationV1],
    suggestions: list[str],
    correction_draft: AssistantCorrectionDraftV1 | None = None,
) -> AssistantAnswerResponseV1:
    answer = AssistantAnswerResponseV1(
        case_id=request.case_id,
        review_run_id=request.review_run_id,
        conversation_id=request.conversation_id,
        answer_scope=scope,  # type: ignore[arg-type]
        direct_answer=text,
        citations=citations,
        suggested_questions=suggestions,
        correction_draft=correction_draft,
        engine=engine,
    )
    _validate_references(answer, request.review_result)
    return answer


def _validate_references(answer: AssistantAnswerResponseV1, result: ReviewRunResultV3) -> None:
    facts_by_id: dict[str, tuple[str, str | None, str]] = {}
    for analysis in result.document_analyses:
        for fact in analysis.facts:
            facts_by_id[fact.fact_id] = (analysis.document_id, None, fact.name)
        for item in analysis.line_items:
            for fact in item.facts:
                facts_by_id[fact.fact_id] = (analysis.document_id, item.item_id, fact.name)
    facts = set(facts_by_id)
    comparisons = {item.comparison_id for item in result.comparisons}
    evidence = {item.evidence_id for item in result.evidence}
    known = {"fact": facts, "comparison": comparisons, "evidence": evidence}
    if any(citation.reference_id not in known[citation.kind] for citation in answer.citations):
        raise ValueError("assistant citation must refer to the supplied review result")
    draft = answer.correction_draft
    if draft is None:
        return
    analyses_by_document = {analysis.document_id: analysis for analysis in result.document_analyses}
    draft_analysis = analyses_by_document.get(draft.document_id)
    if draft_analysis is None:
        raise ValueError(
            "assistant correction draft document must refer to the supplied review result"
        )
    if draft.item_id is not None:
        if not any(entry.item_id == draft.item_id for entry in draft_analysis.line_items):
            raise ValueError("assistant correction draft item must belong to its document")
    if draft.fact_id is None:
        return
    fact_owner = facts_by_id.get(draft.fact_id)
    if fact_owner is None:
        raise ValueError("assistant correction draft fact must refer to the supplied review result")
    fact_document_id, fact_item_id, fact_name = fact_owner
    if fact_document_id != draft.document_id:
        raise ValueError("assistant correction draft fact must belong to its document")
    if draft.item_id is not None and fact_item_id != draft.item_id:
        raise ValueError("assistant correction draft fact must belong to its item")
    if fact_name != draft.field_name.strip().casefold():
        raise ValueError("assistant correction draft field must match its fact")


def _primary_comparison(result: ReviewRunResultV3):
    return next(
        (item for item in result.comparisons if item.status in {"mismatch", "needs_review"}),
        result.comparisons[0] if result.comparisons else None,
    )


def _mentions_case_reference(question: str) -> bool:
    return bool(re.search(r"(?:이\s*사건|현재\s*사건|문서|심사|finding|비교)", question, re.I))


def _classify_question(question: str, result: ReviewRunResultV3) -> _QuestionPolicy:
    """Classify a turn before a provider can choose an answer scope.

    The classifier is deliberately small and deterministic.  Claude may write a
    grounded answer but cannot reinterpret a case question as general advice or
    turn a prohibited request into a case answer.
    """

    if _OUT_OF_SCOPE.search(question):
        return _QuestionPolicy("out_of_scope")
    if len(question) < 3 or question in {"?", "무엇", "확인"}:
        return _QuestionPolicy("clarification_needed")
    if not _mentions_case_reference(question) and any(
        pattern.search(question) for pattern in _GENERAL_TOPIC_ALLOWLIST
    ):
        return _QuestionPolicy("general_customs_guidance")
    if not _mentions_case_reference(question) and _GENERAL_REQUEST.search(question):
        # The assistant may provide only the small general-guidance allowlist;
        # other generic customs questions should not be presented as case advice.
        return _QuestionPolicy("out_of_scope")
    if _EVIDENCE_WORDS.search(question):
        return _QuestionPolicy("evidence_grounded" if result.evidence else "insufficient_evidence")
    has_case_support = bool(result.comparisons) or any(
        analysis.facts or analysis.line_items for analysis in result.document_analyses
    )
    return _QuestionPolicy("case_grounded" if has_case_support else "insufficient_evidence")


def _validate_scope_policy(
    answer: AssistantAnswerResponseV1, request: AssistantAnswerRequestV1
) -> None:
    expected = _classify_question(request.question.strip(), request.review_result)
    if answer.answer_scope != expected.scope:
        raise ValueError("assistant_provider_scope_policy_violation")


def _draft_from_question(
    question: str, result: ReviewRunResultV3
) -> AssistantCorrectionDraftV1 | None:
    match = _CORRECTION_DRAFT.search(question)
    if match is None:
        return None
    document_id = match.group("document")
    known_documents = {analysis.document_id for analysis in result.document_analyses}
    if document_id not in known_documents:
        return None
    return AssistantCorrectionDraftV1(
        document_id=document_id,
        field_name=match.group("field").strip().casefold(),
        new_value=match.group("value").strip(),
        reason=match.group("reason").strip(),
    )


__all__ = [
    "deterministic_assistant_answer",
    "provider_snapshot",
    "response_from_provider_candidate",
]
