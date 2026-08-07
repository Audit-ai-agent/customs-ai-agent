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
from customs_ai_agent.schemas.v3 import EngineV3, FactV3, ReviewRunResultV3
from customs_ai_agent.service.user_text import (
    localize_deterministic_reviewer_prose,
    review_result_markers,
    reviewer_text_is_safe,
)


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
_SUBMISSION_STATUS_QUESTION = re.compile(r"(?:제출|예정|접수|신고\s*(?:여부|완료))")
_CERTIFICATE_HS_QUESTION = re.compile(r"원산지\s*증명서.*(?:hs\s*코드|품목\s*분류)", re.I)
_MULTI_DOCUMENT_HS_QUESTION = re.compile(
    r"(?:동일\s*품목|같은\s*품목|일치).*?(?:hs\s*코드|품목\s*분류)|"
    r"(?:hs\s*코드|품목\s*분류).*?(?:동일|일치|같은)",
    re.I,
)
_GENERAL_INTENT = re.compile(r"(?:역할|차이|기본|일반|란|이란|절차|방법|개념)", re.I)
_DOCUMENT_ALIASES = {
    "invoice": ("상업송장", "인보이스", "invoice", "commercial invoice"),
    "packing_list": ("포장명세서", "packing list", "packing"),
    "bill_of_lading": ("선하증권", "bill of lading", "b/l", "bl 번호"),
    "certificate_of_origin": ("원산지증명서", "certificate of origin", "c/o"),
}

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
    target: Literal["comparison", "certificate_hs", "multi_document_hs", "submission"] = (
        "comparison"
    )


_SAFE_DIRECT_ANSWER = {
    "case_grounded": (
        "현재 사건의 문서와 비교 결과를 함께 확인해 주세요. "
        "이 결과는 담당자 검토를 돕기 위한 안내입니다."
    ),
    "evidence_grounded": (
        "현재 심사에 연결된 법령 근거를 확인해 주세요. 이 근거만으로 자동 판단할 수 없습니다."
    ),
    "general_customs_guidance": (
        f"일반적인 통관 절차와 필요한 서류를 확인해 주세요. {GENERAL_GUIDANCE_DISCLAIMER}"
    ),
    "clarification_needed": "확인하려는 문서나 비교 항목을 조금 더 구체적으로 알려 주세요.",
    "insufficient_evidence": "현재 심사 결과만으로는 이 질문에 답할 근거가 충분하지 않습니다.",
    "out_of_scope": "이 도우미는 현재 심사 결과와 연결된 법령 근거 범위에서만 도와드릴 수 있습니다.",
}

_SAFE_SUGGESTION = "현재 문서 비교에서 확인이 필요한 항목은 무엇인가요?"


def deterministic_assistant_answer(
    request: AssistantAnswerRequestV1, *, engine: EngineV3
) -> AssistantAnswerResponseV1:
    """Return a bounded answer without provider or additional tool calls."""

    question = request.question.strip()
    result = request.review_result
    policy = _classify_question(question, result, recent_turns=request.recent_turns)
    if policy.scope == "out_of_scope":
        return _answer(
            request,
            engine=engine,
            scope="out_of_scope",
            text=(
                "이 도우미는 현재 심사 결과와 연결된 법령 근거 범위에서만 답변합니다. "
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
                "연결된 법령 근거를 보여 주세요.",
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
        text = (
            "현재 심사 결과에는 수입신고확인증의 제출 또는 제출 예정 여부를 확인할 사실이 "
            "포함되어 있지 않습니다. 관련 증빙을 보완한 뒤 다시 확인해 주세요."
            if policy.target == "submission"
            else "현재 심사 결과에는 이 질문을 뒷받침할 충분한 문서 비교 또는 법령 근거가 없습니다."
        )
        return _answer(
            request,
            engine=engine,
            scope="insufficient_evidence",
            text=text,
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
                text="현재 심사 결과에는 이 질문을 뒷받침할 연결된 법령 근거가 없습니다.",
                citations=[],
                suggestions=["문서 비교 결과에서 사람이 확인해야 할 항목은 무엇인가요?"],
            )
        return _answer(
            request,
            engine=engine,
            scope="evidence_grounded",
            text=(
                "현재 심사에는 연결된 법령 근거가 있습니다. "
                "이 근거는 담당자가 원문과 함께 확인해야 하며, 자동 승인 근거로 사용되지 않습니다."
            ),
            citations=[AssistantCitationV1(kind="evidence", reference_id=evidence.evidence_id)],
            suggestions=["이 법령 근거가 어떤 확인 항목과 연결되는지 설명해 주세요."],
        )
    case_text, case_citations = _targeted_case_answer(policy, result)
    if case_text is not None:
        return _answer(
            request,
            engine=engine,
            scope="case_grounded",
            text=case_text,
            citations=case_citations,
            suggestions=[
                "이 비교에 사용한 문서 사실을 보여 주세요.",
                "연결된 법령 근거가 있나요?",
            ],
            correction_draft=_draft_from_question(question, result),
        )
    comparison = _primary_comparison(result)
    if comparison is None:
        return _answer(
            request,
            engine=engine,
            scope="insufficient_evidence",
            text="현재 심사 결과에는 질문을 설명할 수 있는 문서 비교 결과가 없습니다.",
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

    expected_scope = _classify_question(
        request.question.strip(), request.review_result, recent_turns=request.recent_turns
    ).scope
    payload = {
        "schema_version": "1.0",
        "case_id": request.case_id,
        "review_run_id": request.review_run_id,
        "conversation_id": request.conversation_id,
        # Scope is a deterministic boundary, not a model-controlled label.
        # Claude still supplies the answer text, citations, and optional draft.
        "answer_scope": expected_scope,
        "direct_answer": candidate.get("direct_answer"),
        "citations": _known_provider_citations(candidate, request.review_result),
        "suggested_questions": candidate.get("suggested_questions", []),
        "correction_draft": candidate.get("correction_draft"),
        "engine": engine,
    }
    try:
        answer = AssistantAnswerResponseV1.model_validate(payload)
    except ValidationError:
        raise ValueError("assistant_provider_output_invalid") from None
    _validate_scope_policy(answer, request)
    if not _provider_reviewer_text_is_safe(answer, result=request.review_result):
        raise ValueError("assistant_provider_user_text_unsafe")
    _validate_references(answer, request.review_result)
    _validate_scope_requirements(answer)
    expected_draft = _draft_from_question(request.question, request.review_result)
    if expected_draft is not None:
        draft = answer.correction_draft
        if (
            draft is None
            or draft.document_id != expected_draft.document_id
            or draft.field_name != expected_draft.field_name
            or draft.new_value != expected_draft.new_value
            or draft.reason != expected_draft.reason
        ):
            raise ValueError("assistant_provider_correction_draft_required")
    return answer


def provider_answer_allowed(request: AssistantAnswerRequestV1) -> bool:
    """Keep exact snapshot fact answers deterministic instead of model-written.

    Claude may still compose bounded explanations for ordinary case and legal
    evidence questions.  Questions that ask for a specific extracted HS fact
    or a cross-document HS match have a complete code-owned answer already,
    so sending them to the provider would only add ambiguity and cost.
    """

    policy = _classify_question(
        request.question.strip(), request.review_result, recent_turns=request.recent_turns
    )
    return policy.scope in {"case_grounded", "evidence_grounded"} and policy.target == "comparison"


def _known_provider_citations(
    candidate: Mapping[str, Any], result: ReviewRunResultV3
) -> list[dict[str, str]]:
    """Keep only typed citations that refer to the immutable review snapshot."""

    facts = {
        fact.fact_id
        for analysis in result.document_analyses
        for fact in [
            *analysis.facts,
            *(fact for item in analysis.line_items for fact in item.facts),
        ]
    }
    comparisons = {item.comparison_id for item in result.comparisons}
    evidence = {item.evidence_id for item in result.evidence}
    known = {"fact": facts, "comparison": comparisons, "evidence": evidence}
    raw = candidate.get("citations", [])
    if not isinstance(raw, list):
        return []
    return [
        {"kind": kind, "reference_id": reference_id}
        for item in raw
        if isinstance(item, Mapping)
        and isinstance((kind := item.get("kind")), str)
        and kind in known
        and isinstance((reference_id := item.get("reference_id")), str)
        and reference_id in known[kind]
    ]


def _validate_scope_requirements(answer: AssistantAnswerResponseV1) -> None:
    """Apply the same citation and non-case boundaries after provider cleanup."""

    if answer.answer_scope == "case_grounded" and not answer.citations:
        raise ValueError("assistant_provider_case_citations_required")
    if answer.answer_scope == "evidence_grounded":
        if not any(citation.kind == "evidence" for citation in answer.citations):
            raise ValueError("assistant_provider_evidence_citation_required")
    if answer.answer_scope == "general_customs_guidance":
        if answer.citations or answer.correction_draft is not None:
            raise ValueError("assistant_provider_general_boundary_invalid")
        if GENERAL_GUIDANCE_DISCLAIMER not in answer.direct_answer:
            raise ValueError("assistant_provider_general_disclaimer_missing")
    if answer.answer_scope == "out_of_scope" and (
        answer.citations or answer.correction_draft is not None
    ):
        raise ValueError("assistant_provider_out_of_scope_boundary_invalid")


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
    markers = review_result_markers(request.review_result)
    filenames = tuple(analysis.filename for analysis in request.review_result.document_analyses)
    safe_text = _reviewer_text_or_fallback(text, scope=scope, markers=markers, filenames=filenames)
    safe_suggestions = _reviewer_suggestions_or_fallback(
        suggestions, markers=markers, filenames=filenames
    )
    answer = AssistantAnswerResponseV1(
        case_id=request.case_id,
        review_run_id=request.review_run_id,
        conversation_id=request.conversation_id,
        answer_scope=scope,  # type: ignore[arg-type]
        direct_answer=safe_text,
        citations=citations,
        suggested_questions=safe_suggestions,
        correction_draft=correction_draft,
        engine=engine,
    )
    _validate_references(answer, request.review_result)
    return answer


def _provider_reviewer_text_is_safe(
    answer: AssistantAnswerResponseV1, *, result: ReviewRunResultV3
) -> bool:
    """Provider prose may not expose result IDs; citations carry them structurally."""

    return reviewer_text_is_safe(
        (answer.direct_answer, *answer.suggested_questions),
        markers=review_result_markers(result),
        filenames=(analysis.filename for analysis in result.document_analyses),
    )


def _reviewer_text_or_fallback(
    value: str,
    *,
    scope: str,
    markers: frozenset[str],
    filenames: tuple[str, ...],
) -> str:
    """Keep deterministic reviewer prose Korean-first and implementation-free."""

    localized = localize_deterministic_reviewer_prose(value)
    if not reviewer_text_is_safe([localized], markers=markers, filenames=filenames):
        return _SAFE_DIRECT_ANSWER.get(scope, _SAFE_DIRECT_ANSWER["insufficient_evidence"])
    return localized


def _reviewer_suggestions_or_fallback(
    values: list[str], *, markers: frozenset[str], filenames: tuple[str, ...]
) -> list[str]:
    localized = [localize_deterministic_reviewer_prose(value) for value in values]
    if not reviewer_text_is_safe(localized, markers=markers, filenames=filenames):
        return [_SAFE_SUGGESTION]
    return localized


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
    return bool(
        re.search(
            r"(?:이\s*사건|현재\s*사건|문서|심사|finding|비교|document|case|review)",
            question,
            re.I,
        )
    )


def _classify_question(
    question: str,
    result: ReviewRunResultV3,
    *,
    recent_turns: list[Any] | tuple[Any, ...] = (),
) -> _QuestionPolicy:
    """Classify a turn before a provider can choose an answer scope.

    The classifier is deliberately small and deterministic.  Claude may write a
    grounded answer but cannot reinterpret a case question as general advice or
    turn a prohibited request into a case answer.
    """

    if _OUT_OF_SCOPE.search(question):
        return _QuestionPolicy("out_of_scope")
    if len(question) < 3 or question in {"?", "무엇", "확인"}:
        return _QuestionPolicy("clarification_needed")
    # A request for the review's legal basis remains evidence-grounded even
    # when the user explicitly says "this case".  Checking case references
    # first would otherwise route it to a comparison answer and hide the
    # linked Evidence from the assistant response.
    if _EVIDENCE_WORDS.search(question) and not _GENERAL_INTENT.search(question):
        return _QuestionPolicy("evidence_grounded" if result.evidence else "insufficient_evidence")
    case_reference = _question_matches_current_case(question, result, recent_turns=recent_turns)
    if case_reference and not _GENERAL_INTENT.search(question):
        if _SUBMISSION_STATUS_QUESTION.search(question) and not _has_submission_fact(result):
            return _QuestionPolicy("insufficient_evidence", "submission")
        if _MULTI_DOCUMENT_HS_QUESTION.search(question):
            return _QuestionPolicy("case_grounded", "multi_document_hs")
        if _CERTIFICATE_HS_QUESTION.search(question):
            return _QuestionPolicy("case_grounded", "certificate_hs")
        return _QuestionPolicy("case_grounded")
    if not _mentions_case_reference(question) and any(
        pattern.search(question) for pattern in _GENERAL_TOPIC_ALLOWLIST
    ):
        return _QuestionPolicy("general_customs_guidance")
    if not _mentions_case_reference(question) and _GENERAL_REQUEST.search(question):
        # The assistant may provide only the small general-guidance allowlist;
        # other generic customs questions should not be presented as case advice.
        return _QuestionPolicy("out_of_scope")
    has_case_support = bool(result.comparisons) or any(
        analysis.facts or analysis.line_items for analysis in result.document_analyses
    )
    return _QuestionPolicy("case_grounded" if has_case_support else "insufficient_evidence")


def _question_matches_current_case(
    question: str,
    result: ReviewRunResultV3,
    *,
    recent_turns: list[Any] | tuple[Any, ...],
) -> bool:
    """Resolve a question against the immutable snapshot before general routing."""

    normalized_question = _normalized_question(question)
    if _mentions_case_reference(question):
        return True
    for analysis in result.document_analyses:
        if any(
            _normalized_question(alias) in normalized_question
            for alias in _DOCUMENT_ALIASES.get(analysis.document_type, ())
        ):
            return True
        filename = _normalized_question(analysis.filename)
        if filename and filename in normalized_question:
            return True
        stem = _normalized_question(analysis.filename.rsplit(".", 1)[0])
        if len(stem) >= 4 and stem in normalized_question:
            return True
        for fact in _analysis_facts(analysis):
            value = fact.normalized_value or fact.raw_value
            if _is_hs_value(value) and str(value) in question:
                return True
    for finding in result.findings:
        phrase = _normalized_question(finding.message.removeprefix("필수 확인 항목:"))
        if len(phrase) >= 4 and phrase in normalized_question:
            return True
    for action in result.required_actions:
        phrase = _normalized_question(action.message.removeprefix("필수 확인 항목:"))
        if len(phrase) >= 4 and phrase in normalized_question:
            return True
    for turn in recent_turns:
        if getattr(turn, "role", None) != "assistant":
            continue
        content = getattr(turn, "content", "")
        if isinstance(content, str):
            normalized_content = _normalized_question(content)
            if normalized_question in normalized_content:
                return True
            if re.search(r"(?:그|이|해당)\s*(?:항목|내용|문서|결과|근거|사항)", question):
                if any(
                    _normalized_question(alias) in normalized_content
                    for aliases in _DOCUMENT_ALIASES.values()
                    for alias in aliases
                ):
                    return True
    return False


def _normalized_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _is_hs_value(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"\d{6,10}", value))


def _analysis_facts(analysis: Any) -> list[FactV3]:
    return [
        *analysis.facts,
        *(fact for item in analysis.line_items for fact in item.facts),
    ]


def _has_submission_fact(result: ReviewRunResultV3) -> bool:
    return any(
        re.search(r"(?:submit|submission|filing|declaration|제출|접수|신고)", fact.name, re.I)
        for analysis in result.document_analyses
        for fact in _analysis_facts(analysis)
    )


def _targeted_case_answer(
    policy: _QuestionPolicy,
    result: ReviewRunResultV3,
) -> tuple[str | None, list[AssistantCitationV1]]:
    if policy.target == "certificate_hs":
        return _certificate_hs_answer(result)
    if policy.target == "multi_document_hs":
        return _multi_document_hs_answer(result)
    return None, []


def _certificate_hs_answer(
    result: ReviewRunResultV3,
) -> tuple[str | None, list[AssistantCitationV1]]:
    certificate = next(
        (
            analysis
            for analysis in result.document_analyses
            if analysis.document_type == "certificate_of_origin"
        ),
        None,
    )
    if certificate is None:
        return None, []
    facts = [fact for fact in _analysis_facts(certificate) if fact.name in {"hs_code", "hs_codes"}]
    if facts:
        fact = facts[0]
        value = fact.normalized_value or fact.raw_value
        return (
            f"현재 심사 결과에서는 원산지증명서에 HS 코드 {value}가 추출되어 있습니다. "
            "담당자가 원문 인용과 함께 기재 내용을 최종 확인해 주세요.",
            [AssistantCitationV1(kind="fact", reference_id=fact.fact_id)],
        )
    comparison = next(
        (
            item
            for item in result.comparisons
            if item.rule_code == "hs_code_presence"
            and item.left.document_id == certificate.document_id
        ),
        None,
    )
    if comparison is None:
        return None, []
    return (
        "현재 심사 결과에서는 원산지증명서의 HS 코드를 확인하지 못했습니다. "
        "원문에 기재되어 있는지 담당자가 직접 확인해 주세요.",
        [AssistantCitationV1(kind="comparison", reference_id=comparison.comparison_id)],
    )


def _multi_document_hs_answer(
    result: ReviewRunResultV3,
) -> tuple[str | None, list[AssistantCitationV1]]:
    target_types = ("invoice", "packing_list", "certificate_of_origin")
    facts_by_type: dict[str, list[FactV3]] = {}
    for document_type in target_types:
        analysis = next(
            (entry for entry in result.document_analyses if entry.document_type == document_type),
            None,
        )
        if analysis is None:
            continue
        facts_by_type[document_type] = [
            fact for fact in _analysis_facts(analysis) if fact.name in {"hs_code", "hs_codes"}
        ]
    if len(facts_by_type) == len(target_types) and all(facts_by_type.values()):
        values = {
            fact.normalized_value or fact.raw_value
            for facts in facts_by_type.values()
            for fact in facts
        }
        if len(values) == 1:
            citations = [
                AssistantCitationV1(kind="fact", reference_id=facts[0].fact_id)
                for facts in facts_by_type.values()
            ]
            return (
                f"현재 심사 결과에서는 상업송장, 포장명세서, 원산지증명서에서 HS 코드 "
                f"{next(iter(values))}가 동일하게 추출되었습니다. "
                "다만 동일 품목 대응은 담당자가 문서 원문과 함께 최종 확인해 주세요.",
                citations,
            )
    citations = [
        AssistantCitationV1(kind="fact", reference_id=fact.fact_id)
        for facts in facts_by_type.values()
        for fact in facts[:1]
    ]
    if citations:
        return (
            "현재 심사 결과만으로는 세 문서의 HS 코드가 같은 품목을 뜻하는지 충분히 확인할 수 없습니다. "
            "문서별 품목 설명과 원문 인용을 함께 검토해 주세요.",
            citations,
        )
    return None, []


def _validate_scope_policy(
    answer: AssistantAnswerResponseV1, request: AssistantAnswerRequestV1
) -> None:
    expected = _classify_question(
        request.question.strip(), request.review_result, recent_turns=request.recent_turns
    )
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
    "provider_answer_allowed",
    "provider_snapshot",
    "response_from_provider_candidate",
]
