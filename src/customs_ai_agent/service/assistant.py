"""One bounded, grounded assistant turn over an immutable review snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

from pydantic import ValidationError

from customs_ai_agent.schemas.assistant import (
    AssistantAnswerRequestV1,
    AssistantAnswerResponseV1,
    AssistantAnswerSectionV1,
    AssistantAnswerCitationV1,
    AssistantCitationV1,
    AssistantCorrectionDraftV1,
    GENERAL_GUIDANCE_DISCLAIMER,
    GuidanceCitationV1,
    GuidanceEvidenceV1,
    GuidanceResolutionV1,
    KnowledgeAnswerRequestV1,
    KnowledgeAnswerResponseV1,
)
from customs_ai_agent.schemas.v3 import EngineV3, FactV3, ReviewRunResultV3
from customs_ai_agent.service.guidance_registry import (
    GuidanceRegistryError,
    guidance_registry_for_version,
    guidance_scope_directions,
    validate_guidance_source_provenance,
)
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
    r"(?:일반|기본|통관|관세|세율|hs\s*code|품목\s*분류|원산지|fta|과세가격|"
    r"인보이스|상업송장|packing\s*list|b\s*/?\s*l|선하증권|awb|항공화물운송장|"
    r"수입\s*요건|수출\s*요건|신고|납세|운송|무역\s*서류|인코텀즈|incoterms?|"
    r"fob|cif|exw|ddp|수출\s*제한|수출\s*허가)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _GuidanceTarget:
    """One closed official-guidance retrieval target.

    The user question is classified locally into this immutable pair.  Only
    the pair (not user prose) may cross the Agent→Data boundary.
    """

    topic: str
    scope: str
    label: str
    applicable_directions: tuple[Literal["import", "export"], ...]
    pattern: re.Pattern[str]


# The V2 registry distinguishes adjacent subjects that V1 collapsed into a
# broad topic.  These IDs are part of the Data contract, never model output.
_GUIDANCE_TARGETS: tuple[_GuidanceTarget, ...] = (
    _GuidanceTarget(
        "customs_clearance",
        "import_clearance_flow",
        "수입 통관 흐름",
        ("import",),
        re.compile(r"(?:수입\s*통관|수입.*통관)", re.I),
    ),
    _GuidanceTarget(
        "customs_clearance",
        "export_clearance_flow",
        "수출 통관 흐름",
        ("export",),
        re.compile(r"(?:수출\s*통관|수출.*통관)", re.I),
    ),
    _GuidanceTarget(
        "customs_declaration",
        "import_declaration",
        "수입 신고",
        ("import",),
        re.compile(r"(?:수입\s*신고|수입.*신고)", re.I),
    ),
    _GuidanceTarget(
        "customs_declaration",
        "export_declaration",
        "수출 신고",
        ("export",),
        re.compile(r"(?:수출\s*신고|수출.*신고|선적\s*신고)", re.I),
    ),
    _GuidanceTarget(
        "customs_duty",
        "customs_value_basics",
        "과세가격 기초",
        ("import",),
        re.compile(r"(?:과세가격|신고가격|관세평가)", re.I),
    ),
    _GuidanceTarget(
        "customs_duty",
        "duty_payment_basics",
        "관세·납세 기초",
        ("import",),
        re.compile(r"(?:관세(?:액|율)?|세율|납세|관세\s*납부)", re.I),
    ),
    _GuidanceTarget(
        "hs_classification",
        "advance_ruling_or_confirmation",
        "HS 사전심사·확인",
        ("import", "export"),
        re.compile(
            r"(?:hs\s*(?:사전\s*)?(?:심사|확인)|품목\s*분류\s*(?:사전\s*)?(?:심사|확인))", re.I
        ),
    ),
    _GuidanceTarget(
        "hs_classification",
        "classification_principles",
        "HS 품목분류 원칙",
        ("import", "export"),
        re.compile(r"(?:hs\s*code|hs코드|품목\s*분류)", re.I),
    ),
    _GuidanceTarget(
        "origin_verification",
        "origin_verification",
        "원산지 검증",
        ("import", "export"),
        re.compile(r"(?:원산지\s*(?:검증|조사|사후\s*검증))", re.I),
    ),
    _GuidanceTarget(
        "origin_verification",
        "origin_evidence",
        "원산지 증빙",
        ("import", "export"),
        re.compile(r"(?:원산지(?:\s*(?:증명|확인서|입증))?)", re.I),
    ),
    _GuidanceTarget(
        "trade_documents",
        "commercial_and_declaration_documents",
        "상업·신고 관련 서류",
        ("import", "export"),
        re.compile(
            r"(?:상업송장|인보이스|포장명세서|packing\s*list|commercial\s*invoice|무역\s*서류)",
            re.I,
        ),
    ),
    _GuidanceTarget(
        "import_requirements",
        "agency_confirmation_and_supporting_documents",
        "수입 요건·보완서류",
        ("import",),
        re.compile(r"(?:수입\s*요건|수입식품|수입\s*(?:허가|검사)|세관장\s*확인)", re.I),
    ),
    _GuidanceTarget(
        "export_requirements",
        "agency_confirmation_and_supporting_documents",
        "수출 요건·보완서류",
        ("export",),
        re.compile(r"(?:수출\s*요건|수출\s*(?:제한|허가|승인|검사)|세관장\s*확인)", re.I),
    ),
    _GuidanceTarget(
        "transport_documents",
        "bill_of_lading",
        "선하증권(B/L)",
        ("import", "export"),
        re.compile(r"(?:선하\s*증권|bill\s+of\s+lading|b\s*/?\s*l|해상\s*운송)", re.I),
    ),
    _GuidanceTarget(
        "transport_documents",
        "air_waybill",
        "항공화물운송장(AWB)",
        ("import", "export"),
        re.compile(r"(?:air\s*waybill|awb|항공\s*화물\s*운송장|항공\s*운송)", re.I),
    ),
    _GuidanceTarget(
        "fta_procedure",
        "origin_certificate",
        "FTA 원산지증명",
        ("import", "export"),
        re.compile(r"(?:fta.*원산지\s*(?:증명|확인서)|원산지\s*증명.*fta)", re.I),
    ),
    _GuidanceTarget(
        "fta_procedure",
        "preference_procedure",
        "FTA 특혜 절차",
        ("import", "export"),
        re.compile(r"(?:fta\s*(?:절차|적용|특혜)|특혜관세)", re.I),
    ),
    _GuidanceTarget(
        "trade_terms",
        "cost_risk_delivery_terms",
        "무역거래조건",
        ("import", "export"),
        re.compile(
            r"(?:incoterms?|인코텀즈|(?<![A-Za-z])fob(?![A-Za-z])|"
            r"(?<![A-Za-z])cif(?![A-Za-z])|(?<![A-Za-z])exw(?![A-Za-z])|"
            r"(?<![A-Za-z])ddp(?![A-Za-z]))",
            re.I,
        ),
    ),
)
_GUIDANCE_TARGET_BY_PAIR = {(item.topic, item.scope): item for item in _GUIDANCE_TARGETS}
_GUIDANCE_SCOPE_LABELS = {(item.topic, item.scope): item.label for item in _GUIDANCE_TARGETS}
_GENERIC_DIRECTIONAL_TARGETS: tuple[tuple[re.Pattern[str], tuple[str, str]], ...] = (
    (re.compile(r"통관\s*(?:절차|흐름|준비|방법|서류)?", re.I), ("customs_clearance", "")),
    (re.compile(r"(?:신고\s*(?:절차|방법|준비)?|신고\s*서류)", re.I), ("customs_declaration", "")),
    (re.compile(r"운송\s*서류", re.I), ("transport_documents", "")),
)
_CORRECTION_DRAFT = re.compile(
    r"(?:문서|document)\s*[:=]\s*(?P<document>[A-Za-z0-9._:-]+).*?"
    r"(?:필드|field)\s*[:=]\s*(?P<field>[A-Za-z0-9_:-]+).*?"
    r"(?:값|value)\s*[:=]\s*(?P<value>[^;\n]+).*?"
    r"(?:사유|reason)\s*[:=]\s*(?P<reason>[^;\n]+)",
    re.IGNORECASE | re.DOTALL,
)
_SUBMISSION_STATUS_QUESTION = re.compile(
    r"(?:제출|첨부|예정|접수|신고\s*(?:여부|완료)|수리|승인|반려|처리\s*상태)"
)
_CERTIFICATE_HS_QUESTION = re.compile(r"원산지\s*증명서.*(?:hs\s*코드|품목\s*분류)", re.I)
_MULTI_DOCUMENT_HS_QUESTION = re.compile(
    r"(?:동일\s*품목|같은\s*품목|일치).*?(?:hs\s*코드|품목\s*분류)|"
    r"(?:hs\s*코드|품목\s*분류).*?(?:동일|일치|같은)",
    re.I,
)
_GENERAL_INTENT = re.compile(r"(?:역할|차이|기본|일반|란|이란|절차|방법|개념)", re.I)
_APPLICATION_INTENT = re.compile(
    r"(?:적용|영향|보완|다음\s*단계|후속\s*조치|확인\s*(?:항목|사항|해야|할)|"
    r"무엇(?:을|을\s*더)?\s*확인|어떤\s*(?:항목|서류|사항).*확인|먼저\s*확인)",
    re.I,
)
_MIXED_GUIDANCE_INTENT = re.compile(
    r"(?:공식\s*(?:자료|안내)|일반\s*통관|안내|역할|차이|기본|절차|방법|개념|"
    r"적용|영향|보완|다음\s*단계|확인\s*(?:항목|사항|해야))",
    re.I,
)
_GENERAL_COVERAGE_GAP = re.compile(
    r"(?:환급|절세|회피|전략|정확한\s*세액|개별\s*허가|특정\s*업체|실시간\s*처리|"
    r"(?:개별|특정)\s*(?:품목|물품).*(?:허가|세액|hs|품목\s*분류)|"
    r"(?:hs|품목\s*분류).*(?:최종\s*)?확정)",
    re.I,
)
_FOREIGN_IMPORT_REGULATION = re.compile(
    r"(?:외국|해외|미국|중국|일본|유럽|eu|영국|수입국).{0,24}"
    r"(?:수입\s*(?:규제|요건|허가|통관)|통관\s*(?:규제|요건)|규제|허가)",
    re.I,
)
_GUIDANCE_SOURCE_ID = re.compile(r"^guide_[a-f0-9]{64}$")
_GUIDANCE_TOPICS = frozenset(
    {
        "customs_clearance",
        "customs_declaration",
        "customs_duty",
        "export_requirements",
        "fta_procedure",
        "hs_classification",
        "origin_verification",
        "trade_terms",
        "trade_documents",
        "transport_documents",
        "import_requirements",
    }
)
_GUIDANCE_RETRIEVAL_METHODS = frozenset({"registry", "dense", "bm25", "graph"})
_DOCUMENT_ALIASES = {
    "invoice": ("상업송장", "인보이스", "invoice", "commercial invoice"),
    "packing_list": ("포장명세서", "packing list", "packing"),
    "bill_of_lading": ("선하증권", "bill of lading", "b/l", "bl 번호"),
    "air_waybill": ("항공화물운송장", "air waybill", "awb"),
    "certificate_of_origin": ("원산지증명서", "certificate of origin", "c/o"),
    "export_declaration": ("수출신고증", "수출신고", "export declaration"),
    "supporting_evidence": ("보완 증빙", "제출 증빙", "supporting evidence"),
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
    general_topic: str | None = None
    general_scope: str | None = None
    requested_direction: Literal["import", "export", "both"] | None = None
    query_direction: Literal["import", "export", "both"] | None = None
    applicable_directions: tuple[Literal["import", "export"], ...] = ()
    mixed: bool = False
    application_intent: bool = False
    direction_inapplicable: bool = False
    clarification_targets: tuple[_GuidanceTarget, ...] = ()


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
            text=(
                _guidance_clarification_text(policy)
                if policy.clarification_targets
                else "어느 문서, 비교 항목 또는 법령 근거를 확인할지 구체적으로 알려 주세요."
            ),
            citations=[],
            suggestions=(
                _guidance_clarification_suggestions(policy)
                if policy.clarification_targets
                else [
                    "수량 비교 결과를 설명해 주세요.",
                    "연결된 법령 근거를 보여 주세요.",
                ]
            ),
        )
    if policy.scope == "general_customs_guidance":
        # This helper intentionally has no Data guidance context.  The
        # profile runtime resolves this route before calling us, so a direct
        # caller cannot accidentally emit an uncited legal/business answer.
        return _answer(
            request,
            engine=engine,
            scope="insufficient_evidence",
            text=(
                "공식 지식 스냅샷을 확인하지 않아 일반 통관 안내를 제공할 수 없습니다. "
                "출처와 기준일이 확인된 자료로 다시 조회해 주세요."
            ),
            citations=[],
            suggestions=["현재 사건의 문서 비교 결과를 설명해 주세요."],
        )
    if policy.scope == "insufficient_evidence":
        text = (
            _direction_scope_text(policy)
            if policy.direction_inapplicable
            else (
                "현재 심사 결과에는 수입신고확인증의 제출 또는 제출 예정 여부를 확인할 사실이 "
                "포함되어 있지 않습니다. 관련 증빙을 보완한 뒤 다시 확인해 주세요."
                if policy.target == "submission"
                else "현재 심사 결과에는 이 질문을 뒷받침할 충분한 문서 비교 또는 법령 근거가 없습니다."
            )
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
        if not case_citations:
            return _answer(
                request,
                engine=engine,
                scope="insufficient_evidence",
                text=case_text,
                citations=[],
                suggestions=["접수·신고 번호 또는 기관 처리 상태가 보이는 증빙을 등록해 주세요."],
            )
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


def unregistered_document_input_answer(
    request: AssistantAnswerRequestV1,
    *,
    engine: EngineV3,
) -> AssistantAnswerResponseV1:
    """Explain the Case-chat evidence boundary without retaining pasted text.

    This is deliberately emitted before a transcript, Data request, or Claude
    request.  A user message may remain in Backend's authorized conversation,
    but it cannot become a fact, citation, finding, or correction proposal.
    """

    answer = AssistantAnswerResponseV1(
        case_id=request.case_id,
        review_run_id=request.review_run_id,
        conversation_id=request.conversation_id,
        answer_scope="insufficient_evidence",
        direct_answer=(
            "붙여넣은 문서 원문은 현재 심사의 새 근거로 사용하지 않았습니다. "
            "문서를 등록하거나 교체한 뒤 새 심사를 실행하면, 그 결과를 기준으로 다시 확인할 수 있습니다."
        ),
        citations=[],
        suggested_questions=["등록된 문서의 현재 심사 결과에서 확인이 필요한 항목은 무엇인가요?"],
        correction_draft=None,
        case_input_notice="unregistered_document_text",
        engine=engine,
    )
    _validate_references(answer, request.review_result)
    return answer


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
        citations=_answer_citations(citations),
        suggested_questions=safe_suggestions,
        correction_draft=correction_draft,
        engine=engine,
    )
    _validate_references(answer, request.review_result)
    return answer


@dataclass(frozen=True)
class GuidanceContext:
    """Validated response from the one allowed Data guidance query."""

    status: Literal["covered", "no_coverage", "partial", "failed"]
    coverage: Literal["covered", "no_coverage", "partial", "unavailable"]
    snapshot_id: str | None
    as_of: date
    registry_version: Literal["1.0", "2.0"]
    registry_sha256: str
    topic: str | None
    scope: str | None
    applicable_directions: tuple[Literal["import", "export"], ...]
    evidence: tuple[GuidanceEvidenceV1, ...]


def _guidance_citations(context: GuidanceContext) -> list[GuidanceCitationV1]:
    """Project Data source records onto the lean public conversation citation."""

    if context.snapshot_id is None:
        return []
    return [
        GuidanceCitationV1(
            source_id=item.source_id,
            registry_source_id=item.registry_source_id,
            source_kind=item.source_kind,
            source_title=item.title,
            source_uri=item.uri,
            as_of=context.as_of,
            snapshot_id=context.snapshot_id,
            coverage=context.coverage,
        )
        for item in context.evidence
    ]


def _ordered_directions(
    values: object,
) -> tuple[Literal["import", "export"], ...]:
    return tuple(
        cast(Literal["import", "export"], direction)
        for direction in ("import", "export")
        if isinstance(values, (set, frozenset, tuple, list)) and direction in values
    )


def _guidance_scope_label(
    topic: str | None,
    scope: str | None,
    *,
    fallback: str,
) -> str:
    """Resolve a label from the full topic/scope pair, never scope alone.

    ``agency_confirmation_and_supporting_documents`` is intentionally shared
    by the import- and export-requirement topics.  Looking up a scope on its
    own would make an import answer display the export label (or vice versa).
    """

    if topic is not None and scope is not None:
        return _GUIDANCE_SCOPE_LABELS.get((topic, scope), fallback)
    return fallback


def guidance_resolution_for_policy(
    policy: _QuestionPolicy,
    *,
    context: GuidanceContext | None,
    as_of: date,
) -> GuidanceResolutionV1 | None:
    """Expose only safe lookup state for a selected general-guidance scope."""

    if (
        policy.general_topic is None
        or policy.general_scope is None
        or policy.requested_direction is None
        or not policy.applicable_directions
    ):
        return None
    if context is None:
        coverage: Literal["covered", "no_coverage", "partial", "unavailable"] = (
            "no_coverage" if policy.direction_inapplicable else "unavailable"
        )
        return GuidanceResolutionV1(
            topic=policy.general_topic,
            scope=policy.general_scope,
            requested_direction=policy.requested_direction,
            applicable_directions=list(policy.applicable_directions),
            coverage=coverage,
            snapshot_id=None,
            as_of=as_of,
            # A scope-specific request can only be satisfied by V2.  This is
            # a safe failure marker, not a claim that a V2 source was used.
            registry_version="2.0",
        )
    return GuidanceResolutionV1(
        topic=context.topic or policy.general_topic,
        scope=context.scope or policy.general_scope,
        requested_direction=policy.requested_direction,
        applicable_directions=list(context.applicable_directions or policy.applicable_directions),
        coverage=context.coverage,
        snapshot_id=context.snapshot_id,
        as_of=context.as_of,
        registry_version=context.registry_version,
    )


def guidance_registry_sha256_for_policy(
    policy: _QuestionPolicy,
    *,
    context: GuidanceContext | None,
) -> str | None:
    """Return internal-only registry provenance for a selected guidance scope.

    This digest never appears in the user DTO.  It lets Backend independently
    compare the Agent's accepted Data response to the exact copied V1/V2
    manifest before it projects an answer into persistent conversation state.
    For an unavailable, direction-inapplicable V2 lookup, it identifies the
    V2 manifest whose scope boundary Agent applied without claiming a Data
    source or snapshot was used.
    """

    if (
        policy.general_topic is None
        or policy.general_scope is None
        or policy.requested_direction is None
        or not policy.applicable_directions
    ):
        return None
    if context is not None:
        return context.registry_sha256
    return guidance_registry_for_version("2.0").registry_sha256


def _answer_citations(
    citations: list[AssistantCitationV1] | list[GuidanceCitationV1],
) -> list[AssistantAnswerCitationV1]:
    """Widen a homogeneous citation list at the Pydantic union boundary."""

    return [citation for citation in citations]


def guidance_context_from_data(
    response: Mapping[str, Any],
    *,
    profile_mode: Literal["local-development", "functional-uat"],
    expected_as_of: date | None = None,
    expected_trade_direction: Literal["import", "export", "both"] | None = None,
    expected_topic: str | None = None,
    expected_scope: str | None = None,
) -> GuidanceContext:
    """Validate a Data guidance response before it can ground user prose.

    Error bodies and raw retrieval query values are deliberately ignored.  A
    partial response can only be used when it still carries at least one
    validated official source; all other outcomes become an explicit coverage
    limitation instead of a model-completed answer.
    """

    registry_version = response.get("registry_version")
    if response.get("schema_version") != "3.0" or registry_version not in {"1.0", "2.0"}:
        raise ValueError("guidance_response_version_invalid")
    registry_sha256 = response.get("registry_sha256")
    if (
        not isinstance(registry_sha256, str)
        or re.fullmatch(r"[a-f0-9]{64}", registry_sha256) is None
    ):
        raise ValueError("guidance_response_registry_sha256_missing")
    try:
        registry = guidance_registry_for_version(registry_version)
    except GuidanceRegistryError:
        raise ValueError("guidance_response_registry_invalid") from None
    if registry_sha256 != registry.registry_sha256:
        raise ValueError("guidance_response_registry_sha256_mismatch")
    status = response.get("status")
    coverage = response.get("coverage")
    snapshot_id = response.get("snapshot_id")
    raw_as_of = response.get("as_of")
    results = response.get("results")
    if status not in {"covered", "no_coverage", "partial", "failed"}:
        raise ValueError("guidance_response_status_invalid")
    if coverage not in {"covered", "no_coverage", "partial", "unavailable"}:
        raise ValueError("guidance_response_coverage_invalid")
    if snapshot_id is not None and (
        not isinstance(snapshot_id, str) or not 1 <= len(snapshot_id) <= 512
    ):
        raise ValueError("guidance_response_snapshot_invalid")
    try:
        as_of = date.fromisoformat(str(raw_as_of))
    except ValueError:
        raise ValueError("guidance_response_as_of_invalid") from None
    if expected_as_of is not None and as_of != expected_as_of:
        raise ValueError("guidance_response_as_of_mismatch")
    if not isinstance(results, list) or len(results) > 3:
        raise ValueError("guidance_response_results_invalid")
    errors = response.get("errors", [])
    if not isinstance(errors, list) or len(errors) > 4:
        raise ValueError("guidance_response_errors_invalid")
    for error in errors:
        if (
            not isinstance(error, Mapping)
            or not isinstance(error.get("code"), str)
            or not 1 <= len(str(error["code"])) <= 128
            or not isinstance(error.get("message"), str)
            or not 1 <= len(str(error["message"])) <= 2_000
            or not isinstance(error.get("retryable"), bool)
        ):
            raise ValueError("guidance_response_errors_invalid")
    expected_coverage = {
        "covered": "covered",
        "no_coverage": "no_coverage",
        "partial": "partial",
        "failed": "unavailable",
    }[status]
    if coverage != expected_coverage:
        raise ValueError("guidance_response_status_coverage_invalid")
    if status == "partial" and not errors:
        raise ValueError("guidance_response_partial_without_error")
    if status == "failed" and not errors:
        raise ValueError("guidance_response_failed_without_error")
    if status in {"covered", "no_coverage"} and errors:
        raise ValueError("guidance_response_unexpected_error")
    response_topic = response.get("topic")
    if response_topic is not None and response_topic not in _GUIDANCE_TOPICS:
        raise ValueError("guidance_response_topic_invalid")
    if expected_topic is not None and response_topic != expected_topic:
        raise ValueError("guidance_response_topic_mismatch")
    raw_scope = response.get("guidance_scope")
    if registry_version == "1.0":
        if raw_scope is not None:
            raise ValueError("guidance_response_scope_invalid")
        response_scope = "legacy_topic"
    elif not isinstance(raw_scope, str) or not raw_scope:
        raise ValueError("guidance_response_scope_invalid")
    else:
        response_scope = raw_scope
    if (
        registry_version == "2.0"
        and expected_scope is not None
        and response_scope != expected_scope
    ):
        raise ValueError("guidance_response_scope_mismatch")
    applicable_directions: frozenset[Literal["import", "export"]]
    if response_topic is not None:
        try:
            applicable_directions = guidance_scope_directions(
                registry_version=registry_version,
                topic=response_topic,
                scope=response_scope,
            )
        except GuidanceRegistryError:
            raise ValueError("guidance_response_scope_invalid") from None
    else:
        applicable_directions = frozenset()
    evidence: list[GuidanceEvidenceV1] = []
    source_ids: set[str] = set()
    for result in results:
        if not isinstance(result, Mapping) or snapshot_id is None:
            raise ValueError("guidance_response_source_invalid")
        source_id = result.get("source_id")
        retrieval_methods = result.get("retrieval_methods")
        if (
            not isinstance(source_id, str)
            or _GUIDANCE_SOURCE_ID.fullmatch(source_id) is None
            or source_id in source_ids
            or not isinstance(retrieval_methods, list)
            or not 1 <= len(retrieval_methods) <= 4
            or len(retrieval_methods) != len(set(retrieval_methods))
            or any(method not in _GUIDANCE_RETRIEVAL_METHODS for method in retrieval_methods)
            or result.get("topic") != response_topic
        ):
            raise ValueError("guidance_response_source_invalid")
        direction = result.get("direction")
        if direction not in {"import", "export", "both"}:
            raise ValueError("guidance_response_direction_invalid")
        if expected_trade_direction in {"import", "export"} and direction not in {
            expected_trade_direction,
            "both",
        }:
            raise ValueError("guidance_response_direction_mismatch")
        # A request for both directions may legitimately contain a page whose
        # registered binding is import-only or export-only.  The selected V2
        # scope and every individual source are still provenance-validated
        # below; Data's covered/partial status is what asserts whether the
        # full requested bundle was available.  Requiring every citation to
        # spell ``both`` would incorrectly reject a complete bundle made of
        # direction-specific official pages.
        candidate = {
            "source_id": result.get("source_id"),
            "registry_source_id": result.get("registry_source_id"),
            "source_kind": result.get("source_kind"),
            "title": result.get("title"),
            "uri": result.get("uri"),
            "topic": result.get("topic"),
            "direction": result.get("direction"),
            "excerpt": result.get("excerpt"),
            "as_of": as_of,
            "snapshot_id": snapshot_id,
            "law_id": result.get("law_id"),
            "law_name": result.get("law_name"),
            "article_no": result.get("article_no"),
            "version_hash": result.get("version_hash"),
        }
        try:
            item = GuidanceEvidenceV1.model_validate(candidate)
        except ValidationError:
            raise ValueError("guidance_response_source_invalid") from None
        registry_source_id = result.get("registry_source_id")
        if not isinstance(registry_source_id, str):
            raise ValueError("guidance_response_registry_source_missing")
        try:
            validate_guidance_source_provenance(
                profile_mode=profile_mode,
                registry_version=registry_version,
                registry_sha256=registry_sha256,
                registry_source_id=registry_source_id,
                source_kind=item.source_kind,
                title=item.title,
                uri=item.uri,
                topic=item.topic,
                direction=item.direction,
                law_id=item.law_id,
                law_name=item.law_name,
                article_no=item.article_no,
                guidance_scope=response_scope if registry_version == "2.0" else None,
            )
        except GuidanceRegistryError:
            raise ValueError("guidance_response_provenance_invalid") from None
        source_ids.add(item.source_id)
        evidence.append(item)
    if status in {"covered", "partial"} and not evidence:
        raise ValueError("guidance_response_covered_without_sources")
    if status in {"no_coverage", "failed"} and evidence:
        raise ValueError("guidance_response_uncovered_with_sources")
    if status in {"covered", "partial"} and snapshot_id is None:
        raise ValueError("guidance_response_covered_without_snapshot")
    return GuidanceContext(
        status=status,
        coverage=coverage,
        snapshot_id=snapshot_id,
        as_of=as_of,
        registry_version=registry_version,
        registry_sha256=registry_sha256,
        topic=response_topic if isinstance(response_topic, str) else None,
        scope=response_scope if isinstance(response_topic, str) else None,
        applicable_directions=_ordered_directions(applicable_directions),
        evidence=tuple(evidence),
    )


_TOPIC_LABELS = {
    "customs_clearance": "수출입 통관·신고 절차",
    "hs_classification": "HS 품목분류",
    "customs_duty": "과세가격·관세 기초",
    "origin_verification": "원산지 확인",
    "fta_procedure": "FTA 절차",
    "trade_terms": "무역 거래조건",
    "trade_documents": "무역 서류",
    "transport_documents": "운송 서류",
    "import_requirements": "수입 요건",
    "export_requirements": "수출 요건",
    "customs_declaration": "수출입 신고 절차",
}
_KNOWLEDGE_PRESENTATION_STYLES = frozenset({"overview", "checklist", "source_focus"})
_KNOWLEDGE_SUGGESTION_QUESTIONS = {
    "customs_clearance": "수입 통관 절차에서 기본으로 확인할 서류를 알려 주세요.",
    "customs_declaration": "수출입 신고 전에 일반적으로 확인할 사항을 알려 주세요.",
    "customs_duty": "과세가격과 관세를 확인할 때 공식 자료에서 먼저 볼 항목을 알려 주세요.",
    "hs_classification": "HS 품목분류를 확인하는 기본 절차를 알려 주세요.",
    "origin_verification": "원산지증명서는 어떤 경우에 확인하나요?",
    "trade_documents": "상업송장과 포장명세서는 어떤 역할을 하나요?",
    "import_requirements": "수입 요건은 어떤 공식 자료에서 확인하나요?",
    "export_requirements": "수출 요건은 어떤 공식 자료에서 확인하나요?",
    "transport_documents": "B/L과 AWB는 어떤 경우에 구분해 확인하나요?",
    "fta_procedure": "FTA 특혜 절차에서 원산지 확인은 어떻게 시작하나요?",
    "trade_terms": "FOB와 CIF의 기본 차이를 공식 안내 범위에서 알려 주세요.",
}


def _knowledge_suggestions(
    topic: str | None,
    *,
    selected_topics: list[str] | None = None,
) -> list[str]:
    """Render only fixed, code-owned follow-up questions."""

    candidates = selected_topics or [topic or "", "trade_documents", "customs_clearance"]
    questions: list[str] = []
    for candidate in candidates:
        question = _KNOWLEDGE_SUGGESTION_QUESTIONS.get(candidate)
        if question is not None and question not in questions:
            questions.append(question)
        if len(questions) == 3:
            break
    return questions or [_KNOWLEDGE_SUGGESTION_QUESTIONS["customs_clearance"]]


def _knowledge_source_slots(
    context: GuidanceContext,
    slots: list[int] | None = None,
) -> tuple[int, ...]:
    """Validate a provider-selected order without exposing a provider string."""

    if slots is None:
        return tuple(range(min(3, len(context.evidence))))
    if (
        not 1 <= len(slots) <= 3
        or any(isinstance(slot, bool) or not isinstance(slot, int) for slot in slots)
        or len(set(slots)) != len(slots)
        or any(slot < 0 or slot >= len(context.evidence) for slot in slots)
    ):
        raise ValueError("knowledge_provider_source_slots_invalid")
    return tuple(slots)


def _bounded_official_excerpt(text: str) -> str:
    """Keep Data-approved public text readable without adding an interpretation."""

    compact = " ".join(text.split())
    return compact if len(compact) <= 420 else compact[:419].rstrip() + "…"


def _knowledge_template(
    context: GuidanceContext,
    *,
    presentation_style: str,
    source_slots: list[int] | None = None,
) -> str:
    """Render the entire user-visible guidance answer from bounded Data fields.

    The Claude adapter may choose only the fixed presentation style and source
    slot order.  It cannot supply a legal claim, a recommendation sentence,
    source metadata, or a follow-up question for the public response.
    """

    if presentation_style not in _KNOWLEDGE_PRESENTATION_STYLES:
        raise ValueError("knowledge_provider_presentation_style_invalid")
    topic = context.topic or ""
    topic_label = _guidance_scope_label(
        topic,
        context.scope,
        fallback=_TOPIC_LABELS.get(topic, "통관 안내"),
    )
    introductions = {
        "overview": (
            f"공식 지식 스냅샷에 등록된 {topic_label} 자료 {len(context.evidence)}건을 "
            "기준으로 정리한 일반 안내입니다."
        ),
        "checklist": (
            f"{topic_label}와 관련해 공식 자료에서 먼저 확인할 범위를 정리한 일반 안내입니다. "
            f"등록된 자료는 {len(context.evidence)}건입니다."
        ),
        "source_focus": (
            f"{topic_label}에 관한 등록 공식 자료의 발췌를 중심으로 확인할 수 있는 일반 안내입니다."
        ),
    }
    excerpt_lines = ["공식 자료 발췌"]
    for slot in _knowledge_source_slots(context, source_slots):
        source = context.evidence[slot]
        excerpt_lines.append(f"- {source.title}: {_bounded_official_excerpt(source.excerpt)}")
    partial_note = (
        " 일부 공식 자료 범위가 제한되어 있으므로, 적용 요건은 인용된 원문으로 추가 확인해 주세요."
        if context.coverage == "partial"
        else ""
    )
    return "\n\n".join(
        [
            introductions[presentation_style],
            "확인 절차\n1. 인용된 공식 자료와 기준일을 확인합니다.\n"
            "2. 실제 신고·허가·납세 전에는 적용 대상과 요건을 담당자가 원문으로 확인합니다.",
            "\n".join(excerpt_lines),
            f"{GENERAL_GUIDANCE_DISCLAIMER}{partial_note}",
        ]
    )


def deterministic_knowledge_answer(
    request: KnowledgeAnswerRequestV1,
    *,
    engine: EngineV3,
    context: GuidanceContext | None,
) -> KnowledgeAnswerResponseV1:
    """Return a source-bounded standalone answer without a Case snapshot."""

    policy = knowledge_question_policy(request.question.strip(), trade_type=request.trade_type)
    if policy.scope == "out_of_scope":
        return KnowledgeAnswerResponseV1(
            conversation_id=request.conversation_id,
            answer_scope="out_of_scope",
            direct_answer=(
                "통관 지식 도우미는 대한민국 수출입·통관·원산지·FTA·무역서류 범위의 "
                "공식 자료에 근거한 안내만 제공합니다. 웹 검색, 외부 시스템 조회, "
                "비밀값 요청은 처리하지 않습니다."
            ),
            suggested_questions=["수입 통관 절차에서 기본으로 확인할 서류를 알려 주세요."],
            coverage="unavailable",
            snapshot_id=None,
            as_of=request.as_of,
            engine=engine,
        )
    if policy.scope == "clarification_needed":
        return KnowledgeAnswerResponseV1(
            conversation_id=request.conversation_id,
            answer_scope="clarification_needed",
            direct_answer=_guidance_clarification_text(policy),
            suggested_questions=_guidance_clarification_suggestions(policy),
            coverage="unavailable",
            snapshot_id=None,
            as_of=request.as_of,
            engine=engine,
        )
    resolution = guidance_resolution_for_policy(policy, context=context, as_of=request.as_of)
    registry_sha256 = guidance_registry_sha256_for_policy(policy, context=context)
    if policy.direction_inapplicable:
        return KnowledgeAnswerResponseV1(
            conversation_id=request.conversation_id,
            answer_scope="insufficient_evidence",
            direct_answer=_direction_scope_text(policy),
            suggested_questions=[
                "대한민국 수입 또는 수출 중 적용 방향을 선택해 다시 질문해 주세요."
            ],
            coverage="no_coverage",
            snapshot_id=None,
            as_of=request.as_of,
            guidance_resolution=resolution,
            guidance_registry_sha256=registry_sha256,
            engine=engine,
        )
    if context is None or context.status in {"no_coverage", "failed"}:
        return KnowledgeAnswerResponseV1(
            conversation_id=request.conversation_id,
            answer_scope="insufficient_evidence",
            direct_answer=(
                "현재 공식 지식 스냅샷에서 이 질문을 뒷받침할 자료를 확인할 수 없습니다. "
                "추측으로 답하지 않으며, 적용 법령과 최신 기관 안내는 담당자가 공식 원문으로 확인해 주세요."
            ),
            suggested_questions=["수입 통관 절차의 기본 서류를 알려 주세요."],
            coverage=context.coverage if context is not None else "unavailable",
            snapshot_id=context.snapshot_id if context is not None else None,
            as_of=context.as_of if context is not None else request.as_of,
            guidance_resolution=resolution,
            guidance_registry_sha256=registry_sha256,
            engine=engine,
        )
    topic = context.topic or policy.general_topic
    direct_answer = _knowledge_template(context, presentation_style="overview")
    return KnowledgeAnswerResponseV1(
        conversation_id=request.conversation_id,
        answer_scope="general_customs_guidance",
        direct_answer=direct_answer,
        citations=_guidance_citations(context),
        answer_sections=[
            AssistantAnswerSectionV1(
                kind="general_guidance",
                title="일반 통관 안내",
                content=direct_answer,
                citations=_answer_citations(_guidance_citations(context)),
            )
        ],
        suggested_questions=_knowledge_suggestions(topic),
        coverage=context.coverage,
        snapshot_id=context.snapshot_id,
        as_of=context.as_of,
        guidance_resolution=resolution,
        guidance_registry_sha256=registry_sha256,
        engine=engine,
    )


def response_from_knowledge_provider_candidate(
    candidate: Mapping[str, Any],
    request: KnowledgeAnswerRequestV1,
    *,
    engine: EngineV3,
    context: GuidanceContext,
) -> KnowledgeAnswerResponseV1:
    """Accept only bounded presentation choices; never provider prose.

    Claude never chooses source IDs or supplies user-visible legal prose.  The
    Data response is already provenance-validated and Agent code renders the
    complete Korean answer and citation set from those fields alone.
    """

    fallback = deterministic_knowledge_answer(request, engine=engine, context=context)
    if fallback.answer_scope != "general_customs_guidance":
        raise ValueError("knowledge_provider_without_covered_sources")
    if set(candidate) != {"presentation_style", "source_slots", "suggestion_topics"}:
        raise ValueError("knowledge_provider_output_invalid")
    presentation_style = candidate.get("presentation_style")
    source_slots = candidate.get("source_slots")
    suggestion_topics = candidate.get("suggestion_topics")
    if (
        not isinstance(presentation_style, str)
        or presentation_style not in _KNOWLEDGE_PRESENTATION_STYLES
        or not isinstance(source_slots, list)
        or not isinstance(suggestion_topics, list)
        or len(suggestion_topics) > 3
        or not all(isinstance(topic, str) for topic in suggestion_topics)
        or len(suggestion_topics) != len(set(suggestion_topics))
        or any(topic not in _KNOWLEDGE_SUGGESTION_QUESTIONS for topic in suggestion_topics)
    ):
        raise ValueError("knowledge_provider_output_invalid")
    safe_source_slots = cast(list[int], source_slots)
    safe_suggestion_topics = cast(list[str], suggestion_topics)
    try:
        direct_answer = _knowledge_template(
            context,
            presentation_style=presentation_style,
            source_slots=safe_source_slots,
        )
        return KnowledgeAnswerResponseV1(
            conversation_id=request.conversation_id,
            answer_scope="general_customs_guidance",
            direct_answer=direct_answer,
            citations=_guidance_citations(context),
            answer_sections=[
                AssistantAnswerSectionV1(
                    kind="general_guidance",
                    title="일반 통관 안내",
                    content=direct_answer,
                    citations=_answer_citations(_guidance_citations(context)),
                )
            ],
            suggested_questions=_knowledge_suggestions(
                context.topic,
                selected_topics=safe_suggestion_topics,
            ),
            coverage=context.coverage,
            snapshot_id=context.snapshot_id,
            as_of=context.as_of,
            guidance_resolution=fallback.guidance_resolution,
            guidance_registry_sha256=fallback.guidance_registry_sha256,
            engine=engine,
        )
    except ValidationError:
        raise ValueError("knowledge_provider_output_invalid") from None


def guidance_answer_for_case(
    request: AssistantAnswerRequestV1,
    *,
    engine: EngineV3,
    context: GuidanceContext | None,
    standalone_answer: KnowledgeAnswerResponseV1 | None = None,
) -> AssistantAnswerResponseV1:
    """Create an official-guidance response while retaining Case identity.

    This is used when a Case conversation asks a purely general question.  It
    remains a conversation bound to that Case in Backend, but it receives no
    document-derived citation and never presents its statement as a Case
    conclusion.
    """

    standalone = standalone_answer or deterministic_knowledge_answer(
        KnowledgeAnswerRequestV1(
            conversation_id=request.conversation_id,
            question=request.question,
            trade_type=request.review_result.trade_type,
            as_of=request.review_result.review_as_of,
            recent_turns=request.recent_turns,
        ),
        engine=engine,
        context=context,
    )
    if standalone.answer_scope != "general_customs_guidance":
        answer = _answer(
            request,
            engine=engine,
            scope=standalone.answer_scope,
            text=standalone.direct_answer,
            citations=[],
            suggestions=standalone.suggested_questions,
        )
        return answer.model_copy(
            update={
                "guidance_resolution": standalone.guidance_resolution,
                "guidance_registry_sha256": standalone.guidance_registry_sha256,
            }
        )
    section = AssistantAnswerSectionV1(
        kind="general_guidance",
        title="일반 통관 안내",
        content=standalone.direct_answer,
        citations=_answer_citations(standalone.citations),
    )
    answer = AssistantAnswerResponseV1(
        case_id=request.case_id,
        review_run_id=request.review_run_id,
        conversation_id=request.conversation_id,
        answer_scope="general_customs_guidance",
        direct_answer=standalone.direct_answer,
        citations=_answer_citations(standalone.citations),
        answer_sections=[section],
        suggested_questions=standalone.suggested_questions,
        correction_draft=None,
        guidance_resolution=standalone.guidance_resolution,
        guidance_registry_sha256=standalone.guidance_registry_sha256,
        engine=engine,
    )
    _validate_references(answer, request.review_result)
    return answer


def combine_case_and_guidance(
    case_answer: AssistantAnswerResponseV1,
    *,
    context: GuidanceContext | None,
    policy: _QuestionPolicy,
    as_of: date,
) -> AssistantAnswerResponseV1:
    """Expose a mixed request as separate, evidence-bound sections.

    The legacy top-level fields retain the case answer, so a consumer that has
    not yet adopted sections cannot mistake general guidance for a case fact.
    The optional third section is deterministic: it never combines source text
    into a legal conclusion and is emitted only when both citation domains are
    present.
    """

    resolution = guidance_resolution_for_policy(policy, context=context, as_of=as_of)
    registry_sha256 = guidance_registry_sha256_for_policy(policy, context=context)
    case_citations = [
        citation for citation in case_answer.citations if isinstance(citation, AssistantCitationV1)
    ]
    if not case_citations:
        return case_answer.model_copy(
            update={
                "guidance_resolution": resolution,
                "guidance_registry_sha256": registry_sha256,
            }
        )
    case_section = AssistantAnswerSectionV1(
        kind="case_analysis",
        title="현재 사건 기준",
        content=case_answer.direct_answer,
        citations=_answer_citations(case_citations),
    )
    if context is None or context.status not in {"covered", "partial"}:
        return AssistantAnswerResponseV1.model_validate(
            case_answer.model_dump(mode="json", warnings=False)
            | {
                "answer_sections": [case_section.model_dump(mode="json", warnings=False)],
                "guidance_resolution": (
                    resolution.model_dump(mode="json", warnings=False)
                    if resolution is not None
                    else None
                ),
                "guidance_registry_sha256": registry_sha256,
            }
        )
    general_content = _knowledge_template(context, presentation_style="overview")
    general_citations = _guidance_citations(context)
    guidance_section = AssistantAnswerSectionV1(
        kind="general_guidance",
        title="일반 통관 안내",
        content=general_content,
        citations=_answer_citations(general_citations),
    )
    sections: list[AssistantAnswerSectionV1] = [case_section, guidance_section]
    if policy.application_intent:
        application_citations: list[AssistantAnswerCitationV1] = [
            case_citations[0],
            general_citations[0],
        ]
        scope_label = _guidance_scope_label(
            policy.general_topic,
            policy.general_scope,
            fallback="일반 통관 안내",
        )
        partial_note = (
            " 공식 자료 범위가 일부이므로, 아래 항목은 담당자가 인용된 원문과 함께 추가 확인해야 합니다."
            if context.coverage == "partial"
            else ""
        )
        sections.append(
            AssistantAnswerSectionV1(
                kind="case_application",
                title="사건 적용·다음 확인",
                content=(
                    f"현재 사건의 인용된 문서·비교 결과와 {scope_label} 공식 안내를 함께 보면, "
                    "다음 확인을 담당자가 진행해 주세요.\n"
                    "1. 현재 사건의 인용 근거가 일반 안내의 적용 대상과 맞는지 확인합니다.\n"
                    "2. 인용된 공식 자료의 기준일과 실제 신고·보완 요건을 원문으로 확인합니다.\n"
                    "3. 이 안내만으로 신고 수리·허가·FTA 적용 또는 통관 결과를 확정하지 않습니다."
                    f"{partial_note}"
                ),
                citations=application_citations,
            )
        )
    return AssistantAnswerResponseV1.model_validate(
        case_answer.model_dump(mode="json", warnings=False)
        | {
            "answer_sections": [
                section.model_dump(mode="json", warnings=False) for section in sections
            ],
            "citations": [
                *(item.model_dump(mode="json", warnings=False) for item in case_answer.citations),
                *(item.model_dump(mode="json", warnings=False) for item in general_citations),
            ],
            "guidance_resolution": (
                resolution.model_dump(mode="json", warnings=False)
                if resolution is not None
                else None
            ),
            "guidance_registry_sha256": registry_sha256,
        }
    )


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
    case_citations = [
        citation for citation in answer.citations if isinstance(citation, AssistantCitationV1)
    ]
    if any(citation.reference_id not in known[citation.kind] for citation in case_citations):
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


def _guidance_targets_for_question(question: str) -> tuple[_GuidanceTarget, ...]:
    """Return every independent closed retrieval target in a question.

    A first-regex-wins router hid the second subject in questions such as
    "B/L과 AWB".  We instead discover every target, collapse only a more
    specific spelling of the *same* target, and ask the user to choose when
    independent scopes remain.
    """

    selected = [target for target in _GUIDANCE_TARGETS if target.pattern.search(question)]
    selected_pairs = {(target.topic, target.scope) for target in selected}
    # A specific HS confirmation wording also matches the generic
    # classification rule; it remains one scope, not a false compound query.
    if ("hs_classification", "advance_ruling_or_confirmation") in selected_pairs:
        selected = [
            target
            for target in selected
            if (target.topic, target.scope) != ("hs_classification", "classification_principles")
        ]
    # FTA-origin certificate wording is governed by the dedicated FTA scope;
    # do not also turn its ordinary "원산지" substring into a second query.
    if ("fta_procedure", "origin_certificate") in selected_pairs:
        selected = [
            target
            for target in selected
            if (target.topic, target.scope) != ("origin_verification", "origin_evidence")
        ]
    # In a question about the FTA preference procedure, a bare "원산지" is
    # normally an attribute of that procedure rather than a second requested
    # consultation.  Keep an explicit certificate/confirmation/verification
    # request independent so "FTA 특혜와 원산지증명" still asks the user to
    # choose instead of silently selecting a first scope.
    if (
        ("fta_procedure", "preference_procedure") in selected_pairs
        and ("origin_verification", "origin_evidence") in selected_pairs
        and re.search(r"원산지\s*(?:증명|확인서|검증|조사)", question, re.I) is None
    ):
        selected = [
            target
            for target in selected
            if (target.topic, target.scope) != ("origin_verification", "origin_evidence")
        ]
    selected_topics = {target.topic for target in selected}
    for pattern, (topic, _unused_scope) in _GENERIC_DIRECTIONAL_TARGETS:
        if pattern.search(question) and topic not in selected_topics:
            selected.extend(target for target in _GUIDANCE_TARGETS if target.topic == topic)
            selected_topics.add(topic)
    unique: list[_GuidanceTarget] = []
    seen: set[tuple[str, str]] = set()
    for target in selected:
        key = (target.topic, target.scope)
        if key not in seen:
            seen.add(key)
            unique.append(target)
    return tuple(unique)


def _requested_direction(
    question: str,
    *,
    fallback: Literal["import", "export", "both"],
) -> Literal["import", "export", "both"]:
    has_import = bool(re.search(r"수입", question, re.I))
    has_export = bool(re.search(r"수출", question, re.I))
    if "수출입" in question or (has_import and has_export):
        return "both"
    if has_import:
        return "import"
    if has_export:
        return "export"
    return fallback


def _policy_for_guidance_target(
    target: _GuidanceTarget,
    *,
    requested_direction: Literal["import", "export", "both"],
    scope: _AnswerScope,
    mixed: bool = False,
    application_intent: bool = False,
) -> _QuestionPolicy:
    applicable = target.applicable_directions
    if requested_direction == "both":
        query_direction: Literal["import", "export", "both"] = "both"
        # ``both`` is a real request direction, not permission to silently
        # narrow a Korea import-only rule to its import half.  Asking for
        # import/export together where the selected scope is import-only (or
        # export-only) must surface an explicit scope limitation without a
        # Data lookup or model completion.
        direction_inapplicable = len(applicable) != 2
    elif requested_direction in applicable:
        query_direction = requested_direction
        direction_inapplicable = False
    else:
        # Do not substitute an import-only Korean rule for a user asking an
        # export question (or the reverse).  The runtime presents the scope
        # boundary without calling Data or Claude.
        query_direction = requested_direction
        direction_inapplicable = True
    return _QuestionPolicy(
        scope=scope,
        general_topic=target.topic,
        general_scope=target.scope,
        requested_direction=requested_direction,
        query_direction=query_direction,
        applicable_directions=applicable,
        mixed=mixed,
        application_intent=application_intent,
        direction_inapplicable=direction_inapplicable,
    )


def _guidance_clarification_text(policy: _QuestionPolicy) -> str:
    labels = [target.label for target in policy.clarification_targets[:3]]
    choices = ", ".join(labels)
    if choices:
        return (
            "한 번에 하나의 공식 통관 주제 범위만 조회할 수 있습니다. "
            f"먼저 {choices} 중 확인할 한 가지를 선택해 주세요."
        )
    return "수입·수출, HS 품목분류, 원산지·FTA, 통관 서류 중 확인할 한 가지 주제를 알려 주세요."


def _guidance_clarification_suggestions(policy: _QuestionPolicy) -> list[str]:
    """Offer only code-owned single-scope choices after a compound request."""

    suggestions: list[str] = []
    for target in policy.clarification_targets[:3]:
        question = f"{target.label} 공식 안내를 알려 주세요."
        if question not in suggestions:
            suggestions.append(question)
    return suggestions or ["확인하려는 통관 주제 한 가지를 알려 주세요."]


def _direction_scope_text(policy: _QuestionPolicy) -> str:
    label = _guidance_scope_label(
        policy.general_topic,
        policy.general_scope,
        fallback="선택한 통관 안내",
    )
    requested = {"import": "수입", "export": "수출", "both": "수출입"}.get(
        policy.requested_direction or "both", "수출입"
    )
    applicable = "·".join(
        {"import": "수입", "export": "수출"}[direction]
        for direction in policy.applicable_directions
    )
    return (
        f"{label} 공식 안내는 대한민국 기준 {applicable} 범위에만 등록되어 있습니다. "
        f"요청한 {requested} 방향에는 적용 범위가 없어 현재 공식 지식 스냅샷으로 안내할 수 없습니다."
    )


def guidance_topic_for_question(question: str) -> str | None:
    """Return a topic only when the question contains one independent scope."""

    targets = _guidance_targets_for_question(question)
    return targets[0].topic if len(targets) == 1 else None


def knowledge_question_policy(
    question: str,
    *,
    trade_type: Literal["import", "export", "both"] = "both",
) -> _QuestionPolicy:
    """Classify a standalone knowledge turn without a Case snapshot.

    The separate knowledge assistant has no document or review-result access.
    Consequently it can only ask Data for one approved official-source topic,
    request clarification, or refuse a prohibited/uncovered request.
    """

    if _OUT_OF_SCOPE.search(question):
        return _QuestionPolicy("out_of_scope")
    if _FOREIGN_IMPORT_REGULATION.search(question):
        return _QuestionPolicy("out_of_scope")
    if len(question.strip()) < 3 or question.strip() in {"?", "무엇", "확인"}:
        return _QuestionPolicy("clarification_needed")
    if _GENERAL_COVERAGE_GAP.search(question):
        return _QuestionPolicy("insufficient_evidence")
    targets = _guidance_targets_for_question(question)
    if len(targets) > 1:
        return _QuestionPolicy("clarification_needed", clarification_targets=targets)
    if len(targets) == 1:
        target = targets[0]
        requested_direction = _requested_direction(question, fallback=trade_type)
        policy = _policy_for_guidance_target(
            target,
            requested_direction=requested_direction,
            scope="general_customs_guidance",
        )
        if policy.direction_inapplicable:
            return _QuestionPolicy(
                "insufficient_evidence",
                general_topic=policy.general_topic,
                general_scope=policy.general_scope,
                requested_direction=policy.requested_direction,
                query_direction=policy.query_direction,
                applicable_directions=policy.applicable_directions,
                direction_inapplicable=True,
            )
        return policy
    return _QuestionPolicy("out_of_scope")


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
    if _FOREIGN_IMPORT_REGULATION.search(question):
        return _QuestionPolicy("out_of_scope")
    if len(question) < 3 or question in {"?", "무엇", "확인"}:
        return _QuestionPolicy("clarification_needed")
    case_reference = _question_matches_current_case(question, result, recent_turns=recent_turns)
    explicit_case_reference = _mentions_case_reference(question)
    # A request for the review's legal basis remains evidence-grounded even
    # when the user explicitly says "this case".  Checking a general topic
    # first would otherwise hide the linked Evidence behind a broad guide.
    if (
        _EVIDENCE_WORDS.search(question)
        and not _GENERAL_INTENT.search(question)
        and (
            case_reference
            or bool(
                re.search(r"(?:연결된|현재|이|해당)\s*(?:법령|근거)|심사.*(?:법령|근거)", question)
            )
        )
    ):
        return _QuestionPolicy("evidence_grounded" if result.evidence else "insufficient_evidence")
    # Submission and document-status questions remain case-first even when
    # they contain general words such as "절차".  A general guide can never
    # create filing, receipt, acceptance, or rejection evidence.
    if case_reference and _SUBMISSION_STATUS_QUESTION.search(question):
        return _QuestionPolicy(
            "case_grounded" if _has_submission_context(result) else "insufficient_evidence",
            "submission",
        )
    targets = _guidance_targets_for_question(question)
    is_explicit_mixed_scope = (
        case_reference
        and explicit_case_reference
        and len(targets) == 1
        and bool(_MIXED_GUIDANCE_INTENT.search(question))
    )
    if case_reference and not is_explicit_mixed_scope:
        if _MULTI_DOCUMENT_HS_QUESTION.search(question):
            return _QuestionPolicy("case_grounded", "multi_document_hs")
        if _CERTIFICATE_HS_QUESTION.search(question):
            return _QuestionPolicy("case_grounded", "certificate_hs")
        return _QuestionPolicy("case_grounded")
    if _GENERAL_COVERAGE_GAP.search(question):
        return _QuestionPolicy("insufficient_evidence")
    if len(targets) > 1:
        return _QuestionPolicy("clarification_needed", clarification_targets=targets)
    if len(targets) == 1:
        target = targets[0]
        requested_direction = _requested_direction(question, fallback=result.trade_type)
        if case_reference and explicit_case_reference:
            # The user deliberately connected a current case to one general
            # topic.  A later Data lookup is still bounded to that one scope.
            return _policy_for_guidance_target(
                target,
                requested_direction=requested_direction,
                scope="case_grounded",
                mixed=True,
                application_intent=bool(_APPLICATION_INTENT.search(question)),
            )
        policy = _policy_for_guidance_target(
            target,
            requested_direction=requested_direction,
            scope="general_customs_guidance",
        )
        if policy.direction_inapplicable:
            return _QuestionPolicy(
                "insufficient_evidence",
                general_topic=policy.general_topic,
                general_scope=policy.general_scope,
                requested_direction=policy.requested_direction,
                query_direction=policy.query_direction,
                applicable_directions=policy.applicable_directions,
                direction_inapplicable=True,
            )
        return policy
    if case_reference and explicit_case_reference:
        # The user deliberately connected a current case to a general topic.
        # No approved subject was selected, so keep the response case-bound.
        return _QuestionPolicy("case_grounded")
    if _GENERAL_REQUEST.search(question):
        return _QuestionPolicy("out_of_scope")
    # Do not turn an unrecognised question into an arbitrary first comparison.
    # The former fallback made unrelated questions look like case conclusions.
    return _QuestionPolicy("clarification_needed")


def assistant_question_policy(request: AssistantAnswerRequestV1) -> _QuestionPolicy:
    """Expose the fixed routing decision to the profile runtime only."""

    return _classify_question(
        request.question.strip(), request.review_result, recent_turns=request.recent_turns
    )


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


def _has_submission_context(result: ReviewRunResultV3) -> bool:
    """Whether the immutable Run has an attachment or cited filing-state fact.

    A conversation message never creates this context.  An attachment alone
    proves only that a document was added to this Case; an identifier/date or
    status in that document is required before the answer can describe
    document-level filing or acceptance evidence.
    """

    if any(
        analysis.document_type in {"supporting_evidence", "export_declaration"}
        for analysis in result.document_analyses
    ):
        return True
    return any(
        fact.name
        in {
            "submission_receipt_no",
            "export_declaration_no",
            "filing_date",
            "submission_status",
        }
        for analysis in result.document_analyses
        for fact in _analysis_facts(analysis)
    )


def _targeted_case_answer(
    policy: _QuestionPolicy,
    result: ReviewRunResultV3,
) -> tuple[str | None, list[AssistantCitationV1]]:
    if policy.target == "submission":
        return _submission_status_answer(result)
    if policy.target == "certificate_hs":
        return _certificate_hs_answer(result)
    if policy.target == "multi_document_hs":
        return _multi_document_hs_answer(result)
    return None, []


def _submission_status_answer(
    result: ReviewRunResultV3,
) -> tuple[str | None, list[AssistantCitationV1]]:
    """Describe only attachment and document-cited submission evidence.

    This deliberately does not call an outside agency system.  It gives the
    reviewer a useful distinction between a Case attachment, a receipt/declaration
    identifier written in the document, and an explicit acceptance/rejection
    wording, while retaining the source facts as citations.
    """

    relevant = [
        analysis
        for analysis in result.document_analyses
        if analysis.document_type in {"supporting_evidence", "export_declaration"}
        or any(
            fact.name
            in {
                "submission_receipt_no",
                "export_declaration_no",
                "filing_date",
                "submission_status",
            }
            for fact in _analysis_facts(analysis)
        )
    ]
    facts = [fact for analysis in relevant for fact in _analysis_facts(analysis)]
    by_name: dict[str, list[FactV3]] = {}
    for fact in facts:
        by_name.setdefault(fact.name, []).append(fact)
    citations: list[AssistantCitationV1] = []

    def include(*names: str) -> list[FactV3]:
        selected: list[FactV3] = []
        for name in names:
            selected.extend(by_name.get(name, []))
        for fact in selected:
            citation = AssistantCitationV1(kind="fact", reference_id=fact.fact_id)
            if citation not in citations:
                citations.append(citation)
        return selected

    identifiers = include("submission_receipt_no", "export_declaration_no")
    dates = include("filing_date")
    statuses = include("submission_status")
    attached_labels = {
        _DOCUMENT_ALIASES.get(analysis.document_type, ("문서",))[0] for analysis in relevant
    }
    attached_phrase = ", ".join(sorted(attached_labels)) if attached_labels else "관련 증빙"

    if statuses:
        raw_status = " ".join(
            str(item.normalized_value or item.raw_value).casefold() for item in statuses
        )
        if re.search(r"(?:반려|거절|rejected|reject|denied)", raw_status, re.I):
            return (
                f"현재 심사에는 {attached_phrase} 문서가 첨부되어 있고, 문서 원문에는 "
                "기관 처리 상태가 반려 또는 거절로 표시되어 있습니다. 따라서 수리·승인 근거로 "
                "볼 수 없으며, 담당자가 보완 또는 재제출 필요 여부를 확인해 주세요. "
                "이는 외부 기관 시스템의 실시간 조회 결과가 아니라 첨부 문서상 근거입니다.",
                citations,
            )
        if re.search(r"(?:수리|승인|accepted|accept|approved|approve)", raw_status, re.I):
            return (
                f"현재 심사에는 {attached_phrase} 문서가 첨부되어 있고, 문서 원문에서 "
                "수리·승인으로 해석할 수 있는 처리 상태를 확인했습니다. "
                "접수·신고 번호와 제출일도 함께 담당자가 원문으로 확인해 주세요. "
                "이는 외부 기관 시스템의 실시간 조회 결과가 아니라 첨부 문서상 근거입니다.",
                citations,
            )
        return (
            f"현재 심사에는 {attached_phrase} 문서가 첨부되어 있고, 문서 원문에 기관 처리 상태가 "
            "기재되어 있습니다. 그 표현만으로 수리·승인 또는 반려 여부를 확정하지 않으며, "
            "담당자가 원문과 기관 확인 절차를 함께 검토해 주세요. "
            "이는 외부 기관 시스템의 실시간 조회 결과가 아닙니다.",
            citations,
        )
    if identifiers:
        date_note = (
            " 제출일도 함께 기재되어 있습니다."
            if dates
            else " 제출일은 문서에서 확인되지 않았습니다."
        )
        return (
            f"현재 심사에는 {attached_phrase} 문서가 첨부되어 있고, 문서 원문에서 접수·신고 번호를 "
            f"확인했습니다. 이는 기관 제출 근거로 검토할 수 있는 문서상 사실입니다.{date_note} "
            "다만 수리·승인 여부는 별도 처리 상태가 확인되기 전에는 확정할 수 없고, 외부 기관 시스템의 "
            "실시간 조회 결과도 아닙니다.",
            citations,
        )
    if relevant:
        return (
            f"현재 심사에는 {attached_phrase} 문서가 첨부되어 있습니다. 다만 원문에서 접수·신고 번호, "
            "제출일 또는 수리·승인·반려 상태를 확인할 수 없어 기관 제출 또는 처리 상태를 판단할 수 없습니다. "
            "관련 증빙을 보완해 주세요.",
            [],
        )
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
    "GuidanceContext",
    "assistant_question_policy",
    "combine_case_and_guidance",
    "deterministic_assistant_answer",
    "deterministic_knowledge_answer",
    "guidance_answer_for_case",
    "guidance_context_from_data",
    "guidance_topic_for_question",
    "knowledge_question_policy",
    "provider_answer_allowed",
    "provider_snapshot",
    "response_from_provider_candidate",
    "response_from_knowledge_provider_candidate",
    "unregistered_document_input_answer",
]
