from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from typing import Any, Literal

from customs_ai_agent.adapters.llm import (
    ClaudeDocumentAnalyzer,
    LLMError,
    StructuredAgenticResult,
    anthropic_output_schema_sha256,
)
from customs_ai_agent.meaning_catalog import (
    GENERAL_CUSTOMS_GUIDANCE,
    user_action_text,
    user_meaning_for_finding,
    user_meaning_for_issue,
)
from customs_ai_agent.prompts import (
    CASE_ANSWER_V2_PROMPT_ID,
    CASE_ANSWER_VERIFIER_PROMPT_ID,
    EXPERIENCE_ENTAILMENT_PROMPT_ID,
    EXPERIENCE_REPORT_PROMPT_ID,
    load_case_answer_v2_system_prompt,
    load_case_answer_verifier_system_prompt,
    load_case_scope_classifier_system_prompt,
    load_experience_entailment_system_prompt,
    load_experience_report_system_prompt,
)
from customs_ai_agent.schemas.experience import (
    AgentUatCapabilityAttestationV1,
    AnswerScopeClassificationAuditV1,
    AnswerScopeV2,
    AssistantAnswerV2,
    CanonicalClaimV1,
    CaseConversationRequestV2,
    ExperienceGenerationAuditV1,
    ExperienceVerificationAuditV1,
    GroundedPresentationSectionV1,
    GroundedPresentationStatementV1,
    GroundedPresentationV1,
    PresentationSectionCode,
    PresentationStatementRole,
    ReviewExperienceEnvelopeV1,
    UserActionEventV1,
    assistant_answer_verified_content_sha256,
    case_conversation_request_sha256,
    presentation_verified_content_sha256,
)
from customs_ai_agent.schemas.v3 import (
    CitationReferenceV1,
    ReviewRunResultV3,
    canonical_json_sha256,
    stable_object_id,
)
from customs_ai_agent.security_guard import prompt_injection_unmet_reason_codes
from customs_ai_agent.v3_semantics import reconstruct_v3_current_semantics


ExperienceStageCallback = Callable[[dict[str, Any]], None]

_SAFE_SUGGESTED_QUESTIONS = [
    "이 결과가 나온 가장 중요한 이유는 무엇입니까?",
    "담당자가 먼저 확인할 항목은 무엇입니까?",
    "연결된 법령 근거가 이 심사 건에서 어떤 의미입니까?",
]


def _stage_started(
    callback: ExperienceStageCallback | None,
    stage_code: str,
) -> int:
    started_ns = time.perf_counter_ns()
    if callback is not None:
        callback(
            {
                "stage_code": stage_code,
                "state": "started",
                "duration_ms": None,
                "safe_counts": {},
            }
        )
    return started_ns


def _stage_finished(
    callback: ExperienceStageCallback | None,
    stage_code: str,
    started_ns: int,
    *,
    state: Literal["completed", "failed"],
    safe_counts: dict[str, int] | None = None,
) -> None:
    if callback is None:
        return
    callback(
        {
            "stage_code": stage_code,
            "state": state,
            "duration_ms": max(
                1,
                min(
                    300_000,
                    round((time.perf_counter_ns() - started_ns) / 1_000_000),
                ),
            ),
            "safe_counts": safe_counts or {},
        }
    )


_SECTION_ORDER: tuple[PresentationSectionCode, ...] = (
    "conclusion",
    "process",
    "document_comparison",
    "issues_and_causes",
    "regulatory_meaning",
    "business_impact",
    "next_actions",
    "scope_and_limits",
)
_SECTION_TITLES: dict[PresentationSectionCode, str] = {
    "conclusion": "한눈에 보는 결론",
    "process": "Agent가 수행한 검토 과정",
    "document_comparison": "문서 대조 결과",
    "issues_and_causes": "발견된 문제와 원인",
    "regulatory_meaning": "법령 근거와 이 심사 건에서의 의미",
    "business_impact": "업무상 영향",
    "next_actions": "담당자의 다음 조치",
    "scope_and_limits": "검토 범위와 한계",
}
_ANSWER_AREAS: tuple[str, ...] = (
    "direct_answer",
    "explanation",
    "evidence_meaning",
    "impact",
    "next_steps",
)
_ANSWER_AREA_ROLES: dict[str, frozenset[PresentationStatementRole]] = {
    "direct_answer": frozenset(
        {"conclusion", "observation", "reason", "evidence", "impact", "action", "limitation"}
    ),
    "explanation": frozenset({"conclusion", "observation", "reason", "limitation"}),
    "evidence_meaning": frozenset({"evidence", "limitation"}),
    "impact": frozenset({"impact"}),
    "next_steps": frozenset({"action"}),
}
_CASE_LIMITATION = "답변은 현재 심사 회차와 연결된 문서·법령 근거 범위에 한정됩니다."
_GENERAL_GUIDANCE_LIMITATION = (
    "위 내용은 일반 통관 안내이며 현재 심사 건의 판정이나 법적 의무를 뜻하지 않습니다."
)
_DOCUMENT_TYPE_LABELS = {
    "invoice": "상업송장(CI)",
    "packing_list": "포장명세서(PL)",
    "bill_of_lading": "선하증권(B/L)",
    "certificate_of_origin": "원산지증명서(C/O)",
    "unknown": "제출 문서",
}
_FIELD_LABELS = {
    "invoice_no": "송장번호",
    "invoice_number": "송장번호",
    "invoice_date": "송장 발행일",
    "packing_no": "포장명세서 번호",
    "bl_no": "선하증권 번호",
    "certificate_no": "원산지증명서 번호",
    "seller": "판매자",
    "buyer": "구매자",
    "exporter": "수출자",
    "importer": "수입자",
    "shipper": "송하인",
    "consignee": "수하인",
    "notify_party": "착하통지처",
    "vessel": "선박명",
    "voyage_no": "항차",
    "port_of_loading": "선적항",
    "port_of_discharge": "양륙항",
    "departure_date": "출항일",
    "country_of_origin": "원산지",
    "origin_country": "원산지",
    "hs_code": "품목분류번호(HSK)",
    "description": "품명",
    "goods_description": "물품 설명",
    "quantity": "수량",
    "unit_price": "단가",
    "line_amount": "품목 금액",
    "total_amount": "총금액",
    "currency": "통화",
    "packages": "포장 수량",
    "gross_weight_kg": "총중량",
    "total_gross_weight_kg": "전체 총중량",
    "net_weight_kg": "순중량",
    "measurement_cbm": "용적",
    "container_numbers": "컨테이너 번호",
    "seal_numbers": "봉인 번호",
    "shipping_marks": "화인",
}
_FIELD_PRIORITY = {
    name: index
    for index, name in enumerate(
        (
            "invoice_no",
            "invoice_number",
            "invoice_date",
            "packing_no",
            "bl_no",
            "certificate_no",
            "hs_code",
            "description",
            "goods_description",
            "quantity",
            "unit_price",
            "line_amount",
            "total_amount",
            "currency",
            "country_of_origin",
            "origin_country",
            "seller",
            "buyer",
            "exporter",
            "importer",
            "shipper",
            "consignee",
            "vessel",
            "voyage_no",
            "port_of_loading",
            "port_of_discharge",
            "departure_date",
            "packages",
            "gross_weight_kg",
            "total_gross_weight_kg",
            "net_weight_kg",
            "measurement_cbm",
            "container_numbers",
            "seal_numbers",
            "shipping_marks",
        )
    )
}

# Claude may write the connective, explanatory prose, but factual atoms remain
# provider-owned.  These patterns intentionally target values whose invention
# would materially change a customs review: numbers/units, document and HSK
# identifiers, law names, institutions and opaque internal identifiers.  The
# semantic verifier remains responsible for contradictions that contain no new
# lexical atom.
_URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
_HTML_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
_OPAQUE_INTERNAL_RE = re.compile(
    r"(?i)(?:experience_claim|statement|fact|issue|finding|ev|outcome)_[a-f0-9]{32,}"
    r"|(?<![A-Fa-f0-9])[A-Fa-f0-9]{64}(?![A-Fa-f0-9])"
)
_NUMBER_RE = re.compile(
    r"(?<![0-9])(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<unit>%|원|달러|유로|엔|kg|㎏|g|톤|개|건|종|회|일|개월|년|월|조|항|호)?"
    r"(?![0-9])",
    re.IGNORECASE,
)
_ALPHANUMERIC_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9._:/-]{3,}(?![A-Za-z0-9._:/-]))"
    r"(?=[A-Za-z0-9._:/-]*[A-Za-z])(?=[A-Za-z0-9._:/-]*\d)"
    r"[A-Za-z0-9][A-Za-z0-9._:/-]*"
)
_UPPERCASE_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z]{2,}(?:[-_/][A-Z0-9]+)*)(?![A-Za-z0-9])"
)
_KOREAN_LAW_RE = re.compile(
    r"(?<![가-힣])(?:[가-힣]{2,30}(?:시행규칙|시행령|법률|고시)|[가-힣]{2,30}법)"
    r"(?=$|[\s.,;:!?()\[\]{}“”\"']|은|는|이|가|을|를|의|에|에서|으로|로|과|와|도|만)"
)
_NON_LAW_SUFFIXES = (
    "방법",
    "기법",
    "해법",
    "문법",
    "어법",
    "용법",
    "화법",
    "농법",
    "공법",
    "수법",
    "비법",
    "요법",
)
_KOREAN_INSTITUTION_RE = re.compile(
    r"(?<![가-힣])(?:[가-힣]{2,24}\s*)"
    r"(?:관세청|세관|주식회사|유한회사|법인|은행|항만공사|항만|공항)"
    r"(?=$|[\s.,;:!?()\[\]{}“”\"']|은|는|이|가|을|를|의|에|에서|으로|로|과|와|도|만)"
)
_ASCII_ORGANIZATION_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Z][A-Za-z&.-]*\s+){0,4}"
    r"(?:CORP(?:ORATION)?|CO(?:MPANY)?|LTD|INC|LLC)(?![A-Za-z])",
    re.IGNORECASE,
)
_COUNTRY_RE = re.compile(
    r"(?<![가-힣])(?:대한민국|한국|중국|일본|미국|독일|프랑스|영국|베트남|태국|"
    r"인도네시아|인도|호주|캐나다|러시아|대만|홍콩|싱가포르|말레이시아|필리핀|"
    r"네덜란드|이탈리아|스페인|브라질|멕시코)"
    r"(?=$|[\s.,;:!?()\[\]{}“”\"']|은|는|이|가|을|를|의|에|에서|으로|로|과|와|도|만|산)"
)
_KOREAN_PORT_RE = re.compile(
    r"(?<![가-힣])(?:[가-힣]{2,24}(?:항구|항))"
    r"(?=$|[\s.,;:!?()\[\]{}“”\"']|은|는|이|가|을|를|의|에|에서|으로|로|과|와|도|만)"
)
_NON_PORT_SUFFIXES = ("사항", "문항", "조항", "항목")
_GENERAL_GUIDANCE_CASE_ASSERTION_RE = re.compile(
    r"(?:현재|이|해당)\s*(?:심사\s*건|건|case).{0,40}"
    r"(?:판정|결과|의무|적용|승인|자동\s*완료|문제\s*(?:없|있))",
    re.IGNORECASE,
)
_COUNT_UNITS = {"개": "count", "건": "count", "종": "count"}
_PRESENTATION_MIN_TOTAL_MATERIAL_CHARS = 650
_PRESENTATION_MIN_SECTION_CHARS = 28
_PRESENTATION_MIN_MATERIAL_STATEMENT_CHARS = 18
_PRESENTATION_MIN_KOREAN_RATIO = 0.65


def answer_limitation_source(scope: AnswerScopeV2) -> str:
    return _GENERAL_GUIDANCE_LIMITATION if scope == "general_customs_guidance" else _CASE_LIMITATION


_PRESENTATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["statements"],
    "properties": {
        "statements": {
            "type": "array",
            "minItems": 8,
            "maxItems": 100,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["section_code", "role", "text", "source_claim_ids"],
                "properties": {
                    "section_code": {"type": "string", "enum": list(_SECTION_ORDER)},
                    "role": {
                        "type": "string",
                        "enum": [
                            "conclusion",
                            "observation",
                            "reason",
                            "evidence",
                            "impact",
                            "action",
                            "limitation",
                        ],
                    },
                    "text": {"type": "string", "minLength": 1, "maxLength": 4000},
                    "source_claim_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": r"^experience_claim_[a-f0-9]{64}$",
                        },
                    },
                },
            },
        }
    },
}
_ENTAILMENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts"],
    "properties": {
        "verdicts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 100,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement_id", "supported", "reason_code"],
                "properties": {
                    "statement_id": {
                        "type": "string",
                        "pattern": r"^statement_[a-f0-9]{64}$",
                    },
                    "supported": {"type": "boolean"},
                    "reason_code": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                },
            },
        }
    },
}
_CHAT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "direct_answer",
        "explanation",
        "evidence_meaning",
        "impact",
        "next_steps",
        "limitations",
        "claim_selections",
        "citation_references",
        "suggested_questions",
        "used_context_message_ids",
    ],
    "properties": {
        "direct_answer": {"type": "string", "minLength": 1, "maxLength": 4000},
        "explanation": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "evidence_meaning": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "impact": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "next_steps": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "limitations": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "minLength": 1, "maxLength": 2000},
        },
        "claim_selections": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "direct_answer",
                "explanation",
                "evidence_meaning",
                "impact",
                "next_steps",
            ],
            "properties": {
                area: {
                    "type": "array",
                    "maxItems": (1 if area == "direct_answer" else 40),
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "pattern": r"^experience_claim_[a-f0-9]{64}$",
                    },
                }
                for area in (
                    "direct_answer",
                    "explanation",
                    "evidence_meaning",
                    "impact",
                    "next_steps",
                )
            },
        },
        "citation_references": {
            "type": "array",
            "maxItems": 100,
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "reference_id"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["field_source", "issue", "finding", "evidence", "outcome"],
                    },
                    "reference_id": {"type": "string", "minLength": 1, "maxLength": 256},
                },
            },
        },
        "suggested_questions": {
            "type": "array",
            "maxItems": 3,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
        "used_context_message_ids": {
            "type": "array",
            "maxItems": 8,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
    },
}
_ANSWER_VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["supported", "reason_codes"],
    "properties": {
        "supported": {"type": "boolean"},
        "reason_codes": {
            "type": "array",
            "maxItems": 20,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
    },
}
_SCOPE_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer_scope", "reason_code", "guidance_topics"],
    "properties": {
        "answer_scope": {
            "type": "string",
            "enum": [
                "case_grounded",
                "evidence_grounded",
                "general_customs_guidance",
                "clarification_needed",
                "insufficient_evidence",
                "out_of_scope",
            ],
        },
        "reason_code": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "pattern": r"^[a-z][a-z0-9_]*$",
        },
        "guidance_topics": {
            "type": "array",
            "maxItems": 3,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": [
                    "invoice",
                    "packing_list",
                    "bill_of_lading",
                    "certificate_of_origin",
                    "hsk",
                    "hs_hsk_comparison",
                    "customs_review",
                    "import_declaration",
                    "customs_valuation",
                    "duties_and_taxes",
                    "free_trade_agreement",
                    "import_requirements",
                    "inspection_and_supplement",
                    "document_correction",
                    "customs_broker",
                ],
            },
        },
    },
}


def experience_schema_hashes() -> dict[str, str]:
    return {
        "presentation": anthropic_output_schema_sha256(_PRESENTATION_OUTPUT_SCHEMA),
        "entailment": anthropic_output_schema_sha256(_ENTAILMENT_OUTPUT_SCHEMA),
        "chat": anthropic_output_schema_sha256(_CHAT_OUTPUT_SCHEMA),
        "chat_verifier": anthropic_output_schema_sha256(_ANSWER_VERIFIER_SCHEMA),
        "scope_classifier": anthropic_output_schema_sha256(_SCOPE_CLASSIFIER_SCHEMA),
        "scope_classifier_semantic": canonical_json_sha256(_SCOPE_CLASSIFIER_SCHEMA),
    }


def experience_prompt_hashes() -> dict[str, str]:
    return {
        "presentation": hashlib.sha256(
            load_experience_report_system_prompt().encode("utf-8")
        ).hexdigest(),
        "entailment": hashlib.sha256(
            load_experience_entailment_system_prompt().encode("utf-8")
        ).hexdigest(),
        "chat": hashlib.sha256(load_case_answer_v2_system_prompt().encode("utf-8")).hexdigest(),
        "chat_verifier": hashlib.sha256(
            load_case_answer_verifier_system_prompt().encode("utf-8")
        ).hexdigest(),
        "scope_classifier": hashlib.sha256(
            load_case_scope_classifier_system_prompt().encode("utf-8")
        ).hexdigest(),
    }


def _citation(kind: str, reference_id: str) -> CitationReferenceV1:
    return CitationReferenceV1(kind=kind, reference_id=reference_id)  # type: ignore[arg-type]


def _claim(
    section: PresentationSectionCode,
    role: PresentationStatementRole,
    text: str,
    citations: Iterable[CitationReferenceV1],
    *,
    material: bool = True,
) -> CanonicalClaimV1:
    citation_list = list({(item.kind, item.reference_id): item for item in citations}.values())
    payload = {
        "section_code": section,
        "role": role,
        "canonical_text": " ".join(text.split()),
        "citation_references": [
            item.model_dump(mode="json", warnings=False) for item in citation_list
        ],
        "material": material,
    }
    return CanonicalClaimV1.model_validate(
        {
            "claim_id": stable_object_id("experience_claim", payload),
            **payload,
        }
    )


def _normalized_prose(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _normalized_atom(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", normalized).replace(",", "").casefold().strip(".,;:()[]{}“”\"'")


def _factual_atoms(text: str) -> frozenset[str]:
    """Extract high-risk lexical facts without attempting semantic reasoning.

    This deliberately errs on the side of rejecting a Claude candidate.  A
    rejected candidate is regenerated once and then replaced by a detailed
    deterministic report/answer, so false positives reduce fluency rather than
    safety.
    """

    text = unicodedata.normalize("NFKC", text)
    atoms: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        number = match.group("number").replace(",", "")
        atoms.add(f"number:{number}")
        unit = (match.group("unit") or "").casefold()
        if unit:
            atoms.add(f"number-unit:{number}:{_COUNT_UNITS.get(unit, unit)}")
    for pattern, prefix in (
        (_ALPHANUMERIC_IDENTIFIER_RE, "identifier"),
        (_UPPERCASE_IDENTIFIER_RE, "identifier"),
        (_KOREAN_LAW_RE, "law"),
        (_KOREAN_INSTITUTION_RE, "entity"),
        (_ASCII_ORGANIZATION_RE, "entity"),
        (_COUNTRY_RE, "entity"),
        (_KOREAN_PORT_RE, "entity"),
    ):
        for match in pattern.finditer(text):
            atom = _normalized_atom(match.group(0))
            if prefix == "law" and atom.endswith(_NON_LAW_SUFFIXES):
                continue
            if pattern is _KOREAN_PORT_RE and atom.endswith(_NON_PORT_SUFFIXES):
                continue
            if atom:
                atoms.add(f"{prefix}:{atom}")
    for match in re.finditer(r"[“\"]([^“”\"\r\n]{1,80})[”\"]", text):
        literal = _normalized_atom(match.group(1))
        if literal:
            atoms.add(f"value:{literal}")
    labelled_value = re.compile(
        r"(?:원산지|선적항|양륙항|도착항|출발항|선박명|통화|판매자|구매자|수출자|"
        r"수입자|송하인|수하인)(?:은|는|이|가|:)?\s*[“\"]?"
        r"([A-Za-z가-힣][A-Za-z가-힣0-9 .,&()/_-]{0,40}?)"
        r"(?:[”\"]?(?:입니다|이며|이고|로 확인|으로 확인|로 기재|으로 기재))"
    )
    for match in labelled_value.finditer(text):
        atom = _normalized_atom(match.group(1))
        if atom:
            atoms.add(f"value:{atom}")
    return frozenset(atoms)


def _grounded_prose_blockers(
    text: str,
    source_texts: Iterable[str],
    *,
    prefix: str,
    minimum_chars: int = 1,
    require_formal_korean: bool = True,
) -> list[str]:
    normalized = _normalized_prose(text)
    source = "\n".join(_normalized_prose(item) for item in source_texts)
    blockers: list[str] = []
    if len(normalized) < minimum_chars:
        blockers.append(f"{prefix}_too_short")
    if any(
        ord(character) < 32 and character not in "\t\n\r" for character in text
    ) or _HTML_RE.search(normalized):
        blockers.append(f"{prefix}_unsafe_markup")
    if _URL_RE.search(normalized):
        blockers.append(f"{prefix}_url_not_allowed")
    if _OPAQUE_INTERNAL_RE.search(normalized):
        blockers.append(f"{prefix}_internal_identifier_not_allowed")
    if not _factual_atoms(normalized).issubset(_factual_atoms(source)):
        blockers.append(f"{prefix}_unsupported_factual_atom")
    if require_formal_korean and not normalized.endswith("니다."):
        blockers.append(f"{prefix}_formal_style_required")
    return blockers


def _korean_letter_ratio(text: str) -> float:
    hangul = sum("가" <= character <= "힣" for character in text)
    latin = sum(
        ("a" <= character.casefold() <= "z") for character in text if len(character.casefold()) == 1
    )
    denominator = hangul + latin
    return hangul / denominator if denominator else 0.0


def _presentation_statement_options(
    claims: Iterable[CanonicalClaimV1],
) -> list[dict[str, Any]]:
    """Return non-overlapping factual claim groups for Claude to explain.

    Every canonical claim is placed in exactly one option. Claims can be
    combined only when their section and semantic role are identical.  Unlike
    the former finite rendering menu, the model authors new explanatory prose;
    the claim IDs, protected factual atoms and independent entailment verifier
    keep that prose grounded.
    """

    groups: dict[
        tuple[PresentationSectionCode, PresentationStatementRole],
        list[CanonicalClaimV1],
    ] = {}
    for claim in claims:
        groups.setdefault((claim.section_code, claim.role), []).append(claim)
    options: list[dict[str, Any]] = []
    for (section, role), grouped in groups.items():
        chunks: list[list[CanonicalClaimV1]] = []
        current: list[CanonicalClaimV1] = []
        current_citations: set[tuple[str, str]] = set()
        for claim in grouped:
            claim_citations = {
                (str(citation.kind), citation.reference_id)
                for citation in claim.citation_references
            }
            candidate = [*current, claim]
            candidate_citations = current_citations | claim_citations
            candidate_source = " ".join(item.canonical_text for item in candidate)
            if current and (
                len(candidate) > 20
                or len(candidate_citations) > 100
                or len(candidate_source) > 4000
            ):
                chunks.append(current)
                current = [claim]
                current_citations = claim_citations
            else:
                current = candidate
                current_citations = candidate_citations
        if current:
            chunks.append(current)
        for chunk in chunks:
            options.append(
                {
                    "section_code": section,
                    "role": role,
                    "source_claim_ids": [claim.claim_id for claim in chunk],
                    "canonical_claims": [
                        {
                            "claim_id": claim.claim_id,
                            "canonical_text": claim.canonical_text,
                            "citation_references": [
                                citation.model_dump(mode="json", warnings=False)
                                for citation in claim.citation_references
                            ],
                        }
                        for claim in chunk
                    ],
                    "protected_factual_atoms": sorted(
                        _factual_atoms(" ".join(claim.canonical_text for claim in chunk))
                    ),
                }
            )
    return options


def _claim_sources(claims: Iterable[CanonicalClaimV1]) -> list[str]:
    return [claim.canonical_text for claim in claims]


def _field_fact_claims(result: ReviewRunResultV3) -> list[CanonicalClaimV1]:
    """Expose bounded, user-readable document facts for grounded Case questions.

    These claims are used by Chat only. They are deliberately kept out of the
    mandatory report catalog so a four-document review does not become a raw
    field dump.
    """

    document_types = {
        item.document_id: _DOCUMENT_TYPE_LABELS.get(item.document_type, "제출 문서")
        for item in result.document_results
    }
    candidates = [
        item
        for item in result.fact_ledger.entries
        if item.kind == "field_source"
        and item.value.strip().casefold() not in {"", "null", "none", "[]", "{}"}
    ]

    def sort_key(item: Any) -> tuple[int, str, str]:
        field_name = item.label.rsplit(".", 1)[-1]
        return (
            _FIELD_PRIORITY.get(field_name, len(_FIELD_PRIORITY) + 1),
            item.document_id or "",
            item.reference_id,
        )

    claims: list[CanonicalClaimV1] = []
    for entry in sorted(candidates, key=sort_key)[:80]:
        field_name = entry.label.rsplit(".", 1)[-1]
        field_label = _FIELD_LABELS.get(field_name, "문서 기재 항목")
        document_label = document_types.get(entry.document_id or "", "제출 문서")
        value = " ".join(entry.value.split())[:1_000]
        claims.append(
            _claim(
                "document_comparison",
                "observation",
                f"{document_label}에서 {field_label}의 기록값은 “{value}”로 확인되었습니다.",
                [_citation("field_source", entry.reference_id)],
            )
        )
    return claims


def build_answer_claim_catalog(result: ReviewRunResultV3) -> list[CanonicalClaimV1]:
    return [*build_canonical_claim_catalog(result), *_field_fact_claims(result)]


def answer_claim_bindings(
    answer: AssistantAnswerV2,
    result: ReviewRunResultV3,
) -> dict[str, tuple[CanonicalClaimV1, ...]]:
    """Reconstruct the exact canonical claims selected for each Chat area.

    The live evaluator and independent consumers use this projection to prove
    that a fluent but irrelevant answer did not pass merely because it cited
    some fact from the same Case. It never interprets free prose: only the
    provider-owned, verifier-approved renderings are accepted.
    """

    claims = build_answer_claim_catalog(result)
    claim_index = {claim.claim_id: claim for claim in claims}
    values = answer.claim_selections.model_dump(mode="json", warnings=False)
    return {
        area: tuple(claim_index[claim_id] for claim_id in claim_ids if claim_id in claim_index)
        for area, claim_ids in values.items()
    }


def build_canonical_claim_catalog(result: ReviewRunResultV3) -> list[CanonicalClaimV1]:
    facts = {(entry.kind, entry.reference_id): entry for entry in result.fact_ledger.entries}
    current_semantics = reconstruct_v3_current_semantics(result)
    expected_occurrence_references = [
        *(("issue", reference_id) for reference_id in current_semantics.issue_reference_ids),
        *(("finding", reference_id) for reference_id in current_semantics.finding_reference_ids),
    ]
    if any(reference not in facts for reference in expected_occurrence_references):
        raise ValueError("canonical occurrence reference is absent from the FactLedger")
    outcome_entry = next(entry for entry in result.fact_ledger.entries if entry.kind == "outcome")
    outcome = _citation("outcome", outcome_entry.reference_id)
    claims: list[CanonicalClaimV1] = []
    if result.assessment == "no_issue_detected":
        conclusion = (
            "제출된 문서와 현재 조회된 법령 근거 범위에서는 별도의 불일치나 규정상 누락을 "
            "확인하지 못했습니다."
        )
    elif result.assessment == "issues_detected":
        conclusion = "문서 또는 규정 검토에서 담당자가 확인하고 보완해야 할 사항이 발견되었습니다."
    else:
        conclusion = (
            "현재 자료와 근거만으로 심사 결과를 확정하기 어려워 담당자의 추가 확인이 필요합니다."
        )
    claims.append(_claim("conclusion", "conclusion", conclusion, [outcome]))
    claims.append(
        _claim(
            "conclusion",
            "limitation",
            (
                "이 결과는 자동 통관 승인이 아니라 내부 사전심사 결과이며, 현재 환경에서는 "
                "모든 심사 건을 담당자가 최종 확인합니다."
            ),
            [outcome],
        )
    )
    claims.append(
        _claim(
            "process",
            "observation",
            (
                "Agent는 문서 분석, 값 정규화, 문서 연결, 교차검증, 법령 근거 조회, 규정 검토와 "
                "결과 검증을 정해진 순서로 수행했습니다."
            ),
            [outcome],
        )
    )
    if result.action_trace:
        succeeded = sum(item.status == "succeeded" for item in result.action_trace)
        deferred = sum(item.status == "deferred" for item in result.action_trace)
        claims.append(
            _claim(
                "process",
                "observation",
                (
                    f"제한형 추가 조사에서는 {succeeded}개 작업이 완료되었고 "
                    f"{deferred}개 작업은 안전 정책에 따라 담당자 확인으로 남겼습니다."
                ),
                [outcome],
            )
        )
    if result.validation_issues:
        for issue, reference in zip(
            result.validation_issues,
            current_semantics.issue_reference_ids,
            strict=True,
        ):
            meaning = user_meaning_for_issue(issue.issue_code)
            claims.extend(
                [
                    _claim(
                        "document_comparison",
                        "observation",
                        meaning.title + ".",
                        [_citation("issue", reference)],
                    ),
                    _claim(
                        "issues_and_causes",
                        "reason",
                        meaning.explanation,
                        [_citation("issue", reference)],
                    ),
                    _claim(
                        "business_impact",
                        "impact",
                        meaning.impact,
                        [_citation("issue", reference)],
                    ),
                ]
            )
    else:
        claims.append(
            _claim(
                "document_comparison",
                "observation",
                "현재 검증 기준으로 문서 사이의 값 불일치는 발견되지 않았습니다.",
                [outcome],
            )
        )
        claims.append(
            _claim(
                "issues_and_causes",
                "observation",
                "문서 교차검증에서 별도로 설명할 오류 원인은 확인되지 않았습니다.",
                [outcome],
            )
        )
    if result.regulatory_findings:
        for finding, reference in zip(
            result.regulatory_findings,
            current_semantics.finding_reference_ids,
            strict=True,
        ):
            meaning = user_meaning_for_finding(finding.finding_code)
            claims.extend(
                [
                    _claim(
                        "issues_and_causes",
                        "reason",
                        meaning.explanation,
                        [_citation("finding", reference)],
                    ),
                    _claim(
                        "business_impact",
                        "impact",
                        meaning.impact,
                        [_citation("finding", reference)],
                    ),
                ]
            )
    evidence_claims = 0
    for evidence in result.rag_evidence[:12]:
        if ("evidence", evidence.evidence_id) not in facts:
            continue
        identity = " · ".join(
            value for value in (evidence.law_name, evidence.article_no, evidence.title) if value
        )
        methods = {
            "exact_hs": "품목코드 구조 조회",
            "dense": "의미 유사도 검색",
            "bm25": "핵심어 검색",
            "graph": "법령 관계 확장",
        }
        observed_methods = ", ".join(
            methods[item] for item in evidence.retrieval_methods if item in methods
        )
        text = f"현재 심사 건의 법령 근거로 {identity}을(를) 확인했습니다."
        if observed_methods:
            text += f" 이 근거는 {observed_methods} 경로에서 관측되었습니다."
        claims.append(
            _claim(
                "regulatory_meaning",
                "evidence",
                text,
                [_citation("evidence", evidence.evidence_id)],
            )
        )
        evidence_claims += 1
    if not evidence_claims:
        claims.append(
            _claim(
                "regulatory_meaning",
                "limitation",
                "현재 결과에 사용자에게 제시할 수 있는 법령 근거가 충분히 연결되지 않았습니다.",
                [outcome],
            )
        )
    if not result.validation_issues and not result.regulatory_findings:
        claims.append(
            _claim(
                "business_impact",
                "impact",
                (
                    "문서상 보완 항목은 없지만 로컬 검증 환경의 자동 처리 기능이 꺼져 있어 "
                    "담당자 확인 후 다음 업무를 진행해야 합니다."
                ),
                [outcome],
            )
        )
    for action in result.required_actions:
        citations = [
            _citation(facts[(kind, reference_id)].kind, reference_id)
            for reference_id in action.source_reference_ids
            for kind in ("field_source", "issue", "finding", "evidence", "outcome")
            if (kind, reference_id) in facts
        ]
        if not citations:
            citations = [outcome]
        claims.append(
            _claim(
                "next_actions",
                "action",
                user_action_text(action.action_code, action.message),
                citations,
            )
        )
    if not result.required_actions:
        claims.append(
            _claim(
                "next_actions",
                "action",
                "담당자는 문서 대조 결과와 법령 근거를 확인하고 심사 의견을 기록합니다.",
                [outcome],
            )
        )
    claims.extend(
        [
            _claim(
                "scope_and_limits",
                "limitation",
                (
                    "현재 UAT는 텍스트 문서만 처리하며 이미지·PDF의 OCR 결과는 심사 입력으로 "
                    "받지 않습니다."
                ),
                [outcome],
            ),
            _claim(
                "scope_and_limits",
                "limitation",
                (
                    "법령 조회는 보존된 3개 법령 범위의 실제 RAG를 사용하므로 대한민국 통관 법령 "
                    "전체를 대표하지 않습니다."
                ),
                [outcome],
            ),
            _claim(
                "scope_and_limits",
                "limitation",
                (
                    "현재 결과는 법적 통관 승인이나 관세 당국의 최종 판단이 아니며 Production "
                    "자동완료 기능은 꺼져 있습니다."
                ),
                [outcome],
            ),
        ]
    )
    present = {item.section_code for item in claims}
    if present != set(_SECTION_ORDER):
        raise ValueError("canonical presentation catalog is incomplete")
    claim_ids = [item.claim_id for item in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("canonical presentation claim IDs must be unique")
    return claims


def _statement(
    claim_group: list[CanonicalClaimV1],
    text: str | None = None,
) -> GroundedPresentationStatementV1:
    citations = list(
        {
            (citation.kind, citation.reference_id): citation
            for claim in claim_group
            for citation in claim.citation_references
        }.values()
    )
    payload = {
        "role": claim_group[0].role,
        "text": text or " ".join(claim.canonical_text for claim in claim_group),
        "source_claim_ids": [claim.claim_id for claim in claim_group],
        "citation_references": [
            citation.model_dump(mode="json", warnings=False) for citation in citations
        ],
        "material": any(claim.material for claim in claim_group),
    }
    return GroundedPresentationStatementV1.model_validate(
        {
            "statement_id": stable_object_id("statement", payload),
            **payload,
        }
    )


def _headline(result: ReviewRunResultV3) -> str:
    if result.assessment == "no_issue_detected":
        return "문서와 법령 근거 범위에서 별도의 문제를 확인하지 못했습니다"
    if result.assessment == "issues_detected":
        return "담당자가 확인하고 보완해야 할 사항이 발견되었습니다"
    return "현재 근거만으로 확정하기 어려워 담당자 확인이 필요합니다"


def _fallback_presentation(
    result: ReviewRunResultV3,
    claims: list[CanonicalClaimV1],
    *,
    reason_codes: list[str],
    attempt_count: int,
) -> GroundedPresentationV1:
    sections = [
        GroundedPresentationSectionV1(
            section_code=section,
            title=_SECTION_TITLES[section],
            statements=[_statement([claim]) for claim in claims if claim.section_code == section],
        )
        for section in _SECTION_ORDER
    ]
    prompt_hash = experience_prompt_hashes()["presentation"]
    schema_hash = experience_schema_hashes()["presentation"]
    headline = _headline(result)
    payload = {
        "schema_version": "1.0",
        "headline": headline,
        "sections": sections,
        "generation_audit": ExperienceGenerationAuditV1(
            mode="deterministic_fallback",
            task_id=EXPERIENCE_REPORT_PROMPT_ID,
            prompt_sha256=prompt_hash,
            output_schema_sha256=schema_hash,
            attempt_count=attempt_count,
        ),
        "verification_audit": ExperienceVerificationAuditV1(
            status="fallback",
            deterministic_checks_passed=True,
            entailment_verifier="not_run",
            verifier_prompt_sha256=experience_prompt_hashes()["entailment"],
            verifier_schema_sha256=experience_schema_hashes()["entailment"],
            verified_content_sha256=presentation_verified_content_sha256(
                headline,
                sections,
            ),
            checked_statement_count=sum(len(item.statements) for item in sections),
            reason_codes=sorted(set(reason_codes or ["claude_presentation_unavailable"])),
        ),
    }
    payload["presentation_sha256"] = canonical_json_sha256(payload)
    return GroundedPresentationV1.model_validate(payload)


def _candidate_presentation(
    result: ReviewRunResultV3,
    claims: list[CanonicalClaimV1],
    call: StructuredAgenticResult,
    *,
    attempt_count: int,
) -> tuple[list[GroundedPresentationSectionV1], list[str]]:
    claim_index = {claim.claim_id: claim for claim in claims}
    option_index = {
        tuple(option["source_claim_ids"]): tuple(
            claim_index[claim_id] for claim_id in option["source_claim_ids"]
        )
        for option in _presentation_statement_options(claims)
    }
    raw_statements = call.value.get("statements")
    if not isinstance(raw_statements, list):
        return [], ["presentation_statements_invalid"]
    statements_by_section: dict[PresentationSectionCode, list[GroundedPresentationStatementV1]] = {
        section: [] for section in _SECTION_ORDER
    }
    used_claims: set[str] = set()
    blockers: list[str] = []
    for raw in raw_statements:
        if not isinstance(raw, dict):
            blockers.append("presentation_statement_invalid")
            continue
        claim_ids = raw.get("source_claim_ids")
        section = raw.get("section_code")
        role = raw.get("role")
        text = raw.get("text")
        if (
            not isinstance(claim_ids, list)
            or not claim_ids
            or not isinstance(section, str)
            or section not in statements_by_section
            or not isinstance(role, str)
            or not isinstance(text, str)
        ):
            blockers.append("presentation_statement_invalid")
            continue
        source = [claim_index.get(item) for item in claim_ids]
        if any(item is None for item in source):
            blockers.append("presentation_unknown_claim")
            continue
        typed_source = [item for item in source if item is not None]
        if any(item.section_code != section or item.role != role for item in typed_source):
            blockers.append("presentation_claim_type_mismatch")
            continue
        if used_claims.intersection(claim_ids):
            blockers.append("presentation_claim_reused")
            continue
        normalized_text = " ".join(text.split())
        option_claims = option_index.get(tuple(claim_ids))
        if option_claims is None:
            blockers.append("presentation_claim_group_invalid")
            continue
        prose_blockers = _grounded_prose_blockers(
            normalized_text,
            _claim_sources(option_claims),
            prefix="presentation",
            minimum_chars=_PRESENTATION_MIN_MATERIAL_STATEMENT_CHARS,
        )
        if prose_blockers:
            blockers.extend(prose_blockers)
            continue
        used_claims.update(claim_ids)
        statements_by_section[section].append(_statement(typed_source, normalized_text))
    if used_claims != set(claim_index):
        blockers.append("presentation_claim_coverage_incomplete")
    if any(not statements_by_section[section] for section in _SECTION_ORDER):
        blockers.append("presentation_section_missing")
    section_texts = {
        section: " ".join(statement.text for statement in statements_by_section[section])
        for section in _SECTION_ORDER
    }
    if any(
        len(section_texts[section]) < _PRESENTATION_MIN_SECTION_CHARS for section in _SECTION_ORDER
    ):
        blockers.append("presentation_section_too_short")
    material_text = " ".join(
        statement.text
        for section in _SECTION_ORDER
        for statement in statements_by_section[section]
        if statement.material
    )
    if len(material_text) < _PRESENTATION_MIN_TOTAL_MATERIAL_CHARS:
        blockers.append("presentation_material_content_too_short")
    if _korean_letter_ratio(material_text) < _PRESENTATION_MIN_KOREAN_RATIO:
        blockers.append("presentation_korean_ratio_too_low")
    return (
        [
            GroundedPresentationSectionV1(
                section_code=section,
                title=_SECTION_TITLES[section],
                statements=statements_by_section[section],
            )
            for section in _SECTION_ORDER
            if statements_by_section[section]
        ],
        blockers,
    )


def _verify_presentation_with_claude(
    analyzer: ClaudeDocumentAnalyzer,
    claims: list[CanonicalClaimV1],
    sections: list[GroundedPresentationSectionV1],
) -> tuple[StructuredAgenticResult, list[str], str]:
    statements = [statement for section in sections for statement in section.statements]
    verifier_context = {
        "canonical_claims": [claim.model_dump(mode="json", warnings=False) for claim in claims],
        "candidate_statements": [
            statement.model_dump(mode="json", warnings=False) for statement in statements
        ],
    }
    call = analyzer.run_bounded_agentic_task(
        task_id=EXPERIENCE_ENTAILMENT_PROMPT_ID,
        system_prompt=load_experience_entailment_system_prompt(),
        untrusted_context=verifier_context,
        output_schema=_ENTAILMENT_OUTPUT_SCHEMA,
        max_tokens=4096,
    )
    verdicts = call.value.get("verdicts")
    if not isinstance(verdicts, list):
        return call, ["entailment_verdict_invalid"], canonical_json_sha256(verifier_context)
    verdict_index = {
        item.get("statement_id"): item
        for item in verdicts
        if isinstance(item, dict) and isinstance(item.get("statement_id"), str)
    }
    if set(verdict_index) != {statement.statement_id for statement in statements}:
        return call, ["entailment_coverage_incomplete"], canonical_json_sha256(verifier_context)
    blockers = sorted(
        {
            str(item.get("reason_code") or "entailment_unsupported")
            for item in verdict_index.values()
            if item.get("supported") is not True
        }
    )
    return call, blockers, canonical_json_sha256(verifier_context)


def build_grounded_presentation(
    result: ReviewRunResultV3,
    *,
    analyzer: ClaudeDocumentAnalyzer | None,
    stage_callback: ExperienceStageCallback | None = None,
) -> GroundedPresentationV1:
    claims = build_canonical_claim_catalog(result)
    if analyzer is None:
        composition_started = _stage_started(stage_callback, "report_composition")
        verification_started: int | None = None
        try:
            presentation = _fallback_presentation(
                result,
                claims,
                reason_codes=["claude_presentation_unavailable"],
                attempt_count=1,
            )
            _stage_finished(
                stage_callback,
                "report_composition",
                composition_started,
                state="completed",
                safe_counts={
                    "statements": sum(len(section.statements) for section in presentation.sections)
                },
            )
            verification_started = _stage_started(
                stage_callback,
                "grounding_verification",
            )
            # Model validation above is the deterministic claim/citation/hash
            # verifier used by the safe fallback.
            _stage_finished(
                stage_callback,
                "grounding_verification",
                verification_started,
                state="completed",
                safe_counts={
                    "statements": sum(len(section.statements) for section in presentation.sections)
                },
            )
            return presentation
        except Exception:
            _stage_finished(
                stage_callback,
                (
                    "grounding_verification"
                    if verification_started is not None
                    else "report_composition"
                ),
                verification_started or composition_started,
                state="failed",
            )
            raise
    retry_blockers: list[str] = []
    for attempt in range(1, 3):
        generation_started = _stage_started(stage_callback, "report_composition")
        generation_completed = False
        try:
            generation = analyzer.run_bounded_agentic_task(
                task_id=EXPERIENCE_REPORT_PROMPT_ID,
                system_prompt=load_experience_report_system_prompt(),
                untrusted_context={
                    "required_claim_groups": _presentation_statement_options(claims),
                    "fixed_headline": _headline(result),
                    "fixed_section_order": list(_SECTION_ORDER),
                    "quality_requirements": {
                        "minimum_total_material_characters": (
                            _PRESENTATION_MIN_TOTAL_MATERIAL_CHARS
                        ),
                        "minimum_section_characters": _PRESENTATION_MIN_SECTION_CHARS,
                        "minimum_material_statement_characters": (
                            _PRESENTATION_MIN_MATERIAL_STATEMENT_CHARS
                        ),
                        "minimum_korean_letter_ratio": _PRESENTATION_MIN_KOREAN_RATIO,
                        "formal_sentence_ending": "니다.",
                    },
                    "previous_verification_failures": retry_blockers,
                },
                output_schema=_PRESENTATION_OUTPUT_SCHEMA,
                max_tokens=8192,
            )
            _stage_finished(
                stage_callback,
                "report_composition",
                generation_started,
                state="completed",
            )
            generation_completed = True
            if generation.model_id != result.runtime_release.model_id:
                raise ValueError("presentation_model_identity_mismatch")
            sections, blockers = _candidate_presentation(
                result,
                claims,
                generation,
                attempt_count=attempt,
            )
            if blockers:
                retry_blockers = blockers
                continue
            verification_started = _stage_started(
                stage_callback,
                "grounding_verification",
            )
            try:
                verification, blockers, verifier_context_sha256 = _verify_presentation_with_claude(
                    analyzer,
                    claims,
                    sections,
                )
                _stage_finished(
                    stage_callback,
                    "grounding_verification",
                    verification_started,
                    state="completed",
                    safe_counts={"statements": sum(len(item.statements) for item in sections)},
                )
            except (LLMError, ValueError):
                _stage_finished(
                    stage_callback,
                    "grounding_verification",
                    verification_started,
                    state="failed",
                )
                raise
            if verification.model_id != result.runtime_release.model_id:
                blockers.append("entailment_model_identity_mismatch")
            if blockers:
                retry_blockers = blockers
                continue
            headline = _headline(result)
            payload = {
                "schema_version": "1.0",
                "headline": headline,
                "sections": sections,
                "generation_audit": ExperienceGenerationAuditV1(
                    mode="claude",
                    task_id=EXPERIENCE_REPORT_PROMPT_ID,
                    model_id=generation.model_id,
                    prompt_sha256=generation.prompt_sha256,
                    output_schema_sha256=generation.output_schema_sha256,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    latency_ms=generation.latency_ms,
                    attempt_count=attempt,
                ),
                "verification_audit": ExperienceVerificationAuditV1(
                    status="passed",
                    deterministic_checks_passed=True,
                    entailment_verifier="claude",
                    verifier_model_id=verification.model_id,
                    verifier_prompt_sha256=verification.prompt_sha256,
                    verifier_schema_sha256=verification.output_schema_sha256,
                    verifier_context_sha256=verifier_context_sha256,
                    verified_content_sha256=presentation_verified_content_sha256(
                        headline,
                        sections,
                    ),
                    checked_statement_count=sum(len(item.statements) for item in sections),
                ),
            }
            payload["presentation_sha256"] = canonical_json_sha256(payload)
            return GroundedPresentationV1.model_validate(payload)
        except (LLMError, ValueError):
            # A provider exception happens before the normal completed event.
            # Candidate/identity validation failures happen after a successful
            # provider call and therefore retain the completed event above.
            if not generation_completed:
                _stage_finished(
                    stage_callback,
                    "report_composition",
                    generation_started,
                    state="failed",
                )
            retry_blockers = ["claude_presentation_or_verifier_failed"]
    composition_started = _stage_started(stage_callback, "report_composition")
    verification_started = None
    try:
        presentation = _fallback_presentation(
            result,
            claims,
            reason_codes=retry_blockers,
            attempt_count=2,
        )
        _stage_finished(
            stage_callback,
            "report_composition",
            composition_started,
            state="completed",
            safe_counts={
                "statements": sum(len(section.statements) for section in presentation.sections)
            },
        )
        verification_started = _stage_started(
            stage_callback,
            "grounding_verification",
        )
        _stage_finished(
            stage_callback,
            "grounding_verification",
            verification_started,
            state="completed",
            safe_counts={
                "statements": sum(len(section.statements) for section in presentation.sections)
            },
        )
        return presentation
    except Exception:
        _stage_finished(
            stage_callback,
            (
                "grounding_verification"
                if verification_started is not None
                else "report_composition"
            ),
            verification_started or composition_started,
            state="failed",
        )
        raise


def build_user_action_events(
    result: ReviewRunResultV3,
    *,
    presentation: GroundedPresentationV1 | None = None,
) -> list[UserActionEventV1]:
    events: list[UserActionEventV1] = []
    for action in result.required_actions:
        payload = {
            "schema_version": "1.0",
            "action_code": action.action_code,
            "title": "담당자 확인이 필요합니다",
            "description": user_action_text(action.action_code, action.message),
            "status": "needs_attention",
            "source_reference_ids": action.source_reference_ids,
        }
        events.append(
            UserActionEventV1.model_validate(
                {
                    "event_id": stable_object_id("user_action", payload),
                    **payload,
                }
            )
        )
    for trace in result.action_trace:
        if trace.tool_name not in {
            "document.reread_span/v1",
            "evidence.retrieve/v3",
            "human.request/v1",
        }:
            continue
        title = {
            "document.reread_span/v1": "문서 값을 다시 확인했습니다",
            "evidence.retrieve/v3": "법령 근거를 다시 조회했습니다",
            "human.request/v1": "담당자 확인을 요청했습니다",
        }[trace.tool_name]
        payload = {
            "schema_version": "1.0",
            "action_code": trace.tool_name,
            "title": title,
            "description": (
                "안전하게 허용된 범위에서 추가 확인을 수행했습니다."
                if trace.status == "succeeded"
                else "자동 확인을 완료하지 못해 담당자 검토로 남겼습니다."
            ),
            "status": (
                "completed"
                if trace.status == "succeeded"
                else ("deferred" if trace.status == "deferred" else "needs_attention")
            ),
            "source_reference_ids": trace.input_reference_ids,
        }
        events.append(
            UserActionEventV1.model_validate(
                {
                    "event_id": stable_object_id("user_action", payload),
                    **payload,
                }
            )
        )
    if presentation is not None:
        statement_ids = sorted(
            {
                statement.statement_id
                for section in presentation.sections
                for statement in section.statements
            }
        )[:100]
        payload = {
            "schema_version": "1.0",
            "action_code": "explanation.compose/v2",
            "title": "사용자용 심사 설명을 작성했습니다",
            "description": (
                "검증된 심사 사실과 근거만 사용해 사용자용 설명을 작성하고 "
                "근거 결박 검사를 완료했습니다."
            ),
            "status": "completed",
            "source_reference_ids": statement_ids,
        }
        events.append(
            UserActionEventV1.model_validate(
                {
                    "event_id": stable_object_id("user_action", payload),
                    **payload,
                }
            )
        )
    return events


def build_review_experience(
    result: ReviewRunResultV3,
    *,
    analyzer: ClaudeDocumentAnalyzer | None,
    capability: AgentUatCapabilityAttestationV1,
    stage_callback: ExperienceStageCallback | None = None,
) -> ReviewExperienceEnvelopeV1:
    presentation = build_grounded_presentation(
        result,
        analyzer=analyzer,
        stage_callback=stage_callback,
    )
    payload = {
        "schema_version": "1.0",
        "canonical_result": result,
        "user_presentation": presentation,
        "capability_attestation": capability,
        "user_action_events": build_user_action_events(
            result,
            presentation=presentation,
        ),
    }
    payload["envelope_sha256"] = canonical_json_sha256(payload)
    return ReviewExperienceEnvelopeV1.model_validate(payload)


def _non_model_scope_audit(
    *,
    mode: Literal["deterministic_guard", "fallback"],
    scope: AnswerScopeV2,
    reason_code: str,
) -> AnswerScopeClassificationAuditV1:
    hashes = experience_schema_hashes()
    return AnswerScopeClassificationAuditV1(
        mode=mode,
        task_id="customs_case_scope_classifier",
        classified_scope=scope,
        prompt_sha256=experience_prompt_hashes()["scope_classifier"],
        output_schema_sha256=hashes["scope_classifier"],
        semantic_output_schema_sha256=hashes["scope_classifier_semantic"],
        reason_code=reason_code,
        guidance_topics=[],
    )


def _classify_question_scope(
    request: CaseConversationRequestV2,
    analyzer: ClaudeDocumentAnalyzer | None,
) -> tuple[AnswerScopeV2, AnswerScopeClassificationAuditV1]:
    # Only the explicit security guard is deterministic. All ordinary intent
    # classification is delegated to the pinned structured Claude task so a
    # harmless wording variation cannot be rejected by a token list.
    if prompt_injection_unmet_reason_codes(
        [request.question, *(turn.content for turn in request.recent_turns)]
    ):
        return (
            "out_of_scope",
            _non_model_scope_audit(
                mode="deterministic_guard",
                scope="out_of_scope",
                reason_code="prompt_injection_blocked",
            ),
        )
    if analyzer is None:
        return (
            "clarification_needed",
            _non_model_scope_audit(
                mode="fallback",
                scope="clarification_needed",
                reason_code="scope_classifier_unavailable",
            ),
        )
    try:
        call = analyzer.run_bounded_agentic_task(
            task_id="customs_case_scope_classifier",
            system_prompt=load_case_scope_classifier_system_prompt(),
            untrusted_context={
                "question": request.question,
                "recent_turns": [
                    turn.model_dump(mode="json", warnings=False) for turn in request.recent_turns
                ],
                "case_context_available": bool(request.review_result.fact_ledger.entries),
                "evidence_available": bool(request.review_result.rag_evidence),
                "review_disposition": request.review_result.review_disposition,
                "assessment": request.review_result.assessment,
                "scope_definitions": {
                    "case_grounded": "current Case facts, outcome, cause, impact, or action",
                    "evidence_grounded": "current Case Evidence and its meaning",
                    "general_customs_guidance": "general customs term or procedure",
                    "clarification_needed": "possibly relevant but ambiguous",
                    "insufficient_evidence": "requires facts or law outside available scope",
                    "out_of_scope": "clearly unrelated to customs review",
                },
            },
            output_schema=_SCOPE_CLASSIFIER_SCHEMA,
            max_tokens=512,
        )
        if call.model_id != request.review_result.runtime_release.model_id:
            raise ValueError("scope_classifier_model_identity_mismatch")
        raw_scope = call.value.get("answer_scope")
        reason_code = call.value.get("reason_code")
        guidance_topics = call.value.get("guidance_topics")
        if (
            not isinstance(raw_scope, str)
            or not isinstance(reason_code, str)
            or not isinstance(guidance_topics, list)
            or any(not isinstance(topic, str) for topic in guidance_topics)
        ):
            raise ValueError("scope_classifier_output_invalid")
        if raw_scope == "general_customs_guidance":
            if not 1 <= len(guidance_topics) <= 3 or len(guidance_topics) != len(
                set(guidance_topics)
            ):
                raise ValueError("scope_classifier_guidance_topics_invalid")
        elif guidance_topics:
            raise ValueError("scope_classifier_guidance_topics_not_allowed")
        scope: AnswerScopeV2 = raw_scope  # type: ignore[assignment]
        if scope == "evidence_grounded" and not request.review_result.rag_evidence:
            scope = "insufficient_evidence"
            reason_code = "evidence_unavailable_reclassified"
            guidance_topics = []
        audit = AnswerScopeClassificationAuditV1(
            mode="claude",
            task_id="customs_case_scope_classifier",
            classified_scope=scope,
            model_id=call.model_id,
            prompt_sha256=call.prompt_sha256,
            output_schema_sha256=call.output_schema_sha256,
            semantic_output_schema_sha256=call.semantic_output_schema_sha256,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
            latency_ms=call.latency_ms,
            reason_code=reason_code,
            guidance_topics=guidance_topics,
        )
        return scope, audit
    except (LLMError, ValueError, TypeError):
        return (
            "clarification_needed",
            _non_model_scope_audit(
                mode="fallback",
                scope="clarification_needed",
                reason_code="scope_classifier_failed",
            ),
        )


def _answer_source_claims(
    request: CaseConversationRequestV2,
    scope: AnswerScopeV2,
    classification_audit: AnswerScopeClassificationAuditV1,
) -> tuple[list[CanonicalClaimV1], dict[str, str]]:
    claims = build_answer_claim_catalog(request.review_result)
    guidance: dict[str, str] = {}
    if scope == "evidence_grounded":
        claims = [item for item in claims if item.section_code == "regulatory_meaning"]
    elif scope == "case_grounded":
        # The classifier has already established Case scope. Give the answer
        # composer the complete bounded claim catalog instead of applying a
        # second keyword filter that could silently omit the user's intent.
        pass
    elif scope == "general_customs_guidance":
        guidance = {
            topic: GENERAL_CUSTOMS_GUIDANCE[topic] for topic in classification_audit.guidance_topics
        }
        claims = []
    else:
        claims = []
    return claims[:100], guidance


def _fallback_answer(
    request: CaseConversationRequestV2,
    scope: AnswerScopeV2,
    claims: list[CanonicalClaimV1],
    guidance: dict[str, str],
    classification_audit: AnswerScopeClassificationAuditV1,
    *,
    reasons: list[str],
    attempt_count: int,
) -> AssistantAnswerV2:
    selected_claims: list[CanonicalClaimV1] = []
    claim_selections: dict[str, list[str]] = {
        "direct_answer": [],
        "explanation": [],
        "evidence_meaning": [],
        "impact": [],
        "next_steps": [],
    }
    if scope == "out_of_scope":
        status: Literal[
            "answered", "clarification_needed", "insufficient_evidence", "out_of_scope"
        ] = "out_of_scope"
        direct = (
            "이 요청은 통관 심사 지원 범위에서 답변할 수 없습니다. 현재 심사 건의 문서, "
            "법령 근거, 보완 조치 또는 일반 통관 절차를 질문해 주십시오."
        )
        explanation: list[str] = []
    elif scope == "clarification_needed":
        status = "clarification_needed"
        direct = (
            "질문의 대상을 조금 더 구체적으로 알려주시면 현재 심사 결과에 맞추어 설명하겠습니다."
        )
        explanation = [
            "예를 들어 발견된 문제의 원인, 적용된 법령 근거, 업무상 영향 또는 다음 조치 중 무엇을 "
            "확인하려는지 알려주십시오."
        ]
    elif scope == "insufficient_evidence":
        status = "insufficient_evidence"
        direct = (
            "현재 심사 결과에는 이 질문에 답할 수 있는 충분한 법령 근거가 연결되어 있지 않습니다."
        )
        explanation = ["추가 법령 범위 또는 담당자 확인이 필요합니다."]
    elif scope == "general_customs_guidance":
        status = "answered"
        direct = next(iter(guidance.values()))
        explanation = list(guidance.values())[1:]
    else:
        status = "answered"
        direct_claim = claims[0]
        explanation_claims = [
            item for item in claims[1:] if item.role in _ANSWER_AREA_ROLES["explanation"]
        ][:5]
        direct = direct_claim.canonical_text
        explanation = [item.canonical_text for item in explanation_claims]
        claim_selections["direct_answer"] = [direct_claim.claim_id]
        claim_selections["explanation"] = [item.claim_id for item in explanation_claims]
        selected_claims.extend([direct_claim, *explanation_claims])
    evidence_claims = (
        [item for item in claims if item.role in _ANSWER_AREA_ROLES["evidence_meaning"]][:5]
        if status == "answered" and scope != "general_customs_guidance"
        else []
    )
    impact_claims = (
        [item for item in claims if item.role in _ANSWER_AREA_ROLES["impact"]][:5]
        if status == "answered" and scope != "general_customs_guidance"
        else []
    )
    action_claims = (
        [item for item in claims if item.role in _ANSWER_AREA_ROLES["next_steps"]][:5]
        if status == "answered" and scope != "general_customs_guidance"
        else []
    )
    selected_claims.extend([*evidence_claims, *impact_claims, *action_claims])
    claim_selections["evidence_meaning"] = [item.claim_id for item in evidence_claims]
    claim_selections["impact"] = [item.claim_id for item in impact_claims]
    claim_selections["next_steps"] = [item.claim_id for item in action_claims]
    evidence_meaning = [item.canonical_text for item in evidence_claims]
    impact = [item.canonical_text for item in impact_claims]
    next_steps = [item.canonical_text for item in action_claims]
    citations = list(
        {
            (citation.kind, citation.reference_id): citation
            for claim in selected_claims
            for citation in claim.citation_references
        }.values()
    )
    if scope == "general_customs_guidance":
        limitations = [_GENERAL_GUIDANCE_LIMITATION]
    else:
        limitations = [_CASE_LIMITATION]
    answer_content = {
        "request_sha256": case_conversation_request_sha256(request),
        "answer_scope": scope,
        "status": status,
        "direct_answer": direct,
        "explanation": explanation,
        "evidence_meaning": evidence_meaning,
        "impact": impact,
        "next_steps": next_steps,
        "limitations": limitations,
        "claim_selections": claim_selections,
        "citations": citations if status == "answered" else [],
        "suggested_questions": _SAFE_SUGGESTED_QUESTIONS,
        "used_context_message_ids": [turn.message_id for turn in request.recent_turns],
    }
    payload = {
        "schema_version": "2.0",
        **answer_content,
        "generation_audit": ExperienceGenerationAuditV1(
            mode="deterministic_fallback",
            task_id=CASE_ANSWER_V2_PROMPT_ID,
            prompt_sha256=experience_prompt_hashes()["chat"],
            output_schema_sha256=experience_schema_hashes()["chat"],
            attempt_count=attempt_count,
            scope_classification=classification_audit,
        ),
        "verification_audit": ExperienceVerificationAuditV1(
            status="fallback",
            deterministic_checks_passed=True,
            entailment_verifier="not_run",
            verifier_prompt_sha256=experience_prompt_hashes()["chat_verifier"],
            verifier_schema_sha256=experience_schema_hashes()["chat_verifier"],
            verified_content_sha256=assistant_answer_verified_content_sha256(answer_content),
            checked_statement_count=(
                1
                + len(explanation)
                + len(evidence_meaning)
                + len(impact)
                + len(next_steps)
                + len(limitations)
            ),
            reason_codes=sorted(set(reasons or ["claude_answer_unavailable"])),
        ),
    }
    payload["answer_sha256"] = canonical_json_sha256(payload)
    return AssistantAnswerV2.model_validate(payload)


def _answer_values_safe(
    values: dict[str, Any],
    *,
    claims: list[CanonicalClaimV1],
    guidance: dict[str, str],
    allowed_context_ids: set[str],
    scope: AnswerScopeV2,
) -> list[str]:
    blockers: list[str] = []
    raw_areas: dict[str, Any] = {
        "direct_answer": [values.get("direct_answer")],
        "explanation": values.get("explanation"),
        "evidence_meaning": values.get("evidence_meaning"),
        "impact": values.get("impact"),
        "next_steps": values.get("next_steps"),
    }
    if any(
        not isinstance(items, list) or any(not isinstance(item, str) for item in items)
        for items in raw_areas.values()
    ):
        blockers.append("answer_prose_invalid")
        typed_areas: dict[str, list[str]] = {}
    else:
        typed_areas = {
            area: [" ".join(item.split()) for item in items] for area, items in raw_areas.items()
        }
    limitations = values.get("limitations")
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        blockers.append("answer_limitations_invalid")
        typed_limitations: list[str] = []
    else:
        typed_limitations = [" ".join(item.split()) for item in limitations]

    claim_index = {claim.claim_id: claim for claim in claims}
    selected_claims: dict[str, CanonicalClaimV1] = {}
    raw_selections = values.get("claim_selections")
    expected_areas = set(_ANSWER_AREAS)
    if (
        not isinstance(raw_selections, dict)
        or set(raw_selections) != expected_areas
        or any(
            not isinstance(raw_selections.get(area), list)
            or any(not isinstance(item, str) for item in raw_selections[area])
            or len(raw_selections[area]) != len(set(raw_selections[area]))
            for area in expected_areas
        )
    ):
        blockers.append("answer_claim_selections_invalid")
        typed_selections: dict[str, list[str]] = {area: [] for area in expected_areas}
    else:
        typed_selections = {area: list(raw_selections[area]) for area in expected_areas}
    if not blockers:
        if scope in {"case_grounded", "evidence_grounded"}:
            for area, claim_ids in typed_selections.items():
                area_claims = [claim_index.get(claim_id) for claim_id in claim_ids]
                if any(claim is None for claim in area_claims):
                    blockers.append(f"answer_{area}_unknown_claim")
                    continue
                typed_claims = [claim for claim in area_claims if claim is not None]
                if any(claim.role not in _ANSWER_AREA_ROLES[area] for claim in typed_claims):
                    blockers.append(f"answer_{area}_claim_role_invalid")
                    continue
                if bool(typed_areas[area]) != bool(typed_claims):
                    blockers.append(f"answer_{area}_claim_coverage_invalid")
                    continue
                for text in typed_areas[area]:
                    blockers.extend(
                        _grounded_prose_blockers(
                            text,
                            _claim_sources(typed_claims),
                            prefix=f"answer_{area}",
                            minimum_chars=18,
                        )
                    )
                for claim in typed_claims:
                    selected_claims[claim.claim_id] = claim
            if not typed_areas["direct_answer"] or not typed_selections["direct_answer"]:
                blockers.append("answer_direct_answer_missing")
            if not typed_limitations:
                blockers.append("answer_limitation_missing")
            for limitation in typed_limitations:
                blockers.extend(
                    _grounded_prose_blockers(
                        limitation,
                        [_CASE_LIMITATION],
                        prefix="answer_limitation",
                        minimum_chars=18,
                    )
                )
        elif scope == "general_customs_guidance":
            if any(typed_selections.values()):
                blockers.append("answer_guidance_claim_selection_not_allowed")
            guidance_values = list(guidance.values())
            if (
                not guidance_values
                or not typed_areas["direct_answer"]
                or any(typed_areas[area] for area in ("evidence_meaning", "impact", "next_steps"))
            ):
                blockers.append("answer_guidance_coverage_invalid")
            else:
                for area in ("direct_answer", "explanation"):
                    for text in typed_areas[area]:
                        blockers.extend(
                            _grounded_prose_blockers(
                                text,
                                guidance_values,
                                prefix=f"answer_guidance_{area}",
                                minimum_chars=18,
                            )
                        )
                        if _GENERAL_GUIDANCE_CASE_ASSERTION_RE.search(text):
                            blockers.append("answer_guidance_case_assertion_not_allowed")
            if not typed_limitations:
                blockers.append("answer_limitation_missing")
            for limitation in typed_limitations:
                blockers.extend(
                    _grounded_prose_blockers(
                        limitation,
                        [_GENERAL_GUIDANCE_LIMITATION],
                        prefix="answer_guidance_limitation",
                        minimum_chars=18,
                    )
                )
        elif any(typed_selections.values()):
            blockers.append("answer_claim_selection_not_allowed")

    valid_citations = {
        (citation.kind, citation.reference_id)
        for claim in claims
        for citation in claim.citation_references
    }
    raw_citations = values.get("citation_references")
    if not isinstance(raw_citations, list):
        blockers.append("answer_citations_invalid")
    else:
        observed = {
            (item.get("kind"), item.get("reference_id"))
            for item in raw_citations
            if isinstance(item, dict)
        }
        if len(observed) != len(raw_citations) or not observed.issubset(valid_citations):
            blockers.append("answer_unknown_citation")
        if scope in {"case_grounded", "evidence_grounded"} and not observed:
            blockers.append("answer_citation_missing")
        if scope in {"case_grounded", "evidence_grounded"}:
            expected_citations = {
                (citation.kind, citation.reference_id)
                for claim in selected_claims.values()
                for citation in claim.citation_references
            }
            if observed != expected_citations:
                blockers.append("answer_claim_citation_binding_mismatch")
        if scope not in {"case_grounded", "evidence_grounded"} and observed:
            blockers.append("answer_citation_not_allowed")
    used_ids = values.get("used_context_message_ids")
    if not isinstance(used_ids, list) or not set(used_ids).issubset(allowed_context_ids):
        blockers.append("answer_context_reference_invalid")
    return blockers


def answer_case_conversation(
    request: CaseConversationRequestV2,
    *,
    analyzer: ClaudeDocumentAnalyzer | None,
    stage_callback: ExperienceStageCallback | None = None,
) -> AssistantAnswerV2:
    planning_started = _stage_started(stage_callback, "answer_planning")
    try:
        scope, classification_audit = _classify_question_scope(request, analyzer)
        _stage_finished(
            stage_callback,
            "answer_planning",
            planning_started,
            state="completed",
        )
    except Exception:
        _stage_finished(
            stage_callback,
            "answer_planning",
            planning_started,
            state="failed",
        )
        raise
    claims, guidance = _answer_source_claims(request, scope, classification_audit)

    def observed_fallback(
        *,
        reasons: list[str],
        attempt_count: int,
    ) -> AssistantAnswerV2:
        composition_started = _stage_started(stage_callback, "answer_composition")
        verification_started: int | None = None
        try:
            fallback = _fallback_answer(
                request,
                scope,
                claims,
                guidance,
                classification_audit,
                reasons=reasons,
                attempt_count=attempt_count,
            )
            _stage_finished(
                stage_callback,
                "answer_composition",
                composition_started,
                state="completed",
            )
            verification_started = _stage_started(
                stage_callback,
                "grounding_verification",
            )
            _stage_finished(
                stage_callback,
                "grounding_verification",
                verification_started,
                state="completed",
                safe_counts={"citations": len(fallback.citations)},
            )
            return fallback
        except Exception:
            _stage_finished(
                stage_callback,
                (
                    "grounding_verification"
                    if verification_started is not None
                    else "answer_composition"
                ),
                verification_started or composition_started,
                state="failed",
            )
            raise

    if analyzer is None or scope in {
        "out_of_scope",
        "clarification_needed",
        "insufficient_evidence",
    }:
        return observed_fallback(
            reasons=["claude_answer_unavailable"],
            attempt_count=1,
        )
    blockers: list[str] = []
    allowed_context_ids = {turn.message_id for turn in request.recent_turns}
    fixed_status = "answered"
    for attempt in range(1, 3):
        composition_started = _stage_started(stage_callback, "answer_composition")
        composition_completed = False
        try:
            call = analyzer.run_bounded_agentic_task(
                task_id=CASE_ANSWER_V2_PROMPT_ID,
                system_prompt=load_case_answer_v2_system_prompt(),
                untrusted_context={
                    "question": request.question,
                    "answer_scope": scope,
                    "fixed_status": fixed_status,
                    "allowed_claims": [
                        {
                            **item.model_dump(mode="json", warnings=False),
                            "allowed_answer_areas": sorted(
                                area
                                for area, roles in _ANSWER_AREA_ROLES.items()
                                if item.role in roles
                            ),
                            "protected_factual_atoms": sorted(_factual_atoms(item.canonical_text)),
                        }
                        for item in claims
                    ],
                    "general_guidance": {
                        topic: {
                            "canonical_text": text,
                            "allowed_answer_areas": ["direct_answer", "explanation"],
                            "protected_factual_atoms": sorted(_factual_atoms(text)),
                        }
                        for topic, text in guidance.items()
                    },
                    "fixed_limitation_source": (answer_limitation_source(scope)),
                    "recent_turns": [
                        turn.model_dump(mode="json", warnings=False)
                        for turn in request.recent_turns
                    ],
                    "previous_verification_failures": blockers,
                },
                output_schema=_CHAT_OUTPUT_SCHEMA,
                max_tokens=6144,
            )
            _stage_finished(
                stage_callback,
                "answer_composition",
                composition_started,
                state="completed",
            )
            composition_completed = True
            if call.model_id != request.review_result.runtime_release.model_id:
                raise ValueError("answer_model_identity_mismatch")
            normalized_candidate = {
                "direct_answer": " ".join(call.value["direct_answer"].split()),
                "explanation": [" ".join(item.split()) for item in call.value["explanation"]],
                "evidence_meaning": [
                    " ".join(item.split()) for item in call.value["evidence_meaning"]
                ],
                "impact": [" ".join(item.split()) for item in call.value["impact"]],
                "next_steps": [" ".join(item.split()) for item in call.value["next_steps"]],
                "limitations": [" ".join(item.split()) for item in call.value["limitations"]],
                "claim_selections": call.value["claim_selections"],
                "citation_references": call.value["citation_references"],
                "suggested_questions": _SAFE_SUGGESTED_QUESTIONS,
                "used_context_message_ids": call.value["used_context_message_ids"],
            }
            blockers = _answer_values_safe(
                normalized_candidate,
                claims=claims,
                guidance=guidance,
                allowed_context_ids=allowed_context_ids,
                scope=scope,
            )
            if blockers:
                continue
            verification_started = _stage_started(
                stage_callback,
                "grounding_verification",
            )
            verifier_context = {
                "question": request.question,
                "recent_turns": [
                    turn.model_dump(mode="json", warnings=False) for turn in request.recent_turns
                ],
                "answer_scope": scope,
                "allowed_claims": [item.model_dump(mode="json", warnings=False) for item in claims],
                "general_guidance": guidance,
                "fixed_limitation_source": (answer_limitation_source(scope)),
                "candidate_answer": normalized_candidate,
            }
            try:
                verifier = analyzer.run_bounded_agentic_task(
                    task_id=CASE_ANSWER_VERIFIER_PROMPT_ID,
                    system_prompt=load_case_answer_verifier_system_prompt(),
                    untrusted_context=verifier_context,
                    output_schema=_ANSWER_VERIFIER_SCHEMA,
                    max_tokens=1024,
                )
                _stage_finished(
                    stage_callback,
                    "grounding_verification",
                    verification_started,
                    state="completed",
                    safe_counts={
                        "citations": len(normalized_candidate.get("citation_references") or [])
                    },
                )
            except (LLMError, ValueError, KeyError, TypeError):
                _stage_finished(
                    stage_callback,
                    "grounding_verification",
                    verification_started,
                    state="failed",
                )
                raise
            if (
                verifier.model_id != request.review_result.runtime_release.model_id
                or verifier.value.get("supported") is not True
            ):
                blockers = list(verifier.value.get("reason_codes") or [])
                blockers.append("answer_entailment_failed")
                continue
            citations = [
                CitationReferenceV1.model_validate(item)
                for item in normalized_candidate["citation_references"]
            ]
            answer_content = {
                "request_sha256": case_conversation_request_sha256(request),
                "answer_scope": scope,
                "status": fixed_status,
                "direct_answer": normalized_candidate["direct_answer"],
                "explanation": normalized_candidate["explanation"],
                "evidence_meaning": normalized_candidate["evidence_meaning"],
                "impact": normalized_candidate["impact"],
                "next_steps": normalized_candidate["next_steps"],
                "limitations": normalized_candidate["limitations"],
                "claim_selections": normalized_candidate["claim_selections"],
                "citations": citations,
                # Follow-up prompts are product navigation, not model claims.
                # Keep them deterministic so unverified Claude prose cannot be
                # surfaced outside the claim/citation verifier.
                "suggested_questions": _SAFE_SUGGESTED_QUESTIONS,
                "used_context_message_ids": normalized_candidate["used_context_message_ids"],
            }
            payload = {
                "schema_version": "2.0",
                **answer_content,
                "generation_audit": ExperienceGenerationAuditV1(
                    mode="claude",
                    task_id=CASE_ANSWER_V2_PROMPT_ID,
                    model_id=call.model_id,
                    prompt_sha256=call.prompt_sha256,
                    output_schema_sha256=call.output_schema_sha256,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    latency_ms=call.latency_ms,
                    attempt_count=attempt,
                    scope_classification=classification_audit,
                ),
                "verification_audit": ExperienceVerificationAuditV1(
                    status="passed",
                    deterministic_checks_passed=True,
                    entailment_verifier="claude",
                    verifier_model_id=verifier.model_id,
                    verifier_prompt_sha256=verifier.prompt_sha256,
                    verifier_schema_sha256=verifier.output_schema_sha256,
                    verifier_context_sha256=canonical_json_sha256(verifier_context),
                    verified_content_sha256=assistant_answer_verified_content_sha256(
                        answer_content
                    ),
                    checked_statement_count=(
                        1
                        + len(call.value["explanation"])
                        + len(call.value["evidence_meaning"])
                        + len(call.value["impact"])
                        + len(call.value["next_steps"])
                        + len(call.value["limitations"])
                    ),
                ),
            }
            payload["answer_sha256"] = canonical_json_sha256(payload)
            return AssistantAnswerV2.model_validate(payload)
        except (LLMError, ValueError, KeyError, TypeError):
            if not composition_completed:
                _stage_finished(
                    stage_callback,
                    "answer_composition",
                    composition_started,
                    state="failed",
                )
            blockers = ["claude_answer_or_verifier_failed"]
    return observed_fallback(
        reasons=blockers,
        attempt_count=2,
    )


__all__ = [
    "answer_limitation_source",
    "answer_claim_bindings",
    "answer_case_conversation",
    "build_answer_claim_catalog",
    "build_canonical_claim_catalog",
    "build_grounded_presentation",
    "build_review_experience",
    "build_user_action_events",
    "experience_prompt_hashes",
    "experience_schema_hashes",
]
