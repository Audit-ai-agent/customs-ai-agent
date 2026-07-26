from __future__ import annotations

import hashlib
import re
import time
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
)
from customs_ai_agent.schemas.v3 import (
    CitationReferenceV1,
    ReviewRunResultV3,
    canonical_json_sha256,
    stable_object_id,
)
from customs_ai_agent.security_guard import prompt_injection_unmet_reason_codes


ExperienceStageCallback = Callable[[dict[str, Any]], None]


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
                    "customs_review",
                ],
            },
        },
    },
}
_PROTECTED_LITERAL = re.compile(
    r"(?<![0-9A-Za-z가-힣])(?:[A-Z]{2,}[A-Z0-9._:/-]*|\d[\d,./:-]*)(?![0-9A-Za-z가-힣])"
)


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


def build_canonical_claim_catalog(result: ReviewRunResultV3) -> list[CanonicalClaimV1]:
    facts = {(entry.kind, entry.reference_id): entry for entry in result.fact_ledger.entries}
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
        for issue in result.validation_issues:
            reference = next(
                (
                    reference_id
                    for kind, reference_id in facts
                    if kind == "issue" and facts[(kind, reference_id)].label == issue.issue_code
                ),
                outcome_entry.reference_id,
            )
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
        for finding in result.regulatory_findings:
            reference = next(
                (
                    reference_id
                    for kind, reference_id in facts
                    if kind == "finding"
                    and facts[(kind, reference_id)].label == finding.finding_code
                ),
                outcome_entry.reference_id,
            )
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
    payload = {
        "schema_version": "1.0",
        "headline": _headline(result),
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
            checked_statement_count=sum(len(item.statements) for item in sections),
            reason_codes=sorted(set(reason_codes or ["claude_presentation_unavailable"])),
        ),
    }
    payload["presentation_sha256"] = canonical_json_sha256(payload)
    return GroundedPresentationV1.model_validate(payload)


def _literal_check(text: str, source_texts: Iterable[str]) -> bool:
    allowed = " ".join(source_texts)
    return all(token in allowed for token in _PROTECTED_LITERAL.findall(text))


def _candidate_presentation(
    result: ReviewRunResultV3,
    claims: list[CanonicalClaimV1],
    call: StructuredAgenticResult,
    *,
    attempt_count: int,
) -> tuple[list[GroundedPresentationSectionV1], list[str]]:
    claim_index = {claim.claim_id: claim for claim in claims}
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
        if not _literal_check(text, (item.canonical_text for item in typed_source)):
            blockers.append("presentation_unbound_literal")
            continue
        used_claims.update(claim_ids)
        statements_by_section[section].append(_statement(typed_source, " ".join(text.split())))
    if used_claims != set(claim_index):
        blockers.append("presentation_claim_coverage_incomplete")
    if any(not statements_by_section[section] for section in _SECTION_ORDER):
        blockers.append("presentation_section_missing")
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
) -> tuple[StructuredAgenticResult, list[str]]:
    statements = [statement for section in sections for statement in section.statements]
    call = analyzer.run_bounded_agentic_task(
        task_id=EXPERIENCE_ENTAILMENT_PROMPT_ID,
        system_prompt=load_experience_entailment_system_prompt(),
        untrusted_context={
            "canonical_claims": [claim.model_dump(mode="json", warnings=False) for claim in claims],
            "candidate_statements": [
                statement.model_dump(mode="json", warnings=False) for statement in statements
            ],
        },
        output_schema=_ENTAILMENT_OUTPUT_SCHEMA,
        max_tokens=4096,
    )
    verdicts = call.value.get("verdicts")
    if not isinstance(verdicts, list):
        return call, ["entailment_verdict_invalid"]
    verdict_index = {
        item.get("statement_id"): item
        for item in verdicts
        if isinstance(item, dict) and isinstance(item.get("statement_id"), str)
    }
    if set(verdict_index) != {statement.statement_id for statement in statements}:
        return call, ["entailment_coverage_incomplete"]
    blockers = sorted(
        {
            str(item.get("reason_code") or "entailment_unsupported")
            for item in verdict_index.values()
            if item.get("supported") is not True
        }
    )
    return call, blockers


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
                    "canonical_claims": [
                        claim.model_dump(mode="json", warnings=False) for claim in claims
                    ],
                    "fixed_headline": _headline(result),
                    "fixed_section_order": list(_SECTION_ORDER),
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
                verification, blockers = _verify_presentation_with_claude(
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
            payload = {
                "schema_version": "1.0",
                "headline": _headline(result),
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


def build_user_action_events(result: ReviewRunResultV3) -> list[UserActionEventV1]:
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
        "user_action_events": build_user_action_events(result),
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
        ):
            raise ValueError("scope_classifier_output_invalid")
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
    claims = build_canonical_claim_catalog(request.review_result)
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
    return claims[:40], guidance


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
    citations = list(
        {
            (citation.kind, citation.reference_id): citation
            for claim in claims
            for citation in claim.citation_references
        }.values()
    )
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
        direct = claims[0].canonical_text
        explanation = [item.canonical_text for item in claims[1:6]]
    evidence_meaning = (
        [item.canonical_text for item in claims if item.role == "evidence"][:5]
        if status == "answered"
        else []
    )
    impact = (
        [item.canonical_text for item in claims if item.role == "impact"][:5]
        if status == "answered"
        else []
    )
    next_steps = (
        [item.canonical_text for item in claims if item.role == "action"][:5]
        if status == "answered"
        else []
    )
    if scope == "general_customs_guidance":
        limitations = [
            "위 내용은 일반 통관 안내이며 현재 심사 건의 판정이나 법적 의무를 뜻하지 않습니다."
        ]
    else:
        limitations = ["답변은 현재 심사 회차와 연결된 문서·법령 근거 범위에 한정됩니다."]
    payload = {
        "schema_version": "2.0",
        "answer_scope": scope,
        "status": status,
        "direct_answer": direct,
        "explanation": explanation,
        "evidence_meaning": evidence_meaning,
        "impact": impact,
        "next_steps": next_steps,
        "limitations": limitations,
        "citations": citations if status == "answered" else [],
        "suggested_questions": [
            "이 결과가 나온 가장 중요한 이유는 무엇입니까?",
            "담당자가 먼저 확인할 항목은 무엇입니까?",
            "연결된 법령 근거가 이 심사 건에서 어떤 의미입니까?",
        ],
        "used_context_message_ids": [turn.message_id for turn in request.recent_turns],
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
            checked_statement_count=(1 + len(explanation) + len(evidence_meaning) + len(impact)),
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
    source_texts = [item.canonical_text for item in claims] + list(guidance.values())
    prose = [
        values.get("direct_answer"),
        *(values.get("explanation") or []),
        *(values.get("evidence_meaning") or []),
        *(values.get("impact") or []),
        *(values.get("next_steps") or []),
        *(values.get("limitations") or []),
    ]
    if any(not isinstance(item, str) for item in prose):
        blockers.append("answer_prose_invalid")
    elif not _literal_check(" ".join(prose), source_texts):
        blockers.append("answer_unbound_literal")
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
                        item.model_dump(mode="json", warnings=False) for item in claims
                    ],
                    "general_guidance": guidance,
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
            blockers = _answer_values_safe(
                call.value,
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
            try:
                verifier = analyzer.run_bounded_agentic_task(
                    task_id=CASE_ANSWER_VERIFIER_PROMPT_ID,
                    system_prompt=load_case_answer_verifier_system_prompt(),
                    untrusted_context={
                        "answer_scope": scope,
                        "allowed_claims": [
                            item.model_dump(mode="json", warnings=False) for item in claims
                        ],
                        "general_guidance": guidance,
                        "candidate_answer": call.value,
                    },
                    output_schema=_ANSWER_VERIFIER_SCHEMA,
                    max_tokens=1024,
                )
                _stage_finished(
                    stage_callback,
                    "grounding_verification",
                    verification_started,
                    state="completed",
                    safe_counts={"citations": len(call.value.get("citation_references") or [])},
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
                for item in call.value["citation_references"]
            ]
            payload = {
                "schema_version": "2.0",
                "answer_scope": scope,
                "status": fixed_status,
                "direct_answer": " ".join(call.value["direct_answer"].split()),
                "explanation": [" ".join(item.split()) for item in call.value["explanation"]],
                "evidence_meaning": [
                    " ".join(item.split()) for item in call.value["evidence_meaning"]
                ],
                "impact": [" ".join(item.split()) for item in call.value["impact"]],
                "next_steps": [" ".join(item.split()) for item in call.value["next_steps"]],
                "limitations": [" ".join(item.split()) for item in call.value["limitations"]],
                "citations": citations,
                "suggested_questions": [
                    " ".join(item.split()) for item in call.value["suggested_questions"]
                ],
                "used_context_message_ids": call.value["used_context_message_ids"],
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
    "answer_case_conversation",
    "build_canonical_claim_catalog",
    "build_grounded_presentation",
    "build_review_experience",
    "build_user_action_events",
    "experience_prompt_hashes",
    "experience_schema_hashes",
]
