"""Advisory eligibility verdict from the expert-review node.

The verdict summarises what the rule engine and the Evidence service already
found; it is not a clearance decision.  ``requires_human_review`` on the
result stays ``true`` regardless of the verdict, and the Backend never closes
a Case from it.  An LLM may only tighten the rule verdict and must cite
finding or evidence identifiers that exist in the result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from customs_ai_agent.schemas.v3 import (
    ComparisonV3,
    EligibilityReasonV3,
    EligibilityVerdictV3,
    FindingV3,
)


Verdict = Literal["eligible", "ineligible", "indeterminate"]

_STRICTNESS: dict[str, int] = {"eligible": 0, "indeterminate": 1, "ineligible": 2}
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_MAX_REASONS = 64


def rule_verdict(
    *,
    findings: Iterable[FindingV3],
    comparisons_by_id: Mapping[str, ComparisonV3],
    execution_status: str,
    data_status: str,
    retrieval_skipped: bool,
    review_track: str,
) -> EligibilityVerdictV3:
    """Deterministic verdict: blocking mismatches, then open questions, else clear."""

    reasons: list[EligibilityReasonV3] = []
    blocking = False
    open_question = False
    for finding in findings:
        comparison = (
            comparisons_by_id.get(finding.comparison_id)
            if finding.comparison_id is not None
            else None
        )
        if (
            comparison is not None
            and comparison.status == "mismatch"
            and finding.severity in {"high", "critical"}
        ):
            blocking = True
            reasons.append(
                EligibilityReasonV3(
                    code="document_mismatch",
                    message=finding.message,
                    finding_ids=[finding.finding_id],
                )
            )
        elif finding.kind == "regulatory_finding":
            open_question = True
            reasons.append(
                EligibilityReasonV3(
                    code="requirement_confirmation_needed",
                    message=finding.message,
                    finding_ids=[finding.finding_id],
                    evidence_ids=list(finding.evidence_ids)[:64],
                )
            )
        elif comparison is not None and comparison.status == "needs_review":
            open_question = True
            reasons.append(
                EligibilityReasonV3(
                    code="comparison_needs_review",
                    message=finding.message,
                    finding_ids=[finding.finding_id],
                )
            )
        elif finding.kind == "validation_issue":
            open_question = True
            reasons.append(
                EligibilityReasonV3(
                    code="extraction_warning",
                    message=finding.message,
                    finding_ids=[finding.finding_id],
                )
            )
    if blocking:
        return EligibilityVerdictV3(
            verdict="ineligible",
            source="rule",
            reasons=[reason for reason in reasons if reason.code == "document_mismatch"][
                :_MAX_REASONS
            ],
        )
    if execution_status != "succeeded":
        reasons.append(
            EligibilityReasonV3(
                code="execution_incomplete",
                message="자동 심사가 완전히 끝나지 않아 적격 여부를 판단할 수 없습니다.",
            )
        )
        open_question = True
    if data_status != "ok" or retrieval_skipped:
        reasons.append(
            EligibilityReasonV3(
                code="evidence_unavailable" if not retrieval_skipped else "evidence_not_requested",
                message=(
                    "법령 근거를 조회하지 않아 요건 충족 여부를 판단할 수 없습니다."
                    if retrieval_skipped
                    else "법령 근거 조회가 완전하지 않아 요건 충족 여부를 판단할 수 없습니다."
                ),
            )
        )
        open_question = True
    if review_track == "detailed_review":
        reasons.append(
            EligibilityReasonV3(
                code="low_confidence",
                message="추출·비교·근거 신호가 임계값에 미치지 못해 정밀 검토가 필요합니다.",
            )
        )
        open_question = True
    if open_question:
        return EligibilityVerdictV3(
            verdict="indeterminate", source="rule", reasons=reasons[:_MAX_REASONS]
        )
    return EligibilityVerdictV3(
        verdict="eligible",
        source="rule",
        reasons=[
            EligibilityReasonV3(
                code="no_blocking_finding",
                message="문서 간 불일치와 미확인 요건이 없어 담당자 확인 대상으로만 제시합니다.",
            )
        ],
    )


def merge_llm_verdict(
    rule: EligibilityVerdictV3,
    answer: Mapping[str, Any],
    *,
    known_finding_ids: frozenset[str],
    known_evidence_ids: frozenset[str],
) -> EligibilityVerdictV3:
    """Accept an LLM verdict only when it is at least as strict and fully cited.

    Anything malformed keeps the rule verdict untouched (fail-closed towards
    the deterministic answer).
    """

    verdict = answer.get("verdict")
    if verdict not in _STRICTNESS or _STRICTNESS[verdict] < _STRICTNESS[rule.verdict]:
        return rule
    raw_reasons = answer.get("reasons")
    if not isinstance(raw_reasons, list):
        return rule
    accepted: list[EligibilityReasonV3] = []
    for raw in raw_reasons[:_MAX_REASONS]:
        if not isinstance(raw, dict):
            return rule
        code = raw.get("code")
        message = raw.get("message")
        finding_ids = raw.get("finding_ids", [])
        evidence_ids = raw.get("evidence_ids", [])
        if (
            not isinstance(code, str)
            or not _REASON_CODE.match(code)
            or not isinstance(message, str)
            or not 1 <= len(message.strip()) <= 2000
            or not isinstance(finding_ids, list)
            or not isinstance(evidence_ids, list)
            or not all(isinstance(value, str) for value in finding_ids)
            or not all(isinstance(value, str) for value in evidence_ids)
            or not set(finding_ids) <= known_finding_ids
            or not set(evidence_ids) <= known_evidence_ids
            or not (finding_ids or evidence_ids)
        ):
            return rule
        accepted.append(
            EligibilityReasonV3(
                code=f"llm_{code}"[:128],
                message=message.strip(),
                finding_ids=list(dict.fromkeys(finding_ids))[:64],
                evidence_ids=list(dict.fromkeys(evidence_ids))[:64],
            )
        )
    if verdict != "eligible" and not accepted:
        # A stricter verdict must point at something in the result.
        return rule
    reasons = [*rule.reasons, *accepted][:_MAX_REASONS]
    return EligibilityVerdictV3(verdict=verdict, source="rule+llm", reasons=reasons)


__all__ = ["merge_llm_verdict", "rule_verdict"]
