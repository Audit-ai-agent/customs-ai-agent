from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from customs_ai_agent.schemas.models import ReviewRunResultV2


SAFE_CLAIM_FRAMES: dict[str, dict[str, str]] = {
    "summary": {
        "plain": "",
        "summary_brief": "요약하면, ",
        "summary_label": "사전심사 요약: ",
    },
    "document_consistency": {
        "plain": "",
        "document_result": "문서 정합성 확인 결과, ",
        "document_review": "문서 검토 결과: ",
    },
    "regulatory_review": {
        "plain": "",
        "regulatory_result": "규정 근거 검토 결과, ",
        "regulatory_review": "규정 검토 결과: ",
    },
    "required_actions": {
        "plain": "",
        "action_brief": "필요 조치: ",
        "action_owner": "담당자 조치 사항: ",
    },
    "answer": {
        "plain": "",
        "answer_grounded": "확인된 Case 근거에 따르면, ",
        "answer_evidence": "근거 확인 결과: ",
    },
}

ACTIVE_FAILURE_REASON_CODES = frozenset(
    {
        "active_bounded_execution_failed",
        "active_tool_execution_incomplete",
        "agent_budget_exhausted",
    }
)
PRODUCTION_RELEASE_ACTION_MESSAGE = (
    "Claude extraction과 법령 범위에 대한 독립 검수 및 Release 승인이 필요합니다."
)
SUPPORTING_DOCUMENT_ACTION_MESSAGE = "요건서류의 유효성·대상 범위·신뢰 검증을 완료해야 합니다."
ACTIVE_FAILURE_ACTION_MESSAGE = "제한된 Agent 조사를 완료하지 못해 사람 검토가 필요합니다."


@dataclass(frozen=True, slots=True)
class ExpectedV3Semantics:
    current_fact_entries: tuple[dict[str, Any], ...]
    field_reference_ids: tuple[str, ...]
    issue_reference_ids: tuple[str, ...]
    finding_reference_ids: tuple[str, ...]
    evidence_reference_ids: tuple[str, ...]
    outcome_reference_id: str


def citable_regulatory_evidence_ids(result: ReviewRunResultV2) -> tuple[str, ...]:
    """Return current Evidence that may support a regulatory assessment.

    ``blocked`` records preserve retrieval audit context but are explicitly not
    authority for an automatic regulatory conclusion.
    """

    return tuple(
        item.evidence_id
        for item in result.rag_evidence
        if item.eligibility in {"supporting", "finding"}
    )


def has_citable_regulatory_evidence(result: ReviewRunResultV2) -> bool:
    return bool(citable_regulatory_evidence_ids(result))


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_object_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{_canonical_json_sha256(value)}"


def _display_value(value: Any) -> str:
    if isinstance(value, str):
        rendered = value.strip()
    else:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
    return (rendered or "(empty)")[:4000]


def _fact_payload(
    *,
    kind: str,
    reference_id: str,
    label: str,
    value: Any,
    document_id: str | None = None,
) -> dict[str, Any]:
    rendered = _display_value(value)
    canonical = {
        "kind": kind,
        "reference_id": reference_id,
        "label": label,
        "value": rendered,
        "document_id": document_id,
    }
    return {
        "fact_id": _stable_object_id("fact", canonical),
        **canonical,
        "source_sha256": _canonical_json_sha256(canonical),
    }


def reconstruct_v3_current_semantics(result: ReviewRunResultV2) -> ExpectedV3Semantics:
    """Rebuild the provider-owned current FactLedger projection from V2 sources."""

    entries: list[dict[str, Any]] = []
    field_references: list[str] = []
    for document in result.document_results:
        for field in document.fields:
            reference_id = _stable_object_id(
                "field",
                {
                    "document_id": document.document_id,
                    "field_path": f"fields.{field.name}",
                    "source": field.source.model_dump(mode="json"),
                },
            )
            field_references.append(reference_id)
            entries.append(
                _fact_payload(
                    kind="field_source",
                    reference_id=reference_id,
                    label=f"{document.document_id}.fields.{field.name}",
                    value=(
                        field.normalized_value
                        if field.normalized_value is not None
                        else field.value
                    ),
                    document_id=document.document_id,
                )
            )
        for line_item in document.line_items:
            for field in line_item.fields:
                reference_id = _stable_object_id(
                    "field",
                    {
                        "document_id": document.document_id,
                        "field_path": f"line_items.{line_item.item_id}.{field.name}",
                        "source": field.source.model_dump(mode="json"),
                    },
                )
                field_references.append(reference_id)
                entries.append(
                    _fact_payload(
                        kind="field_source",
                        reference_id=reference_id,
                        label=(
                            f"{document.document_id}.line_items.{line_item.item_id}.{field.name}"
                        ),
                        value=(
                            field.normalized_value
                            if field.normalized_value is not None
                            else field.value
                        ),
                        document_id=document.document_id,
                    )
                )

    issue_references: list[str] = []
    for index, issue in enumerate(result.validation_issues):
        reference_id = _stable_object_id(
            "issue",
            {"index": index, "issue": issue.model_dump(mode="json")},
        )
        issue_references.append(reference_id)
        entries.append(
            _fact_payload(
                kind="issue",
                reference_id=reference_id,
                label=issue.issue_code,
                value=issue.message,
                document_id=issue.source_document_id,
            )
        )

    finding_references: list[str] = []
    for index, finding in enumerate(result.regulatory_findings):
        reference_id = _stable_object_id(
            "finding",
            {"index": index, "finding": finding.model_dump(mode="json")},
        )
        finding_references.append(reference_id)
        entries.append(
            _fact_payload(
                kind="finding",
                reference_id=reference_id,
                label=finding.finding_code,
                value=finding.message,
                document_id=(finding.source.document_id if finding.source is not None else None),
            )
        )

    evidence_references: list[str] = []
    for evidence in result.rag_evidence:
        evidence_references.append(evidence.evidence_id)
        entries.append(
            _fact_payload(
                kind="evidence",
                reference_id=evidence.evidence_id,
                label=evidence.title,
                value=evidence.text,
            )
        )

    outcome_payload = {
        "status": result.status,
        "requires_human_review": result.requires_human_review,
        "retrieval_status": result.retrieval_status,
        "retrieval_coverage": result.retrieval_coverage,
    }
    outcome_reference = _stable_object_id("outcome", outcome_payload)
    entries.append(
        _fact_payload(
            kind="outcome",
            reference_id=outcome_reference,
            label="deterministic_outcome",
            value=outcome_payload,
        )
    )
    entries.sort(key=lambda item: (str(item["kind"]), str(item["reference_id"])))
    return ExpectedV3Semantics(
        current_fact_entries=tuple(entries),
        field_reference_ids=tuple(field_references),
        issue_reference_ids=tuple(issue_references),
        finding_reference_ids=tuple(finding_references),
        evidence_reference_ids=tuple(evidence_references),
        outcome_reference_id=outcome_reference,
    )


def expected_narrative_claims(
    result: ReviewRunResultV2,
    semantics: ExpectedV3Semantics,
    *,
    release_approved: bool,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return an answer-first, code-owned explanation with exact citations.

    Claude may select a presentation frame for these claims, but it cannot
    create, remove, reorder, or change them.  The wording intentionally follows
    the reviewer-facing sequence ``conclusion -> cause/impact -> evidence ->
    action`` while Fact/Issue/Finding/Outcome ownership remains deterministic.
    """

    outcome_citation = ({"kind": "outcome", "reference_id": semantics.outcome_reference_id},)
    if result.validation_issues or result.regulatory_findings:
        summary_text = (
            "사전심사 결론: 확인이 필요한 문서 불일치 또는 미충족 규정 요건이 있어 "
            "담당자 검토가 필요합니다."
        )
    elif (
        result.retrieval_status == "ok"
        and result.retrieval_coverage == "complete"
        and has_citable_regulatory_evidence(result)
    ):
        summary_text = (
            "사전심사 결론: 현재 검토 범위에서는 문서 불일치나 미충족 규정 요건이 "
            "확인되지 않았습니다."
        )
    else:
        summary_text = (
            "사전심사 결론: 규정 근거가 충분하지 않아 결과를 확정할 수 없으며 "
            "담당자 검토가 필요합니다."
        )

    document_consistency = [
        {
            "text": (
                f"문서 검토 원인({issue.issue_code}): {issue.message} "
                "이 검증 문제가 해결되기 전에는 관련 문서의 정합성을 확정할 수 없습니다."
            ),
            "citations": ({"kind": "issue", "reference_id": reference_id},),
        }
        for issue, reference_id in zip(
            result.validation_issues,
            semantics.issue_reference_ids,
            strict=True,
        )
    ]
    if not document_consistency:
        document_consistency.append(
            {
                "text": (
                    "문서 검토 결과: 결정론적 문서 교차검증에서 불일치가 확인되지 않았습니다."
                ),
                "citations": outcome_citation,
            }
        )

    regulatory_review: list[dict[str, Any]] = []
    citable_evidence_ids = citable_regulatory_evidence_ids(result)
    for finding, reference_id in zip(
        result.regulatory_findings,
        semantics.finding_reference_ids,
        strict=True,
    ):
        regulatory_review.append(
            {
                "text": (
                    f"규정 검토 원인({finding.finding_code}): {finding.message} "
                    "이 요건이 충족되기 전에는 규정 검토를 완료할 수 없습니다."
                ),
                "citations": (
                    {"kind": "finding", "reference_id": reference_id},
                    *(
                        {"kind": "evidence", "reference_id": evidence_id}
                        for evidence_id in finding.evidence_ids
                    ),
                ),
            }
        )
    if not regulatory_review:
        if citable_evidence_ids:
            regulatory_review.append(
                {
                    "text": (
                        "규정 검토 결과: 실제 조회된 규정 Evidence 범위에서는 "
                        "별도의 미충족 요건이 확인되지 않았습니다."
                    ),
                    "citations": (
                        {
                            "kind": "evidence",
                            "reference_id": citable_evidence_ids[0],
                        },
                        *outcome_citation,
                    ),
                }
            )
        else:
            regulatory_review.append(
                {
                    "text": (
                        "규정 검토 결과: 인용 가능한 규정 근거가 없어 규정 판단을 보류합니다."
                    ),
                    "citations": outcome_citation,
                }
            )

    required_actions: list[dict[str, Any]] = []
    if result.requires_human_review:
        required_actions.append(
            {
                "text": (
                    "다음 조치: 현재 결과가 담당자 검토를 요구하므로, 담당자는 관련 원문, "
                    "추출값, 규정 Evidence를 확인한 뒤 필요한 수정 또는 승인 결정을 "
                    "기록해야 합니다."
                ),
                "citations": outcome_citation,
            }
        )
    if result.status == "blocked":
        required_actions.append(
            {
                "text": (
                    "다음 조치: 표시된 차단 원인을 먼저 해소한 뒤 새로운 Review Run으로 "
                    "다시 실행해야 합니다."
                ),
                "citations": outcome_citation,
            }
        )
    if not required_actions:
        required_actions.append(
            {
                "text": (
                    "다음 조치: 승인된 자동완료 범위에서는 추가 확인이 필요하지 않습니다."
                    if release_approved
                    else (
                        "다음 조치: 현재 결과는 Release Gate 승인 전까지 자동완료에 "
                        "사용할 수 없으므로 담당자가 최종 확인해야 합니다."
                    )
                ),
                "citations": outcome_citation,
            }
        )

    return {
        "summary": ({"text": summary_text, "citations": outcome_citation},),
        "document_consistency": tuple(document_consistency),
        "regulatory_review": tuple(regulatory_review),
        "required_actions": tuple(required_actions),
    }


def expected_required_actions(
    result: ReviewRunResultV2,
    semantics: ExpectedV3Semantics,
    *,
    completion_eligible: bool,
    supporting_verification_incomplete: bool,
    active_failure_codes: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Rebuild the only user-visible action projection authorized by policy."""

    actions: list[dict[str, Any]] = []
    for issue, reference_id in zip(
        result.validation_issues,
        semantics.issue_reference_ids,
        strict=True,
    ):
        actions.append(
            {
                "action_code": f"resolve_{issue.issue_code.casefold()}",
                "message": issue.message,
                "source_reference_ids": [reference_id],
            }
        )
    for finding, reference_id in zip(
        result.regulatory_findings,
        semantics.finding_reference_ids,
        strict=True,
    ):
        actions.append(
            {
                "action_code": f"resolve_{finding.finding_code.casefold()}",
                "message": finding.message,
                "source_reference_ids": [reference_id, *finding.evidence_ids],
            }
        )
    if not completion_eligible:
        actions.append(
            {
                "action_code": "obtain_production_release_approval",
                "message": PRODUCTION_RELEASE_ACTION_MESSAGE,
                "source_reference_ids": [semantics.outcome_reference_id],
            }
        )
    if supporting_verification_incomplete:
        actions.append(
            {
                "action_code": "verify_supporting_document",
                "message": SUPPORTING_DOCUMENT_ACTION_MESSAGE,
                "source_reference_ids": [semantics.outcome_reference_id],
            }
        )
    for reason_code in sorted(set(active_failure_codes)):
        if reason_code not in ACTIVE_FAILURE_REASON_CODES:
            raise ValueError("active failure action code is outside the policy allowlist")
        actions.append(
            {
                "action_code": reason_code,
                "message": ACTIVE_FAILURE_ACTION_MESSAGE,
                "source_reference_ids": [semantics.outcome_reference_id],
            }
        )
    return tuple(actions)


def allowed_framed_claim_texts(section: str, canonical_text: str) -> frozenset[str]:
    return frozenset(f"{prefix}{canonical_text}" for prefix in SAFE_CLAIM_FRAMES[section].values())


def render_framed_claim(canonical_text: str, frame: str, *, section: str) -> str:
    frames = SAFE_CLAIM_FRAMES.get(section)
    if not canonical_text.strip() or frames is None or frame not in frames:
        raise ValueError("claim frame is not allowed for the declared section")
    rendered = f"{frames[frame]}{canonical_text}"
    if len(rendered) > 4000:
        raise ValueError("rendered canonical claim exceeds the wire limit")
    return rendered


__all__ = [
    "ExpectedV3Semantics",
    "ACTIVE_FAILURE_ACTION_MESSAGE",
    "ACTIVE_FAILURE_REASON_CODES",
    "PRODUCTION_RELEASE_ACTION_MESSAGE",
    "SAFE_CLAIM_FRAMES",
    "SUPPORTING_DOCUMENT_ACTION_MESSAGE",
    "allowed_framed_claim_texts",
    "citable_regulatory_evidence_ids",
    "expected_narrative_claims",
    "expected_required_actions",
    "has_citable_regulatory_evidence",
    "reconstruct_v3_current_semantics",
    "render_framed_claim",
]
