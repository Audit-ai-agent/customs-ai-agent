"""Deterministic, source-linked eight-section review report.

The report deliberately does not ask an LLM to invent a narrative.  Every
claim carries one or more identifiers already present in the public review
result, so a caller can follow it back to a fact, comparison, or Evidence
record.  This is also the safe fallback when a functional-UAT provider is
unavailable after extraction.
"""

from __future__ import annotations

from collections.abc import Iterable

from customs_ai_agent.schemas.v3 import (
    ComparisonV3,
    DocumentAnalysisV3,
    EvidenceV3,
    FactV3,
    FindingV3,
    HumanProvidedSourceV3,
    ReportClaimV3,
    ReportSectionV3,
    RequiredActionV1,
)
from customs_ai_agent.service.user_text import (
    contains_internal_result_marker,
    localize_deterministic_reviewer_prose,
    review_result_markers,
    reviewer_text_is_safe,
)


def build_grounded_report_sections(
    *,
    analyses: list[DocumentAnalysisV3],
    comparisons: list[ComparisonV3],
    evidence: list[EvidenceV3],
    findings: list[FindingV3],
    actions: list[RequiredActionV1],
    evidence_retrieval_skip_reason: str | None = None,
) -> list[ReportSectionV3]:
    """Build all required sections using only identifiers already in the result."""

    markers = review_result_markers(
        {
            "document_analyses": [analysis.model_dump(mode="json") for analysis in analyses],
            "comparisons": [comparison.model_dump(mode="json") for comparison in comparisons],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "findings": [finding.model_dump(mode="json") for finding in findings],
            "required_actions": [action.model_dump(mode="json") for action in actions],
        }
    )
    filenames = tuple(analysis.filename for analysis in analyses)
    fact_ids = [fact.fact_id for analysis in analyses for fact in _facts(analysis)]
    primary_comparison = _choose_comparison(comparisons)
    support = _support(
        fact_ids=fact_ids,
        comparison_id=primary_comparison.comparison_id if primary_comparison else None,
        evidence_id=evidence[0].evidence_id if evidence else None,
    )
    issue_comparison = next(
        (item for item in comparisons if item.status in {"mismatch", "needs_review"}),
        primary_comparison,
    )
    issue_support = _support(
        fact_ids=[],
        comparison_id=issue_comparison.comparison_id if issue_comparison else None,
        evidence_id=evidence[0].evidence_id if evidence else None,
    )
    finding = findings[0] if findings else None
    finding_support = _support(
        fact_ids=finding.fact_ids if finding else [],
        comparison_id=finding.comparison_id if finding else None,
        evidence_id=(finding.evidence_ids[0] if finding and finding.evidence_ids else None),
        fallback=issue_support,
    )
    evidence_support = _support(
        fact_ids=[],
        comparison_id=primary_comparison.comparison_id if primary_comparison else None,
        evidence_id=evidence[0].evidence_id if evidence else None,
        fallback=support,
    )
    action_support = _support(
        fact_ids=[],
        comparison_id=(finding.comparison_id if finding else None),
        evidence_id=(finding.evidence_ids[0] if finding and finding.evidence_ids else None),
        fallback=issue_support,
    )
    comparison_text = _reviewer_sentence(
        issue_comparison.message if issue_comparison is not None else None,
        "문서 사이의 비교 결과를 확인해야 합니다.",
        markers=markers,
        filenames=filenames,
    )
    evidence_text = (
        f"연결된 법령 근거는 {len(evidence)}건입니다."
        if evidence
        else _evidence_skip_message(evidence_retrieval_skip_reason)
    )
    finding_text = _reviewer_sentence(
        finding.message if finding is not None else None,
        "자동으로 확정할 수 없는 항목이 있어 담당자 검토가 필요합니다.",
        markers=markers,
        filenames=filenames,
    )
    action_text = _reviewer_sentence(
        actions[0].message if actions else None,
        "문서와 법령 근거를 담당자가 직접 확인해 주세요.",
        markers=markers,
        filenames=filenames,
    )
    sections = [
        _section(
            "conclusion",
            "결론",
            f"문서 심사는 자동 완료하지 않으며, 확인 항목 {len(findings)}건을 사람 검토로 남겼습니다.",
            finding_support,
        ),
        _section(
            "execution_process",
            "실행 과정",
            "문서 추출·정규화·대응·비교와 제한된 근거 조회를 순서대로 수행했습니다.",
            support,
        ),
        _section("document_comparison", "문서 비교", comparison_text, issue_support),
        _section("issues_and_causes", "문제와 원인", finding_text, finding_support),
        _section("legal_evidence", "법령 근거", evidence_text, evidence_support),
        _section(
            "impact",
            "영향",
            "확인되지 않은 비교값이나 법령 근거는 담당자의 추가 확인 없이는 사건 판단에 사용할 수 없습니다.",
            issue_support,
        ),
        _section("next_actions", "다음 조치", action_text, action_support),
        _section(
            "limitations",
            "한계",
            "이 결과는 텍스트 문서와 연결된 법령 근거만 사용한 검토 보조 결과이며 최종 통관 판단이 아닙니다.",
            issue_support,
        ),
    ]
    human_facts: list[tuple[FactV3, HumanProvidedSourceV3]] = []
    for analysis in analyses:
        for fact in _facts(analysis):
            if isinstance(fact.source, HumanProvidedSourceV3):
                human_facts.append((fact, fact.source))
    if not human_facts:
        return sections
    correction_ids = list(
        dict.fromkeys(source.correction_proposal_id for _fact, source in human_facts)
    )
    correction_fact_ids = [fact.fact_id for fact, _source in human_facts]
    execution = sections[1]
    sections[1] = execution.model_copy(
        update={
            "claims": [
                *execution.claims,
                ReportClaimV3(
                    claim_id=(
                        "report:execution_process:human-corrections:" + ",".join(correction_ids)
                    )[:256],
                    text=(
                        "승인된 수정 내용을 새 심사에 반영했으며, 원문 인용은 그대로 보존했습니다."
                    ),
                    fact_ids=correction_fact_ids,
                    comparison_ids=[],
                    evidence_ids=[],
                ),
            ]
        }
    )
    return sections


def _evidence_skip_message(reason: str | None) -> str:
    if reason == "no_hs_qualified_commodity":
        return "HS 코드가 확인된 품목이 없어 법령 근거 조회를 실행하지 않았습니다. 해당 문서의 HS 코드를 보완해 주세요."
    if reason == "export_destination_country_missing":
        return (
            "수출 목적국 코드가 문서에서 확인되지 않아 법령 근거 조회를 실행하지 않았습니다. "
            "수출신고증 또는 운송 문서의 두 자리 목적국 코드를 보완해 주세요."
        )
    return "연결 가능한 법령 근거가 없어 담당자 확인이 필요합니다."


def validate_grounded_report_sections(
    sections: list[ReportSectionV3],
    *,
    analyses: list[DocumentAnalysisV3],
    comparisons: list[ComparisonV3],
    evidence: list[EvidenceV3],
) -> None:
    """Reject a provider report that cites an unknown public object."""

    expected = {
        "conclusion",
        "execution_process",
        "document_comparison",
        "issues_and_causes",
        "legal_evidence",
        "impact",
        "next_actions",
        "limitations",
    }
    if len(sections) != 8 or {section.section for section in sections} != expected:
        raise ValueError("report_sections_invalid")
    facts = {fact.fact_id for analysis in analyses for fact in _facts(analysis)}
    comparison_ids = {item.comparison_id for item in comparisons}
    evidence_ids = {item.evidence_id for item in evidence}
    for section in sections:
        for claim in section.claims:
            if not set(claim.fact_ids) <= facts:
                raise ValueError("report_claim_fact_invalid")
            if not set(claim.comparison_ids) <= comparison_ids:
                raise ValueError("report_claim_comparison_invalid")
            if not set(claim.evidence_ids) <= evidence_ids:
                raise ValueError("report_claim_evidence_invalid")
    expected_corrections: set[str] = set()
    for analysis in analyses:
        for fact in _facts(analysis):
            if isinstance(fact.source, HumanProvidedSourceV3):
                expected_corrections.add(fact.fact_id)
    if not expected_corrections:
        return
    execution = next(section for section in sections if section.section == "execution_process")
    for fact_id in expected_corrections:
        if not any(
            fact_id in claim.fact_ids
            and "원문 인용" in claim.text
            and not contains_internal_result_marker(claim.text)
            for claim in execution.claims
        ):
            raise ValueError("report_human_correction_claim_missing")


def _facts(analysis: DocumentAnalysisV3) -> Iterable[FactV3]:
    yield from analysis.facts
    for item in analysis.line_items:
        yield from item.facts


def _choose_comparison(comparisons: list[ComparisonV3]) -> ComparisonV3 | None:
    return next(
        (item for item in comparisons if item.status in {"mismatch", "needs_review"}),
        comparisons[0] if comparisons else None,
    )


def _reviewer_sentence(
    value: str | None,
    fallback: str,
    *,
    markers: frozenset[str],
    filenames: tuple[str, ...],
) -> str:
    """Render only Korean-first reviewer prose outside structured citations."""

    if not value or contains_internal_result_marker(value, markers=markers):
        return fallback
    localized = localize_deterministic_reviewer_prose(value)
    if not reviewer_text_is_safe([localized], markers=markers, filenames=filenames):
        return fallback
    return localized


def _support(
    *,
    fact_ids: list[str],
    comparison_id: str | None,
    evidence_id: str | None,
    fallback: tuple[list[str], list[str], list[str]] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    if comparison_id:
        return [], [comparison_id], []
    if evidence_id:
        return [], [], [evidence_id]
    if fact_ids:
        return [fact_ids[0]], [], []
    if fallback is not None:
        return fallback
    # Every review has at least the document-type comparison.  This branch is
    # defensive and is intentionally never used for a valid review result.
    raise ValueError("grounded report has no available reference")


def _section(
    name: str,
    title: str,
    text: str,
    support: tuple[list[str], list[str], list[str]],
) -> ReportSectionV3:
    fact_ids, comparison_ids, evidence_ids = support
    reference = (fact_ids or comparison_ids or evidence_ids)[0]
    return ReportSectionV3(
        section=name,  # type: ignore[arg-type]
        title=title,
        claims=[
            ReportClaimV3(
                claim_id=f"report:{name}:{reference}"[:256],
                text=text,
                fact_ids=fact_ids,
                comparison_ids=comparison_ids,
                evidence_ids=evidence_ids,
            )
        ],
    )


__all__ = ["build_grounded_report_sections", "validate_grounded_report_sections"]
