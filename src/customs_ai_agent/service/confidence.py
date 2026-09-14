"""Deterministic confidence signals for one review run.

Every number here is derived from facts the review already recorded: whether
operands were grounded in verbatim quotes, whether a focused re-extraction
confirmed a comparison, and what the Data Evidence service returned.  No model
opinion enters the score, so two runs over the same input produce the same
confidence and the same review track.  Both tracks are reviewed by a person;
the track only tells the reviewer how much to expect from the automatic pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from customs_ai_agent.schemas.v3 import (
    ComparisonV3,
    ConfidenceComponentsV3,
    ConfidenceV3,
    DocumentAnalysisV3,
    FindingV3,
)


DataStatus = Literal["ok", "partial", "failed"]

_MISMATCH_BASE = 0.90
_NEEDS_REVIEW_BASE = 0.60
_WARNING_FINDING = 0.50
_EVIDENCE_FINDING = {"ok": 0.85, "partial": 0.60, "failed": 0.30}
_GROUNDING_FACTOR = {2: 1.0, 1: 0.75, 0: 0.5}
_AGREEMENT_BONUS = 0.05
_UNAVAILABLE_PENALTY = 0.15
_UNKNOWN_TYPE_FACTOR = 0.4
_EVIDENCE_COMPONENT = {"ok": 1.0, "partial": 0.6, "failed": 0.0}
_EVIDENCE_SKIPPED_COMPONENT = 0.7
_WEIGHTS = (0.4, 0.4, 0.2)


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 4)


def comparison_key(
    comparison: ComparisonV3,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Identity of a comparison that survives a re-extraction.

    ``comparison_id`` hashes operand fact ids and values, so the same rule on
    the same documents gets a new id after a focused re-read.  The rule code
    and operand locations are stable across the audit loop.
    """

    return (
        comparison.rule_code,
        comparison.left.document_id,
        comparison.left.item_id,
        comparison.right.document_id,
        comparison.right.item_id,
    )


def finding_confidence(
    finding: FindingV3,
    *,
    comparison: ComparisonV3 | None,
    preliminary_status: str | None,
    reextracted_document_ids: frozenset[str],
    unavailable_document_ids: frozenset[str],
    data_status: DataStatus,
) -> float:
    """Confidence that this finding reflects the documents, not the extractor."""

    if finding.kind == "regulatory_finding":
        return _clamp(_EVIDENCE_FINDING[data_status])
    if comparison is None:
        return _clamp(_WARNING_FINDING)
    base = _MISMATCH_BASE if comparison.status == "mismatch" else _NEEDS_REVIEW_BASE
    grounded = sum(1 for operand in (comparison.left, comparison.right) if operand.fact_ids)
    value = base * _GROUNDING_FACTOR[grounded]
    involved = {
        document_id
        for document_id in (comparison.left.document_id, comparison.right.document_id)
        if document_id is not None
    }
    if involved & unavailable_document_ids:
        value -= _UNAVAILABLE_PENALTY
    elif involved & reextracted_document_ids and preliminary_status == comparison.status:
        value += _AGREEMENT_BONUS
    return _clamp(value)


def _grounded_fact_ratio(analysis: DocumentAnalysisV3) -> float:
    facts = list(analysis.facts)
    for item in analysis.line_items:
        facts.extend(item.facts)
    if not facts:
        # A document with no usable fact contributes the weakest extraction
        # signal short of an unknown type.
        return 0.5
    # Facts survive rendering only when their quote was verified, so the
    # ratio measures how many candidate facts the extractor could ground.
    dropped = sum(
        1
        for warning in analysis.warnings
        if warning.code.endswith("_invalid") or warning.code.startswith("fact_")
    )
    return len(facts) / (len(facts) + dropped)


def extraction_component(analyses: Iterable[DocumentAnalysisV3]) -> float:
    values: list[float] = []
    for analysis in analyses:
        type_factor = _UNKNOWN_TYPE_FACTOR if analysis.document_type == "unknown" else 1.0
        values.append(type_factor * _grounded_fact_ratio(analysis))
    if not values:
        return 0.0
    return _clamp(sum(values) / len(values))


def comparison_component(comparisons: Iterable[ComparisonV3]) -> float:
    executed = [comparison for comparison in comparisons if comparison.status != "skip"]
    if not executed:
        return 0.5
    needs_review = sum(1 for comparison in executed if comparison.status == "needs_review")
    return _clamp(1.0 - 0.5 * needs_review / len(executed))


def evidence_component(data_status: DataStatus, *, retrieval_skipped: bool) -> float:
    if retrieval_skipped:
        return _EVIDENCE_SKIPPED_COMPONENT
    return _EVIDENCE_COMPONENT[data_status]


def review_confidence(
    *,
    analyses: Iterable[DocumentAnalysisV3],
    comparisons: Iterable[ComparisonV3],
    data_status: DataStatus,
    retrieval_skipped: bool,
    threshold: float,
) -> ConfidenceV3:
    components = ConfidenceComponentsV3(
        extraction=extraction_component(analyses),
        comparison=comparison_component(comparisons),
        evidence=evidence_component(data_status, retrieval_skipped=retrieval_skipped),
    )
    overall = _clamp(
        _WEIGHTS[0] * components.extraction
        + _WEIGHTS[1] * components.comparison
        + _WEIGHTS[2] * components.evidence
    )
    rounded_threshold = round(threshold, 4)
    return ConfidenceV3(
        overall=overall,
        threshold=rounded_threshold,
        review_track="expedited_check" if overall >= rounded_threshold else "detailed_review",
        components=components,
    )


def attach_finding_confidence(
    findings: Iterable[FindingV3],
    *,
    comparisons_by_id: Mapping[str, ComparisonV3],
    preliminary_status_by_key: Mapping[
        tuple[str, str | None, str | None, str | None, str | None], str
    ],
    reextracted_document_ids: frozenset[str],
    unavailable_document_ids: frozenset[str],
    data_status: DataStatus,
) -> list[FindingV3]:
    scored: list[FindingV3] = []
    for finding in findings:
        comparison = (
            comparisons_by_id.get(finding.comparison_id)
            if finding.comparison_id is not None
            else None
        )
        preliminary_status = (
            preliminary_status_by_key.get(comparison_key(comparison))
            if comparison is not None
            else None
        )
        scored.append(
            finding.model_copy(
                update={
                    "confidence": finding_confidence(
                        finding,
                        comparison=comparison,
                        preliminary_status=preliminary_status,
                        reextracted_document_ids=reextracted_document_ids,
                        unavailable_document_ids=unavailable_document_ids,
                        data_status=data_status,
                    )
                }
            )
        )
    return scored


__all__ = [
    "attach_finding_confidence",
    "comparison_component",
    "comparison_key",
    "evidence_component",
    "extraction_component",
    "finding_confidence",
    "review_confidence",
]
