"""Text-only wire models for the active Backend-to-Agent endpoint.

The endpoint keeps its historical ``v3`` route and request shape.  Result
``3.2`` keeps the V3.1 source-grounded workflow and adds an inspectable plan,
an eight-section grounded report, and explicitly labelled human corrections.
None of the values is an approval signal.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class V3WireModel(BaseModel):
    """Reject undeclared wire properties at the Agent boundary."""

    model_config = ConfigDict(extra="forbid")


class TextDocumentInputV3(V3WireModel):
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    content_text: str = Field(min_length=1, max_length=200_000)
    declared_type: str | None = Field(default=None, max_length=64)


class ServiceReviewRunRequestV3(V3WireModel):
    case_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_run_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date
    trade_type: Literal["import", "export"]
    jurisdiction: Literal["KR"]
    documents: list[TextDocumentInputV3] = Field(min_length=1, max_length=20)
    parent_review_run_id: str | None = Field(
        default=None,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    applied_correction_proposal_ids: list[str] = Field(default_factory=list, max_length=50)
    human_corrections: list["HumanCorrectionInputV3"] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _document_ids_are_unique(self) -> "ServiceReviewRunRequestV3":
        document_ids = [document.document_id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id values must be unique within a review run")
        proposal_ids = self.applied_correction_proposal_ids
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ValueError("applied correction proposal IDs must be unique")
        correction_ids = [item.correction_proposal_id for item in self.human_corrections]
        if len(correction_ids) != len(set(correction_ids)):
            raise ValueError("human correction proposal IDs must be unique")
        if set(correction_ids) != set(proposal_ids):
            raise ValueError("human corrections must exactly match applied correction proposal IDs")
        if any(item.document_id not in set(document_ids) for item in self.human_corrections):
            raise ValueError("human correction document_id must refer to an input document")
        return self

    def input_manifest_wire(self) -> dict[str, Any]:
        """Return exactly the compact request shape sent by the Backend.

        Pydantic materialises the optional child-run fields after decoding a
        baseline request.  They were absent on the HTTP wire, so including
        their ``None``/empty defaults in the result binding hash would make a
        valid Backend request fail closed.  Only a child run carries those
        three fields; nested optional correction values are likewise omitted.
        """

        wire = self.model_dump(mode="json", exclude_none=True, warnings=False)
        if self.parent_review_run_id is None:
            wire.pop("applied_correction_proposal_ids", None)
            wire.pop("human_corrections", None)
        return wire


class ReviewReportV3(V3WireModel):
    overall_status: Literal["needs_review", "blocked"]
    summary: str = Field(min_length=1, max_length=4000)
    disclaimer: str = Field(min_length=1, max_length=2000)
    sections: list["ReportSectionV3"] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _has_exactly_one_of_each_report_section(self) -> "ReviewReportV3":
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
        if {section.section for section in self.sections} != expected:
            raise ValueError("report must include each required section exactly once")
        return self


class EvidenceV3(V3WireModel):
    evidence_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=20_000)
    source_uri: str = Field(min_length=1, max_length=4000)
    source_type: str = Field(min_length=1, max_length=128)
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    law_id: str | None = Field(default=None, max_length=512)
    law_name: str | None = Field(default=None, max_length=2000)


class TemporalContextV3(V3WireModel):
    """Dates that bind one Evidence response to its immutable sources.

    This is provenance metadata, not a decision or an assertion that the
    HSK/requirements snapshots apply to every historical review date.  Data
    owns its construction; Agent preserves the validated values for Backend
    persistence and the user-facing snapshot notice.
    """

    law_as_of: date
    hsk_snapshot_as_of: date
    requirements_snapshot_as_of: date


class TemporalCoverageV3(V3WireModel):
    """Verified availability for each date-bearing Evidence source family.

    This deliberately does not infer coverage from the dates in
    ``TemporalContextV3``.  A historical request can retain the baseline HSK
    or CCCT date while Data explicitly marks that family unavailable.
    """

    law: Literal["covered", "unavailable"]
    hsk: Literal["covered", "unavailable"]
    requirements: Literal["covered", "unavailable"]


class SourceV3(V3WireModel):
    """A verbatim quote and its zero-based occurrence in one input document."""

    source_kind: Literal["document"] = "document"
    document_id: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=1, max_length=4000)
    occurrence: int = Field(ge=0, le=100_000)


class HumanProvidedSourceV3(V3WireModel):
    """A reviewer-supplied fact that deliberately has no invented document quote."""

    source_kind: Literal["human_provided"] = "human_provided"
    document_id: str = Field(min_length=1, max_length=128)
    correction_proposal_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2000)


class HumanCorrectionInputV3(V3WireModel):
    """One already-approved correction for a new child review run."""

    correction_proposal_id: str = Field(min_length=1, max_length=128)
    document_id: str = Field(min_length=1, max_length=128)
    item_id: str | None = Field(default=None, max_length=256)
    fact_id: str | None = Field(default=None, max_length=256)
    field_name: str = Field(min_length=1, max_length=64)
    new_value: str = Field(min_length=1, max_length=2000)
    unit: str | None = Field(default=None, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)
    source_kind: Literal["human_provided"] = "human_provided"


class FactV3(V3WireModel):
    fact_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=64)
    raw_value: str = Field(min_length=1, max_length=2000)
    normalized_value: str | None = Field(default=None, max_length=2000)
    unit: str | None = Field(default=None, max_length=64)
    source: SourceV3 | HumanProvidedSourceV3


class LineItemV3(V3WireModel):
    item_id: str = Field(min_length=1, max_length=256)
    source: SourceV3
    facts: list[FactV3] = Field(default_factory=list, max_length=32)


class ExtractionWarningV3(V3WireModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)


class DocumentAnalysisV3(V3WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    filename: str = Field(min_length=1, max_length=255)
    declared_type: str | None = Field(default=None, max_length=64)
    document_type: Literal[
        "invoice",
        "packing_list",
        "bill_of_lading",
        "air_waybill",
        "certificate_of_origin",
        "export_declaration",
        "supporting_evidence",
        "unknown",
    ]
    classification_source: SourceV3 | None = None
    extraction_status: Literal["complete", "needs_review"]
    facts: list[FactV3] = Field(default_factory=list, max_length=64)
    line_items: list[LineItemV3] = Field(default_factory=list, max_length=100)
    warnings: list[ExtractionWarningV3] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _facts_belong_to_document(self) -> "DocumentAnalysisV3":
        if (
            self.classification_source
            and self.classification_source.document_id != self.document_id
        ):
            raise ValueError("classification source must refer to its document")
        fact_ids = [fact.fact_id for fact in self.facts]
        for item in self.line_items:
            if item.source.document_id != self.document_id:
                raise ValueError("line item source must refer to its document")
            fact_ids.extend(fact.fact_id for fact in item.facts)
            if any(fact.source.document_id != self.document_id for fact in item.facts):
                raise ValueError("line item facts must refer to their document")
        if any(fact.source.document_id != self.document_id for fact in self.facts):
            raise ValueError("document facts must refer to their document")
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact_id values must be unique within a document analysis")
        return self


class ComparisonOperandV3(V3WireModel):
    document_id: str | None = Field(default=None, max_length=128)
    item_id: str | None = Field(default=None, max_length=256)
    fact_ids: list[str] = Field(default_factory=list, max_length=8)
    display_value: str | None = Field(default=None, max_length=4000)


class ComparisonV3(V3WireModel):
    comparison_id: str = Field(min_length=1, max_length=256)
    rule_code: str = Field(min_length=1, max_length=128)
    status: Literal["pass", "skip", "mismatch", "needs_review"]
    left: ComparisonOperandV3
    right: ComparisonOperandV3
    expected_value: str | None = Field(default=None, max_length=2000)
    observed_value: str | None = Field(default=None, max_length=2000)
    unit: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


class ReextractionV3(V3WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    attempt_count: Literal[1] = 1
    requested_fields: list[str] = Field(min_length=1, max_length=32)
    status: Literal["completed", "unavailable"]


class WorkflowStageV3(V3WireModel):
    stage: Literal[
        "classify_documents",
        "extract_facts",
        "normalize_facts",
        "match_documents",
        "validate_documents",
        "reextract_documents",
        "retrieve_evidence",
        "generate_result",
    ]
    status: Literal["completed", "skipped", "needs_review", "failed"]
    attempt_count: int = Field(ge=0, le=40)
    summary: str = Field(min_length=1, max_length=2000)


class PlannerActionV3(V3WireModel):
    """A high-level action from the intentionally small Agent tool surface."""

    cycle: int = Field(ge=1, le=2)
    action: Literal[
        "analyze_documents",
        "apply_human_corrections",
        "compare_documents",
        "reextract_documents",
        "retrieve_evidence",
        "generate_report",
    ]
    status: Literal["completed", "skipped", "needs_review", "failed"]
    summary: str = Field(min_length=1, max_length=2000)
    correction_proposal_ids: list[str] = Field(default_factory=list, max_length=50)


class PlannerTraceV3(V3WireModel):
    cycle_count: int = Field(ge=1, le=2)
    actions: list[PlannerActionV3] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def _is_bounded_and_ordered(self) -> "PlannerTraceV3":
        if any(action.cycle > self.cycle_count for action in self.actions):
            raise ValueError("planner action cycle must not exceed cycle_count")
        if self.actions != sorted(self.actions, key=lambda action: action.cycle):
            raise ValueError("planner actions must be ordered by cycle")
        return self


class WorkflowV3(V3WireModel):
    status: Literal["completed", "partial", "failed"]
    stages: list[WorkflowStageV3] = Field(min_length=1, max_length=8)
    reextractions: list[ReextractionV3] = Field(default_factory=list, max_length=20)
    planner: PlannerTraceV3

    @model_validator(mode="after")
    def _stages_are_unique(self) -> "WorkflowV3":
        stages = [stage.stage for stage in self.stages]
        if len(stages) != len(set(stages)):
            raise ValueError("workflow stage values must be unique")
        reextraction_ids = [entry.document_id for entry in self.reextractions]
        if len(reextraction_ids) != len(set(reextraction_ids)):
            raise ValueError("a document can be re-extracted at most once")
        return self


class FindingV3(V3WireModel):
    finding_id: str = Field(min_length=1, max_length=256)
    kind: Literal["validation_issue", "regulatory_finding"]
    code: str = Field(min_length=1, max_length=128)
    severity: Literal["low", "medium", "high", "critical"]
    message: str = Field(min_length=1, max_length=4000)
    item_id: str | None = Field(default=None, max_length=128)
    evidence_ids: list[str] = Field(default_factory=list, max_length=256)
    comparison_id: str | None = Field(default=None, max_length=256)
    document_ids: list[str] = Field(default_factory=list, max_length=20)
    fact_ids: list[str] = Field(default_factory=list, max_length=32)


class RequiredActionV1(V3WireModel):
    action_code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    finding_ids: list[str] = Field(default_factory=list, max_length=256)


class ReportClaimV3(V3WireModel):
    claim_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=4000)
    fact_ids: list[str] = Field(default_factory=list, max_length=64)
    comparison_ids: list[str] = Field(default_factory=list, max_length=64)
    evidence_ids: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _is_grounded(self) -> "ReportClaimV3":
        if not (self.fact_ids or self.comparison_ids or self.evidence_ids):
            raise ValueError("report claim must reference a fact, comparison, or evidence")
        return self


class ReportSectionV3(V3WireModel):
    section: Literal[
        "conclusion",
        "execution_process",
        "document_comparison",
        "issues_and_causes",
        "legal_evidence",
        "impact",
        "next_actions",
        "limitations",
    ]
    title: str = Field(min_length=1, max_length=128)
    claims: list[ReportClaimV3] = Field(min_length=1, max_length=32)


class EngineV3(V3WireModel):
    kind: Literal["deterministic", "claude"]
    model_id: str = Field(min_length=1, max_length=256)


class ReviewRunResultV3(V3WireModel):
    case_id: str = Field(min_length=1, max_length=128)
    review_run_id: str = Field(min_length=1, max_length=128)
    review_as_of: date
    trade_type: Literal["import", "export"]
    jurisdiction: Literal["KR"]
    status: Literal["needs_review", "blocked"]
    requires_human_review: Literal[True] = True
    execution_status: Literal["succeeded", "partial", "failed"]
    assessment: Literal["no_issue_detected", "issues_detected", "indeterminate"]
    input_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    report: ReviewReportV3
    workflow: WorkflowV3
    document_analyses: list[DocumentAnalysisV3] = Field(min_length=1, max_length=20)
    comparisons: list[ComparisonV3] = Field(default_factory=list, max_length=2000)
    evidence: list[EvidenceV3] = Field(default_factory=list, max_length=10_000)
    findings: list[FindingV3] = Field(default_factory=list, max_length=2000)
    required_actions: list[RequiredActionV1] = Field(min_length=1, max_length=2001)
    # Additive provenance from the Data Evidence response.  Old three-law
    # results may legitimately omit it; the new law-history profile supplies
    # all three dates as one atomic context.
    temporal_context: TemporalContextV3 | None = None
    # Additive availability state.  Law-history runs require this companion
    # object; legacy serialized results may still omit it.
    temporal_coverage: TemporalCoverageV3 | None = None
    result_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    schema_version: Literal["3.2"] = "3.2"
    engine: EngineV3 | None = None

    @model_validator(mode="after")
    def _human_only_and_hashed(self) -> "ReviewRunResultV3":
        if self.status == "blocked" and self.execution_status != "failed":
            raise ValueError("blocked result must have failed execution_status")
        if self.status == "needs_review" and self.report.overall_status != "needs_review":
            raise ValueError("needs_review result must have needs_review report status")
        if (
            self.temporal_context is not None
            and self.temporal_context.law_as_of != self.review_as_of
        ):
            raise ValueError("temporal context law_as_of must match review_as_of")
        if self.temporal_coverage is not None and self.temporal_context is None:
            raise ValueError("temporal coverage requires temporal context")
        comparison_ids = [comparison.comparison_id for comparison in self.comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("comparison_id values must be unique")
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("finding_id values must be unique")
        known_findings = set(finding_ids)
        known_comparisons = set(comparison_ids)
        known_evidence = {entry.evidence_id for entry in self.evidence}
        known_facts: set[str] = set()
        for analysis in self.document_analyses:
            known_facts.update(fact.fact_id for fact in analysis.facts)
            for item in analysis.line_items:
                known_facts.update(fact.fact_id for fact in item.facts)
        if any(
            finding.comparison_id is not None and finding.comparison_id not in known_comparisons
            for finding in self.findings
        ):
            raise ValueError("finding comparison_id must refer to a comparison")
        if any(
            finding_id not in known_findings
            for action in self.required_actions
            for finding_id in action.finding_ids
        ):
            raise ValueError("action finding_ids must refer to findings")
        for section in self.report.sections:
            for claim in section.claims:
                if not set(claim.fact_ids) <= known_facts:
                    raise ValueError("report claim fact_ids must refer to extracted facts")
                if not set(claim.comparison_ids) <= known_comparisons:
                    raise ValueError("report claim comparison_ids must refer to comparisons")
                if not set(claim.evidence_ids) <= known_evidence:
                    raise ValueError("report claim evidence_ids must refer to evidence")
        if self.result_sha256 != review_result_v3_sha256(self):
            raise ValueError("result_sha256 does not match the public result")
        return self


def _canonical_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported canonical value: {type(value)!r}")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        default=_canonical_default,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def review_result_v3_sha256(result: ReviewRunResultV3 | Mapping[str, Any]) -> str:
    if isinstance(result, ReviewRunResultV3):
        payload = result.model_dump(mode="json", warnings=False)
    else:
        payload = dict(result)
    payload.pop("result_sha256", None)
    # ``temporal_context`` was added after existing immutable Review Runs had
    # already been hashed.  A null context carries no provenance, so omit only
    # that null additive field to keep those legacy snapshots readable.  A
    # present date bundle remains covered by the result hash.
    if payload.get("temporal_context") is None:
        payload.pop("temporal_context", None)
    if payload.get("temporal_coverage") is None:
        payload.pop("temporal_coverage", None)
    return canonical_json_sha256(payload)


__all__ = [
    "ComparisonOperandV3",
    "ComparisonV3",
    "DocumentAnalysisV3",
    "EngineV3",
    "EvidenceV3",
    "ExtractionWarningV3",
    "FactV3",
    "FindingV3",
    "HumanCorrectionInputV3",
    "HumanProvidedSourceV3",
    "LineItemV3",
    "PlannerActionV3",
    "PlannerTraceV3",
    "ReextractionV3",
    "ReportClaimV3",
    "ReportSectionV3",
    "RequiredActionV1",
    "ReviewReportV3",
    "ReviewRunResultV3",
    "ServiceReviewRunRequestV3",
    "SourceV3",
    "TextDocumentInputV3",
    "TemporalContextV3",
    "TemporalCoverageV3",
    "WorkflowStageV3",
    "WorkflowV3",
    "canonical_json_sha256",
    "review_result_v3_sha256",
]
