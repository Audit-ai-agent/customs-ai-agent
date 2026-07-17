from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


DocumentType = Literal[
    "bill_of_lading",
    "invoice",
    "packing_list",
    "certificate_of_origin",
    "unknown",
]
Severity = Literal["low", "medium", "high", "critical"]
ReviewStatus = Literal["completed", "needs_review", "blocked"]
TradeType = Literal["import", "export"]
RetrievalStatus = Literal["ok", "partial", "failed"]
Coverage = Literal["complete", "partial", "unknown"]
HsResolution = Literal["exact", "unique_descendant", "ambiguous", "invalid", "missing"]


def stable_evidence_id(source_record_id: str, version_hash: str) -> str:
    canonical = f"{source_record_id}\x1f{version_hash}".encode("utf-8")
    return "ev_" + hashlib.sha256(canonical).hexdigest()


class BBox(BaseModel):
    page: int = Field(ge=1)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def _valid_box(self) -> "BBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox x2/y2 must be greater than x1/y1")
        return self


class FieldEvidence(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    bbox: BBox | None = None
    raw_text: str | None = None
    text_start: int | None = Field(default=None, ge=0)
    text_end: int | None = Field(default=None, ge=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _valid_text_span(self) -> "FieldEvidence":
        if (self.text_start is None) != (self.text_end is None):
            raise ValueError("text_start and text_end must be provided together")
        if (
            self.text_start is not None
            and self.text_end is not None
            and self.text_end <= self.text_start
        ):
            raise ValueError("text_end must be greater than text_start")
        return self


class ExtractedField(BaseModel):
    name: str
    value: Any
    normalized_value: Any | None = None
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal[
        "rule_heuristic", "source_quote_grounded_uncalibrated"
    ] = "rule_heuristic"
    evidence: FieldEvidence


class DocumentInput(BaseModel):
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    content_text: str = Field(min_length=1, max_length=200_000)
    declared_type: DocumentType | None = None
    requirement_document_codes: list[str] = Field(default_factory=list)


class ReviewRunRequest(BaseModel):
    scenario_id: str | None = Field(default=None, min_length=1, max_length=128)
    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    documents: list[DocumentInput] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
    )
    llm_mode: Literal["mock", "llm"] = "mock"
    review_run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date = Field(default_factory=date.today)
    trade_type: TradeType = "import"
    jurisdiction: Literal["KR"] = "KR"
    uploaded_requirement_document_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _scenario_or_documents(self) -> "ReviewRunRequest":
        if bool(self.scenario_id) == bool(self.documents):
            raise ValueError("exactly one of scenario_id or documents is required")
        if self.documents and sum(len(doc.content_text) for doc in self.documents) > 500_000:
            raise ValueError("combined document text exceeds the per-run safety limit")
        if self.documents:
            document_ids = [document.document_id for document in self.documents]
            if len(document_ids) != len(set(document_ids)):
                raise ValueError("document_id values must be unique within a review run")
        self.uploaded_requirement_document_codes = sorted(
            {
                code.strip()
                for code in self.uploaded_requirement_document_codes
                if code.strip()
            }
        )
        return self


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelCallRecord(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    provider: Literal["anthropic"]
    model_id: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=64)
    request_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)
    stop_reason: str = Field(min_length=1, max_length=64)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class ModelErrorDetail(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    retryable: bool
    provider: Literal["anthropic"] | None = None
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_version: str | None = Field(default=None, min_length=1, max_length=64)
    latency_ms: int = Field(default=0, ge=0)
    provider_request_id: str | None = Field(default=None, max_length=256)
    provider_status_code: int | None = Field(default=None, ge=100, le=599)


class RetrievalQueryItem(WireModel):
    item_id: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000)
    hs_code: str | None = Field(default=None, min_length=1, max_length=64)
    origin_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    destination_country: str = Field(pattern=r"^[A-Z]{2}$")


class EvidenceRetrievalRequest(WireModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=128)
    review_run_id: str = Field(min_length=1, max_length=128)
    review_as_of: date
    jurisdiction: Literal["KR"]
    trade_type: TradeType
    top_k: int = Field(default=8, ge=1, le=100)
    items: list[RetrievalQueryItem] = Field(min_length=1, max_length=100)
    uploaded_requirement_document_codes: list[str] = Field(default_factory=list)
    validation_issue_codes: list[str] = Field(default_factory=list)
    query_intent: str = Field(default="required_import_documents", min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _unique_item_ids(self) -> "EvidenceRetrievalRequest":
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("retrieval request item_id values must be unique")
        if len(self.uploaded_requirement_document_codes) != len(
            set(self.uploaded_requirement_document_codes)
        ):
            raise ValueError("uploaded requirement document codes must be unique")
        if len(self.validation_issue_codes) != len(set(self.validation_issue_codes)):
            raise ValueError("validation issue codes must be unique")
        if any(
            not code or len(code) > 128
            for code in self.uploaded_requirement_document_codes
        ):
            raise ValueError("uploaded requirement document codes must be 1..128 chars")
        if any(not code or len(code) > 128 for code in self.validation_issue_codes):
            raise ValueError("validation issue codes must be 1..128 chars")
        return self


class UserCorrection(BaseModel):
    document_id: str
    field_path: str
    value: Any
    confidence: float = Field(default=1.0, ge=0, le=1)


class ValidationIssue(BaseModel):
    issue_code: str
    severity: Severity
    message: str
    expected_value: str | None = None
    observed_value: str | None = None
    source_document_id: str | None = None
    target_document_id: str | None = None
    source_bbox: BBox | None = None
    target_bbox: BBox | None = None
    confidence: float = Field(default=0.95, ge=0, le=1)
    confidence_basis: Literal[
        "deterministic_rule", "source_field_minimum"
    ] = "deterministic_rule"
    needs_regulatory_review: bool = False


class RetrievalErrorDetail(WireModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class RagEvidence(WireModel):
    evidence_id: str = Field(pattern=r"^ev_[a-f0-9]{64}$")
    source_record_id: str = Field(min_length=1)
    source_type: Literal["law_article", "customs_requirement", "hsk_entry", "notice"]
    item_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    law_id: str | None = None
    law_name: str | None = None
    article_no: str | None = None
    source_uri: str = Field(min_length=1)
    valid_from: date | None = None
    valid_to: date | None = None
    version_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    matched_hs_code: str | None = None
    trade_type: TradeType
    required_document_code: str | None = None
    required_document_name: str | None = None
    agencies: list[str] = Field(default_factory=list)
    retrieval_methods: list[Literal["exact_hs", "dense", "bm25", "graph"]] = Field(
        default_factory=list
    )
    eligibility: Literal["finding", "supporting", "blocked"]
    score: float | None = Field(default=None, ge=0, le=1)
    raw_scores: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_evidence_metadata(self) -> "RagEvidence":
        if self.evidence_id != stable_evidence_id(self.source_record_id, self.version_hash):
            raise ValueError(
                "evidence_id must be SHA-256 of source_record_id and version_hash"
            )
        if self.matched_hs_code and not (
            self.matched_hs_code.isdigit() and len(self.matched_hs_code) in {6, 10}
        ):
            raise ValueError("matched_hs_code must be 6 or 10 digits")
        if len(self.agencies) != len(set(self.agencies)):
            raise ValueError("agencies must be unique")
        if len(self.retrieval_methods) != len(set(self.retrieval_methods)):
            raise ValueError("retrieval_methods must be unique")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be before valid_from")
        if self.eligibility == "finding" and not (
            self.required_document_code and self.required_document_name
        ):
            raise ValueError("finding evidence requires document code and name")
        if self.source_type == "customs_requirement" and self.eligibility == "finding":
            required = (
                self.law_id,
                self.law_name,
                self.valid_from,
                self.matched_hs_code,
                self.required_document_code,
                self.required_document_name,
            )
            if any(value is None for value in required) or not self.agencies:
                raise ValueError(
                    "finding customs requirement requires law, date, matched HS, "
                    "document and agency provenance"
                )
        return self


class RetrievalItemResult(WireModel):
    item_id: str = Field(min_length=1)
    hs_resolution: HsResolution
    resolved_hsk10: str | None = Field(default=None, pattern=r"^\d{10}$")
    candidate_hsk10s: list[str] = Field(default_factory=list)
    coverage: Coverage
    requirements_found: bool
    evidence: list[RagEvidence] = Field(default_factory=list)
    errors: list[RetrievalErrorDetail] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_is_item_bound(self) -> "RetrievalItemResult":
        if any(item.item_id != self.item_id for item in self.evidence):
            raise ValueError("every evidence item_id must match its retrieval result item_id")
        if any(not (len(code) == 10 and code.isdigit()) for code in self.candidate_hsk10s):
            raise ValueError("candidate_hsk10s must contain only 10-digit codes")
        if len(self.candidate_hsk10s) != len(set(self.candidate_hsk10s)):
            raise ValueError("candidate_hsk10s must be unique")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within an item result")
        if self.hs_resolution in {"exact", "unique_descendant"}:
            if self.resolved_hsk10 is None:
                raise ValueError(f"resolved_hsk10 is required for {self.hs_resolution}")
            if self.candidate_hsk10s and self.resolved_hsk10 not in self.candidate_hsk10s:
                raise ValueError("candidate_hsk10s must contain resolved_hsk10")
        elif self.hs_resolution == "ambiguous":
            if self.resolved_hsk10 is not None or len(self.candidate_hsk10s) < 2:
                raise ValueError(
                    "ambiguous resolution requires no resolved code and at least two candidates"
                )
        elif self.resolved_hsk10 is not None or self.candidate_hsk10s:
            raise ValueError(
                f"{self.hs_resolution} resolution cannot contain resolved or candidate codes"
            )
        finding_requirements = [
            item
            for item in self.evidence
            if item.source_type == "customs_requirement" and item.eligibility == "finding"
        ]
        if self.requirements_found != bool(finding_requirements):
            raise ValueError(
                "requirements_found must match finding-eligible customs requirement evidence"
            )
        return self


class EvidenceRetrievalResponse(WireModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    status: RetrievalStatus
    knowledge_snapshot_id: str | None = None
    results: list[RetrievalItemResult] = Field(default_factory=list)
    errors: list[RetrievalErrorDetail] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_results(self) -> "EvidenceRetrievalResponse":
        item_ids = [result.item_id for result in self.results]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("retrieval response result item_id values must be unique")
        has_result_problem = any(
            result.coverage != "complete" or bool(result.errors) for result in self.results
        )
        if self.status == "ok" and (
            not self.knowledge_snapshot_id or self.errors or has_result_problem
        ):
            raise ValueError("ok response requires a snapshot and complete, error-free results")
        if self.status == "partial" and not (self.errors or has_result_problem):
            raise ValueError("partial response requires an incomplete or errored result")
        if self.status == "failed" and not (
            self.errors or any(result.errors for result in self.results)
        ):
            raise ValueError("failed response requires a response or result error")
        return self


class RegulatoryFinding(BaseModel):
    finding_code: str
    severity: Severity
    message: str
    item_id: str = Field(min_length=1)
    required_document_code: str = Field(min_length=1)
    required_document_name: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal[
        "retrieval_score", "structured_requirement_policy"
    ] = "structured_requirement_policy"


class AuditEvent(BaseModel):
    stage: str
    event: str
    message: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class XaiReport(BaseModel):
    case_id: str
    overall_status: ReviewStatus
    summary: dict[str, Any]
    document_summary: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    regulatory_findings: list[dict[str, Any]]
    rag_evidence: list[dict[str, Any]]
    disclaimer: str = (
        "This result is a pre-clearance review aid. Final legal decisions must "
        "be made by a responsible human reviewer."
    )


class ReviewRunResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str
    review_run_id: str
    review_as_of: date
    trade_type: TradeType
    jurisdiction: str
    status: ReviewStatus
    requires_human_review: bool
    retrieval_status: RetrievalStatus
    retrieval_coverage: Coverage
    knowledge_snapshot_id: str | None = None
    retrieval_request: EvidenceRetrievalRequest | None = None
    retrieval_results: list[RetrievalItemResult] = Field(default_factory=list)
    retrieval_errors: list[RetrievalErrorDetail] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    regulatory_findings: list[RegulatoryFinding] = Field(default_factory=list)
    rag_evidence: list[RagEvidence] = Field(default_factory=list)
    report: XaiReport
    audit_trail: list[AuditEvent] = Field(default_factory=list)
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    model_errors: list[ModelErrorDetail] = Field(default_factory=list)


class ChatAnswer(BaseModel):
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    grounded: bool
