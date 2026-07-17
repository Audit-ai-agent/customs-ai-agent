from __future__ import annotations

from typing import Any, TypedDict


class ReviewState(TypedDict, total=False):
    case_id: str
    review_run_id: str
    review_as_of: str
    trade_type: str
    jurisdiction: str
    uploaded_requirement_document_codes: list[str]
    documents: list[dict[str, Any]]
    document_ids: list[str]
    classified_documents: dict[str, dict[str, Any]]
    model_analyses: dict[str, dict[str, Any]]
    model_calls: list[dict[str, Any]]
    model_errors: list[dict[str, Any]]
    extracted_documents: dict[str, dict[str, Any]]
    normalized_documents: dict[str, dict[str, Any]]
    pairing_result: dict[str, Any]
    validation_issues: list[dict[str, Any]]
    rag_evidence: list[dict[str, Any]]
    retrieval_request_id: str
    retrieval_request: dict[str, Any]
    retrieval_status: str
    retrieval_coverage: str
    knowledge_snapshot_id: str | None
    retrieval_results: list[dict[str, Any]]
    retrieval_errors: list[dict[str, Any]]
    regulatory_findings: list[dict[str, Any]]
    report_json: dict[str, Any] | None
    retry_count: int
    current_stage: str
    requires_human_review: bool
    final_status: str | None
    audit_trail: list[dict[str, Any]]
    llm_mode: str
