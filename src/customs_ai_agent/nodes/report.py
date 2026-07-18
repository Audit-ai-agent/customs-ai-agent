from __future__ import annotations

from typing import Any

from customs_ai_agent.nodes.common import audit
from customs_ai_agent.policies.outcome import determine_outcome
from customs_ai_agent.schemas.models import XaiReport


def _count_by_severity(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "critical": sum(1 for item in items if item.get("severity") == "critical"),
        "high": sum(1 for item in items if item.get("severity") == "high"),
        "medium": sum(1 for item in items if item.get("severity") == "medium"),
        "low": sum(1 for item in items if item.get("severity") == "low"),
    }


def _document_summary(state: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for doc in (state.get("extracted_documents") or {}).values():
        summary.append(
            {
                "document_id": doc["document_id"],
                "document_type": doc["document_type"],
                "extraction_status": doc.get("extraction_status", "failed"),
                "extraction_confidence": doc.get("extraction_confidence", 0.0),
            }
        )
    return summary


def _issue_view(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_code": issue["issue_code"],
        "severity": issue["severity"],
        "message": issue["message"],
        "expected_value": issue.get("expected_value"),
        "observed_value": issue.get("observed_value"),
        "source_document_id": issue.get("source_document_id"),
        "target_document_id": issue.get("target_document_id"),
        "source_bbox": issue.get("source_bbox"),
        "target_bbox": issue.get("target_bbox"),
        "confidence": issue.get("confidence"),
        "confidence_basis": issue.get("confidence_basis"),
        "needs_regulatory_review": issue.get("needs_regulatory_review", False),
        "recommendation": "Review the source document values and confirm the corrected value before clearance submission.",
    }


def generate_xai_report(state: dict[str, Any]) -> dict[str, Any]:
    validation_issues = state.get("validation_issues") or []
    findings = state.get("regulatory_findings") or []
    all_items = [*validation_issues, *findings]
    severities = _count_by_severity(all_items)
    doc_summary = _document_summary(state)
    avg_confidence = (
        sum(doc.get("extraction_confidence", 0.0) for doc in doc_summary) / len(doc_summary)
        if doc_summary
        else 0.0
    )
    overall_status, _, outcome_reasons = determine_outcome(state)
    report = XaiReport(
        case_id=state["case_id"],
        overall_status=overall_status,
        summary={
            "total_documents": len(doc_summary),
            "total_issues": len(all_items),
            **severities,
            "average_extraction_confidence": round(avg_confidence, 4),
            "retrieval_status": state.get("retrieval_status"),
            "retrieval_coverage": state.get("retrieval_coverage"),
            "knowledge_snapshot_id": state.get("knowledge_snapshot_id"),
            "outcome_reasons": outcome_reasons,
        },
        document_summary=doc_summary,
        issues=[_issue_view(issue) for issue in validation_issues],
        regulatory_findings=findings,
        rag_evidence=state.get("rag_evidence") or [],
    ).model_dump(mode="json")
    return {
        "report_json": report,
        "current_stage": "generate_xai_report",
        "audit_trail": audit(
            state,
            "generate_xai_report",
            "completed",
            "Generated XAI report JSON.",
            {"overall_status": report["overall_status"]},
        ),
    }
