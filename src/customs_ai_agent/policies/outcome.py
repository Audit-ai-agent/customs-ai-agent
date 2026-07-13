from __future__ import annotations

from typing import Any

from customs_ai_agent.schemas.models import ReviewStatus


def retrieval_completion_blockers(state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if state.get("retrieval_status") != "ok":
        blockers.append("retrieval_status_not_ok")
    if state.get("retrieval_coverage") != "complete":
        blockers.append("retrieval_coverage_not_complete")
    if not state.get("knowledge_snapshot_id"):
        blockers.append("knowledge_snapshot_missing")
    if state.get("retrieval_errors"):
        blockers.append("retrieval_errors_present")

    results = state.get("retrieval_results") or []
    invoice_item_ids = {
        item.get("item_id")
        for doc in (state.get("normalized_documents") or {}).values()
        if doc.get("document_type") == "invoice"
        for item in doc.get("line_items") or []
        if item.get("item_id")
    }
    result_item_ids = {result.get("item_id") for result in results if result.get("item_id")}
    if not invoice_item_ids or invoice_item_ids != result_item_ids:
        blockers.append("item_results_incomplete")

    for result in results:
        item_id = result.get("item_id") or "unknown"
        if result.get("hs_resolution") not in {"exact", "unique_descendant"}:
            blockers.append(f"{item_id}:hs_not_resolved")
        if result.get("coverage") != "complete":
            blockers.append(f"{item_id}:coverage_not_complete")
        if result.get("errors"):
            blockers.append(f"{item_id}:errors_present")

        evidence = result.get("evidence") or []
        if any(item.get("item_id") != item_id for item in evidence):
            blockers.append(f"{item_id}:evidence_item_mismatch")
        if result.get("requirements_found"):
            usable_requirements = [
                item
                for item in evidence
                if item.get("eligibility") == "finding"
                and item.get("required_document_code")
                and item.get("required_document_name")
                and item.get("evidence_id")
                and item.get("source_record_id")
                and item.get("source_uri")
                and item.get("version_hash")
            ]
            if not usable_requirements:
                blockers.append(f"{item_id}:requirements_not_evidence_verified")

    return sorted(set(blockers))


def determine_outcome(state: dict[str, Any]) -> tuple[ReviewStatus, bool, list[str]]:
    issues = state.get("validation_issues") or []
    findings = state.get("regulatory_findings") or []
    reviewed_items = [*issues, *findings]
    blockers = retrieval_completion_blockers(state)

    if any(item.get("severity") == "critical" for item in reviewed_items):
        return "blocked", True, [*blockers, "critical_finding"]

    confidences = [
        float(doc.get("extraction_confidence") or 0)
        for doc in (state.get("extracted_documents") or {}).values()
    ]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    if state.get("requires_human_review"):
        blockers.append("human_review_flag")
    if any(item.get("severity") in {"medium", "high"} for item in reviewed_items):
        blockers.append("material_finding")
    if int(state.get("retry_count") or 0) > 0:
        blockers.append("self_correction_performed")
    if average_confidence < 0.90:
        blockers.append("extraction_confidence_below_threshold")

    blockers = sorted(set(blockers))
    if blockers:
        return "needs_review", True, blockers
    return "completed", False, []
