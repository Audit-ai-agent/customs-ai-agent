from __future__ import annotations

from typing import Any

from customs_ai_agent.schemas.models import ReviewStatus


REQUIRED_DOCUMENT_FIELDS: dict[str, frozenset[str]] = {
    "bill_of_lading": frozenset(
        {"bl_no", "shipper", "consignee", "packages", "gross_weight_kg"}
    ),
    "invoice": frozenset(
        {"invoice_no", "seller", "buyer", "currency", "total_amount"}
    ),
    "packing_list": frozenset({"packing_no", "seller", "buyer", "packages"}),
    "certificate_of_origin": frozenset(
        {
            "certificate_no",
            "exporter",
            "importer",
            "origin_country",
            "hs_codes",
            "goods_description",
        }
    ),
}
REQUIRED_LINE_ITEM_FIELDS: dict[str, frozenset[str]] = {
    "invoice": frozenset({"description", "hs_code", "quantity", "amount"}),
    "packing_list": frozenset(
        {"description", "quantity", "gross_weight_kg", "package_count"}
    ),
}


def _has_value(field: dict[str, Any] | None) -> bool:
    if not field:
        return False
    value = field.get("value")
    return value is not None and value != "" and value != []


def extraction_completion_blockers(state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for document_id, document in (state.get("extracted_documents") or {}).items():
        document_type = str(document.get("document_type") or "unknown")
        if document_type == "unknown":
            blockers.append(f"{document_id}:document_type_unknown")
            continue
        fields = document.get("fields") or {}
        for name in REQUIRED_DOCUMENT_FIELDS.get(document_type, frozenset()):
            if not _has_value(fields.get(name)):
                blockers.append(f"{document_id}:required_field_missing:{name}")
        required_line_fields = REQUIRED_LINE_ITEM_FIELDS.get(document_type)
        if required_line_fields is None:
            continue
        line_items = document.get("line_items") or []
        if not line_items:
            blockers.append(f"{document_id}:required_line_items_missing")
            continue
        for index, item in enumerate(line_items, start=1):
            for name in required_line_fields:
                if not _has_value(item.get(name)):
                    blockers.append(
                        f"{document_id}:line_{index}:required_field_missing:{name}"
                    )
    return sorted(set(blockers))


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
    blockers = [
        *retrieval_completion_blockers(state),
        *extraction_completion_blockers(state),
    ]

    has_critical_finding = any(
        item.get("severity") == "critical" for item in reviewed_items
    )
    llm_is_uncalibrated = state.get("llm_mode") == "llm"
    if has_critical_finding:
        blockers.append("critical_finding")
        # Until a separate HITL approval contract exists, a Claude-derived
        # field binding may request review but cannot autonomously block a case.
        if not llm_is_uncalibrated:
            return "blocked", True, sorted(set(blockers))

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
    if llm_is_uncalibrated:
        blockers.append("llm_extraction_uncalibrated")
    if average_confidence < 0.90:
        blockers.append("extraction_confidence_below_threshold")

    blockers = sorted(set(blockers))
    if blockers:
        return "needs_review", True, blockers
    return "completed", False, []
