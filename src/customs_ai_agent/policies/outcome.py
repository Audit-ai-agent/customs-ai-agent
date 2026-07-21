from __future__ import annotations

from typing import Any

from customs_ai_agent.schemas.models import ReviewStatus
from customs_ai_agent.policies.case_profile import required_document_types


REQUIRED_DOCUMENT_FIELDS: dict[str, frozenset[str]] = {
    "bill_of_lading": frozenset(
        {
            "bl_no",
            "shipper",
            "consignee",
            "vessel",
            "voyage_no",
            "port_of_loading",
            "port_of_discharge",
            "container_numbers",
            "seal_numbers",
            "packages",
            "gross_weight_kg",
            "measurement_cbm",
        }
    ),
    "invoice": frozenset(
        {
            "invoice_no",
            "invoice_date",
            "seller",
            "buyer",
            "currency",
            "total_amount",
            "departure_date",
            "vessel",
            "voyage_no",
            "port_of_loading",
            "port_of_discharge",
            "country_of_origin",
            "shipping_marks",
            "packages",
        }
    ),
    "packing_list": frozenset(
        {
            "packing_no",
            "invoice_no",
            "invoice_date",
            "bl_no",
            "seller",
            "buyer",
            "departure_date",
            "vessel",
            "voyage_no",
            "port_of_loading",
            "port_of_discharge",
            "country_of_origin",
            "shipping_marks",
            "container_numbers",
            "seal_numbers",
            "packages",
            "total_gross_weight_kg",
            "total_measurement_cbm",
        }
    ),
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
    "invoice": frozenset(
        {
            "description",
            "hs_code",
            "quantity",
            "unit_price",
            "amount",
            "origin_country",
        }
    ),
    "packing_list": frozenset(
        {
            "description",
            "quantity",
            "net_weight_kg",
            "gross_weight_kg",
            "package_count",
        }
    ),
}


def _has_value(field: dict[str, Any] | None) -> bool:
    if not field:
        return False
    value = field.get("value")
    return value is not None and value != "" and value != []


def extraction_completion_blockers(state: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for conflict in state.get("declared_document_type_conflicts") or []:
        document_id = conflict.get("document_id") or "unknown"
        blockers.append(f"{document_id}:declared_document_type_mismatch")
    pairing = state.get("pairing_result") or {}
    if pairing.get("document_type_conflicts"):
        blockers.append("pairing_document_type_conflict")
    if pairing.get("missing_required_document_types"):
        blockers.append("pairing_required_document_missing")
    if pairing.get("line_item_pairing_ambiguities"):
        blockers.append("line_item_pairing_ambiguous")
    if pairing.get("unmatched_invoice_item_ids"):
        blockers.append("invoice_line_item_unmatched")
    if pairing and pairing.get("document_set_status") != "deterministic":
        blockers.append("document_set_not_deterministic")
    if pairing and pairing.get("line_item_pairing_status") not in {
        "deterministic",
        "deterministic_with_unscored_packing_extras",
    }:
        blockers.append("line_item_pairing_not_deterministic")
    present_document_types = {
        str(document.get("document_type") or "unknown")
        for document in (state.get("extracted_documents") or {}).values()
    }
    for required_type in required_document_types(str(state.get("trade_type") or "import")):
        if required_type not in present_document_types:
            blockers.append(f"required_document_missing:{required_type}")
    for document_id, document in (state.get("extracted_documents") or {}).items():
        if document.get("extraction_status") != "completed":
            blockers.append(f"{document_id}:extraction_failed")
            continue
        document_type = str(document.get("document_type") or "unknown")
        if document_type == "unknown":
            blockers.append(f"{document_id}:document_type_unknown")
            continue
        fields = document.get("fields") or {}
        for name in REQUIRED_DOCUMENT_FIELDS.get(document_type, frozenset()):
            if not _has_value(fields.get(name)):
                blockers.append(f"{document_id}:required_field_missing:{name}")
        if document_type == "bill_of_lading" and not any(
            _has_value(fields.get(name))
            for name in ("freight_prepaid", "freight_collect", "freight_terms")
        ):
            blockers.append(f"{document_id}:required_field_missing:freight_terms")
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
                    blockers.append(f"{document_id}:line_{index}:required_field_missing:{name}")
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


def determine_outcome(
    state: dict[str, Any],
    *,
    extraction_release_approved: bool = False,
) -> tuple[ReviewStatus, bool, list[str]]:
    issues = state.get("validation_issues") or []
    findings = state.get("regulatory_findings") or []
    reviewed_items = [*issues, *findings]
    blockers = [
        *retrieval_completion_blockers(state),
        *extraction_completion_blockers(state),
    ]

    has_critical_finding = any(item.get("severity") == "critical" for item in reviewed_items)
    llm_is_uncalibrated = state.get("llm_mode") == "llm" and not extraction_release_approved
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
    if average_confidence < 0.90 and not extraction_release_approved:
        blockers.append("extraction_confidence_below_threshold")

    blockers = sorted(set(blockers))
    if blockers:
        return "needs_review", True, blockers
    return "completed", False, []
