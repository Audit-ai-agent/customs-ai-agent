from __future__ import annotations

from decimal import Decimal
from typing import Any

from customs_ai_agent.nodes.common import audit, normalize_text
from customs_ai_agent.nodes.normalize import is_valid_hs_code
from customs_ai_agent.policies.severity import SEVERITY_POLICY
from customs_ai_agent.schemas.models import BBox, ValidationIssue


def _first_doc(state: dict[str, Any], doc_type: str) -> tuple[str | None, dict[str, Any] | None]:
    for doc_id, doc in (state.get("normalized_documents") or {}).items():
        if doc.get("document_type") == doc_type:
            return doc_id, doc
    return None, None


def _bbox(doc: dict[str, Any] | None, field_name: str) -> BBox | None:
    if not doc:
        return None
    evidence = (doc.get("field_evidence") or {}).get(field_name) or {}
    value = evidence.get("bbox")
    return BBox.model_validate(value) if value is not None else None


def _issue(
    code: str,
    message: str,
    expected: Any = None,
    observed: Any = None,
    source_document_id: str | None = None,
    target_document_id: str | None = None,
    source_bbox: BBox | None = None,
    target_bbox: BBox | None = None,
) -> dict[str, Any]:
    return ValidationIssue(
        issue_code=code,
        severity=SEVERITY_POLICY[code],
        message=message,
        expected_value=None if expected is None else str(expected),
        observed_value=None if observed is None else str(observed),
        source_document_id=source_document_id,
        target_document_id=target_document_id,
        source_bbox=source_bbox,
        target_bbox=target_bbox,
        confidence=0.98,
    ).model_dump(mode="json")


def _line_by_id(doc: dict[str, Any] | None, item_id: str | None) -> dict[str, Any] | None:
    if not doc or not item_id:
        return None
    for item in doc.get("line_items") or []:
        if item.get("item_id") == item_id:
            return item
    return None


def _has_medium_plus(issues: list[dict[str, Any]]) -> bool:
    return any(issue["severity"] in {"medium", "high", "critical"} for issue in issues)


def validate_cross_documents(state: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    bl_id, bl = _first_doc(state, "bill_of_lading")
    invoice_id, invoice = _first_doc(state, "invoice")
    packing_id, packing = _first_doc(state, "packing_list")
    co_id, co = _first_doc(state, "certificate_of_origin")

    if invoice:
        total = invoice.get("fields", {}).get("total_amount")
        amounts = [item.get("amount") for item in invoice.get("line_items") or [] if item.get("amount") is not None]
        if total is not None and amounts:
            calculated = sum(amounts, Decimal("0"))
            if abs(calculated - total) > Decimal("1.00"):
                issues.append(
                    _issue(
                        "INV_TOTAL_MISMATCH",
                        "Invoice line item sum does not match total amount.",
                        calculated,
                        total,
                        invoice_id,
                        invoice_id,
                        _bbox(invoice, "total_amount"),
                        _bbox(invoice, "total_amount"),
                    )
                )

    pairing = state.get("pairing_result") or {}
    for link in pairing.get("line_item_links") or []:
        inv_line = _line_by_id(invoice, link.get("invoice_item_id"))
        pack_line = _line_by_id(packing, link.get("packing_item_id"))
        if not inv_line or not pack_line or link.get("score", 0) < 0.70:
            issues.append(
                _issue(
                    "LINE_ITEM_UNMATCHED",
                    "Invoice and packing list line item matching failed.",
                    link.get("invoice_item_id"),
                    link.get("packing_item_id"),
                    invoice_id,
                    packing_id,
                )
            )
            continue
        if inv_line.get("quantity") != pack_line.get("quantity"):
            issues.append(
                _issue(
                    "QTY_MISMATCH_INVOICE_PACKING",
                    "Invoice quantity and packing list quantity do not match.",
                    inv_line.get("quantity"),
                    pack_line.get("quantity"),
                    invoice_id,
                    packing_id,
                )
            )

    if bl and packing:
        bl_weight = bl.get("fields", {}).get("gross_weight_kg")
        packing_weights = [
            item.get("gross_weight_kg")
            for item in packing.get("line_items") or []
            if item.get("gross_weight_kg") is not None
        ]
        if bl_weight is not None and packing_weights:
            packing_weight = sum(packing_weights, Decimal("0"))
            if abs(bl_weight - packing_weight) > Decimal("1.0"):
                issues.append(
                    _issue(
                        "GROSS_WEIGHT_MISMATCH",
                        "B/L gross weight and packing list gross weight do not match.",
                        bl_weight,
                        packing_weight,
                        bl_id,
                        packing_id,
                        _bbox(bl, "gross_weight_kg"),
                        None,
                    )
                )

        bl_containers = set(bl.get("fields", {}).get("container_numbers") or [])
        packing_containers = set(packing.get("fields", {}).get("container_numbers") or [])
        if bl_containers and packing_containers and bl_containers != packing_containers:
            issues.append(
                _issue(
                    "CONTAINER_NO_MISMATCH",
                    "B/L and packing list container numbers do not match.",
                    sorted(bl_containers),
                    sorted(packing_containers),
                    bl_id,
                    packing_id,
                    _bbox(bl, "container_numbers"),
                    _bbox(packing, "container_numbers"),
                )
            )

    if packing:
        for item in packing.get("line_items") or []:
            net = item.get("net_weight_kg")
            gross = item.get("gross_weight_kg")
            if net is not None and gross is not None and net > gross:
                issues.append(
                    _issue(
                        "NET_GROSS_WEIGHT_INVALID",
                        "Net weight is greater than gross weight.",
                        f"net <= {gross}",
                        net,
                        packing_id,
                        packing_id,
                    )
                )

    if bl and invoice:
        checks = [
            ("shipper", "seller"),
            ("consignee", "buyer"),
        ]
        for left, right in checks:
            left_value = normalize_text(bl.get("fields", {}).get(left))
            right_value = normalize_text(invoice.get("fields", {}).get(right))
            if left_value and right_value and left_value != right_value:
                issues.append(
                    _issue(
                        "SHIPPER_CONSIGNEE_MISMATCH",
                        "Party information differs across documents.",
                        bl.get("fields", {}).get(left),
                        invoice.get("fields", {}).get(right),
                        bl_id,
                        invoice_id,
                        _bbox(bl, left),
                        _bbox(invoice, right),
                    )
                )

    if invoice:
        for item in invoice.get("line_items") or []:
            hs_code = item.get("hs_code")
            if not hs_code:
                issues.append(
                    _issue(
                        "HS_CODE_MISSING",
                        "Invoice line item is missing an HS code.",
                        "HS code present",
                        None,
                        invoice_id,
                        invoice_id,
                    )
                )
            elif not is_valid_hs_code(hs_code):
                issues.append(
                    _issue(
                        "HS_CODE_FORMAT_INVALID",
                        "Invoice line item has an invalid HS code format.",
                        "6 or 10 digits",
                        hs_code,
                        invoice_id,
                        invoice_id,
                    )
                )

    if invoice and co:
        invoice_origins = {
            item.get("origin_country")
            for item in invoice.get("line_items") or []
            if item.get("origin_country")
        }
        co_origin = co.get("fields", {}).get("origin_country")
        if invoice_origins and co_origin and invoice_origins != {co_origin}:
            issues.append(
                _issue(
                    "ORIGIN_MISMATCH",
                    "Certificate of origin and invoice origin do not match.",
                    sorted(invoice_origins),
                    co_origin,
                    invoice_id,
                    co_id,
                    None,
                    _bbox(co, "origin_country"),
                )
            )

    return {
        "validation_issues": issues,
        "current_stage": "validate_cross_documents",
        "requires_human_review": state.get("requires_human_review", False) or _has_medium_plus(issues),
        "audit_trail": audit(
            state,
            "validate_cross_documents",
            "completed",
            "Validated cross-document consistency.",
            {"issue_count": len(issues)},
        ),
    }
