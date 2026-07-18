from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from customs_ai_agent.nodes.common import audit, normalize_text
from customs_ai_agent.nodes.normalize import is_valid_hs_code
from customs_ai_agent.policies.severity import SEVERITY_POLICY
from customs_ai_agent.policies.case_profile import required_document_types
from customs_ai_agent.schemas.models import BBox, ValidationIssue


def _unique_doc(
    state: dict[str, Any],
    doc_type: str,
) -> tuple[str | None, dict[str, Any] | None, list[str]]:
    matches = [
        (doc_id, doc)
        for doc_id, doc in (state.get("normalized_documents") or {}).items()
        if doc.get("document_type") == doc_type
    ]
    if len(matches) == 1:
        return matches[0][0], matches[0][1], []
    if len(matches) > 1:
        return None, None, sorted(doc_id for doc_id, _ in matches)
    return None, None, []


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


def _field(doc: dict[str, Any] | None, *names: str) -> Any:
    if not doc:
        return None
    fields = doc.get("fields") or {}
    for name in names:
        value = fields.get(name)
        if value is not None and value != "" and value != []:
            return value
    return None


def _raw_field(doc: dict[str, Any] | None, name: str) -> Any:
    if not doc:
        return None
    return (doc.get("raw_fields") or {}).get(name)


def _display_quantum(value: Any) -> Decimal | None:
    """Return the smallest displayed decimal unit without inventing tolerance."""

    if value is None:
        return None
    matches = list(re.finditer(r"[-+]?\d[\d,]*(?:\.(\d+))?", str(value)))
    if not matches:
        return None
    decimals = len(matches[-1].group(1) or "")
    return Decimal("1").scaleb(-decimals)


def _rounding_tolerance(*raw_values: Any) -> Decimal:
    return sum(
        (
            quantum / Decimal("2")
            for raw_value in raw_values
            if (quantum := _display_quantum(raw_value)) is not None
        ),
        Decimal("0"),
    )


def _normalized_collection(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return tuple(sorted(normalized for item in values if (normalized := normalize_text(item))))


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
        return _normalized_collection(left) == _normalized_collection(right)
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        try:
            return Decimal(str(left)) == Decimal(str(right))
        except Exception:
            return False
    return normalize_text(left) == normalize_text(right)


def _package_signature(value: Any) -> tuple[int, str] | None:
    if value is None:
        return None
    matches = re.findall(
        r"(?i)(\d[\d,]*)\s*(CARTONS?|CTNS?|PACKAGES?|PKGS?|PALLETS?|BOXES?|CASES?)\b",
        str(value),
    )
    if not matches:
        return None
    count, unit = matches[-1]
    normalized_unit = {
        "CARTON": "CARTON",
        "CARTONS": "CARTON",
        "CTN": "CARTON",
        "CTNS": "CARTON",
        "PACKAGE": "PACKAGE",
        "PACKAGES": "PACKAGE",
        "PKG": "PACKAGE",
        "PKGS": "PACKAGE",
        "PALLET": "PALLET",
        "PALLETS": "PALLET",
        "BOX": "BOX",
        "BOXES": "BOX",
        "CASE": "CASE",
        "CASES": "CASE",
    }[unit.upper()]
    return int(count.replace(",", "")), normalized_unit


def _same_packages(left: Any, right: Any) -> bool:
    left_signature = _package_signature(left)
    right_signature = _package_signature(right)
    if left_signature is not None and right_signature is not None:
        return left_signature == right_signature
    return _same_value(left, right)


def _split_vessel_voyage(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if " / " not in text:
        return text or None, None
    vessel, voyage = text.rsplit(" / ", 1)
    return vessel.strip() or None, voyage.strip() or None


def _single_invoice_origin(invoice: dict[str, Any] | None) -> Any:
    direct = _field(invoice, "country_of_origin", "origin_country")
    if direct:
        return direct
    origins = {
        item.get("origin_country")
        for item in (invoice or {}).get("line_items") or []
        if item.get("origin_country")
    }
    return next(iter(origins)) if len(origins) == 1 else None


def _compare_document_fields(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    source_id: str | None,
    source_doc: dict[str, Any] | None,
    source_name: str,
    target_id: str | None,
    target_doc: dict[str, Any] | None,
    target_name: str,
    source_value: Any | None = None,
    target_value: Any | None = None,
    comparator: Any = _same_value,
) -> None:
    left = _field(source_doc, source_name) if source_value is None else source_value
    right = _field(target_doc, target_name) if target_value is None else target_value
    if left is None or right is None:
        missing = [
            label
            for label, value in (
                (f"{source_id or 'source'}:{source_name}", left),
                (f"{target_id or 'target'}:{target_name}", right),
            )
            if value is None
        ]
        issues.append(
            _issue(
                "VALIDATION_INPUT_MISSING",
                "A required cross-document comparison could not be evaluated.",
                f"{source_name} and {target_name}",
                ", ".join(missing),
                source_id,
                target_id,
                _bbox(source_doc, source_name),
                _bbox(target_doc, target_name),
            )
        )
        return
    if comparator(left, right):
        return
    issues.append(
        _issue(
            code,
            message,
            left,
            right,
            source_id,
            target_id,
            _bbox(source_doc, source_name),
            _bbox(target_doc, target_name),
        )
    )


def _freight_flag(value: Any) -> bool:
    return normalize_text(value) in {
        "1",
        "true",
        "x",
        "y",
        "yes",
        "collect",
        "prepaid",
    }


def _has_medium_plus(issues: list[dict[str, Any]]) -> bool:
    return any(issue["severity"] in {"medium", "high", "critical"} for issue in issues)


def _cap_llm_issue_confidence(
    state: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    if state.get("llm_mode") != "llm":
        return
    documents = state.get("normalized_documents") or {}
    all_confidences = [
        float(document.get("extraction_confidence") or 0.0) for document in documents.values()
    ]
    for issue in issues:
        relevant_ids = {
            issue.get("source_document_id"),
            issue.get("target_document_id"),
        }
        relevant_confidences = [
            float(documents[document_id].get("extraction_confidence") or 0.0)
            for document_id in relevant_ids
            if document_id in documents
        ]
        source_confidence = min(relevant_confidences or all_confidences or [0.0])
        issue["confidence"] = min(float(issue["confidence"]), source_confidence)
        issue["confidence_basis"] = "source_field_minimum"


def validate_cross_documents(state: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    selected = {
        document_type: _unique_doc(state, document_type)
        for document_type in (
            "bill_of_lading",
            "invoice",
            "packing_list",
            "certificate_of_origin",
        )
    }
    for document_type, (_, _, conflicting_ids) in selected.items():
        if conflicting_ids:
            issues.append(
                _issue(
                    "DUPLICATE_DOCUMENT_TYPE",
                    "Multiple documents of one type make the case set ambiguous.",
                    f"at most one {document_type}",
                    ", ".join(conflicting_ids),
                    conflicting_ids[0],
                    conflicting_ids[1],
                )
            )
    for document_type in required_document_types(str(state.get("trade_type") or "import")):
        document_id, document, conflicting_ids = selected[document_type]
        if document is None and not conflicting_ids:
            issues.append(
                _issue(
                    "REQUIRED_DOCUMENT_MISSING",
                    "A document required by the active case profile is missing.",
                    document_type,
                    None,
                    document_id,
                    document_id,
                )
            )
    bl_id, bl, _ = selected["bill_of_lading"]
    invoice_id, invoice, _ = selected["invoice"]
    packing_id, packing, _ = selected["packing_list"]
    co_id, co, _ = selected["certificate_of_origin"]

    if invoice:
        total = invoice.get("fields", {}).get("total_amount")
        line_items = invoice.get("line_items") or []
        amounts = [item.get("amount") for item in line_items if item.get("amount") is not None]
        for item in line_items:
            quantity = item.get("quantity")
            unit_price = item.get("unit_price")
            amount = item.get("amount")
            if quantity is None or unit_price is None or amount is None:
                issues.append(
                    _issue(
                        "VALIDATION_INPUT_MISSING",
                        "Invoice line arithmetic requires quantity, unit price and amount.",
                        "quantity, unit_price and amount",
                        item.get("item_id"),
                        invoice_id,
                        invoice_id,
                    )
                )
                continue
            raw_fields = item.get("raw_fields") or {}
            unit_quantum = _display_quantum(raw_fields.get("unit_price"))
            amount_tolerance = _rounding_tolerance(raw_fields.get("amount"))
            tolerance = amount_tolerance
            if unit_quantum is not None:
                tolerance += abs(quantity) * unit_quantum / Decimal("2")
            calculated = quantity * unit_price
            if abs(calculated - amount) > tolerance:
                issues.append(
                    _issue(
                        "INV_LINE_AMOUNT_MISMATCH",
                        "Invoice line amount falls outside the interval implied by "
                        "displayed quantity and unit-price precision.",
                        f"{calculated} ± {tolerance}",
                        amount,
                        invoice_id,
                        invoice_id,
                    )
                )
        if total is None or not amounts:
            issues.append(
                _issue(
                    "VALIDATION_INPUT_MISSING",
                    "Invoice total reconciliation requires a total and line amounts.",
                    "total_amount and at least one line amount",
                    None,
                    invoice_id,
                    invoice_id,
                )
            )
        else:
            calculated = sum(amounts, Decimal("0"))
            tolerance = _rounding_tolerance(
                _raw_field(invoice, "total_amount"),
                *[
                    (item.get("raw_fields") or {}).get("amount")
                    for item in line_items
                    if item.get("amount") is not None
                ],
            )
            if abs(calculated - total) > tolerance:
                issues.append(
                    _issue(
                        "INV_TOTAL_MISMATCH",
                        "Invoice line item sum falls outside the interval implied by "
                        "displayed amount precision.",
                        f"{calculated} ± {tolerance}",
                        total,
                        invoice_id,
                        invoice_id,
                        _bbox(invoice, "total_amount"),
                        _bbox(invoice, "total_amount"),
                    )
                )

    if invoice and packing:
        for code, message, invoice_field, packing_field in (
            (
                "INVOICE_NO_MISMATCH",
                "Invoice number differs between the invoice and packing list.",
                "invoice_no",
                "invoice_no",
            ),
            (
                "INVOICE_DATE_MISMATCH",
                "Invoice date differs between the invoice and packing list.",
                "invoice_date",
                "invoice_date",
            ),
            (
                "SELLER_MISMATCH",
                "Seller differs between the invoice and packing list.",
                "seller",
                "seller",
            ),
            (
                "DEPARTURE_DATE_MISMATCH",
                "Departure date differs between the invoice and packing list.",
                "departure_date",
                "departure_date",
            ),
            (
                "PORT_OF_LOADING_MISMATCH",
                "Origin port differs between the invoice and packing list.",
                "port_of_loading",
                "port_of_loading",
            ),
            (
                "PORT_OF_DISCHARGE_MISMATCH",
                "Destination port differs between the invoice and packing list.",
                "port_of_discharge",
                "port_of_discharge",
            ),
            (
                "SHIPPING_MARKS_MISMATCH",
                "Shipping marks differ between the invoice and packing list.",
                "shipping_marks",
                "shipping_marks",
            ),
        ):
            _compare_document_fields(
                issues,
                code=code,
                message=message,
                source_id=invoice_id,
                source_doc=invoice,
                source_name=invoice_field,
                target_id=packing_id,
                target_doc=packing,
                target_name=packing_field,
            )
        _compare_document_fields(
            issues,
            code="CONSIGNEE_MISMATCH",
            message="Consignee differs between the invoice and packing list.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="consignee",
            target_id=packing_id,
            target_doc=packing,
            target_name="consignee",
            source_value=_field(invoice, "consignee", "buyer"),
            target_value=_field(packing, "consignee", "buyer"),
        )
        _compare_document_fields(
            issues,
            code="COUNTRY_OF_ORIGIN_MISMATCH",
            message="Country of origin differs between the invoice and packing list.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="country_of_origin",
            target_id=packing_id,
            target_doc=packing,
            target_name="country_of_origin",
            source_value=_single_invoice_origin(invoice),
        )
        _compare_document_fields(
            issues,
            code="PACKAGE_MISMATCH",
            message="Package count or kind differs between the invoice and packing list.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="packages",
            target_id=packing_id,
            target_doc=packing,
            target_name="packages",
            comparator=_same_packages,
        )
        invoice_vessel, invoice_voyage = _split_vessel_voyage(_field(invoice, "vessel"))
        packing_vessel, packing_voyage = _split_vessel_voyage(_field(packing, "vessel"))
        _compare_document_fields(
            issues,
            code="VESSEL_MISMATCH",
            message="Vessel differs between the invoice and packing list.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="vessel",
            target_id=packing_id,
            target_doc=packing,
            target_name="vessel",
            source_value=invoice_vessel,
            target_value=packing_vessel,
        )
        _compare_document_fields(
            issues,
            code="VOYAGE_NO_MISMATCH",
            message="Voyage number differs between the invoice and packing list.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="voyage_no",
            target_id=packing_id,
            target_doc=packing,
            target_name="voyage_no",
            source_value=_field(invoice, "voyage_no") or invoice_voyage,
            target_value=_field(packing, "voyage_no") or packing_voyage,
        )

    if bl and packing:
        bl_no = bl.get("fields", {}).get("bl_no")
        packing_bl_no = packing.get("fields", {}).get("bl_no")
        if (
            bl_no is not None
            and packing_bl_no is not None
            and normalize_text(bl_no) != normalize_text(packing_bl_no)
        ):
            issues.append(
                _issue(
                    "BL_NO_MISMATCH",
                    "B/L number differs between the bill of lading and packing list.",
                    bl_no,
                    packing_bl_no,
                    bl_id,
                    packing_id,
                    _bbox(bl, "bl_no"),
                    _bbox(packing, "bl_no"),
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
        packing_total_weight = packing.get("fields", {}).get("total_gross_weight_kg")
        packing_weights = [
            item.get("gross_weight_kg")
            for item in packing.get("line_items") or []
            if item.get("gross_weight_kg") is not None
        ]
        if bl_weight is None or (packing_total_weight is None and not packing_weights):
            issues.append(
                _issue(
                    "VALIDATION_INPUT_MISSING",
                    "B/L and packing-list gross weights are required for reconciliation.",
                    "gross_weight_kg and total_gross_weight_kg or line gross weights",
                    None,
                    bl_id,
                    packing_id,
                )
            )
        else:
            packing_weight = (
                packing_total_weight
                if packing_total_weight is not None
                else sum(packing_weights, Decimal("0"))
            )
            weight_tolerance = _rounding_tolerance(
                _raw_field(bl, "gross_weight_kg"),
                _raw_field(packing, "total_gross_weight_kg"),
            )
            if abs(bl_weight - packing_weight) > weight_tolerance:
                issues.append(
                    _issue(
                        "GROSS_WEIGHT_MISMATCH",
                        "B/L gross weight and packing list gross weight do not match.",
                        f"{bl_weight} ± {weight_tolerance}",
                        packing_weight,
                        bl_id,
                        packing_id,
                        _bbox(bl, "gross_weight_kg"),
                        None,
                    )
                )

        _compare_document_fields(
            issues,
            code="CONTAINER_NO_MISMATCH",
            message="B/L and packing list container numbers do not match.",
            source_id=bl_id,
            source_doc=bl,
            source_name="container_numbers",
            target_id=packing_id,
            target_doc=packing,
            target_name="container_numbers",
        )
        _compare_document_fields(
            issues,
            code="SEAL_NO_MISMATCH",
            message="B/L and packing list seal numbers do not match.",
            source_id=bl_id,
            source_doc=bl,
            source_name="seal_numbers",
            target_id=packing_id,
            target_doc=packing,
            target_name="seal_numbers",
        )

        for code, message, bl_field, packing_field in (
            (
                "PORT_OF_LOADING_MISMATCH",
                "Port of loading differs between the B/L and packing list.",
                "port_of_loading",
                "port_of_loading",
            ),
            (
                "PORT_OF_DISCHARGE_MISMATCH",
                "Port of discharge differs between the B/L and packing list.",
                "port_of_discharge",
                "port_of_discharge",
            ),
        ):
            _compare_document_fields(
                issues,
                code=code,
                message=message,
                source_id=bl_id,
                source_doc=bl,
                source_name=bl_field,
                target_id=packing_id,
                target_doc=packing,
                target_name=packing_field,
            )
        bl_vessel, bl_embedded_voyage = _split_vessel_voyage(_field(bl, "vessel"))
        packing_vessel, packing_embedded_voyage = _split_vessel_voyage(_field(packing, "vessel"))
        _compare_document_fields(
            issues,
            code="VESSEL_MISMATCH",
            message="Vessel differs between the B/L and packing list.",
            source_id=bl_id,
            source_doc=bl,
            source_name="vessel",
            target_id=packing_id,
            target_doc=packing,
            target_name="vessel",
            source_value=bl_vessel,
            target_value=packing_vessel,
        )
        _compare_document_fields(
            issues,
            code="VOYAGE_NO_MISMATCH",
            message="Voyage number differs between the B/L and packing list.",
            source_id=bl_id,
            source_doc=bl,
            source_name="voyage_no",
            target_id=packing_id,
            target_doc=packing,
            target_name="voyage_no",
            source_value=_field(bl, "voyage_no") or bl_embedded_voyage,
            target_value=_field(packing, "voyage_no") or packing_embedded_voyage,
        )
        _compare_document_fields(
            issues,
            code="SHIPPER_CONSIGNEE_MISMATCH",
            message="Shipper differs between the B/L and packing list seller.",
            source_id=bl_id,
            source_doc=bl,
            source_name="shipper",
            target_id=packing_id,
            target_doc=packing,
            target_name="seller",
        )
        _compare_document_fields(
            issues,
            code="CONSIGNEE_MISMATCH",
            message="Consignee differs between the B/L and packing list.",
            source_id=bl_id,
            source_doc=bl,
            source_name="consignee",
            target_id=packing_id,
            target_doc=packing,
            target_name="consignee",
            target_value=_field(packing, "consignee", "buyer"),
        )
        _compare_document_fields(
            issues,
            code="PACKAGE_MISMATCH",
            message="Package count or kind differs between the B/L and packing list.",
            source_id=bl_id,
            source_doc=bl,
            source_name="packages",
            target_id=packing_id,
            target_doc=packing,
            target_name="packages",
            comparator=_same_packages,
        )
        bl_measurement = _field(bl, "measurement_cbm")
        packing_measurement = _field(packing, "total_measurement_cbm")
        if bl_measurement is None or packing_measurement is None:
            issues.append(
                _issue(
                    "VALIDATION_INPUT_MISSING",
                    "B/L and packing-list measurements are required for reconciliation.",
                    "measurement_cbm and total_measurement_cbm",
                    None,
                    bl_id,
                    packing_id,
                )
            )
        else:
            measurement_tolerance = _rounding_tolerance(
                _raw_field(bl, "measurement_cbm"),
                _raw_field(packing, "total_measurement_cbm"),
            )
            if abs(bl_measurement - packing_measurement) > measurement_tolerance:
                issues.append(
                    _issue(
                        "MEASUREMENT_MISMATCH",
                        "B/L and packing-list measurements differ outside displayed precision.",
                        f"{bl_measurement} ± {measurement_tolerance}",
                        packing_measurement,
                        bl_id,
                        packing_id,
                        _bbox(bl, "measurement_cbm"),
                        _bbox(packing, "total_measurement_cbm"),
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
            (
                "shipper",
                "seller",
                "SELLER_MISMATCH",
                "B/L shipper differs from the invoice seller.",
            ),
            (
                "consignee",
                "buyer",
                "CONSIGNEE_MISMATCH",
                "B/L consignee differs from the invoice buyer.",
            ),
        ]
        for left, right, code, message in checks:
            _compare_document_fields(
                issues,
                code=code,
                message=message,
                source_id=bl_id,
                source_doc=bl,
                source_name=left,
                target_id=invoice_id,
                target_doc=invoice,
                target_name=right,
            )

    if bl:
        fields = bl.get("fields") or {}
        prepaid_present = "freight_prepaid" in fields
        collect_present = "freight_collect" in fields
        freight_terms = _field(bl, "freight_terms")
        # A B/L must express exactly one freight-payment term.  Absence is not a
        # clean state: the final plan explicitly treats both "both present" and
        # "neither present" as controlled defects.
        invalid_freight = True
        if prepaid_present or collect_present:
            invalid_freight = (
                int(_freight_flag(fields.get("freight_prepaid")))
                + int(_freight_flag(fields.get("freight_collect")))
                != 1
            )
        elif freight_terms:
            normalized_terms = normalize_text(freight_terms)
            has_prepaid = normalized_terms is not None and "prepaid" in normalized_terms
            has_collect = normalized_terms is not None and "collect" in normalized_terms
            invalid_freight = has_prepaid == has_collect
        if invalid_freight:
            issues.append(
                _issue(
                    "FREIGHT_TERMS_INVALID",
                    "Freight prepaid and collect indicators must be mutually exclusive.",
                    "exactly one of prepaid or collect",
                    {
                        "prepaid": fields.get("freight_prepaid"),
                        "collect": fields.get("freight_collect"),
                        "terms": freight_terms,
                    },
                    bl_id,
                    bl_id,
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
        if not invoice_origins or not co_origin:
            issues.append(
                _issue(
                    "VALIDATION_INPUT_MISSING",
                    "C/O origin validation requires invoice and certificate origin values.",
                    "invoice line origins and certificate origin_country",
                    None,
                    invoice_id,
                    co_id,
                )
            )
        elif invoice_origins != {co_origin}:
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
        _compare_document_fields(
            issues,
            code="EXPORTER_MISMATCH",
            message="Certificate exporter differs from the invoice seller.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="seller",
            target_id=co_id,
            target_doc=co,
            target_name="exporter",
        )
        _compare_document_fields(
            issues,
            code="IMPORTER_MISMATCH",
            message="Certificate importer differs from the invoice buyer or consignee.",
            source_id=invoice_id,
            source_doc=invoice,
            source_name="buyer",
            target_id=co_id,
            target_doc=co,
            target_name="importer",
            source_value=_field(invoice, "buyer", "consignee"),
        )
        invoice_hs_codes = {
            item.get("hs_code") for item in invoice.get("line_items") or [] if item.get("hs_code")
        }
        co_hs_codes = set(_field(co, "hs_codes") or [])
        if not invoice_hs_codes or not co_hs_codes:
            issues.append(
                _issue(
                    "VALIDATION_INPUT_MISSING",
                    "C/O HS validation requires invoice and certificate HS codes.",
                    "invoice line hs_code and certificate hs_codes",
                    None,
                    invoice_id,
                    co_id,
                )
            )
        elif invoice_hs_codes != co_hs_codes:
            issues.append(
                _issue(
                    "HS_CODE_MISMATCH",
                    "Certificate and invoice HS code sets do not match.",
                    sorted(invoice_hs_codes),
                    sorted(co_hs_codes),
                    invoice_id,
                    co_id,
                    None,
                    _bbox(co, "hs_codes"),
                )
            )

    _cap_llm_issue_confidence(state, issues)
    return {
        "validation_issues": issues,
        "current_stage": "validate_cross_documents",
        "requires_human_review": state.get("requires_human_review", False)
        or _has_medium_plus(issues),
        "audit_trail": audit(
            state,
            "validate_cross_documents",
            "completed",
            "Validated cross-document consistency.",
            {"issue_count": len(issues)},
        ),
    }
