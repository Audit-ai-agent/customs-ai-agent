from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from customs_ai_agent.nodes.common import audit, parse_decimal


COUNTRIES = {
    "NO": "NO",
    "NORWAY": "NO",
    "VN": "VN",
    "VIETNAM": "VN",
    "CN": "CN",
    "CHINA": "CN",
    "KR": "KR",
    "KOREA": "KR",
    "SOUTH KOREA": "KR",
    "US": "US",
    "USA": "US",
    "UNITED STATES": "US",
}


def normalize_currency(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace("$", "")
    if text in {"US", "USD", "DOLLAR", "DOLLARS"}:
        return "USD"
    if text in {"KRW", "WON"}:
        return "KRW"
    return text or None


def normalize_country(value: Any) -> str | None:
    if value is None:
        return None
    return COUNTRIES.get(str(value).strip().upper(), str(value).strip().upper() or None)


def normalize_hs_code(value: Any) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    return digits


def is_valid_hs_code(value: str | None) -> bool:
    return bool(value and len(value) in {6, 10} and value.isdigit())


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    formats = ["%Y.%m.%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d %Y", "%B %d %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_weight_kg(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(",", "")
    if "O" in text:
        return None
    amount = parse_decimal(text)
    if amount is None:
        return None
    if re.search(r"\b(MT|METRIC TON|TONNE|TONNES)\b", text):
        return amount * Decimal("1000")
    return amount


def parse_quantity(value: Any) -> tuple[Decimal | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip().upper()
    amount = parse_decimal(text)
    unit_match = re.search(r"\b(KG|KGS|CTN|CTNS|PCS|EA|MT)\b", text)
    unit = unit_match.group(1) if unit_match else None
    if unit in {"KGS"}:
        unit = "KG"
    if unit == "MT" and amount is not None:
        amount *= Decimal("1000")
        unit = "KG"
    return amount, unit


def _value(field_data: dict[str, Any] | None) -> Any:
    return None if field_data is None else field_data.get("value")


def _evidence(field_data: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if field_data is None else field_data.get("evidence")


def _normalize_field(name: str, value: Any) -> Any:
    if name in {"gross_weight_kg"}:
        return parse_weight_kg(value)
    if name in {"total_amount"}:
        return parse_decimal(value)
    if name in {"currency"}:
        return normalize_currency(value)
    if name in {"invoice_date", "issue_date"}:
        return normalize_date(value)
    if name in {"origin_country"}:
        return normalize_country(value)
    if name in {"hs_codes"} and isinstance(value, list):
        return [normalize_hs_code(item) for item in value if normalize_hs_code(item)]
    return value


def _normalize_line_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {"item_id": item["item_id"], "field_evidence": {}}
    for key, value in item.items():
        if key == "item_id":
            continue
        raw_value = _value(value)
        normalized["field_evidence"][key] = _evidence(value)
        if key == "quantity":
            quantity, unit = parse_quantity(raw_value)
            normalized["quantity"] = quantity
            normalized["unit"] = unit
        elif key in {"unit_price", "amount"}:
            normalized[key] = parse_decimal(raw_value)
        elif key in {"net_weight_kg", "gross_weight_kg"}:
            normalized[key] = parse_weight_kg(raw_value)
        elif key == "package_count":
            amount = parse_decimal(raw_value)
            normalized[key] = int(amount) if amount is not None else None
        elif key == "hs_code":
            normalized[key] = normalize_hs_code(raw_value)
        elif key == "origin_country":
            normalized[key] = normalize_country(raw_value)
        else:
            normalized[key] = raw_value
    return normalized


def normalize_documents(state: dict[str, Any]) -> dict[str, Any]:
    normalized_docs: dict[str, dict[str, Any]] = {}

    for document_id, doc in (state.get("extracted_documents") or {}).items():
        fields: dict[str, Any] = {}
        field_evidence: dict[str, Any] = {}
        raw_fields: dict[str, Any] = {}
        for name, field_data in doc.get("fields", {}).items():
            raw_value = _value(field_data)
            raw_fields[name] = raw_value
            fields[name] = _normalize_field(name, raw_value)
            field_evidence[name] = _evidence(field_data)

        normalized_docs[document_id] = {
            "document_id": document_id,
            "filename": doc["filename"],
            "document_type": doc["document_type"],
            "fields": fields,
            "raw_fields": raw_fields,
            "field_evidence": field_evidence,
            "line_items": [_normalize_line_item(item) for item in doc.get("line_items", [])],
            "extraction_confidence": doc.get("extraction_confidence", 0.0),
        }

    return {
        "normalized_documents": normalized_docs,
        "current_stage": "normalize_documents",
        "audit_trail": audit(
            state,
            "normalize_documents",
            "completed",
            "Normalized extracted values.",
            {"document_count": len(normalized_docs)},
        ),
    }
