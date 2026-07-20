from __future__ import annotations

import re
from datetime import date, datetime
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
    "REPUBLIC OF KOREA": "KR",
    "KOREA, REPUBLIC OF": "KR",
    "US": "US",
    "USA": "US",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
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
    text = str(value).strip()
    if not text:
        return None
    # Only formatting separators used by legitimate HS/HSK displays may be
    # removed.  Silently deleting letters or arbitrary punctuation can turn an
    # untrusted value such as ``ABC731815`` into a different, apparently valid
    # classification code and send the wrong scope to the Evidence Service.
    compact = re.sub(r"[ .-]", "", text)
    if not compact or re.fullmatch(r"[0-9]+", compact) is None:
        return None
    return compact


def is_valid_hs_code(value: str | None) -> bool:
    return bool(
        isinstance(value, str) and re.fullmatch(r"(?:[0-9]{6}|[0-9]{10})", value) is not None
    )


def is_ambiguous_numeric_date(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return len(_numeric_day_month_candidates(text)) > 1


def _numeric_day_month_candidates(text: str) -> set[date]:
    match = re.fullmatch(
        r"([0-9]{1,2})([./-])([0-9]{1,2})\2([0-9]{4})",
        text,
    )
    if match is None:
        return set()
    separator = match.group(2)
    candidates: set[date] = set()
    for fmt in (f"%m{separator}%d{separator}%Y", f"%d{separator}%m{separator}%Y"):
        try:
            candidates.add(datetime.strptime(text, fmt).date())
        except ValueError:
            continue
    return candidates


def normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    formats = ["%Y.%m.%d", "%Y-%m-%d", "%b %d %Y", "%B %d %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    numeric_candidates = _numeric_day_month_candidates(text)
    if len(numeric_candidates) == 1:
        return numeric_candidates.pop().isoformat()
    # Two distinct valid interpretations are deliberately not resolved using
    # an implicit locale.  Validation converts this None into a deterministic
    # DATE_FORMAT_AMBIGUOUS issue and routes the run to HITL.
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
    if name in {
        "gross_weight_kg",
        "net_weight_kg",
        "total_net_weight_kg",
        "total_gross_weight_kg",
    }:
        return parse_weight_kg(value)
    if name in {
        "total_amount",
        "measurement_cbm",
        "total_measurement_cbm",
    }:
        return parse_decimal(value)
    if name in {"currency"}:
        return normalize_currency(value)
    if name in {
        "departure_date",
        "invoice_date",
        "issue_date",
        "on_board_date",
    }:
        return normalize_date(value)
    if name in {"origin_country", "country_of_origin"}:
        return normalize_country(value)
    if name in {"hs_codes"} and isinstance(value, list):
        return [normalize_hs_code(item) for item in value if normalize_hs_code(item)]
    return value


def _normalize_line_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "item_id": item["item_id"],
        "field_evidence": {},
        "raw_fields": {},
    }
    for key, value in item.items():
        if key == "item_id":
            continue
        raw_value = _value(value)
        normalized["raw_fields"][key] = raw_value
        normalized["field_evidence"][key] = _evidence(value)
        if key == "quantity":
            quantity, unit = parse_quantity(raw_value)
            normalized["quantity"] = quantity
            normalized["unit"] = unit
        elif key in {"unit_price", "amount", "measurement_cbm"}:
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
