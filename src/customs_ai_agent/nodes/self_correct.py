from __future__ import annotations

import copy
from typing import Any

from customs_ai_agent.nodes.common import audit


NUMERIC_FIELDS = {
    "gross_weight_kg",
    "net_weight_kg",
    "quantity",
    "total_amount",
    "amount",
    "unit_price",
    "hs_code",
}


def _correct_field(name: str, data: dict[str, Any]) -> bool:
    if name not in NUMERIC_FIELDS:
        return False
    value = data.get("value")
    if not isinstance(value, str) or "O" not in value.upper():
        return False
    corrected = value.replace("O", "0").replace("o", "0")
    if corrected == value:
        return False
    data["value"] = corrected
    data["confidence"] = max(float(data.get("confidence") or 0), 0.93)
    # Evidence describes what was observed in the source and is immutable.
    # The corrected value is an interpretation, not a replacement source quote.
    data["value_source"] = "deterministic_ocr_correction"
    return True


def _has_low_confidence(state: dict[str, Any]) -> bool:
    for doc in (state.get("extracted_documents") or {}).values():
        for name, field_data in (doc.get("fields") or {}).items():
            value = field_data.get("value")
            if (
                name in NUMERIC_FIELDS
                and float(field_data.get("confidence") or 0) < 0.75
                and isinstance(value, str)
                and "O" in value.upper()
            ):
                return True
        for item in doc.get("line_items") or []:
            for name, field_data in item.items():
                if name == "item_id":
                    continue
                value = field_data.get("value")
                if (
                    name in NUMERIC_FIELDS
                    and float(field_data.get("confidence") or 0) < 0.75
                    and isinstance(value, str)
                    and "O" in value.upper()
                ):
                    return True
    return False


def has_low_confidence_extraction(state: dict[str, Any]) -> bool:
    return _has_low_confidence(state)


def self_correct_extraction(state: dict[str, Any]) -> dict[str, Any]:
    retry_count = int(state.get("retry_count") or 0)
    extracted = copy.deepcopy(state.get("extracted_documents") or {})
    changed = 0

    if retry_count < 2:
        for doc in extracted.values():
            for name, field_data in (doc.get("fields") or {}).items():
                if _correct_field(name, field_data):
                    changed += 1
            for item in doc.get("line_items") or []:
                for name, field_data in item.items():
                    if name != "item_id" and _correct_field(name, field_data):
                        changed += 1
            confidences = [value["confidence"] for value in doc.get("fields", {}).values()]
            for item in doc.get("line_items") or []:
                confidences.extend(value["confidence"] for key, value in item.items() if key != "item_id")
            doc["extraction_confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "extracted_documents": extracted,
        "retry_count": retry_count + 1,
        "current_stage": "self_correct_extraction",
        "requires_human_review": state.get("requires_human_review", False) or changed > 0,
        "audit_trail": audit(
            state,
            "self_correct_extraction",
            "completed",
            "Ran targeted extraction self-correction.",
            {"changed_fields": changed, "retry_count": retry_count + 1},
        ),
    }
