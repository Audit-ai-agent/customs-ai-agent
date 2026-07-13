from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from customs_ai_agent.schemas.models import AuditEvent, BBox, ExtractedField, FieldEvidence


def audit(
    state: dict[str, Any],
    stage: str,
    event: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events = list(state.get("audit_trail") or [])
    events.append(
        AuditEvent(
            stage=stage,
            event=event,
            message=message,
            metadata=metadata or {},
        ).model_dump(mode="json")
    )
    return events


def extract_label(text: str, *labels: str) -> str | None:
    for label in labels:
        pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$", re.I | re.M)
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            return value if value else None
    return None


def extract_list_label(text: str, *labels: str) -> list[str]:
    value = extract_label(text, *labels)
    if not value:
        return []
    parts = re.split(r"[,;/]+", value)
    return [part.strip() for part in parts if part.strip()]


def make_bbox(index: int) -> BBox:
    y1 = 40.0 + index * 24.0
    return BBox(page=1, x1=40.0, y1=y1, x2=300.0, y2=y1 + 18.0)


def confidence_for_value(value: Any) -> float:
    if value is None or value == "" or value == []:
        return 0.0
    raw = str(value)
    if re.search(r"\d[Oo]+|\b[Oo]{2}\b", raw):
        return 0.62
    if "?" in raw:
        return 0.55
    return 0.94


def field(document_id: str, name: str, value: Any, index: int) -> dict[str, Any]:
    confidence = confidence_for_value(value)
    extracted = ExtractedField(
        name=name,
        value=value,
        confidence=confidence,
        evidence=FieldEvidence(
            document_id=document_id,
            page=1,
            bbox=make_bbox(index),
            raw_text=None if value is None else str(value),
            confidence=confidence,
        ),
    )
    return extracted.model_dump(mode="json")


def parse_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text.casefold() if text else None
