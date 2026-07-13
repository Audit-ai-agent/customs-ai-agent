from __future__ import annotations

import re
from typing import Any

from customs_ai_agent.nodes.common import audit, extract_label, extract_list_label, field


def _item_lines(text: str) -> list[str]:
    return [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().upper().startswith("ITEM:")
    ]


def _parse_item(line: str) -> dict[str, str | None]:
    parts = [part.strip() for part in line.split("|")]
    item: dict[str, str | None] = {"description": parts[0] if parts else None}
    key_map = {
        "HS": "hs_code",
        "QTY": "quantity",
        "UNIT PRICE": "unit_price",
        "AMOUNT": "amount",
        "ORIGIN": "origin_country",
        "NET WEIGHT": "net_weight_kg",
        "GROSS WEIGHT": "gross_weight_kg",
        "PACKAGE COUNT": "package_count",
    }
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = [piece.strip() for piece in part.split(":", 1)]
        mapped = key_map.get(key.upper())
        if mapped:
            item[mapped] = value or None
    return item


def _extract_bl(document_id: str, text: str, filename: str) -> dict[str, Any]:
    fields = {
        "bl_no": field(document_id, "bl_no", extract_label(text, "B/L NO"), 1),
        "shipper": field(document_id, "shipper", extract_label(text, "SHIPPER"), 2),
        "consignee": field(document_id, "consignee", extract_label(text, "CONSIGNEE"), 3),
        "vessel": field(document_id, "vessel", extract_label(text, "VESSEL"), 4),
        "voyage_no": field(document_id, "voyage_no", extract_label(text, "VOYAGE NO"), 5),
        "port_of_loading": field(document_id, "port_of_loading", extract_label(text, "PORT OF LOADING"), 6),
        "port_of_discharge": field(document_id, "port_of_discharge", extract_label(text, "PORT OF DISCHARGE"), 7),
        "container_numbers": field(document_id, "container_numbers", extract_list_label(text, "CONTAINER NO", "CONTAINER NUMBERS"), 8),
        "seal_numbers": field(document_id, "seal_numbers", extract_list_label(text, "SEAL NO", "SEAL NUMBERS"), 9),
        "packages": field(document_id, "packages", extract_label(text, "PACKAGES"), 10),
        "gross_weight_kg": field(document_id, "gross_weight_kg", extract_label(text, "GROSS WEIGHT"), 11),
    }
    return _doc("bill_of_lading", document_id, filename, fields, [])


def _extract_invoice(document_id: str, text: str, filename: str) -> dict[str, Any]:
    fields = {
        "invoice_no": field(document_id, "invoice_no", extract_label(text, "INVOICE NO"), 1),
        "invoice_date": field(document_id, "invoice_date", extract_label(text, "INVOICE DATE"), 2),
        "seller": field(document_id, "seller", extract_label(text, "SELLER"), 3),
        "buyer": field(document_id, "buyer", extract_label(text, "BUYER"), 4),
        "incoterms": field(document_id, "incoterms", extract_label(text, "INCOTERMS"), 5),
        "currency": field(document_id, "currency", extract_label(text, "CURRENCY"), 6),
        "total_amount": field(document_id, "total_amount", extract_label(text, "TOTAL AMOUNT"), 7),
    }
    line_items = []
    for index, raw_item in enumerate(_item_lines(text), start=1):
        parsed = _parse_item(raw_item)
        line_items.append(
            {
                "item_id": f"{document_id}_line_{index}",
                **{
                    name: field(document_id, name, value, 20 + index + offset)
                    for offset, (name, value) in enumerate(parsed.items())
                },
            }
        )
    return _doc("invoice", document_id, filename, fields, line_items)


def _extract_packing(document_id: str, text: str, filename: str) -> dict[str, Any]:
    fields = {
        "packing_no": field(document_id, "packing_no", extract_label(text, "PACKING NO"), 1),
        "seller": field(document_id, "seller", extract_label(text, "SELLER"), 2),
        "buyer": field(document_id, "buyer", extract_label(text, "BUYER"), 3),
        "container_numbers": field(document_id, "container_numbers", extract_list_label(text, "CONTAINER NO", "CONTAINER NUMBERS"), 4),
        "packages": field(document_id, "packages", extract_label(text, "PACKAGES"), 5),
    }
    line_items = []
    for index, raw_item in enumerate(_item_lines(text), start=1):
        parsed = _parse_item(raw_item)
        line_items.append(
            {
                "item_id": f"{document_id}_line_{index}",
                **{
                    name: field(document_id, name, value, 20 + index + offset)
                    for offset, (name, value) in enumerate(parsed.items())
                },
            }
        )
    return _doc("packing_list", document_id, filename, fields, line_items)


def _extract_co(document_id: str, text: str, filename: str) -> dict[str, Any]:
    hs_codes = re.split(r"[,;/\s]+", extract_label(text, "HS CODE", "HS CODES") or "")
    hs_codes = [code for code in hs_codes if code]
    fields = {
        "certificate_no": field(document_id, "certificate_no", extract_label(text, "CERTIFICATE NO"), 1),
        "exporter": field(document_id, "exporter", extract_label(text, "EXPORTER"), 2),
        "importer": field(document_id, "importer", extract_label(text, "IMPORTER"), 3),
        "origin_country": field(document_id, "origin_country", extract_label(text, "ORIGIN COUNTRY"), 4),
        "hs_codes": field(document_id, "hs_codes", hs_codes, 5),
        "goods_description": field(document_id, "goods_description", extract_label(text, "GOODS DESCRIPTION"), 6),
        "issue_date": field(document_id, "issue_date", extract_label(text, "ISSUE DATE"), 7),
    }
    return _doc("certificate_of_origin", document_id, filename, fields, [])


def _doc(
    doc_type: str,
    document_id: str,
    filename: str,
    fields: dict[str, dict[str, Any]],
    line_items: list[dict[str, Any]],
) -> dict[str, Any]:
    confidences = [value["confidence"] for value in fields.values()]
    for item in line_items:
        confidences.extend(value["confidence"] for key, value in item.items() if key != "item_id")
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "document_id": document_id,
        "filename": filename,
        "document_type": doc_type,
        "fields": fields,
        "line_items": line_items,
        "extraction_status": "completed",
        "extraction_confidence": round(avg_confidence, 4),
    }


def extract_documents(state: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, dict[str, Any]] = {}
    by_id = {doc["document_id"]: doc for doc in state.get("documents") or []}
    for document_id, classified in (state.get("classified_documents") or {}).items():
        doc = by_id[document_id]
        doc_type = classified["document_type"]
        if doc_type == "invoice":
            extracted[document_id] = _extract_invoice(document_id, doc["content_text"], doc["filename"])
        elif doc_type == "packing_list":
            extracted[document_id] = _extract_packing(document_id, doc["content_text"], doc["filename"])
        elif doc_type == "bill_of_lading":
            extracted[document_id] = _extract_bl(document_id, doc["content_text"], doc["filename"])
        elif doc_type == "certificate_of_origin":
            extracted[document_id] = _extract_co(document_id, doc["content_text"], doc["filename"])
        else:
            extracted[document_id] = _doc("unknown", document_id, doc["filename"], {}, [])

    return {
        "extracted_documents": extracted,
        "current_stage": "extract_documents",
        "audit_trail": audit(
            state,
            "extract_documents",
            "completed",
            "Extracted structured fields from documents.",
            {"document_count": len(extracted)},
        ),
    }
