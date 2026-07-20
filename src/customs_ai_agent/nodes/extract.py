from __future__ import annotations

import copy
import re
from typing import Any

from customs_ai_agent.identifiers import stable_line_item_id
from customs_ai_agent.nodes.common import audit, extract_label, extract_list_label, field


def _item_lines(text: str) -> list[tuple[str, int]]:
    lines: list[tuple[str, int]] = []
    for match in re.finditer(r"^\s*ITEM\s*:\s*(.*?)\s*$", text, re.IGNORECASE | re.MULTILINE):
        captured = match.group(1)
        stripped = captured.strip()
        if stripped:
            leading = len(captured) - len(captured.lstrip())
            lines.append((stripped, match.start(1) + leading))
    return lines


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
        "port_of_loading": field(
            document_id, "port_of_loading", extract_label(text, "PORT OF LOADING"), 6
        ),
        "port_of_discharge": field(
            document_id, "port_of_discharge", extract_label(text, "PORT OF DISCHARGE"), 7
        ),
        "container_numbers": field(
            document_id,
            "container_numbers",
            extract_list_label(text, "CONTAINER NO", "CONTAINER NUMBERS"),
            8,
        ),
        "seal_numbers": field(
            document_id, "seal_numbers", extract_list_label(text, "SEAL NO", "SEAL NUMBERS"), 9
        ),
        "packages": field(document_id, "packages", extract_label(text, "PACKAGES"), 10),
        "gross_weight_kg": field(
            document_id, "gross_weight_kg", extract_label(text, "GROSS WEIGHT"), 11
        ),
        "measurement_cbm": field(
            document_id, "measurement_cbm", extract_label(text, "MEASUREMENT", "CBM"), 12
        ),
        "freight_prepaid": field(
            document_id,
            "freight_prepaid",
            extract_label(text, "FREIGHT PREPAID"),
            13,
        ),
        "freight_collect": field(
            document_id,
            "freight_collect",
            extract_label(text, "FREIGHT COLLECT"),
            14,
        ),
        "freight_terms": field(
            document_id,
            "freight_terms",
            extract_label(text, "FREIGHT TERMS"),
            15,
        ),
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
        "departure_date": field(
            document_id, "departure_date", extract_label(text, "DEPARTURE DATE"), 8
        ),
        "vessel": field(document_id, "vessel", extract_label(text, "VESSEL"), 9),
        "voyage_no": field(document_id, "voyage_no", extract_label(text, "VOYAGE NO"), 10),
        "port_of_loading": field(
            document_id, "port_of_loading", extract_label(text, "PORT OF LOADING"), 11
        ),
        "port_of_discharge": field(
            document_id, "port_of_discharge", extract_label(text, "PORT OF DISCHARGE"), 12
        ),
        "country_of_origin": field(
            document_id,
            "country_of_origin",
            extract_label(text, "COUNTRY OF ORIGIN", "ORIGIN COUNTRY"),
            13,
        ),
        "shipping_marks": field(
            document_id, "shipping_marks", extract_label(text, "SHIPPING MARKS", "MARKS"), 14
        ),
        "packages": field(document_id, "packages", extract_label(text, "PACKAGES"), 15),
    }
    line_items = []
    for index, (raw_item, line_start) in enumerate(_item_lines(text), start=1):
        parsed = _parse_item(raw_item)
        line_items.append(
            {
                "item_id": stable_line_item_id(document_id, index),
                **{
                    name: field(
                        document_id,
                        name,
                        value,
                        20 + index + offset,
                        text_start=(
                            line_start + raw_item.find(str(value))
                            if value is not None and raw_item.find(str(value)) >= 0
                            else None
                        ),
                        text_end=(
                            line_start + raw_item.find(str(value)) + len(str(value))
                            if value is not None and raw_item.find(str(value)) >= 0
                            else None
                        ),
                    )
                    for offset, (name, value) in enumerate(parsed.items())
                },
            }
        )
    return _doc("invoice", document_id, filename, fields, line_items)


def _extract_packing(document_id: str, text: str, filename: str) -> dict[str, Any]:
    fields = {
        "packing_no": field(document_id, "packing_no", extract_label(text, "PACKING NO"), 1),
        "invoice_no": field(document_id, "invoice_no", extract_label(text, "INVOICE NO"), 2),
        "invoice_date": field(document_id, "invoice_date", extract_label(text, "INVOICE DATE"), 3),
        "bl_no": field(document_id, "bl_no", extract_label(text, "B/L NO"), 4),
        "seller": field(document_id, "seller", extract_label(text, "SELLER"), 5),
        "buyer": field(document_id, "buyer", extract_label(text, "BUYER"), 6),
        "departure_date": field(
            document_id, "departure_date", extract_label(text, "DEPARTURE DATE"), 7
        ),
        "vessel": field(document_id, "vessel", extract_label(text, "VESSEL"), 8),
        "voyage_no": field(document_id, "voyage_no", extract_label(text, "VOYAGE NO"), 9),
        "port_of_loading": field(
            document_id, "port_of_loading", extract_label(text, "PORT OF LOADING"), 10
        ),
        "port_of_discharge": field(
            document_id, "port_of_discharge", extract_label(text, "PORT OF DISCHARGE"), 11
        ),
        "country_of_origin": field(
            document_id,
            "country_of_origin",
            extract_label(text, "COUNTRY OF ORIGIN", "ORIGIN COUNTRY"),
            12,
        ),
        "shipping_marks": field(
            document_id, "shipping_marks", extract_label(text, "SHIPPING MARKS", "MARKS"), 13
        ),
        "container_numbers": field(
            document_id,
            "container_numbers",
            extract_list_label(text, "CONTAINER NO", "CONTAINER NUMBERS"),
            14,
        ),
        "seal_numbers": field(
            document_id,
            "seal_numbers",
            extract_list_label(text, "SEAL NO", "SEAL NUMBERS"),
            15,
        ),
        "packages": field(document_id, "packages", extract_label(text, "PACKAGES"), 16),
        "total_gross_weight_kg": field(
            document_id,
            "total_gross_weight_kg",
            extract_label(text, "TOTAL GROSS WEIGHT", "GROSS WEIGHT"),
            17,
        ),
        "total_measurement_cbm": field(
            document_id,
            "total_measurement_cbm",
            extract_label(text, "TOTAL MEASUREMENT", "MEASUREMENT", "CBM"),
            18,
        ),
    }
    line_items = []
    for index, (raw_item, line_start) in enumerate(_item_lines(text), start=1):
        parsed = _parse_item(raw_item)
        line_items.append(
            {
                "item_id": stable_line_item_id(document_id, index),
                **{
                    name: field(
                        document_id,
                        name,
                        value,
                        20 + index + offset,
                        text_start=(
                            line_start + raw_item.find(str(value))
                            if value is not None and raw_item.find(str(value)) >= 0
                            else None
                        ),
                        text_end=(
                            line_start + raw_item.find(str(value)) + len(str(value))
                            if value is not None and raw_item.find(str(value)) >= 0
                            else None
                        ),
                    )
                    for offset, (name, value) in enumerate(parsed.items())
                },
            }
        )
    return _doc("packing_list", document_id, filename, fields, line_items)


def _extract_co(document_id: str, text: str, filename: str) -> dict[str, Any]:
    hs_codes = re.split(r"[,;/\s]+", extract_label(text, "HS CODE", "HS CODES") or "")
    hs_codes = [code for code in hs_codes if code]
    fields = {
        "certificate_no": field(
            document_id, "certificate_no", extract_label(text, "CERTIFICATE NO"), 1
        ),
        "exporter": field(document_id, "exporter", extract_label(text, "EXPORTER"), 2),
        "importer": field(document_id, "importer", extract_label(text, "IMPORTER"), 3),
        "origin_country": field(
            document_id, "origin_country", extract_label(text, "ORIGIN COUNTRY"), 4
        ),
        "hs_codes": field(document_id, "hs_codes", hs_codes, 5),
        "goods_description": field(
            document_id, "goods_description", extract_label(text, "GOODS DESCRIPTION"), 6
        ),
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


def _failed_doc(
    document_id: str,
    filename: str,
    *,
    input_modality: str,
    extraction_provider: str,
    model_id: str | None = None,
    prompt_version: str | None = None,
    ocr_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only fail-closed extraction representation.

    A failed provider/schema/OCR/grounding boundary is not a completed
    ``unknown`` extraction.  It has no business values and can therefore be
    safely serialized by the public V2 contract even when OCR failed before a
    ledger could be produced.
    """

    result: dict[str, Any] = {
        "document_id": document_id,
        "filename": filename,
        "document_type": "unknown",
        "fields": {},
        "line_items": [],
        "extraction_status": "failed",
        "extraction_confidence": 0.0,
        "extraction_provider": extraction_provider,
        "input_modality": input_modality,
        "model_id": model_id,
        "prompt_version": prompt_version,
    }
    if ocr_layout is not None:
        result["ocr_layout"] = ocr_layout
    return result


def extract_documents(
    state: dict[str, Any],
) -> dict[str, Any]:
    extracted: dict[str, dict[str, Any]] = {}
    model_errors = list(state.get("model_errors") or [])
    by_id = {doc["document_id"]: doc for doc in state.get("documents") or []}
    llm_mode = state.get("llm_mode") == "llm"
    for document_id, classified in (state.get("classified_documents") or {}).items():
        doc = by_id[document_id]
        doc_type = classified["document_type"]
        stored_analysis = (state.get("model_analyses") or {}).get(document_id)
        if llm_mode and stored_analysis and stored_analysis.get("mapped_extraction"):
            # This snapshot was grounded while the one-shot payload was present.
            # Extraction is intentionally incapable of obtaining source bytes.
            extracted[document_id] = copy.deepcopy(stored_analysis["mapped_extraction"])
        elif llm_mode:
            # A provider/schema/OCR/grounding failure is an explicit failed
            # extraction. It never falls back to the deterministic extractor
            # and never masquerades as a completed unknown document.
            metadata = (stored_analysis or {}).get("metadata") or {}
            raw_ledger = (stored_analysis or {}).get("ocr_layout_ledger")
            extracted[document_id] = _failed_doc(
                document_id,
                doc["filename"],
                input_modality=str(
                    metadata.get("input_modality") or classified.get("input_modality") or "text"
                ),
                extraction_provider=str(
                    metadata.get("provider") or classified.get("provider") or "anthropic"
                ),
                model_id=metadata.get("model_id"),
                prompt_version=metadata.get("prompt_version"),
                ocr_layout=(
                    {
                        key: raw_ledger[key]
                        for key in (
                            "schema_version",
                            "ledger_id",
                            "provider_id",
                            "engine_id",
                            "engine_version",
                            "reading_order_policy_id",
                            "model_id",
                            "model_assets_lock_sha256",
                        )
                    }
                    if isinstance(raw_ledger, dict)
                    else None
                ),
            )
        elif doc_type == "invoice":
            extracted[document_id] = _extract_invoice(
                document_id, str(doc.get("content_text") or ""), doc["filename"]
            )
        elif doc_type == "packing_list":
            extracted[document_id] = _extract_packing(
                document_id, str(doc.get("content_text") or ""), doc["filename"]
            )
        elif doc_type == "bill_of_lading":
            extracted[document_id] = _extract_bl(
                document_id, str(doc.get("content_text") or ""), doc["filename"]
            )
        elif doc_type == "certificate_of_origin":
            extracted[document_id] = _extract_co(
                document_id, str(doc.get("content_text") or ""), doc["filename"]
            )
        else:
            extracted[document_id] = _doc("unknown", document_id, doc["filename"], {}, [])

    return {
        "extracted_documents": extracted,
        "model_errors": model_errors,
        "current_stage": "extract_documents",
        "audit_trail": audit(
            state,
            "extract_documents",
            "completed",
            "Extracted structured fields from documents.",
            {
                "document_count": len(extracted),
                "provider": "anthropic" if llm_mode else "rules",
                "failed_document_count": sum(
                    doc["extraction_status"] != "completed" or doc["document_type"] == "unknown"
                    for doc in extracted.values()
                ),
            },
        ),
    }
