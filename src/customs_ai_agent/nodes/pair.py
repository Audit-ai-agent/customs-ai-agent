from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from customs_ai_agent.nodes.common import audit

fuzz: Any = None
try:
    from rapidfuzz import fuzz as _rapidfuzz

    fuzz = _rapidfuzz
except ImportError:  # pragma: no cover
    pass


def _score(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    if fuzz:
        return fuzz.token_set_ratio(left, right) / 100.0
    return SequenceMatcher(None, left.casefold(), right.casefold()).ratio()


def _first_doc(state: dict[str, Any], doc_type: str) -> tuple[str | None, dict[str, Any] | None]:
    for doc_id, doc in (state.get("normalized_documents") or {}).items():
        if doc.get("document_type") == doc_type:
            return doc_id, doc
    return None, None


def pair_documents(state: dict[str, Any]) -> dict[str, Any]:
    invoice_id, invoice = _first_doc(state, "invoice")
    packing_id, packing = _first_doc(state, "packing_list")
    links: list[dict[str, Any]] = []

    if invoice and packing:
        packing_items = packing.get("line_items") or []
        for invoice_item in invoice.get("line_items") or []:
            best_item = None
            best_score = 0.0
            for packing_item in packing_items:
                score = _score(invoice_item.get("description"), packing_item.get("description"))
                if score > best_score:
                    best_score = score
                    best_item = packing_item
            links.append(
                {
                    "invoice_item_id": invoice_item.get("item_id"),
                    "packing_item_id": best_item.get("item_id") if best_item else None,
                    "score": round(best_score, 4),
                }
            )

    result = {
        "invoice_document_id": invoice_id,
        "packing_list_document_id": packing_id,
        "line_item_links": links,
    }
    return {
        "pairing_result": result,
        "current_stage": "pair_documents",
        "audit_trail": audit(
            state,
            "pair_documents",
            "completed",
            "Paired shipment and line items.",
            {"line_item_links": len(links)},
        ),
    }
