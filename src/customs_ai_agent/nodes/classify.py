from __future__ import annotations

from typing import Any

from customs_ai_agent.nodes.common import audit


def _classify(text: str, filename: str) -> tuple[str, float]:
    content = f"{filename}\n{text}".upper()
    if "COMMERCIAL INVOICE" in content or "INVOICE NO" in content:
        return "invoice", 0.97
    if "PACKING LIST" in content or "PACKING NO" in content:
        return "packing_list", 0.97
    if "BILL OF LADING" in content or "B/L NO" in content:
        return "bill_of_lading", 0.97
    if "CERTIFICATE OF ORIGIN" in content or "ORIGIN COUNTRY" in content:
        return "certificate_of_origin", 0.96
    return "unknown", 0.40


def classify_documents(state: dict[str, Any]) -> dict[str, Any]:
    classified: dict[str, dict[str, Any]] = {}
    requires_human_review = state.get("requires_human_review", False)

    for doc in state.get("documents") or []:
        doc_type = doc.get("declared_type")
        confidence = 1.0 if doc_type else None
        if not doc_type:
            doc_type, confidence = _classify(doc["content_text"], doc["filename"])
        if doc_type == "unknown" or (confidence or 0) < 0.70:
            requires_human_review = True
        classified[doc["document_id"]] = {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "document_type": doc_type,
            "confidence": confidence,
        }

    return {
        "classified_documents": classified,
        "current_stage": "classify_documents",
        "requires_human_review": requires_human_review,
        "audit_trail": audit(
            state,
            "classify_documents",
            "completed",
            "Classified documents.",
            {"document_count": len(classified)},
        ),
    }
