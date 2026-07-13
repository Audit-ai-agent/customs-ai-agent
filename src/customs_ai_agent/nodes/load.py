from __future__ import annotations

from typing import Any

from customs_ai_agent.nodes.common import audit


def load_case(state: dict[str, Any]) -> dict[str, Any]:
    documents = state.get("documents") or []
    return {
        "document_ids": [doc["document_id"] for doc in documents],
        "current_stage": "load_case",
        "audit_trail": audit(
            state,
            "load_case",
            "completed",
            f"Loaded {len(documents)} documents.",
        ),
    }
