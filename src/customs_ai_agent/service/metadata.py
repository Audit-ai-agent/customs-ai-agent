"""Non-secret operational metadata for the two local Agent profiles."""

from __future__ import annotations

import os
from typing import Any

from customs_ai_agent.adapters.llm import DEFAULT_CLAUDE_MODEL
from customs_ai_agent.contracts import schema_sha256
from customs_ai_agent.service.settings import ServiceSettings


SERVICE_API_VERSION = "internal-agent-service/v2"


def build_metadata(settings: ServiceSettings) -> dict[str, Any]:
    """Return only the facts a local peer needs to call this service."""

    if settings.service_mode == "functional-uat":
        engine = {
            "kind": "claude",
            "model_id": os.environ.get("CUSTOMS_AGENT_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip()
            or DEFAULT_CLAUDE_MODEL,
        }
    else:
        engine = {"kind": "deterministic", "model_id": "development-fixture-deterministic"}
    return {
        "service_api_version": SERVICE_API_VERSION,
        "service_mode": settings.service_mode,
        "human_review_only": True,
        "capabilities": ["text", "evidence_http", "profile_mode_handshake"],
        "limits": {
            "max_body_bytes": settings.max_body_bytes,
            "max_documents": settings.max_documents,
            "max_text_characters": settings.max_text_characters,
        },
        "engine": engine,
        "schemas": {
            "review_request_v3_sha256": schema_sha256("backend", "v3", "review-request"),
            "review_result_v3_sha256": schema_sha256("backend", "v3", "review-result"),
            "evidence_retrieval_v3_sha256": schema_sha256("evidence", "v3", "rag-retrieval"),
            "evidence_readiness_v3_sha256": schema_sha256("evidence", "v3", "readiness"),
        },
    }


__all__ = ["SERVICE_API_VERSION", "build_metadata"]
