from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from customs_ai_agent.adapters.llm import DEFAULT_CLAUDE_MODEL
from customs_ai_agent.contracts import schema_sha256
from customs_ai_agent.contracts.evidence_runtime_release import (
    load_evidence_runtime_release_lock,
)
from customs_ai_agent.prompts import (
    DOCUMENT_ANALYSIS_PROMPT_ID,
    DOCUMENT_ANALYSIS_PROMPT_VERSION,
    load_document_analysis_system_prompt,
)
from customs_ai_agent.service.settings import ServiceSettings


SERVICE_API_VERSION = "internal-agent-service/v1"


def build_metadata(settings: ServiceSettings) -> dict[str, Any]:
    try:
        package_version = version("customs-ai-agent")
    except PackageNotFoundError:
        package_version = "0.3.0"
    evidence_lock = load_evidence_runtime_release_lock()
    prompt_bytes = load_document_analysis_system_prompt().encode("utf-8")
    return {
        "agent": {
            "build_sha": settings.build_sha,
            "package_version": package_version,
            "wheel_sha256": settings.wheel_sha256,
        },
        "capabilities": ["text"],
        "evidence_runtime_release": (
            {
                "release_state": "unreleased",
            }
            if evidence_lock.identity is None
            else {
                "evaluation_report_sha256": (evidence_lock.identity.evaluation_report_sha256),
                "knowledge_generation_id": (evidence_lock.identity.knowledge_generation_id),
                "provider_release": evidence_lock.identity.provider_release,
                "provider_revision": evidence_lock.identity.provider_revision,
                "release_state": "released",
                "retrieval_schema_sha256": (evidence_lock.identity.retrieval_schema_sha256),
                "runtime_release_sha256": (evidence_lock.identity.runtime_release_sha256),
            }
        ),
        "model": {
            "id": DEFAULT_CLAUDE_MODEL,
            "prompt_id": DOCUMENT_ANALYSIS_PROMPT_ID,
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompt_version": DOCUMENT_ANALYSIS_PROMPT_VERSION,
        },
        "schemas": {
            "review_request_v2_sha256": schema_sha256("backend", "v2", "review-request"),
            "review_result_v2_sha256": schema_sha256("backend", "v2", "review-result"),
        },
        "service_api_version": SERVICE_API_VERSION,
    }


__all__ = ["SERVICE_API_VERSION", "build_metadata"]
