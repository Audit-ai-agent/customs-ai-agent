from __future__ import annotations

import copy
import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from typing import Any

from customs_ai_agent.build_info import BuildManifestError, load_packaged_build_manifest
from customs_ai_agent.adapters.llm import DEFAULT_CLAUDE_MODEL
from customs_ai_agent.contracts import schema_sha256
from customs_ai_agent.contracts.evidence_runtime_release import (
    load_evidence_runtime_release_lock,
    load_pinned_evidence_provider_runtime_identity,
)
from customs_ai_agent.contracts.evidence_v3 import (
    EvidenceV3ContractError,
    load_evidence_runtime_release_lock_v3,
)
from customs_ai_agent.agentic import runtime_release_identity
from customs_ai_agent.prompts import (
    DOCUMENT_ANALYSIS_PROMPT_ID,
    DOCUMENT_ANALYSIS_PROMPT_VERSION,
    load_document_analysis_system_prompt,
)
from customs_ai_agent.release_profile import load_release_profile, load_reviewer_trust_store
from customs_ai_agent.assertion_trust import load_assertion_trust_store
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.settings import ServiceConfigurationError
from customs_ai_agent.service.release_admission import (
    ProductionReleaseAdmissionError,
    assert_production_release_admission,
)


SERVICE_API_VERSION = "internal-agent-service/v2"
LEGACY_SERVICE_API_VERSION = "internal-agent-service/v1"


def _service_openapi_sha256() -> str:
    try:
        raw = (
            files("customs_ai_agent.contracts")
            .joinpath("service", "v1", "openapi.lock.json")
            .read_bytes()
        )
        payload = json.loads(raw.decode("utf-8"))
        digest = payload["canonical_sha256"]
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        raise ServiceConfigurationError("customs_agent_service_openapi_lock_invalid") from None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "customs-agent-service-openapi-lock/v1"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ServiceConfigurationError("customs_agent_service_openapi_lock_invalid")
    return digest


def build_metadata(settings: ServiceSettings) -> dict[str, Any]:
    try:
        build_manifest = load_packaged_build_manifest(
            required=settings.environment == "production",
        )
    except BuildManifestError:
        raise ServiceConfigurationError("customs_agent_build_manifest_invalid") from None
    if build_manifest is not None and settings.build_sha != build_manifest.git_sha:
        raise ServiceConfigurationError("customs_agent_build_sha_mismatch")
    try:
        package_version = version("customs-ai-agent")
    except PackageNotFoundError:
        package_version = "0.3.0"
    evidence_lock = load_evidence_runtime_release_lock()
    evidence_provider_contract = load_pinned_evidence_provider_runtime_identity()
    release_profile = load_release_profile()
    reviewer_trust_store = load_reviewer_trust_store()
    assertion_trust_store = load_assertion_trust_store()
    try:
        evidence_v3_lock = load_evidence_runtime_release_lock_v3()
    except EvidenceV3ContractError:
        raise ServiceConfigurationError("customs_agent_evidence_v3_lock_invalid") from None
    evidence_v3_identity = evidence_v3_lock.identity
    if evidence_v3_identity is not None and (
        evidence_lock.identity is None
        or evidence_v3_identity.base_runtime_release != evidence_lock.identity
    ):
        raise ServiceConfigurationError("customs_agent_evidence_release_lock_mismatch")
    configured_model = os.environ.get(
        "CUSTOMS_AGENT_CLAUDE_MODEL",
        DEFAULT_CLAUDE_MODEL,
    ).strip()
    advertised_model = configured_model or DEFAULT_CLAUDE_MODEL
    runtime_release = runtime_release_identity(
        model_id=advertised_model,
        evidence_snapshot_id=(
            evidence_v3_identity.base_runtime_release.knowledge_snapshot_id
            if evidence_v3_identity is not None
            else None
        ),
        evidence_runtime_v3_sha256=(
            evidence_v3_identity.runtime_release_v3_sha256
            if evidence_v3_identity is not None
            else None
        ),
    )
    release_admission_error: str | None = None
    try:
        assert_production_release_admission(configured_model_id=advertised_model)
    except ProductionReleaseAdmissionError as exc:
        release_admission_error = exc.code
    active_authorized = (
        release_admission_error is None and release_profile.bounded_adaptive_active_enabled
    )
    prompt_bytes = load_document_analysis_system_prompt().encode("utf-8")
    return {
        "agent": {
            "build_sha": settings.build_sha,
            "package_version": package_version,
            "wheel_sha256": settings.wheel_sha256,
            "source_snapshot_sha256": (
                build_manifest.source_snapshot_sha256 if build_manifest is not None else "unknown"
            ),
        },
        "capabilities": [
            "text",
            "bounded_planning_shadow",
            "bounded_active_executor_injectable",
            "citation_verified_narrative",
            "grounded_case_answer",
            "immutable_revision",
        ],
        "transport_limits": {
            "max_body_bytes": settings.max_body_bytes,
            "max_documents": settings.max_documents,
            "max_text_characters": settings.max_text_characters,
            "text_only": True,
            "empty_visual_placeholders_accepted": True,
        },
        "agentic_limits": {
            "max_cycles": 2,
            "max_tool_calls": 6,
            "ocr_enabled": False,
            "planner_mode": "active" if active_authorized else "shadow",
            "active_execution_authorized_by_release": active_authorized,
            "active_tool_handlers_registered": True,
            "trusted_assertion_verifier_configured": (
                assertion_trust_store.release_state == "active"
            ),
            "effective_execution_mode": "active" if active_authorized else "shadow",
        },
        "automatic_completion_release": {
            "profile_id": release_profile.profile_id,
            "release_state": release_profile.release_state,
            "kill_switch_active": release_profile.kill_switch_active,
            "bounded_adaptive_active_enabled": (release_profile.bounded_adaptive_active_enabled),
            "profile_sha256": release_profile.profile_sha256,
            "review_subject_sha256": release_profile.review_subject_sha256,
            "model_revision": release_profile.model_revision,
            "evaluation_report_sha256": release_profile.evaluation_report_sha256,
            "reviewer_trust_store": {
                "release_state": reviewer_trust_store.release_state,
                "trust_store_sha256": reviewer_trust_store.trust_store_sha256,
                "trusted_reviewer_count": len(
                    {item.reviewer_id for item in reviewer_trust_store.keys}
                ),
            },
            "assertion_trust_store": {
                "release_state": assertion_trust_store.release_state,
                "trust_domain": assertion_trust_store.trust_domain,
                "trust_store_sha256": assertion_trust_store.trust_store_sha256,
                "active_key_count": len(
                    [item for item in assertion_trust_store.keys if item.status == "active"]
                ),
            },
            "production_admission": {
                "admitted": release_admission_error is None,
                "reason_code": release_admission_error,
            },
        },
        "evidence_provider_contract": evidence_provider_contract.to_wire(),
        "evidence_runtime_release": (
            {
                "release_state": "unreleased",
            }
            if evidence_lock.identity is None
            else {
                "release_state": "released",
                "identity": evidence_lock.identity.to_wire(),
            }
        ),
        "evidence_runtime_release_v3": (
            {"release_state": "unreleased"}
            if evidence_v3_identity is None
            else {
                "release_state": "released",
                "identity": evidence_v3_identity.to_wire(),
            }
        ),
        "model": {
            "id": advertised_model,
            "prompt_id": DOCUMENT_ANALYSIS_PROMPT_ID,
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompt_version": DOCUMENT_ANALYSIS_PROMPT_VERSION,
        },
        "runtime_release": runtime_release.model_dump(mode="json", warnings=False),
        "schemas": {
            "service_openapi_sha256": _service_openapi_sha256(),
            "review_request_v2_sha256": schema_sha256("backend", "v2", "review-request"),
            "review_result_v2_sha256": schema_sha256("backend", "v2", "review-result"),
            "review_request_v3_sha256": schema_sha256("backend", "v3", "review-request"),
            "review_result_v3_sha256": schema_sha256("backend", "v3", "review-result"),
        },
        "service_api_version": SERVICE_API_VERSION,
    }


def build_legacy_metadata(settings: ServiceSettings) -> dict[str, Any]:
    """Preserve the published V1 metadata wire while V2 adds capabilities."""

    metadata = copy.deepcopy(build_metadata(settings))
    metadata["capabilities"] = ["text"]
    metadata.pop("transport_limits", None)
    metadata.pop("agentic_limits", None)
    metadata.pop("automatic_completion_release", None)
    metadata["schemas"] = {
        key: value
        for key, value in metadata["schemas"].items()
        if key.startswith("review_request_v2") or key.startswith("review_result_v2")
    }
    metadata["service_api_version"] = LEGACY_SERVICE_API_VERSION
    return metadata


__all__ = [
    "LEGACY_SERVICE_API_VERSION",
    "SERVICE_API_VERSION",
    "build_legacy_metadata",
    "build_metadata",
]
