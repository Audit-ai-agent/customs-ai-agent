from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from customs_ai_agent.build_info import package_code_snapshot_sha256
from customs_ai_agent.experience import experience_prompt_hashes, experience_schema_hashes
from customs_ai_agent.schemas.experience import AgentUatCapabilityAttestationV1
from customs_ai_agent.schemas.v3 import canonical_json_sha256


UAT_TOOL_REGISTRY_ID = "customs-agent-tools/text-uat-v2"
_PRODUCTION_TOOL_REGISTRY_ID = "customs-agent-tools/text-v1"
UAT_ALLOWED_TOOLS = (
    "document.reread_span/v1",
    "evidence.retrieve/v3",
    "human.request/v1",
    "explanation.compose/v2",
)
# ReviewResultV3 is a frozen public contract. Its canonical plan/trace can only
# represent the inherited v1 explanation action. The UAT-only v2 renderer is
# attested by UAT_ALLOWED_TOOLS and recorded additively in the Experience
# envelope instead of widening that public contract.
UAT_CANONICAL_REVIEW_TOOLS = (
    "document.reread_span/v1",
    "evidence.retrieve/v3",
    "human.request/v1",
    "explanation.compose/v1",
)
UAT_RESOLVABLE_GAPS = ("field_uncertainty", "evidence_incomplete")


class UatCapabilityError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _tool_registry_sha256() -> str:
    return canonical_json_sha256(
        {
            "registry_id": UAT_TOOL_REGISTRY_ID,
            "inherits_production_registry": _PRODUCTION_TOOL_REGISTRY_ID,
            "allowed_tools": list(UAT_ALLOWED_TOOLS),
            "resolvable_gaps": list(UAT_RESOLVABLE_GAPS),
            "max_cycles": 2,
            "max_tool_calls": 6,
            "max_rereads_per_field": 1,
            "outcome_write_authority": False,
            "auto_completion_enabled": False,
        }
    )


def _bundle_sha256(values: dict[str, str]) -> str:
    return canonical_json_sha256(dict(sorted(values.items())))


def issue_uat_capability(
    *,
    agent_commit: str,
    model_id: str,
    evidence_runtime_v3_sha256: str,
    issued_at: datetime,
    expires_at: datetime,
    adaptive_active: bool,
) -> AgentUatCapabilityAttestationV1:
    if issued_at.tzinfo is None or expires_at.tzinfo is None:
        raise UatCapabilityError("uat_capability_clock_invalid")
    issued_wire = issued_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    expires_wire = expires_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "profile_id": (
            "customs-agent-local-uat/"
            + canonical_json_sha256(
                {
                    "agent_commit": agent_commit,
                    "model_id": model_id,
                    "evidence_runtime_v3_sha256": evidence_runtime_v3_sha256,
                    "issued_at": issued_wire,
                    "expires_at": expires_wire,
                }
            )[:24]
        ),
        "environment": "local_uat",
        "evidence_profile": "preserved-3law-v3-scenario-eval",
        "loopback_only": True,
        # Hash the exact JSON wire representation. Pydantic serializes UTC as
        # ``Z``; using datetime objects here would hash ``+00:00`` first and
        # produce a self-hash that cannot survive model validation.
        "issued_at": issued_wire,
        "expires_at": expires_wire,
        "agent_commit": agent_commit,
        "agent_code_snapshot_sha256": package_code_snapshot_sha256(),
        "model_id": model_id,
        "prompt_bundle_sha256": _bundle_sha256(experience_prompt_hashes()),
        "schema_bundle_sha256": _bundle_sha256(experience_schema_hashes()),
        "tool_registry_id": UAT_TOOL_REGISTRY_ID,
        "tool_registry_sha256": _tool_registry_sha256(),
        "evidence_runtime_v3_sha256": evidence_runtime_v3_sha256,
        "max_cycles": 2,
        "max_tool_calls": 6,
        "max_rereads_per_field": 1,
        "adaptive_active": adaptive_active,
        "auto_completion_enabled": False,
    }
    payload["capability_sha256"] = canonical_json_sha256(payload)
    return AgentUatCapabilityAttestationV1.model_validate(payload)


def capability_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_uat_capability(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_agent_commit: str | None,
    expected_model_id: str,
    expected_evidence_runtime_v3_sha256: str,
    require_active: bool,
    now: datetime | None = None,
) -> AgentUatCapabilityAttestationV1:
    candidate = Path(path).expanduser().resolve(strict=True)
    if capability_file_sha256(candidate) != expected_file_sha256:
        raise UatCapabilityError("uat_capability_file_sha256_mismatch")
    try:
        raw = candidate.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise UatCapabilityError("uat_capability_file_invalid") from None
    if not isinstance(payload, dict):
        raise UatCapabilityError("uat_capability_file_invalid")
    try:
        capability = AgentUatCapabilityAttestationV1.model_validate(payload)
    except Exception:
        raise UatCapabilityError("uat_capability_contract_invalid") from None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise UatCapabilityError("uat_capability_clock_invalid")
    identity_mismatch = (
        (expected_agent_commit is not None and capability.agent_commit != expected_agent_commit)
        or capability.agent_code_snapshot_sha256 != package_code_snapshot_sha256()
        or capability.model_id != expected_model_id
        or capability.prompt_bundle_sha256 != _bundle_sha256(experience_prompt_hashes())
        or capability.schema_bundle_sha256 != _bundle_sha256(experience_schema_hashes())
        or capability.tool_registry_id != UAT_TOOL_REGISTRY_ID
        or capability.tool_registry_sha256 != _tool_registry_sha256()
        or capability.evidence_runtime_v3_sha256 != expected_evidence_runtime_v3_sha256
    )
    if identity_mismatch:
        raise UatCapabilityError("uat_capability_identity_mismatch")
    if not capability.issued_at <= current < capability.expires_at:
        raise UatCapabilityError("uat_capability_expired")
    if require_active and not capability.adaptive_active:
        raise UatCapabilityError("uat_capability_active_mode_disabled")
    if capability.auto_completion_enabled:
        raise UatCapabilityError("uat_capability_auto_completion_forbidden")
    return capability


def assert_runtime_uat_capability(
    capability: AgentUatCapabilityAttestationV1 | None,
    *,
    agent_code_snapshot_sha256: str,
    model_id: str,
    evidence_runtime_v3_sha256: str | None,
    expected_agent_commit: str | None = None,
    require_active: bool = True,
    now: datetime | None = None,
) -> bool:
    if capability is None:
        return False
    try:
        # The capability is mutable as an in-process Pydantic object. Rebuild
        # it from its complete wire representation at every authority boundary
        # so its self-hash, lifetime, and exact schema are checked again.
        current_capability = AgentUatCapabilityAttestationV1.model_validate(
            capability.model_dump(mode="json", warnings=False)
        )
    except Exception:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return False
    return (
        current_capability.environment == "local_uat"
        and current_capability.loopback_only is True
        and current_capability.evidence_profile == "preserved-3law-v3-scenario-eval"
        and current_capability.issued_at <= current < current_capability.expires_at
        and (
            expected_agent_commit is None
            or current_capability.agent_commit == expected_agent_commit
        )
        and current_capability.agent_code_snapshot_sha256 == agent_code_snapshot_sha256
        and current_capability.model_id == model_id
        and current_capability.prompt_bundle_sha256 == _bundle_sha256(experience_prompt_hashes())
        and current_capability.schema_bundle_sha256 == _bundle_sha256(experience_schema_hashes())
        and current_capability.tool_registry_id == UAT_TOOL_REGISTRY_ID
        and current_capability.tool_registry_sha256 == _tool_registry_sha256()
        and current_capability.evidence_runtime_v3_sha256 == evidence_runtime_v3_sha256
        and (not require_active or current_capability.adaptive_active)
        and current_capability.auto_completion_enabled is False
    )


__all__ = [
    "UAT_ALLOWED_TOOLS",
    "UAT_CANONICAL_REVIEW_TOOLS",
    "UAT_RESOLVABLE_GAPS",
    "UAT_TOOL_REGISTRY_ID",
    "UatCapabilityError",
    "assert_runtime_uat_capability",
    "capability_file_sha256",
    "issue_uat_capability",
    "load_uat_capability",
]
