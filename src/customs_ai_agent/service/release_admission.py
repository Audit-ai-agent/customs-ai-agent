"""Fail-closed admission for the Production Agent V3 process.

This module deliberately performs only local, immutable release checks.  It
does not call Claude or Data.  The service readiness probe performs the live
Data handshake after this local Gate has admitted the process.
"""

from __future__ import annotations

from dataclasses import dataclass

from customs_ai_agent.agentic import agentic_semantic_schema_hashes, runtime_release_identity
from customs_ai_agent.contracts import schema_sha256
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseLockError,
    EvidenceRuntimeReleaseUnreleasedError,
    load_released_evidence_runtime_identity,
)
from customs_ai_agent.contracts.evidence_v3 import (
    EvidenceV3ContractError,
    EvidenceV3UnreleasedError,
    load_released_evidence_runtime_identity_v3,
)
from customs_ai_agent.release_profile import load_release_profile
from customs_ai_agent.assertion_trust import load_assertion_trust_store
from customs_ai_agent.schemas.v3 import RuntimeReleaseIdentityV1


@dataclass(frozen=True, slots=True)
class ProductionReleaseAdmissionError(RuntimeError):
    """Content-free, allowlisted Production release admission failure."""

    code: str

    def __str__(self) -> str:
        return self.code


def assert_production_release_admission(
    *,
    configured_model_id: str,
) -> RuntimeReleaseIdentityV1:
    """Return the exact admitted identity or fail before any external call.

    Production V3 is one indivisible release: released Data V2/V3 locks, an
    approved two-reviewer Claude profile, exact model revision, executable
    package snapshot, prompts, Schemas, policy, and Tool registry must all
    match.  A caller cannot selectively bypass one of these bindings.
    """

    try:
        base = load_released_evidence_runtime_identity()
    except EvidenceRuntimeReleaseUnreleasedError:
        raise ProductionReleaseAdmissionError("evidence_runtime_release_unreleased") from None
    except EvidenceRuntimeReleaseLockError:
        raise ProductionReleaseAdmissionError("evidence_runtime_release_lock_invalid") from None
    try:
        evidence_v3 = load_released_evidence_runtime_identity_v3()
    except EvidenceV3UnreleasedError:
        raise ProductionReleaseAdmissionError("evidence_runtime_v3_release_unreleased") from None
    except EvidenceV3ContractError:
        raise ProductionReleaseAdmissionError("evidence_runtime_v3_release_invalid") from None
    if evidence_v3.base_runtime_release != base:
        raise ProductionReleaseAdmissionError("evidence_runtime_release_identity_mismatch")

    try:
        assertion_store = load_assertion_trust_store()
    except (OSError, TypeError, UnicodeError, ValueError):
        raise ProductionReleaseAdmissionError("assertion_trust_store_invalid") from None
    if assertion_store.release_state != "active":
        raise ProductionReleaseAdmissionError("assertion_trust_store_unreleased")

    try:
        profile = load_release_profile()
    except (OSError, TypeError, UnicodeError, ValueError):
        raise ProductionReleaseAdmissionError("claude_release_profile_invalid") from None
    if (
        profile.release_state != "approved"
        or profile.kill_switch_active
        or not profile.auto_completion_enabled
    ):
        raise ProductionReleaseAdmissionError("claude_release_profile_unapproved")
    # Production traffic uses an exact dated model identifier.  An alias may be
    # useful for diagnostics but cannot satisfy a signed release identity.
    if profile.model_id != configured_model_id or profile.model_revision != configured_model_id:
        raise ProductionReleaseAdmissionError("claude_configured_model_identity_mismatch")

    try:
        runtime = runtime_release_identity(
            model_id=configured_model_id,
            evidence_snapshot_id=base.knowledge_snapshot_id,
            evidence_runtime_v3_sha256=evidence_v3.runtime_release_v3_sha256,
        )
    except Exception:
        raise ProductionReleaseAdmissionError("agent_runtime_release_identity_invalid") from None
    expected_observed = (
        (profile.agent_code_snapshot_sha256, runtime.agent_code_snapshot_sha256),
        (profile.model_id, runtime.model_id),
        (profile.model_revision, runtime.model_revision),
        (profile.extraction_prompt_sha256, runtime.extraction_prompt_sha256),
        (profile.coordinator_prompt_sha256, runtime.coordinator_prompt_sha256),
        (profile.narrative_prompt_sha256, runtime.narrative_prompt_sha256),
        (profile.chat_prompt_sha256, runtime.chat_prompt_sha256),
        (profile.extraction_schema_sha256, runtime.extraction_schema_sha256),
        (profile.coordinator_schema_sha256, runtime.coordinator_schema_sha256),
        (profile.narrative_schema_sha256, runtime.narrative_schema_sha256),
        (profile.chat_schema_sha256, runtime.chat_schema_sha256),
        (
            profile.coordinator_semantic_schema_sha256,
            agentic_semantic_schema_hashes()["coordinator"],
        ),
        (
            profile.narrative_semantic_schema_sha256,
            agentic_semantic_schema_hashes()["narrative"],
        ),
        (
            profile.chat_semantic_schema_sha256,
            agentic_semantic_schema_hashes()["chat"],
        ),
        (profile.request_schema_sha256, schema_sha256("backend", "v3", "review-request")),
        (profile.result_schema_sha256, runtime.result_schema_sha256),
        (profile.policy_id, runtime.policy_id),
        (profile.tool_registry_id, runtime.tool_registry_id),
        (profile.profile_sha256, runtime.release_profile_sha256),
        (profile.review_subject_sha256, runtime.release_profile_subject_sha256),
        (
            profile.evaluation_report_sha256,
            runtime.release_profile_evaluation_report_sha256,
        ),
        (profile.reviewer_trust_store_sha256, runtime.reviewer_trust_store_sha256),
        (profile.assertion_trust_store_sha256, runtime.assertion_trust_store_sha256),
        (assertion_store.trust_store_sha256, runtime.assertion_trust_store_sha256),
        (assertion_store.release_state, runtime.assertion_trust_store_state),
        (evidence_v3.runtime_release_v3_sha256, runtime.evidence_runtime_v3_sha256),
        (base.knowledge_snapshot_id, runtime.evidence_snapshot_id),
    )
    if any(expected != observed for expected, observed in expected_observed):
        raise ProductionReleaseAdmissionError("agent_runtime_release_identity_mismatch")
    return runtime


__all__ = [
    "ProductionReleaseAdmissionError",
    "assert_production_release_admission",
]
