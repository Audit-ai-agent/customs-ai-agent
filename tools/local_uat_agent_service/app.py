"""Loopback-only, non-release Agent service for local user UAT.

This composition root deliberately lives under ``tools``.  It is absent from
the production wheel and never calls the Production ``AgentRuntime``
constructor or release-admission function.  The caller must provide a strict
manifest containing the exact diagnostic Evidence V3 identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from customs_ai_agent.adapters.llm import ClaudeDocumentAnalyzer, ClaudeSettings
from customs_ai_agent.assertion_trust import (
    AssertionTrustStoreV1,
    SupportingDocumentAssertionVerifier,
)
from customs_ai_agent.contracts.evidence_v3 import EvidenceRuntimeReleaseIdentityV3
from customs_ai_agent.runtime import (
    AgentRuntime,
    RuntimeConfigurationError,
    RuntimeExecutionError,
    RuntimeInputError,
)
from customs_ai_agent.experience import build_review_experience
from customs_ai_agent.schemas.v3 import (
    ReviewRevisionResultV1,
    ReviewRunResultV3,
    review_result_v3_sha256,
)
from customs_ai_agent.schemas.experience import AgentUatCapabilityAttestationV1
from customs_ai_agent.uat_capability import UatCapabilityError, load_uat_capability
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.controller import ServiceRuntime
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings
from tools.agent_execution_inspector.evidence_v3_conformance import (
    LoopbackEvidenceV3DiagnosticRetriever,
)


LOCAL_UAT_ACKNOWLEDGEMENT = "LOCAL_UAT_DIAGNOSTIC_NOT_PRODUCTION"
_MANIFEST_SCHEMA_VERSION = "customs-agent-local-uat-manifest/v2"
_MANIFEST_SCHEMA_VERSION_V3 = "customs-agent-local-uat-manifest/v3"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "acknowledgement",
        "diagnostic_only",
        "text_only",
        "mock_fallback_allowed",
        "auto_completion_allowed",
        "evidence_mode",
        "evidence_base_url",
        "evidence_runtime_identity",
        "assertion_fixture_mode",
        "assertion_trust_store_path",
        "assertion_trust_store_sha256",
    }
)
_MANIFEST_V3_FIELDS = frozenset(
    {
        *_MANIFEST_FIELDS,
        "enhanced_experience_enabled",
        "uat_adaptive_active",
        "uat_capability_profile_path",
        "uat_capability_profile_sha256",
        "agent_commit",
    }
)
EvidenceMode = Literal[
    "synthetic-v3",
    "preserved-3law-v3-diagnostic",
    "preserved-3law-v3-scenario-eval",
]


@dataclass(frozen=True, slots=True)
class _EvidencePolicyIdentityPin:
    requirement_policy_release: str
    requirement_policy_manifest_sha256: str
    coverage_matrix_sha256: str
    reviewer_trust_store_sha256: str | None = None


_EVIDENCE_MODES = frozenset(
    {
        "synthetic-v3",
        "preserved-3law-v3-diagnostic",
        "preserved-3law-v3-scenario-eval",
    }
)
_ACTUAL_STORE_EVIDENCE_MODES = frozenset(
    {
        "preserved-3law-v3-diagnostic",
        "preserved-3law-v3-scenario-eval",
    }
)
_EVIDENCE_POLICY_MODES: Mapping[EvidenceMode, str] = {
    "synthetic-v3": "SYNTHETIC_FIXTURE",
    "preserved-3law-v3-diagnostic": "ACTUAL_CODES_HUMAN_ONLY",
    "preserved-3law-v3-scenario-eval": "SEALED_TEST_ONLY_ACTUAL_CCCT_HUMAN_ONLY",
}
_EVIDENCE_POLICY_IDENTITY_PINS: Mapping[EvidenceMode, _EvidencePolicyIdentityPin] = {
    "synthetic-v3": _EvidencePolicyIdentityPin(
        requirement_policy_release="test-only-conformance.no-production-policy.v1",
        requirement_policy_manifest_sha256=(
            "882d18df864cf19149589705c27e9d94b8441e80c8d6bef397a7507cf67a9c90"
        ),
        coverage_matrix_sha256=("a47fd7e683b7e0a270c5d4b76365034b2f24a0b4f2a52ba1c69421480c1f5263"),
    ),
    "preserved-3law-v3-diagnostic": _EvidencePolicyIdentityPin(
        requirement_policy_release="test-only-conformance.no-production-policy.v1",
        requirement_policy_manifest_sha256=(
            "54aa31a0baed9a8eeeea57f9c436a78b8d687e244538491a46e34af46238cf7a"
        ),
        coverage_matrix_sha256=("a47fd7e683b7e0a270c5d4b76365034b2f24a0b4f2a52ba1c69421480c1f5263"),
    ),
    "preserved-3law-v3-scenario-eval": _EvidencePolicyIdentityPin(
        requirement_policy_release="test-only-three-law-scenario-eval-human-only.v1",
        requirement_policy_manifest_sha256=(
            "2467226fe64c84f8fdc0b41a8ab44ebaba04e6ba7280cd3e4bd56bd2e9c38c95"
        ),
        coverage_matrix_sha256=("21012116b4fddf0ae0e0bb12343558aca472373480d32d535e6b59875800c444"),
        reviewer_trust_store_sha256=(
            "7864482157b701b44c457bd35c7721b31ffc31564ea43d83948aa27e08233637"
        ),
    ),
}
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_ASSERTION_TRUST_STORE_BYTES = 1024 * 1024
_LOCAL_UAT_REASON = "local_uat_non_release"
_ASSERTION_FIXTURE_MODE = "LOOPBACK_TEST_ONLY_HUMAN_APPROVER"
_UAT_APPROVER_AUTHORITY_ID = (
    "backend:principal-sha256:6cc4551495f1ec8da505fdeaa4f34c17eeb8c234bcaeb6ca184f92a0cdd6318d"
)
_THREE_LAW_EVIDENCE_HTTP_TIMEOUT_SECONDS = 195.0
_SERVICE_ENV_KEYS = frozenset(
    {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN",
        "CUSTOMS_AGENT_ENVIRONMENT",
        "CUSTOMS_AGENT_BUILD_SHA",
        "CUSTOMS_AGENT_WHEEL_SHA256",
        "CUSTOMS_AGENT_HOST",
        "CUSTOMS_AGENT_MAX_CONCURRENCY",
        "EVIDENCE_SERVICE_TOKEN",
        "EVIDENCE_SERVICE_BASE_URL",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS",
        "EVIDENCE_SERVICE_TOKEN_HEADER",
        "CUSTOMS_AGENT_PORT",
        "CUSTOMS_AGENT_MAX_BODY_BYTES",
        "CUSTOMS_AGENT_MAX_DOCUMENTS",
        "CUSTOMS_AGENT_MAX_TEXT_CHARACTERS",
        "CUSTOMS_AGENT_BODY_READ_TIMEOUT_SECONDS",
        "CUSTOMS_AGENT_QUEUE_WAIT_SECONDS",
        "CUSTOMS_AGENT_RUN_TIMEOUT_SECONDS",
        "CUSTOMS_AGENT_REPLAY_TTL_SECONDS",
        "CUSTOMS_AGENT_REPLAY_MAX_ENTRIES",
        "CUSTOMS_AGENT_REPLAY_MAX_BYTES",
        "CUSTOMS_AGENT_READINESS_TTL_SECONDS",
        "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS",
        "CUSTOMS_AGENT_SHUTDOWN_GRACE_SECONDS",
        "CUSTOMS_AGENT_LOG_LEVEL",
        "CUSTOMS_ENHANCED_AGENT_EXPERIENCE",
        "CUSTOMS_UAT_ADAPTIVE_ACTIVE",
        "CUSTOMS_AGENT_UAT_CAPABILITY_PATH",
    }
)


class LocalUatConfigurationError(RuntimeError):
    """Content-free startup error safe to surface to a local operator."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey
        value[key] = item
    return value


def _reject_nonfinite(_value: str) -> Any:
    raise ValueError


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _literal_loopback_origin(value: object) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise LocalUatConfigurationError("uat_evidence_base_url_invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise LocalUatConfigurationError("uat_evidence_base_url_invalid") from None
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise LocalUatConfigurationError("uat_evidence_base_url_not_loopback")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class LocalUatManifest:
    evidence_mode: EvidenceMode
    evidence_base_url: str
    evidence_runtime_identity: EvidenceRuntimeReleaseIdentityV3
    assertion_fixture_mode: Literal[
        "DISABLED",
        "LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
    ]
    assertion_trust_store: AssertionTrustStoreV1 | None
    manifest_sha256: str
    enhanced_experience_enabled: bool = False
    uat_adaptive_active: bool = False
    uat_capability_profile_path: str | None = None
    uat_capability_profile_sha256: str | None = None
    agent_commit: str | None = None

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": (
                _MANIFEST_SCHEMA_VERSION_V3
                if self.enhanced_experience_enabled
                else _MANIFEST_SCHEMA_VERSION
            ),
            "manifest_sha256": self.manifest_sha256,
            "evidence_mode": self.evidence_mode,
            "evidence_policy_mode": _EVIDENCE_POLICY_MODES[self.evidence_mode],
            "evidence_profile": _evidence_profile_metadata(self.evidence_mode),
            "evidence_base_url": self.evidence_base_url,
            "evidence_runtime_identity": self.evidence_runtime_identity.to_wire(),
            "assertion_fixture_mode": self.assertion_fixture_mode,
            "assertion_trust_store_sha256": (
                self.assertion_trust_store.trust_store_sha256
                if self.assertion_trust_store is not None
                else None
            ),
            "trusted_assertion_verifier_configured": self.assertion_trust_store is not None,
            "diagnostic_only": True,
            "text_only": True,
            "mock_fallback_allowed": False,
            "auto_completion_allowed": False,
            "enhanced_experience_enabled": self.enhanced_experience_enabled,
            "uat_adaptive_active": self.uat_adaptive_active,
            "uat_capability_file_sha256": self.uat_capability_profile_sha256,
        }


def _evidence_profile_metadata(mode: EvidenceMode) -> dict[str, Any]:
    """Return non-secret UI hints without asserting provider internals.

    A V3 response proves the exact provider/runtime identity and preserves
    Evidence provenance.  It does not prove that the provider opened any
    particular database.  Store counts and execution claims therefore remain
    the Backend/harness's responsibility after independent corroboration.
    """

    profile_kinds = {
        "synthetic-v3": "synthetic_contract_control",
        "preserved-3law-v3-diagnostic": "preserved_three_law_rag_diagnostic",
        "preserved-3law-v3-scenario-eval": "preserved_three_law_rag_scenario_evaluation",
    }
    return {
        "profile_id": mode,
        "profile_kind": profile_kinds[mode],
        "transport": "evidence_http_v3",
        "identity_binding": "exact_readiness_and_response_headers",
        "direct_database_access": False,
        "mock_fallback_allowed": False,
        "release_eligible": False,
        "provider_execution_corroboration": "backend_harness_required",
        "rag_quality_status": "not_attested_by_agent",
        "result_provenance_paths": [
            "knowledge_snapshot_id",
            "rag_evidence[].source_record_id",
            "rag_evidence[].source_uri",
            "rag_evidence[].version_hash",
            "rag_evidence[].retrieval_methods",
            "rag_evidence[].score",
            "rag_evidence[].raw_scores",
        ],
    }


def _evidence_timeout_seconds(mode: EvidenceMode) -> float:
    # The actual-store provider performs fail-closed full-store attestation
    # before and after retrieval under its own 180-second service deadline.
    # Its HTTP consumer must have a strictly larger budget so a valid terminal
    # response is not discarded at the same deadline boundary.
    return (
        _THREE_LAW_EVIDENCE_HTTP_TIMEOUT_SECONDS if mode in _ACTUAL_STORE_EVIDENCE_MODES else 30.0
    )


def _readiness_policy(mode: EvidenceMode) -> tuple[float, float]:
    # The actual Data request itself has a 75-second hard deadline. Give that
    # request enough time to finish and cache the result for the maximum
    # bounded TTL; otherwise a 30-second controller timeout leaves the
    # single-threaded probe running and every retry observes SERVICE_BUSY.
    return (60.0, 90.0) if mode in _ACTUAL_STORE_EVIDENCE_MODES else (5.0, 5.0)


def _assert_evidence_policy_identity(
    mode: EvidenceMode,
    identity: EvidenceRuntimeReleaseIdentityV3,
) -> None:
    """Bind the operator-selected profile to an independently pinned policy identity."""

    pin = _EVIDENCE_POLICY_IDENTITY_PINS[mode]
    if (
        identity.requirement_policy_release != pin.requirement_policy_release
        or identity.requirement_policy_manifest_sha256 != pin.requirement_policy_manifest_sha256
        or identity.coverage_matrix_sha256 != pin.coverage_matrix_sha256
        or (
            pin.reviewer_trust_store_sha256 is not None
            and identity.reviewer_trust_store_sha256 != pin.reviewer_trust_store_sha256
        )
    ):
        raise LocalUatConfigurationError("uat_evidence_policy_identity_mismatch")


def _is_local_uat_secret_path(path: Path) -> bool:
    parts = tuple(part.casefold() for part in path.parts)
    return any(
        parts[index : index + 2] == ("_local", "secrets") for index in range(max(0, len(parts) - 1))
    )


def _load_uat_assertion_trust_store(
    path_value: object,
    expected_sha256: object,
) -> AssertionTrustStoreV1:
    if (
        not isinstance(path_value, str)
        or path_value != path_value.strip()
        or not path_value
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise LocalUatConfigurationError("uat_assertion_trust_store_config_invalid")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        raise LocalUatConfigurationError("uat_assertion_trust_store_path_invalid")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not _is_local_uat_secret_path(resolved):
            raise LocalUatConfigurationError("uat_assertion_trust_store_path_invalid")
        raw = resolved.read_bytes()
    except LocalUatConfigurationError:
        raise
    except OSError:
        raise LocalUatConfigurationError("uat_assertion_trust_store_unavailable") from None
    if not raw or len(raw) > _MAX_ASSERTION_TRUST_STORE_BYTES:
        raise LocalUatConfigurationError("uat_assertion_trust_store_size_invalid")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
        store = AssertionTrustStoreV1.model_validate(payload)
    except (UnicodeError, ValueError, RecursionError):
        raise LocalUatConfigurationError("uat_assertion_trust_store_invalid") from None
    active_keys = [item for item in store.keys if item.status == "active"]
    today = date.today()
    if (
        store.release_state != "active"
        or store.trust_store_sha256 != expected_sha256
        or len(store.keys) != 1
        or len(active_keys) != 1
        or active_keys[0].authority_id != _UAT_APPROVER_AUTHORITY_ID
        or active_keys[0].method != "human_approver"
        or not active_keys[0].key_id.startswith("local-uat-")
        or not active_keys[0].valid_from <= today <= active_keys[0].valid_until
    ):
        raise LocalUatConfigurationError("uat_assertion_trust_store_policy_invalid")
    return store


def load_local_uat_manifest(path: str | os.PathLike[str]) -> LocalUatManifest:
    manifest_path = Path(path).expanduser()
    try:
        raw = manifest_path.read_bytes()
    except OSError:
        raise LocalUatConfigurationError("uat_manifest_unavailable") from None
    if not raw or len(raw) > _MAX_MANIFEST_BYTES:
        raise LocalUatConfigurationError("uat_manifest_size_invalid")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, ValueError, RecursionError):
        raise LocalUatConfigurationError("uat_manifest_invalid") from None
    if not isinstance(payload, dict):
        raise LocalUatConfigurationError("uat_manifest_shape_invalid")
    schema_version = payload.get("schema_version")
    expected_fields = (
        _MANIFEST_V3_FIELDS if schema_version == _MANIFEST_SCHEMA_VERSION_V3 else _MANIFEST_FIELDS
    )
    if set(payload) != expected_fields:
        raise LocalUatConfigurationError("uat_manifest_shape_invalid")
    if (
        schema_version not in {_MANIFEST_SCHEMA_VERSION, _MANIFEST_SCHEMA_VERSION_V3}
        or payload.get("acknowledgement") != LOCAL_UAT_ACKNOWLEDGEMENT
        or payload.get("diagnostic_only") is not True
        or payload.get("text_only") is not True
        or payload.get("mock_fallback_allowed") is not False
        or payload.get("auto_completion_allowed") is not False
        or payload.get("evidence_mode") not in _EVIDENCE_MODES
    ):
        raise LocalUatConfigurationError("uat_manifest_policy_invalid")
    base_url = _literal_loopback_origin(payload.get("evidence_base_url"))
    try:
        identity = EvidenceRuntimeReleaseIdentityV3.from_mapping(
            payload.get("evidence_runtime_identity")
        )
    except (TypeError, ValueError):
        raise LocalUatConfigurationError("uat_evidence_identity_invalid") from None
    _assert_evidence_policy_identity(payload["evidence_mode"], identity)
    scenario_eval = payload["evidence_mode"] == "preserved-3law-v3-scenario-eval"
    fixture_mode = payload.get("assertion_fixture_mode")
    if scenario_eval:
        if fixture_mode != _ASSERTION_FIXTURE_MODE:
            raise LocalUatConfigurationError("uat_assertion_fixture_policy_invalid")
        assertion_trust_store = _load_uat_assertion_trust_store(
            payload.get("assertion_trust_store_path"),
            payload.get("assertion_trust_store_sha256"),
        )
    else:
        if (
            fixture_mode != "DISABLED"
            or payload.get("assertion_trust_store_path") is not None
            or payload.get("assertion_trust_store_sha256") is not None
        ):
            raise LocalUatConfigurationError("uat_assertion_fixture_policy_invalid")
        assertion_trust_store = None
    enhanced_experience_enabled = schema_version == _MANIFEST_SCHEMA_VERSION_V3 and (
        payload.get("enhanced_experience_enabled") is True
    )
    uat_adaptive_active = schema_version == _MANIFEST_SCHEMA_VERSION_V3 and (
        payload.get("uat_adaptive_active") is True
    )
    capability_path_value = payload.get("uat_capability_profile_path")
    capability_sha_value = payload.get("uat_capability_profile_sha256")
    agent_commit_value = payload.get("agent_commit")
    if schema_version == _MANIFEST_SCHEMA_VERSION_V3:
        if (
            not enhanced_experience_enabled
            or not scenario_eval
            or not isinstance(capability_path_value, str)
            or not isinstance(capability_sha_value, str)
            or len(capability_sha_value) != 64
            or any(character not in "0123456789abcdef" for character in capability_sha_value)
            or not isinstance(agent_commit_value, str)
            or len(agent_commit_value) != 40
            or any(character not in "0123456789abcdef" for character in agent_commit_value)
        ):
            raise LocalUatConfigurationError("uat_capability_manifest_policy_invalid")
        capability_path = Path(capability_path_value)
        if not capability_path.is_absolute() or capability_path.is_symlink():
            raise LocalUatConfigurationError("uat_capability_path_invalid")
        try:
            resolved_capability_path = capability_path.resolve(strict=True)
        except OSError:
            raise LocalUatConfigurationError("uat_capability_unavailable") from None
        if not resolved_capability_path.is_file() or not _is_local_uat_secret_path(
            resolved_capability_path
        ):
            raise LocalUatConfigurationError("uat_capability_path_invalid")
        capability_path_value = str(resolved_capability_path)
    elif any(
        value is not None
        for value in (
            capability_path_value,
            capability_sha_value,
            agent_commit_value,
        )
    ):
        raise LocalUatConfigurationError("uat_capability_manifest_policy_invalid")
    return LocalUatManifest(
        evidence_mode=payload["evidence_mode"],
        evidence_base_url=base_url,
        evidence_runtime_identity=identity,
        assertion_fixture_mode=fixture_mode,
        assertion_trust_store=assertion_trust_store,
        manifest_sha256=hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        enhanced_experience_enabled=enhanced_experience_enabled,
        uat_adaptive_active=uat_adaptive_active,
        uat_capability_profile_path=capability_path_value,
        uat_capability_profile_sha256=capability_sha_value,
        agent_commit=agent_commit_value,
    )


def _environment_values(env_file: str | os.PathLike[str]) -> dict[str, str]:
    path = Path(env_file).expanduser()
    if not path.is_file():
        raise LocalUatConfigurationError("uat_env_file_unavailable")
    values = dotenv_values(path, verbose=False)
    result: dict[str, str] = {}
    for name in _SERVICE_ENV_KEYS:
        process_value = os.environ.get(name)
        file_value = values.get(name)
        value = process_value if process_value is not None else file_value
        if isinstance(value, str):
            result[name] = value
    return result


def local_uat_service_settings(
    manifest: LocalUatManifest,
    env_file: str | os.PathLike[str],
) -> ServiceSettings:
    source = _environment_values(env_file)
    configured_environment = source.get("CUSTOMS_AGENT_ENVIRONMENT", "test").strip().casefold()
    configured_host = source.get("CUSTOMS_AGENT_HOST", "127.0.0.1").strip()
    configured_concurrency = source.get("CUSTOMS_AGENT_MAX_CONCURRENCY", "1").strip()
    configured_evidence_url = (
        source.get("EVIDENCE_SERVICE_BASE_URL", manifest.evidence_base_url).strip().rstrip("/")
    )
    configured_origins = {
        item.strip().rstrip("/")
        for item in source.get(
            "EVIDENCE_SERVICE_ALLOWED_ORIGINS", manifest.evidence_base_url
        ).split(",")
        if item.strip()
    }
    configured_token_header = source.get("EVIDENCE_SERVICE_TOKEN_HEADER", "Authorization").strip()
    readiness_ttl, readiness_timeout = _readiness_policy(manifest.evidence_mode)
    configured_readiness_ttl = source.get(
        "CUSTOMS_AGENT_READINESS_TTL_SECONDS",
        str(readiness_ttl),
    ).strip()
    configured_readiness_timeout = source.get(
        "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS",
        str(readiness_timeout),
    ).strip()
    try:
        readiness_conflict = (
            float(configured_readiness_ttl) != readiness_ttl
            or float(configured_readiness_timeout) != readiness_timeout
        )
    except ValueError:
        readiness_conflict = True
    if (
        configured_environment not in {"development", "test"}
        or configured_host != "127.0.0.1"
        or configured_concurrency != "1"
        or configured_evidence_url != manifest.evidence_base_url
        or configured_origins != {manifest.evidence_base_url}
        or configured_token_header.casefold() != "authorization"
        or readiness_conflict
    ):
        raise LocalUatConfigurationError("uat_env_boundary_conflict")
    source.update(
        {
            "CUSTOMS_AGENT_ENVIRONMENT": "test",
            "CUSTOMS_AGENT_HOST": "127.0.0.1",
            "CUSTOMS_AGENT_MAX_CONCURRENCY": "1",
            "EVIDENCE_SERVICE_BASE_URL": manifest.evidence_base_url,
            "EVIDENCE_SERVICE_ALLOWED_ORIGINS": manifest.evidence_base_url,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
            "CUSTOMS_AGENT_READINESS_TTL_SECONDS": str(readiness_ttl),
            "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS": str(readiness_timeout),
            "CUSTOMS_ENHANCED_AGENT_EXPERIENCE": (
                "true" if manifest.enhanced_experience_enabled else "false"
            ),
            "CUSTOMS_UAT_ADAPTIVE_ACTIVE": ("true" if manifest.uat_adaptive_active else "false"),
        }
    )
    try:
        settings = ServiceSettings.from_environment(source)
    except ServiceConfigurationError as exc:
        raise LocalUatConfigurationError(exc.code) from None
    if (
        settings.environment != "test"
        or settings.host != "127.0.0.1"
        or settings.max_concurrency != 1
        or settings.run_timeout_seconds <= _evidence_timeout_seconds(manifest.evidence_mode)
        or settings.enhanced_agent_experience != manifest.enhanced_experience_enabled
        or settings.uat_adaptive_active != manifest.uat_adaptive_active
    ):
        raise LocalUatConfigurationError("uat_service_boundary_invalid")
    return settings


def _diagnostic_result(result: ReviewRunResultV3) -> ReviewRunResultV3:
    """Make non-Production assurance and automatic completion structural."""

    prior_reasons = set(result.completion_eligibility.unmet_reason_codes)
    if result.completion_eligibility.eligible or prior_reasons == {
        "narrative_release_identity_incomplete"
    }:
        # Reframing a release-approved narrative after generation would require
        # recompiling its canonical claims, trace outputs, and result lineage.
        # This diagnostic adapter has no such authority and therefore refuses
        # that state rather than emitting a self-contradictory result.
        raise RuntimeExecutionError(
            "uat_release_result_forbidden",
            "LOCAL UAT accepts only an already non-release review result.",
        )
    payload = result.model_dump(mode="python", warnings=False)
    eligibility = dict(payload["completion_eligibility"])
    eligibility.update(
        {
            "eligible": False,
            "profile_id": None,
            "unmet_reason_codes": sorted(
                {*eligibility.get("unmet_reason_codes", []), _LOCAL_UAT_REASON}
            ),
        }
    )
    payload["completion_eligibility"] = eligibility
    payload["assurance_level"] = "diagnostic"
    if payload["review_disposition"] != "blocked":
        payload["review_disposition"] = "human_review_required"
        # V1/V2 fields are an immutable provider-owned review projection.  A
        # diagnostic overlay must not rewrite those facts after the canonical
        # FactLedger, Gap plan, action trace, and narrative were built.  V3's
        # assurance, eligibility, and disposition are the sole authorization
        # boundary consumed by the LOCAL UAT workspace.
    payload["result_sha256"] = "0" * 64
    payload["result_sha256"] = review_result_v3_sha256(payload)
    return ReviewRunResultV3.model_validate(payload)


class LocalUatDiagnosticRuntime:
    """Actual Claude + explicit diagnostic Evidence runtime with no Mock path."""

    def __init__(
        self,
        core: AgentRuntime,
        retriever: LoopbackEvidenceV3DiagnosticRetriever,
        *,
        model_id: str,
    ) -> None:
        self._core = core
        self._retriever = retriever
        self.model_id = model_id

    def review_json(self, _payload: bytes) -> Any:
        raise RuntimeInputError(
            "uat_v1_disabled",
            "LOCAL UAT exposes only the V3 Agent contract through service API v2.",
        )

    def review_v3_json(self, payload: bytes) -> ReviewRunResultV3:
        return _diagnostic_result(self._core.review_v3_json(payload))

    def revision_v3_json(self, payload: bytes) -> ReviewRevisionResultV1:
        result = self._core.revision_v3_json(payload)
        return ReviewRevisionResultV1(
            parent_result_sha256=result.parent_result_sha256,
            child_result=_diagnostic_result(result.child_result),
            applied_decision_ids=result.applied_decision_ids,
        )

    def answer_v3_json(self, payload: bytes) -> Any:
        return self._core.answer_v3_json(payload)

    def assert_uat_capability_current(
        self,
        *,
        require_active: bool = False,
    ) -> AgentUatCapabilityAttestationV1:
        return self._core.assert_uat_capability_current(
            require_active=require_active,
        )

    def review_experience_json(self, payload: bytes) -> Any:
        capability = self.assert_uat_capability_current()
        result = _diagnostic_result(self._core.review_v3_json(payload))
        return build_review_experience(
            result,
            analyzer=self._core._analyzer,
            capability=capability,
        )

    def answer_v2_json(self, payload: bytes) -> Any:
        return self._core.answer_v2_json(payload)

    def review_experience_stream_json(
        self,
        payload: bytes,
        stage_callback: Callable[[dict[str, Any]], None],
    ) -> Any:
        return self._core.review_experience_stream_json(
            payload,
            stage_callback,
            _result_transform=_diagnostic_result,
        )

    def answer_v2_stream_json(
        self,
        payload: bytes,
        stage_callback: Callable[[dict[str, Any]], None],
    ) -> Any:
        return self._core.answer_v2_stream_json(payload, stage_callback)

    @property
    def uat_capability(self) -> AgentUatCapabilityAttestationV1 | None:
        return getattr(self._core, "_uat_capability", None)

    def preflight(self) -> object:
        return self._retriever.preflight()

    def close(self) -> None:
        self._core.close()


def build_local_uat_runtime(
    manifest: LocalUatManifest,
    env_file: str | os.PathLike[str],
    *,
    evidence_service_token: str,
) -> LocalUatDiagnosticRuntime:
    """Construct the only real LOCAL UAT runtime composition.

    The Production constructor is intentionally not called: it must continue
    to reject the packaged unreleased locks.  Dependencies are concrete,
    caller-explicit, and loopback-bound; there is no Mock or fallback branch.
    """

    claude_settings = ClaudeSettings.from_environment(env_file)
    analyzer = ClaudeDocumentAnalyzer.from_settings(claude_settings)
    try:
        retriever = LoopbackEvidenceV3DiagnosticRetriever(
            manifest.evidence_base_url,
            conformance_identity=manifest.evidence_runtime_identity,
            service_token=evidence_service_token,
            acknowledge_non_release=True,
            timeout_seconds=_evidence_timeout_seconds(manifest.evidence_mode),
            max_retries=0,
        )
    except Exception:
        analyzer.close()
        raise
    core = object.__new__(AgentRuntime)
    core._analyzer = analyzer
    core._closed = False
    core._config = None  # type: ignore[assignment]
    core._retriever = None  # type: ignore[assignment]
    core._retriever_v3 = retriever
    core._assertion_verifier = (
        SupportingDocumentAssertionVerifier(manifest.assertion_trust_store)
        if manifest.assertion_trust_store is not None
        else None
    )
    core._uat_expected_agent_commit = manifest.agent_commit
    capability: AgentUatCapabilityAttestationV1 | None = None
    if manifest.enhanced_experience_enabled:
        if (
            manifest.uat_capability_profile_path is None
            or manifest.uat_capability_profile_sha256 is None
            or manifest.agent_commit is None
        ):
            analyzer.close()
            raise LocalUatConfigurationError("uat_capability_manifest_policy_invalid")
        try:
            capability = load_uat_capability(
                manifest.uat_capability_profile_path,
                expected_file_sha256=manifest.uat_capability_profile_sha256,
                expected_agent_commit=manifest.agent_commit,
                expected_model_id=claude_settings.model,
                expected_evidence_runtime_v3_sha256=(
                    manifest.evidence_runtime_identity.runtime_release_v3_sha256
                ),
                require_active=manifest.uat_adaptive_active,
            )
        except UatCapabilityError as exc:
            analyzer.close()
            raise LocalUatConfigurationError(exc.code) from None
    core._uat_capability = capability
    core._secret_markers = (
        claude_settings.api_key.get_secret_value(),
        evidence_service_token,
    )
    return LocalUatDiagnosticRuntime(
        core,
        retriever,
        model_id=claude_settings.model,
    )


def _uat_metadata(
    base: Mapping[str, Any],
    manifest: LocalUatManifest,
    *,
    model_id: str,
    capability: AgentUatCapabilityAttestationV1 | None = None,
) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(base))
    metadata["status"] = "ready"
    metadata["service_mode"] = "local_user_uat_diagnostic"
    metadata["assurance_level"] = "diagnostic"
    metadata["uat"] = manifest.safe_metadata()
    metadata["model"]["id"] = model_id
    metadata["evidence_runtime_release_v3"] = {
        "release_state": "diagnostic",
        "identity": manifest.evidence_runtime_identity.to_wire(),
    }
    metadata["agentic_limits"].update(
        {
            "planner_mode": ("active" if manifest.uat_adaptive_active else "shadow"),
            "active_execution_authorized_by_release": False,
            "effective_execution_mode": (
                "uat_active" if manifest.uat_adaptive_active else "shadow"
            ),
            "uat_adaptive_active": manifest.uat_adaptive_active,
        }
    )
    metadata["enhanced_experience_enabled"] = manifest.enhanced_experience_enabled
    metadata["uat_capability_status"] = "valid" if capability is not None else "disabled"
    metadata["uat_capability_sha256"] = (
        capability.capability_sha256 if capability is not None else None
    )
    metadata["uat_capability_file_sha256"] = manifest.uat_capability_profile_sha256
    metadata["uat_capability_expires_at"] = (
        capability.expires_at.isoformat() if capability is not None else None
    )
    metadata["uat_capability_attestation"] = (
        capability.model_dump(mode="json", warnings=False) if capability is not None else None
    )
    metadata["agent_commit"] = manifest.agent_commit
    metadata["automatic_completion_release"]["uat_forced_disabled"] = True
    return metadata


def _boundary_error(code: str) -> JSONResponse:
    request_id = str(uuid4())
    return JSONResponse(
        status_code=503,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Request-ID": request_id,
            "X-Customs-Service-Mode": "local-user-uat-diagnostic",
        },
        content={
            "error": {
                "code": code,
                "message": "Production readiness is unavailable in LOCAL UAT.",
                "request_id": request_id,
                "retryable": False,
            }
        },
    )


def _assert_uat_service_boundary(
    manifest: LocalUatManifest,
    settings: ServiceSettings,
    acknowledgement: object,
) -> None:
    if acknowledgement is not True:
        raise LocalUatConfigurationError("uat_acknowledgement_required")
    if (
        settings.environment == "production"
        or settings.host != "127.0.0.1"
        or settings.max_concurrency != 1
        or settings.evidence_base_url != manifest.evidence_base_url
    ):
        raise LocalUatConfigurationError("uat_service_boundary_invalid")


def _compose_local_uat_app(
    *,
    manifest: LocalUatManifest,
    settings: ServiceSettings,
    acknowledge_local_uat: Literal[True],
    runtime: ServiceRuntime,
    readiness_probe: Callable[[], object],
    model_id: str,
) -> FastAPI:
    """Compose HTTP transport over already explicit UAT dependencies.

    This private seam exists for tests. The public entrypoint below has no
    runtime injection parameter and therefore cannot select a Mock provider.
    """

    _assert_uat_service_boundary(manifest, settings, acknowledge_local_uat)
    if not model_id.strip():
        raise LocalUatConfigurationError("uat_model_identity_missing")

    app = create_app(
        settings,
        runtime_factory=lambda _settings: runtime,
        readiness_probe_factory=lambda _settings: readiness_probe,
    )
    app.title = "Customs AI Agent LOCAL USER UAT Service"
    app.description = (
        "Loopback-only text UAT with actual Claude and an explicitly pinned "
        "diagnostic Evidence V3 provider. Never Production evidence."
    )
    capability = getattr(runtime, "uat_capability", None)
    if capability is not None and not isinstance(
        capability,
        AgentUatCapabilityAttestationV1,
    ):
        raise LocalUatConfigurationError("uat_capability_contract_invalid")
    app.state.build_metadata = _uat_metadata(
        app.state.build_metadata,
        manifest,
        model_id=model_id,
        capability=capability,
    )
    app.state.build_metadata_v3 = _uat_metadata(
        app.state.build_metadata_v3,
        manifest,
        model_id=model_id,
        capability=capability,
    )

    @app.middleware("http")
    async def local_uat_boundary(request: Request, call_next: Callable[..., Any]) -> Any:
        if request.url.path == "/health/ready":
            return _boundary_error("production_readiness_unavailable_in_local_uat")
        if request.url.path.startswith("/internal/agent/v1/"):
            return JSONResponse(
                status_code=404,
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                    "X-Customs-Service-Mode": "local-user-uat-diagnostic",
                },
                content={"detail": "Not Found"},
            )
        response = await call_next(request)
        response.headers.setdefault("X-Customs-Service-Mode", "local-user-uat-diagnostic")
        return response

    @app.get("/health/uat-ready", summary="LOCAL UAT dependency readiness")
    async def uat_ready(request: Request) -> JSONResponse:
        controller = request.app.state.controller
        if manifest.enhanced_experience_enabled:
            try:
                controller.assert_uat_capability_current(require_active=False)
            except RuntimeConfigurationError:
                return _boundary_error("uat_capability_invalid_or_expired")
        if not await controller.ready():
            return _boundary_error("uat_dependency_unready")
        return JSONResponse(
            status_code=200,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "X-Customs-Service-Mode": "local-user-uat-diagnostic",
            },
            content={
                "status": "uat-ready",
                "diagnostic_only": True,
                "text_only": True,
                "mock_fallback_allowed": False,
                "auto_completion_allowed": False,
                "assurance_level": "diagnostic",
                "evidence_mode": manifest.evidence_mode,
                "evidence_policy_mode": _EVIDENCE_POLICY_MODES[manifest.evidence_mode],
                "assertion_fixture_mode": manifest.assertion_fixture_mode,
                "assertion_trust_store_sha256": (
                    manifest.assertion_trust_store.trust_store_sha256
                    if manifest.assertion_trust_store is not None
                    else None
                ),
                "trusted_assertion_verifier_configured": (
                    manifest.assertion_trust_store is not None
                ),
                "enhanced_experience_enabled": manifest.enhanced_experience_enabled,
                "uat_adaptive_active": manifest.uat_adaptive_active,
                "uat_capability_status": ("valid" if capability is not None else "disabled"),
                "uat_capability_sha256": (
                    capability.capability_sha256 if capability is not None else None
                ),
                "uat_capability_file_sha256": manifest.uat_capability_profile_sha256,
                "uat_capability_expires_at": (
                    capability.expires_at.isoformat() if capability is not None else None
                ),
                "uat_capability_attestation": (
                    capability.model_dump(mode="json", warnings=False)
                    if capability is not None
                    else None
                ),
                "agent_commit": manifest.agent_commit,
                "manifest_sha256": manifest.manifest_sha256,
            },
        )

    return app


def create_local_uat_app(
    *,
    manifest: LocalUatManifest,
    settings: ServiceSettings,
    acknowledge_local_uat: Literal[True],
    env_file: str | os.PathLike[str],
) -> FastAPI:
    """Create the actual-Claude, no-Mock LOCAL USER UAT service."""

    _assert_uat_service_boundary(manifest, settings, acknowledge_local_uat)
    if settings.evidence_service_token is None:
        raise LocalUatConfigurationError("uat_runtime_secrets_missing")
    runtime = build_local_uat_runtime(
        manifest,
        env_file,
        evidence_service_token=settings.evidence_service_token.get_secret_value(),
    )
    try:
        return _compose_local_uat_app(
            manifest=manifest,
            settings=settings,
            acknowledge_local_uat=acknowledge_local_uat,
            runtime=runtime,
            readiness_probe=runtime.preflight,
            model_id=runtime.model_id,
        )
    except Exception:
        runtime.close()
        raise


__all__ = [
    "LOCAL_UAT_ACKNOWLEDGEMENT",
    "LocalUatConfigurationError",
    "LocalUatDiagnosticRuntime",
    "LocalUatManifest",
    "build_local_uat_runtime",
    "create_local_uat_app",
    "load_local_uat_manifest",
    "local_uat_service_settings",
]
