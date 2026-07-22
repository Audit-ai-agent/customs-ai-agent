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
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import dotenv_values
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from customs_ai_agent.adapters.llm import ClaudeDocumentAnalyzer, ClaudeSettings
from customs_ai_agent.contracts.evidence_v3 import EvidenceRuntimeReleaseIdentityV3
from customs_ai_agent.runtime import AgentRuntime, RuntimeInputError
from customs_ai_agent.schemas.v3 import (
    ReviewRevisionResultV1,
    ReviewRunResultV3,
    review_result_v3_sha256,
)
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.controller import ServiceRuntime
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings
from tools.agent_execution_inspector.evidence_v3_conformance import (
    LoopbackEvidenceV3ConformanceRetriever,
)


LOCAL_UAT_ACKNOWLEDGEMENT = "LOCAL_UAT_DIAGNOSTIC_NOT_PRODUCTION"
_MANIFEST_SCHEMA_VERSION = "customs-agent-local-uat-manifest/v1"
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
    }
)
_EVIDENCE_MODES = frozenset({"synthetic-v3"})
_MAX_MANIFEST_BYTES = 1024 * 1024
_LOCAL_UAT_REASON = "local_uat_non_release"
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
    evidence_mode: Literal["synthetic-v3"]
    evidence_base_url: str
    evidence_runtime_identity: EvidenceRuntimeReleaseIdentityV3
    manifest_sha256: str

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "evidence_mode": self.evidence_mode,
            "evidence_base_url": self.evidence_base_url,
            "evidence_runtime_identity": self.evidence_runtime_identity.to_wire(),
            "diagnostic_only": True,
            "text_only": True,
            "mock_fallback_allowed": False,
            "auto_completion_allowed": False,
        }


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
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise LocalUatConfigurationError("uat_manifest_shape_invalid")
    if (
        payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION
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
    return LocalUatManifest(
        evidence_mode=payload["evidence_mode"],
        evidence_base_url=base_url,
        evidence_runtime_identity=identity,
        manifest_sha256=hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
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
    if (
        configured_environment not in {"development", "test"}
        or configured_host != "127.0.0.1"
        or configured_concurrency != "1"
        or configured_evidence_url != manifest.evidence_base_url
        or configured_origins != {manifest.evidence_base_url}
        or configured_token_header.casefold() != "authorization"
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
    ):
        raise LocalUatConfigurationError("uat_service_boundary_invalid")
    return settings


def _diagnostic_result(result: ReviewRunResultV3) -> ReviewRunResultV3:
    """Make non-Production assurance and automatic completion structural."""

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
        # Keep the inherited V2 compatibility fields semantically aligned so
        # older Backend/UI projections cannot mistake a diagnostic result for
        # a completed review merely by ignoring the V3 disposition.
        payload["status"] = "needs_review"
        payload["requires_human_review"] = True
        payload["report"] = dict(payload["report"])
        payload["report"]["overall_status"] = "needs_review"
    payload["result_sha256"] = "0" * 64
    payload["result_sha256"] = review_result_v3_sha256(payload)
    return ReviewRunResultV3.model_validate(payload)


class LocalUatDiagnosticRuntime:
    """Actual Claude + explicit diagnostic Evidence runtime with no Mock path."""

    def __init__(
        self,
        core: AgentRuntime,
        retriever: LoopbackEvidenceV3ConformanceRetriever,
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
        retriever = LoopbackEvidenceV3ConformanceRetriever(
            manifest.evidence_base_url,
            conformance_identity=manifest.evidence_runtime_identity,
            service_token=evidence_service_token,
            acknowledge_non_release=True,
            timeout_seconds=30,
            max_retries=0,
        )
    except Exception:
        analyzer.close()
        raise
    core = object.__new__(AgentRuntime)
    core._analyzer = analyzer
    core._closed = False
    core._config = None
    core._retriever = None
    core._retriever_v3 = retriever
    core._assertion_verifier = None
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
            "planner_mode": "shadow",
            "active_execution_authorized_by_release": False,
            "effective_execution_mode": "shadow",
        }
    )
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
    app.state.build_metadata = _uat_metadata(
        app.state.build_metadata,
        manifest,
        model_id=model_id,
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
