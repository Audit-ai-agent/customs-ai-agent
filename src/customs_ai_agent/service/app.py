from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from customs_ai_agent.adapters.retrieval_v3 import fetch_evidence_runtime_readiness_v3
from customs_ai_agent.adapters.llm import DEFAULT_CLAUDE_MODEL
from customs_ai_agent.assertion_trust import SupportingDocumentAssertionVerifier
from customs_ai_agent.contracts import get_schema
from customs_ai_agent.runtime import (
    AgentRuntime,
    RuntimeConfig,
    RuntimeConfigurationError,
)
from customs_ai_agent.service.auth import bearer_authorized
from customs_ai_agent.service.controller import (
    AgentServiceController,
    ServiceRuntime,
)
from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.input import (
    ERROR_RESPONSE_SCHEMA,
    TEXT_REVIEW_REQUEST_SCHEMA,
    inspect_text_request,
    inspect_v3_auxiliary_request,
    raw_header_values,
    read_bounded_json_body,
)
from customs_ai_agent.schemas.v3 import (
    GroundedAnswerV1,
    GroundedQuestionV1,
    ReviewRevisionRequestV1,
    ReviewRevisionResultV1,
)
from customs_ai_agent.schemas.experience import (
    AssistantAnswerV2,
    CaseConversationRequestV2,
    ReviewExperienceEnvelopeV1,
    ReviewRevisionExperienceResultV1,
    ReviewStageEventV1,
)
from customs_ai_agent.schemas.v3 import canonical_json_sha256
from customs_ai_agent.experience import experience_prompt_hashes, experience_schema_hashes
from customs_ai_agent.meaning_catalog import meaning_catalog_sha256
from customs_ai_agent.service.metadata import build_legacy_metadata, build_metadata
from customs_ai_agent.service.release_admission import (
    ProductionReleaseAdmissionError,
    assert_production_release_admission,
)
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.telemetry import safe_log


LOGGER = logging.getLogger("customs_ai_agent.service")
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}
_BEARER_SECURITY: list[dict[str, list[str]]] = [{"ServiceBearer": []}]
_PRE_EXECUTION_503_RESPONSE: dict[str, Any] = {
    "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
    "headers": {
        "X-Agent-Execution-State": {
            "description": (
                "Present only when the authenticated request was rejected by the dependency "
                "admission Gate before controller.submit and runtime execution."
            ),
            "schema": {"type": "string", "enum": ["not-started"]},
        }
    },
}


class _UnavailableRuntime:
    """Keep liveness observable while production release admission is closed."""

    def assert_uat_capability_current(self, *, require_active: bool = False) -> None:
        del require_active
        raise RuntimeConfigurationError(
            "runtime_dependency_unready",
            "The Agent runtime dependency Gate is closed.",
        )

    def review_json(self, _payload: bytes) -> object:
        raise RuntimeConfigurationError(
            "runtime_dependency_unready",
            "The Agent runtime dependency Gate is closed.",
        )

    def close(self) -> None:
        return None

    review_v3_json = review_json
    revision_v3_json = review_json
    revision_experience_v3_json = review_json
    answer_v3_json = review_json
    review_experience_json = review_json
    answer_v2_json = review_json

    def review_experience_stream_json(
        self,
        payload: bytes,
        _stage_callback: Callable[[dict[str, Any]], None],
    ) -> object:
        return self.review_json(payload)

    def answer_v2_stream_json(
        self,
        payload: bytes,
        _stage_callback: Callable[[dict[str, Any]], None],
    ) -> object:
        return self.review_json(payload)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid4())


def _stage_event(
    *,
    operation_id: str,
    sequence: int,
    stage_code: str,
    state: str,
    duration_ms: int | None = None,
    safe_counts: dict[str, int] | None = None,
) -> ReviewStageEventV1:
    payload = {
        "schema_version": "1.0",
        "operation_id": operation_id,
        "sequence": sequence,
        "stage_code": stage_code,
        "state": state,
        "occurred_at": (datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        "duration_ms": duration_ms,
        "safe_counts": safe_counts or {},
        "message_key": f"agent.stage.{stage_code}.{state}",
    }
    payload["event_sha256"] = canonical_json_sha256(payload)
    return ReviewStageEventV1.model_validate(payload)


def _ndjson(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _error_response(request: Request, error: ServiceHttpError) -> JSONResponse:
    request_id = _request_id(request)
    headers = {**_NO_STORE_HEADERS, "X-Request-ID": request_id}
    if error.execution_state == "not-started":
        headers["X-Agent-Execution-State"] = "not-started"
    if error.retry_after_seconds is not None:
        headers["Retry-After"] = str(error.retry_after_seconds)
    safe_log(
        LOGGER,
        logging.WARNING,
        "request_rejected",
        error_code=error.code,
        request_id=request_id,
        retryable=error.retryable,
        status_code=error.status_code,
    )
    return JSONResponse(
        status_code=error.status_code,
        headers=headers,
        content={
            "error": {
                "code": error.code,
                "message": error.public_message,
                "request_id": request_id,
                "retryable": error.retryable,
            }
        },
    )


def _authorize(request: Request, settings: ServiceSettings) -> None:
    values = raw_header_values(request, "authorization")
    if not values:
        raise ServiceHttpError("authentication_required", 401)
    if len(values) != 1 or not bearer_authorized(values[0], settings):
        raise ServiceHttpError("authentication_invalid", 403)


def _production_runtime(settings: ServiceSettings) -> ServiceRuntime:
    # Release admission precedes even loading Claude credentials/settings.  A
    # safe local model identifier is sufficient to bind the signed profile;
    # secrets are loaded only after every immutable V3 release Gate passes.
    configured_model_id = os.environ.get(
        "CUSTOMS_AGENT_CLAUDE_MODEL",
        DEFAULT_CLAUDE_MODEL,
    ).strip()
    try:
        assert_production_release_admission(configured_model_id=configured_model_id)
    except ProductionReleaseAdmissionError as exc:
        raise RuntimeConfigurationError(
            exc.code,
            "The immutable Agent/Data Production release Gate is closed.",
        ) from None
    config = RuntimeConfig.from_environment(
        evidence_base_url=settings.evidence_base_url,
        evidence_service_token=settings.evidence_service_token,
        evidence_allowed_origins=settings.evidence_allowed_origins,
        verification_assertion_verifier=(SupportingDocumentAssertionVerifier.from_packaged_store()),
    )
    if config.claude_settings.model != configured_model_id:
        raise RuntimeConfigurationError(
            "claude_configured_model_identity_mismatch",
            "The configured Claude model changed during release admission.",
        )
    if config.verification_assertion_verifier is None:
        raise RuntimeConfigurationError(
            "trusted_assertion_verifier_unconfigured",
            "Production supporting-document trust verification is not configured.",
        )
    return AgentRuntime(config)


def _production_readiness_probe(settings: ServiceSettings) -> Callable[[], object]:
    def probe() -> object:
        token = (
            settings.evidence_service_token.get_secret_value()
            if settings.evidence_service_token is not None
            else None
        )
        return fetch_evidence_runtime_readiness_v3(
            settings.evidence_base_url,
            service_token=token,
            allowed_origins=settings.evidence_allowed_origins,
            timeout_seconds=settings.readiness_timeout_seconds,
            max_retries=0,
        )

    return probe


def create_app(
    settings: ServiceSettings | None = None,
    *,
    runtime_factory: Callable[[ServiceSettings], ServiceRuntime] | None = None,
    readiness_probe_factory: Callable[[ServiceSettings], Callable[[], object]] | None = None,
) -> FastAPI:
    """Create one text-only internal Agent service process."""

    active_settings = settings or ServiceSettings.from_environment()
    metadata = build_metadata(active_settings)
    legacy_metadata = build_legacy_metadata(active_settings)
    review_result_schema = get_schema("backend", "v2", "review-result")
    review_request_v3_schema = get_schema("backend", "v3", "review-request")
    review_result_v3_schema = get_schema("backend", "v3", "review-result")
    revision_request_schema = ReviewRevisionRequestV1.model_json_schema(mode="validation")
    revision_result_schema = ReviewRevisionResultV1.model_json_schema(mode="validation")
    revision_experience_result_schema = ReviewRevisionExperienceResultV1.model_json_schema(
        mode="validation"
    )
    grounded_question_schema = GroundedQuestionV1.model_json_schema(mode="validation")
    grounded_answer_schema = GroundedAnswerV1.model_json_schema(mode="validation")
    experience_envelope_schema = ReviewExperienceEnvelopeV1.model_json_schema(mode="validation")
    conversation_request_schema = CaseConversationRequestV2.model_json_schema(mode="validation")
    assistant_answer_schema = AssistantAnswerV2.model_json_schema(mode="validation")
    make_runtime = runtime_factory or _production_runtime
    make_readiness_probe = readiness_probe_factory or _production_readiness_probe

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime_available = True
        try:
            runtime = make_runtime(active_settings)
        except RuntimeConfigurationError:
            # An unreleased Data lock is an expected deployment state. Keep
            # liveness and generic 503 readiness available without constructing
            # Claude or accepting a review.
            runtime = _UnavailableRuntime()
            runtime_available = False
            safe_log(LOGGER, logging.WARNING, "runtime_dependency_unready")
        dependency_probe = make_readiness_probe(active_settings)

        def readiness_probe() -> object:
            if not runtime_available:
                raise RuntimeError("runtime unavailable")
            return dependency_probe()

        controller = AgentServiceController(
            settings=active_settings,
            runtime=runtime,
            readiness_probe=readiness_probe,
            logger=LOGGER,
        )
        app.state.controller = controller
        app.state.body_admission = asyncio.Semaphore(active_settings.max_concurrency)
        safe_log(LOGGER, logging.INFO, "service_started")
        try:
            yield
        finally:
            await controller.shutdown()
            safe_log(LOGGER, logging.INFO, "service_stopped")

    app = FastAPI(
        title="Customs AI Agent Internal Service",
        version="2.0.0",
        description=(
            "Private text-only service adapter for the evidence-bound Customs AI Agent. "
            "End-user authentication, Case state, and durable Job idempotency are Backend-owned."
        ),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.service_settings = active_settings
    app.state.build_metadata = metadata
    app.state.build_metadata_v1 = legacy_metadata
    experience_metadata = copy.deepcopy(metadata)
    experience_metadata.update(
        {
            "service_api_version": "internal-agent-service/v3",
            "enhanced_experience_enabled": active_settings.enhanced_agent_experience,
            "uat_adaptive_active": active_settings.uat_adaptive_active,
            "auto_completion_enabled": False,
            "meaning_catalog_sha256": meaning_catalog_sha256(),
            "experience_prompt_hashes": experience_prompt_hashes(),
            "experience_schema_hashes": experience_schema_hashes(),
        }
    )
    experience_metadata["capabilities"] = [
        *experience_metadata["capabilities"],
        "grounded_commercial_presentation",
        "grounded_revision_presentation",
        "multiturn_case_assistant_v2",
        "safe_ndjson_stage_events",
        "loopback_uat_bounded_active_tools",
    ]
    app.state.build_metadata_v3 = experience_metadata

    async def admitted_body(request: Request) -> bytes:
        body_admission: asyncio.Semaphore = request.app.state.body_admission
        try:
            await asyncio.wait_for(
                body_admission.acquire(),
                timeout=active_settings.queue_wait_seconds,
            )
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            return await read_bounded_json_body(request, active_settings)
        finally:
            body_admission.release()

    @app.middleware("http")
    async def request_identity(request: Request, call_next: Callable[..., Any]) -> Any:
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.exception_handler(ServiceHttpError)
    async def service_error_handler(request: Request, exc: ServiceHttpError) -> JSONResponse:
        return _error_response(request, exc)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(request, ServiceHttpError("internal_error", 500))

    @app.get(
        "/health/live",
        summary="Process liveness",
        responses={200: {"description": "The ASGI process is alive."}},
    )
    async def live() -> JSONResponse:
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content={"status": "alive"},
        )

    @app.get(
        "/health/ready",
        summary="Dependency-aware readiness",
        responses={
            200: {"description": "The local runtime and pinned Evidence dependency are ready."},
            503: {
                "description": "The service is draining or Evidence is not ready.",
                "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
            },
        },
    )
    async def ready(request: Request) -> JSONResponse:
        controller: AgentServiceController = request.app.state.controller
        if not await controller.ready():
            raise ServiceHttpError("dependency_unready", 503, retryable=True)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content={"status": "ready"},
        )

    @app.get(
        "/internal/agent/v1/metadata",
        summary="Safe build and capability metadata",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            200: {"description": "Non-secret release identity and capabilities."},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def service_metadata(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=request.app.state.build_metadata_v1,
        )

    @app.get(
        "/internal/agent/v1/metrics",
        summary="Process-local safe metrics",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            200: {"description": "Label-free process metrics without document identifiers."},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def service_metrics(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        controller: AgentServiceController = request.app.state.controller
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=controller.metrics.snapshot(),
        )

    @app.post(
        "/internal/agent/v1/reviews",
        summary="Execute one text-only Agent review",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": TEXT_REVIEW_REQUEST_SCHEMA}},
            },
        },
        responses={
            200: {
                "description": "A redacted Agent ReviewResult V2.",
                "content": {
                    "application/json": {
                        "schema": review_result_schema,
                    }
                },
                "headers": {
                    "X-Idempotent-Replay": {
                        "description": (
                            "True only when this process replayed the same review_run_id "
                            "and canonical request."
                        ),
                        "schema": {"type": "string", "enum": ["true", "false"]},
                    },
                    "X-Request-ID": {
                        "description": "Opaque per-request correlation identifier.",
                        "schema": {"type": "string", "format": "uuid"},
                    },
                },
            },
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            409: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            408: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def review(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_text_request(payload, active_settings)
        controller: AgentServiceController = request.app.state.controller
        # This is a dependency admission Gate, not a health hint. It runs
        # before the runtime worker, so an unsealed/mismatched Data release can
        # never spend Claude tokens.
        if not await controller.ready():
            raise ServiceHttpError(
                "dependency_unready",
                503,
                retryable=True,
                execution_state="not-started",
            )
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=inspection.review_run_id,
            request_hash=inspection.canonical_sha256,
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        headers = {
            **_NO_STORE_HEADERS,
            "X-Idempotent-Replay": "true" if replayed else "false",
        }
        return JSONResponse(
            status_code=200,
            headers=headers,
            content=outcome.result,
        )

    @app.get(
        "/internal/agent/v2/metadata",
        summary="V3 capability and release metadata",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            200: {"description": "Non-secret V3 release identity and capability limits."},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def service_metadata_v2(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=request.app.state.build_metadata,
        )

    @app.post(
        "/internal/agent/v2/reviews",
        summary="Execute one text-only bounded adaptive Agent review",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": review_request_v3_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": review_result_v3_schema}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            409: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def review_v3(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_text_request(
            payload,
            active_settings,
            accept_empty_visual_placeholders=True,
        )
        controller: AgentServiceController = request.app.state.controller
        if not await controller.ready():
            raise ServiceHttpError(
                "dependency_unready",
                503,
                retryable=True,
                execution_state="not-started",
            )
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=f"v3:{inspection.review_run_id}",
            request_hash=inspection.canonical_sha256,
            runtime_method_name="review_v3_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(
            status_code=200,
            headers={
                **_NO_STORE_HEADERS,
                "X-Idempotent-Replay": "true" if replayed else "false",
            },
            content=outcome.result,
        )

    @app.post(
        "/internal/agent/v2/revisions",
        summary="Execute an approved immutable child review",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": revision_request_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": revision_result_schema}}},
            409: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
        },
    )
    async def revision_v3(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_v3_auxiliary_request(
            payload,
            operation="revision",
            settings=active_settings,
        )
        controller: AgentServiceController = request.app.state.controller
        if not await controller.ready():
            raise ServiceHttpError(
                "dependency_unready",
                503,
                retryable=True,
                execution_state="not-started",
            )
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=f"revision:{inspection.replay_id}",
            request_hash=inspection.canonical_sha256,
            runtime_method_name="revision_v3_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
            )
        return JSONResponse(
            status_code=200,
            headers={
                **_NO_STORE_HEADERS,
                "X-Idempotent-Replay": "true" if replayed else "false",
            },
            content=outcome.result,
        )

    @app.post(
        "/internal/agent/v3/revisions",
        summary="Execute an approved child review with verified user presentation",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": revision_request_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": revision_experience_result_schema}}},
            409: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
        },
    )
    async def revision_experience_v3(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_v3_auxiliary_request(
            payload,
            operation="revision",
            settings=active_settings,
        )
        controller = await _require_enhanced_ready(request)
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=f"revision-experience:{inspection.replay_id}",
            request_hash=inspection.canonical_sha256,
            runtime_method_name="revision_experience_v3_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
            )
        return JSONResponse(
            status_code=200,
            headers={
                **_NO_STORE_HEADERS,
                "X-Idempotent-Replay": "true" if replayed else "false",
            },
            content=outcome.result,
        )

    @app.post(
        "/internal/agent/v2/answers",
        summary="Answer one evidence-grounded Case question",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": grounded_question_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": grounded_answer_schema}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
        },
    )
    async def answer_v3(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_v3_auxiliary_request(
            payload,
            operation="answer",
            settings=active_settings,
        )
        controller: AgentServiceController = request.app.state.controller
        if not await controller.ready():
            raise ServiceHttpError(
                "dependency_unready",
                503,
                retryable=True,
                execution_state="not-started",
            )
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=f"answer:{inspection.replay_id}",
            request_hash=inspection.canonical_sha256,
            runtime_method_name="answer_v3_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
            )
        return JSONResponse(
            status_code=200,
            headers={
                **_NO_STORE_HEADERS,
                "X-Idempotent-Replay": "true" if replayed else "false",
            },
            content=outcome.result,
        )

    @app.get(
        "/internal/agent/v3/metadata",
        summary="Enhanced Agent experience capability metadata",
        responses={
            200: {"content": {"application/json": {"schema": {"type": "object"}}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def service_metadata_v3(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=request.app.state.build_metadata_v3,
        )

    async def _require_enhanced_ready(request: Request) -> AgentServiceController:
        if not active_settings.enhanced_agent_experience:
            raise ServiceHttpError(
                "enhanced_experience_disabled",
                503,
                retryable=False,
                execution_state="not-started",
            )
        controller: AgentServiceController = request.app.state.controller
        try:
            # This is deliberately ordered before dependency readiness. An
            # expired or identity-drifted local capability must produce zero
            # Claude, Evidence, or Tool calls.
            controller.assert_uat_capability_current(require_active=False)
        except RuntimeConfigurationError:
            raise ServiceHttpError(
                "uat_capability_invalid_or_expired",
                503,
                retryable=False,
            ) from None
        if not await controller.ready():
            raise ServiceHttpError(
                "dependency_unready",
                503,
                retryable=True,
                execution_state="not-started",
            )
        return controller

    @app.post(
        "/internal/agent/v3/reviews",
        summary="Execute a review with verified user presentation",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": review_request_v3_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": experience_envelope_schema}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
        },
    )
    async def review_experience(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_text_request(
            payload,
            active_settings,
            accept_empty_visual_placeholders=True,
        )
        controller = await _require_enhanced_ready(request)
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=f"experience:{inspection.review_run_id}",
            request_hash=inspection.canonical_sha256,
            runtime_method_name="review_experience_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(
            status_code=200,
            headers={
                **_NO_STORE_HEADERS,
                "X-Idempotent-Replay": "true" if replayed else "false",
            },
            content=outcome.result,
        )

    @app.post(
        "/internal/agent/v3/answers",
        summary="Answer one multi-turn Case or customs question",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": conversation_request_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": assistant_answer_schema}}},
            409: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
        },
    )
    async def answer_experience(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_v3_auxiliary_request(
            payload,
            operation="answer_v2",
            settings=active_settings,
        )
        controller = await _require_enhanced_ready(request)
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=f"answer-v2:{inspection.replay_id}",
            request_hash=inspection.canonical_sha256,
            runtime_method_name="answer_v2_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(
            status_code=200,
            headers={
                **_NO_STORE_HEADERS,
                "X-Idempotent-Replay": "true" if replayed else "false",
            },
            content=outcome.result,
        )

    @app.post(
        "/internal/agent/v3/reviews/stream",
        summary="Execute a review and stream safe verified stages as NDJSON",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": review_request_v3_schema}},
            },
        },
    )
    async def review_experience_stream(request: Request) -> StreamingResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_text_request(
            payload,
            active_settings,
            accept_empty_visual_placeholders=True,
        )
        controller = await _require_enhanced_ready(request)

        async def generate() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)

            def observe(event: dict[str, Any]) -> None:
                if not queue.full():
                    queue.put_nowait(event)

            submitted = asyncio.create_task(
                controller.submit(
                    payload=payload,
                    review_run_id=f"experience:{inspection.review_run_id}",
                    request_hash=inspection.canonical_sha256,
                    runtime_method_name="review_experience_stream_json",
                    stage_callback=observe,
                )
            )
            sequence = 0
            while not submitted.done() or not queue.empty():
                try:
                    observed = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                event = _stage_event(
                    operation_id=inspection.review_run_id,
                    sequence=sequence,
                    stage_code=str(observed["stage_code"]),
                    state=str(observed["state"]),
                    duration_ms=observed.get("duration_ms"),
                    safe_counts=observed.get("safe_counts") or {},
                )
                yield _ndjson(
                    {
                        "type": "stage_event",
                        "event": event.model_dump(mode="json"),
                    }
                )
                sequence += 1
            outcome, _replayed = await submitted
            if outcome.result is None:
                yield _ndjson(
                    {
                        "type": "stage_event",
                        "event": _stage_event(
                            operation_id=inspection.review_run_id,
                            sequence=sequence,
                            stage_code="failed",
                            state="failed",
                        ).model_dump(mode="json"),
                    }
                )
                yield _ndjson(
                    {
                        "type": "error",
                        "error": {
                            "code": outcome.error_code or "internal_error",
                            "retryable": outcome.retryable,
                        },
                    }
                )
                return
            yield _ndjson({"type": "result", "result": outcome.result})

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers=_NO_STORE_HEADERS,
        )

    @app.post(
        "/internal/agent/v3/answers/stream",
        summary="Answer a multi-turn question and stream safe stages as NDJSON",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": conversation_request_schema}},
            },
        },
    )
    async def answer_experience_stream(request: Request) -> StreamingResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_v3_auxiliary_request(
            payload,
            operation="answer_v2",
            settings=active_settings,
        )
        controller = await _require_enhanced_ready(request)
        operation_id = inspection.replay_id[-64:]

        async def generate() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)

            def observe(event: dict[str, Any]) -> None:
                if not queue.full():
                    queue.put_nowait(event)

            submitted = asyncio.create_task(
                controller.submit(
                    payload=payload,
                    review_run_id=f"answer-v2:{inspection.replay_id}",
                    request_hash=inspection.canonical_sha256,
                    runtime_method_name="answer_v2_stream_json",
                    stage_callback=observe,
                )
            )
            sequence = 0
            while not submitted.done() or not queue.empty():
                try:
                    observed = await asyncio.wait_for(queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                event = _stage_event(
                    operation_id=operation_id,
                    sequence=sequence,
                    stage_code=str(observed["stage_code"]),
                    state=str(observed["state"]),
                    duration_ms=observed.get("duration_ms"),
                    safe_counts=observed.get("safe_counts") or {},
                )
                yield _ndjson(
                    {
                        "type": "stage_event",
                        "event": event.model_dump(mode="json"),
                    }
                )
                sequence += 1
            outcome, _replayed = await submitted
            if outcome.result is None:
                yield _ndjson(
                    {
                        "type": "stage_event",
                        "event": _stage_event(
                            operation_id=operation_id,
                            sequence=sequence,
                            stage_code="failed",
                            state="failed",
                        ).model_dump(mode="json"),
                    }
                )
                yield _ndjson(
                    {
                        "type": "error",
                        "error": {
                            "code": outcome.error_code or "internal_error",
                            "retryable": outcome.retryable,
                        },
                    }
                )
                return
            yield _ndjson({"type": "result", "result": outcome.result})

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers=_NO_STORE_HEADERS,
        )

    original_openapi = app.openapi

    def service_openapi() -> dict[str, Any]:
        schema = original_openapi()
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["ServiceBearer"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "opaque",
            "description": "High-entropy internal service credential.",
        }
        return schema

    app.openapi = service_openapi  # type: ignore[method-assign]
    return app


__all__ = ["create_app"]
