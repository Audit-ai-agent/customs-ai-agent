from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from customs_ai_agent.adapters.retrieval import fetch_evidence_readiness
from customs_ai_agent.contracts import get_schema
from customs_ai_agent.runtime import AgentRuntime, RuntimeConfig
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
    raw_header_values,
    read_bounded_json_body,
)
from customs_ai_agent.service.metadata import build_metadata
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.telemetry import safe_log


LOGGER = logging.getLogger("customs_ai_agent.service")
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}
_BEARER_SECURITY: list[dict[str, list[str]]] = [{"ServiceBearer": []}]


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid4())


def _error_response(request: Request, error: ServiceHttpError) -> JSONResponse:
    request_id = _request_id(request)
    headers = {**_NO_STORE_HEADERS, "X-Request-ID": request_id}
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
    return AgentRuntime(
        RuntimeConfig.from_environment(
            evidence_base_url=settings.evidence_base_url,
            evidence_service_token=settings.evidence_service_token,
            evidence_token_header=settings.evidence_token_header,
            evidence_allowed_origins=settings.evidence_allowed_origins,
        )
    )


def _production_readiness_probe(settings: ServiceSettings) -> Callable[[], object]:
    def probe() -> object:
        token = (
            settings.evidence_service_token.get_secret_value()
            if settings.evidence_service_token is not None
            else None
        )
        return fetch_evidence_readiness(
            settings.evidence_base_url,
            service_token=token,
            token_header=settings.evidence_token_header,
            allowed_origins=settings.evidence_allowed_origins,
            timeout_seconds=settings.readiness_timeout_seconds,
            max_retries=0,
            contract_identity_mode="required",
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
    review_result_schema = get_schema("backend", "v2", "review-result")
    make_runtime = runtime_factory or _production_runtime
    make_readiness_probe = readiness_probe_factory or _production_readiness_probe

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = make_runtime(active_settings)
        controller = AgentServiceController(
            settings=active_settings,
            runtime=runtime,
            readiness_probe=make_readiness_probe(active_settings),
            logger=LOGGER,
        )
        app.state.controller = controller
        safe_log(LOGGER, logging.INFO, "service_started")
        try:
            yield
        finally:
            await controller.shutdown()
            safe_log(LOGGER, logging.INFO, "service_stopped")

    app = FastAPI(
        title="Customs AI Agent Internal Service",
        version="1.0.0",
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
            content=request.app.state.build_metadata,
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
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def review(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await read_bounded_json_body(request, active_settings)
        inspection = inspect_text_request(payload, active_settings)
        controller: AgentServiceController = request.app.state.controller
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
