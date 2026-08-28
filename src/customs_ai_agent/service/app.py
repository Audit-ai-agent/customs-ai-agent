"""Loopback-only HTTP surface for the two supported Agent profiles.

The service exposes one bounded review and one synchronous grounded-assistant
turn.  Backend retains ownership of users, conversations, Case state, and all
human correction decisions.
"""

from __future__ import annotations

import re

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from customs_ai_agent.contracts import get_schema
from customs_ai_agent.schemas.raster_text_extraction import RasterTextExtractionResultV1
from customs_ai_agent.service.auth import bearer_authorized
from customs_ai_agent.service.controller import AgentServiceController, ServiceRuntime
from customs_ai_agent.service.document_extraction import (
    DisabledDocumentPreviewEngine,
    DocumentPreviewExecutionError,
    DocumentPreviewInputError,
    LocalDocumentPreviewEngine,
    extract_document_preview,
)
from customs_ai_agent.service.document_input import (
    read_bounded_document_body,
    read_bounded_raster_document_body,
    read_bounded_raster_text_document_body,
)
from customs_ai_agent.service.document_preview_runtime import resolve_document_preview_engine
from customs_ai_agent.service.raster_document_extraction import (
    DisabledRasterDocumentPreviewEngine,
    LocalRasterDocumentPreviewEngine,
    RasterDocumentPreviewExecutionError,
    RasterDocumentPreviewInputError,
    RasterDocumentPreviewUnavailable,
    extract_raster_document_preview,
)
from customs_ai_agent.service.raster_document_preview_runtime import (
    AutoActivatingSourceAttestedRasterPreviewRuntime,
    RasterDocumentPreviewRuntime,
    StaticRasterDocumentPreviewRuntime,
    resolve_raster_document_preview_engine,
)
from customs_ai_agent.service.raster_text_extraction import (
    DisabledRasterTextPreviewEngine,
    LocalRasterTextPreviewEngine,
    RasterTextPreviewExecutionError,
    RasterTextPreviewInputError,
    extract_raster_text_preview,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    AutoActivatingSourceAttestedRasterTextPreviewRuntime,
    RasterTextPreviewRuntime,
    StaticRasterTextPreviewRuntime,
    resolve_raster_text_preview_engine,
)
from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.progress import PROGRESS
from customs_ai_agent.service.input import (
    ERROR_RESPONSE_SCHEMA,
    inspect_assistant_request,
    inspect_knowledge_request,
    inspect_text_request,
    raw_header_values,
    read_bounded_json_body,
)
from customs_ai_agent.service.metadata import build_metadata
from customs_ai_agent.service.profiles import (
    SERVICE_MODE_HEADER,
    build_profile_runtime,
    profile_readiness_probe,
)
from customs_ai_agent.service.runtime_errors import RuntimeConfigurationError
from customs_ai_agent.service.settings import ServiceSettings
from customs_ai_agent.service.telemetry import safe_log


LOGGER = logging.getLogger("customs_ai_agent.service")
_SAFE_PROGRESS_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}
_BEARER_SECURITY: list[dict[str, list[str]]] = [{"ServiceBearer": []}]
_PRE_EXECUTION_503_RESPONSE: dict[str, Any] = {
    "content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}},
    "headers": {
        "X-Agent-Execution-State": {
            "description": "The request was rejected before review execution.",
            "schema": {"type": "string", "enum": ["not-started"]},
        }
    },
}
_RASTER_TEXT_AUTO_ACTIVATION_MODE = "source-attested-synthetic-raster-text-v1"
_RASTER_TEXT_PREWARM_POLL_SECONDS = 1.0


class _UnavailableRuntime:
    """Preserve liveness while a local profile dependency is unavailable."""

    def review_v3_json(self, _payload: bytes) -> object:
        raise RuntimeConfigurationError(
            "runtime_dependency_unready",
            "The Agent profile dependency is not ready.",
        )

    def assistant_answer_json(self, _payload: bytes) -> object:
        raise RuntimeConfigurationError(
            "runtime_dependency_unready",
            "The Agent profile dependency is not ready.",
        )

    def knowledge_answer_json(self, _payload: bytes) -> object:
        raise RuntimeConfigurationError(
            "runtime_dependency_unready",
            "The Agent profile dependency is not ready.",
        )

    def close(self) -> None:
        return None


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid4())


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


def _profile_runtime(settings: ServiceSettings) -> ServiceRuntime:
    return build_profile_runtime(settings)


def _profile_readiness_probe(settings: ServiceSettings) -> Callable[[], object]:
    return profile_readiness_probe(settings)


def _raster_text_prewarm_callable(
    *,
    settings: ServiceSettings,
    runtime: RasterTextPreviewRuntime,
    injected_engine: LocalRasterTextPreviewEngine | None,
) -> Callable[[], object] | None:
    """Return the one supported automatic warmup hook, if this process owns it.

    Static and test-injected preview paths must remain inert.  The automatic
    pointer mode can start with no pointer at all, so the hook is deliberately
    selected without calling ``status()`` or loading any material on the event
    loop.  Runtime implementations own the fail-closed pointer validation.
    """

    if (
        settings.raster_text_preview_auto_activation_mode
        != _RASTER_TEXT_AUTO_ACTIVATION_MODE
        or injected_engine is not None
    ):
        return None
    candidate = getattr(runtime, "prewarm", None)
    return candidate if callable(candidate) else None


async def _run_raster_text_prewarm_loop(prewarm: Callable[[], object]) -> None:
    """Prewarm at most one automatic text binding, then poll for replacements."""

    while True:
        try:
            # Native sidecar startup is blocking by design.  Keeping it in a
            # single worker thread leaves FastAPI startup/readiness and the
            # status endpoint responsive while no warm binding exists yet.
            await asyncio.to_thread(prewarm)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Pointer absence/drift normally returns an unavailable status.
            # This narrow guard covers an implementation fault without
            # exposing raw model/package errors or stopping future polls.
            safe_log(LOGGER, logging.WARNING, "raster_text_prewarm_failed")
        await asyncio.sleep(_RASTER_TEXT_PREWARM_POLL_SECONDS)


def create_app(
    settings: ServiceSettings | None = None,
    *,
    runtime_factory: Callable[[ServiceSettings], ServiceRuntime] | None = None,
    readiness_probe_factory: Callable[[ServiceSettings], Callable[[], object]] | None = None,
    document_preview_engine: LocalDocumentPreviewEngine | None = None,
    raster_document_preview_engine: LocalRasterDocumentPreviewEngine | None = None,
    raster_text_preview_engine: LocalRasterTextPreviewEngine | None = None,
) -> FastAPI:
    """Create the text-review service plus its isolated preview boundary."""

    active_settings = settings or ServiceSettings.from_environment()
    review_request_schema = get_schema("backend", "v3", "review-request")
    review_result_schema = get_schema("backend", "v3", "review-result")
    assistant_request_schema = get_schema("backend", "v3", "assistant-answer-request")
    assistant_response_schema = get_schema("backend", "v3", "assistant-answer-response")
    knowledge_request_schema = get_schema("backend", "v3", "knowledge-answer-request")
    knowledge_response_schema = get_schema("backend", "v3", "knowledge-answer-response")
    document_extraction_request_schema = get_schema("backend", "v3", "document-extraction-request")
    document_extraction_result_schema = get_schema("backend", "v3", "document-extraction-result")
    raster_document_extraction_request_schema = get_schema(
        "backend", "v3", "raster-document-extraction-request"
    )
    raster_document_extraction_result_schema = get_schema(
        "backend", "v3", "raster-document-extraction-result"
    )
    raster_text_extraction_request_schema = get_schema(
        "backend", "v3", "raster-text-extraction-request"
    )
    raster_text_extraction_result_schema = get_schema(
        "backend", "v3", "raster-text-extraction-result"
    )
    make_runtime = runtime_factory or _profile_runtime
    make_readiness_probe = readiness_probe_factory or _profile_readiness_probe
    # Direct injection remains a test-only seam.  Normal startup may resolve
    # only the fixed local approval pointer and never accepts an engine/model
    # URL or a fallback selector through the environment.  The explicit
    # source-attested mode still begins unavailable; it validates a later
    # atomic local pointer replacement per PDF request.
    preview_resolution = (
        None
        if document_preview_engine is not None
        else resolve_document_preview_engine(
            approval_path=active_settings.document_preview_approval_path,
            approval_sha256=active_settings.document_preview_approval_sha256,
            source_attested_auto_activation=(
                active_settings.document_preview_auto_activation_mode == "source-attested-synthetic"
            ),
        )
    )
    active_document_preview_engine = (
        document_preview_engine
        if document_preview_engine is not None
        else preview_resolution.engine
        if preview_resolution is not None
        else DisabledDocumentPreviewEngine()
    )
    # Raster retains the hash-pinned static pointer.  Its explicit automatic
    # source-attested mode is a separate immutable binding facade: the same
    # pointer snapshot selects both OCR and reviewer-page rendering on each
    # request, and a bad replacement clears the prior binding.
    raster_preview_resolution = (
        None
        if (
            raster_document_preview_engine is not None
            or active_settings.raster_document_preview_auto_activation_mode
            == "source-attested-synthetic-raster-v1"
        )
        else resolve_raster_document_preview_engine(
            approval_path=active_settings.raster_document_preview_approval_path,
            approval_sha256=active_settings.raster_document_preview_approval_sha256,
        )
    )
    active_raster_document_preview_runtime: RasterDocumentPreviewRuntime = (
        StaticRasterDocumentPreviewRuntime(
            engine=raster_document_preview_engine,
            available=False,
        )
        if raster_document_preview_engine is not None
        else AutoActivatingSourceAttestedRasterPreviewRuntime(
            approval_path=active_settings.raster_document_preview_approval_path
        )
        if active_settings.raster_document_preview_auto_activation_mode
        == "source-attested-synthetic-raster-v1"
        and active_settings.raster_document_preview_approval_path is not None
        else StaticRasterDocumentPreviewRuntime(
            engine=(
                raster_preview_resolution.engine
                if raster_preview_resolution is not None
                else DisabledRasterDocumentPreviewEngine()
            ),
            page_renderer=(
                None
                if raster_preview_resolution is None
                else raster_preview_resolution.page_renderer
            ),
            raster_profile_revision=(
                None
                if raster_preview_resolution is None
                else raster_preview_resolution.raster_profile_revision
            ),
            available=(
                raster_preview_resolution is not None
                and raster_preview_resolution.status == "approved"
            ),
        )
    )
    raster_text_preview_resolution = (
        None
        if (
            raster_text_preview_engine is not None
            or active_settings.raster_text_preview_auto_activation_mode
            == "source-attested-synthetic-raster-text-v1"
        )
        else resolve_raster_text_preview_engine(
            approval_path=active_settings.raster_text_preview_approval_path,
            approval_sha256=active_settings.raster_text_preview_approval_sha256,
        )
    )
    active_raster_text_preview_runtime: RasterTextPreviewRuntime = (
        StaticRasterTextPreviewRuntime(engine=raster_text_preview_engine, available=False)
        if raster_text_preview_engine is not None
        else AutoActivatingSourceAttestedRasterTextPreviewRuntime(
            approval_path=active_settings.raster_text_preview_approval_path
        )
        if active_settings.raster_text_preview_auto_activation_mode
        == "source-attested-synthetic-raster-text-v1"
        and active_settings.raster_text_preview_approval_path is not None
        else StaticRasterTextPreviewRuntime(
            engine=(
                raster_text_preview_resolution.engine
                if raster_text_preview_resolution is not None
                else DisabledRasterTextPreviewEngine()
            ),
            raster_text_profile_revision=(
                None
                if raster_text_preview_resolution is None
                else raster_text_preview_resolution.raster_text_profile_revision
            ),
            available=(
                raster_text_preview_resolution is not None
                and raster_text_preview_resolution.status == "available"
            ),
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime_available = True
        try:
            runtime = make_runtime(active_settings)
        except RuntimeConfigurationError:
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
        app.state.document_preview_admission = asyncio.Semaphore(active_settings.max_concurrency)
        # The approved raster profile is CPU-thread-bound and owns exactly one
        # warm native sidecar.  Keep OCR and reviewer-page rendering on the
        # same admission lane so a viewer request cannot oversubscribe it or
        # race a profile replacement with an unrelated native job.
        app.state.raster_document_preview_admission = asyncio.Semaphore(
            active_settings.raster_max_concurrency
        )
        raster_text_prewarm = _raster_text_prewarm_callable(
            settings=active_settings,
            runtime=active_raster_text_preview_runtime,
            injected_engine=raster_text_preview_engine,
        )
        # Do not await native startup here.  In automatic pointer mode this
        # one task also notices a pointer written after service startup and a
        # later atomic pointer replacement.  It serially awaits each blocking
        # call, so it never creates overlapping native warmups.
        raster_text_prewarm_task: asyncio.Task[None] | None = (
            asyncio.create_task(_run_raster_text_prewarm_loop(raster_text_prewarm))
            if raster_text_prewarm is not None
            else None
        )
        app.state.raster_text_prewarm_task = raster_text_prewarm_task
        safe_log(LOGGER, logging.INFO, "service_started")
        try:
            yield
        finally:
            if raster_text_prewarm_task is not None:
                raster_text_prewarm_task.cancel()
                # ``to_thread`` cannot interrupt a native call itself, but
                # waiting for the loop task to observe cancellation before
                # ``close`` avoids scheduling another poll during shutdown.
                await asyncio.gather(raster_text_prewarm_task, return_exceptions=True)
                app.state.raster_text_prewarm_task = None
            # The automatic runtime owns a long-lived child process.  Static
            # and test runtimes provide the same no-op lifecycle method.
            active_raster_document_preview_runtime.close()
            shutdown_raster_text = getattr(active_raster_text_preview_runtime, "shutdown", None)
            if callable(shutdown_raster_text):
                shutdown_raster_text()
            else:
                # Compatibility for focused injected runtimes owned by tests.
                active_raster_text_preview_runtime.close()
            await controller.shutdown()
            safe_log(LOGGER, logging.INFO, "service_stopped")

    app = FastAPI(
        title="Customs AI Agent Internal Service",
        version="2.0.0",
        description=(
            "Loopback-only text review service. Backend owns users, Case state, uploads, "
            "and final human decisions."
        ),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.service_settings = active_settings
    app.state.build_metadata = build_metadata(active_settings)
    app.state.raster_text_prewarm_task = None
    app.state.document_preview_activation_status = (
        "injected"
        if document_preview_engine is not None
        else preview_resolution.status
        if preview_resolution is not None
        else "disabled"
    )
    app.state.document_preview_activation_reason = (
        None if preview_resolution is None else preview_resolution.reason_code
    )
    app.state.raster_document_preview_activation_status = (
        "injected"
        if raster_document_preview_engine is not None
        else "auto-pending"
        if active_settings.raster_document_preview_auto_activation_mode
        == "source-attested-synthetic-raster-v1"
        else raster_preview_resolution.status
        if raster_preview_resolution is not None
        else "disabled"
    )
    app.state.raster_document_preview_activation_reason = (
        None if raster_preview_resolution is None else raster_preview_resolution.reason_code
    )
    if active_settings.raster_document_preview_auto_activation_mode == (
        "source-attested-synthetic-raster-v1"
    ):
        app.state.raster_document_preview_activation_reason = (
            "raster_source_attested_auto_pending"
        )
    # Kept for compatibility with existing local test seams.  Requests use
    # the unified runtime facade below, never this mutable app-state object.
    app.state.raster_document_page_renderer = (
        None if raster_preview_resolution is None else raster_preview_resolution.page_renderer
    )
    app.state.raster_text_preview_activation_status = (
        "injected"
        if raster_text_preview_engine is not None
        else "auto-pending"
        if active_settings.raster_text_preview_auto_activation_mode
        == "source-attested-synthetic-raster-text-v1"
        else raster_text_preview_resolution.status
        if raster_text_preview_resolution is not None
        else "disabled"
    )
    app.state.raster_text_preview_activation_reason = (
        None
        if raster_text_preview_resolution is None
        else raster_text_preview_resolution.reason_code
    )
    if active_settings.raster_text_preview_auto_activation_mode == (
        "source-attested-synthetic-raster-text-v1"
    ):
        app.state.raster_text_preview_activation_reason = "raster_text_source_attested_auto_pending"

    async def admitted_body(request: Request) -> bytes:
        admission: asyncio.Semaphore = request.app.state.body_admission
        try:
            await asyncio.wait_for(admission.acquire(), timeout=active_settings.queue_wait_seconds)
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
            admission.release()

    async def admitted_document(request: Request) -> object:
        """Read raw bytes only at the dedicated preview boundary."""

        admission: asyncio.Semaphore = request.app.state.body_admission
        try:
            await asyncio.wait_for(admission.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            return await read_bounded_document_body(request, active_settings)
        finally:
            admission.release()

    async def admitted_raster_document(request: Request) -> object:
        """Read raw v2 OCR bytes only at the isolated raster boundary."""

        admission: asyncio.Semaphore = request.app.state.body_admission
        try:
            await asyncio.wait_for(admission.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            return await read_bounded_raster_document_body(request, active_settings)
        finally:
            admission.release()

    async def admitted_raster_text_document(request: Request) -> object:
        """Read raw text-OCR bytes only at the independent v1 boundary."""

        admission: asyncio.Semaphore = request.app.state.body_admission
        try:
            await asyncio.wait_for(admission.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            return await read_bounded_raster_text_document_body(request, active_settings)
        finally:
            admission.release()

    async def execute_document_preview(admitted: object) -> dict[str, Any]:
        """Run only an injected local adapter under the normal capacity bound."""

        # Keep the binary value in this narrow function and avoid propagating
        # it into the review controller, transcript code, or structured logs.
        from customs_ai_agent.service.document_input import InspectedDocumentExtractionRequest

        if not isinstance(admitted, InspectedDocumentExtractionRequest):
            raise ServiceHttpError("internal_error", 500)
        capacity: asyncio.Semaphore = app.state.document_preview_admission
        try:
            await asyncio.wait_for(capacity.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    extract_document_preview,
                    active_document_preview_engine,
                    preview_input=admitted.preview_input,
                    structure=admitted.structure,
                    settings=active_settings,
                ),
                timeout=active_settings.run_timeout_seconds,
            )
        except TimeoutError:
            raise ServiceHttpError("document_preview_timeout", 504, retryable=True) from None
        except DocumentPreviewInputError:
            raise ServiceHttpError("document_decode_invalid", 422) from None
        except DocumentPreviewExecutionError:
            raise ServiceHttpError("document_preview_failed", 502, retryable=True) from None
        except Exception:
            raise ServiceHttpError("document_preview_failed", 502, retryable=True) from None
        finally:
            capacity.release()
        return result.model_dump(mode="json", warnings=False)

    async def execute_raster_document_preview(admitted: object) -> dict[str, Any]:
        """Run an explicitly injected raster adapter under the preview bound."""

        from customs_ai_agent.service.document_input import InspectedRasterDocumentExtractionRequest

        if not isinstance(admitted, InspectedRasterDocumentExtractionRequest):
            raise ServiceHttpError("internal_error", 500)
        capacity: asyncio.Semaphore = app.state.raster_document_preview_admission
        try:
            await asyncio.wait_for(capacity.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    extract_raster_document_preview,
                    active_raster_document_preview_runtime,
                    raster_input=admitted.raster_input,
                    structure=admitted.structure,
                    settings=active_settings,
                ),
                timeout=active_settings.raster_run_timeout_seconds,
            )
        except TimeoutError:
            # ``to_thread`` cannot forcibly stop Python work.  The production
            # warm-session runtime reacts by terminating/reaping its child;
            # closing here prevents a timed-out request from retaining that
            # sidecar for a later caller.
            active_raster_document_preview_runtime.close()
            raise ServiceHttpError("raster_document_preview_timeout", 504, retryable=True) from None
        except RasterDocumentPreviewInputError:
            raise ServiceHttpError("document_decode_invalid", 422) from None
        except RasterDocumentPreviewExecutionError:
            raise ServiceHttpError("raster_document_preview_failed", 502, retryable=True) from None
        except Exception:
            raise ServiceHttpError("raster_document_preview_failed", 502, retryable=True) from None
        finally:
            capacity.release()
        return result.model_dump(mode="json", warnings=False)

    async def execute_raster_document_page(admitted: object, *, page_number: int) -> Response:
        """Return one approved PDFium page only to the Backend proxy boundary."""

        from customs_ai_agent.service.document_input import InspectedRasterDocumentExtractionRequest

        if not isinstance(admitted, InspectedRasterDocumentExtractionRequest):
            raise ServiceHttpError("internal_error", 500)
        if admitted.raster_input.metadata.media_type != "application/pdf" or page_number < 1:
            raise ServiceHttpError("document_decode_invalid", 422)
        capacity: asyncio.Semaphore = app.state.raster_document_preview_admission
        try:
            await asyncio.wait_for(capacity.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        try:
            rendered = await asyncio.wait_for(
                asyncio.to_thread(
                    active_raster_document_preview_runtime.render_pdf_page,
                    admitted.raster_input,
                    admitted.structure,
                    page_number=page_number,
                ),
                timeout=active_settings.raster_page_render_timeout_seconds,
            )
        except TimeoutError:
            active_raster_document_preview_runtime.close()
            raise ServiceHttpError("raster_document_page_timeout", 504, retryable=True) from None
        except RasterDocumentPreviewInputError:
            raise ServiceHttpError("document_decode_invalid", 422) from None
        except RasterDocumentPreviewUnavailable:
            raise ServiceHttpError("raster_page_renderer_unavailable", 503, retryable=True) from None
        except Exception:
            raise ServiceHttpError("raster_document_page_failed", 502, retryable=True) from None
        finally:
            capacity.release()
        return Response(
            content=rendered.content,
            status_code=200,
            media_type=rendered.media_type,
            headers={
                **_NO_STORE_HEADERS,
                "X-Document-SHA256": rendered.source_sha256,
                "X-Raster-Profile-Revision": rendered.raster_profile_revision,
                "X-Raster-Page-Number": str(rendered.page_number),
                "X-Raster-Page-Width": str(rendered.width),
                "X-Raster-Page-Height": str(rendered.height),
                "X-Raster-Page-SHA256": rendered.content_sha256,
                "X-Raster-Renderer-Revision": rendered.renderer_revision,
            },
        )

    async def execute_raster_text_preview(admitted: object) -> dict[str, Any]:
        """Run the independent text-only OCR profile under the shared CPU lane."""

        from customs_ai_agent.service.document_input import InspectedRasterTextExtractionRequest

        if not isinstance(admitted, InspectedRasterTextExtractionRequest):
            raise ServiceHttpError("internal_error", 500)
        # Text and structured OCR compete for the same bounded local CPU and
        # sidecar budget.  One shared lane preserves their measured deadlines.
        capacity: asyncio.Semaphore = app.state.raster_document_preview_admission
        try:
            await asyncio.wait_for(capacity.acquire(), timeout=active_settings.queue_wait_seconds)
        except TimeoutError:
            raise ServiceHttpError(
                "service_busy",
                503,
                retryable=True,
                retry_after_seconds=max(1, round(active_settings.queue_wait_seconds)),
            ) from None
        worker: asyncio.Task[RasterTextExtractionResultV1] | None = None
        release_when_worker_finishes = False

        def release_after_worker(done: asyncio.Task[RasterTextExtractionResultV1]) -> None:
            # ``to_thread`` cannot be cancelled while native decode/OCR is
            # executing.  Consume a late exception and keep the shared CPU
            # lane occupied until the owned worker has actually returned;
            # otherwise a timed-out request could oversubscribe OCR work.
            try:
                done.exception()
            except (asyncio.CancelledError, Exception):
                pass
            capacity.release()

        try:
            worker = asyncio.create_task(
                asyncio.to_thread(
                    extract_raster_text_preview,
                    active_raster_text_preview_runtime,
                    raster_input=admitted.raster_text_input,
                    structure=admitted.structure,
                    settings=active_settings,
                )
            )
            result = await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=active_settings.raster_run_timeout_seconds,
            )
        except TimeoutError:
            active_raster_text_preview_runtime.close()
            if worker is not None:
                worker.add_done_callback(release_after_worker)
                release_when_worker_finishes = True
            raise ServiceHttpError("raster_text_preview_timeout", 504, retryable=True) from None
        except RasterTextPreviewInputError:
            raise ServiceHttpError("document_decode_invalid", 422) from None
        except RasterTextPreviewExecutionError:
            raise ServiceHttpError("raster_text_preview_failed", 502, retryable=True) from None
        except Exception:
            raise ServiceHttpError("raster_text_preview_failed", 502, retryable=True) from None
        finally:
            if not release_when_worker_finishes:
                capacity.release()
        return result.model_dump(mode="json", warnings=False)

    @app.middleware("http")
    async def request_identity(request: Request, call_next: Callable[..., Any]) -> Any:
        request.state.request_id = str(uuid4())
        if request.url.path.startswith("/internal/agent/") and raw_header_values(
            request, SERVICE_MODE_HEADER
        ) != [active_settings.service_mode]:
            response = _error_response(request, ServiceHttpError("service_mode_mismatch", 403))
            response.headers[SERVICE_MODE_HEADER] = active_settings.service_mode
            return response
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.state.request_id)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault(SERVICE_MODE_HEADER, active_settings.service_mode)
        return response

    @app.exception_handler(ServiceHttpError)
    async def service_error_handler(request: Request, exc: ServiceHttpError) -> JSONResponse:
        return _error_response(request, exc)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _exc: Exception) -> JSONResponse:
        return _error_response(request, ServiceHttpError("internal_error", 500))

    @app.get("/health/live", summary="Process liveness")
    async def live() -> JSONResponse:
        return JSONResponse(status_code=200, headers=_NO_STORE_HEADERS, content={"status": "alive"})

    @app.get("/health/ready", summary="Profile dependency readiness")
    async def ready(request: Request) -> JSONResponse:
        controller: AgentServiceController = request.app.state.controller
        if not await controller.ready():
            raise ServiceHttpError("dependency_unready", 503, retryable=True)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content={"status": "ready", "service_mode": active_settings.service_mode},
        )

    @app.get(
        "/internal/agent/v2/metadata",
        summary="Local profile metadata",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            200: {"description": "Non-secret local profile metadata."},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def metadata(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=request.app.state.build_metadata,
        )

    @app.post(
        "/internal/agent/v2/reviews",
        summary="Execute one text-only human-review Agent pass",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": review_request_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": review_result_schema}}},
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
    async def review(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspection = inspect_text_request(payload, active_settings)
        controller: AgentServiceController = request.app.state.controller
        outcome, replayed = await controller.submit(
            payload=payload,
            review_run_id=inspection.review_run_id,
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

    @app.get(
        "/internal/agent/v3/reviews/{review_run_id}/progress",
        summary="Read display-only live progress for one in-flight review",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            404: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def review_progress(request: Request, review_run_id: str) -> JSONResponse:
        _authorize(request, active_settings)
        if not _SAFE_PROGRESS_RUN_ID.fullmatch(review_run_id):
            raise ServiceHttpError("request_invalid", 422)
        snapshot = PROGRESS.snapshot(review_run_id)
        if snapshot is None:
            raise ServiceHttpError("review_progress_not_found", 404)
        return JSONResponse(status_code=200, headers=_NO_STORE_HEADERS, content=snapshot)

    @app.post(
        "/internal/agent/v2/document-extractions",
        summary="Create one human-review-only local document extraction preview",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "parameters": [
                {
                    "in": "header",
                    "name": "X-Document-Asset-ID",
                    "required": True,
                    "schema": document_extraction_request_schema["properties"]["document_asset_id"],
                },
                {
                    "in": "header",
                    "name": "X-Extraction-Request-ID",
                    "required": True,
                    "schema": document_extraction_request_schema["properties"][
                        "extraction_request_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Document-SHA256",
                    "required": True,
                    "schema": document_extraction_request_schema["properties"]["source_sha256"],
                },
            ],
            "requestBody": {
                "required": True,
                "description": (
                    "Identity-encoded raw bytes only. Content-Type must be application/pdf, "
                    "image/jpeg, or image/png; multipart, JSON, and base64 are rejected."
                ),
                "content": {
                    "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                    "image/png": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            "x-customs-request-metadata-schema": document_extraction_request_schema,
        },
        responses={
            200: {"content": {"application/json": {"schema": document_extraction_result_schema}}},
            400: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            408: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def document_extraction(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        admitted = await admitted_document(request)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=await execute_document_preview(admitted),
        )

    @app.get(
        "/internal/agent/v3/raster-document-preview-status",
        summary="Return only the active raster profile identity for Backend job binding",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            200: {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "schema_version",
                                "availability",
                                "raster_profile_revision",
                            ],
                            "properties": {
                                "schema_version": {"type": "string", "const": "1.0"},
                                "availability": {
                                    "type": "string",
                                    "enum": ["available", "unavailable"],
                                },
                                "raster_profile_revision": {
                                    "type": ["string", "null"],
                                    "maxLength": 128,
                                },
                            },
                        }
                    }
                }
            },
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def raster_document_preview_status(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        status = active_raster_document_preview_runtime.status()
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content={
                "schema_version": "1.0",
                "availability": status.availability,
                "raster_profile_revision": status.raster_profile_revision,
            },
        )

    @app.get(
        "/internal/agent/v3/raster-text-preview-status",
        summary="Return only the active text-only raster profile identity for Backend job binding",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            200: {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "schema_version",
                                "availability",
                                "raster_text_profile_revision",
                            ],
                            "properties": {
                                "schema_version": {"type": "string", "const": "1.0"},
                                "availability": {
                                    "type": "string",
                                    "enum": ["available", "unavailable"],
                                },
                                "raster_text_profile_revision": {
                                    "type": ["string", "null"],
                                    "maxLength": 128,
                                },
                            },
                        }
                    }
                }
            },
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def raster_text_preview_status(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        status = active_raster_text_preview_runtime.status()
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content={
                "schema_version": "1.0",
                "availability": status.availability,
                "raster_text_profile_revision": status.raster_text_profile_revision,
            },
        )

    @app.post(
        "/internal/agent/v3/raster-text-extractions",
        summary="Create one human-review-only local OCR text draft without layout/table claims",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "parameters": [
                {
                    "in": "header",
                    "name": "X-Document-Asset-ID",
                    "required": True,
                    "schema": raster_text_extraction_request_schema["properties"][
                        "document_asset_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Extraction-Request-ID",
                    "required": True,
                    "schema": raster_text_extraction_request_schema["properties"][
                        "extraction_request_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Document-SHA256",
                    "required": True,
                    "schema": raster_text_extraction_request_schema["properties"]["source_sha256"],
                },
                {
                    "in": "header",
                    "name": "X-Raster-Text-Profile-Revision",
                    "required": True,
                    "schema": raster_text_extraction_request_schema["properties"][
                        "raster_text_profile_revision"
                    ],
                },
            ],
            "requestBody": {
                "required": True,
                "description": (
                    "Identity-encoded raw bytes only. Content-Type must be application/pdf, "
                    "image/jpeg, or image/png; multipart, JSON, and base64 are rejected."
                ),
                "content": {
                    "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                    "image/png": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            "x-customs-request-metadata-schema": raster_text_extraction_request_schema,
        },
        responses={
            200: {
                "content": {"application/json": {"schema": raster_text_extraction_result_schema}}
            },
            400: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            408: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def raster_text_extraction(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        admitted = await admitted_raster_text_document(request)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=await execute_raster_text_preview(admitted),
        )

    @app.post(
        "/internal/agent/v3/raster-document-extractions",
        summary="Create one human-review-only local raster OCR/layout candidate",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "parameters": [
                {
                    "in": "header",
                    "name": "X-Document-Asset-ID",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "document_asset_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Extraction-Request-ID",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "extraction_request_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Document-SHA256",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "source_sha256"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Raster-Profile-Revision",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "raster_profile_revision"
                    ],
                },
            ],
            "requestBody": {
                "required": True,
                "description": (
                    "Identity-encoded raw bytes only. Content-Type must be application/pdf, "
                    "image/jpeg, or image/png; multipart, JSON, and base64 are rejected."
                ),
                "content": {
                    "application/pdf": {"schema": {"type": "string", "format": "binary"}},
                    "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                    "image/png": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            "x-customs-request-metadata-schema": raster_document_extraction_request_schema,
        },
        responses={
            200: {
                "content": {"application/json": {"schema": raster_document_extraction_result_schema}}
            },
            400: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            408: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def raster_document_extraction(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        admitted = await admitted_raster_document(request)
        return JSONResponse(
            status_code=200,
            headers=_NO_STORE_HEADERS,
            content=await execute_raster_document_preview(admitted),
        )

    @app.post(
        "/internal/agent/v3/raster-document-pages/{page_number}",
        summary="Render one approved PDFium raster page for the Backend reviewer proxy",
        response_class=Response,
        openapi_extra={
            "security": _BEARER_SECURITY,
            "parameters": [
                {
                    "in": "path",
                    "name": "page_number",
                    "required": True,
                    "schema": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                {
                    "in": "header",
                    "name": "X-Document-Asset-ID",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "document_asset_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Extraction-Request-ID",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "extraction_request_id"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Document-SHA256",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "source_sha256"
                    ],
                },
                {
                    "in": "header",
                    "name": "X-Raster-Profile-Revision",
                    "required": True,
                    "schema": raster_document_extraction_request_schema["properties"][
                        "raster_profile_revision"
                    ],
                },
            ],
            "requestBody": {
                "required": True,
                "description": (
                    "Identity-encoded raw PDF bytes only. This internal endpoint is for an "
                    "authenticated Backend no-store reviewer proxy, never a browser client."
                ),
                "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
            },
            "x-customs-raster-page-response-headers": {
                "X-Document-SHA256": "input/source binding",
                "X-Raster-Profile-Revision": "approved raster profile binding",
                "X-Raster-Page-Number": "rendered page number",
                "X-Raster-Page-Width": "PNG pixel width",
                "X-Raster-Page-Height": "PNG pixel height",
                "X-Raster-Page-SHA256": "returned PNG SHA-256",
                "X-Raster-Renderer-Revision": "PDFium/DPI renderer binding",
            },
        },
        responses={
            200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}},
            400: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            408: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def raster_document_page(request: Request, page_number: int) -> Response:
        _authorize(request, active_settings)
        admitted = await admitted_raster_document(request)
        return await execute_raster_document_page(admitted, page_number=page_number)

    @app.post(
        "/internal/agent/v2/assistant/answers",
        summary="Answer one grounded question about an immutable review result",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": assistant_request_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": assistant_response_schema}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def assistant_answer(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspect_assistant_request(payload)
        controller: AgentServiceController = request.app.state.controller
        outcome = await controller.execute(
            payload=payload,
            runtime_method_name="assistant_answer_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(status_code=200, headers=_NO_STORE_HEADERS, content=outcome.result)

    @app.post(
        "/internal/agent/v3/law-search",
        summary="Proxy one reviewer-facing free-text statute search to Data",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def law_search(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        controller: AgentServiceController = request.app.state.controller
        outcome = await controller.execute(payload=payload, runtime_method_name="law_search_json")
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(status_code=200, headers=_NO_STORE_HEADERS, content=outcome.result)

    @app.post(
        "/internal/agent/v3/law-catalog",
        summary="Proxy the searchable-statute catalog from Data",
        openapi_extra={"security": _BEARER_SECURITY},
        responses={
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def law_catalog(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        controller: AgentServiceController = request.app.state.controller
        outcome = await controller.execute(payload=payload, runtime_method_name="law_catalog_json")
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(status_code=200, headers=_NO_STORE_HEADERS, content=outcome.result)

    @app.post(
        "/internal/agent/v2/knowledge/answers",
        summary="Answer one official-source grounded customs knowledge question",
        openapi_extra={
            "security": _BEARER_SECURITY,
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": knowledge_request_schema}},
            },
        },
        responses={
            200: {"content": {"application/json": {"schema": knowledge_response_schema}}},
            401: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            403: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            413: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            415: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            422: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            502: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
            503: _PRE_EXECUTION_503_RESPONSE,
            504: {"content": {"application/json": {"schema": ERROR_RESPONSE_SCHEMA}}},
        },
    )
    async def knowledge_answer(request: Request) -> JSONResponse:
        _authorize(request, active_settings)
        payload = await admitted_body(request)
        inspect_knowledge_request(payload)
        controller: AgentServiceController = request.app.state.controller
        outcome = await controller.execute(
            payload=payload,
            runtime_method_name="knowledge_answer_json",
        )
        if outcome.result is None:
            raise ServiceHttpError(
                outcome.error_code or "internal_error",
                outcome.status_code,
                retryable=outcome.retryable,
                retry_after_seconds=outcome.retry_after_seconds,
            )
        return JSONResponse(status_code=200, headers=_NO_STORE_HEADERS, content=outcome.result)

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
