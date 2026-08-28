from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Callable, Sequence

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from customs_ai_agent.schemas.raster_text_extraction import RasterTextPageV1
import customs_ai_agent.service.app as app_module
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.raster_document_extraction import RasterizedPageV2
from customs_ai_agent.service.raster_text_extraction import (
    MediaAwareRasterTextEngine,
    RasterTextLedgerV1,
)
from customs_ai_agent.service.raster_text_preview_runtime import RasterTextPreviewStatusV1
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"


def _settings(**changes: object) -> ServiceSettings:
    baseline = ServiceSettings(
        bearer_token=SecretStr(_TOKEN),
        evidence_base_url="http://127.0.0.1:18083",
        evidence_service_token=SecretStr(_TOKEN),
    )
    return replace(baseline, **changes)


def _png(width: int = 2, height: int = 3) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def _jpeg_with_orientation(orientation: int) -> bytes:
    tiff = (
        b"II\x2a\x00\x08\x00\x00\x00"
        + b"\x01\x00"
        + b"\x12\x01\x03\x00\x01\x00\x00\x00"
        + orientation.to_bytes(2, "little")
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    exif = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    sof0 = b"\xff\xc0\x00\x11\x08\x00\x03\x00\x02\x03" + (b"\x01\x11\x00" * 3)
    return b"\xff\xd8" + app1 + sof0 + b"\xff\xd9"


def _headers(content: bytes, media_type: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_TOKEN}",
        "X-Customs-Service-Mode": "local-development",
        "Content-Type": media_type,
        "X-Document-Asset-ID": "asset-raster-text-1",
        "X-Extraction-Request-ID": "extract-raster-text-1",
        "X-Document-SHA256": hashlib.sha256(content).hexdigest(),
        "X-Raster-Text-Profile-Revision": "synthetic-raster-text-v1",
    }


def _status_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_TOKEN}",
        "X-Customs-Service-Mode": "local-development",
    }


def _auto_text_settings(tmp_path: Path) -> ServiceSettings:
    return _settings(
        raster_text_preview_approval_path=(
            tmp_path
            / "_local"
            / "data"
            / "document-preview"
            / "approvals"
            / "raster-text-active.json"
        ),
        raster_text_preview_auto_activation_mode="source-attested-synthetic-raster-text-v1",
    )


def _wait_until(predicate: Callable[[], bool], *, timeout_seconds: float = 3.5) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_default_text_endpoint_is_content_free_unavailable() -> None:
    content = _png()
    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["canonical_text"] is None
    assert body["canonical_text_sha256"] is None
    assert body["pages"] == []
    assert body["warnings"] == [{"code": "raster_text_engine_unavailable"}]
    assert app.state.raster_text_preview_activation_status == "disabled"


def test_text_only_engine_returns_editable_canonical_text_without_layout_claims() -> None:
    captured: dict[str, object] = {}

    class Adapter:
        def rasterize_pdf(self, _input: object, _structure: object) -> Sequence[RasterizedPageV2]:
            raise AssertionError("image must not use PDF rendering")

        def decode_image(self, _input: object, _structure: object) -> RasterizedPageV2:
            return RasterizedPageV2(page_number=1, width=2, height=3, pixels=b"\x00" * 18)

    class Provider:
        def extract_text(
            self, pages: Sequence[RasterizedPageV2], **kwargs: object
        ) -> RasterTextLedgerV1:
            captured["pages"] = tuple(pages)
            return RasterTextLedgerV1(
                source_sha256=str(kwargs["source_sha256"]),
                raster_text_profile_revision=str(kwargs["raster_text_profile_revision"]),
                provider_revision="direct-text-test-v1",
                pages=(
                    RasterTextPageV1(page_number=1, text="COMMERCIAL INVOICE\r\nTOTAL 10.00"),
                ),
            )

    content = _png()
    app = create_app(
        _settings(),
        raster_text_preview_engine=MediaAwareRasterTextEngine(Adapter(), Provider()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "preview_ready"
    assert body["canonical_text"] == "COMMERCIAL INVOICE\nTOTAL 10.00"
    assert body["pages"] == [
        {"page_number": 1, "text": "COMMERCIAL INVOICE\nTOTAL 10.00", "confidence": None}
    ]
    assert "tables" not in body and "field_candidates" not in body and "bbox" not in body
    assert captured["pages"]


def test_valid_zero_text_ocr_result_is_content_free_needs_review() -> None:
    class Adapter:
        def rasterize_pdf(self, _input: object, _structure: object) -> Sequence[RasterizedPageV2]:
            raise AssertionError("image must not use PDF rendering")

        def decode_image(self, _input: object, _structure: object) -> RasterizedPageV2:
            return RasterizedPageV2(page_number=1, width=2, height=3, pixels=b"\x00" * 18)

    class Provider:
        def extract_text(
            self, _pages: Sequence[RasterizedPageV2], **kwargs: object
        ) -> RasterTextLedgerV1:
            # A successful recognizer can legitimately find no text.  This is
            # intentionally distinct from a sidecar error/unavailable state.
            return RasterTextLedgerV1(
                source_sha256=str(kwargs["source_sha256"]),
                raster_text_profile_revision=str(kwargs["raster_text_profile_revision"]),
                provider_revision="direct-text-test-v1",
                pages=(),
            )

    content = _png()
    app = create_app(
        _settings(),
        raster_text_preview_engine=MediaAwareRasterTextEngine(Adapter(), Provider()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )
    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "1.0",
        "document_asset_id": "asset-raster-text-1",
        "extraction_request_id": "extract-raster-text-1",
        "raster_text_profile_revision": "synthetic-raster-text-v1",
        "status": "needs_review",
        "media_type": "image/png",
        "byte_size": len(content),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "canonical_text": None,
        "canonical_text_sha256": None,
        "extractor_revision": None,
        "pages": [],
        "warnings": [],
        "needs_review_reasons": [{"code": "ocr_text_not_found"}],
    }


def test_rotated_jpeg_is_needs_review_before_text_engine_execution() -> None:
    content = _jpeg_with_orientation(6)
    calls = 0

    class Engine:
        def extract(self, _input: object, _structure: object) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("rotated JPEG must not reach OCR")

    app = create_app(_settings(), raster_text_preview_engine=Engine())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/jpeg"),
        )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["needs_review_reasons"] == [
        {"code": "jpeg_exif_orientation_unsupported"}
    ]
    assert calls == 0


def test_multi_page_input_is_needs_review_before_text_provider_execution() -> None:
    class Adapter:
        def rasterize_pdf(self, _input: object, _structure: object) -> Sequence[RasterizedPageV2]:
            return (
                RasterizedPageV2(page_number=1, width=2, height=3, pixels=b"\x00" * 18),
                RasterizedPageV2(page_number=2, width=2, height=3, pixels=b"\x00" * 18),
            )

        def decode_image(self, _input: object, _structure: object) -> RasterizedPageV2:
            raise AssertionError("PDF must not use image decode")

    class Provider:
        def extract_text(self, *_args: object, **_kwargs: object) -> RasterTextLedgerV1:
            raise AssertionError("unmeasured multi-page PDF must not reach OCR")

    content = b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF\n"
    app = create_app(
        _settings(),
        raster_text_preview_engine=MediaAwareRasterTextEngine(Adapter(), Provider()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "application/pdf"),
        )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["needs_review_reasons"] == [{"code": "multi_page_pdf_unsupported"}]


def test_known_multi_page_pdf_is_rejected_before_rasterization() -> None:
    class Adapter:
        def rasterize_pdf(self, _input: object, _structure: object) -> Sequence[RasterizedPageV2]:
            raise AssertionError("known multi-page PDF must not reach the renderer")

        def decode_image(self, _input: object, _structure: object) -> RasterizedPageV2:
            raise AssertionError("PDF must not use image decode")

    class Provider:
        def extract_text(self, *_args: object, **_kwargs: object) -> RasterTextLedgerV1:
            raise AssertionError("known multi-page PDF must not reach OCR")

    content = (
        b"%PDF-1.7\n"
        b"1 0 obj<</Type /Page>>endobj\n"
        b"2 0 obj<</Type /Page>>endobj\n%%EOF\n"
    )
    app = create_app(
        _settings(),
        raster_text_preview_engine=MediaAwareRasterTextEngine(Adapter(), Provider()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "application/pdf"),
        )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["needs_review_reasons"] == [{"code": "multi_page_pdf_unsupported"}]


def test_text_openapi_has_raw_binary_contract_and_no_layout_fields() -> None:
    schema = create_app(_settings()).openapi()
    endpoint = schema["paths"]["/internal/agent/v3/raster-text-extractions"]["post"]
    assert set(endpoint["requestBody"]["content"]) == {
        "application/pdf",
        "image/jpeg",
        "image/png",
    }
    metadata = endpoint["x-customs-request-metadata-schema"]
    result = endpoint["responses"]["200"]["content"]["application/json"]["schema"]
    assert metadata["required"][-1] == "raster_text_profile_revision"
    assert result["$id"].endswith("/v3/raster-text-extraction-result.schema.json")
    assert "tables" not in result["properties"]
    assert "field_candidates" not in result["properties"]
    assert "base64" not in metadata["properties"]


def test_text_auto_activation_settings_require_fixed_path_without_static_hash(tmp_path: Path) -> None:
    pointer = (
        tmp_path
        / "_local"
        / "data"
        / "document-preview"
        / "approvals"
        / "raster-text-active.json"
    )
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
        "EVIDENCE_SERVICE_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_PATH": str(pointer),
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_AUTO_ACTIVATION": (
            "source-attested-synthetic-raster-text-v1"
        ),
    }
    settings = ServiceSettings.from_environment(source)
    assert settings.raster_text_preview_approval_path == pointer
    assert settings.raster_text_preview_approval_sha256 is None
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_raster_text_preview_auto_activation_incomplete",
    ):
        ServiceSettings.from_environment(
            {**source, "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_SHA256": "a" * 64}
        )


def test_auto_raster_text_prewarm_is_nonblocking_and_reloads_late_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing pointer must not block readiness, and later replacements warm serially."""

    first_revision = "synthetic-raster-text-v1:" + "1" * 24
    replacement_revision = "synthetic-raster-text-v1:" + "2" * 24

    class FakeAutoRuntime:
        def __init__(self) -> None:
            self._lock = threading.RLock()
            self.pending_revision: str | None = None
            self.active_revision: str | None = None
            self.first_started = threading.Event()
            self.allow_first = threading.Event()
            self.first_finished = threading.Event()
            self.prewarm_calls = 0
            self.inflight = 0
            self.max_inflight = 0
            self.close_calls = 0

        def prewarm(self) -> RasterTextPreviewStatusV1:
            with self._lock:
                self.prewarm_calls += 1
                call_number = self.prewarm_calls
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
            try:
                if call_number == 1:
                    self.first_started.set()
                    assert self.allow_first.wait(timeout=5)
                with self._lock:
                    self.active_revision = self.pending_revision
                    status = self.status()
                if call_number == 1:
                    self.first_finished.set()
                return status
            finally:
                with self._lock:
                    self.inflight -= 1

        def status(self) -> RasterTextPreviewStatusV1:
            with self._lock:
                return RasterTextPreviewStatusV1(
                    availability="available" if self.active_revision is not None else "unavailable",
                    raster_text_profile_revision=self.active_revision,
                )

        def close(self) -> None:
            with self._lock:
                self.close_calls += 1
            self.allow_first.set()

    runtime = FakeAutoRuntime()
    monkeypatch.setattr(
        app_module,
        "AutoActivatingSourceAttestedRasterTextPreviewRuntime",
        lambda **_kwargs: runtime,
    )
    app = create_app(_auto_text_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as client:
        # ``prewarm`` is deliberately blocked.  Lifespan startup and status
        # remain usable because the native work runs in a worker thread.
        assert runtime.first_started.wait(timeout=1)
        first_status = client.get(
            "/internal/agent/v3/raster-text-preview-status", headers=_status_headers()
        )
        assert first_status.status_code == 200
        assert first_status.json() == {
            "schema_version": "1.0",
            "availability": "unavailable",
            "raster_text_profile_revision": None,
        }
        assert runtime.max_inflight == 1

        # The first warmup represents pointer absence.  A later pointer is
        # noticed by the same one-second poll rather than a status request
        # triggering a cold start itself.
        runtime.allow_first.set()
        assert runtime.first_finished.wait(timeout=1)
        with runtime._lock:
            runtime.pending_revision = first_revision
        assert _wait_until(
            lambda: client.get(
                "/internal/agent/v3/raster-text-preview-status", headers=_status_headers()
            ).json()["raster_text_profile_revision"]
            == first_revision
        )

        with runtime._lock:
            runtime.pending_revision = replacement_revision
        assert _wait_until(
            lambda: client.get(
                "/internal/agent/v3/raster-text-preview-status", headers=_status_headers()
            ).json()["raster_text_profile_revision"]
            == replacement_revision
        )
        assert runtime.max_inflight == 1

    assert runtime.close_calls == 1


def test_auto_raster_text_prewarm_shutdown_cancels_loop_then_closes_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shutdown must not leave a blocked poll task or schedule another warmup."""

    class BlockingAutoRuntime:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.returned = threading.Event()
            self.close_calls = 0
            self.prewarm_calls = 0

        def prewarm(self) -> RasterTextPreviewStatusV1:
            self.prewarm_calls += 1
            self.started.set()
            assert self.release.wait(timeout=5)
            self.returned.set()
            return RasterTextPreviewStatusV1("unavailable", None)

        def status(self) -> RasterTextPreviewStatusV1:
            return RasterTextPreviewStatusV1("unavailable", None)

        def close(self) -> None:
            self.close_calls += 1
            self.release.set()

    runtime = BlockingAutoRuntime()
    monkeypatch.setattr(
        app_module,
        "AutoActivatingSourceAttestedRasterTextPreviewRuntime",
        lambda **_kwargs: runtime,
    )
    app = create_app(_auto_text_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False):
        assert runtime.started.wait(timeout=1)
        assert app.state.raster_text_prewarm_task is not None

    assert app.state.raster_text_prewarm_task is None
    assert runtime.close_calls == 1
    assert runtime.returned.wait(timeout=1)
    # Cancellation prevents the periodic loop from scheduling a second call
    # after ``close`` releases the original worker.
    time.sleep(1.05)
    assert runtime.prewarm_calls == 1


def test_timeout_keeps_the_shared_raster_lane_until_the_owned_worker_returns() -> None:
    """A cancelled async wait must not permit concurrent native OCR work."""

    class SlowEngine:
        def extract(self, _input: object, _structure: object) -> object:
            time.sleep(1.15)
            from customs_ai_agent.service.raster_text_extraction import RasterTextPreviewUnavailable

            raise RasterTextPreviewUnavailable("slow test worker")

    content = _png()
    app = create_app(
        _settings(raster_run_timeout_seconds=1, queue_wait_seconds=0.01),
        raster_text_preview_engine=SlowEngine(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )
        second = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )
        time.sleep(1.2)
        third = client.post(
            "/internal/agent/v3/raster-text-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )
    assert first.status_code == 504
    assert second.status_code == 503
    # The late worker released the lane after it actually finished.  The next
    # request owns it and has its own bounded timeout instead of overlapping.
    assert third.status_code == 504
