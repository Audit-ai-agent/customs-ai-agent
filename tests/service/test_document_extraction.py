from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionWarningV1,
    DocumentPreviewPageV1,
    DocumentSpanV1,
    PreviewCandidateV1,
)
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.document_extraction import LayeredLocalPreviewEngine
from customs_ai_agent.service.settings import ServiceSettings


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


def _pdf() -> bytes:
    return b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF\n"


def _headers(content: bytes, media_type: str, *, asset_id: str = "asset-1") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_TOKEN}",
        "X-Customs-Service-Mode": "local-development",
        "Content-Type": media_type,
        "X-Document-Asset-ID": asset_id,
        "X-Extraction-Request-ID": "extract-1",
        "X-Document-SHA256": hashlib.sha256(content).hexdigest(),
    }


def _image_candidate() -> PreviewCandidateV1:
    return PreviewCandidateV1(
        status="preview_ready",
        extractor_revision="fake-image-v1",
        pages=[
            DocumentPreviewPageV1(
                page_number=1,
                width=2,
                height=3,
                coordinate_unit="pixels",
                spans=[
                    DocumentSpanV1(
                        span_id="page-1-span-1",
                        ordinal=1,
                        reading_order=1,
                        text="INVOICE NO INV-001",
                        bbox={"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.1},
                    )
                ],
            )
        ],
        warnings=[],
    )


class _ImageEngine:
    def __init__(self) -> None:
        self.contents: list[bytes] = []

    def extract(self, preview_input: object) -> PreviewCandidateV1:
        content = getattr(preview_input, "content")
        self.contents.append(content)
        return _image_candidate()


def test_disabled_default_returns_content_free_unavailable_preview() -> None:
    content = _png()
    app = create_app(_settings())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/document-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "schema_version": "1.0",
        "document_asset_id": "asset-1",
        "extraction_request_id": "extract-1",
        "status": "unavailable",
        "media_type": "image/png",
        "byte_size": len(content),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "canonical_text_sha256": None,
        "extractor_revision": None,
        "pages": [],
        "warnings": [{"code": "preview_engine_unavailable"}],
    }


def test_explicit_source_attested_auto_mode_starts_pending_without_a_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / "_local" / "data" / "document-preview" / "approvals" / "pdf-preview-active.json"
    app = create_app(
        _settings(
            document_preview_approval_path=pointer,
            document_preview_auto_activation_mode="source-attested-synthetic",
        )
    )
    content = _pdf()

    assert app.state.document_preview_activation_status == "auto-pending"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/document-extractions",
            content=content,
            headers=_headers(content, "application/pdf"),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["extractor_revision"] is None


def test_preview_returns_only_extracted_spans_not_binary_or_filename(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "RAW_BINARY_MUST_NOT_BE_LOGGED"
    content = _png() + marker.encode("ascii")
    engine = _ImageEngine()
    app = create_app(_settings(), document_preview_engine=engine)
    caplog.set_level(logging.INFO)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/document-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "preview_ready"
    assert body["pages"][0]["spans"][0]["text"] == "INVOICE NO INV-001"
    assert "content" not in body
    assert "filename" not in body
    assert engine.contents == [content]
    # The service logger allows only event/code/request metadata.  Neither
    # binary bytes nor preview text belong in its records.
    assert "INVOICE NO INV-001" not in caplog.text  # type: ignore[attr-defined]
    assert marker not in caplog.text  # type: ignore[attr-defined]


def test_magic_and_hash_fail_closed_before_adapter_execution() -> None:
    content = _png()
    engine = _ImageEngine()
    app = create_app(_settings(), document_preview_engine=engine)

    with TestClient(app, raise_server_exceptions=False) as client:
        magic = client.post(
            "/internal/agent/v2/document-extractions",
            content=b"not-a-pdf",
            headers=_headers(b"not-a-pdf", "application/pdf"),
        )
        headers = _headers(content, "image/png")
        headers["X-Document-SHA256"] = "0" * 64
        digest = client.post(
            "/internal/agent/v2/document-extractions",
            content=content,
            headers=headers,
        )

    assert magic.status_code == 422
    assert magic.json()["error"]["code"] == "document_magic_mismatch"
    assert digest.status_code == 422
    assert digest.json()["error"]["code"] == "document_hash_mismatch"
    assert engine.contents == []


def test_image_pixel_limit_is_enforced_before_adapter_execution() -> None:
    content = _png(width=2_000, height=2_000)
    engine = _ImageEngine()
    app = create_app(_settings(max_document_image_pixels=3_000_000), document_preview_engine=engine)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/document-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "document_image_limits_exceeded"
    assert engine.contents == []


def test_pdf_text_layer_is_used_before_raster_ocr() -> None:
    calls: list[str] = []

    class TextLayer:
        def extract_pdf_text_layer(self, _input: object) -> PreviewCandidateV1 | None:
            calls.append("text-layer")
            return PreviewCandidateV1(
                status="preview_ready",
                extractor_revision="fake-pdf-text-v1",
                pages=[
                    DocumentPreviewPageV1(
                        page_number=1,
                        width=612,
                        height=792,
                        coordinate_unit="pdf_points",
                        spans=[
                            DocumentSpanV1(
                                span_id="page-1-span-1",
                                ordinal=1,
                                reading_order=1,
                                text="PDF TEXT LAYER",
                            )
                        ],
                    )
                ],
                warnings=[DocumentExtractionWarningV1(code="text_layer_used")],
            )

    class Raster:
        def extract_raster_or_image(self, _input: object) -> PreviewCandidateV1:
            calls.append("raster")
            raise AssertionError("PDF text layer should have been used first")

    content = _pdf()
    app = create_app(
        _settings(), document_preview_engine=LayeredLocalPreviewEngine(TextLayer(), Raster())
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/document-extractions",
            content=content,
            headers=_headers(content, "application/pdf"),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "preview_ready"
    assert calls == ["text-layer"]


def test_openapi_exposes_raw_binary_preview_contract_without_base64() -> None:
    schema = create_app(_settings()).openapi()
    endpoint = schema["paths"]["/internal/agent/v2/document-extractions"]["post"]

    assert set(endpoint["requestBody"]["content"]) == {
        "application/pdf",
        "image/jpeg",
        "image/png",
    }
    assert endpoint["responses"]["200"]["content"]["application/json"]["schema"]["$id"].endswith(
        "/v3/document-extraction-result.schema.json"
    )
    metadata = endpoint["x-customs-request-metadata-schema"]
    result = endpoint["responses"]["200"]["content"]["application/json"]["schema"]
    assert "base64" not in metadata["properties"]
    assert "filename" not in metadata["properties"]
    assert "base64" not in result["properties"]
    assert "filename" not in result["properties"]
