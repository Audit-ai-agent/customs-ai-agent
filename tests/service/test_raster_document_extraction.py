from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from customs_ai_agent.schemas.document_extraction import BoundingBoxV1
from customs_ai_agent.schemas.raster_document_extraction import (
    RasterPreviewCandidateV2,
)
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.raster_document_extraction import (
    MediaAwareRasterPreviewEngine,
    RasterDocumentPreviewUnsupported,
    RasterLedgerCellV2,
    RasterLedgerFieldV2,
    RasterLedgerSpanV2,
    RasterLedgerTableV2,
    RasterOcrLayoutLedgerV2,
    RasterizedPageV2,
    raster_ledger_to_preview_candidate,
)
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
        "X-Document-Asset-ID": "asset-raster-1",
        "X-Extraction-Request-ID": "extract-raster-1",
        "X-Document-SHA256": hashlib.sha256(content).hexdigest(),
        "X-Raster-Profile-Revision": "synthetic-raster-v1",
    }


def test_raster_default_is_content_free_unavailable_and_v1_remains_separate() -> None:
    content = _png()
    app = create_app(_settings())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-document-extractions",
            content=content,
            headers=_headers(content, "image/png"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "schema_version": "2.0",
        "document_asset_id": "asset-raster-1",
        "extraction_request_id": "extract-raster-1",
        "raster_profile_revision": "synthetic-raster-v1",
        "status": "unavailable",
        "media_type": "image/png",
        "byte_size": len(content),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "canonical_text_sha256": None,
        "extractor_revision": None,
        "document_type": "unknown",
        "classification_confidence": None,
        "pages": [],
        "field_candidates": [],
        "tables": [],
        "warnings": [{"code": "raster_engine_unavailable"}],
        "needs_review_reasons": [],
    }
    assert app.state.raster_document_preview_activation_status == "disabled"


def test_raster_requires_profile_revision_header() -> None:
    content = _png()
    headers = _headers(content, "image/png")
    del headers["X-Raster-Profile-Revision"]
    app = create_app(_settings())

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-document-extractions",
            content=content,
            headers=headers,
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "document_request_invalid"


def test_rotated_jpeg_is_needs_review_before_engine_execution() -> None:
    content = _jpeg_with_orientation(6)
    calls = 0

    class Engine:
        def extract(self, _input: object, _structure: object) -> RasterPreviewCandidateV2:
            nonlocal calls
            calls += 1
            raise AssertionError("rotated JPEG must not reach OCR")

    app = create_app(_settings(), raster_document_preview_engine=Engine())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-document-extractions",
            content=content,
            headers=_headers(content, "image/jpeg"),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["needs_review_reasons"] == [
        {"code": "jpeg_exif_orientation_unsupported"}
    ]
    assert calls == 0


def test_multi_page_pdf_is_needs_review_before_unmeasured_ocr_execution() -> None:
    class Adapter:
        def rasterize_pdf(self, _input: object, _structure: object) -> list[RasterizedPageV2]:
            raise RasterDocumentPreviewUnsupported("multi_page_pdf_unsupported")

        def decode_image(self, _input: object, _structure: object) -> RasterizedPageV2:
            raise AssertionError("PDF must not use image decoding")

    class Provider:
        def extract_layout(self, *_args: object, **_kwargs: object) -> RasterOcrLayoutLedgerV2:
            raise AssertionError("multi-page PDF must not reach OCR")

    content = b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n2 0 obj<</Type /Page>>endobj\n%%EOF\n"
    app = create_app(
        _settings(),
        raster_document_preview_engine=MediaAwareRasterPreviewEngine(Adapter(), Provider()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-document-extractions",
            content=content,
            headers=_headers(content, "application/pdf"),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["needs_review_reasons"] == [{"code": "multi_page_pdf_unsupported"}]


def test_media_aware_engine_passes_only_rasterized_pdf_pages_to_provider() -> None:
    captured: dict[str, object] = {}

    class Adapter:
        def rasterize_pdf(self, raster_input: object, _structure: object) -> list[RasterizedPageV2]:
            captured["pdf_bytes"] = getattr(raster_input, "content")
            return [RasterizedPageV2(page_number=1, width=100, height=200, pixels=b"raster")]

        def decode_image(self, _raster_input: object, _structure: object) -> RasterizedPageV2:
            raise AssertionError("PDF must use PDF rasterizer")

    class Provider:
        def extract_layout(
            self, pages: Sequence[RasterizedPageV2], **kwargs: object
        ) -> RasterOcrLayoutLedgerV2:
            page = tuple(pages)[0]
            assert isinstance(page, RasterizedPageV2)
            captured["provider_pages"] = tuple(pages)
            return RasterOcrLayoutLedgerV2(
                source_sha256=str(kwargs["source_sha256"]),
                raster_profile_revision=str(kwargs["raster_profile_revision"]),
                provider_revision="fake-ppstructure-v3",
                pages=(page,),
                spans=(
                    RasterLedgerSpanV2(
                        span_id="span-1",
                        page_number=1,
                        ordinal=1,
                        reading_order=1,
                        text="COMMERCIAL INVOICE",
                        bbox={"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.1},
                        confidence=0.99,
                    ),
                ),
                document_type="commercial_invoice",
                classification_confidence=0.99,
            )

    content = b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF\n"
    app = create_app(
        _settings(),
        raster_document_preview_engine=MediaAwareRasterPreviewEngine(Adapter(), Provider()),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-document-extractions",
            content=content,
            headers=_headers(content, "application/pdf"),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "preview_ready"
    assert captured["pdf_bytes"] == content
    pages = captured["provider_pages"]
    assert isinstance(pages, tuple)
    assert all(isinstance(page, RasterizedPageV2) for page in pages)


def test_missing_geometry_is_a_review_draft_not_a_preview_ready_server_error() -> None:
    candidate = raster_ledger_to_preview_candidate(
        RasterOcrLayoutLedgerV2(
            source_sha256="a" * 64,
            raster_profile_revision="synthetic-raster-v1",
            provider_revision="fake-ppstructure-v3",
            pages=(RasterizedPageV2(page_number=1, width=100, height=200, pixels=b"raster"),),
            spans=(
                RasterLedgerSpanV2(
                    span_id="span-1",
                    page_number=1,
                    ordinal=1,
                    reading_order=1,
                    text="COMMERCIAL INVOICE",
                    bbox=None,
                    confidence=0.99,
                ),
            ),
            document_type="commercial_invoice",
            classification_confidence=0.99,
        )
    )
    assert candidate.status == "needs_review"
    assert [reason.code for reason in candidate.needs_review_reasons or []] == ["layout_incomplete"]


def test_ledger_projection_preserves_provider_fields_and_table_cells() -> None:
    """The provider-owned structure is passed through without reconstruction.

    This guards the boundary separately from the PP-Structure projector: if a
    provider supplies evidence-backed fields and cells, neither the ledger
    conversion nor public preview projection may erase or re-order them.
    """

    def bbox(x: float, y: float, width: float, height: float) -> BoundingBoxV1:
        return BoundingBoxV1(x=x, y=y, width=width, height=height)

    spans = (
        RasterLedgerSpanV2(
            span_id="span-2",
            page_number=1,
            ordinal=2,
            reading_order=2,
            text="value-b",
            bbox=bbox(0.30, 0.10, 0.20, 0.10),
            confidence=0.97,
        ),
        RasterLedgerSpanV2(
            span_id="span-1",
            page_number=1,
            ordinal=1,
            reading_order=1,
            text="value-a",
            bbox=bbox(0.10, 0.10, 0.10, 0.10),
            confidence=0.98,
        ),
    )
    candidate = raster_ledger_to_preview_candidate(
        RasterOcrLayoutLedgerV2(
            source_sha256="a" * 64,
            raster_profile_revision="synthetic-raster-v1",
            provider_revision="fake-ppstructure-v3",
            pages=(RasterizedPageV2(page_number=1, width=100, height=200, pixels=b"raster"),),
            spans=spans,
            document_type="commercial_invoice",
            classification_confidence=0.98,
            fields=(
                RasterLedgerFieldV2(
                    field_id="field_a",
                    value="value-b",
                    confidence=0.97,
                    evidence_span_ids=("span-1", "span-2"),
                    bbox=bbox(0.10, 0.10, 0.40, 0.10),
                ),
            ),
            tables=(
                RasterLedgerTableV2(
                    table_id="items",
                    page_number=1,
                    bbox=bbox(0.10, 0.40, 0.60, 0.30),
                    cells=(
                        RasterLedgerCellV2(
                            cell_id="items-r1-c0",
                            page_number=1,
                            row_index=1,
                            column_index=0,
                            text="value-b",
                            bbox=bbox(0.10, 0.60, 0.30, 0.10),
                            confidence=0.97,
                            evidence_span_ids=("span-2",),
                        ),
                        RasterLedgerCellV2(
                            cell_id="items-r0-c0",
                            page_number=1,
                            row_index=0,
                            column_index=0,
                            text="value-a",
                            bbox=bbox(0.10, 0.40, 0.30, 0.10),
                            confidence=0.98,
                            evidence_span_ids=("span-1",),
                        ),
                    ),
                ),
            ),
        )
    )

    pages = candidate.pages or []
    fields = candidate.field_candidates or []
    tables = candidate.tables or []
    assert candidate.status == "preview_ready"
    assert [span.span_id for span in pages[0].spans] == ["span-1", "span-2"]
    assert [field.field_id for field in fields] == ["field_a"]
    assert fields[0].evidence_span_ids == ["span-1", "span-2"]
    assert len(tables) == 1
    assert [row.row_index for row in tables[0].rows] == [0, 1]
    assert [
        (cell.row_index, cell.column_index, cell.cell_id)
        for row in tables[0].rows
        for cell in row.cells
    ] == [(0, 0, "items-r0-c0"), (1, 0, "items-r1-c0")]


def test_raster_page_endpoint_requires_a_sealed_pdfium_renderer_not_an_injected_ocr_engine() -> (
    None
):
    content = b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF\n"
    app = create_app(_settings(), raster_document_preview_engine=object())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v3/raster-document-pages/1",
            content=content,
            headers=_headers(content, "application/pdf"),
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "raster_page_renderer_unavailable"


def test_openapi_exposes_v3_binary_contract_without_base64() -> None:
    schema = create_app(_settings()).openapi()
    endpoint = schema["paths"]["/internal/agent/v3/raster-document-extractions"]["post"]

    assert set(endpoint["requestBody"]["content"]) == {
        "application/pdf",
        "image/jpeg",
        "image/png",
    }
    metadata = endpoint["x-customs-request-metadata-schema"]
    result = endpoint["responses"]["200"]["content"]["application/json"]["schema"]
    assert metadata["required"][-1] == "raster_profile_revision"
    assert result["$id"].endswith("/v3/raster-document-extraction-result.schema.json")
    assert "base64" not in metadata["properties"]
    assert "base64" not in result["properties"]
    page_endpoint = schema["paths"]["/internal/agent/v3/raster-document-pages/{page_number}"][
        "post"
    ]
    assert set(page_endpoint["requestBody"]["content"]) == {"application/pdf"}
    assert set(page_endpoint["responses"]["200"]["content"]) == {"image/png"}
    assert page_endpoint["x-customs-raster-page-response-headers"]["X-Raster-Page-SHA256"]


def test_raster_auto_activation_settings_require_fixed_path_without_static_hash(
    tmp_path: Path,
) -> None:
    pointer = (
        tmp_path
        / "_local"
        / "data"
        / "document-preview"
        / "approvals"
        / "raster-preview-active.json"
    )
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
        "EVIDENCE_SERVICE_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        "CUSTOMS_AGENT_RASTER_PREVIEW_APPROVAL_PATH": str(pointer),
        "CUSTOMS_AGENT_RASTER_PREVIEW_AUTO_ACTIVATION": "source-attested-synthetic-raster-v1",
    }

    settings = ServiceSettings.from_environment(source)
    assert settings.raster_document_preview_approval_path == pointer
    assert settings.raster_document_preview_approval_sha256 is None
    assert (
        settings.raster_document_preview_auto_activation_mode
        == "source-attested-synthetic-raster-v1"
    )
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_raster_preview_auto_activation_incomplete",
    ):
        ServiceSettings.from_environment(
            {**source, "CUSTOMS_AGENT_RASTER_PREVIEW_APPROVAL_SHA256": "a" * 64}
        )


def test_raster_runtime_has_a_separate_serial_sixty_second_bound() -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
        "EVIDENCE_SERVICE_TOKEN": _TOKEN,
        "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        # Preserve the independent non-raster document-preview setting.
        "CUSTOMS_AGENT_RUN_TIMEOUT_SECONDS": "420",
        "CUSTOMS_AGENT_RASTER_RUN_TIMEOUT_SECONDS": "60",
        "CUSTOMS_AGENT_RASTER_PAGE_RENDER_TIMEOUT_SECONDS": "20",
    }

    settings = ServiceSettings.from_environment(source)
    assert settings.run_timeout_seconds == 420
    assert settings.raster_run_timeout_seconds == 60
    assert settings.raster_page_render_timeout_seconds == 20
    assert settings.raster_max_concurrency == 1

    with pytest.raises(
        ServiceConfigurationError, match="customs_agent_raster_run_timeout_seconds_invalid"
    ):
        ServiceSettings.from_environment(
            {**source, "CUSTOMS_AGENT_RASTER_RUN_TIMEOUT_SECONDS": "61"}
        )
    with pytest.raises(
        ServiceConfigurationError, match="customs_agent_raster_max_concurrency_invalid"
    ):
        ServiceSettings.from_environment({**source, "CUSTOMS_AGENT_RASTER_MAX_CONCURRENCY": "2"})


def test_raster_status_endpoint_is_authenticated_no_store_and_content_free(tmp_path: Path) -> None:
    pointer = (
        tmp_path
        / "_local"
        / "data"
        / "document-preview"
        / "approvals"
        / "raster-preview-active.json"
    )
    pointer.parent.mkdir(parents=True)
    settings = _settings(
        raster_document_preview_approval_path=pointer,
        raster_document_preview_auto_activation_mode="source-attested-synthetic-raster-v1",
    )
    app = create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        missing_auth = client.get(
            "/internal/agent/v3/raster-document-preview-status",
            headers={"X-Customs-Service-Mode": "local-development"},
        )
        response = client.get(
            "/internal/agent/v3/raster-document-preview-status",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                "X-Customs-Service-Mode": "local-development",
            },
        )

    assert missing_auth.status_code == 401
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "schema_version": "1.0",
        "availability": "unavailable",
        "raster_profile_revision": None,
    }
    assert app.state.raster_document_preview_activation_status == "auto-pending"
